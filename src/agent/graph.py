"""Préparation des données : audit, choix humains, nettoyage et export Parquet.

Les propositions sont déterministes ; le traitement est exécuté par
src.services.preprocessing. Le CSV d’origine reste inchangé.
"""

import hashlib
import difflib
import json
import math
import uuid
from typing import TypedDict
from pathlib import Path
import polars as pl
import re
from collections import Counter
from datetime import timedelta
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from src.services.data_loader import inspect_dataset, load_lazy_dataset
from src.services.preprocessing import run_preprocessing

ALLOWED_OPERATIONS = {'parse_time', 'sort_time', 'handle_missing', 'handle_nan', 'interpolate', 'remove_duplicates', 'resample', 'handle_outliers', 'normalize_categories', 'regularize_timestamps'}
TEMPORAL_TYPES = {'datetime': 'Datetime', 'date': 'Date', 'time': 'Time', 'unix_s': 'Int64', 'unix_ms': 'Int64', 'elapsed_s': 'Float64', 'sample_index': 'Int64', 'implicit': None, 'none': None}
OUTLIER_METHODS = {'iqr'}
OUTLIER_ACTIONS = {'keep', 'drop'}

class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""
    # Chemin du fichier d’origine
    source_dataset_path: str
    # Résultats de l’audit Polars : lacunes, anomalies, doublons et temps.
    quality_audit: dict
    # Colonne temporelle, type, format et fréquence éventuelle.
    time_settings: dict
    # résumé des propositions de nettoyage.
    recommendation_summary: str
    # opérations proposées avant approbation.
    cleaning_proposals: list[dict]
    # Opérations explicitement acceptées par l’utilisateur.
    approved_operations: list[dict]
    # Réponses aux questions métier et ajustements.
    user_answers: dict[str, str]
    # Propositions avec statut de validation et erreurs éventuelles.
    checked_proposals: list[dict]
    # Interprétation des réponses sur les pannes et les pics.
    business_context: dict
    # True si les étapes suivantes exigent zéro valeur manquante.
    requires_complete_data: bool
    # Réponse originale yes/no/unknown, conservée pour le rapport.
    completeness_response: str



# 1. Détecter les valeurs atypiques selon la règle IQR.
def detect_outliers_iqr(lf: pl.LazyFrame, columns: list[str], time_column: str | None=None, factor: float=1.5) -> pl.DataFrame:
    """Return an independent report of IQR outliers."""
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or (not math.isfinite(factor)) or (factor <= 0):
        raise ValueError('IQR factor must be positive and finite.')
    schema = lf.collect_schema()
    if time_column is not None and time_column not in schema:
        raise ValueError(f'Unknown time column: {time_column}')
    invalid_columns = [column for column in columns if column not in schema or not schema[column].is_numeric()]
    if invalid_columns:
        raise ValueError(f'Unknown or nonnumeric columns: {invalid_columns}')
    if not columns:
        empty_schema = {'row_index': pl.UInt64, 'column': pl.String, 'value': pl.Float64, 'lower_bound': pl.Float64, 'upper_bound': pl.Float64}
        if time_column is not None:
            time_type = schema[time_column]
            empty_schema_with_time = {'row_index': pl.UInt64, time_column: time_type}
            for column_name, column_type in empty_schema.items():
                if column_name != 'row_index':
                    empty_schema_with_time[column_name] = column_type
            empty_schema = empty_schema_with_time
        return pl.DataFrame(schema=empty_schema)
    indexed = lf.with_row_index('row_index')
    reports = []
    for column in columns:
        values = indexed.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
        bounds = values.select(pl.col(column).quantile(0.25).alias('q1'), pl.col(column).quantile(0.75).alias('q3')).collect()
        q1 = bounds['q1'][0]
        q3 = bounds['q3'][0]
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        selected = [pl.col('row_index'), pl.lit(column).alias('column'), pl.col(column).cast(pl.Float64).alias('value'), pl.lit(float(lower)).alias('lower_bound'), pl.lit(float(upper)).alias('upper_bound')]
        if time_column is not None:
            selected.insert(1, pl.col(time_column))
        detected = indexed.filter(((pl.col(column) < lower) | (pl.col(column) > upper)).fill_null(False)).select(selected).collect()
        reports.append(detected)
    if not reports:
        empty_schema = {'row_index': pl.UInt64, 'column': pl.String, 'value': pl.Float64, 'lower_bound': pl.Float64, 'upper_bound': pl.Float64}
        if time_column is not None:
            time_type = schema[time_column]
            empty_schema_with_time = {'row_index': pl.UInt64, time_column: time_type}
            for column_name, column_type in empty_schema.items():
                if column_name != 'row_index':
                    empty_schema_with_time[column_name] = column_type
            empty_schema = empty_schema_with_time
        return pl.DataFrame(schema=empty_schema)
    return pl.concat(reports, how='vertical')



# 2. Repérer les pics isolés qui pourraient être interpolés.
def inspect_iqr_interpolation_evidence(lf: pl.LazyFrame, outliers: pl.DataFrame, time_column: str | None, factor: float=1.5) -> list[dict]:
    """Find isolated IQR spikes with stable, observed temporal neighbors.

    The output contains row identities, not replacement values. This is
    evidence for a human decision, never proof of a sensor malfunction.
    """
    if outliers.is_empty() or time_column is None:
        return []
    schema = lf.collect_schema()
    if schema.get(time_column) not in (pl.Date, pl.Datetime):
        return []
    columns = sorted(set(outliers.get_column('column').to_list()))
    frame = lf.select([time_column, *columns]).collect()
    if frame.height < 5:
        return []
    times = frame.get_column(time_column).to_list()
    if any((t is None for t in times)):
        return []
    deltas = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    positive = [d for d in deltas if d > 0]
    if len(positive) != len(deltas):
        return []
    from collections import Counter
    cadence, freq = Counter((round(d, 6) for d in positive)).most_common(1)[0]
    if cadence <= 0 or freq / len(positive) < 0.8:
        return []
    by_column = {}
    for item in outliers.to_dicts():
        by_column.setdefault(item['column'], []).append(item)
    evidence = []
    for column, flags in by_column.items():
        values = frame.get_column(column).to_list()
        flagged = {int(x['row_index']) for x in flags}
        candidates = []
        rejected = {'edge_or_adjacent': 0, 'irregular_time': 0, 'missing_neighbors': 0, 'unstable_neighbors': 0}
        for item in flags:
            i = int(item['row_index'])
            if i < 1 or i >= len(values) - 1 or i - 1 in flagged or (i + 1 in flagged):
                rejected['edge_or_adjacent'] += 1
                continue
            if abs(deltas[i - 1] - cadence) > 0.05 * cadence or abs(deltas[i] - cadence) > 0.05 * cadence:
                rejected['irregular_time'] += 1
                continue
            left, right = (values[i - 1], values[i + 1])
            if any((v is None or not isinstance(v, (int, float)) or (not math.isfinite(v)) for v in (left, right))):
                rejected['missing_neighbors'] += 1
                continue
            width = float(item['upper_bound'] - item['lower_bound'])
            iqr = width / (2 * factor + 1)
            if iqr <= 0 or abs(float(right) - float(left)) > 0.5 * iqr:
                rejected['unstable_neighbors'] += 1
                continue
            estimate = (float(left) + float(right)) / 2
            candidates.append({'row_index': i, 'timestamp': str(times[i]), 'observed': float(item['value']), 'neighbor_left': float(left), 'neighbor_right': float(right), 'estimated_linear': estimate, 'iqr_lower': float(item['lower_bound']), 'iqr_upper': float(item['upper_bound'])})
        evidence.append({'column': column, 'flagged_count': len(flags), 'eligible_count': len(candidates), 'eligible_row_indices': [x['row_index'] for x in candidates], 'examples': candidates[:3], 'rejected': rejected, 'cadence_seconds': cadence, 'cadence_dominance': round(freq / len(positive), 3), 'rule': 'isolated IQR flag; two finite unflagged immediate neighbors; regular time steps; neighbor difference <= 0.5 IQR'})
    return evidence



# 3. Réaliser l’audit global du fichier.
def inspect_node(state: AgentState) -> dict:
    """Inspect the dataset and summarize IQR outliers."""
    lf = load_lazy_dataset(state['source_dataset_path'])
    report = inspect_dataset(file_path=state['source_dataset_path'], detailed=False, lf=lf)
    schema = lf.collect_schema()
    numeric_columns = [name for name, dtype in schema.items() if dtype.is_numeric()]
    timestamp_columns = report.get('temporal_info', {}).get('timestamp_columns', [])
    time_column = timestamp_columns[0] if len(timestamp_columns) == 1 else None
    outliers = detect_outliers_iqr(lf, columns=numeric_columns, time_column=time_column, factor=1.5)
    outlier_summary = outliers.group_by('column').agg(pl.len().alias('outlier_count')).to_dicts() if not outliers.is_empty() else []
    report['outlier_summary'] = outlier_summary
    report['iqr_interpolation_evidence'] = inspect_iqr_interpolation_evidence(lf, outliers, time_column, factor=1.5)
    inspect_additional_quality(lf, report)
    print_quality_audit(report)
    return {'quality_audit': report}

def print_quality_audit(report: dict) -> None:
    """Only show a short audit summary; details remain in the JSON report."""
    missing = [c for c in report.get('columns', []) if c.get('missing_count', 0)]
    outliers = [c for c in report.get('outlier_summary', []) if c.get('outlier_count', 0)]
    print(f'\nAudit : {len(missing)} colonnes avec lacunes, {len(outliers)} avec valeurs IQR atypiques.')
    if report.get('exact_duplicate_count'):
        print(f"Doublons exacts : {report['exact_duplicate_count']}")
    time_quality = report.get('time_quality', {})
    if time_quality.get('repeated_timestamp_count'):
        print(f"Horodatages répétés : {time_quality['repeated_timestamp_count']}")

def inspect_additional_quality(lf: pl.LazyFrame, report: dict) -> None:
    """Inspect temporal ordering, exact duplicates and category variants."""
    schema = lf.collect_schema()
    columns = list(schema)
    if not columns:
        return
    try:
        exact = lf.select(pl.struct(pl.all()).is_duplicated().sum().alias('count')).collect().item()
        report['exact_duplicate_count'] = int(exact or 0)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        print(f'[WARNING] Exact duplicate inspection skipped: {exc}', flush=True)
    temporal = report.get('temporal_info', {}).get('timestamp_columns', [])
    if len(temporal) == 1 and temporal[0] in schema:
        name = temporal[0]
        try:
            previous = pl.col(name).shift(1)
            metrics = lf.select((pl.col(name) < previous).fill_null(False).sum().alias('reversals'), pl.col(name).is_duplicated().sum().alias('repeated')).collect().row(0, named=True)
            report['time_quality'] = {'column': name, 'reversal_count': int(metrics['reversals'] or 0), 'repeated_timestamp_count': int(metrics['repeated'] or 0)}
        except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
            print(f'[WARNING] Temporal quality inspection skipped: {exc}', flush=True)
    variants = []
    for name, dtype in schema.items():
        if dtype not in (pl.String, pl.Categorical):
            continue
        try:
            values = lf.select(pl.col(name).drop_nulls().unique()).collect().to_series().to_list()
            if len(values) > 200:
                continue
            groups = {}
            for value in values:
                if not isinstance(value, str):
                    continue
                key = value.strip().casefold()
                groups.setdefault(key, []).append(value)
            collisions = [items for items in groups.values() if len(items) > 1]
            if collisions:
                variants.append({'column': name, 'examples': collisions[:3]})
        except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
            print(f'[WARNING] Category inspection skipped for {name}: {exc}', flush=True)
    report['category_variants'] = variants
    if len(temporal) == 1 and temporal[0] in schema:
        name = temporal[0]
        if schema[name] in (pl.Date, pl.Datetime):
            try:
                unique_times = lf.select(pl.col(name).drop_nulls().unique().sort()).collect().get_column(name)
                if len(unique_times) >= 3:
                    deltas = [(b - a).total_seconds() for a, b in zip(unique_times[:-1], unique_times[1:])]
                    positive = [d for d in deltas if d > 0]
                    if positive:
                        cadence, frequency = Counter(positive).most_common(1)[0]
                        dominance = frequency / len(positive)
                        gaps = [{'after': str(unique_times[i]), 'before': str(unique_times[i + 1]), 'missing_estimate': int(round(delta / cadence)) - 1} for i, delta in enumerate(deltas) if delta > cadence * 1.5 and abs(delta / cadence - round(delta / cadence)) < 0.05]
                        report['cadence_quality'] = {'column': name, 'cadence_seconds': cadence, 'dominance': round(dominance, 3), 'missing_timestamp_estimate': sum((g['missing_estimate'] for g in gaps)) if dominance >= 0.8 else None, 'gap_examples': gaps[:5] if dominance >= 0.8 else []}
            except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError) as exc:
                print(f'[WARNING] Cadence inspection skipped: {exc}', flush=True)



# 4. Vérifier les informations temporelles.
def validate_temporal_answer(answer: dict, report: dict) -> dict:
    """
    Validate temporal metadata supplied by the user.

    This checks the column name and declared metadata.
    It does not convert the dataset.
    """
    if not isinstance(answer, dict):
        raise ValueError('Expected a dictionary.')
    kind = answer.get('temporal_kind')
    column = answer.get('column')
    dtype = answer.get('target_dtype')
    fmt = answer.get('format')
    rate = answer.get('sampling_rate_hz')
    if kind not in TEMPORAL_TYPES:
        raise ValueError('Unknown temporal kind.')
    if dtype != TEMPORAL_TYPES[kind]:
        raise ValueError('Incorrect target data type.')
    columns = {item['name']: item['dtype'] for item in report['columns']}
    if kind in {'implicit', 'none'}:
        if column is not None:
            raise ValueError('No column should be specified for this temporal kind.')
    elif column not in columns:
        available = ', '.join(columns.keys())
        raise ValueError(f"Temporal column '{column}' was not found. Available columns: {available}")
    if fmt is not None and (not isinstance(fmt, str)):
        raise ValueError('Invalid temporal format.')
    if kind in {'datetime', 'date', 'time'}:
        if columns[column] in {'String', 'Utf8'} and (not fmt):
            raise ValueError('A format is required for textual dates/times.')
    if kind in {'implicit', 'sample_index'} and rate is None:
        raise ValueError('A sampling rate is required.')
    if rate is not None:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or (not math.isfinite(rate)) or (rate <= 0):
            raise ValueError('Sampling rate must be positive and finite.')
    return {'column': column, 'temporal_kind': kind, 'target_dtype': dtype, 'format': fmt, 'sampling_rate_hz': rate}



# 5. Demander une précision sur le temps si nécessaire.
def clarify_node(state: AgentState) -> dict:
    """
    Request temporal metadata only when necessary.

    If exactly one column already has a recognized temporal
    type, use it provisionally without changing the data.
    """
    report = state['quality_audit']
    temporal = report['temporal_info']
    detected = temporal.get('timestamp_columns', [])
    if len(detected) == 1:
        column = detected[0]
        dtype = next((item['dtype'] for item in report['columns'] if item['name'] == column))
        if dtype.startswith('Datetime'):
            kind = 'datetime'
        elif dtype == 'Date':
            kind = 'date'
        else:
            kind = 'time'
        return {'time_settings': {'column': column, 'temporal_kind': kind, 'target_dtype': TEMPORAL_TYPES[kind], 'format': None, 'sampling_rate_hz': None}}
    answer = interrupt({'type': 'temporal_clarification', 'message': 'Please specify the temporal column, its meaning, type and format.', 'columns': report['columns'], 'temporal_kinds': list(TEMPORAL_TYPES)})
    return {'time_settings': validate_temporal_answer(answer, report)}



# 6. Proposer des nettoyages avec des règles Python prudentes.
def recommend_node(state: AgentState) -> dict:
    """Build cautious, column-specific proposals from measured evidence."""
    report = state['quality_audit']
    time_column = state.get('time_settings', {}).get('column')
    complete = state.get('requires_complete_data', False)
    outliers = {x['column']: x['outlier_count'] for x in report.get('outlier_summary', [])}
    answers = state.get('user_answers', {})
    cause = answers.get('question_1', '').strip().lower()
    outlier_meaning = answers.get('question_2', '').strip().lower()
    recs = []

    def add(operation, column, reason, risk, parameters):
        recs.append({'id': f'rec_{len(recs) + 1}', 'operation': operation, 'column': column, 'reason': reason, 'risk': risk, 'parameters': parameters, 'requires_clarification': False})
    for col in report.get('columns', []):
        count = int(col.get('missing_count') or 0)
        if count == 0:
            continue
        name = col['name']
        dtype = str(col.get('dtype', ''))
        run = int(col.get('max_missing_run') or count)
        pct = float(col.get('null_percentage') or 0)
        numeric = dtype.startswith(('Int', 'UInt', 'Float', 'Decimal'))
        flagged = outliers.get(name, 0)
        if not numeric:
            method = 'keep'
            why = 'Catégorie : ne pas inventer de valeur sans connaître son sens.'
        elif name == time_column:
            method = 'keep'
            why = 'Horodatage : ne pas inventer de dates.'
        elif cause == 'panne' and run > 12:
            method = 'keep'
            why = 'Panne signalée et longue interruption : éviter une imputation artificielle.'
        elif cause == 'mesure' and run <= 3 and (pct <= 3) and time_column:
            method = 'linear'
            why = 'Mesure manquante ponctuelle : interpolation temporelle envisageable.'
        elif run <= 2 and pct <= 1 and time_column:
            method = 'linear'
            why = 'Très courte lacune : interpolation locale envisageable.'
        elif run <= 6 and pct <= 3:
            method = 'median' if flagged else 'mean'
            why = 'Lacunes courtes avec valeurs IQR atypiques : médiane robuste.' if flagged else 'Lacunes courtes sans valeurs IQR atypiques : moyenne possible.'
        elif run <= 12 and pct <= 5 and (not flagged):
            method = 'median'
            why = 'Lacunes modérées et peu nombreuses : médiane envisageable.'
        else:
            method = 'keep'
            why = 'Lacunes longues ou nombreuses : imputation non justifiée automatiquement.'
        if cause == 'panne' and run <= 12:
            why += ' Cause déclarée : panne ; vérifier la continuité avant imputation.'
        elif cause == 'inconnue':
            why += ' Cause inconnue : prudence sur les valeurs estimées.'
        if flagged and outlier_meaning == 'erreurs':
            why += ' Valeurs IQR suspectées erronées : examen conseillé, aucune suppression automatique.'
        elif flagged and outlier_meaning in {'pics réels', 'pics reels', 'réels', 'reels'}:
            why += ' Pics déclarés réels : les préserver ; IQR ne prouve pas une erreur.'
        if complete and method == 'keep':
            why += ' Le modèle exige des données complètes : décision métier nécessaire.'
        reason = f'{count} valeurs manquantes ({pct:g} %), lacune max {run}, {flagged} outliers IQR. {why}'
        add('handle_missing', name, reason, 'Une imputation peut modifier la distribution et les relations temporelles.', {'method': method})
    if report.get('exact_duplicate_count', 0):
        add('remove_duplicates', None, 'Lignes identiques détectées.', 'Certaines répétitions peuvent être voulues.', {'keep': 'first'})
    quality = report.get('time_quality', {})
    if quality.get('column') and quality.get('reversal_count', 0):
        add('sort_time', quality['column'], 'Horodatages désordonnés.', "Le tri change l'ordre des lignes.", {'ascending': True})
    cadence = report.get('cadence_quality', {})
    if cadence.get('missing_timestamp_estimate') and cadence.get('cadence_seconds'):
        add('regularize_timestamps', cadence['column'], 'Horodatages manquants.', 'La cadence doit être confirmée.', {'every_seconds': cadence['cadence_seconds'], 'fill_new_rows': 'null'})
    for item in report.get('category_variants', []):
        add('normalize_categories', item['column'], 'Variantes de catégories détectées.', 'La casse peut être significative.', {'strip_whitespace': True, 'case': 'lower'})
    return {'cleaning_proposals': recs, 'recommendation_summary': 'Propositions fondées sur l’audit et les réponses métier, ajustables par l’utilisateur.'}



# 7. Recueillir les réponses métier.
def clarify_recommendations_node(state: AgentState) -> dict:
    """Collect business context and optional adjustments once."""
    recs = state.get('cleaning_proposals', [])
    if not any((c.get('missing_count') for c in state['quality_audit'].get('columns', []))) and (not state['quality_audit'].get('outlier_summary')):
        return {}
    questions = ['Cause des lacunes (panne / mesure / inconnue) ?', 'Les outliers IQR sont-ils des pics réels, des erreurs, ou inconnus ?', 'Ajustements facultatifs colonne=méthode, séparés par des virgules (ex. ASE_NO2=median, QDP_CO=median, QDP_CO=outlier_drop). Méthodes : keep, median, mean, linear, forward, backward, drop, unknown (catégories), outlier_drop (suppression de lignes IQR). Entrée = conserver les propositions.']
    answer = interrupt({'type': 'cleaning_clarification', 'questions': questions, 'message': 'Trois réponses maximum, sans confirmation répétée.', 'recommendations': recs, 'outlier_columns': [x['column'] for x in state['quality_audit'].get('outlier_summary', [])], 'missing_columns': [c['name'] for c in state['quality_audit'].get('columns', []) if c.get('missing_count')], 'dtypes': {col['name']: str(col.get('dtype', '')) for col in state['quality_audit'].get('columns', [])}})
    if not isinstance(answer, dict):
        raise ValueError('Invalid clarification response')
    return {'user_answers': answer}



# 8. Appliquer les ajustements demandés par l’utilisateur.
def refine_recommendations_node(state: AgentState) -> dict:
    """Apply independent missing-value and outlier choices for each column."""
    answers = state.get('user_answers', {})
    report = state['quality_audit']
    dtypes = {c['name']: str(c.get('dtype', '')) for c in report.get('columns', [])}
    missing = {r['column'] for r in state.get('cleaning_proposals', []) if r['operation'] == 'handle_missing'}
    outliers = {x['column'] for x in report.get('outlier_summary', [])}
    choices = {}
    for part in answers.get('question_3', '').split(','):
        if not part.strip():
            continue
        column, method = (value.strip() for value in part.split('=', 1))
        method = method.lower()
        choices.setdefault(column, []).append(method)
    updated = []
    for rec in state.get('cleaning_proposals', []):
        item = dict(rec)
        column = item.get('column')
        if item['operation'] == 'handle_missing':
            methods = [m for m in choices.get(column, []) if m != 'outlier_drop']
            if methods:
                method = methods[0]
                item['parameters'] = {'method': 'constant', 'value': 'Unknown'} if method == 'unknown' else {'method': method}
                item['reason'] += f' Ajustement métier : {method}.'
        updated.append(item)
    for column, methods in choices.items():
        if 'outlier_drop' not in methods:
            continue
        if column not in outliers or not dtypes[column].startswith(('Int', 'UInt', 'Float', 'Decimal')):
            continue
        updated.append({'id': f'rec_{len(updated) + 1}', 'operation': 'handle_outliers', 'column': column, 'parameters': {'method': 'iqr', 'action': 'drop', 'factor': 1.5}, 'reason': 'Suppression IQR explicitement demandée pour cette colonne.', 'risk': 'Supprime des lignes entières et peut retirer des pics réels.', 'requires_clarification': False})
    return {'cleaning_proposals': updated, 'business_context': {'missing': answers.get('question_1', ''), 'outliers': answers.get('question_2', '')}}



# 9. Vérifier les paramètres de chaque opération.
def validate_operation(rec: dict, report: dict, temporal_config: dict) -> list[str]:
    """Return blocking reasons; an empty list means technically approvable.

    This checks schema and parameter compatibility, not statistical suitability.
    """
    errors = []
    columns = {item['name']: item['dtype'] for item in report['columns']}
    op, column = (rec.get('operation'), rec.get('column'))
    params = rec.get('parameters', {})
    numeric = ('Int', 'UInt', 'Float', 'Decimal')
    if op not in ALLOWED_OPERATIONS:
        return [f'Unsupported operation: {op}']
    if not isinstance(params, dict):
        return ['Parameters must be a dictionary']
    if column is not None and column not in columns:
        errors.append(f'Unknown column: {column}')
    if op in {'handle_missing', 'handle_nan', 'interpolate', 'handle_outliers', 'parse_time'} and column is None:
        errors.append('A target column is required')
    dtype = columns.get(column, '')
    is_numeric = dtype.startswith(numeric)
    method = params.get('method')
    if op == 'handle_missing':
        allowed = {'forward', 'backward', 'linear', 'mean', 'median', 'drop', 'keep', 'constant'}
        if method not in allowed:
            errors.append(f'Invalid missing-value method: {method}')
        if method in {'linear', 'mean', 'median'} and (not is_numeric):
            errors.append('Numeric imputation requires a numeric column')
        if method == 'constant' and params.get('value') is None:
            errors.append('Constant replacement requires a non-null value')
        if method == 'constant' and is_numeric and (isinstance(params.get('value'), bool) or not isinstance(params.get('value'), (int, float)) or (not math.isfinite(params['value']))):
            errors.append('Numeric constant must be a finite number')
    elif op == 'interpolate':
        if not is_numeric:
            errors.append('Interpolation requires a numeric column')
        if method not in {'linear', 'forward', 'backward'}:
            errors.append('Unsupported interpolation method')
    elif op == 'handle_nan':
        if not dtype.startswith('Float'):
            errors.append('NaN handling requires a float column')
        if method not in {'forward', 'backward', 'linear', 'mean', 'median', 'drop', 'keep', 'constant'}:
            errors.append('Unsupported NaN handling method')
        if method == 'constant' and (isinstance(params.get('value'), bool) or not isinstance(params.get('value'), (int, float)) or (not math.isfinite(params['value']))):
            errors.append('NaN replacement requires a finite numeric constant')
    elif op == 'sort_time':
        time_col = temporal_config.get('column')
        if not time_col or time_col not in columns:
            errors.append('No valid temporal column configured')
        if column not in {None, time_col}:
            errors.append('Sort column must match configured time column')
    elif op == 'parse_time':
        if not dtype:
            errors.append('Temporal column missing')
        if dtype in {'String', 'Utf8'} and (not (params.get('format') or temporal_config.get('format'))):
            errors.append('Parsing textual timestamps requires a format')
    elif op == 'resample':
        time_column = temporal_config.get('column')
        if not time_column or time_column not in columns:
            errors.append('Resampling requires a configured temporal column')
        if column not in {None, time_column}:
            errors.append('Resampling column must match the temporal column')
        if not isinstance(params.get('every'), str) or not params['every'].strip():
            errors.append('Resampling requires a nonempty interval')
        if method not in {'mean', 'sum', 'min', 'max', 'first', 'last'}:
            errors.append('Unsupported resampling aggregation')
        value_columns = params.get('value_columns')
        if not isinstance(value_columns, list) or not value_columns or any((not isinstance(name, str) or name not in columns or (not columns[name].startswith(('Int', 'UInt', 'Float', 'Decimal'))) for name in value_columns)) or (len(value_columns) != len(set(value_columns))):
            errors.append('Resampling requires explicitly selected numeric columns')
        if params.get('allow_column_loss') is not True:
            errors.append('Resampling requires explicit column-loss approval')
    elif op == 'handle_outliers':
        if not is_numeric:
            errors.append('Outlier handling requires a numeric column')
        if method not in OUTLIER_METHODS:
            errors.append('Unsupported outlier detection method')
        if params.get('action') not in OUTLIER_ACTIONS:
            errors.append("Outlier action must be 'keep' or 'drop'")
        factor = params.get('factor', 1.5)
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or (not math.isfinite(factor)) or (factor <= 0):
            errors.append('IQR factor must be positive and finite')
    elif op == 'regularize_timestamps':
        cadence = report.get('cadence_quality', {})
        if column != cadence.get('column'):
            errors.append('Regularization requires the inspected time column')
        if not cadence.get('missing_timestamp_estimate'):
            errors.append('No confirmed missing timestamps on a dominant cadence')
        if params.get('every_seconds') != cadence.get('cadence_seconds'):
            errors.append('Cadence must match inspected evidence')
        if params.get('fill_new_rows') != 'null':
            errors.append('New timestamps must have null measurements')
    elif op == 'normalize_categories':
        if dtype not in {'String', 'Categorical', 'Enum'}:
            errors.append('Category normalization requires a categorical column')
        if params.get('strip_whitespace') is not True:
            errors.append('Whitespace normalization must be explicit')
        if params.get('case') not in {'lower', 'upper', 'preserve'}:
            errors.append('Invalid category case normalization')
        if column not in {item.get('column') for item in report.get('category_variants', [])}:
            errors.append('No observed category variants justify normalization')
    elif op == 'remove_duplicates':
        if not report.get('exact_duplicate_count', 0):
            errors.append('No exact duplicates detected')
        if params.get('keep', 'first') not in {'first', 'last'}:
            errors.append('Invalid duplicate retention policy')
        subset = params.get('subset')
        if subset is not None:
            errors.append('Subset-based deduplication is not supported; only exact full-row duplicates can be removed')
    return errors



# 10. Valider techniquement les propositions.
def technical_validation_node(state: AgentState) -> dict:
    """Validate all recommendations and expose precise rejection reasons."""
    validated = []
    for rec in state.get('cleaning_proposals', []):
        reasons = validate_operation(rec, state['quality_audit'], state.get('time_settings', {}))
        warnings = ['Statistical uncertainty explicitly acknowledged by the user.'] if rec.get('risk_acknowledged') else ['Statistical justification remains uncertain.'] if rec.get('requires_clarification') else []
        validated.append({**rec, 'technical_status': 'rejected' if reasons else 'needs_clarification' if rec.get('requires_clarification') else 'valid', 'validation_errors': reasons, 'validation_warnings': warnings})
    return {'checked_proposals': validated}



# 11. Attendre l’approbation humaine.
def approval_node(state: AgentState) -> dict:
    """Validate optional completeness fixes and require one final approval."""
    recommendations = state.get('checked_proposals', [])
    approvable = [rec for rec in recommendations if rec['technical_status'] == 'valid' and (not (rec['operation'] == 'handle_missing' and rec.get('parameters', {}).get('method') == 'keep'))]
    answer = interrupt({'type': 'cleaning_approval', 'recommendations': recommendations, 'requires_complete': state.get('requires_complete_data', False), 'inspection_report': state.get('quality_audit', {}), 'business_interpretation': state.get('business_context', {})})
    if not isinstance(answer, dict):
        raise ValueError('Invalid approval response.')
    extra = answer.get('extra_recommendations', [])
    if not isinstance(extra, list):
        raise ValueError('Invalid additional recommendations.')
    existing_ids = {rec['id'] for rec in recommendations}
    checked = []
    for rec in extra:
        if not isinstance(rec, dict) or rec.get('id') in existing_ids:
            raise ValueError('Invalid or duplicate additional recommendation.')
        errors = validate_operation(rec, state['quality_audit'], state.get('time_settings', {}))
        if errors:
            raise ValueError('Invalid completeness operation: ' + '; '.join(errors))
        checked.append({**rec, 'technical_status': 'valid', 'validation_errors': []})
        existing_ids.add(rec['id'])
    approvable += checked
    approved_ids = answer.get('approved_ids', [])
    if not isinstance(approved_ids, list) or not all((isinstance(x, str) for x in approved_ids)) or len(approved_ids) != len(set(approved_ids)) or (not set(approved_ids) <= {rec['id'] for rec in approvable}):
        raise ValueError('Invalid approved IDs.')
    return {'cleaning_proposals': state.get('cleaning_proposals', []) + extra, 'checked_proposals': recommendations + checked, 'approved_operations': [rec for rec in approvable if rec['id'] in approved_ids]}



# 12. Relier les étapes dans LangGraph.
def build_graph():
    """Build the recommendation, clarification, and approval workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node('inspect', inspect_node)
    workflow.add_node('clarify', clarify_node)
    workflow.add_node('recommend', recommend_node)
    workflow.add_node('clarify_recommendations', clarify_recommendations_node)
    workflow.add_node('refine_recommendations', refine_recommendations_node)
    workflow.add_node('validate_technical', technical_validation_node)
    workflow.add_node('approve', approval_node)
    workflow.add_edge(START, 'inspect')
    workflow.add_edge('inspect', 'clarify')
    workflow.add_edge('clarify', 'clarify_recommendations')
    workflow.add_edge('clarify_recommendations', 'recommend')
    workflow.add_edge('recommend', 'refine_recommendations')
    workflow.add_edge('refine_recommendations', 'validate_technical')
    workflow.add_edge('validate_technical', 'approve')
    workflow.add_edge('approve', END)
    return workflow.compile(checkpointer=InMemorySaver())



# 13. Dialogues avec l’utilisateur dans le terminal.
def ask_temporal_answer(payload: dict) -> dict:
    """
    Ask the user to describe how time is represented.

    The user selects an existing column by number to avoid
    spelling mistakes and case-sensitivity problems.
    """
    print('\n=== TEMPORAL INFORMATION ===')
    print('\nThe agent needs to understand how time is represented in your dataset.')
    columns = payload['columns']
    print('\nAvailable columns in your dataset:')
    for index, item in enumerate(columns, start=1):
        print(f"  {index}. {item['name']} (detected type: {item['dtype']})")
    print('\nHow is time represented?')
    choices = {'1': ('datetime', 'Datetime'), '2': ('date', 'Date'), '3': ('time', 'Time'), '4': ('unix_s', 'Int64'), '5': ('unix_ms', 'Int64'), '6': ('elapsed_s', 'Float64'), '7': ('sample_index', 'Int64'), '8': ('implicit', None), '9': ('none', None)}
    descriptions = {'1': 'Date and time (e.g. 2026-01-01 14:30:00)', '2': 'Date only (e.g. 2026-01-01)', '3': 'Time only (e.g. 14:30:00)', '4': 'Unix timestamp in seconds', '5': 'Unix timestamp in milliseconds', '6': 'Elapsed time in seconds (e.g. 0.01, 0.02)', '7': 'Sample index (e.g. 0, 1, 2, 3)', '8': 'No time column, but sampling rate is known', '9': 'No time information is available'}
    for key, description in descriptions.items():
        print(f'  {key}. {description}')
    while True:
        choice = input('\nChoose the number describing your time data: ').strip()
        if choice in choices:
            break
        print('Invalid choice. Enter a number from 1 to 9.')
    kind, target_dtype = choices[choice]
    column = None
    if kind not in {'implicit', 'none'}:
        while True:
            selected = input('\nEnter the NUMBER of the column containing this time information: ').strip()
            if selected.isdigit():
                index = int(selected) - 1
                if 0 <= index < len(columns):
                    column = columns[index]['name']
                    break
            print('Invalid column number. Select one from the list above.')
    print(f"\nExpected temporal type: {target_dtype or 'not applicable'}")
    print('This is the type the agent will use to interpret the temporal information.')
    fmt = None
    if kind in {'datetime', 'date', 'time'}:
        print('\nExamples of date/time formats:')
        print('  %Y-%m-%d           -> 2026-01-01')
        print('  %d/%m/%Y           -> 01/01/2026')
        print('  %Y-%m-%d %H:%M:%S  -> 2026-01-01 14:30:00')
        print('  %H:%M:%S           -> 14:30:00')
        print('\nIf the selected column is already recognized as Date, Datetime or Time, press Enter.')
        fmt = input('Otherwise, enter the format used by the values in that column: ').strip() or None
    rate = None
    if kind in {'implicit', 'sample_index'}:
        print('\nSampling rate means the number of measurements recorded per second.')
        print('Example: 1000 Hz means 1000 measurements per second.')
        while True:
            raw = input('Enter the sampling rate in Hz (e.g. 1000): ').strip()
            try:
                rate = float(raw)
                if math.isfinite(rate) and rate > 0:
                    break
            except ValueError:
                pass
            print('Please enter a positive number (e.g. 1000 or 50000).')
    print('\n=== YOUR TEMPORAL CONFIGURATION ===')
    print(f"Column: {column or 'none'}")
    print(f'Temporal meaning: {kind}')
    print(f'Target type: {target_dtype}')
    print(f"Format: {fmt or 'not specified'}")
    print(f"Sampling rate: {rate or 'not specified'}")
    return {'column': column, 'temporal_kind': kind, 'target_dtype': target_dtype, 'format': fmt, 'sampling_rate_hz': rate}

def ask_cleaning_clarification(payload: dict) -> dict:
    """Ask two business questions and validate all adjustments in one input."""
    questions = payload['questions']
    dtypes = payload['dtypes']
    missing = set(payload.get('missing_columns', []))
    outliers = set(payload.get('outlier_columns', []))
    print('\n--- Questions métier ---')
    answers = {'question_1': input(f'{questions[0]}\n> ').strip(), 'question_2': input(f'{questions[1]}\n> ').strip()}
    while True:
        raw = input(f'{questions[2]}\n> ').strip()
        errors, seen = ([], set())
        for part in filter(None, (x.strip() for x in raw.split(','))):
            if '=' not in part:
                errors.append(f'Format incorrect : {part} (colonne=méthode).')
                continue
            column, method = (x.strip() for x in part.split('=', 1))
            method = method.lower()
            if column not in dtypes:
                hint = difflib.get_close_matches(column, dtypes, n=1, cutoff=0.6)
                errors.append(f'Colonne inconnue : {column}. ' + (f'Vouliez-vous dire {hint[0]} ?' if hint else ''))
                continue
            if (column, method) in seen:
                errors.append(f'Ajustement répété : {column}={method}.')
                continue
            seen.add((column, method))
            numeric = dtypes[column].startswith(('Int', 'UInt', 'Float', 'Decimal'))
            allowed = {'keep', 'median', 'mean', 'linear', 'forward', 'backward', 'drop'} if numeric else {'keep', 'forward', 'backward', 'drop', 'unknown'}
            if method == 'outlier_drop' and column in outliers and numeric:
                continue
            if column not in missing or method not in allowed:
                errors.append(f'Méthode non applicable : {column}={method}.')
        missing_methods = {}
        for column, method in seen:
            if method != 'outlier_drop':
                missing_methods.setdefault(column, set()).add(method)
        for column, methods in missing_methods.items():
            if len(methods) > 1:
                errors.append(f"Plusieurs méthodes pour les lacunes de {column} : {', '.join(sorted(methods))}.")
        if errors:
            print('\n--- Saisie à corriger ---')
            for error in errors:
                print('-', error)
            print('Les deux premières réponses sont conservées.\n')
            continue
        answers['question_3'] = raw
        return answers



# 14. Afficher les choix et avertissements avant confirmation.
def ask_approval(payload: dict) -> dict:
    """Offer an optional completeness fix, then approve all operations once."""
    print('\n--- Opérations proposées ---')
    candidates = []
    for rec in payload.get('recommendations', []):
        label = rec.get('column') or 'global'
        if rec.get('technical_status') != 'valid':
            print(f"[NON APPLICABLE] {rec['id']} ({label}): " + '; '.join(rec.get('validation_errors', [])))
            continue
        if rec['operation'] == 'handle_missing' and rec.get('parameters', {}).get('method') == 'keep':
            continue
        candidates.append(rec)
        print(f"[PROPOSÉ] {rec['id']} | {label}: {rec['operation']} {rec.get('parameters', {})}")
        print(f"  Motif : {rec.get('reason', '')}")
        print()
    extra = []
    if payload.get('requires_complete'):
        affected = {c['name']: str(c.get('dtype', '')) for c in payload.get('inspection_report', {}).get('columns', []) if (c.get('missing_count') or 0) > 0}
        treated = {rec.get('column') for rec in candidates if rec['operation'] in {'handle_missing', 'handle_nan', 'interpolate'} and rec.get('parameters', {}).get('method') != 'keep'}
        unresolved = {name: dtype for name, dtype in affected.items() if name not in treated}
        print('\n--- Exigence de données complètes ---')
        if unresolved:
            print('Colonnes encore sans traitement des lacunes : ' + ', '.join(unresolved))
            print('Ces colonnes peuvent rester incomplètes après nettoyage.')
            print('1 = Ajouter des imputations ; 2 = Supprimer les lignes incomplètes ; 3 = Conserver les lacunes')
            while True:
                choice = input('Votre choix (1/2/3) : ').strip()
                if choice in {'1', '2', '3'}:
                    break
                print('Choisissez 1, 2 ou 3.')
            if choice == '1':
                print('Saisissez colonne=méthode pour chaque colonne à traiter.')
                print('Exemple : ASE_NO=median, Location=unknown. Entrée = conserver les lacunes.')
                methods = {}
                while True:
                    raw = input('Imputations supplémentaires : ').strip()
                    if not raw:
                        print('Aucune imputation ajoutée : lacunes conservées.')
                        break
                    methods, errors = ({}, [])
                    for part in filter(None, (p.strip() for p in raw.split(','))):
                        if '=' not in part:
                            errors.append(f'Format invalide : {part}')
                            continue
                        column, method = (x.strip() for x in part.split('=', 1))
                        method = method.lower()
                        if column not in unresolved:
                            errors.append(f'Colonne inconnue ou déjà traitée : {column}')
                            continue
                        numeric = unresolved[column].startswith(('Int', 'UInt', 'Float', 'Decimal'))
                        allowed = {'median', 'mean', 'linear', 'forward', 'backward'} if numeric else {'unknown', 'forward', 'backward'}
                        if method not in allowed or column in methods:
                            errors.append(f'Méthode invalide ou répétée : {column}={method}')
                        else:
                            methods[column] = method
                    if not errors and methods:
                        break
                    print('; '.join(errors) if errors else 'Indiquez au moins une imputation.')
                for column, method in methods.items():
                    params = {'method': 'constant', 'value': 'Unknown'} if method == 'unknown' else {'method': method}
                    extra.append({'operation': 'handle_missing', 'column': column, 'parameters': params, 'reason': 'Traitement supplémentaire demandé pour obtenir des données complètes.', 'risk': "L'imputation peut déformer les mesures, surtout sur de longues lacunes."})
            elif choice == '2':
                for column in unresolved:
                    extra.append({'operation': 'handle_missing', 'column': column, 'parameters': {'method': 'drop'}, 'reason': 'Suppression des lignes incomplètes explicitement demandée.', 'risk': 'Des lignes entières et leurs mesures seront perdues.'})
            else:
                print('Lacunes conservées : le Parquet pourra rester incomplet.')
        else:
            print('Toutes les colonnes incomplètes ont un traitement proposé.')
            print('La complétude réelle sera vérifiée après export.')
    used_ids = {rec['id'] for rec in payload.get('recommendations', [])}
    next_id = 1
    for rec in extra:
        while f'rec_{next_id}' in used_ids:
            next_id += 1
        rec['id'] = f'rec_{next_id}'
        used_ids.add(rec['id'])
        next_id += 1
        candidates.append(rec)
        print(f"[AJOUTÉ] {rec['id']} | {rec['column']}: {rec['parameters']}")
        print()
    if not candidates:
        print('Aucune opération proposée : export Parquet sans nettoyage.')
        return {'approved_ids': [], 'extra_recommendations': extra}
    choices = {rec['id'] for rec in candidates} | {rec['column'] for rec in candidates if rec.get('column')}
    while True:
        raw = input('Colonnes ou IDs à REFUSER (virgules, Entrée = aucune) : ').strip()
        rejected = {x.strip() for x in raw.split(',') if x.strip()}
        unknown = rejected - choices
        if not unknown:
            break
        print('Noms ou IDs inconnus :', ', '.join(sorted(unknown)))
    chosen = [rec for rec in candidates if rec['id'] not in rejected and rec.get('column') not in rejected]
    print(f'Opérations sélectionnées : {len(chosen)}')
    if not chosen:
        return {'approved_ids': [], 'extra_recommendations': extra}
    business = payload.get('business_interpretation', {})
    cause = str(business.get('missing', '')).strip().lower()
    meaning = str(business.get('outliers', '')).strip().lower()
    columns = {col['name']: col for col in payload.get('inspection_report', {}).get('columns', [])}
    warnings = []
    for rec in chosen:
        name = rec.get('column')
        operation = rec.get('operation')
        method = rec.get('parameters', {}).get('method')
        if operation == 'handle_outliers' and meaning in {'pics réels', 'pics reels', 'réels', 'reels', 'pics', 'réel', 'reel'} and (rec.get('parameters', {}).get('action') == 'drop'):
            warnings.append(f"{rec['id']} ({name}) : pics déclarés réels, mais suppression IQR demandée ; des mesures valides peuvent être perdues.")
        if cause == 'panne' and operation in {'handle_missing', 'handle_nan', 'interpolate'} and (method in {'linear', 'forward', 'backward'}):
            run = int(columns.get(name, {}).get('max_missing_run') or 0)
            if run > 12:
                warnings.append(f"{rec['id']} ({name}) : propagation/interpolation sur une panne de {run} mesures consécutives ; valeurs artificielles possibles.")
    if warnings:
        print('\n--- Avertissements avant approbation ---')
        for warning in warnings:
            print('[ATTENTION] ' + warning)
        print('La confirmation OUI ci-dessous vaut aussi acceptation de ces risques.')
    confirmation = input('Confirmer leur exécution ? (taper OUI) : ').strip().upper()
    return {'approved_ids': [rec['id'] for rec in chosen] if confirmation == 'OUI' else [], 'extra_recommendations': extra}

def summarize_missing_policy(state: dict) -> dict:
    """Record why missing values are retained or proposed for imputation."""
    report = state.get('quality_audit', {})
    affected = [{'column': col['name'], 'missing_count': col.get('missing_count', 0), 'null_count': col.get('null_count', 0), 'nan_count': col.get('nan_count', 0), 'max_missing_run': col.get('max_missing_run')} for col in report.get('columns', []) if (col.get('missing_count', 0) or 0) > 0]
    response = state.get('completeness_response', 'unknown')
    proposals = [rec for rec in state.get('cleaning_proposals', []) if rec.get('operation') in {'handle_missing', 'handle_nan'}]
    approved_ids = {rec['id'] for rec in state.get('approved_operations', [])}
    if not affected:
        decision = 'No missing values detected; no imputation needed.'
    elif response == 'no':
        decision = 'Missing values retained: downstream completeness is not required.'
    elif response == 'unknown':
        decision = 'Missing values retained pending downstream requirements.'
    elif not proposals:
        decision = 'Complete data required, but no sufficiently justified imputation method was identified; unresolved gaps remain.'
    elif not any((rec['id'] in approved_ids for rec in proposals)):
        decision = 'Complete data required; imputation candidates exist but none was approved. Missing values remain unresolved.'
    else:
        decision = 'Complete data required; approved imputation proposals may address some gaps. Completeness is not guaranteed.'
    return {'downstream_completeness_requirement': response, 'affected_columns': affected, 'imputation_proposal_ids': [rec['id'] for rec in proposals], 'approved_imputation_ids': [rec['id'] for rec in proposals if rec['id'] in approved_ids], 'decision': decision}



# 15. Conserver les décisions dans un rapport JSON.
def export_recommendation_report(state: dict, dataset_path: str) -> Path:
    """Export complete audit and decision evidence without altering the dataset."""
    output_dir = Path(dataset_path).parent / 'audit_reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'recommendations_{uuid.uuid4().hex[:12]}.json'
    payload = {'dataset': Path(dataset_path).name, 'inspection_report': state.get('quality_audit', {}), 'temporal_config': state.get('time_settings', {}), 'missing_values_policy': summarize_missing_policy(state), 'recommendations': state.get('cleaning_proposals', []), 'validated_recommendations': state.get('checked_proposals', []), 'approved_recommendations': state.get('approved_operations', []), 'clarification_answers': state.get('user_answers', {}), 'business_interpretation': state.get('business_context', {}), 'summary': state.get('recommendation_summary', ''), 'dataset_modified': False}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    return output_path

def dataset_quality_summary(lf: pl.LazyFrame) -> dict:
    """Count rows and missing values (null or NaN) in each column."""
    schema = lf.collect_schema()
    expressions = [pl.len().alias('__row_count')]
    for name, dtype in schema.items():
        missing = pl.col(name).is_null()
        if dtype in (pl.Float32, pl.Float64):
            missing = missing | pl.col(name).is_nan().fill_null(False)
        expressions.append(missing.sum().alias(name))
    counts = lf.select(expressions).collect().row(0, named=True)
    return {'rows': counts.pop('__row_count'), 'missing': {name: int(count) for name, count in counts.items() if count}}



# 16. Vérifier le Parquet produit.
def print_postprocessing_check(file_path: str, parquet_path: str, complete_required: bool, approved: list[dict]) -> dict:
    """Read exported Parquet and check observable effects, without changing files."""
    before = dataset_quality_summary(load_lazy_dataset(file_path))
    after = dataset_quality_summary(pl.scan_parquet(str(parquet_path)))
    original_columns = set(load_lazy_dataset(file_path).collect_schema().names())
    exported_columns = set(pl.scan_parquet(str(parquet_path)).collect_schema().names())
    if original_columns != exported_columns:
        print('[ATTENTION] Colonnes différentes dans le Parquet exporté.')
    print('\n--- Vérification du Parquet exporté ---')
    print(f"Lignes : {before['rows']} → {after['rows']}")
    print(f'Opérations approuvées : {len(approved)}')
    print(f"Colonnes encore incomplètes : {len(after['missing'])}")
    if after['missing']:
        print('Valeurs manquantes restantes : ' + ', '.join((f'{name}={count}' for name, count in after['missing'].items())))
    warnings = []
    for rec in approved:
        operation = rec.get('operation')
        column = rec.get('column')
        method = rec.get('parameters', {}).get('method')
        old = before['missing'].get(column, 0)
        new = after['missing'].get(column, 0)
        if operation in {'handle_missing', 'handle_nan', 'interpolate'} and method != 'keep':
            if new and new >= old and (method != 'drop'):
                warnings.append(f'{column} ({method}) : aucune réduction nette des lacunes')
            elif new:
                warnings.append(f'{column} ({method}) : {new} lacunes restantes')
        elif operation == 'handle_outliers' and rec.get('parameters', {}).get('action') == 'drop':
            if after['rows'] == before['rows']:
                warnings.append(f'{column} : aucune ligne supprimée au total ; vérifier le filtre IQR')
    if complete_required and after['missing']:
        warnings.append('Exigence de données complètes NON satisfaite')
    for warning in dict.fromkeys(warnings):
        print('[À VÉRIFIER] ' + warning)
    if not warnings:
        print('Aucune anomalie détectée par les contrôles de résultat.')
    print('Note : le bilan global ne prouve pas chaque opération isolément, notamment quand plusieurs filtres suppriment des lignes.')
    return {'before': before, 'after': after, 'warnings': warnings}



# 17. Exécuter le parcours complet sans modifier la source.
def main():
    """
    Run the data preparation and cleaning approval stage in the terminal.

    The dataset is automatically located in dataset/sample/.
    Each team member provides their own Ollama URL.
    """
    print('=== Data analysis agent: preparation stage ===')
    project_root = Path(__file__).resolve().parents[3]
    sample_dir = project_root / 'time-series-agent' / 'datasets' / 'sample'
    if not sample_dir.is_dir():
        alternative = project_root / 'datasets' / 'sample'
        if alternative.is_dir():
            sample_dir = alternative
    if not sample_dir.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {sample_dir}')
    supported_extensions = {'.csv', '.parquet', '.xlsx', '.xls'}
    datasets = sorted((path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() in supported_extensions))
    if not datasets:
        raise FileNotFoundError(f'No supported dataset found in {sample_dir}')
    if len(datasets) > 1:
        raise ValueError('Multiple datasets found in dataset/sample/. Keep only one dataset for automatic selection.')
    file_path = str(datasets[0])
    with open(file_path, 'rb') as source:
        original_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    print(f'\nDataset detected: {datasets[0].name}')
    graph = build_graph()
    config = {'configurable': {'thread_id': str(uuid.uuid4())}}
    print('\nMissing values are often supported by analysis libraries and plots, but some downstream models require complete data.')
    while True:
        response = input('Do downstream steps require complete data without null/NaN? (yes/no/unknown): ').strip().lower()
        if response in {'yes', 'no', 'unknown', ''}:
            break
        print('Answer yes, no or unknown.')
    result = graph.invoke({'source_dataset_path': file_path, 'requires_complete_data': response == 'yes', 'completeness_response': response or 'unknown'}, config=config)
    while result.get('__interrupt__'):
        payload = result['__interrupt__'][0].value
        request_type = payload.get('type')
        if request_type == 'temporal_clarification':
            report = graph.get_state(config).values['quality_audit']
            while True:
                answer = ask_temporal_answer(payload)
                try:
                    validate_temporal_answer(answer, report)
                    break
                except ValueError as error:
                    print(f'Invalid answer: {error}')
        elif request_type == 'cleaning_clarification':
            answer = ask_cleaning_clarification(payload)
        elif request_type == 'cleaning_approval':
            answer = ask_approval(payload)
        else:
            raise ValueError(f'Unknown interruption: {request_type}')
        result = graph.invoke(Command(resume=answer), config=config)
    invalid = [rec for rec in result.get('checked_proposals', []) if rec.get('technical_status') != 'valid' and rec.get('parameters', {}).get('method') != 'keep']
    for rec in invalid:
        print(f"[NON APPLICABLE] {rec['id']} : " + '; '.join(rec.get('validation_errors', [])))
    approved = result.get('approved_operations', [])
    report_path = export_recommendation_report(result, file_path)
    output_dir = sample_dir.parent / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'rb') as source:
        current_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    if current_hash != original_hash:
        raise RuntimeError('Source dataset changed after inspection.')
    if approved:
        processed = run_preprocessing(file_path=file_path, approved_recommendations=approved, expected_source_sha256=original_hash, temporal_config=result.get('time_settings', {}), output_dir=str(output_dir))
        output_path = processed['output_file_path']
    else:
        output_path = output_dir / f'{Path(file_path).stem}_processed_{uuid.uuid4().hex[:12]}.parquet'
        temporary = output_path.with_suffix('.tmp.parquet')
        try:
            load_lazy_dataset(file_path).sink_parquet(temporary)
            with open(file_path, 'rb') as source:
                current_hash = hashlib.file_digest(source, 'sha256').hexdigest()
            if current_hash != original_hash:
                raise RuntimeError('Source dataset changed during export.')
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
    print_postprocessing_check(file_path, str(output_path), bool(result.get('requires_complete_data')), approved)
    with open(file_path, 'rb') as source:
        final_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    if final_hash != original_hash:
        raise RuntimeError('Le fichier source a changé pendant le traitement.')
    print(f'Parquet : {output_path}')
    print(f'Rapport JSON : {report_path}')
    print('Source inchangée (SHA256 vérifié).')

if __name__ == '__main__':
    main()