"""Plotly figures for time series stored as Parquet files.

Every figure function takes a parquet path and returns a plotly Figure.
Large datasets are thinned out so that the figures stay light.
`save_figure` writes a figure as JSON, so that the UI can reload it with
`plotly.io.read_json` (the LLM never receives the figure itself).
"""

import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MAX_POINTS = 5_000  # max points per curve in a line chart
MAX_SAMPLE = 10_000  # max rows for histogram / boxplot / scatter

DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")
DATE_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?")


# ============================================================
# HELPERS
# ============================================================

def check_columns(df, columns):
    """Raise an error if a column does not exist."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )


def check_numeric(df, columns):
    """Raise an error if a column is not numeric (booleans are rejected)."""
    for c in columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            raise ValueError(f"Column '{c}' is not numeric.")


def check_value_columns(df, columns):
    """Check that the columns are given, exist and are numeric."""
    if not columns:
        raise ValueError("At least one column must be provided.")
    check_columns(df, columns)
    check_numeric(df, columns)


def sample_rows(data, n=MAX_SAMPLE):
    """Random sample (same result every time) if the data is too big."""
    if len(data) > n:
        return data.sample(n=n, random_state=0)
    return data


def decimate_minmax(df, value_cols, n=MAX_POINTS):
    """Thin out a time-ordered DataFrame while keeping the spikes.

    The rows are split in consecutive buckets and, in each bucket, the rows
    holding the min and the max of every value column are kept. A plain
    "one row out of k" can erase an isolated peak, this cannot.
    Note: where a bucket keeps no value (all missing), the gap is not drawn.
    """
    if len(df) <= n:
        return df

    n_buckets = max(1, n // (2 * len(value_cols)))
    bucket = np.arange(len(df)) * n_buckets // len(df)

    keep = set()
    for col in value_cols:
        s = pd.Series(df[col].to_numpy(dtype="float64"))
        valid = s.notna().to_numpy()
        if not valid.any():
            continue
        g = s[valid].groupby(bucket[valid])
        keep.update(g.idxmin().tolist())
        keep.update(g.idxmax().tolist())

    if not keep:
        return df.iloc[:: -(-len(df) // n)]
    return df.iloc[sorted(keep)]


def parse_bound(value, tz, is_end):
    """Turn a user date string into a (Timestamp, inclusive_end) pair.

    Accepted: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM[:SS]" (optionally with an
    offset). A date-only end covers the whole day. If the column has a time
    zone, a bound without offset is read in that time zone.
    """
    text = str(value).strip()
    date_only = bool(DATE_ONLY.fullmatch(text))
    if not (date_only or DATE_TIME.fullmatch(text)):
        raise ValueError(
            f"Invalid date '{value}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
        )

    ts = pd.Timestamp(text)
    if tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    elif tz is None and ts.tzinfo is not None:
        raise ValueError(
            f"Date '{value}' has a UTC offset but the time column has no time zone."
        )

    if is_end and date_only:
        return ts + pd.Timedelta(days=1), False  # exclusive: whole day included
    return ts, True


def load_data(parquet_path, time_col=None, start_date=None, end_date=None, columns=None):
    """Read the parquet file and optionally filter it on the time column.

    columns: columns to read (None = all). time_col is added automatically.
    start_date / end_date: see parse_bound. If the time column is numeric
    (seconds, sample index...), they are read as numbers.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    import pyarrow.parquet as pq

    available = pq.ParquetFile(path).schema_arrow.names
    to_read = None
    if columns is not None:
        wanted = list(columns) + ([time_col] if time_col else [])
        missing = [c for c in wanted if c not in available]
        if missing:
            raise ValueError(
                f"Missing columns: {missing}. Available columns: {available}"
            )
        to_read = list(dict.fromkeys(wanted))  # unique, order kept

    df = pd.read_parquet(path, columns=to_read)
    if df.empty:
        raise ValueError("The file contains no data.")

    if time_col is None:
        if start_date is not None or end_date is not None:
            raise ValueError("time_col is required to filter by date.")
        return df

    check_columns(df, [time_col])
    is_number = pd.api.types.is_numeric_dtype(df[time_col]) and not pd.api.types.is_bool_dtype(
        df[time_col]
    )

    if not is_number:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except (ValueError, TypeError) as error:
            raise ValueError(f"Column '{time_col}' cannot be read as dates.") from error

    if is_number:
        if start_date is not None:
            df = df[df[time_col] >= float(start_date)]
        if end_date is not None:
            df = df[df[time_col] <= float(end_date)]
    else:
        tz = df[time_col].dt.tz
        if start_date is not None:
            start, _ = parse_bound(start_date, tz, is_end=False)
            df = df[df[time_col] >= start]
        if end_date is not None:
            end, inclusive = parse_bound(end_date, tz, is_end=True)
            df = df[df[time_col] <= end] if inclusive else df[df[time_col] < end]

    if df.empty:
        raise ValueError("No data found for the selected dates.")

    return df.sort_values(time_col)


def save_figure(fig, output_dir, prefix="fig"):
    """Write the figure as JSON and return the path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}_{uuid4().hex[:8]}.json"
    path.write_text(fig.to_json(), encoding="utf-8")
    return path


# =========================== FIGURES ============================

def plot_time_series(
    parquet_path,
    time_col,
    value_cols,
    start_date=None,
    end_date=None,
    normalize=False,
):
    """Plot one or more variables over time.

    If time_col is None, the row number is used as x axis.
    normalize=True rescales each curve (mean 0, std 1) so that
    curves with very different scales can be compared.
    """
    if not value_cols:
        raise ValueError("At least one column must be provided.")
    df = load_data(parquet_path, time_col, start_date, end_date, columns=value_cols)
    check_value_columns(df, value_cols)

    df = decimate_minmax(df, value_cols)

    if time_col is None:
        x = list(range(len(df)))
    else:
        x = df[time_col]

    fig = go.Figure()

    for col in value_cols:
        y = df[col]
        if normalize and y.std() > 0:
            y = (y - y.mean()) / y.std()

        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=col))

    fig.update_layout(
        title="Time Series",
        xaxis_title=time_col or "row number",
        yaxis_title="Normalized value" if normalize else "Value",
        hovermode="x unified",
    )
    return fig


def plot_histogram(
    parquet_path,
    column,
    bins=30,
    time_col=None,
    start_date=None,
    end_date=None,
):
    """Plot the distribution of one numeric variable."""
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError("bins must be a positive integer.")

    df = load_data(parquet_path, time_col, start_date, end_date, columns=[column])
    check_value_columns(df, [column])

    values = sample_rows(df[column].dropna())

    fig = go.Figure(go.Histogram(x=values, nbinsx=bins, name=column))
    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Frequency",
    )
    return fig


def plot_boxplot(
    parquet_path,
    columns,
    time_col=None,
    start_date=None,
    end_date=None,
    group_by=None,
):
    """Plot the distribution of one or more variables.

    group_by: name of a categorical column (e.g. "machine"). One box is drawn
    per group. It needs a single value column.
    """
    if group_by is not None and len(columns or []) != 1:
        raise ValueError("group_by needs exactly one value column.")

    read = list(columns or []) + ([group_by] if group_by else [])
    df = load_data(parquet_path, time_col, start_date, end_date, columns=read)
    check_value_columns(df, columns)

    fig = go.Figure()

    if group_by is None:
        for col in columns:
            values = sample_rows(df[col].dropna())
            fig.add_trace(go.Box(y=values, name=col, boxmean=True))
    else:
        check_columns(df, [group_by])
        col = columns[0]
        for key, part in df.groupby(group_by, sort=True):
            values = sample_rows(part[col].dropna())
            fig.add_trace(go.Box(y=values, name=str(key), boxmean=True))

    fig.update_layout(
        title=f"Boxplot of {columns[0]} by {group_by}" if group_by else "Boxplot",
        yaxis_title="Value",
    )
    return fig


def plot_scatter(
    parquet_path,
    x_col,
    y_col,
    time_col=None,
    start_date=None,
    end_date=None,
):
    """Plot the relationship between two numeric variables."""
    df = load_data(parquet_path, time_col, start_date, end_date, columns=[x_col, y_col])
    check_value_columns(df, [x_col, y_col])

    pairs = sample_rows(df[[x_col, y_col]].dropna())

    fig = go.Figure(
        go.Scattergl(
            x=pairs[x_col],
            y=pairs[y_col],
            mode="markers",
            name=f"{x_col} vs {y_col}",
        )
    )
    fig.update_layout(
        title=f"{x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig


def plot_correlation_heatmap(
    parquet_path,
    columns=None,
    time_col=None,
    start_date=None,
    end_date=None,
):
    """Plot the correlation matrix of numeric variables.

    If columns is None, all numeric columns are used (except time_col).
    """
    df = load_data(parquet_path, time_col, start_date, end_date, columns=columns)

    if columns is None:
        numeric_df = df.select_dtypes(include="number")
        if time_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[time_col])
    else:
        check_value_columns(df, columns)
        numeric_df = df[columns]

    if numeric_df.shape[1] < 2:
        raise ValueError("At least two numeric columns are required.")

    corr = numeric_df.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar_title="Correlation",
        )
    )
    fig.update_layout(title="Correlation Heatmap")
    return fig