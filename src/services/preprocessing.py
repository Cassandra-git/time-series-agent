
"""Polars preprocessing services and approved-operation execution."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import os
from pathlib import Path
from uuid import uuid4

import polars as pl

from src.services.data_loader import load_lazy_dataset


def _is_temporal(dtype: pl.DataType) -> bool:
    """Identify Date and Datetime types, including parameterized Datetime."""
    return dtype.base_type() in (pl.Date, pl.Datetime)


def remove_duplicates(
    lf: pl.LazyFrame,
    keep: str = "first",
) -> pl.LazyFrame:
    """Remove exact duplicate rows in a stable order."""
    if keep not in {"first", "last"}:
        raise ValueError("Unsupported duplicate retention strategy.")

    return lf.unique(keep=keep, maintain_order=True)


def time_formating(
    lf: pl.LazyFrame,
    time_column: str,
    time_format: str | None,
    approved_operations: list[str],
) -> pl.LazyFrame:
    """Parse and/or sort the specified temporal column."""
    schema = lf.collect_schema()

    if time_column not in schema:
        raise ValueError(f"Unknown time column: {time_column}")

    if not approved_operations or not set(approved_operations).issubset(
        {"parse_time", "sort_time"}
    ):
        raise ValueError("Unsupported temporal operation.")

    if "parse_time" in approved_operations:
        dtype = schema[time_column]

        if dtype == pl.String:
            if not time_format:
                raise ValueError("An explicit timestamp format is required.")

            lf = lf.with_columns(
                pl.col(time_column).str.strptime(
                    pl.Datetime,
                    format=time_format,
                    strict=True,
                )
            )

        elif not _is_temporal(dtype):
            raise TypeError("Timestamp column cannot be parsed.")

    if "sort_time" in approved_operations:
        lf = lf.sort(time_column)

    return lf


def handle_missing(
    lf: pl.LazyFrame,
    concerned_column: str,
    selected_operation: str,
    constant_value: object = None,
    handle_nan: bool = False,
) -> pl.LazyFrame:
    """Apply one missing-value strategy to one column."""
    schema = lf.collect_schema()

    if concerned_column not in schema:
        raise ValueError(f"Unknown column: {concerned_column}")

    
    dtype = schema[concerned_column]

    if handle_nan:
        if dtype not in (pl.Float32, pl.Float64):
            raise TypeError("NaN handling requires a float column.")

        if selected_operation != "keep":
            lf = lf.with_columns(
                pl.col(concerned_column)
                .fill_nan(None)
                .alias(concerned_column)
            )
    expression = pl.col(concerned_column)

    if selected_operation == "keep":
        return lf

    if selected_operation in {
        "mean", "median", "interpolation"
    } and not dtype.is_numeric():
        raise TypeError("Numeric imputation requires a numeric column.")

    if selected_operation == "mean":
        return lf.with_columns(
            expression.fill_null(expression.mean())
            .alias(concerned_column)
        )

    if selected_operation == "median":
        return lf.with_columns(
            expression.fill_null(expression.median())
            .alias(concerned_column)
        )

    if selected_operation == "interpolation":
        return lf.with_columns(
            expression.interpolate().alias(concerned_column)
        )

    if selected_operation == "drop":
        return lf.drop_nulls(subset=[concerned_column])

    if selected_operation == "forward_fill":
        return lf.with_columns(
            expression.forward_fill().alias(concerned_column)
        )

    if selected_operation == "backward_fill":
        return lf.with_columns(
            expression.backward_fill().alias(concerned_column)
        )

    if selected_operation == "Unknown":
        if dtype != pl.String:
            raise TypeError(
                "'Unknown' replacement requires a string column."
            )

        return lf.with_columns(
            expression.fill_null("Unknown").alias(concerned_column)
        )

    if selected_operation == "constant":
        if constant_value is None:
            raise ValueError("A non-null constant is required.")

        return lf.with_columns(
            expression.fill_null(
                pl.lit(constant_value).cast(dtype, strict=True)
            ).alias(concerned_column)
        )

    raise ValueError(
        f"Unsupported missing-value operation: {selected_operation}"
    )


def handle_outliers(
    lf: pl.LazyFrame,
    column: str,
    selected_operation: str,
    factor: float = 1.5,
) -> pl.LazyFrame:
    """Detect IQR outliers and apply a column-wide approved action."""
    schema = lf.collect_schema()

    if column not in schema:
        raise ValueError(f"Unknown column: {column}")

    if not schema[column].is_numeric():
        raise TypeError("Outlier handling requires a numeric column.")

    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(factor)
        or factor <= 0
    ):
        raise ValueError("IQR factor must be positive and finite.")

    allowed = {
        "keep", "drop", "replace_null",
        "interpolation", "marqueur",
    }

    if selected_operation not in allowed:
        raise ValueError("Unsupported outlier action.")

    if selected_operation == "keep":
        return lf

    finite_values = pl.col(column).cast(pl.Float64).filter(
        pl.col(column).cast(pl.Float64).is_finite()
    )

    bounds = lf.select(
        finite_values.quantile(0.25).alias("q1"),
        finite_values.quantile(0.75).alias("q3"),
    ).collect()

    q1 = bounds["q1"][0]
    q3 = bounds["q3"][0]

    if q1 is None or q3 is None:
        return lf

    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    finite_column = pl.col(column).cast(pl.Float64)
    is_outlier = (
        finite_column.is_finite()
        & ((finite_column < lower) | (finite_column > upper))
    ).fill_null(False)

    if selected_operation == "drop":
        return lf.filter(~is_outlier)

    if selected_operation == "replace_null":
        return lf.with_columns(
            pl.when(is_outlier)
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
        )

    if selected_operation == "interpolation":
        cleaned = lf.with_columns(
            pl.when(is_outlier)
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
        )

        return cleaned.with_columns(
            pl.col(column).interpolate().alias(column)
        )

    marker = f"{column}_is_outlier"

    if marker in schema:
        raise ValueError(f"Marker column already exists: {marker}")

    return lf.with_columns(is_outlier.alias(marker))


def normalize_categories(
    lf: pl.LazyFrame,
    columns: list[str],
    case: str = "lower",
    strip_spaces: bool = True,
) -> pl.LazyFrame:
    """Normalize approved string and categorical columns."""
    schema = lf.collect_schema()

    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Provide unique category columns.")

    if case not in {"lower", "upper", "preserve"}:
        raise ValueError("Unsupported category case transformation.")

    expressions = []

    for column in columns:
        if column not in schema:
            raise ValueError(f"Unknown column: {column}")

        if schema[column].base_type() not in {
            pl.String, pl.Categorical, pl.Enum
        }:
            raise TypeError(
                f"Column is not categorical or string: {column}"
            )

        expression = pl.col(column).cast(pl.String)

        if strip_spaces:
            expression = expression.str.strip_chars()

        if case == "lower":
            expression = expression.str.to_lowercase()
        elif case == "upper":
            expression = expression.str.to_uppercase()

        expressions.append(expression.alias(column))

    return lf.with_columns(expressions)


def resample(
    lf: pl.LazyFrame,
    time_column: str,
    frequency: str,
    value_columns: list[str],
    aggregation: str = "mean",
) -> pl.LazyFrame:
    """Aggregate selected numeric columns at an approved frequency."""
    schema = lf.collect_schema()

    if time_column not in schema:
        raise ValueError(f"Unknown time column: {time_column}")

    if not _is_temporal(schema[time_column]):
        raise TypeError("Time column must be Date or Datetime.")

    if (
        not value_columns
        or len(value_columns) != len(set(value_columns))
        or time_column in value_columns
    ):
        raise ValueError("Invalid resampling value columns.")

    for column in value_columns:
        if column not in schema or not schema[column].is_numeric():
            raise ValueError(
                f"Unknown or nonnumeric value column: {column}"
            )

    if aggregation not in {
        "mean", "median", "sum", "min",
        "max", "first", "last"
    }:
        raise ValueError("Unsupported resampling aggregation.")

    if not isinstance(frequency, str) or not frequency:
        raise ValueError("Invalid resampling frequency.")

    null_timestamps = lf.select(
        pl.col(time_column).is_null().sum()
    ).collect().item()

    if null_timestamps:
        raise ValueError(
            "Resampling would discard null timestamps."
        )

    expressions = [
        getattr(pl.col(column), aggregation)().alias(column)
        for column in value_columns
    ]

    return (
        lf.sort(time_column)
        .group_by_dynamic(
            index_column=time_column,
            every=frequency,
        )
        .agg(expressions)
        .sort(time_column)
    )


def regularize_timestamps(
    lf: pl.LazyFrame,
    time_column: str,
    frequency: str | None = None,
) -> pl.LazyFrame:
    """Add missing time-grid rows without inventing measurement values."""
    schema = lf.collect_schema()

    if time_column not in schema:
        raise ValueError(f"Unknown time column: {time_column}")

    if not _is_temporal(schema[time_column]):
        raise TypeError("Time column must be Date or Datetime.")

    df = lf.collect().sort(time_column)
    timestamps = df[time_column].to_list()

    if any(timestamp is None for timestamp in timestamps):
        raise ValueError("Null timestamps must be handled first.")

    if len(timestamps) < 2:
        raise ValueError("At least two timestamps are required.")

    deltas = [
        (right - left).total_seconds()
        for left, right in zip(timestamps, timestamps[1:])
    ]

    if any(delta <= 0 for delta in deltas):
        raise ValueError(
            "Timestamps must be unique and strictly increasing."
        )

    if frequency is None:
        seconds = Counter(
            round(delta, 6) for delta in deltas
        ).most_common(1)[0][0]

        if seconds <= 0:
            raise ValueError("Invalid inferred cadence.")

        frequency = f"{int(seconds * 1_000_000)}us"

    if not isinstance(frequency, str) or not frequency:
        raise ValueError("Invalid regularization frequency.")

    if schema[time_column].base_type() == pl.Date:
        regular_times = pl.date_range(
            start=timestamps[0],
            end=timestamps[-1],
            interval=frequency,
            eager=True,
        )
    else:
        regular_times = pl.datetime_range(
            start=timestamps[0],
            end=timestamps[-1],
            interval=frequency,
            eager=True,
        )

    grid = pl.DataFrame({time_column: regular_times})

    # Reject grids that would silently discard existing observations.
    if not set(timestamps).issubset(
        set(grid[time_column].to_list())
    ):
        raise ValueError(
            "Selected frequency does not preserve original timestamps."
        )

    return grid.join(
        df,
        on=time_column,
        how="left",
    ).lazy()


def _file_sha256(path: Path) -> str:
    """Fingerprint the original dataset."""
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _approved_frequency(parameters: dict) -> str:
    """Read an explicit frequency or convert audited integer seconds."""
    frequency = parameters.get("frequency")

    if frequency is None:
        frequency = parameters.get("every")

    if frequency is not None:
        if not isinstance(frequency, str) or not frequency:
            raise ValueError("Invalid frequency.")
        return frequency

    seconds = parameters.get("every_seconds")

    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("An explicit positive frequency is required.")

    microseconds = round(seconds * 1_000_000)
    if microseconds <= 0:
        raise ValueError("Frequency is below microsecond precision.")
    return f"{microseconds}us"


def run_preprocessing(
    file_path: str,
    approved_recommendations: list[dict],
    output_dir: str = "datasets/processed",
    expected_source_sha256: str | None = None,
    sheet_name: str | None = None,
    temporal_config: dict | None = None
) -> dict:
    """
    Execute only approved cleaning operations using shared LazyFrame tools.

    All transformations are cumulative. The source is never overwritten.
    The final Parquet is published only after successful execution.
    """
    temporal_config = temporal_config or {}
    
    if not approved_recommendations:
        return {
            "status": "skipped",
            "output_file_path": None,
            "applied_operations": [],
        }

    source = Path(file_path).resolve(strict=True)

    if expected_source_sha256 is None:
        raise ValueError(
            "A source fingerprint from the audit is required."
        )

    if _file_sha256(source) != expected_source_sha256:
        raise RuntimeError(
            "Source dataset changed after the audit."
        )

    # Import here to avoid a circular import between tools and services.
    from src.tools.cleaning_tools import (
        categories_tool,
        duplicates_tool,
        missing_values_tool,
        outliers_tool,
        regularization_tool,
        resampling_tool,
        time_tool,
    )

    if source.suffix.lower() in {".xlsx", ".xls"}:
        lf = load_lazy_dataset(
            str(source),
            sheet_name=sheet_name,
        )
    else:
        if sheet_name is not None:
            raise ValueError(
                "sheet_name is only supported for Excel."
            )
        lf = load_lazy_dataset(str(source))

    applied = []

    # Keep the approved order. Never invent or reorder operations.
    for recommendation in approved_recommendations:
        operation = recommendation["operation"]
        column = recommendation.get("column")
        parameters = recommendation.get("parameters") or {}

        if operation == "interpolate_iqr_outliers":
            raise ValueError(
                "Targeted IQR interpolation has been removed. "
                "Remove this recommendation before approving preprocessing."
            )

        if operation in {"handle_missing", "handle_nan", "interpolate"}:
            method = parameters.get("method")
            if method is None:
                raise ValueError(f"Missing approved method for {operation}.")

            lf = missing_values_tool(
                lf,
                column=column,
                method=method,
                value=parameters.get("value"),
                handle_nan=(operation == "handle_nan"),
            )

        elif operation == "handle_outliers":
            if parameters.get("method", "iqr") != "iqr":
                raise ValueError(
                    "Only IQR detection is supported."
                )

            lf = outliers_tool(
                lf,
                column=column,
                action=parameters["action"],
                factor=parameters.get("factor", 1.5),
            )

        elif operation == "normalize_categories":
            if "columns" in parameters:
                columns = parameters["columns"]

                if columns != [column]:
                    raise ValueError(
                        "Approved category columns do not match the target."
                    )
            else:
                columns = [column]

            lf = categories_tool(
                lf,
                columns=columns,
                case=parameters.get("case", "lower"),
                strip_whitespace=parameters.get(
                    "strip_whitespace", True
                ),
            )

        elif operation in {"parse_time", "sort_time"}:
            temporal_column = temporal_config.get("column")
            lf = time_tool(
                lf,
                time_column=column or temporal_column,
                operations=[operation],
                time_format=(
                    parameters.get("format")
                    or temporal_config.get("format")
                ),
            )

        elif operation == "remove_duplicates":
            lf = duplicates_tool(
                lf,
                keep=parameters.get("keep", "first"),
            )

        elif operation == "regularize_timestamps":
            if parameters.get("fill_new_rows", "null") != "null":
                raise ValueError(
                    "Regularization only inserts null measurements."
                )

            lf = regularization_tool(
                lf,
                time_column=column or temporal_config.get("column"),
                frequency=_approved_frequency(parameters),
            )

        elif operation == "resample":
            value_columns = parameters.get("value_columns")

            if (
                not isinstance(value_columns, list)
                or not value_columns
                or len(value_columns) != len(set(value_columns))
            ):
                raise ValueError(
                    "Resampling requires explicitly approved value_columns."
                )

            if parameters.get("allow_column_loss") is not True:
                raise ValueError(
                    "Resampling requires explicit column-loss approval."
                )

            # Prevent silently dropping non-aggregated columns.
            time_column = column or temporal_config.get("column")
            removed_columns = (
                set(lf.collect_schema().names())
                - {time_column, *value_columns}
            )

            if removed_columns:
                print(
                    "Approved resampling will remove columns: "
                    f"{sorted(removed_columns)}"
                )

            lf = resampling_tool(
                lf,
                time_column=column or temporal_config.get("column"),
                frequency=_approved_frequency(parameters),
                value_columns=value_columns,
                aggregation=parameters["method"],
            )

        else:
            raise ValueError(
                f"Unsupported approved operation: {operation}"
            )

        applied.append({
            "id": recommendation["id"],
            "operation": operation,
            "column": column,
        })

    # Execute the accumulated lazy transformations.
    df = lf.collect()

    if _file_sha256(source) != expected_source_sha256:
        raise RuntimeError(
            "Source dataset changed during preprocessing."
        )

    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    # A unique output name prevents overwriting earlier results.
    output = directory / (
        f"{source.stem}_processed_{uuid4().hex}.parquet"
    )
    temporary = directory / (
        f".{output.name}.{uuid4().hex}.tmp"
    )

    try:
        df.write_parquet(temporary)

        # Verify the temporary file before publishing it.
        check = pl.read_parquet(temporary)

        if check.shape != df.shape or check.schema != df.schema:
            raise RuntimeError("Parquet output verification failed.")

        # Exclusive creation: never replace an existing output.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "status": "success",
        "source_file": str(source),
        "source_sha256": expected_source_sha256,
        "output_file_path": str(output),
        "processed_rows": df.height,
        "processed_columns": df.width,
        "applied_operations": applied,
    }
