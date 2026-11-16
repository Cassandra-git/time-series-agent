from datetime import datetime

import polars as pl

from services.data_loader import scan_dataset
from services.statistics import summarize_numeric_column


def inspect_dataset(path: str) -> dict:
    """List the names and types of the columns in a Parquet file."""
    schema = scan_dataset(path).collect_schema()
    return {
        "columns": [
            {"name": name, "dtype": str(dtype), "numeric": dtype.is_numeric()}
            for name, dtype in schema.items()
        ]
    }


def describe_column(
    path: str,
    column: str,
    *,
    time_column: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Summarize a numeric column, optionally over [start, end).

    For a time window, supply the exact Datetime column name and both bounds
    as ISO 8601 timestamps. Include offsets if the column has a time zone.
    """
    frame = scan_dataset(path)
    if time_column is not None or start is not None or end is not None:
        if time_column is None or start is None or end is None:
            raise ValueError("Supply time_column, start, and end together.")

        dtype = frame.collect_schema().get(time_column)
        if dtype is None:
            raise ValueError(f"Unknown time column: {time_column}")
        if not isinstance(dtype, pl.Datetime):
            raise ValueError(f"Time column must have a Datetime type: {time_column}")

        try:
            lower = datetime.fromisoformat(start)
            upper = datetime.fromisoformat(end)
        except ValueError as error:
            raise ValueError("Use ISO 8601 timestamps for start and end.") from error

        zoned = dtype.time_zone is not None
        if (lower.tzinfo is not None) != zoned or (upper.tzinfo is not None) != zoned:
            raise ValueError("Use offsets on both bounds only for a zoned time column.")
        if lower >= upper:
            raise ValueError("Start must be earlier than end.")

        lower_expr = pl.lit(lower)
        upper_expr = pl.lit(upper)
        if zoned:
            lower_expr = lower_expr.dt.convert_time_zone(dtype.time_zone)
            upper_expr = upper_expr.dt.convert_time_zone(dtype.time_zone)

        frame = frame.filter(pl.col(time_column).is_between(lower_expr, upper_expr, closed="left"))

    summary = summarize_numeric_column(frame, column)
    return {"column": column, **summary}