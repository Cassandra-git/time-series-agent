import polars as pl


def summarize_numeric_column(frame: pl.LazyFrame, column: str) -> dict:
    """Compute descriptive statistics for an explicitly selected column."""
    schema = frame.collect_schema()
    if column not in schema:
        raise ValueError(f"Unknown column: {column}")
    if not schema[column].is_numeric():
        raise ValueError(f"Column is not numeric: {column}")

    raw = pl.col(column)
    value = pl.when(raw.is_finite()).then(raw).otherwise(None)
    return frame.select(
        pl.len().alias("row_count"),
        raw.null_count().alias("null_count"),
        (raw.is_not_null() & ~raw.is_finite()).sum().alias("non_finite_count"),
        value.count().alias("valid_count"),
        value.min().alias("min"),
        value.max().alias("max"),
        value.mean().alias("mean"),
        value.median().alias("median"),
        value.std(ddof=1).alias("std"), # écart-type échantillon
        value.quantile(0.25, interpolation="linear").alias("q1"),
        value.quantile(0.75, interpolation="linear").alias("q3"), 
    ).collect().row(0, named=True)
