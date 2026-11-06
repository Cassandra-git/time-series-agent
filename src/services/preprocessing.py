import polars as pl

def remove_duplicates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Remove duplicate rows."""
    return lf.unique()