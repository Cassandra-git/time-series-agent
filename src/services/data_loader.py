from pathlib import Path

import polars as pl


def scan_dataset(path: str | Path) -> pl.LazyFrame:
    """Read a Parquet dataset without assuming any column names."""
    path = Path(path)

    if path.suffix.lower() != ".parquet":
        raise ValueError("Expected a .parquet file.")
    if not path.is_file():
        raise FileNotFoundError(path)

    return pl.scan_parquet(path)