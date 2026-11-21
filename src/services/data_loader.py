from pathlib import Path
import math

import polars as pl




SUPPORTED_FORMATS = {".csv", ".parquet", ".xlsx", ".xls"}





def scan_dataset(path: str | Path) -> pl.LazyFrame:
    """Read a Parquet dataset without assuming any column names."""
    path = Path(path)

    if path.suffix.lower() != ".parquet":
        raise ValueError("Expected a .parquet file.")
    if not path.is_file():
        raise FileNotFoundError(path)

    return pl.scan_parquet(path)

def convert_to_parquet(file_path: str | Path, output_dir: str | Path) -> Path:
    """Return a Parquet copy of a CSV/Excel file (a Parquet file is returned as is).

    The copy is named after the content hash, so loading the same file twice
    reuses it. The source file is never modified.
    """
    import hashlib

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    if path.suffix.lower() == ".parquet":
        return path
    with open(path, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()[:12]
    output = Path(output_dir) / f"{path.stem}_raw_{digest}.parquet"
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.parquet")
        try:
            load_lazy_dataset(str(path)).sink_parquet(temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def detect_csv_separator(file_path: str) -> str:
    """Detect the CSV separator from the header."""
    with open(file_path, "r", encoding="utf-8-sig") as file:
        header = file.readline()

    comma_count = header.count(",")
    semicolon_count = header.count(";")

    if semicolon_count > comma_count:
        return ";"

    return ","



def load_lazy_dataset(
    file_path: str,
    sheet_name: str | None = None,
) -> pl.LazyFrame:
    """
    Load a CSV, Parquet, or Excel dataset as a LazyFrame.

    CSV and Parquet are scanned lazily.
    Excel is loaded into memory before conversion to LazyFrame.
    """
    path = Path(file_path)

    # Check that the file exists.
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    extension = path.suffix.lower()

    # Reject unsupported formats.
    if extension not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )


    if sheet_name is not None and extension not in {".xlsx", ".xls"}:
        raise ValueError("sheet_name is only supported for Excel files.")

    if extension == ".csv":
        separator = detect_csv_separator(file_path)
        return pl.scan_csv(
            file_path,
            separator=separator,
            try_parse_dates=True,
            null_values=["NA"],
        )

    if extension == ".parquet":
        return pl.scan_parquet(path)

    # Excel cannot be scanned lazily.
    options = {"engine": "calamine"}
    if sheet_name is not None:
        options["sheet_name"] = sheet_name
    df = pl.read_excel(path, **options)
    if not isinstance(df, pl.DataFrame):
        raise ValueError("Please select a single Excel worksheet.")
    return df.lazy()


def inspect_column_quality(
    lf: pl.LazyFrame,
    schema: pl.Schema,
) -> dict[str, dict]:
    """Compute missing-run and distribution statistics per column."""

    result = {}

    for name, dtype in schema.items():
        is_missing = pl.col(name).is_null()

        if dtype in (pl.Float32, pl.Float64):
            is_missing = is_missing | pl.col(name).is_nan()

        run_start = (
            is_missing
            & ~is_missing.shift(1).fill_null(False)
        )

        run_id = run_start.cast(pl.UInt32).cum_sum()

        max_run = (
            lf.select(
                is_missing.alias("_missing"),
                run_id.alias("_run_id"),
            )
            .filter(pl.col("_missing"))
            .group_by("_run_id")
            .agg(pl.len().alias("_length"))
            .select(pl.col("_length").max())
            .collect()
            .item()
        )

        expressions = [
            is_missing.sum().alias("missing_count"),
            run_start.sum().alias("missing_run_count"),
        ]

        if dtype.is_numeric():
            values = pl.col(name).cast(pl.Float64)

            if dtype in (pl.Float32, pl.Float64):
                values = values.filter(values.is_finite())

            expressions.extend([
                values.min().alias("min"),
                values.max().alias("max"),
                values.mean().alias("mean"),
                values.std().alias("std"),
                values.quantile(0.25).alias("q1"),
                values.median().alias("median"),
                values.quantile(0.75).alias("q3"),
            ])

        stats = lf.select(expressions).collect().to_dicts()[0]

        result[name] = {
            **stats,
            "max_missing_run": int(max_run or 0),
        }

    return result



def inspect_dataset(
    file_path: str,
    sheet_name: str | None = None,
    detailed: bool = False,
    sampling_rate_hz: float | None = None,
    lf: pl.LazyFrame | None = None,
) -> dict:
    """
    Inspect a time-series dataset without modifying it.

    Supports explicit timestamps and implicit time axes.

    Args:
        file_path:
            Path to a CSV, Parquet, or Excel dataset.

        sheet_name:
            Optional worksheet name for Excel files.

        detailed:
            If True, calculate the number of exact duplicate rows.
            This may be expensive for large datasets.

        sampling_rate_hz:
            Optional sampling frequency in Hz.
            Used when timestamps are absent, for example
            with vibration signals sampled at 50 kHz.

    Returns:
        A JSON-compatible dictionary containing:
        - Dataset dimensions
        - Column types
        - Missing value statistics
        - Temporal information
        - Optional duplicate count
        - Small data preview
    """

    
    # STEP 1: Validate the sampling frequency.
    # Sampling frequency must be strictly positive and finite. Reject NaN and infinity as well.
    if sampling_rate_hz is not None:
        if (
            isinstance(sampling_rate_hz, bool)
            or not isinstance(sampling_rate_hz, (int, float))
            or not math.isfinite(sampling_rate_hz)
            or sampling_rate_hz <= 0
        ):
            raise ValueError(
                "sampling_rate_hz must be a positive, finite number."
            )


    # STEP 2: Load the dataset.
    if lf is None:
        lf = load_lazy_dataset(file_path, sheet_name=sheet_name)


    # Retrieve the schema without collecting the full data.
    schema = lf.collect_schema()


    # STEP 3: Build statistical expressions.
    expressions = [
        pl.len().alias("__inspection_row_count")
    ]

    # Count null values for every column.
    # Count NaN values for floating-point columns.
    for index, (name, dtype) in enumerate(schema.items()):

        expressions.append(
            pl.col(name)
            .null_count()
            .alias(f"__inspection_null_{index}")
        )

        if dtype in (pl.Float32, pl.Float64):
            expressions.append(
                pl.col(name)
                .is_nan()
                .sum()
                .alias(f"__inspection_nan_{index}")
            )

 
    # STEP 4: Execute the statistical expressions.
    # This triggers actual computation.
    # The resulting DataFrame contains one row.
    stats = lf.select(expressions).collect()

    row_count = int(
        stats["__inspection_row_count"][0]
    )


    # STEP 5: Build the column quality report.
    columns = []

    for index, (name, dtype) in enumerate(schema.items()):

        null_count = int(
            stats[f"__inspection_null_{index}"][0]
        )

        nan_count = 0

        if dtype in (pl.Float32, pl.Float64):
            nan_count = int(
                stats[f"__inspection_nan_{index}"][0]
            )

        columns.append({
            "name": name,
            "dtype": str(dtype),
            "null_count": null_count,
            "nan_count": nan_count,
            "missing_count": null_count + nan_count,
            "null_percentage": (
                round(100 * null_count / row_count, 2)
                if row_count else 0.0
            ),
        })

    # STEP 5B: Compute per-column quality statistics.
    quality_stats = inspect_column_quality(lf, schema)

    for column in columns:
        column.update(quality_stats[column["name"]])

    # STEP 6: Detect explicit temporal columns.
    # These columns are already recognized as temporal.
    temporal_columns = [
        name
        for name, dtype in schema.items()
        if dtype.base_type() in (
            pl.Date,
            pl.Datetime,
            pl.Time,
        )
    ]

  
    # STEP 7: Analyze the temporal representation.
    temporal_info = {
        "mode": "unknown",
        "timestamp_columns": temporal_columns,
        "sampling_rate_hz": None,
        "sampling_interval_seconds": None,
        "duration_seconds": None,
        "requires_timestamp_generation": False,
    }

    if temporal_columns:
        # The dataset contains an explicit time column.
        temporal_info["mode"] = "explicit"

    elif sampling_rate_hz is not None:
        # No timestamp column was detected, but the
        # sampling frequency is known.
        #
        # Assume each row corresponds to one sample,
        # and samples are uniformly spaced.
        sampling_interval = 1.0 / sampling_rate_hz

        # Elapsed time between the first and last samples.
        # For N samples, there are N-1 intervals.
        duration = (
            (row_count - 1) / sampling_rate_hz
            if row_count > 0 else 0.0
        )

        temporal_info.update({
            "mode": "implicit",
            "sampling_rate_hz": sampling_rate_hz,
            "sampling_interval_seconds": sampling_interval,
            "duration_seconds": duration,
            "requires_timestamp_generation": True,
        })

    # If there are no temporal columns and no sampling frequency, temporal mode remains "unknown".
    # The agent should ask the user for metadata.

    # STEP 8: Count exact duplicate rows (optional).
    duplicate_rows = None

    if detailed and schema:
        duplicate_result = (
            lf.select(
                (
                    pl.len()
                    - pl.struct(pl.all()).n_unique()
                ).alias("__duplicates")
            )
            .collect()
        )

        duplicate_rows = int(
            duplicate_result["__duplicates"][0]
        )

    # STEP 9: Retrieve a small preview.
    preview = lf.head(5).collect().to_dicts()

    # Convert values to strings so the preview can be serialized, including date and datetime objects.
    serializable_preview = [
        {
            key: str(value) if value is not None else None
            for key, value in row.items()
        }
        for row in preview
    ]

    # --------------------------------------------------
    # STEP 10: Return the inspection report.
    # --------------------------------------------------

    return {
        "file_name": Path(file_path).name,
        "file_format": Path(file_path).suffix.lower(),
        "sheet_name": sheet_name,
        "row_count": row_count,
        "column_count": len(schema),
        "columns": columns,
        "temporal_columns": temporal_columns,
        "temporal_info": temporal_info,
        "duplicate_rows": duplicate_rows,
        "preview": serializable_preview,
    }
