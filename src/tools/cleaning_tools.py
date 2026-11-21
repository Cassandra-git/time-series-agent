"""Thin wrappers for inspection and Polars LazyFrame preprocessing."""

from __future__ import annotations

import polars as pl
from langchain_core.tools import tool

from services.data_loader import inspect_dataset
from services.preprocessing import (
    handle_missing,
    handle_outliers,
    normalize_categories,
    remove_duplicates,
    resample,
    regularize_timestamps,
    time_formating,
)


@tool
def inspect_dataset_tool(file_path: str, detailed: bool = False) -> dict:
    """Inspect a dataset without modifying it or calling an LLM."""
    return inspect_dataset(file_path=file_path, detailed=detailed)


def missing_values_tool(
    lf: pl.LazyFrame,
    column: str,
    method: str,
    value: object = None,
    handle_nan: bool = False,
) -> pl.LazyFrame:
    """Apply a missing-value strategy to one column."""
    aliases = {
        "linear": "interpolation",
        "interpolation": "interpolation",
        "forward": "forward_fill",
        "forward_fill": "forward_fill",
        "backward": "backward_fill",
        "backward_fill": "backward_fill",
        "mean": "mean",
        "median": "median",
        "drop": "drop",
        "keep": "keep",
        "constant": "constant",
        "Unknown": "Unknown",
    }
    if method not in aliases:
        raise ValueError(f"Unsupported missing-value method: {method}")
    return handle_missing(
        lf,
        concerned_column=column,
        selected_operation=aliases[method],
        constant_value=value,
        handle_nan=handle_nan,
    )


def outliers_tool(
    lf: pl.LazyFrame,
    column: str,
    action: str,
    factor: float = 1.5,
) -> pl.LazyFrame:
    """Apply an approved IQR outlier action."""
    return handle_outliers(lf, column=column, selected_operation=action, factor=factor)


def categories_tool(
    lf: pl.LazyFrame,
    columns: list[str],
    case: str = "lower",
    strip_whitespace: bool = True,
) -> pl.LazyFrame:
    """Normalize approved categorical columns."""
    return normalize_categories(lf, columns=columns, case=case, strip_spaces=strip_whitespace)


def time_tool(
    lf: pl.LazyFrame,
    time_column: str,
    operations: list[str],
    time_format: str | None = None,
) -> pl.LazyFrame:
    """Parse or sort a timestamp column."""
    return time_formating(
        lf,
        time_column=time_column,
        time_format=time_format,
        approved_operations=operations,
    )


def duplicates_tool(lf: pl.LazyFrame, keep: str = "first") -> pl.LazyFrame:
    """Remove exact duplicate rows."""
    return remove_duplicates(lf, keep=keep)


def resampling_tool(
    lf: pl.LazyFrame,
    time_column: str,
    frequency: str,
    value_columns: list[str],
    aggregation: str,
) -> pl.LazyFrame:
    """Resample approved numeric columns."""
    return resample(
        lf,
        time_column=time_column,
        frequency=frequency,
        value_columns=value_columns,
        aggregation=aggregation,
    )


def regularization_tool(
    lf: pl.LazyFrame,
    time_column: str,
    frequency: str,
) -> pl.LazyFrame:
    """Insert missing timestamps with null measurements."""
    return regularize_timestamps(lf, time_column=time_column, frequency=frequency)
