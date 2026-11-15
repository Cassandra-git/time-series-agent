"""Smoke tests for cumulative approved preprocessing."""

import hashlib
from pathlib import Path

import polars as pl
import pytest

from src.services.preprocessing import run_preprocessing


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cumulative_preprocessing(tmp_path):
    source = tmp_path / "sample.parquet"
    output_dir = tmp_path / "processed"
    original = pl.DataFrame({
        "timestamp": [
            "2026-01-01 00:00:00",
            "2026-01-01 01:00:00",
            "2026-01-01 02:00:00",
        ],
        "temperature": [10.0, None, 14.0],
        "station": [" Paris ", "PARIS", "Paris"],
    })
    original.write_parquet(source)
    original_bytes = source.read_bytes()

    approved = [
        {
            "id": "rec_1",
            "operation": "parse_time",
            "column": "timestamp",
            "parameters": {"format": "%Y-%m-%d %H:%M:%S"},
        },
        {
            "id": "rec_2",
            "operation": "sort_time",
            "column": "timestamp",
            "parameters": {},
        },
        {
            "id": "rec_3",
            "operation": "handle_missing",
            "column": "temperature",
            "parameters": {"method": "linear"},
        },
        {
            "id": "rec_4",
            "operation": "normalize_categories",
            "column": "station",
            "parameters": {"case": "lower", "strip_whitespace": True},
        },
    ]

    result = run_preprocessing(
        file_path=str(source),
        approved_recommendations=approved,
        expected_source_sha256=_sha256(source),
        temporal_config={
            "column": "timestamp",
            "format": "%Y-%m-%d %H:%M:%S",
        },
        output_dir=str(output_dir),
    )
    processed = pl.read_parquet(result["output_file_path"])

    assert result["status"] == "success"
    assert processed.height == 3
    assert processed["temperature"].to_list() == [10.0, 12.0, 14.0]
    assert processed["station"].to_list() == ["paris", "paris", "paris"]
    assert processed.schema["timestamp"].base_type() == pl.Datetime
    assert source.read_bytes() == original_bytes
    assert Path(result["output_file_path"]).resolve() != source.resolve()


def test_source_change_is_rejected(tmp_path):
    source = tmp_path / "sample.parquet"
    pl.DataFrame({"value": [1, 2, 3]}).write_parquet(source)
    old_hash = _sha256(source)
    pl.DataFrame({"value": [4, 5, 6]}).write_parquet(source)

    with pytest.raises(RuntimeError, match="changed"):
        run_preprocessing(
            file_path=str(source),
            approved_recommendations=[
                {
                    "id": "rec_1",
                    "operation": "remove_duplicates",
                    "column": None,
                    "parameters": {},
                }
            ],
            expected_source_sha256=old_hash,
            output_dir=str(tmp_path / "processed"),
        )
    assert not (tmp_path / "processed").exists()


def test_unapproved_operation_is_rejected(tmp_path):
    source = tmp_path / "sample.parquet"
    pl.DataFrame({"value": [1, 2, 3]}).write_parquet(source)

    with pytest.raises(ValueError, match="Unsupported approved operation"):
        run_preprocessing(
            file_path=str(source),
            approved_recommendations=[
                {
                    "id": "rec_1",
                    "operation": "unknown_operation",
                    "column": "value",
                    "parameters": {},
                }
            ],
            expected_source_sha256=_sha256(source),
            output_dir=str(tmp_path / "processed"),
        )
    assert not (tmp_path / "processed").exists()
