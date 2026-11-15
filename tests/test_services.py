"""Tests for dataset inspection, without Ollama."""

from pathlib import Path

import pytest

from src.services.data_loader import inspect_dataset
from src.tools.cleaning_tools import inspect_dataset_tool


def _sample_dataset() -> Path:
    """Find the sample dataset without depending on the working directory."""
    project = Path(__file__).resolve().parents[1]
    candidates = [
        project / "data" / "sample" / " ASE4.csv",
        project / "datasets" / "sample" / "ASE4.csv",
        project / "time-series-agent" / "datasets" / "sample" / "ASE4.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("sensor_data.csv is not available in a sample directory")


def test_inspect_dataset_service():
    """Inspect a real sample using the Polars service."""
    result = inspect_dataset(str(_sample_dataset()))
    assert isinstance(result, dict)
    assert "row_count" in result
    assert "columns" in result
    assert result["row_count"] > 0
    print("\\nNumber of rows:", result["row_count"])
    print("Columns:", result["columns"])


def test_inspect_dataset_tool():
    """Invoke the existing LangChain inspection tool without an LLM."""
    result = inspect_dataset_tool.invoke({
        "file_path": str(_sample_dataset()),
        "detailed": True,
    })
    assert isinstance(result, dict)
    assert "row_count" in result
    assert "duplicate_rows" in result or "exact_duplicate_count" in result
    print("\\nFull inspection report:", result)
