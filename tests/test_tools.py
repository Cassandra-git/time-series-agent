import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import polars as pl

from tools.analysis_tools import describe_column, inspect_dataset


class AnalysisToolsTests(unittest.TestCase):
    def test_discovers_columns_and_summarizes_selected_measure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.parquet"
            pl.DataFrame({
                "station": ["A"] * 4,
                "Pression (bar)": [1.0, 3.0, None, float("inf")],
            }).write_parquet(path)
            self.assertEqual(inspect_dataset(str(path))["columns"], [
                {"name": "station", "dtype": "String", "numeric": False},
                {"name": "Pression (bar)", "dtype": "Float64", "numeric": True},
            ])
            summary = describe_column(str(path), "Pression (bar)")
            self.assertAlmostEqual(summary.pop("std"), 2 ** 0.5)
            self.assertEqual(summary, {
                "column": "Pression (bar)",
                "row_count": 4,
                "null_count": 1,
                "non_finite_count": 1,
                "valid_count": 2,
                "min": 1.0,
                "max": 3.0,
                "mean": 2.0,
                "median": 2.0,
                "q1": 1.5,
                "q3": 2.5,
            })
            for column in ("station", "unknown"):
                with self.subTest(column=column), self.assertRaises(ValueError):
                    describe_column(str(path), column)

    def test_summarizes_only_rows_in_the_requested_time_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.parquet"
            pl.DataFrame({
                "Recorded at": [datetime(2026, 11, 14, hour, minute) for hour, minute in (
                    (7, 59), (8, 0), (8, 30), (9, 0),
                )],
                "Pression (bar)": [100.0, 10.0, 14.0, 100.0],
            }).write_parquet(path)

            summary = describe_column(
                str(path), "Pression (bar)",
                time_column="Recorded at",
                start="2026-11-14T08:00:00",
                end="2026-11-14T09:00:00",
            )
            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(summary["mean"], 12.0)
            self.assertEqual(describe_column(str(path), "Pression (bar)")["mean"], 56.0)

            with self.assertRaisesRegex(ValueError, "Supply time_column"):
                describe_column(str(path), "Pression (bar)", time_column="Recorded at")
