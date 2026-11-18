"""LangChain tools that draw charts.

Each tool saves the figure as JSON and returns only its path and a short
summary (never the figure). Errors are returned as {"error": "..."} so that
the LLM can read them and retry with corrected arguments.
"""

import logging
import os

from langchain_core.tools import tool

from services import plotting

logger = logging.getLogger(__name__)


def _output_dir():
    return os.environ.get("TS_AGENT_OUTPUT_DIR", "outputs/figures")


def _run(build, prefix):
    """Build a figure, save it, and return a small dict for the LLM."""
    try:
        fig = build()
        path = plotting.save_figure(fig, _output_dir(), prefix)
    except (ValueError, FileNotFoundError) as error:
        return {"error": str(error)}
    except Exception as error:  # keep the agent alive on unexpected failures
        logger.exception("Visualisation tool failed")
        return {"error": f"Unexpected {type(error).__name__}: {error}"}

    return {
        "figure_path": str(path),
        "title": fig.layout.title.text,
        "n_traces": len(fig.data),
    }


@tool
def plot_time_series_tool(
    parquet_path: str,
    value_cols: list[str],
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    normalize: bool = False,
) -> dict:
    """Line chart of one or more numeric columns over time.

    Use for trends, gaps and comparing sensors. Set normalize=True when the
    columns have very different scales (z-score per curve). Dates are
    'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'; a date-only end_date includes
    the whole day. Column names must match the dataset exactly.
    """
    return _run(
        lambda: plotting.plot_time_series(
            parquet_path, time_col, value_cols, start_date, end_date, normalize
        ),
        "timeseries",
    )


@tool
def plot_histogram_tool(
    parquet_path: str,
    column: str,
    bins: int = 30,
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Histogram of one numeric column, to inspect its distribution."""
    return _run(
        lambda: plotting.plot_histogram(
            parquet_path, column, bins, time_col, start_date, end_date
        ),
        "histogram",
    )


@tool
def plot_boxplot_tool(
    parquet_path: str,
    columns: list[str],
    group_by: str | None = None,
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Box plot of numeric columns, to compare spread and spot outliers.

    With group_by (a categorical column such as 'Location'), pass a single
    numeric column: one box is drawn per group.
    """
    return _run(
        lambda: plotting.plot_boxplot(
            parquet_path, columns, time_col, start_date, end_date, group_by
        ),
        "boxplot",
    )


@tool
def plot_scatter_tool(
    parquet_path: str,
    x_col: str,
    y_col: str,
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Scatter plot of two numeric columns, to inspect a relationship."""
    return _run(
        lambda: plotting.plot_scatter(
            parquet_path, x_col, y_col, time_col, start_date, end_date
        ),
        "scatter",
    )


@tool
def plot_correlation_heatmap_tool(
    parquet_path: str,
    columns: list[str] | None = None,
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Correlation matrix of numeric columns (all numeric columns if none given)."""
    return _run(
        lambda: plotting.plot_correlation_heatmap(
            parquet_path, columns, time_col, start_date, end_date
        ),
        "heatmap",
    )


VISUALISATION_TOOLS = [
    plot_time_series_tool,
    plot_histogram_tool,
    plot_boxplot_tool,
    plot_scatter_tool,
    plot_correlation_heatmap_tool,
]