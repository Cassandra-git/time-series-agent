"""Chart tools for the agent.

Two flavours of the same five charts, both thin wrappers over services/plotting.py:

* `VISUALISATION_TOOLS`: LangChain @tool objects that take `parquet_path`
  (for a LangChain / LangGraph agent).
* `bind_visualisation_tools(parquet_path)`: plain functions bound to ONE file,
  as expected by `answer_question(..., visualisation_tools=...)` in builder.py.
  The model never has to type a path. Optional arguments use "" for "not
  given", and lists / booleans are coerced because a small model sometimes
  sends "a,b" instead of ["a", "b"].

A tool never returns the figure: it saves it as JSON and returns the path, the
title and the number of curves. Errors come back as {"error": "..."} so the
model can read them and retry.
"""

import logging
import os

from langchain_core.tools import tool

from ts_agent.services import plotting

logger = logging.getLogger(__name__)


def _output_dir() -> str:
    return os.environ.get("TS_AGENT_OUTPUT_DIR", "outputs/figures")


def _run(build, prefix: str) -> dict:
    """Build a figure, save it as JSON, return a small dict for the LLM."""
    try:
        fig = build()
        path = plotting.save_figure(fig, _output_dir(), prefix)
    except (ValueError, FileNotFoundError, TypeError, KeyError) as error:
        return {"error": str(error)}
    except Exception as error:  # never crash the agent loop
        logger.exception("Visualisation tool failed")
        return {"error": f"Unexpected {type(error).__name__}: {error}"}
    return {"figure_path": str(path), "title": fig.layout.title.text, "n_traces": len(fig.data)}


# ============================================================
# LangChain tools (the file path is an argument)
# ============================================================

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
            parquet_path, time_col, value_cols, start_date, end_date, normalize),
        "timeseries")


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
            parquet_path, column, bins, time_col, start_date, end_date),
        "histogram")


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

    With group_by (a categorical column), pass a single numeric column:
    one box is drawn per group.
    """
    return _run(
        lambda: plotting.plot_boxplot(
            parquet_path, columns, time_col, start_date, end_date, group_by),
        "boxplot")


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
            parquet_path, x_col, y_col, time_col, start_date, end_date),
        "scatter")


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
            parquet_path, columns, time_col, start_date, end_date),
        "heatmap")


VISUALISATION_TOOLS = [
    plot_time_series_tool,
    plot_histogram_tool,
    plot_boxplot_tool,
    plot_scatter_tool,
    plot_correlation_heatmap_tool,
]


# ============================================================
# Plain functions bound to one file (for builder.answer_question)
# ============================================================

def _as_list(value) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value or [])


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "oui"}
    return bool(value)


def _or_none(value):
    return value or None


def bind_visualisation_tools(parquet_path: str) -> tuple:
    """Return the five chart functions, all bound to `parquet_path`."""

    def plot_time_series(
        value_cols: list[str],
        time_col: str = "",
        start_date: str = "",
        end_date: str = "",
        normalize: bool = False,
    ) -> dict:
        """Line chart of one or more numeric columns over time.

        Use for trends, gaps, and comparing sensors. Set normalize=True when the
        columns have very different scales (each curve is rescaled).

        Args:
            value_cols: Exact names of the numeric columns to plot.
            time_col: Exact name of the time column. Empty = use the row number.
            start_date: Optional start, 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
            end_date: Optional end. A date without time includes the whole day.
            normalize: True to compare curves with different scales.
        """
        return _run(
            lambda: plotting.plot_time_series(
                parquet_path, _or_none(time_col), _as_list(value_cols),
                _or_none(start_date), _or_none(end_date), _as_bool(normalize)),
            "timeseries")

    def plot_histogram(
        column: str, bins: int = 30, time_col: str = "", start_date: str = "", end_date: str = ""
    ) -> dict:
        """Histogram of one numeric column, to inspect its distribution.

        Args:
            column: Exact name of the numeric column.
            bins: Number of bins.
            time_col: Time column, only needed to filter by date.
            start_date: Optional start, 'YYYY-MM-DD'.
            end_date: Optional end, 'YYYY-MM-DD'.
        """
        return _run(
            lambda: plotting.plot_histogram(
                parquet_path, column, int(bins), _or_none(time_col),
                _or_none(start_date), _or_none(end_date)),
            "histogram")

    def plot_boxplot(
        columns: list[str], group_by: str = "", time_col: str = "",
        start_date: str = "", end_date: str = ""
    ) -> dict:
        """Box plot of numeric columns, to compare spread and spot outliers.

        Args:
            columns: Exact names of the numeric columns. Only one if group_by is used.
            group_by: Optional categorical column: one box per group.
            time_col: Time column, only needed to filter by date.
            start_date: Optional start, 'YYYY-MM-DD'.
            end_date: Optional end, 'YYYY-MM-DD'.
        """
        return _run(
            lambda: plotting.plot_boxplot(
                parquet_path, _as_list(columns), _or_none(time_col),
                _or_none(start_date), _or_none(end_date), _or_none(group_by)),
            "boxplot")

    def plot_scatter(
        x_col: str, y_col: str, time_col: str = "", start_date: str = "", end_date: str = ""
    ) -> dict:
        """Scatter plot of two numeric columns, to inspect a relationship.

        Args:
            x_col: Exact name of the column on the x axis.
            y_col: Exact name of the column on the y axis.
            time_col: Time column, only needed to filter by date.
            start_date: Optional start, 'YYYY-MM-DD'.
            end_date: Optional end, 'YYYY-MM-DD'.
        """
        return _run(
            lambda: plotting.plot_scatter(
                parquet_path, x_col, y_col, _or_none(time_col),
                _or_none(start_date), _or_none(end_date)),
            "scatter")

    def plot_correlation_heatmap(
        columns: list[str] = None, time_col: str = "", start_date: str = "", end_date: str = ""
    ) -> dict:
        """Correlation matrix of numeric columns.

        Args:
            columns: Exact column names. Leave empty to use all numeric columns.
            time_col: Time column, only needed to filter by date (excluded from the matrix).
            start_date: Optional start, 'YYYY-MM-DD'.
            end_date: Optional end, 'YYYY-MM-DD'.
        """
        cols = _as_list(columns) or None
        return _run(
            lambda: plotting.plot_correlation_heatmap(
                parquet_path, cols, _or_none(time_col),
                _or_none(start_date), _or_none(end_date)),
            "heatmap")

    return (plot_time_series, plot_histogram, plot_boxplot, plot_scatter, plot_correlation_heatmap)