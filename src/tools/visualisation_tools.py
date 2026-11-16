"""LangChain tools exposing the plotting services to the agent.

Rules followed by every tool:
* the figure is saved to disk (Plotly JSON) and NEVER returned to the LLM;
* the tool returns a short text: "Figure created: <path> (...)";
* errors are returned as "ERROR: ..." so the agent can fix its call and retry;
* list arguments tolerate a plain string, a comma-separated string or a
  JSON string, because small models often send those instead of a list.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

from langchain_core.tools import tool

from src.services import plotting
from src.state import resolve_path

FIG_DIR = Path(os.getenv("TS_AGENT_FIG_DIR", "outputs/figures"))


# ============================================================
# HELPERS
# ============================================================

def _as_list(value: list[str] | str | None) -> list[str] | None:
    """Coerce an LLM-provided value to a list of column names."""
    if value is None:
        return None

    if isinstance(value, list):
        return [str(item).strip() for item in value]

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed]
            except json.JSONDecodeError:
                pass

        return [part.strip() for part in text.split(",") if part.strip()]

    raise ValueError("Expected a column name or a list of column names.")


def _none_if_empty(value: str | None) -> str | None:
    """Models sometimes send "" or "null" for optional arguments."""
    if value is None:
        return None

    value = str(value).strip()

    return None if value == "" or value.lower() in {"null", "none"} else value


def _save_figure(fig, kind: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{kind}_{uuid4().hex[:8]}.json"
    fig.write_json(path)
    return str(path)


def _run(kind: str, details: str, build: Callable) -> str:
    """Build a figure, save it, and return a compact message."""
    try:
        fig = build()
    except (ValueError, TypeError, FileNotFoundError, KeyError) as exc:
        return f"ERROR: {exc}"

    path = _save_figure(fig, kind)

    return f"Figure created: {path} ({kind}; {details})"


# ============================================================
# TOOLS
# ============================================================

@tool
def plot_time_series_tool(
    dataset_id: str,
    value_cols: list[str] | str,
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    normalize: bool = False,
) -> str:
    """Plot one or more numeric columns over time (line chart).

    Use this when the user wants to see how a variable evolves, or to
    compare several signals. Long signals are automatically decimated.

    Args:
        dataset_id: id of the dataset to plot, e.g. "ds_1".
        value_cols: column name or list of column names to plot.
        time_col: time column (datetime or numeric). Omit to use row index.
        start_date: optional lower bound, e.g. "2024-01-01".
        end_date: optional upper bound (date-only values include that day).
        normalize: set True to z-score each curve when scales differ a lot.
    """
    try:
        cols = _as_list(value_cols)
    except ValueError as exc:
        return f"ERROR: {exc}"

    return _run(
        "time_series",
        f"columns={cols}, time_col={time_col}",
        lambda: plotting.plot_time_series(
            resolve_path(dataset_id),
            _none_if_empty(time_col),
            cols or [],
            _none_if_empty(start_date),
            _none_if_empty(end_date),
            normalize=normalize,
        ),
    )


@tool
def plot_histogram_tool(
    dataset_id: str,
    column: str,
    bins: int = 30,
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Plot the distribution (histogram) of ONE numeric column.

    Use this when the user asks about the distribution or spread of a
    variable.

    Args:
        dataset_id: id of the dataset to plot, e.g. "ds_1".
        column: numeric column name.
        bins: number of bins (positive integer).
        time_col: only needed together with start_date / end_date.
        start_date: optional lower bound.
        end_date: optional upper bound.
    """
    try:
        n_bins = int(bins)
    except (TypeError, ValueError):
        return "ERROR: bins must be a positive integer."

    return _run(
        "histogram",
        f"column={column}, bins={n_bins}",
        lambda: plotting.plot_histogram(
            resolve_path(dataset_id),
            column,
            n_bins,
            _none_if_empty(time_col),
            _none_if_empty(start_date),
            _none_if_empty(end_date),
        ),
    )


@tool
def plot_boxplot_tool(
    dataset_id: str,
    columns: list[str] | str,
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Plot boxplots of one or more numeric columns (median, quartiles,
    outliers). Use this to compare spreads or spot outliers.

    Args:
        dataset_id: id of the dataset to plot, e.g. "ds_1".
        columns: column name or list of column names.
        time_col: only needed together with start_date / end_date.
        start_date: optional lower bound.
        end_date: optional upper bound.
    """
    try:
        cols = _as_list(columns)
    except ValueError as exc:
        return f"ERROR: {exc}"

    return _run(
        "boxplot",
        f"columns={cols}",
        lambda: plotting.plot_boxplot(
            resolve_path(dataset_id),
            cols or [],
            _none_if_empty(time_col),
            _none_if_empty(start_date),
            _none_if_empty(end_date),
        ),
    )


@tool
def plot_scatter_tool(
    dataset_id: str,
    x_col: str,
    y_col: str,
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Plot the relationship between TWO numeric columns (scatter plot).

    Use this when the user asks whether two variables are related.

    Args:
        dataset_id: id of the dataset to plot, e.g. "ds_1".
        x_col: numeric column for the x axis.
        y_col: numeric column for the y axis.
        time_col: only needed together with start_date / end_date.
        start_date: optional lower bound.
        end_date: optional upper bound.
    """
    return _run(
        "scatter",
        f"x={x_col}, y={y_col}",
        lambda: plotting.plot_scatter(
            resolve_path(dataset_id),
            x_col,
            y_col,
            _none_if_empty(time_col),
            _none_if_empty(start_date),
            _none_if_empty(end_date),
        ),
    )


@tool
def plot_correlation_heatmap_tool(
    dataset_id: str,
    columns: list[str] | str | None = None,
    time_col: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Plot the correlation matrix of numeric columns as a heatmap.

    Use this to see which variables move together. If columns is omitted,
    all numeric columns are used.

    Args:
        dataset_id: id of the dataset to plot, e.g. "ds_1".
        columns: optional column name(s); at least two are needed.
        time_col: excluded from the matrix; needed for date filtering.
        start_date: optional lower bound.
        end_date: optional upper bound.
    """
    try:
        cols = _as_list(columns)
    except ValueError as exc:
        return f"ERROR: {exc}"

    return _run(
        "correlation_heatmap",
        f"columns={cols or 'all numeric'}",
        lambda: plotting.plot_correlation_heatmap(
            resolve_path(dataset_id),
            cols,
            _none_if_empty(time_col),
            _none_if_empty(start_date),
            _none_if_empty(end_date),
        ),
    )


VISUALISATION_TOOLS = [
    plot_time_series_tool,
    plot_histogram_tool,
    plot_boxplot_tool,
    plot_scatter_tool,
    plot_correlation_heatmap_tool,
]