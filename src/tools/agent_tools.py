"""Tools given to the conversational agent.

Every tool reads the active dataset from the agent state (InjectedState), so
the model never types a file path except in load_dataset. The tools are thin
wrappers: the work is done by services/ and by the existing tool modules.

Tools that change the active dataset (load_dataset, clean_dataset) return a
LangGraph Command that updates the state and adds the ToolMessage.
"""

import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from services.data_loader import SUPPORTED_FORMATS, convert_to_parquet
from services.data_loader import inspect_dataset as audit_dataset
from tools import analysis_tools
from tools.visualisation_tools import (
    plot_boxplot_tool,
    plot_correlation_heatmap_tool,
    plot_histogram_tool,
    plot_scatter_tool,
    plot_time_series_tool,
)

State = Annotated[dict, InjectedState]
CallId = Annotated[str, InjectedToolCallId]


class NoDatasetError(ValueError):
    pass


def _dataset(state: dict) -> str:
    path = state.get("dataset_path")
    if not path:
        raise NoDatasetError("No dataset is loaded. Ask the user for a CSV, Excel or Parquet file.")
    return path


def _reply(call_id: str, content: dict, **update) -> Command:
    message = ToolMessage(json.dumps(content, ensure_ascii=False, default=str), tool_call_id=call_id)
    return Command(update={**update, "messages": [message]})


def dataset_state(file_path: str, processed_dir: str) -> dict:
    """State fields for a newly selected dataset (used by load_dataset and by the UI)."""
    path = Path(file_path).expanduser()
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {path.suffix!r}; use one of {sorted(SUPPORTED_FORMATS)}.")
    parquet = convert_to_parquet(path, processed_dir)
    return {
        "source_path": str(path),
        "dataset_path": str(parquet),
        "dataset_name": path.name,
        "cleaned": False,
    }


def make_agent_tools(processed_dir: str) -> list:
    """Build the agent's tool list. `processed_dir` receives converted and cleaned files."""

    # ------------------------------------------------------------ data

    @tool
    def load_dataset(file_path: str, tool_call_id: CallId) -> Command:
        """Load a CSV, Excel or Parquet file given by the user and make it the active dataset."""
        try:
            update = dataset_state(file_path, processed_dir)
        except (OSError, ValueError) as error:
            return _reply(tool_call_id, {"error": str(error)})
        content = {"loaded": update["dataset_name"], **analysis_tools.inspect_dataset(update["dataset_path"])}
        return _reply(tool_call_id, content, **update)

    @tool
    def inspect_dataset(state: State) -> dict:
        """List the exact column names and types of the active dataset."""
        return analysis_tools.inspect_dataset(_dataset(state))

    @tool
    def audit_data_quality(state: State) -> dict:
        """Report rows, missing values per column, duplicates and the detected time column."""
        report = audit_dataset(_dataset(state))
        keep = ("name", "dtype", "missing_count", "null_percentage", "max_missing_run")
        return {
            "row_count": report["row_count"],
            "duplicate_rows": report["duplicate_rows"],
            "temporal_info": report["temporal_info"],
            "columns": [{k: c.get(k) for k in keep} for c in report["columns"]],
        }

    @tool
    def clean_dataset(state: State, tool_call_id: CallId) -> Command:
        """Start the guided cleaning workflow (audit, questions to the user, approval, export).

        The user answers the questions directly in the interface. When it ends,
        the cleaned file becomes the active dataset.
        """
        # Imported here: agent.graph imports services that import tools (cycle).
        from agent.graph import run_cleaning_workflow

        source = state.get("source_path") or _dataset(state)
        # `interrupt` pauses the agent and sends each question to the interface.
        # thread_id is tied to this tool call so that resuming replays the same run.
        outcome = run_cleaning_workflow(source, processed_dir, interrupt, thread_id=tool_call_id)
        content = {
            "cleaned_dataset": Path(outcome["parquet_path"]).name,
            "approved_operations": [
                {"operation": op["operation"], "column": op.get("column"), "parameters": op.get("parameters")}
                for op in outcome["approved_operations"]
            ],
            "rejected_operations": outcome["rejected_operations"],
            "rows_before_after": [outcome["check"]["before"]["rows"], outcome["check"]["after"]["rows"]],
            "missing_values_after": outcome["check"]["after"]["missing"],
            "warnings": outcome["check"]["warnings"],
        }
        return _reply(tool_call_id, content, dataset_path=outcome["parquet_path"], cleaned=True)

    # ------------------------------------------------------------ statistics

    @tool
    def describe_column(column: str, state: State) -> dict:
        """Summarize a numeric column over the whole file (count, mean, std, min, max, quartiles)."""
        return analysis_tools.describe_column(_dataset(state), column)

    @tool
    def describe_period(column: str, time_column: str, start: str, end: str, state: State) -> dict:
        """Summarize a numeric column over [start, end), with ISO 8601 bounds such as 2026-01-31T00:00:00."""
        return analysis_tools.describe_column(
            _dataset(state), column, time_column=time_column, start=start, end=end)

    # ------------------------------------------------------------ charts

    @tool
    def plot_time_series(
        value_cols: list[str],
        state: State,
        time_col: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        normalize: bool = False,
    ) -> dict:
        """Line chart of numeric columns over time (trends, gaps, comparing sensors).

        normalize=True rescales each curve when scales differ. Dates are
        'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        """
        return plot_time_series_tool.func(
            _dataset(state), value_cols, time_col, start_date, end_date, normalize)

    @tool
    def plot_histogram(
        column: str,
        state: State,
        bins: int = 30,
        time_col: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Histogram of one numeric column, to inspect its distribution."""
        return plot_histogram_tool.func(_dataset(state), column, bins, time_col, start_date, end_date)

    @tool
    def plot_boxplot(
        columns: list[str],
        state: State,
        group_by: str | None = None,
        time_col: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Box plot of numeric columns, to compare spread and spot outliers.

        With group_by (a categorical column), pass a single numeric column.
        """
        return plot_boxplot_tool.func(_dataset(state), columns, group_by, time_col, start_date, end_date)

    @tool
    def plot_scatter(
        x_col: str,
        y_col: str,
        state: State,
        time_col: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Scatter plot of two numeric columns, to inspect their relationship."""
        return plot_scatter_tool.func(_dataset(state), x_col, y_col, time_col, start_date, end_date)

    @tool
    def plot_correlation_heatmap(
        state: State,
        columns: list[str] | None = None,
        time_col: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Correlation matrix of numeric columns (all numeric columns if none are given)."""
        return plot_correlation_heatmap_tool.func(_dataset(state), columns, time_col, start_date, end_date)

    return [
        load_dataset,
        inspect_dataset,
        audit_data_quality,
        clean_dataset,
        describe_column,
        describe_period,
        plot_time_series,
        plot_histogram,
        plot_boxplot,
        plot_scatter,
        plot_correlation_heatmap,
    ]
