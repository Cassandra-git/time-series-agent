from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


# ============================================================
# HELPERS
# ============================================================

def _load_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """Load a cleaned Parquet file into a DataFrame."""

    path = Path(parquet_path)

    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError("The Parquet file contains no data.")

    return df


def _check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Check that all requested columns exist."""

    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _prepare_dataframe(
    parquet_path: str | Path,
    time_col: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    """Load the Parquet file and optionally filter by date."""

    df = _load_parquet(parquet_path)

    if start_date is not None or end_date is not None:
        if time_col is None:
            raise ValueError("A time column is required for date filtering.")

    if time_col is not None:
        _check_columns(df, [time_col])

        if start_date is not None:
            df = df[df[time_col] >= pd.Timestamp(start_date)]

        if end_date is not None:
            df = df[df[time_col] <= pd.Timestamp(end_date)]

    if df.empty:
        raise ValueError("No data found for the selected criteria.")

    return df


def _validate_value_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> None:
    """Check that at least one requested column exists."""

    if not columns:
        raise ValueError("At least one column must be provided.")

    _check_columns(df, columns)


# ============================================================
# TIME SERIES
# ============================================================

def plot_time_series(
    parquet_path: str | Path,
    time_col: str,
    value_cols: list[str],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None
) -> go.Figure:
    """Plot one or more variables over time."""

    df = _prepare_dataframe(
        parquet_path, time_col, start_date, end_date
    )

    _validate_value_columns(df, value_cols)

    df = df.sort_values(by=time_col)

    fig = go.Figure()

    for col in value_cols:
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[col],
                mode="lines",
                name=col
            )
        )

    fig.update_layout(
        title="Time Series",
        xaxis_title=time_col,
        yaxis_title="Value",
        hovermode="x unified",
        showlegend=True
    )

    return fig


# ============================================================
# HISTOGRAM
# ============================================================

def plot_histogram(
    parquet_path: str | Path,
    column: str,
    bins: int = 30,
    time_col: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None
) -> go.Figure:
    """Plot the distribution of a numeric variable."""

    df = _prepare_dataframe(
        parquet_path, time_col, start_date, end_date
    )

    _validate_value_columns(df, [column])

    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError("bins must be a positive integer.")

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df[column],
            nbinsx=bins,
            name=column
        )
    )

    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Frequency"
    )

    return fig


# ============================================================
# BOXPLOT
# ============================================================

def plot_boxplot(
    parquet_path: str | Path,
    columns: list[str],
    time_col: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None
) -> go.Figure:
    """Plot the distribution of one or more variables."""

    df = _prepare_dataframe(
        parquet_path, time_col, start_date, end_date
    )

    _validate_value_columns(df, columns)

    fig = go.Figure()

    for col in columns:
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                boxmean=True
            )
        )

    fig.update_layout(
        title="Boxplot",
        yaxis_title="Value",
        showlegend=True
    )

    return fig


# ============================================================
# SCATTER PLOT
# ============================================================

def plot_scatter(
    parquet_path: str | Path,
    x_col: str,
    y_col: str,
    time_col: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None
) -> go.Figure:
    """Plot the relationship between two variables."""

    df = _prepare_dataframe(
        parquet_path, time_col, start_date, end_date
    )

    _validate_value_columns(df, [x_col, y_col])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            name=f"{x_col} vs {y_col}"
        )
    )

    fig.update_layout(
        title=f"{x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col
    )

    return fig


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(
    parquet_path: str | Path,
    columns: list[str] | None = None,
    time_col: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None
) -> go.Figure:
    """Plot the correlation matrix of numeric variables."""

    df = _prepare_dataframe(
        parquet_path, time_col, start_date, end_date
    )

    if columns is not None:
        _validate_value_columns(df, columns)
        numeric_df = df[columns]
    else:
        numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError("At least two numeric columns are required.")

    correlation = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar_title="Correlation"
        )
    )

    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Variables",
        yaxis_title="Variables"
    )

    return fig
