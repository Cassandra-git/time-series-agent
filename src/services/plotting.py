"""Plotly figures for time series stored as Parquet files.

Every function takes a parquet path and returns a plotly Figure.
Large datasets are thinned out so that the figures stay light.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

MAX_POINTS = 50_000  # above this, we keep only some of the points


# ============================================================
# HELPERS
# ============================================================

def check_columns(df, columns):
    """Raise an error if a column does not exist."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )


def check_numeric(df, columns):
    """Raise an error if a column is not numeric."""
    for c in columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Column '{c}' is not numeric.")


def check_value_columns(df, columns):
    """Check that the columns are given, exist and are numeric."""
    if not columns:
        raise ValueError("At least one column must be provided.")
    check_columns(df, columns)
    check_numeric(df, columns)


def sample_rows(data, n=MAX_POINTS):
    """Random sample (same result every time) if the data is too big."""
    if len(data) > n:
        return data.sample(n=n, random_state=0)
    return data


def load_data(parquet_path, time_col=None, start_date=None, end_date=None):
    """Read the parquet file and optionally filter it on the time column.

    start_date / end_date are strings like "2024-01-31".
    If end_date has no time, the whole day is included.
    If the time column is numeric (seconds, sample index...),
    the dates are read as numbers.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("The file contains no data.")

    if time_col is None:
        if start_date is not None or end_date is not None:
            raise ValueError("time_col is required to filter by date.")
        return df

    check_columns(df, [time_col])
    is_number = pd.api.types.is_numeric_dtype(df[time_col])

    if not is_number:
        df[time_col] = pd.to_datetime(df[time_col])

    if start_date is not None:
        if is_number:
            df = df[df[time_col] >= float(start_date)]
        else:
            df = df[df[time_col] >= pd.Timestamp(start_date)]

    if end_date is not None:
        if is_number:
            df = df[df[time_col] <= float(end_date)]
        elif len(end_date) <= 10:
            df = df[df[time_col] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
        else:
            df = df[df[time_col] <= pd.Timestamp(end_date)]

    if df.empty:
        raise ValueError("No data found for the selected dates.")

    return df.sort_values(time_col)


# ============================================================
# TIME SERIES
# ============================================================

def plot_time_series(
    parquet_path,
    time_col,
    value_cols,
    start_date=None,
    end_date=None,
    normalize=False,
):
    """Plot one or more variables over time.

    If time_col is None, the row number is used as x axis.
    normalize=True rescales each curve (mean 0, std 1) so that
    curves with very different scales can be compared.
    """
    df = load_data(parquet_path, time_col, start_date, end_date)
    check_value_columns(df, value_cols)

    # Keep one point out of "step" if there are too many points
    step = max(1, len(df) // MAX_POINTS)
    df = df.iloc[::step]

    if time_col is None:
        x = list(range(len(df)))
    else:
        x = df[time_col]

    fig = go.Figure()

    for col in value_cols:
        y = df[col]
        if normalize and y.std() > 0:
            y = (y - y.mean()) / y.std()

        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=col))

    fig.update_layout(
        title="Time Series",
        xaxis_title=time_col or "row number",
        yaxis_title="Normalized value" if normalize else "Value",
        hovermode="x unified",
    )
    return fig


# ============================================================
# HISTOGRAM
# ============================================================

def plot_histogram(
    parquet_path,
    column,
    bins=30,
    time_col=None,
    start_date=None,
    end_date=None,
):
    """Plot the distribution of one numeric variable."""
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer.")

    df = load_data(parquet_path, time_col, start_date, end_date)
    check_value_columns(df, [column])

    values = sample_rows(df[column].dropna())

    fig = go.Figure(go.Histogram(x=values, nbinsx=bins, name=column))
    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Frequency",
    )
    return fig


# ============================================================
# BOXPLOT
# ============================================================

def plot_boxplot(
    parquet_path,
    columns,
    time_col=None,
    start_date=None,
    end_date=None,
):
    """Plot the distribution of one or more variables."""
    df = load_data(parquet_path, time_col, start_date, end_date)
    check_value_columns(df, columns)

    fig = go.Figure()

    for col in columns:
        values = sample_rows(df[col].dropna())
        fig.add_trace(go.Box(y=values, name=col, boxmean=True))

    fig.update_layout(title="Boxplot", yaxis_title="Value")
    return fig


# ============================================================
# SCATTER PLOT
# ============================================================

def plot_scatter(
    parquet_path,
    x_col,
    y_col,
    time_col=None,
    start_date=None,
    end_date=None,
):
    """Plot the relationship between two numeric variables."""
    df = load_data(parquet_path, time_col, start_date, end_date)
    check_value_columns(df, [x_col, y_col])

    pairs = sample_rows(df[[x_col, y_col]].dropna())

    fig = go.Figure(
        go.Scattergl(
            x=pairs[x_col],
            y=pairs[y_col],
            mode="markers",
            name=f"{x_col} vs {y_col}",
        )
    )
    fig.update_layout(
        title=f"{x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(
    parquet_path,
    columns=None,
    time_col=None,
    start_date=None,
    end_date=None,
):
    """Plot the correlation matrix of numeric variables.

    If columns is None, all numeric columns are used (except time_col).
    """
    df = load_data(parquet_path, time_col, start_date, end_date)

    if columns is None:
        numeric_df = df.select_dtypes(include="number")
        if time_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[time_col])
    else:
        check_value_columns(df, columns)
        numeric_df = df[columns]

    if numeric_df.shape[1] < 2:
        raise ValueError("At least two numeric columns are required.")

    corr = numeric_df.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar_title="Correlation",
        )
    )
    fig.update_layout(title="Correlation Heatmap")
    return fig