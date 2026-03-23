"""
preprocessing.py
================
Data loading and preprocessing pipeline for the SPY financial time series.

Responsibilities
----------------
- Load the raw CSV (Yahoo Finance multi-index format) and flatten headers.
- Compute log returns and simple (arithmetic) returns.
- Validate data for index integrity, duplicates, and price sanity.
- Expose a single ``preprocess_pipeline()`` entry-point used by every
  notebook and downstream module.

-------------------------------
Log returns are preferred throughout this project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
"""Standard annualisation constant for daily equity data."""

RAW_DATA_PATH: Path = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "spy_price.csv"
)
"""Default path to the downloaded SPY price CSV."""


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_raw_data(filepath: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and flatten the raw SPY price CSV.

    Yahoo Finance downloads produce a two-row header:

        Row 0 (Price):  Close  High  Low  Open  Volume
        Row 1 (Ticker): SPY    SPY   SPY  SPY   SPY

    This function flattens those into simple column names and returns a
    DatetimeIndex-indexed DataFrame.

    Parameters
    ----------
    filepath:
        Path to the raw CSV.  Defaults to ``data/raw/spy_price.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: Close, High, Low, Open, Volume  |  Index: Date
    """
    df = pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)

    # Drop the ticker-level header. We only need to track SPY.
    df.columns = df.columns.droplevel(1)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Remove any all-NaN rows that can appear at the CSV boundary.
    df.dropna(how="all", inplace=True)

    # Guarantee chronological order.
    df.sort_index(inplace=True)

    return df


# ---------------------------------------------------------------------------
# Return Calculations
# ---------------------------------------------------------------------------


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute continuously-compounded (log) returns.

    r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    prices:
        Time series of adjusted closing prices.

    Returns
    -------
    pd.Series
        Log return series.  First observation is NaN by construction.
    """
    return np.log(prices / prices.shift(1)).rename("log_return")


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """Compute simple (arithmetic) percentage returns.

    r_t = (P_t - P_{t-1}) / P_{t-1}

    Simple returns are used when computing portfolio-level cumulative
    wealth, since log returns are not additive across assets.

    Parameters
    ----------
    prices:
        Time series of adjusted closing prices.

    Returns
    -------
    pd.Series
        Simple return series.  First observation is NaN.
    """
    return prices.pct_change().rename("simple_return")


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def compute_drawdown(prices: pd.Series) -> pd.DataFrame:
    """Compute rolling drawdown from the running peak price.

    DD_t = (P_t - max_{s<=t} P_s) / max_{s<=t} P_s

    Drawdown is a key regime indicator: persistent and deep drawdowns signal bear-market regimes; 
    flat drawdown near zero indicates bull regimes.

    Parameters
    ----------
    prices:
        Adjusted close price series.

    Returns
    -------
    pd.DataFrame
        Columns:
        - ``drawdown``    : current drawdown (values in [-1, 0])
        - ``rolling_max`` : running peak price
    """
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return pd.DataFrame(
        {"drawdown": drawdown, "rolling_max": rolling_max},
        index=prices.index,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_data(df: pd.DataFrame) -> None:
    """Run sanity checks on the raw DataFrame.

    Raises
    ------
    AssertionError
        On any integrity failure (non-monotonic index, duplicates,
        negative or missing prices).
    """
    assert df.index.is_monotonic_increasing, (
        "Date index is not sorted chronologically."
    )
    assert not df.index.duplicated().any(), (
        "Duplicate timestamps detected in index."
    )
    assert df["Close"].notna().all(), (
        "NaN values found in the Close price series."
    )
    assert (df["Close"] > 0).all(), (
        "Non-positive prices detected — check the source data."
    )


# ---------------------------------------------------------------------------
# Public Pipeline
# ---------------------------------------------------------------------------


def preprocess_pipeline(filepath: str | Path = RAW_DATA_PATH) -> dict:
    """End-to-end preprocessing pipeline.

    This is the single entry-point used by all notebooks and modules.

    Parameters
    ----------
    filepath:
        Path to the raw CSV.  Defaults to ``data/raw/spy_price.csv``.

    Returns
    -------
    dict with keys:

    ``prices``
        Daily adjusted close prices (pd.Series).
    ``log_returns``
        Log return series with the opening NaN dropped.
    ``simple_returns``
        Simple return series with the opening NaN dropped.
    ``drawdown``
        DataFrame with columns ``drawdown`` and ``rolling_max``.
    ``raw_df``
        Full OHLCV DataFrame exactly as loaded from disk.
    """
    raw_df = load_raw_data(filepath)
    validate_data(raw_df)

    prices = raw_df["Close"].rename("price")
    log_returns = compute_log_returns(prices).dropna()
    simple_returns = compute_simple_returns(prices).dropna()
    drawdown = compute_drawdown(prices)

    return {
        "prices": prices,
        "log_returns": log_returns,
        "simple_returns": simple_returns,
        "drawdown": drawdown,
        "raw_df": raw_df,
    }
