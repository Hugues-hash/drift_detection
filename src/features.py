"""
features.py
===========
Feature engineering for financial regime detection.

Computes a set of rolling statistical features of a return series across multiple time scales.  

These features serve two purposes:
1. Inputs to the regime detector: HMM use these features to distinguish low-volatility from high-volatility regimes.
2. Inputs to the predictive model: Ridge regression uses lagged versions of these features as predictors for next-day returns.

Features
----------------
Rolling statistics  : mean, volatility, realized volatility, skewness, kurtosis
Risk / momentum     : rolling Sharpe, cumulative momentum, drawdown slope
Structural          : volatility-of-volatility, autocorrelation
Lagged returns      : AR-type raw predictors

Windows: 20, 60, 252 trading days. This covers one month, one quarter, and one year of history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing import TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Individual Feature Functions
# ---------------------------------------------------------------------------


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Annualised rolling mean of returns.

    Captures the trend direction within the window.

    Parameters
    ----------
    series:
        Daily return series.
    window:
        Look-back window in trading days.

    Returns
    -------
    pd.Series named ``roll_mean_{window}d``.
    """
    return (series.rolling(window).mean() * TRADING_DAYS_PER_YEAR).rename(f"roll_mean_{window}d")


def rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    """Annualised rolling volatility (standard deviation).

    sigma_annual = sigma_daily * sqrt(252)

    Implied volatility tracks expectations and rolling historical volatility tracks realised conditions.

    Parameters
    ----------
    series:
        Daily return series.
    window:
        Look-back window in trading days.

    Returns
    -------
    pd.Series named ``roll_vol_{window}d``.
    """
    return (series.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)).rename(f"roll_vol_{window}d")


def realized_volatility(series: pd.Series, window: int) -> pd.Series:
    """Realized volatility: square root of rolling sum of squared returns.

    RV_t = sqrt( sum_{s=t-w+1}^{t} r_s^2 )

    Parameters
    ----------
    series:
        Daily return series.
    window:
        Look-back window in trading days.

    Returns
    -------
    pd.Series named ``realized_vol_{window}d``.
    """
    return np.sqrt(series.pow(2).rolling(window).sum()).rename(f"realized_vol_{window}d")


def rolling_skewness(series: pd.Series, window: int) -> pd.Series:
    """Rolling skewness of returns.

    Parameters
    ----------
    series:
        Daily return series.
    window:
        Look-back window in trading days.

    Returns
    -------
    pd.Series named ``roll_skew_{window}d``.
    """
    return series.rolling(window).skew().rename(f"roll_skew_{window}d")


def rolling_kurtosis(series: pd.Series, window: int) -> pd.Series:
    """Rolling excess kurtosis.

    Spikes in kurtosis indicate fat-tailed return distributions which is a reliable signature 
    of high-volatility / crisis regimes where extreme moves become far more frequent than a Gaussian would predict.

    Parameters
    ----------
    series:
        Daily return series.
    window:
        Look-back window in trading days.

    Returns
    -------
    pd.Series named ``roll_kurt_{window}d``.
    """
    return series.rolling(window).kurt().rename(f"roll_kurt_{window}d")


def rolling_sharpe(series: pd.Series, window: int, rf: float = 0.0) -> pd.Series:
    """Rolling annualised Sharpe ratio.

    Sharpe_t = (mu_t - rf) / sigma_t * sqrt(252)

    A composite risk-adjusted return measure. Dropping Sharpe during periods of stable volatility signals deteriorating return quality
    which is a precursor of regime transitions.

    Parameters
    ----------
    series:
        Daily return series.
    window:
        Look-back window in trading days.
    rf:
        Daily risk-free rate (default 0, i.e. excess return = raw return).

    Returns
    -------
    pd.Series named ``roll_sharpe_{window}d``.
    """
    excess = series - rf
    sharpe = (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe.rename(f"roll_sharpe_{window}d")


def momentum(series: pd.Series, window: int) -> pd.Series:
    """Rolling cumulative return (momentum signal).

    mom_t = prod_{s=t-w+1}^{t} (1 + r_s) - 1

    Positive momentum is associated with trending bull regimes; negative momentum with bear regimes.  
    The 12-month minus 1-month momentum factor is one of the most robust anomalies in empirical finance.

    Parameters
    ----------
    series:
        Daily simple return series (not log returns).
    window:
        Look-back window in trading days.

    Returns
    -------
    pd.Series named ``momentum_{window}d``.
    """
    cum_ret = (1 + series).rolling(window).apply(np.prod, raw=True) - 1
    return cum_ret.rename(f"momentum_{window}d")


def volatility_of_volatility(series: pd.Series, vol_window: int = 20, vov_window: int = 60) -> pd.Series:
    """Volatility of volatility (VoV).

    VoV = std( roll_vol_t, ..., roll_vol_{t-vov_window} )

    VoV measures how much volatility itself is changing over time.
    Rising VoV is a forward-looking indicator of regime transitions: as market uncertainty about the regime grows, the vol of vol spikes
    before the actual vol shift fully materialises.

    Parameters
    ----------
    series:
        Daily return series.
    vol_window:
        Inner window for estimating rolling volatility.
    vov_window:
        Outer window for estimating volatility of those vol estimates.

    Returns
    -------
    pd.Series named ``vov_{vol_window}_{vov_window}d``.
    """
    vol = series.rolling(vol_window).std()
    return vol.rolling(vov_window).std().rename(f"vov_{vol_window}_{vov_window}d")


def rolling_autocorrelation(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    """Rolling lag-1 autocorrelation.

    Low (near-zero) autocorrelation signals efficient / random-walk market.
    High positive autocorrelation   signals trending / momentum regime.
    High negative autocorrelation   signals mean-reverting / oversold regime.

    Parameters
    ----------
    series:
        Daily return series.
    window:
        Look-back window in trading days.
    lag:
        Lag order (default 1 = consecutive day autocorrelation).

    Returns
    -------
    pd.Series named ``roll_autocorr_{window}d_lag{lag}``.
    """
    return series.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=False).rename(f"roll_autocorr_{window}d_lag{lag}")


# ---------------------------------------------------------------------------
# Feature Matrix Builder
# ---------------------------------------------------------------------------


def build_feature_matrix(returns: pd.Series, windows: list | None = None, include_vov: bool = True, include_autocorr: bool = True) -> pd.DataFrame:
    """Build a comprehensive feature matrix from a return series.

    Features are computed at multiple time scales to capture both
    short-term market dynamics and longer structural trends.

    Parameters
    ----------
    returns:
        Log or simple return series indexed by date.
    windows:
        Rolling windows in trading days.  Defaults to ``[20, 60, 252]``.
    include_vov:
        Include volatility-of-volatility feature (default True).
    include_autocorr:
        Include rolling autocorrelation feature (default True).
        This is the slowest feature to compute; disable for speed.

    Returns
    -------
    pd.DataFrame
        Feature matrix.  Each column is one engineered feature.
        Rows with NaN values (from rolling warm-up) are **retained**;
        callers must handle them explicitly (typically ``dropna()``).
    """
    if windows is None:
        windows = [20, 60, 252]

    feature_list: list[pd.Series] = []

    for w in windows:
        feature_list.append(rolling_mean(returns, w))
        feature_list.append(rolling_volatility(returns, w))
        feature_list.append(realized_volatility(returns, w))
        feature_list.append(rolling_skewness(returns, w))
        feature_list.append(rolling_kurtosis(returns, w))
        feature_list.append(rolling_sharpe(returns, w))
        feature_list.append(momentum(returns, w))

    if include_vov:
        feature_list.append(volatility_of_volatility(returns))

    if include_autocorr:
        # Only compute at the medium window to limit run time.
        feature_list.append(rolling_autocorrelation(returns, windows[1]))

    features = pd.concat(feature_list, axis=1)
    features.index.name = "Date"
    return features


def add_lagged_returns(features: pd.DataFrame, returns: pd.Series, lags: list | None = None) -> pd.DataFrame:
    """Append lagged return columns to an existing feature matrix.

    Lagged returns are the raw AR-type predictors used by the Ridge
    regression model.

    Parameters
    ----------
    features:
        Existing feature DataFrame.
    returns:
        Daily return series.
    lags:
        Lag indices to include.  Defaults to ``[1, 2, 3, 5]``.

    Returns
    -------
    pd.DataFrame
        Feature matrix with additional ``return_lag{k}`` columns.
    """
    if lags is None:
        lags = [1, 2, 3, 5]

    lag_df = pd.concat([returns.shift(lag).rename(f"return_lag{lag}") for lag in lags], axis=1)
    return pd.concat([features, lag_df], axis=1)
