"""
rolling_stats.py
================
Statistical drift detection for financial time series.

Implements three complementary methods for detecting distribution shift:


1. **Rolling KS Test** (Kolmogorov-Smirnov)
   Non-parametric two-sample test comparing the distribution of a sliding current window against a fixed reference window. Sensitive
   to *any* distributional change, not just mean or variance shifts.

2. **PSI** (Population Stability Index)
   Industry standard for monitoring feature drift in deployed ML models.It originated in credit risk model monitoring.
   PSI < 0.10 -> stable | 0.10–0.20 -> moderate | > 0.20 -> severe drift. 

3. **KL Divergence** (Kullback-Leibler)
   Information-theoretic measure of how much information is lost when the reference distribution is used to approximate the current one.

Reference windows default to one trading year (252 days).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats



# ---------------------------------------------------------------------------
# 1. Rolling KS Test — Distribution Shift
# ---------------------------------------------------------------------------

def rolling_ks_test(series: pd.Series, ref_window: int = 252, test_window: int = 63) -> pd.DataFrame:
    """Rolling two-sample Kolmogorov-Smirnov test.

    At each time step, compares the distribution of the most recent ``test_window`` observations against the preceding ``ref_window``
    observations.

    A large KS statistic (and p-value < 0.05) means the two samples are unlikely to come from the same distribution (a distribution
    shift has occurred).

    The KS test makes no assumption about the shape of either distribution, making it robust to heavy tails and regime transitions.

    Parameters
    ----------
    series:
        Return series.
    ref_window:
        Size of the reference (historical) window in trading days.
    test_window:
        Size of the current (test) window in trading days.
        Default 63 ≈ one quarter.

    Returns
    -------
    pd.DataFrame
        Columns:
        - ``ks_stat``    : KS test statistic (in [0, 1])
        - ``ks_pvalue``  : p-value of the two-sample KS test
        - ``drift_flag`` : True when p-value < 0.05 (significant shift)
    """
    min_idx = ref_window + test_window
    results = []

    for i in range(min_idx, len(series)):
        ref = series.iloc[i - test_window - ref_window : i - test_window].values
        cur = series.iloc[i - test_window : i].values
        stat, pval = stats.ks_2samp(ref, cur)
        results.append((series.index[i], stat, pval))

    df = pd.DataFrame(results, columns=["Date", "ks_stat", "ks_pvalue"])
    df.set_index("Date", inplace=True)
    df["drift_flag"] = df["ks_pvalue"] < 0.05

    return df


# ---------------------------------------------------------------------------
# 2. PSI — Population Stability Index
# ---------------------------------------------------------------------------

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-8) -> float:
    """Compute the Population Stability Index (PSI).

    PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)

    where ref_i and cur_i are the fraction of observations in bin i for the reference and current distributions respectively.

    Bin edges are defined by percentiles of the reference distribution to prevent look-ahead bias.

    Interpretation
    --------------
    PSI < 0.10   → Negligible drift (model is stable)
    0.10 - 0.20  → Moderate drift  (monitor closely)
    PSI > 0.20   → Significant drift (retrain / investigate)

    Parameters
    ----------
    reference:
        Baseline (historical) sample.
    current:
        Current sample to compare against the baseline.
    n_bins:
        Number of histogram bins.  Default 10.
    eps:
        Small constant to avoid log(0) when a bin is empty.

    Returns
    -------
    float
        PSI value (non-negative; 0 = identical distributions).
    """
    # Define bin edges from the reference distribution only.
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)

    # Clip to avoid log(0) for empty bins.
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# 3. KL Divergence — Information-Theoretic Drift
# ---------------------------------------------------------------------------


def compute_kl_divergence(reference: np.ndarray, current: np.ndarray, n_bins: int = 50, eps: float = 1e-8) -> float:
    """Kullback-Leibler divergence D_KL(current || reference).

    KL divergence measures the information lost when reference is used to approximate current.  
    
    It is asymmetric: D_KL(P||Q) != D_KL(Q||P).

    Both distributions are estimated non-parametrically via histograms with uniform bin widths spanning the combined range.

    Parameters
    ----------
    reference:
        Baseline (historical) empirical distribution.
    current:
        Current empirical distribution.
    n_bins:
        Number of histogram bins for density estimation.
    eps:
        Numerical stability constant.

    Returns
    -------
    float
        KL divergence (>= 0; = 0 iff distributions are identical).
    """
    combined_min = min(reference.min(), current.min())
    combined_max = max(reference.max(), current.max())
    bins = np.linspace(combined_min, combined_max, n_bins + 1)

    p_hist, _ = np.histogram(reference, bins=bins, density=True)
    q_hist, _ = np.histogram(current, bins=bins, density=True)

    p_hist = np.clip(p_hist, eps, None)
    q_hist = np.clip(q_hist, eps, None)

    return float(stats.entropy(q_hist, p_hist))


# ---------------------------------------------------------------------------
# Rolling Drift Pipeline
# ---------------------------------------------------------------------------


def compute_rolling_drift(series: pd.Series, ref_window: int = 252, step: int = 21) -> pd.DataFrame:
    """Compute PSI and KL divergence on a rolling basis.

    At each step, the preceding ``ref_window`` observations form the reference distribution. 
    The following ``ref_window // 4`` (≈ one quarter) form the current distribution.

    Both metrics are computed at monthly intervals (step ≈ 21 trading days) to produce a time series of drift intensity.

    Parameters
    ----------
    series:
        Return series.
    ref_window:
        Reference window size in trading days (default 252 = 1 year).
    step:
        Number of days to advance between measurements (default 21 ≈ 1 month).

    Returns
    -------
    pd.DataFrame
        Columns: ``psi``, ``kl_div``
    """
    test_window = ref_window // 4      # ~63 days (one quarter)
    min_start = ref_window + test_window
    records = []

    for i in range(min_start, len(series), step):
        ref = series.iloc[i - test_window - ref_window : i - test_window].values
        cur = series.iloc[i - test_window : i].values
        psi = compute_psi(ref, cur)
        kl = compute_kl_divergence(ref, cur)
        records.append((series.index[i], psi, kl))

    df = pd.DataFrame(records, columns=["Date", "psi", "kl_div"])
    df.set_index("Date", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Legacy helper (kept for backwards compatibility with notebook 01)
# ---------------------------------------------------------------------------


def compute_rolling_variance(data: pd.Series, window: int = 252) -> pd.DataFrame:
    """Compute rolling standard deviation.

    Parameters
    ----------
    data:
        Input return series.
    window:
        Rolling window size (default 252 trading days).

    Returns
    -------
    pd.DataFrame
        Single column ``rolling_std``.
    """
    rolling_std = data.rolling(window=window).std()
    return pd.DataFrame({"rolling_std": rolling_std})
