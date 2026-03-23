"""
evaluation.py
=============
Model evaluation and monitoring for the drift-detection project.

This module is the monitoring layer: it connects statistical drift signals (from ``rolling_stats.py``) with model performance signals
(from ``model.py``) to answer the central question:

    Does distribution shift predict model performance degradation?

Responsibilities
----------------
- Financial performance metrics (Sharpe, directional accuracy, RMSE, MAE).
- Rolling model performance tracking over time.
- Alignment and correlation of drift metrics with model metrics.
- Regime-stratified performance analysis (how does the model perform within each detected regime?).
- Visualisation helpers: four-panel monitoring dashboard, regime bar charts, and drift signal plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rolling_stats import compute_rolling_drift, rolling_ks_test


# ---------------------------------------------------------------------------
# Style defaults (applied globally when this module is imported)
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Financial Metrics
# ---------------------------------------------------------------------------


def annualised_sharpe(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    """Annualised Sharpe ratio.

    Sharpe = (mean(r) - rf) / std(r) * sqrt(periods)

    Parameters
    ----------
    returns:
        Period return series.
    rf:
        Per-period risk-free rate (default 0).
    periods:
        Number of periods per year (252 for daily data).

    Returns
    -------
    float
    """
    excess = returns - rf
    if excess.std() == 0:
        return np.nan
    return float((excess.mean() / excess.std()) * np.sqrt(periods))


def max_drawdown(prices: np.ndarray) -> float:
    """Maximum drawdown from cumulative peak (value in [-1, 0]).

    Parameters
    ----------
    prices:
        Cumulative wealth series (not returns).

    Returns
    -------
    float
    """
    roll_max = np.maximum.accumulate(prices)
    drawdown = (prices - roll_max) / roll_max
    return float(drawdown.min())


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the correct directional sign.

    Directional accuracy > 0.5 is a necessary (not sufficient)
    condition for a profitable long/short strategy.

    Parameters
    ----------
    y_true, y_pred:
        Actual and predicted returns.

    Returns
    -------
    float in [0, 1].
    """
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a full suite of regression and financial evaluation metrics.

    Returns
    -------
    dict with keys: rmse, mae, r2, directional_accuracy.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Rolling Performance
# ---------------------------------------------------------------------------


def rolling_prediction_metrics(pred_df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Compute rolling RMSE, MAE, and directional accuracy.

    Parameters
    ----------
    pred_df:
        DataFrame with columns ``y_true`` and ``y_pred``.
    window:
        Rolling window in trading days.  Default 63 ≈ one quarter.

    Returns
    -------
    pd.DataFrame
        Columns: rolling_rmse, rolling_mae, rolling_dir_acc.
    """
    sq_err = (pred_df["y_true"] - pred_df["y_pred"]).pow(2)
    abs_err = (pred_df["y_true"] - pred_df["y_pred"]).abs()
    correct_dir = (
        np.sign(pred_df["y_true"]) == np.sign(pred_df["y_pred"])).astype(float)

    return pd.DataFrame(
        {
            "rolling_rmse": np.sqrt(sq_err.rolling(window).mean()),
            "rolling_mae": abs_err.rolling(window).mean(),
            "rolling_dir_acc": correct_dir.rolling(window).mean(),
        }
    )


# ---------------------------------------------------------------------------
# Drift-Performance Correlation
# ---------------------------------------------------------------------------


def drift_performance_correlation(returns: pd.Series, pred_df: pd.DataFrame, drift_window: int = 252) -> pd.DataFrame:
    """Align drift metrics with model performance and merge into one table.

    This is the analytical core of the monitoring pipeline.  
    It tests the hypothesis that PSI / KS-statistic spikes *precede* RMSE spikes.

    Parameters
    ----------
    returns:
        Raw return series.
    pred_df:
        Walk-forward predictions with columns ``y_true`` and ``y_pred``.
    drift_window:
        Reference window for drift computation.

    Returns
    -------
    pd.DataFrame
        Merged table with drift metrics and rolling performance metrics.
    """
    drift_df = compute_rolling_drift(returns, ref_window=drift_window)
    ks_df = rolling_ks_test(returns, ref_window=drift_window)
    perf_df = rolling_prediction_metrics(pred_df)

    merged = pd.concat([drift_df, ks_df[["ks_stat", "ks_pvalue", "drift_flag"]], perf_df], axis=1)
    merged.dropna(inplace=True)
    return merged


def print_correlation_summary(merged_df: pd.DataFrame) -> None:
    """Print Pearson correlations between drift metrics and RMSE.

    Statistically significant positive correlation between drift intensity and RMSE confirms that distribution shift degrades model performance.
    """
    drift_cols = ["psi", "kl_div", "ks_stat"]
    perf_cols = ["rolling_rmse", "rolling_dir_acc"]

    print("\n── Drift → Performance Correlation (Pearson r) ──────────────────────")
    for dc in drift_cols:
        for pc in perf_cols:
            if dc not in merged_df or pc not in merged_df:
                continue
            common = merged_df[[dc, pc]].dropna()
            if len(common) < 10:
                continue
            r, p = pearsonr(common[dc], common[pc])
            sig = (
                "***" if p < 0.001
                else "**" if p < 0.01
                else "*" if p < 0.05
                else ""
            )
            print(
                f"  {dc:10s} vs {pc:22s}: r = {r:+.3f}  p = {p:.4f}  {sig}"
            )
    print()


# ---------------------------------------------------------------------------
# Regime-Stratified Analysis
# ---------------------------------------------------------------------------


def regime_performance_table(pred_df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Compute model metrics separately for each detected regime.

    Parameters
    ----------
    pred_df:
        Predictions with columns ``y_true`` and ``y_pred``.
    regimes:
        Integer regime labels aligned with ``pred_df``.

    Returns
    -------
    pd.DataFrame
        Per-regime RMSE, directional accuracy, and sample count.
    """
    from regime_detector import REGIME_NAMES

    aligned = pd.concat([pred_df, regimes.rename("regime")], axis=1).dropna()

    rows = []
    for regime_id, grp in aligned.groupby("regime"):
        metrics = compute_model_metrics(
            grp["y_true"].values, grp["y_pred"].values
        )
        metrics["regime"] = REGIME_NAMES.get(int(regime_id), f"regime_{regime_id}")
        metrics["n_days"] = len(grp)
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("regime")
    return df


# ---------------------------------------------------------------------------
# Visualisation Helpers
# ---------------------------------------------------------------------------


def plot_regime_overlay(prices: pd.Series, regimes: pd.Series, title: str = "Market Regime Overlay on SPY Price",
    figsize: tuple = (14, 5)) -> plt.Figure:
    """Plot price series with coloured regime shading.

    Parameters
    ----------
    prices:
        Adjusted close price series.
    regimes:
        Integer regime labels with DatetimeIndex.
    title:
        Plot title.
    figsize:
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    from regime_detector import REGIME_NAMES

    palette = {0: "#2ecc71", 1: "#e74c3c", 2: "#f39c12"}   # green / red / orange

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(prices.index, prices.values, color="#2c3e50", linewidth=0.9, label="SPY Close")

    # Shade regime periods.
    aligned_regimes = regimes.reindex(prices.index, method="ffill")
    current_regime = None
    start_date = None

    for date, regime in aligned_regimes.items():
        if regime != current_regime:
            if current_regime is not None and start_date is not None:
                ax.axvspan(
                    start_date,
                    date,
                    alpha=0.20,
                    color=palette.get(int(current_regime), "grey"),
                    lw=0,
                )
            current_regime = regime
            start_date = date

    # Close the last span.
    if current_regime is not None and start_date is not None:
        ax.axvspan(
            start_date,
            aligned_regimes.index[-1],
            alpha=0.20,
            color=palette.get(int(current_regime), "grey"),
            lw=0,
        )

    # Legend patches.
    patches = [
        mpatches.Patch(
            color=palette.get(k, "grey"),
            alpha=0.5,
            label=REGIME_NAMES.get(k, f"regime_{k}"),
        )
        for k in sorted(palette)
        if k in aligned_regimes.unique()
    ]
    ax.legend(handles=[ax.lines[0]] + patches, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig


def plot_drift_signals(drift_df: pd.DataFrame, ks_df: pd.DataFrame, figsize: tuple = (14, 8)) -> plt.Figure:
    """Two-panel drift signal chart: PSI (top) and KS statistic (bottom).

    Parameters
    ----------
    drift_df:
        Output of ``compute_rolling_drift`` (columns: psi, kl_div).
    ks_df:
        Output of ``rolling_ks_test`` (columns: ks_stat, drift_flag).
    figsize:
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)

    # Panel 1: PSI
    ax = axes[0]
    ax.plot(drift_df.index, drift_df["psi"], color="#e74c3c", linewidth=1.2, label="PSI")
    ax.axhline(0.10, color="orange", linestyle="--", linewidth=0.9, label="PSI = 0.10 (moderate)")
    ax.axhline(0.20, color="red", linestyle="--", linewidth=0.9, label="PSI = 0.20 (severe)")
    ax.set_title("Population Stability Index (PSI)")
    ax.set_ylabel("PSI")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: KS Statistic
    ax = axes[1]
    ax.plot(ks_df.index, ks_df["ks_stat"], color="#2980b9", linewidth=1.2, label="KS Statistic")

    # Shade periods where drift is statistically significant (p < 0.05).
    if "drift_flag" in ks_df.columns:
        in_drift = False
        drift_start = None
        for date, flag in ks_df["drift_flag"].items():
            if flag and not in_drift:
                drift_start = date
                in_drift = True
            elif not flag and in_drift:
                ax.axvspan(drift_start, date, color="red", alpha=0.08, lw=0)
                in_drift = False
        if in_drift:
            ax.axvspan(drift_start, ks_df.index[-1], color="red", alpha=0.08, lw=0)

    ax.set_title("Rolling KS Test Statistic (p < 0.05 shaded red)")
    ax.set_ylabel("KS Statistic")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Distribution Drift Signals — SPY Log Returns", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_monitoring_dashboard(pred_df: pd.DataFrame, merged_df: pd.DataFrame, figsize: tuple = (16, 14)) -> plt.Figure:
    """Four-panel model monitoring dashboard.

    Panels (top to bottom):
    1. Actual vs. predicted returns.
    2. Rolling RMSE over time.
    3. PSI drift metric over time.
    4. KS test statistic with drift-alarm shading.

    Parameters
    ----------
    pred_df:
        Columns ``y_true`` and ``y_pred``.
    merged_df:
        Output of ``drift_performance_correlation``.
    figsize:
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Panel 1: Actual vs Predicted
    ax = axes[0]
    ax.plot(pred_df.index, pred_df["y_true"],
            label="Actual", color="#2c3e50", alpha=0.55, linewidth=0.7)
    ax.plot(pred_df.index, pred_df["y_pred"],
            label="Predicted", color="#e67e22", alpha=0.80, linewidth=0.7)
    ax.set_title("Actual vs. Predicted Log Returns")
    ax.set_ylabel("Log Return")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: Rolling RMSE
    ax = axes[1]
    if "rolling_rmse" in merged_df.columns:
        ax.plot(merged_df.index, merged_df["rolling_rmse"],
                color="#8e44ad", linewidth=1.2)
    ax.set_title("Rolling RMSE (63-day window)")
    ax.set_ylabel("RMSE")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 3: PSI
    ax = axes[2]
    if "psi" in merged_df.columns:
        ax.plot(merged_df.index, merged_df["psi"],
                color="#e74c3c", linewidth=1.2, label="PSI")
        ax.axhline(0.10, color="orange", linestyle="--",
                   linewidth=0.8, label="PSI=0.10 (moderate)")
        ax.axhline(0.20, color="red", linestyle="--",
                   linewidth=0.8, label="PSI=0.20 (severe)")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Population Stability Index")
    ax.set_ylabel("PSI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 4: KS Statistic
    ax = axes[3]
    if "ks_stat" in merged_df.columns:
        ax.plot(merged_df.index, merged_df["ks_stat"],
                color="#2980b9", linewidth=1.2, label="KS Statistic")
        if "drift_flag" in merged_df.columns:
            in_drift = False
            drift_start = None
            for date, flag in merged_df["drift_flag"].items():
                if flag and not in_drift:
                    drift_start = date
                    in_drift = True
                elif not flag and in_drift:
                    ax.axvspan(drift_start, date, color="red", alpha=0.10, lw=0)
                    in_drift = False
            if in_drift:
                ax.axvspan(drift_start, merged_df.index[-1],
                           color="red", alpha=0.10, lw=0)
        ax.legend(fontsize=8)
    ax.set_title("KS Test Statistic (drift alarms shaded)")
    ax.set_ylabel("KS Statistic")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "Model Monitoring Dashboard — SPY Return Prediction",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_regime_performance_bar(regime_perf_df: pd.DataFrame, metric: str = "directional_accuracy", figsize: tuple = (10, 5)) -> plt.Figure:
    """Bar chart of a performance metric stratified by regime.

    Parameters
    ----------
    regime_perf_df:
        Output of ``regime_performance_table``.
    metric:
        Column from ``regime_perf_df`` to plot.
    figsize:
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    df = regime_perf_df[[metric]].dropna()
    colors = [
        "#2ecc71" if (metric == "directional_accuracy" and v > 0.5)
        else "#e74c3c"
        for v in df[metric]
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df.index, df[metric], color=colors, edgecolor="white", width=0.5)

    if metric == "directional_accuracy":
        ax.axhline(0.5, color="grey", linestyle="--",
                   linewidth=1, label="Random baseline (50%)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    for bar, val in zip(bars, df[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_title(f"Model {metric.replace('_', ' ').title()} by Market Regime")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Regime")
    fig.tight_layout()
    return fig
