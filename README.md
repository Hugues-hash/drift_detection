# Regime Detection in Financial Time Series

An implementation of **market regime detection and distribution drift monitoring** applied to the S&P 500 ETF (SPY).

---

## Problem Statement

Financial time series are **non-stationary**: their statistical properties (mean, variance) change over time. Models trained on historical data silently degrade when the market regime shifts which is a critical failure in deployed quantitative systems.

This project investigates:
- How distribution shift can be **detected** using statistical methods
- How **predictive model performance** evolves under regime transitions
- Whether **drift signals correlate with performance degradation** (enabling proactive monitoring)

---

## Methods

### Regime Detection
| Method | Type | Description |
|---|---|---|
| **Hidden Markov Model** | Probabilistic | Gaussian HMM with 2 latent states (low/high volatility); learns transition dynamics and per-regime emission distributions via Baum-Welch EM |

### Distribution Drift
| Metric | Description |
|---|---|
| **PSI** (Population Stability Index) | From credit risk monitoring: PSI < 0.10 stable, 0.10–0.20 moderate, > 0.20 severe |
| **KS Test** (Kolmogorov-Smirnov) | Non-parametric two-sample test comparing rolling current vs. reference window distributions |
| **KL Divergence** | Information-theoretic measure of how much the current distribution diverges from the reference |

### Predictive Model
- **Ridge Regression** as a stable, interpretable baseline
- **Walk-forward cross-validation** (train=252d, test=63d, step=21d). This avoids look-ahead bias
- StandardScaler fit on training fold only at each step

---

## Project Structure

```
drift_detection/
├── data/
│   └── raw/
│       └── spy_price.csv             # SPY daily data (2005–2026), from Yahoo Finance
│
├── notebooks/
│   ├── 01_exploration.ipynb          # EDA: data validation, returns, rolling stats
│   ├── 02_feature_engineering.ipynb  # Feature engineering: multi-scale rolling features
│   ├── 03_regime_detection.ipynb     # HMM, CUSUM, rolling KS (regime analysis)
│   └── 04_model_monitoring.ipynb     # Walk-forward backtest + drift-performance correlation
│
├── src/
│   ├── preprocessing.py              # Data loading, returns, drawdown, validation pipeline
│   ├── features.py                   # Rolling mean/vol/skew/kurt, momentum, VoV, autocorr
│   ├── rolling_stats.py              # CUSUM, rolling KS test, PSI, KL divergence
│   ├── regime_detector.py            # HMMRegimeDetector, VolatilityRegimeDetector, CUSUM change points
│   ├── model.py                      # Walk-forward Ridge regression, PredictionRecord, aggregation
│   └── evaluation.py                 # Metrics, drift-performance correlation, monitoring dashboard
│
└── requirements.txt
```

---

## Notebooks

### `01_exploration.ipynb`
Exploratory data analysis: data integrity checks, daily return distributions, rolling statistics.

### `02_feature_engineering.ipynb`
Feature construction at three time scales (20d, 60d, 252d): rolling volatility, higher moments (skewness, kurtosis), Sharpe ratio, momentum, volatility-of-volatility, autocorrelation. ADF stationarity tests. Feature correlation heatmap.

### `03_regime_detection.ipynb`
- **HMM**: fits a 2-state Gaussian HMM, overlays regimes on SPY price chart, prints transition matrix and per-regime emission statistics.
- **Rolling KS test**: quantifies distribution shift over time.
- Method comparison and regime agreement analysis.

### `04_model_monitoring.ipynb`
Walk-forward Ridge regression backtest → rolling RMSE and directional accuracy → PSI / KL / KS drift metrics → Pearson correlation between drift intensity and model error → regime-stratified performance → feature importance evolution over time → full monitoring dashboard.

---

## Dataset

| Property | Value |
|---|---|
| Asset | S&P 500 ETF (SPY) |
| Frequency | Daily |
| Period | Jan 2005 – Feb 2026 |
| Source | Yahoo Finance (public data via `yfinance`) |
| Observations | ~5,300 trading days |

---

## Key Results

- **Regime persistence**: The HMM transition matrix confirms that both low- and high-volatility regimes are highly persistent (P(stay) > 0.95), consistent with empirical findings in the regime-switching literature.
- **Crisis detection**: KS tests successfully identify structural breaks at known market events (GFC 2008, COVID 2020, 2022 rate-hike bear market).
- **Drift–performance link**: PSI and KS statistics are positively correlated with rolling RMSE — distribution drift predicts model performance degradation.
- **Regime-conditional performance**: The baseline model performs significantly worse in high-volatility regimes, confirming that crisis dynamics are not captured by a model trained on calm-market data.

---

## Installation

```bash
git clone https://github.com/<your-username>/drift_detection.git
cd drift_detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Then open the notebooks in order (01 → 04).

---

## Skills Demonstrated

- **Financial time series analysis**: returns, drawdown, rolling statistics, stationarity testing
- **Statistical ML**: Hidden Markov Models (Baum-Welch EM), Ridge regression
- **MLOps / model monitoring**: PSI, KS test, KL divergence, walk-forward backtesting, drift-performance correlation
- **Software engineering**: modular Python package, clean API design, docstrings, type hints, no global state
- **Quantitative research**: hypothesis formulation, empirical testing, interpretation of results in a finance context

---

*Built with Python · pandas · scikit-learn · hmmlearn · scipy · matplotlib · seaborn*
