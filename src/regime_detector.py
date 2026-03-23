"""
regime_detector.py
==================
Regime detection for financial time series.

Implements one approach to identify distinct market regimes (e.g. bull / bear / high-volatility crisis):

1. Hidden Markov Model (HMM)
   A probabilistic graphical model with latent states.  At each time step the market is assumed to be in one of *n* hidden regimes.
   Each with its own Gaussian emission distribution over returns.  The model learns:
   - Transition probabilities A_{ij} = P(regime_t = j | regime_{t-1} = i)
   - Emission parameters (mu_i, sigma_i) per regime
   - Initial state distribution pi
   
   After fitting, states are sorted by volatility so that regime 0 is
   always the lowest-volatility (calm) state.

"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn(
        "hmmlearn is not installed.  HMMRegimeDetector will not be available.\n"
        "Install it with: pip install hmmlearn",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Regime label mapping  (index to readable name)
# ---------------------------------------------------------------------------

REGIME_NAMES: dict[int, str] = {
    0: "low_volatility",    # calm / bull market
    1: "high_volatility",   # stress / bear market
    2: "transition",        # intermediate (used with 3-state HMM)
}


# ---------------------------------------------------------------------------
# Hidden Markov Model
# ---------------------------------------------------------------------------


class HMMRegimeDetector:
    """Gaussian Hidden Markov Model for regime detection.

    Fits a GaussianHMM to a return series and assigns each trading day
    to a discrete latent regime.  After fitting, regimes are re-labelled
    in ascending order of their emission volatility, so regime 0 is
    always the quietest state.

    Parameters
    ----------
    n_regimes:
        Number of hidden states.  Default 2 (low / high volatility).
    n_iter:
        Maximum EM iterations (convergence typically < 200).
    covariance_type:
        Emission covariance structure.  ``"full"`` is the standard choice
        for univariate series; ``"diag"`` is equivalent for 1-D.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        n_iter: int = 1000,
        covariance_type: str = "full",
        random_state: int = 1,
    ) -> None:
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required.  Install with: pip install hmmlearn"
            )
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self._model: "GaussianHMM | None" = None
        # Maps sorted label index to original HMM state index
        self._state_order: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series | np.ndarray) -> "HMMRegimeDetector":
        """Fit the HMM to a return series via Baum-Welch EM.

        Parameters
        ----------
        returns:
            Daily log return series.

        Returns
        -------
        self
        """
        X = np.asarray(returns).reshape(-1, 1)

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(X)

        # Sort states by emission std (ascending) to fix label-switching.
        stds = np.array(
            [np.sqrt(model.covars_[i][0, 0]) for i in range(self.n_regimes)]
        )
        self._state_order = np.argsort(stds)   # state_order[0] = quietest state
        self._model = model
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, returns: pd.Series | np.ndarray) -> np.ndarray:
        """Predict the most likely regime sequence (Viterbi algorithm).

        Parameters
        ----------
        returns:
            Daily log return series.

        Returns
        -------
        np.ndarray of int
            Regime labels (0 = low-vol, 1 = high-vol, …).
        """
        X = np.asarray(returns).reshape(-1, 1)
        raw_states = self._model.predict(X)
        # Re-map to sorted labels
        inv_order = np.argsort(self._state_order)
        return inv_order[raw_states]

    def predict_proba(self, returns: pd.Series | np.ndarray) -> np.ndarray:
        """Return smoothed posterior state probabilities.

        Parameters
        ----------
        returns:
            Daily log return series.

        Returns
        -------
        np.ndarray, shape (T, n_regimes)
            Posterior probability of being in each regime at each step.
        """
        X = np.asarray(returns).reshape(-1, 1)
        _, posterior = self._model.score_samples(X)
        return posterior[:, self._state_order]

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Fit the model and return regime labels as a named pd.Series.

        Parameters
        ----------
        returns:
            Daily log return series (pd.Series with DatetimeIndex).

        Returns
        -------
        pd.Series
            Integer regime labels, same index as ``returns``.
        """
        self.fit(returns)
        labels = self.predict(returns)
        return pd.Series(labels, index=returns.index, name="hmm_regime")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def transition_matrix(self) -> pd.DataFrame:
        """Regime transition probability matrix A.

        A_{ij} = P(regime_{t} = j | regime_{t-1} = i)

        Rows sum to 1.  High diagonal values -> persistent regimes.
        """
        if self._model is None:
            raise RuntimeError("Call .fit() before accessing properties.")
        A = self._model.transmat_[np.ix_(self._state_order, self._state_order)]
        names = [
            REGIME_NAMES.get(i, f"regime_{i}") for i in range(self.n_regimes)
        ]
        return pd.DataFrame(A, index=names, columns=names)

    @property
    def emission_stats(self) -> pd.DataFrame:
        """Mean and volatility of each regime's emission distribution."""
        if self._model is None:
            raise RuntimeError("Call .fit() before accessing properties.")
        rows = []
        for i, orig_i in enumerate(self._state_order):
            mu = self._model.means_[orig_i, 0]
            sigma = np.sqrt(self._model.covars_[orig_i][0, 0])
            rows.append(
                {
                    "regime": REGIME_NAMES.get(i, f"regime_{i}"),
                    "mean_daily_return": round(mu, 6),
                    "daily_vol": round(sigma, 6),
                    "annualised_vol_%": round(sigma * np.sqrt(252) * 100, 2),
                    "annualised_return_%": round(mu * 252 * 100, 2),
                }
            )
        return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Regime Statistics
# ---------------------------------------------------------------------------

def regime_statistics(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Compute descriptive statistics for each detected regime.

    Parameters
    ----------
    returns:
        Daily log return series.
    regimes:
        Integer regime labels with the same (or overlapping) DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Per-regime statistics: mean return, volatility, Sharpe ratio,
        skewness, excess kurtosis, and percentage of total time.
    """
    aligned = pd.concat([returns, regimes], axis=1).dropna()
    ret_col = aligned.columns[0]
    reg_col = aligned.columns[1]

    rows = []
    for regime_id, group in aligned.groupby(reg_col):
        r = group[ret_col]
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        pct_time = 100.0 * len(r) / len(aligned)

        rows.append(
            {
                "regime": REGIME_NAMES.get(int(regime_id), f"regime_{regime_id}"),
                "n_days": len(r),
                "pct_time_%": round(pct_time, 1),
                "ann_return_%": round(ann_ret * 100, 2),
                "ann_vol_%": round(ann_vol * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "skewness": round(r.skew(), 3),
                "excess_kurtosis": round(r.kurt(), 3),
            }
        )

    return pd.DataFrame(rows).set_index("regime")
