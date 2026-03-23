"""
model.py
========
Baseline predictive model for next-day SPY returns.

Architecture
------------
The model is a Ridge regression with temporal walk-forward cross-validation.  
The choice is intentional: the project goal is NOT to maximise prediction accuracy but to provide a stable and interpretable
baseline whose performance degradation under regime transitions can be measured reliably.

Walk-forward validation
-----------------------
At each step:
  1. Train a Ridge model on a fixed-size ``train_window`` of history.
  2. Evaluate on the next ``test_window`` of out-of-sample observations.
  3. Advance by ``step`` days and repeat.

This strictly respects temporal ordering and prevents any look-ahead bias which is critical in financial machine learning.
Because data leakage silently inflates apparent performance metrics.

Output
------
``walk_forward_backtest`` returns a list of ``PredictionRecord`` objects that store both raw predictions and computed metrics for each fold.
``predictions_to_dataframe`` then aggregates these into a DataFrame for analysis and visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Prediction Record (one per CV fold)
# ---------------------------------------------------------------------------


@dataclass
class PredictionRecord:
    """Container for a single walk-forward fold's predictions and metadata.

    Attributes
    ----------
    train_start, train_end : pd.Timestamp
        First and last dates in the training window.
    test_start, test_end : pd.Timestamp
        First and last dates in the test window.
    y_true, y_pred : np.ndarray
        Actual and predicted returns for the test window.
    feature_names : list of str
        Names of the features used in this fold.
    coefficients : np.ndarray
        Fitted Ridge coefficients (useful for tracking feature importance over time and detecting regime-driven weight shifts).
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    y_true: np.ndarray
    y_pred: np.ndarray
    feature_names: list = field(default_factory=list)
    coefficients: np.ndarray = field(default_factory=lambda: np.array([]))

    # ------------------------------------------------------------------
    # Computed metrics (properties, no storage cost)
    # ------------------------------------------------------------------

    @property
    def rmse(self) -> float:
        """Root Mean Square Error."""
        return float(np.sqrt(mean_squared_error(self.y_true, self.y_pred)))

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        return float(mean_absolute_error(self.y_true, self.y_pred))

    @property
    def r2(self) -> float:
        """Coefficient of determination R^2."""
        return float(r2_score(self.y_true, self.y_pred))

    @property
    def directional_accuracy(self) -> float:
        """Fraction of predictions with the correct directional sign.

        A value > 0.5 means the model has *some* directional edge;
        < 0.5 means it is worse than a coin-flip on direction.
        """
        correct = np.sign(self.y_true) == np.sign(self.y_pred)
        return float(correct.mean())


# ---------------------------------------------------------------------------
# Walk-Forward Cross-Validation
# ---------------------------------------------------------------------------

def walk_forward_backtest(features: pd.DataFrame, target: pd.Series, train_window: int = 252, test_window: int = 63,
    step: int = 21, alpha: float = 1.0) -> list[PredictionRecord]:
    """Walk-forward backtesting with Ridge regression.

    At each fold:
    1. Extract a ``train_window`` training slice.
    2. Standardise features (fit scaler on train only).
    3. Fit Ridge(alpha=``alpha``) on the standardised training set.
    4. Predict on the next ``test_window`` days.
    5. Advance start by ``step`` days.

    Parameters
    ----------
    features:
        Feature matrix (rows = trading days, columns = engineered features).
        NaN rows are removed after alignment with the target.
    target:
        Next-day log return (the label).  Aligned with features on index.
    train_window:
        Training set size in trading days.  Default 252 = 1 year.
    test_window:
        Out-of-sample evaluation window per fold.  Default 63 ≈ 1 quarter.
    step:
        How many days to advance between folds.  Default 21 ≈ 1 month.
        When step < test_window the windows overlap. In that case the last fold's prediction is used per day in ``get_full_prediction_series``.
    alpha:
        Ridge regularisation strength.  Higher alpha → more shrinkage.

    Returns
    -------
    list of PredictionRecord
        One record per complete fold.
    """
    # Align features and target. Drop any rows with NaN.
    aligned = pd.concat(
        [features, target.rename("__target__")], axis=1
    ).dropna()

    X_all = aligned.drop(columns="__target__").values
    y_all = aligned["__target__"].values
    idx = aligned.index
    feat_names = list(aligned.drop(columns="__target__").columns)

    records: list[PredictionRecord] = []
    start = 0

    while start + train_window + test_window <= len(aligned):
        # 1. Slice
        tr = slice(start, start + train_window)
        te = slice(start + train_window, start + train_window + test_window)

        X_train, y_train = X_all[tr], y_all[tr]
        X_test, y_test = X_all[te], y_all[te]

        # 2. Standardise (scaler fit on train only)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # 3. Train
        model = Ridge(alpha=alpha)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

        records.append(
            PredictionRecord(
                train_start=idx[start],
                train_end=idx[start + train_window - 1],
                test_start=idx[start + train_window],
                test_end=idx[start + train_window + test_window - 1],
                y_true=y_test,
                y_pred=y_pred,
                feature_names=feat_names,
                coefficients=model.coef_.copy(),
            )
        )

        start += step

    return records


# ---------------------------------------------------------------------------
# Aggregation Helpers
# ---------------------------------------------------------------------------

def predictions_to_dataframe(records: list[PredictionRecord]) -> pd.DataFrame:
    """Summarise walk-forward results into a tidy per-fold DataFrame.

    Parameters
    ----------
    records:
        Output of ``walk_forward_backtest``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``test_start``, with columns:
        rmse, mae, r2, directional_accuracy, n_test.
    """
    rows = []
    for r in records:
        rows.append(
            {
                "test_start": r.test_start,
                "test_end": r.test_end,
                "rmse": r.rmse,
                "mae": r.mae,
                "r2": r.r2,
                "directional_accuracy": r.directional_accuracy,
                "n_test": len(r.y_true),
            }
        )
    return pd.DataFrame(rows).set_index("test_start")


def get_full_prediction_series(records: list[PredictionRecord], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reconstruct a continuous prediction series from walk-forward records.

    When ``step < test_window``, adjacent folds overlap.  In that case the most recent prediction for each date is used (last-write wins),
    which is the prediction made with the most up-to-date training data.

    Parameters
    ----------
    records:
        Output of ``walk_forward_backtest``.
    index:
        Full DatetimeIndex of the aligned feature/target dataset (used to
        locate exact date positions for each fold's test window).

    Returns
    -------
    pd.DataFrame
        Columns ``y_true`` and ``y_pred``, indexed by date.
    """
    rows = []
    for r in records:
        n = len(r.y_true)
        # Find the start position of the test window in the full index.
        start_pos = index.get_loc(r.test_start)
        dates = index[start_pos : start_pos + n]
        for d, yt, yp in zip(dates, r.y_true, r.y_pred):
            rows.append({"date": d, "y_true": yt, "y_pred": yp})

    df = pd.DataFrame(rows).set_index("date")
    # If overlapping, keep the latest (most-recent-data) prediction per day.
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


def feature_importance_over_time(records: list[PredictionRecord]) -> pd.DataFrame:
    """Track how Ridge coefficients evolve across walk-forward folds.

    Regime transitions often manifest as abrupt changes in which features are most predictive.  
    Plotting these coefficient paths is a useful diagnostic for understanding model behaviour under distribution shift.

    Parameters
    ----------
    records:
        Output of ``walk_forward_backtest``.

    Returns
    -------
    pd.DataFrame
        Rows = folds (indexed by test_start), columns = feature names,
        values = Ridge coefficient for that fold.
    """
    if not records or len(records[0].coefficients) == 0:
        raise ValueError("Records do not contain coefficients.")

    data = {
        r.test_start: dict(zip(r.feature_names, r.coefficients))
        for r in records
    }
    df = pd.DataFrame(data).T
    df.index.name = "test_start"
    return df
