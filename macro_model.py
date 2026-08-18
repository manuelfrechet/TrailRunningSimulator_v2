from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# V0 macro model
# -----------------------------------------------------------------------------
#
# Purpose:
#   Learn a coarse relationship between:
#
#       distance from start
#       cumulative ascent
#       cumulative descent
#
#   and:
#
#       cumulative elapsed time
#
# Model form:
#
#   standardized linear + quadratic + interaction terms
#
# This is deliberately simple for V0.
#
# The macro model is completely independent from the micro model.
# It does NOT modify, constrain, blend with, or correct the micro prediction.
# -----------------------------------------------------------------------------


@dataclass
class MacroModel:
    feature_names: list[str]
    means: pd.Series
    scales: pd.Series
    coefficients: np.ndarray
    intercept: float

    residual_median_s: float
    residual_q10_s: float
    residual_q90_s: float

    training_rows: int
    training_activities: int

    training_mae_s: float
    training_rmse_s: float
    training_r2: float

    min_distance_m: float
    max_distance_m: float
    min_cumulative_ascent_m: float
    max_cumulative_ascent_m: float
    min_cumulative_descent_m: float
    max_cumulative_descent_m: float

    def predict_cumulative_time(
        self,
        distance_from_start_m: Any,
        cumulative_ascent_m: Any,
        cumulative_descent_m: Any,
    ) -> np.ndarray:
        """
        Predict cumulative elapsed time from macro state variables.
        """

        frame = pd.DataFrame(
            {
                "distance_from_start_m": np.atleast_1d(
                    distance_from_start_m
                ),
                "cumulative_ascent_m": np.atleast_1d(
                    cumulative_ascent_m
                ),
                "cumulative_descent_m": np.atleast_1d(
                    cumulative_descent_m
                ),
            }
        )

        X = _build_design_matrix(
            frame=frame,
            means=self.means,
            scales=self.scales,
        )

        prediction = (
            self.intercept
            + X @ self.coefficients
        )

        return prediction

    def summary(self) -> dict[str, Any]:
        return {
            "training_rows": self.training_rows,
            "training_activities": self.training_activities,
            "training_mae_s": self.training_mae_s,
            "training_rmse_s": self.training_rmse_s,
            "training_r2": self.training_r2,
            "residual_median_s": self.residual_median_s,
            "residual_q10_s": self.residual_q10_s,
            "residual_q90_s": self.residual_q90_s,
            "min_distance_m": self.min_distance_m,
            "max_distance_m": self.max_distance_m,
            "min_cumulative_ascent_m": self.min_cumulative_ascent_m,
            "max_cumulative_ascent_m": self.max_cumulative_ascent_m,
            "min_cumulative_descent_m": self.min_cumulative_descent_m,
            "max_cumulative_descent_m": self.max_cumulative_descent_m,
            "feature_names": list(self.feature_names),
        }


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_FEATURES = [
    "distance_from_start_m",
    "cumulative_ascent_m",
    "cumulative_descent_m",
]

RIDGE_LAMBDA = 1e-4


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return one clean numeric Series.
    """
    if df is None or df.empty:
        return pd.Series(
            dtype="float64",
            index=df.index if df is not None else None,
        )

    obj = df.loc[:, column]

    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]

    return pd.to_numeric(
        obj,
        errors="coerce",
    )


def _build_standardized_base(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Standardize the three macro variables.
    """

    base = pd.DataFrame(
        {
            "distance_from_start_m": _safe_numeric_series(
                frame,
                "distance_from_start_m",
            ),
            "cumulative_ascent_m": _safe_numeric_series(
                frame,
                "cumulative_ascent_m",
            ),
            "cumulative_descent_m": _safe_numeric_series(
                frame,
                "cumulative_descent_m",
            ),
        }
    )

    means = base.mean(
        axis=0,
        skipna=True,
    )

    scales = base.std(
        axis=0,
        skipna=True,
    )

    scales = scales.replace(
        0.0,
        1.0,
    )

    scales = scales.fillna(1.0)

    standardized = (
        base.fillna(means)
        - means
    ) / scales

    return (
        standardized,
        means,
        scales,
    )


def _build_design_matrix(
    frame: pd.DataFrame,
    means: pd.Series,
    scales: pd.Series,
) -> np.ndarray:
    """
    Build V0 macro design matrix.

    Variables:
        d
        A+
        A-

    plus:

        d²
        A+²
        A-²
        d*A+
        d*A-
        A+*A-
    """

    base = pd.DataFrame(
        {
            "distance_from_start_m": _safe_numeric_series(
                frame,
                "distance_from_start_m",
            ),
            "cumulative_ascent_m": _safe_numeric_series(
                frame,
                "cumulative_ascent_m",
            ),
            "cumulative_descent_m": _safe_numeric_series(
                frame,
                "cumulative_descent_m",
            ),
        }
    )

    base = (
        base.fillna(means)
    )

    z = (
        base - means
    ) / scales

    d = z["distance_from_start_m"].to_numpy(
        dtype=float
    )

    a = z["cumulative_ascent_m"].to_numpy(
        dtype=float
    )

    de = z["cumulative_descent_m"].to_numpy(
        dtype=float
    )

    X = np.column_stack(
        [
            d,
            a,
            de,
            d * d,
            a * a,
            de * de,
            d * a,
            d * de,
            a * de,
        ]
    )

    return X


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
) -> tuple[float, np.ndarray]:
    """
    Ridge regression with an unpenalized intercept.
    """

    n_samples, n_features = X.shape

    X_augmented = np.column_stack(
        [
            np.ones(n_samples),
            X,
        ]
    )

    penalty = np.eye(
        n_features + 1
    )

    penalty[0, 0] = 0.0

    lhs = (
        X_augmented.T @ X_augmented
        + ridge_lambda * penalty
    )

    rhs = (
        X_augmented.T @ y
    )

    beta = np.linalg.solve(
        lhs,
        rhs,
    )

    return (
        float(beta[0]),
        beta[1:].astype(float),
    )


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """
    Return MAE, RMSE and R2.
    """

    residuals = (
        y_true
        - y_pred
    )

    mae = float(
        np.mean(
            np.abs(residuals)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    ss_res = float(
        np.sum(
            residuals ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                y_true
                - np.mean(y_true)
            ) ** 2
        )
    )

    if ss_tot > 0:
        r2 = (
            1.0
            - ss_res / ss_tot
        )
    else:
        r2 = np.nan

    return (
        mae,
        rmse,
        r2,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def fit_macro_model(
    learning_df: pd.DataFrame,
) -> MacroModel:
    """
    Fit the V0 macro model from the complete historical transition dataset.

    The macro model uses cumulative elapsed time at each historical position.
    """

    if learning_df is None or learning_df.empty:
        raise ValueError(
            "Historical learning dataset is empty."
        )

    required_columns = [
        "distance_from_start_m",
        "cumulative_ascent_m",
        "cumulative_descent_m",
        "elapsed_time_s",
    ]

    missing = [
        column
        for column in required_columns
        if column not in learning_df.columns
    ]

    if missing:
        raise ValueError(
            "Macro model is missing required columns: "
            + ", ".join(missing)
        )

    df = learning_df.copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required_columns
    ).reset_index(
        drop=True
    )

    if len(df) < 10:
        raise ValueError(
            "Not enough historical rows to fit the macro model."
        )

    # -------------------------------------------------------------------------
    # Standardization
    # -------------------------------------------------------------------------

    standardized, means, scales = (
        _build_standardized_base(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Design matrix
    # -------------------------------------------------------------------------

    X = _build_design_matrix(
        df,
        means,
        scales,
    )

    y = df[
        "elapsed_time_s"
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    intercept, coefficients = _fit_ridge(
        X,
        y,
        RIDGE_LAMBDA,
    )

    y_pred = (
        intercept
        + X @ coefficients
    )

    mae, rmse, r2 = _metrics(
        y,
        y_pred,
    )

    residuals = (
        y
        - y_pred
    )

    residual_median = float(
        np.median(
            residuals
        )
    )

    residual_q10 = float(
        np.quantile(
            residuals,
            0.10,
        )
    )

    residual_q90 = float(
        np.quantile(
            residuals,
            0.90,
        )
    )

    n_activities = (
        int(
            df["activity_id"].nunique()
        )
        if "activity_id" in df.columns
        else 0
    )

    feature_names = [
        "distance",
        "cumulative_ascent",
        "cumulative_descent",
        "distance_squared",
        "cumulative_ascent_squared",
        "cumulative_descent_squared",
        "distance_x_ascent",
        "distance_x_descent",
        "ascent_x_descent",
    ]

    return MacroModel(
        feature_names=feature_names,
        means=means,
        scales=scales,
        coefficients=coefficients,
        intercept=intercept,
        residual_median_s=residual_median,
        residual_q10_s=residual_q10,
        residual_q90_s=residual_q90,
        training_rows=int(
            len(df)
        ),
        training_activities=n_activities,
        training_mae_s=mae,
        training_rmse_s=rmse,
        training_r2=r2,
        min_distance_m=float(
            df["distance_from_start_m"].min()
        ),
        max_distance_m=float(
            df["distance_from_start_m"].max()
        ),
        min_cumulative_ascent_m=float(
            df["cumulative_ascent_m"].min()
        ),
        max_cumulative_ascent_m=float(
            df["cumulative_ascent_m"].max()
        ),
        min_cumulative_descent_m=float(
            df["cumulative_descent_m"].min()
        ),
        max_cumulative_descent_m=float(
            df["cumulative_descent_m"].max()
        ),
    )


def predict_macro_cumulative_time(
    model: MacroModel,
    distance_from_start_m: Any,
    cumulative_ascent_m: Any,
    cumulative_descent_m: Any,
) -> np.ndarray:
    """
    Predict cumulative elapsed time.
    """

    if model is None:
        raise ValueError(
            "Macro model is None."
        )

    prediction = model.predict_cumulative_time(
        distance_from_start_m=distance_from_start_m,
        cumulative_ascent_m=cumulative_ascent_m,
        cumulative_descent_m=cumulative_descent_m,
    )

    return prediction


def predict_macro_profile(
    model: MacroModel,
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add independent macro predictions to a normalized GPX profile.

    This function does NOT calculate micro predictions.
    """

    if model is None:
        raise ValueError(
            "Macro model is None."
        )

    if profile_df is None or profile_df.empty:
        return pd.DataFrame()

    required_columns = [
        "distance_from_start_m",
        "cumulative_ascent_m",
        "cumulative_descent_m",
    ]

    missing = [
        column
        for column in required_columns
        if column not in profile_df.columns
    ]

    if missing:
        raise ValueError(
            "GPX profile is missing required macro columns: "
            + ", ".join(missing)
        )

    result = profile_df.copy()

    predicted_cumulative = (
        model.predict_cumulative_time(
            result["distance_from_start_m"],
            result["cumulative_ascent_m"],
            result["cumulative_descent_m"],
        )
    )

    result["macro_predicted_cumulative_time_s"] = (
        predicted_cumulative
    )

    # -------------------------------------------------------------------------
    # Derive macro segment time from the cumulative macro curve.
    # -------------------------------------------------------------------------

    cumulative = result[
        "macro_predicted_cumulative_time_s"
    ].to_numpy(
        dtype=float
    )

    segment_times = np.diff(
        cumulative,
        prepend=0.0,
    )

    # If the profile has an explicit starting row at 0 m, its prediction should
    # represent zero elapsed time rather than the first positive segment.
    if len(segment_times) > 0:
        segment_times[0] = cumulative[0]

    result["macro_predicted_time_s"] = (
        segment_times
    )

    return result
  
