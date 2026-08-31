from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from config import GPX_SEGMENT_LENGTH_M


# =============================================================================
# Macro Model 2
# =============================================================================
#
# Purpose:
#
#     Test a monotonic version of the current V0 macro regression.
#
# Model 1 remains completely untouched in macro_model.py.
#
# Model 2 uses the SAME:
#
#     - three variables
#     - nine polynomial / interaction basis terms
#     - no intercept
#     - historical target
#
# The ONLY methodological change is:
#
#     beta_i >= 0
#
# for every coefficient.
#
# Basis:
#
#     d
#     A+
#     A-
#     d^2
#     A+^2
#     A-^2
#     d*A+
#     d*A-
#     A+*A-
#
# Since d, A+ and A- are non-negative and non-decreasing along the course,
# every basis term is also non-negative and non-decreasing.
#
# Therefore a coefficient vector with:
#
#     beta_i >= 0
#
# guarantees a monotonic cumulative prediction with respect to the three
# cumulative course variables.
#
# The model is still anchored at:
#
#     M(0,0,0) = 0
#
# because there is no intercept and every basis term is zero at the origin.
#
# -----------------------------------------------------------------------------
# Important:
#
# This module is a MODEL 2 EXPERIMENT.
#
# It must not replace Model 1 automatically.
# Both models should be fitted on the same historical FIT corpus and compared
# before any decision is made.
# =============================================================================


RIDGE_LAMBDA = 1e-4

OPTIMIZATION_TOLERANCE = 1e-10

MAX_OPTIMIZATION_ITERATIONS = 5000


# =============================================================================
# Feature definition
# =============================================================================

FEATURE_NAMES = [
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


BASE_VARIABLES = [
    "distance_from_start_m",
    "cumulative_ascent_m",
    "cumulative_descent_m",
]


# =============================================================================
# Data container
# =============================================================================

@dataclass
class MacroModel2:

    feature_names: list[str]

    scales: pd.Series

    coefficients: np.ndarray

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

    optimization_success: bool
    optimization_status: int
    optimization_message: str

    def predict_cumulative_time(
        self,
        distance_from_start_m: Any,
        cumulative_ascent_m: Any,
        cumulative_descent_m: Any,
    ) -> np.ndarray:
        """
        Predict cumulative elapsed time.

        Because the coefficients are constrained to be non-negative and the
        basis is monotonic in the cumulative course variables, this prediction
        is monotonic in those variables.
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
            scales=self.scales,
        )

        prediction = (
            X
            @ self.coefficients
        )

        return np.asarray(
            prediction,
            dtype=float,
        )

    def summary(
        self,
    ) -> dict[str, Any]:

        return {
            "model": "Macro Model 2 - non-negative constrained regression",

            "training_rows": (
                self.training_rows
            ),

            "training_activities": (
                self.training_activities
            ),

            "training_mae_s": (
                self.training_mae_s
            ),

            "training_rmse_s": (
                self.training_rmse_s
            ),

            "training_r2": (
                self.training_r2
            ),

            "residual_median_s": (
                self.residual_median_s
            ),

            "residual_q10_s": (
                self.residual_q10_s
            ),

            "residual_q90_s": (
                self.residual_q90_s
            ),

            "min_distance_m": (
                self.min_distance_m
            ),

            "max_distance_m": (
                self.max_distance_m
            ),

            "min_cumulative_ascent_m": (
                self.min_cumulative_ascent_m
            ),

            "max_cumulative_ascent_m": (
                self.max_cumulative_ascent_m
            ),

            "min_cumulative_descent_m": (
                self.min_cumulative_descent_m
            ),

            "max_cumulative_descent_m": (
                self.max_cumulative_descent_m
            ),

            "feature_names": list(
                self.feature_names
            ),

            "gpx_segment_length_m": (
                GPX_SEGMENT_LENGTH_M
            ),

            "all_coefficients_non_negative": bool(
                np.all(
                    self.coefficients
                    >= -OPTIMIZATION_TOLERANCE
                )
            ),

            "optimization_success": (
                self.optimization_success
            ),

            "optimization_status": (
                self.optimization_status
            ),

            "optimization_message": (
                self.optimization_message
            ),
        }


# =============================================================================
# Helpers
# =============================================================================

def _safe_numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return one clean numeric Series.
    """

    obj = df.loc[
        :,
        column,
    ]

    if isinstance(
        obj,
        pd.DataFrame,
    ):
        obj = obj.iloc[
            :,
            0,
        ]

    return pd.to_numeric(
        obj,
        errors="coerce",
    )


def _prepare_macro_variables(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the three macro variables.
    """

    return pd.DataFrame(
        {
            "distance_from_start_m": (
                _safe_numeric_series(
                    frame,
                    "distance_from_start_m",
                )
            ),

            "cumulative_ascent_m": (
                _safe_numeric_series(
                    frame,
                    "cumulative_ascent_m",
                )
            ),

            "cumulative_descent_m": (
                _safe_numeric_series(
                    frame,
                    "cumulative_descent_m",
                )
            ),
        }
    )


def _calculate_scales(
    variables: pd.DataFrame,
) -> pd.Series:
    """
    Calculate numerical scales.

    Variables are NOT centered.

    This preserves the physical origin:

        d = 0
        A+ = 0
        A- = 0
    """

    scales = variables.std(
        axis=0,
        skipna=True,
    )

    scales = scales.replace(
        0.0,
        1.0,
    )

    scales = scales.fillna(
        1.0
    )

    scales = scales.clip(
        lower=1e-9,
    )

    return scales


def _build_design_matrix(
    frame: pd.DataFrame,
    scales: pd.Series,
) -> np.ndarray:
    """
    Build the same nine-term basis as Macro Model 1.

    Variables are standardized by scale only.

    No centering is applied.
    """

    variables = (
        _prepare_macro_variables(
            frame
        )
    )

    variables = (
        variables.fillna(
            0.0
        )
    )

    d = (
        variables[
            "distance_from_start_m"
        ]
        / float(
            scales[
                "distance_from_start_m"
            ]
        )
    ).to_numpy(
        dtype=float
    )

    ascent = (
        variables[
            "cumulative_ascent_m"
        ]
        / float(
            scales[
                "cumulative_ascent_m"
            ]
        )
    ).to_numpy(
        dtype=float
    )

    descent = (
        variables[
            "cumulative_descent_m"
        ]
        / float(
            scales[
                "cumulative_descent_m"
            ]
        )
    ).to_numpy(
        dtype=float
    )

    return np.column_stack(
        [
            d,
            ascent,
            descent,
            d * d,
            ascent * ascent,
            descent * descent,
            d * ascent,
            d * descent,
            ascent * descent,
        ]
    )


def _fit_nonnegative_ridge(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
) -> tuple[
    np.ndarray,
    bool,
    int,
    str,
]:
    """
    Solve non-negative ridge regression.

    Objective:

        ||X beta - y||²
        +
        lambda ||beta||²

    subject to:

        beta >= 0

    We implement ridge by augmenting the least-squares system:

        X_aug =
            [ X                  ]
            [ sqrt(lambda) * I  ]

        y_aug =
            [ y ]
            [ 0 ]

    and solve the bounded least-squares problem with:

        0 <= beta_i < infinity
    """

    if X.ndim != 2:
        raise ValueError(
            "Macro Model 2 design matrix must be 2-dimensional."
        )

    if y.ndim != 1:
        raise ValueError(
            "Macro Model 2 target vector must be 1-dimensional."
        )

    if len(X) != len(y):
        raise ValueError(
            "Macro Model 2 design matrix and target vector "
            "have incompatible lengths."
        )

    if len(y) == 0:
        raise ValueError(
            "Macro Model 2 target vector is empty."
        )

    n_features = X.shape[1]

    sqrt_lambda = float(
        np.sqrt(
            max(
                0.0,
                ridge_lambda,
            )
        )
    )

    regularization = (
        sqrt_lambda
        * np.eye(
            n_features
        )
    )

    X_augmented = np.vstack(
        [
            X,
            regularization,
        ]
    )

    y_augmented = np.concatenate(
        [
            y,
            np.zeros(
                n_features,
                dtype=float,
            ),
        ]
    )

    lower_bounds = np.zeros(
        n_features,
        dtype=float,
    )

    upper_bounds = np.full(
        n_features,
        np.inf,
        dtype=float,
    )

    result = lsq_linear(
        X_augmented,
        y_augmented,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        method="trf",
        tol=OPTIMIZATION_TOLERANCE,
        lsmr_tol=OPTIMIZATION_TOLERANCE,
        max_iter=MAX_OPTIMIZATION_ITERATIONS,
    )

    coefficients = np.asarray(
        result.x,
        dtype=float,
    )

    # Numerical optimizer noise can produce tiny negative values around
    # zero. Those are mathematically zero for our purposes.
    coefficients[
        np.abs(
            coefficients
        )
        <= OPTIMIZATION_TOLERANCE
    ] = 0.0

    return (
        coefficients,
        bool(
            result.success
        ),
        int(
            result.status
        ),
        str(
            result.message
        ),
    )


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Calculate MAE, RMSE and R².
    """

    residuals = (
        y_true
        - y_pred
    )

    mae = float(
        np.mean(
            np.abs(
                residuals
            )
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
                - np.mean(
                    y_true
                )
            ) ** 2
        )
    )

    if ss_tot > 0.0:

        r2 = float(
            1.0
            - ss_res
            / ss_tot
        )

    else:

        r2 = np.nan

    return (
        mae,
        rmse,
        r2,
    )


# =============================================================================
# Fit Model 2
# =============================================================================

def fit_macro_model2(
    learning_df: pd.DataFrame,
) -> MacroModel2:
    """
    Fit Macro Model 2 using the complete historical learning corpus.

    Target:

        elapsed_time_s

    Inputs:

        distance_from_start_m
        cumulative_ascent_m
        cumulative_descent_m

    Mathematical constraint:

        beta_i >= 0

    Physical origin:

        M(0,0,0) = 0
    """

    if (
        learning_df is None
        or learning_df.empty
    ):

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
            "Macro Model 2 is missing required columns: "
            + ", ".join(
                missing
            )
        )

    df = learning_df.copy()

    # -------------------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------------------

    for column in required_columns:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=required_columns
        )
        .reset_index(
            drop=True
        )
    )

    if len(df) < 10:

        raise ValueError(
            "Not enough historical rows to fit Macro Model 2."
        )

    # -------------------------------------------------------------------------
    # Historical variables
    # -------------------------------------------------------------------------

    variables = (
        _prepare_macro_variables(
            df
        )
    )

    scales = (
        _calculate_scales(
            variables
        )
    )

    # -------------------------------------------------------------------------
    # Design matrix
    # -------------------------------------------------------------------------

    X = _build_design_matrix(
        df,
        scales,
    )

    y = df[
        "elapsed_time_s"
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------------------
    # Constrained fit
    # -------------------------------------------------------------------------

    (
        coefficients,
        optimization_success,
        optimization_status,
        optimization_message,
    ) = _fit_nonnegative_ridge(
        X=X,
        y=y,
        ridge_lambda=RIDGE_LAMBDA,
    )

    # -------------------------------------------------------------------------
    # Training predictions
    # -------------------------------------------------------------------------

    y_pred = (
        X
        @ coefficients
    )

    (
        mae,
        rmse,
        r2,
    ) = _calculate_metrics(
        y_true=y,
        y_pred=y_pred,
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

    # -------------------------------------------------------------------------
    # Number of historical activities
    # -------------------------------------------------------------------------

    if (
        "activity_id"
        in df.columns
    ):

        training_activities = int(
            df[
                "activity_id"
            ].nunique()
        )

    else:

        training_activities = 0

    # -------------------------------------------------------------------------
    # Basic coefficient validation.
    # -------------------------------------------------------------------------

    if np.any(
        coefficients
        < -OPTIMIZATION_TOLERANCE
    ):

        raise RuntimeError(
            "Macro Model 2 optimization returned a materially negative "
            "coefficient despite the non-negative constraint."
        )

    return MacroModel2(
        feature_names=list(
            FEATURE_NAMES
        ),

        scales=(
            scales
        ),

        coefficients=(
            coefficients
        ),

        residual_median_s=(
            residual_median
        ),

        residual_q10_s=(
            residual_q10
        ),

        residual_q90_s=(
            residual_q90
        ),

        training_rows=int(
            len(
                df
            )
        ),

        training_activities=(
            training_activities
        ),

        training_mae_s=(
            mae
        ),

        training_rmse_s=(
            rmse
        ),

        training_r2=(
            r2
        ),

        min_distance_m=float(
            df[
                "distance_from_start_m"
            ].min()
        ),

        max_distance_m=float(
            df[
                "distance_from_start_m"
            ].max()
        ),

        min_cumulative_ascent_m=float(
            df[
                "cumulative_ascent_m"
            ].min()
        ),

        max_cumulative_ascent_m=float(
            df[
                "cumulative_ascent_m"
            ].max()
        ),

        min_cumulative_descent_m=float(
            df[
                "cumulative_descent_m"
            ].min()
        ),

        max_cumulative_descent_m=float(
            df[
                "cumulative_descent_m"
            ].max()
        ),

        optimization_success=(
            optimization_success
        ),

        optimization_status=(
            optimization_status
        ),

        optimization_message=(
            optimization_message
        ),
    )


# =============================================================================
# Public prediction helper
# =============================================================================

def predict_macro_model2_cumulative_time(
    model: MacroModel2,
    distance_from_start_m: Any,
    cumulative_ascent_m: Any,
    cumulative_descent_m: Any,
) -> np.ndarray:
    """
    Predict cumulative elapsed time with Macro Model 2.
    """

    if model is None:

        raise ValueError(
            "Macro Model 2 is None."
        )

    return model.predict_cumulative_time(
        distance_from_start_m=(
            distance_from_start_m
        ),
        cumulative_ascent_m=(
            cumulative_ascent_m
        ),
        cumulative_descent_m=(
            cumulative_descent_m
        ),
    )


# =============================================================================
# Prediction of a full normalized profile
# =============================================================================

def predict_macro_model2_profile(
    model: MacroModel2,
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict Macro Model 2 cumulative and segment times for a normalized GPX
    profile.

    This is primarily intended for diagnostics and direct comparison against
    Macro Model 1.

    The function does NOT alter the profile and does NOT clip negative segment
    times.

    Under the mathematical constraints they should not occur except for
    negligible floating-point noise.
    """

    if model is None:

        raise ValueError(
            "Macro Model 2 is None."
        )

    if (
        profile_df is None
        or profile_df.empty
    ):

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
            "Profile is missing required Macro Model 2 columns: "
            + ", ".join(
                missing
            )
        )

    result = (
        profile_df.copy()
    )

    cumulative_prediction = (
        model.predict_cumulative_time(
            distance_from_start_m=(
                result[
                    "distance_from_start_m"
                ]
            ),
            cumulative_ascent_m=(
                result[
                    "cumulative_ascent_m"
                ]
            ),
            cumulative_descent_m=(
                result[
                    "cumulative_descent_m"
                ]
            ),
        )
    )

    result[
        "macro_model2_predicted_cumulative_time_s"
    ] = cumulative_prediction

    cumulative = result[
        "macro_model2_predicted_cumulative_time_s"
    ].to_numpy(
        dtype=float
    )

    result[
        "macro_model2_predicted_time_s"
    ] = np.diff(
        cumulative,
        prepend=0.0,
    )

    # -------------------------------------------------------------------------
    # Tiny negative numerical values can occur at machine precision.
    #
    # We do NOT replace meaningful negatives.
    # -------------------------------------------------------------------------

    tiny_negative = (
        (
            result[
                "macro_model2_predicted_time_s"
            ]
            < 0.0
        )
        & (
            np.abs(
                result[
                    "macro_model2_predicted_time_s"
                ]
            )
            <= OPTIMIZATION_TOLERANCE
        )
    )

    result.loc[
        tiny_negative,
        "macro_model2_predicted_time_s",
    ] = 0.0

    result[
        "macro_model2_negative_increment"
    ] = (
        result[
            "macro_model2_predicted_time_s"
        ]
        < 0.0
    )

    return result


# =============================================================================
# Compare Model 2 cumulative monotonicity
# =============================================================================

def check_macro_model2_monotonicity(
    model: MacroModel2,
    profile_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Check whether Macro Model 2 produces any negative segment increments on
    a supplied normalized trajectory.
    """

    predicted = (
        predict_macro_model2_profile(
            model,
            profile_df,
        )
    )

    if predicted.empty:

        return {
            "segments": 0,
            "negative_segments": 0,
            "minimum_segment_time_s": np.nan,
            "total_negative_time_s": 0.0,
        }

    segment_times = predicted[
        "macro_model2_predicted_time_s"
    ].to_numpy(
        dtype=float
    )

    negative = (
        segment_times
        < 0.0
    )

    negative_values = (
        segment_times[
            negative
        ]
    )

    return {
        "segments": int(
            len(
                segment_times
            )
        ),

        "negative_segments": int(
            negative.sum()
        ),

        "minimum_segment_time_s": float(
            np.min(
                segment_times
            )
        ),

        "total_negative_time_s": float(
            np.abs(
                negative_values
            ).sum()
        ),
    }
