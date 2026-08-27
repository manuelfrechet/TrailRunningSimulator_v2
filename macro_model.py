from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import GPX_SEGMENT_LENGTH_M


# =============================================================================
# V0 Macro model
# =============================================================================
#
# Learn:
#
#     cumulative_elapsed_time
#     =
#     M(
#         distance_from_start,
#         cumulative_ascent,
#         cumulative_descent
#     )
#
# V0 basis:
#
#     linear terms
#     quadratic terms
#     pairwise interaction terms
#
# Physical constraint:
#
#     M(0, 0, 0) = 0
#
# No free intercept.
#
# Macro and micro remain independent.
# =============================================================================


RIDGE_LAMBDA = 1e-4


# =============================================================================
# Data container
# =============================================================================

@dataclass
class MacroModel:

    feature_names: list[str]

    # Scaling only.
    # Variables are NOT centered because the physical origin must remain zero.
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

    def predict_cumulative_time(
        self,
        distance_from_start_m: Any,
        cumulative_ascent_m: Any,
        cumulative_descent_m: Any,
    ) -> np.ndarray:
        """
        Predict cumulative elapsed time.

        The fitted function satisfies:

            M(0, 0, 0) = 0
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

    Duplicate-column cases are handled defensively.
    """

    if (
        df is None
        or df.empty
    ):

        return pd.Series(
            dtype="float64",
            index=(
                df.index
                if df is not None
                else None
            ),
        )

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
    Extract the three canonical macro variables.
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

    Variables are deliberately NOT centered.

    Therefore the physical origin remains:

        distance = 0
        ascent   = 0
        descent  = 0
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
    Build the anchored V0 macro basis.

    Variables:

        d
        A+
        A-

    Terms:

        d
        A+
        A-
        d²
        A+²
        A-²
        d*A+
        d*A-
        A+*A-

    Every term is zero at the race origin.
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


def _fit_ridge_without_intercept(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    """
    Ridge regression without an intercept.

    This structurally enforces:

        M(0,0,0) = 0
    """

    if X.ndim != 2:
        raise ValueError(
            "Macro design matrix must be 2-dimensional."
        )

    if y.ndim != 1:
        raise ValueError(
            "Macro target vector must be 1-dimensional."
        )

    if len(y) == 0:
        raise ValueError(
            "Macro target vector is empty."
        )

    n_features = (
        X.shape[1]
    )

    penalty = np.eye(
        n_features
    )

    lhs = (
        X.T
        @ X
        + ridge_lambda
        * penalty
    )

    rhs = (
        X.T
        @ y
    )

    coefficients = np.linalg.solve(
        lhs,
        rhs,
    )

    return coefficients.astype(
        float
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


def _validate_profile_spacing(
    profile_df: pd.DataFrame,
) -> None:
    """
    Verify that a prediction profile follows the configured GPX segment
    spacing.

    This does NOT affect the macro mathematical model.

    It exists only to prevent an accidental mismatch such as:
        configuration = 100 m
        supplied profile = 50 m rows
    """

    if (
        profile_df is None
        or profile_df.empty
    ):
        return

    distance = pd.to_numeric(
        profile_df[
            "distance_from_start_m"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if len(distance) == 0:
        return

    if not np.isfinite(
        distance
    ).all():

        raise ValueError(
            "Profile contains invalid distance values."
        )

    # Rows represent segment endpoints.
    # The first endpoint should therefore equal one configured segment.
    expected_first = float(
        GPX_SEGMENT_LENGTH_M
    )

    if not np.isclose(
        distance[0],
        expected_first,
        atol=1e-6,
        rtol=0.0,
    ):

        raise ValueError(
            "Profile spacing does not match config.py. "
            f"Expected first endpoint at "
            f"{GPX_SEGMENT_LENGTH_M:.2f} m, "
            f"found {distance[0]:.2f} m."
        )

    if len(distance) >= 2:

        spacing = np.diff(
            distance
        )

        if not np.allclose(
            spacing,
            GPX_SEGMENT_LENGTH_M,
            atol=1e-6,
            rtol=0.0,
        ):

            raise ValueError(
                "Profile segment spacing does not match "
                "GPX_SEGMENT_LENGTH_M from config.py."
            )


# =============================================================================
# Fit macro model
# =============================================================================

def fit_macro_model(
    learning_df: pd.DataFrame,
) -> MacroModel:
    """
    Fit the V0 macro model using the complete historical learning corpus.

    Target:
        elapsed_time_s

    Inputs:
        distance_from_start_m
        cumulative_ascent_m
        cumulative_descent_m

    Constraint:
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
            "Macro model is missing required columns: "
            + ", ".join(
                missing
            )
        )

    df = learning_df.copy()

    # -------------------------------------------------------------------------
    # Required numeric variables
    # -------------------------------------------------------------------------

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
    # Variables and numerical scales
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
    # Fit
    # -------------------------------------------------------------------------

    coefficients = (
        _fit_ridge_without_intercept(
            X=X,
            y=y,
            ridge_lambda=(
                RIDGE_LAMBDA
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Training diagnostics
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

    if "activity_id" in df.columns:

        training_activities = int(
            df[
                "activity_id"
            ].nunique()
        )

    else:

        training_activities = 0

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
        feature_names=(
            feature_names
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
    )


# =============================================================================
# Prediction API
# =============================================================================

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


def predict_macro_profile(
    model: MacroModel,
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add independent macro predictions to a normalized GPX profile.

    The profile rows represent segment endpoints.

    Macro cumulative prediction:

        M(distance, cumulative ascent, cumulative descent)

    Segment time:

        M(current endpoint)
        -
        M(previous endpoint)

    Because M(0,0,0)=0, the first segment is simply the first cumulative
    prediction.
    """

    if model is None:

        raise ValueError(
            "Macro model is None."
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
            "Profile is missing required macro columns: "
            + ", ".join(
                missing
            )
        )

    # -------------------------------------------------------------------------
    # Explicitly ensure that GPX normalization and model prediction are using
    # the same project configuration.
    # -------------------------------------------------------------------------

    _validate_profile_spacing(
        profile_df
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
        "macro_predicted_cumulative_time_s"
    ] = cumulative_prediction

    cumulative = result[
        "macro_predicted_cumulative_time_s"
    ].to_numpy(
        dtype=float
    )

    result[
        "macro_predicted_time_s"
    ] = np.diff(
        cumulative,
        prepend=0.0,
    )

    return result
