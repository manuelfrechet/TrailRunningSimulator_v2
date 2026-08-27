from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import GPX_SEGMENT_LENGTH_M


# =============================================================================
# V0 Micro analog model
# =============================================================================
#
# Historical state vector:
#
#   1. distance_from_start_m
#   2. cumulative_ascent_m
#   3. cumulative_descent_m
#   4. elapsed_time_s
#   5. segment_ascent_m
#   6. segment_descent_m
#   7. segment_grade_pct
#
# Target:
#
#   actual_segment_time_s
#
# Historical learning density:
#
#   controlled upstream by LEARNING_STEP_M in fit_learning.py.
#
# Historical transition length:
#
#   GPX_SEGMENT_LENGTH_M
#
# Therefore, with:
#
#   LEARNING_STEP_M = 1 m
#   GPX_SEGMENT_LENGTH_M = 100 m
#
# the historical library contains:
#
#   state at 0 m -> observed next 100 m
#   state at 1 m -> observed next 100 m
#   state at 2 m -> observed next 100 m
#   ...
#
# V0 method:
#
#   - standardize all 7 dimensions using the historical corpus
#   - build one KD-tree
#   - query the 2 closest historical states
#   - inverse-distance interpolate their observed transition times
#
# IMPORTANT:
#
#   - no macro dependency
#   - no macro correction
#   - no clipping
#   - no hand-tuned feature weights
# =============================================================================


STATE_COLUMNS = [
    "distance_from_start_m",
    "cumulative_ascent_m",
    "cumulative_descent_m",
    "elapsed_time_s",
    "segment_ascent_m",
    "segment_descent_m",
    "segment_grade_pct",
]

TARGET_COLUMN = "actual_segment_time_s"

N_ANALOGUES = 2

EPSILON = 1e-12


# =============================================================================
# Data container
# =============================================================================

@dataclass
class MicroModel:

    state_columns: list[str]

    means: pd.Series
    scales: pd.Series

    historical_states: np.ndarray
    historical_target_times_s: np.ndarray

    historical_activity_ids: np.ndarray
    historical_activity_names: np.ndarray
    historical_distances_m: np.ndarray

    tree: cKDTree

    training_rows: int
    training_activities: int

    # -------------------------------------------------------------------------
    # Query construction
    # -------------------------------------------------------------------------

    def build_query(
        self,
        distance_from_start_m: Any,
        cumulative_ascent_m: Any,
        cumulative_descent_m: Any,
        elapsed_time_s: Any,
        segment_ascent_m: Any,
        segment_descent_m: Any,
        segment_grade_pct: Any,
    ) -> pd.DataFrame:
        """
        Build one or more query states.

        Each query describes:

            state at the beginning of the transition

        plus:

            terrain over the next GPX_SEGMENT_LENGTH_M.
        """

        return pd.DataFrame(
            {
                "distance_from_start_m": (
                    np.atleast_1d(
                        distance_from_start_m
                    )
                ),
                "cumulative_ascent_m": (
                    np.atleast_1d(
                        cumulative_ascent_m
                    )
                ),
                "cumulative_descent_m": (
                    np.atleast_1d(
                        cumulative_descent_m
                    )
                ),
                "elapsed_time_s": (
                    np.atleast_1d(
                        elapsed_time_s
                    )
                ),
                "segment_ascent_m": (
                    np.atleast_1d(
                        segment_ascent_m
                    )
                ),
                "segment_descent_m": (
                    np.atleast_1d(
                        segment_descent_m
                    )
                ),
                "segment_grade_pct": (
                    np.atleast_1d(
                        segment_grade_pct
                    )
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Query standardization
    # -------------------------------------------------------------------------

    def _standardize_query(
        self,
        query_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Standardize a query using historical corpus means and scales.
        """

        query = query_df[
            self.state_columns
        ].copy()

        for column in self.state_columns:

            query[column] = (
                pd.to_numeric(
                    query[column],
                    errors="coerce",
                )
            )

        query = query.fillna(
            self.means
        )

        standardized = (
            query
            - self.means
        ) / self.scales

        return standardized.to_numpy(
            dtype=float
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict(
        self,
        query_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict one or more transition times.

        The transition length is GPX_SEGMENT_LENGTH_M.

        Returns:
            micro_predicted_time_s

            analogue_1_distance
            analogue_1_time_s
            analogue_1_activity_id
            analogue_1_activity_name
            analogue_1_distance_from_start_m

            analogue_2_distance
            analogue_2_time_s
            analogue_2_activity_id
            analogue_2_activity_name
            analogue_2_distance_from_start_m
        """

        if (
            query_df is None
            or query_df.empty
        ):

            return pd.DataFrame()

        missing = [
            column
            for column in self.state_columns
            if column not in query_df.columns
        ]

        if missing:

            raise ValueError(
                "Micro query is missing required columns: "
                + ", ".join(
                    missing
                )
            )

        query_standardized = (
            self._standardize_query(
                query_df
            )
        )

        distances, indices = (
            self.tree.query(
                query_standardized,
                k=N_ANALOGUES,
            )
        )

        distances = np.asarray(
            distances,
            dtype=float,
        )

        indices = np.asarray(
            indices,
            dtype=int,
        )

        # cKDTree can return one-dimensional output depending on shape.
        if distances.ndim == 1:

            distances = (
                distances.reshape(
                    -1,
                    1,
                )
            )

        if indices.ndim == 1:

            indices = (
                indices.reshape(
                    -1,
                    1,
                )
            )

        if distances.shape[1] < N_ANALOGUES:

            raise ValueError(
                "Micro model could not return the required "
                f"{N_ANALOGUES} historical analogues."
            )

        analogue_times = (
            self.historical_target_times_s[
                indices
            ]
        )

        predictions = np.empty(
            len(
                query_df
            ),
            dtype=float,
        )

        for row_index in range(
            len(
                query_df
            )
        ):

            row_distances = distances[
                row_index
            ]

            row_times = analogue_times[
                row_index
            ]

            # -----------------------------------------------------------------
            # Exact historical state.
            # -----------------------------------------------------------------

            if (
                row_distances[0]
                <= EPSILON
            ):

                predictions[
                    row_index
                ] = float(
                    row_times[0]
                )

                continue

            # -----------------------------------------------------------------
            # Inverse-distance interpolation.
            # -----------------------------------------------------------------

            weights = (
                1.0
                / (
                    row_distances
                    + EPSILON
                )
            )

            predictions[
                row_index
            ] = float(
                np.sum(
                    weights
                    * row_times
                )
                / np.sum(
                    weights
                )
            )

        result = query_df[
            self.state_columns
        ].copy()

        result[
            "micro_predicted_time_s"
        ] = predictions

        # ---------------------------------------------------------------------
        # Analogue 1
        # ---------------------------------------------------------------------

        result[
            "analogue_1_distance"
        ] = distances[
            :,
            0,
        ]

        result[
            "analogue_1_time_s"
        ] = analogue_times[
            :,
            0,
        ]

        result[
            "analogue_1_activity_id"
        ] = (
            self.historical_activity_ids[
                indices[
                    :,
                    0,
                ]
            ]
        )

        result[
            "analogue_1_activity_name"
        ] = (
            self.historical_activity_names[
                indices[
                    :,
                    0,
                ]
            ]
        )

        result[
            "analogue_1_distance_from_start_m"
        ] = (
            self.historical_distances_m[
                indices[
                    :,
                    0,
                ]
            ]
        )

        # ---------------------------------------------------------------------
        # Analogue 2
        # ---------------------------------------------------------------------

        result[
            "analogue_2_distance"
        ] = distances[
            :,
            1,
        ]

        result[
            "analogue_2_time_s"
        ] = analogue_times[
            :,
            1,
        ]

        result[
            "analogue_2_activity_id"
        ] = (
            self.historical_activity_ids[
                indices[
                    :,
                    1,
                ]
            ]
        )

        result[
            "analogue_2_activity_name"
        ] = (
            self.historical_activity_names[
                indices[
                    :,
                    1,
                ]
            ]
        )

        result[
            "analogue_2_distance_from_start_m"
        ] = (
            self.historical_distances_m[
                indices[
                    :,
                    1,
                ]
            ]
        )

        return result

    # -------------------------------------------------------------------------
    # Single transition prediction
    # -------------------------------------------------------------------------

    def predict_one(
        self,
        *,
        distance_from_start_m: float,
        cumulative_ascent_m: float,
        cumulative_descent_m: float,
        elapsed_time_s: float,
        segment_ascent_m: float,
        segment_descent_m: float,
        segment_grade_pct: float,
    ) -> dict[str, Any]:
        """
        Predict one future GPX_SEGMENT_LENGTH_M transition.
        """

        query_df = self.build_query(
            distance_from_start_m=(
                distance_from_start_m
            ),
            cumulative_ascent_m=(
                cumulative_ascent_m
            ),
            cumulative_descent_m=(
                cumulative_descent_m
            ),
            elapsed_time_s=(
                elapsed_time_s
            ),
            segment_ascent_m=(
                segment_ascent_m
            ),
            segment_descent_m=(
                segment_descent_m
            ),
            segment_grade_pct=(
                segment_grade_pct
            ),
        )

        result = self.predict(
            query_df
        )

        if result.empty:

            raise ValueError(
                "Micro prediction returned no result."
            )

        return (
            result.iloc[
                0
            ].to_dict()
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return information about the historical analogue library.
        """

        return {
            "training_rows": (
                self.training_rows
            ),
            "training_activities": (
                self.training_activities
            ),
            "n_state_variables": len(
                self.state_columns
            ),
            "state_columns": list(
                self.state_columns
            ),
            "n_analogues": (
                N_ANALOGUES
            ),
            "distance_metric": (
                "standardized Euclidean"
            ),
            "interpolation": (
                "inverse-distance weighted"
            ),
            "segment_length_m": (
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


def _prepare_training_data(
    learning_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean historical state and target data.

    The actual transition length has already been established upstream by
    fit_learning.py using GPX_SEGMENT_LENGTH_M.
    """

    required_columns = (
        STATE_COLUMNS
        + [
            TARGET_COLUMN,
            "activity_id",
            "activity_name",
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in learning_df.columns
    ]

    if missing:

        raise ValueError(
            "Micro model is missing required columns: "
            + ", ".join(
                missing
            )
        )

    training_df = learning_df[
        required_columns
    ].copy()

    for column in (
        STATE_COLUMNS
        + [
            TARGET_COLUMN,
        ]
    ):

        training_df[
            column
        ] = pd.to_numeric(
            training_df[
                column
            ],
            errors="coerce",
        )

    training_df = (
        training_df.dropna(
            subset=(
                STATE_COLUMNS
                + [
                    TARGET_COLUMN,
                ]
            )
        )
        .copy()
    )

    # Observed transition duration must be positive.
    training_df = training_df[
        training_df[
            TARGET_COLUMN
        ] > 0.0
    ].copy()

    if training_df.empty:

        return pd.DataFrame()

    return training_df.reset_index(
        drop=True
    )


# =============================================================================
# Fit micro model
# =============================================================================

def fit_micro_model(
    learning_df: pd.DataFrame,
) -> MicroModel:
    """
    Build the V0 historical analogue library and KD-tree.

    fit_learning.py has already created one observed transition for every
    LEARNING_STEP_M starting position, with each target covering
    GPX_SEGMENT_LENGTH_M.
    """

    if (
        learning_df is None
        or learning_df.empty
    ):

        raise ValueError(
            "Historical learning dataset is empty."
        )

    training_df = (
        _prepare_training_data(
            learning_df
        )
    )

    if training_df.empty:

        raise ValueError(
            "No valid historical rows remain for the micro model."
        )

    if (
        len(
            training_df
        )
        < N_ANALOGUES
    ):

        raise ValueError(
            "Not enough historical rows to build the micro analogue model."
        )

    # -------------------------------------------------------------------------
    # Historical state matrix
    # -------------------------------------------------------------------------

    state_frame = pd.DataFrame(
        index=training_df.index
    )

    for column in STATE_COLUMNS:

        state_frame[
            column
        ] = _safe_numeric_series(
            training_df,
            column,
        )

    means = state_frame.mean(
        axis=0,
        skipna=True,
    )

    scales = state_frame.std(
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
        lower=1e-9
    )

    standardized_states = (
        state_frame
        - means
    ) / scales

    historical_states = (
        standardized_states.to_numpy(
            dtype=float
        )
    )

    # -------------------------------------------------------------------------
    # Build one KD-tree over the complete historical corpus.
    # -------------------------------------------------------------------------

    tree = cKDTree(
        historical_states
    )

    # -------------------------------------------------------------------------
    # Historical observed transition times.
    # -------------------------------------------------------------------------

    historical_target_times_s = (
        training_df[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )
    )

    # -------------------------------------------------------------------------
    # Historical identifiers for diagnostics.
    # -------------------------------------------------------------------------

    historical_activity_ids = (
        training_df[
            "activity_id"
        ].to_numpy()
    )

    historical_activity_names = (
        training_df[
            "activity_name"
        ]
        .astype(
            str
        )
        .to_numpy()
    )

    historical_distances_m = (
        training_df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
    )

    training_activities = int(
        training_df[
            "activity_id"
        ].nunique()
    )

    return MicroModel(
        state_columns=list(
            STATE_COLUMNS
        ),
        means=(
            means
        ),
        scales=(
            scales
        ),
        historical_states=(
            historical_states
        ),
        historical_target_times_s=(
            historical_target_times_s
        ),
        historical_activity_ids=(
            historical_activity_ids
        ),
        historical_activity_names=(
            historical_activity_names
        ),
        historical_distances_m=(
            historical_distances_m
        ),
        tree=(
            tree
        ),
        training_rows=int(
            len(
                training_df
            )
        ),
        training_activities=(
            training_activities
        ),
    )


# =============================================================================
# Public prediction helper
# =============================================================================

def predict_micro_segment(
    model: MicroModel,
    *,
    distance_from_start_m: float,
    cumulative_ascent_m: float,
    cumulative_descent_m: float,
    elapsed_time_s: float,
    segment_ascent_m: float,
    segment_descent_m: float,
    segment_grade_pct: float,
) -> dict[str, Any]:
    """
    Predict one future GPX_SEGMENT_LENGTH_M segment.
    """

    if model is None:

        raise ValueError(
            "Micro model is None."
        )

    return model.predict_one(
        distance_from_start_m=(
            distance_from_start_m
        ),
        cumulative_ascent_m=(
            cumulative_ascent_m
        ),
        cumulative_descent_m=(
            cumulative_descent_m
        ),
        elapsed_time_s=(
            elapsed_time_s
        ),
        segment_ascent_m=(
            segment_ascent_m
        ),
        segment_descent_m=(
            segment_descent_m
        ),
        segment_grade_pct=(
            segment_grade_pct
        ),
    )
    
