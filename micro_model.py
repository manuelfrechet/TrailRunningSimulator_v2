from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# V0 Micro analog model
# -----------------------------------------------------------------------------
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
# V0 method:
#
#   - standardize all 7 dimensions using the historical corpus
#   - build one KD-tree
#   - query the 2 closest historical states
#   - interpolate their observed next-50 m times using inverse distance
#
# Important:
#
#   - no macro model dependency
#   - no macro correction
#   - no clipping
#   - no weighting chosen by hand
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------

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
        Build a query dataframe with exactly the same seven dimensions
        used by the historical model.
        """

        return pd.DataFrame(
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
                "elapsed_time_s": np.atleast_1d(
                    elapsed_time_s
                ),
                "segment_ascent_m": np.atleast_1d(
                    segment_ascent_m
                ),
                "segment_descent_m": np.atleast_1d(
                    segment_descent_m
                ),
                "segment_grade_pct": np.atleast_1d(
                    segment_grade_pct
                ),
            }
        )

    def _standardize_query(
        self,
        query_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Standardize a query using the historical means/scales.
        """

        query = query_df[
            self.state_columns
        ].copy()

        for column in self.state_columns:
            query[column] = pd.to_numeric(
                query[column],
                errors="coerce",
            )

        # Missing query values are replaced with the historical mean.
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

    def predict(
        self,
        query_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict one or more next-50 m times.

        Returns one row per query with:
            micro_predicted_time_s
            analogue distances
            analogue observed times
            analogue activity IDs/names
            analogue historical positions
        """

        if query_df is None or query_df.empty:
            return pd.DataFrame()

        missing = [
            column
            for column in self.state_columns
            if column not in query_df.columns
        ]

        if missing:
            raise ValueError(
                "Micro query is missing required columns: "
                + ", ".join(missing)
            )

        query_standardized = self._standardize_query(
            query_df
        )

        distances, indices = self.tree.query(
            query_standardized,
            k=N_ANALOGUES,
        )

        # ---------------------------------------------------------------------
        # cKDTree returns 1D arrays for k=1 and 2D arrays for k=2.
        # Normalize explicitly.
        # ---------------------------------------------------------------------

        if N_ANALOGUES == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        distances = np.asarray(
            distances,
            dtype=float,
        )

        indices = np.asarray(
            indices,
            dtype=int,
        )

        # ---------------------------------------------------------------------
        # Retrieve analogue target times.
        # ---------------------------------------------------------------------

        analogue_times = (
            self.historical_target_times_s[
                indices
            ]
        )

        # ---------------------------------------------------------------------
        # Inverse-distance interpolation.
        # ---------------------------------------------------------------------

        predictions = np.empty(
            len(query_df),
            dtype=float,
        )

        for row_index in range(
            len(query_df)
        ):

            row_distances = distances[
                row_index
            ]

            row_times = analogue_times[
                row_index
            ]

            # Exact / practically exact match.
            if row_distances[0] <= EPSILON:
                predictions[
                    row_index
                ] = float(
                    row_times[0]
                )
                continue

            weights = 1.0 / (
                row_distances
                + EPSILON
            )

            predictions[
                row_index
            ] = float(
                np.sum(
                    weights
                    * row_times
                )
                / np.sum(weights)
            )

        # ---------------------------------------------------------------------
        # Build result.
        # ---------------------------------------------------------------------

        result = query_df[
            self.state_columns
        ].copy()

        result[
            "micro_predicted_time_s"
        ] = predictions

        # Analogue 1.
        result[
            "analogue_1_distance"
        ] = distances[
            :, 0
        ]

        result[
            "analogue_1_time_s"
        ] = analogue_times[
            :, 0
        ]

        result[
            "analogue_1_activity_id"
        ] = self.historical_activity_ids[
            indices[:, 0]
        ]

        result[
            "analogue_1_activity_name"
        ] = self.historical_activity_names[
            indices[:, 0]
        ]

        result[
            "analogue_1_distance_from_start_m"
        ] = self.historical_distances_m[
            indices[:, 0]
        ]

        # Analogue 2.
        result[
            "analogue_2_distance"
        ] = distances[
            :, 1
        ]

        result[
            "analogue_2_time_s"
        ] = analogue_times[
            :, 1
        ]

        result[
            "analogue_2_activity_id"
        ] = self.historical_activity_ids[
            indices[:, 1]
        ]

        result[
            "analogue_2_activity_name"
        ] = self.historical_activity_names[
            indices[:, 1]
        ]

        result[
            "analogue_2_distance_from_start_m"
        ] = self.historical_distances_m[
            indices[:, 1]
        ]

        return result

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
        Convenience method for one GPX 50 m query.
        """

        query_df = self.build_query(
            distance_from_start_m=distance_from_start_m,
            cumulative_ascent_m=cumulative_ascent_m,
            cumulative_descent_m=cumulative_descent_m,
            elapsed_time_s=elapsed_time_s,
            segment_ascent_m=segment_ascent_m,
            segment_descent_m=segment_descent_m,
            segment_grade_pct=segment_grade_pct,
        )

        result = self.predict(
            query_df
        )

        if result.empty:
            raise ValueError(
                "Micro prediction returned no result."
            )

        return result.iloc[0].to_dict()

    def summary(self) -> dict[str, Any]:
        """
        Return basic information about the historical analog library.
        """

        return {
            "training_rows": self.training_rows,
            "training_activities": self.training_activities,
            "n_state_variables": len(
                self.state_columns
            ),
            "state_columns": list(
                self.state_columns
            ),
            "n_analogues": N_ANALOGUES,
            "distance_metric": (
                "standardized Euclidean"
            ),
            "interpolation": (
                "inverse-distance weighted"
            ),
        }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return one clean numeric 1D Series.
    """

    obj = df.loc[:, column]

    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]

    return pd.to_numeric(
        obj,
        errors="coerce",
    )


def _prepare_training_data(
    learning_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean the historical state/target data.
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
            + ", ".join(missing)
        )

    columns = required_columns

    df = learning_df[
        columns
    ].copy()

    for column in (
        STATE_COLUMNS
        + [
            TARGET_COLUMN,
        ]
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=(
            STATE_COLUMNS
            + [
                TARGET_COLUMN,
            ]
        )
    ).copy()

    # Target must be physically positive.
    df = df[
        df[TARGET_COLUMN] > 0.0
    ].copy()

    if df.empty:
        return pd.DataFrame()

    return df.reset_index(
        drop=True
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def fit_micro_model(
    learning_df: pd.DataFrame,
) -> MicroModel:
    """
    Build the V0 historical analog library and KD-tree.
    """

    if learning_df is None or learning_df.empty:
        raise ValueError(
            "Historical learning dataset is empty."
        )

    training_df = _prepare_training_data(
        learning_df
    )

    if training_df.empty:
        raise ValueError(
            "No valid historical rows remain for the micro model."
        )

    # -------------------------------------------------------------------------
    # Historical state matrix
    # -------------------------------------------------------------------------

    state_frame = pd.DataFrame(
        index=training_df.index
    )

    for column in STATE_COLUMNS:
        state_frame[column] = _safe_numeric_series(
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
    # Build KD-tree ONCE.
    # -------------------------------------------------------------------------

    tree = cKDTree(
        historical_states
    )

    # -------------------------------------------------------------------------
    # Historical outcomes.
    # -------------------------------------------------------------------------

    historical_target_times_s = (
        training_df[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    historical_activity_ids = (
        training_df[
            "activity_id"
        ]
        .to_numpy()
    )

    historical_activity_names = (
        training_df[
            "activity_name"
        ]
        .astype(str)
        .to_numpy()
    )

    historical_distances_m = (
        training_df[
            "distance_from_start_m"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # -------------------------------------------------------------------------
    # Activity count.
    # -------------------------------------------------------------------------

    training_activities = int(
        training_df[
            "activity_id"
        ].nunique()
    )

    return MicroModel(
        state_columns=list(
            STATE_COLUMNS
        ),
        means=means,
        scales=scales,
        coefficients=np.empty(
            0,
            dtype=float,
        ),
        historical_states=historical_states,
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
        tree=tree,
        training_rows=int(
            len(training_df)
        ),
        training_activities=(
            training_activities
        ),
    )


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
    Predict one future 50 m segment.
    """

    if model is None:
        raise ValueError(
            "Micro model is None."
        )

    return model.predict_one(
        distance_from_start_m=distance_from_start_m,
        cumulative_ascent_m=cumulative_ascent_m,
        cumulative_descent_m=cumulative_descent_m,
        elapsed_time_s=elapsed_time_s,
        segment_ascent_m=segment_ascent_m,
        segment_descent_m=segment_descent_m,
        segment_grade_pct=segment_grade_pct,
    )
  
