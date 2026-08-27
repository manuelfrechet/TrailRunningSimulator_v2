from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import GPX_SEGMENT_LENGTH_M
from macro_model import MacroModel
from micro_model import MicroModel


# =============================================================================
# V0 simulator
# =============================================================================
#
# The GPX profile contains one row per normalized segment END.
#
# With:
#
#     GPX_SEGMENT_LENGTH_M = 100 m
#
# rows represent:
#
#     0 -> 100
#     100 -> 200
#     200 -> 300
#     ...
#
# For every segment:
#
#     1. determine the terrain state at the segment start
#     2. predict the segment with the macro model
#     3. predict the segment with the micro model
#     4. advance both clocks
#     5. apply any aid-station stop to both clocks
#     6. continue to the next segment
#
# Macro and micro remain independent.
#
# The micro model receives the CURRENT simulated micro elapsed time because
# elapsed_time_s is one of its seven state variables.
#
# -----------------------------------------------------------------------------
# IMPORTANT V0 PHYSICAL CONSTRAINT
# -----------------------------------------------------------------------------
#
# The polynomial macro model can occasionally produce:
#
#     M(X[k+1]) < M(X[k])
#
# which would imply a negative segment duration.
#
# For V0:
#
#     macro_segment_time = max(raw_macro_segment_time, 0)
#
# We also preserve the raw value so that diagnostics can tell us whether
# clipping is rare and harmless or frequent and problematic.
# =============================================================================


# =============================================================================
# Required GPX profile columns
# =============================================================================

REQUIRED_PROFILE_COLUMNS = [
    "distance_from_start_m",
    "ascent_m",
    "descent_m",
    "cumulative_ascent_m",
    "cumulative_descent_m",
    "grade_pct",
]


# =============================================================================
# Profile validation
# =============================================================================

def _validate_profile(
    profile_df: pd.DataFrame,
) -> None:
    """
    Validate the normalized GPX profile.
    """
    if profile_df is None or profile_df.empty:
        raise ValueError(
            "GPX profile is empty."
        )

    missing = [
        column
        for column in REQUIRED_PROFILE_COLUMNS
        if column not in profile_df.columns
    ]

    if missing:
        raise ValueError(
            "GPX profile is missing required columns: "
            + ", ".join(missing)
        )

    distance = pd.to_numeric(
        profile_df[
            "distance_from_start_m"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        distance
    ).all():
        raise ValueError(
            "GPX profile contains invalid distance values."
        )

    if len(distance) == 0:
        raise ValueError(
            "GPX profile contains no segments."
        )

    # First row represents the end of the first segment.
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
            "GPX profile does not match config.py. "
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
                "GPX profile segment spacing does not match "
                "GPX_SEGMENT_LENGTH_M from config.py."
            )


# =============================================================================
# Segment start state
# =============================================================================

def _get_segment_start_state(
    profile_df: pd.DataFrame,
    row_index: int,
) -> dict[str, float]:
    """
    Return the terrain state at the START of one segment.

    The GPX profile stores segment END rows.

    Therefore:

        row 0 = 0 -> segment_length
        row 1 = segment_length -> 2 * segment_length
        ...
    """
    if row_index == 0:
        return {
            "distance_from_start_m": 0.0,
            "cumulative_ascent_m": 0.0,
            "cumulative_descent_m": 0.0,
        }

    previous = profile_df.iloc[
        row_index - 1
    ]

    return {
        "distance_from_start_m": float(
            previous[
                "distance_from_start_m"
            ]
        ),
        "cumulative_ascent_m": float(
            previous[
                "cumulative_ascent_m"
            ]
        ),
        "cumulative_descent_m": float(
            previous[
                "cumulative_descent_m"
            ]
        ),
    }


# =============================================================================
# Macro prediction
# =============================================================================

def _predict_macro_segment(
    macro_model: MacroModel,
    start_state: dict[str, float],
    end_row: pd.Series,
) -> tuple[float, float]:
    """
    Predict one segment with the macro model.

    Returns:

        macro_segment_time_s
        raw_macro_segment_time_s

    The raw value is retained so the simulator can report whether the
    physical non-negative-duration constraint was activated.
    """

    start_prediction = (
        macro_model.predict_cumulative_time(
            distance_from_start_m=np.array(
                [
                    start_state[
                        "distance_from_start_m"
                    ]
                ],
                dtype=float,
            ),
            cumulative_ascent_m=np.array(
                [
                    start_state[
                        "cumulative_ascent_m"
                    ]
                ],
                dtype=float,
            ),
            cumulative_descent_m=np.array(
                [
                    start_state[
                        "cumulative_descent_m"
                    ]
                ],
                dtype=float,
            ),
        )
    )

    end_prediction = (
        macro_model.predict_cumulative_time(
            distance_from_start_m=np.array(
                [
                    float(
                        end_row[
                            "distance_from_start_m"
                        ]
                    )
                ],
                dtype=float,
            ),
            cumulative_ascent_m=np.array(
                [
                    float(
                        end_row[
                            "cumulative_ascent_m"
                        ]
                    )
                ],
                dtype=float,
            ),
            cumulative_descent_m=np.array(
                [
                    float(
                        end_row[
                            "cumulative_descent_m"
                        ]
                    )
                ],
                dtype=float,
            ),
        )
    )

    raw_segment_time_s = float(
        end_prediction[0]
        - start_prediction[0]
    )

    if not np.isfinite(
        raw_segment_time_s
    ):
        raise ValueError(
            "Macro model returned an invalid segment time."
        )

    # -------------------------------------------------------------------------
    # V0 physical constraint.
    #
    # Elapsed time cannot move backwards.
    # -------------------------------------------------------------------------

    segment_time_s = max(
        raw_segment_time_s,
        0.0,
    )

    return (
        segment_time_s,
        raw_segment_time_s,
    )


# =============================================================================
# Micro prediction
# =============================================================================

def _predict_micro_segment(
    micro_model: MicroModel,
    start_state: dict[str, float],
    current_micro_elapsed_s: float,
    end_row: pd.Series,
) -> dict[str, Any]:
    """
    Predict one segment with the micro analogue model.

    The elapsed_time_s supplied to the micro model is the CURRENT simulated
    micro elapsed time.
    """

    prediction = micro_model.predict_one(
        distance_from_start_m=(
            start_state[
                "distance_from_start_m"
            ]
        ),
        cumulative_ascent_m=(
            start_state[
                "cumulative_ascent_m"
            ]
        ),
        cumulative_descent_m=(
            start_state[
                "cumulative_descent_m"
            ]
        ),
        elapsed_time_s=(
            current_micro_elapsed_s
        ),
        segment_ascent_m=float(
            end_row[
                "ascent_m"
            ]
        ),
        segment_descent_m=float(
            end_row[
                "descent_m"
            ]
        ),
        segment_grade_pct=float(
            end_row[
                "grade_pct"
            ]
        ),
    )

    predicted_time_s = float(
        prediction[
            "micro_predicted_time_s"
        ]
    )

    if not np.isfinite(
        predicted_time_s
    ):
        raise ValueError(
            "Micro model returned an invalid segment time."
        )

    if predicted_time_s < 0.0:
        raise ValueError(
            "Micro model returned a negative segment time."
        )

    return prediction


# =============================================================================
# Aid-station handling
# =============================================================================

def _get_aid_station_info(
    row: pd.Series,
) -> tuple[str, float]:
    """
    Return:

        aid_station_name
        stop_minutes

    No station means:

        ""
        0.0
    """

    station_name = ""

    if (
        "aid_station_name"
        in row.index
    ):

        value = row[
            "aid_station_name"
        ]

        if (
            value is not None
            and not pd.isna(value)
        ):

            station_name = str(
                value
            )

    stop_minutes = 0.0

    if (
        "aid_station_stop_min"
        in row.index
    ):

        value = row[
            "aid_station_stop_min"
        ]

        if (
            value is not None
            and not pd.isna(value)
        ):

            stop_minutes = max(
                0.0,
                float(value),
            )

    return (
        station_name,
        stop_minutes,
    )


# =============================================================================
# Main simulation
# =============================================================================

def simulate_race(
    gpx_profile_df: pd.DataFrame,
    macro_model: MacroModel,
    micro_model: MicroModel,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Simulate one GPX trajectory.

    Simulation advances one GPX_SEGMENT_LENGTH_M segment at a time.

    Macro:
        predicts terrain-based segment time independently.

    Micro:
        searches the historical library using the current simulated state.

    Aid stations:
        stop time is added to BOTH clocks after arrival.

    Returns:
        simulation dataframe
        race summary
    """

    if macro_model is None:
        raise ValueError(
            "macro_model is required."
        )

    if micro_model is None:
        raise ValueError(
            "micro_model is required."
        )

    _validate_profile(
        gpx_profile_df
    )

    profile = (
        gpx_profile_df
        .copy()
        .sort_values(
            "distance_from_start_m",
            kind="mergesort",
        )
        .reset_index(
            drop=True,
        )
    )

    # -------------------------------------------------------------------------
    # Independent simulation clocks.
    # -------------------------------------------------------------------------

    macro_elapsed_s = 0.0
    micro_elapsed_s = 0.0

    rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Segment-by-segment simulation.
    # -------------------------------------------------------------------------

    for row_index in range(
        len(profile)
    ):

        end_row = profile.iloc[
            row_index
        ]

        start_state = (
            _get_segment_start_state(
                profile,
                row_index,
            )
        )

        # =====================================================================
        # MACRO
        # =====================================================================

        (
            macro_segment_time_s,
            raw_macro_segment_time_s,
        ) = _predict_macro_segment(
            macro_model=macro_model,
            start_state=start_state,
            end_row=end_row,
        )

        macro_time_clipped = (
            raw_macro_segment_time_s
            < 0.0
        )

        # =====================================================================
        # MICRO
        # =====================================================================

        micro_prediction = (
            _predict_micro_segment(
                micro_model=micro_model,
                start_state=start_state,
                current_micro_elapsed_s=(
                    micro_elapsed_s
                ),
                end_row=end_row,
            )
        )

        micro_segment_time_s = float(
            micro_prediction[
                "micro_predicted_time_s"
            ]
        )

        # =====================================================================
        # ARRIVAL TIMES
        # =====================================================================

        macro_arrival_s = (
            macro_elapsed_s
            + macro_segment_time_s
        )

        micro_arrival_s = (
            micro_elapsed_s
            + micro_segment_time_s
        )

        # =====================================================================
        # MODEL DIFFERENCE
        # =====================================================================

        segment_difference_s = (
            micro_segment_time_s
            - macro_segment_time_s
        )

        arrival_difference_s = (
            micro_arrival_s
            - macro_arrival_s
        )

        # =====================================================================
        # AID STATION
        # =====================================================================

        (
            aid_station_name,
            aid_stop_minutes,
        ) = _get_aid_station_info(
            end_row
        )

        aid_stop_s = (
            aid_stop_minutes
            * 60.0
        )

        # =====================================================================
        # DEPARTURE TIMES
        #
        # Aid-station time is added AFTER arrival.
        # =====================================================================

        macro_departure_s = (
            macro_arrival_s
            + aid_stop_s
        )

        micro_departure_s = (
            micro_arrival_s
            + aid_stop_s
        )

        departure_difference_s = (
            micro_departure_s
            - macro_departure_s
        )

        # =====================================================================
        # SAVE RESULT
        # =====================================================================

        rows.append(
            {
                # -----------------------------------------------------------------
                # Position
                # -----------------------------------------------------------------

                "distance_from_start_m": float(
                    end_row[
                        "distance_from_start_m"
                    ]
                ),

                "segment_start_m": float(
                    start_state[
                        "distance_from_start_m"
                    ]
                ),

                "segment_end_m": float(
                    end_row[
                        "distance_from_start_m"
                    ]
                ),

                # -----------------------------------------------------------------
                # Terrain
                # -----------------------------------------------------------------

                "ascent_m": float(
                    end_row[
                        "ascent_m"
                    ]
                ),

                "descent_m": float(
                    end_row[
                        "descent_m"
                    ]
                ),

                "cumulative_ascent_m": float(
                    end_row[
                        "cumulative_ascent_m"
                    ]
                ),

                "cumulative_descent_m": float(
                    end_row[
                        "cumulative_descent_m"
                    ]
                ),

                "grade_pct": float(
                    end_row[
                        "grade_pct"
                    ]
                ),

                # -----------------------------------------------------------------
                # Macro
                # -----------------------------------------------------------------

                "raw_macro_predicted_time_s": (
                    raw_macro_segment_time_s
                ),

                "macro_predicted_time_s": (
                    macro_segment_time_s
                ),

                "macro_time_clipped": (
                    macro_time_clipped
                ),

                # -----------------------------------------------------------------
                # Micro
                # -----------------------------------------------------------------

                "micro_predicted_time_s": (
                    micro_segment_time_s
                ),

                # -----------------------------------------------------------------
                # Segment comparison
                # -----------------------------------------------------------------

                "micro_minus_macro_s": (
                    segment_difference_s
                ),

                # -----------------------------------------------------------------
                # Cumulative arrival times
                # -----------------------------------------------------------------

                "macro_arrival_time_s": (
                    macro_arrival_s
                ),

                "micro_arrival_time_s": (
                    micro_arrival_s
                ),

                "micro_minus_macro_arrival_s": (
                    arrival_difference_s
                ),

                # -----------------------------------------------------------------
                # Aid station
                # -----------------------------------------------------------------

                "aid_station_name": (
                    aid_station_name
                ),

                "aid_station_stop_min": (
                    aid_stop_minutes
                ),

                "aid_station_stop_s": (
                    aid_stop_s
                ),

                # -----------------------------------------------------------------
                # Departure times after aid station
                # -----------------------------------------------------------------

                "macro_departure_time_s": (
                    macro_departure_s
                ),

                "micro_departure_time_s": (
                    micro_departure_s
                ),

                "micro_minus_macro_departure_s": (
                    departure_difference_s
                ),

                # -----------------------------------------------------------------
                # Micro analogue diagnostics
                # -----------------------------------------------------------------

                "analogue_1_distance": (
                    micro_prediction[
                        "analogue_1_distance"
                    ]
                ),

                "analogue_1_time_s": (
                    micro_prediction[
                        "analogue_1_time_s"
                    ]
                ),

                "analogue_1_activity_id": (
                    micro_prediction[
                        "analogue_1_activity_id"
                    ]
                ),

                "analogue_1_activity_name": (
                    micro_prediction[
                        "analogue_1_activity_name"
                    ]
                ),

                "analogue_1_distance_from_start_m": (
                    micro_prediction[
                        "analogue_1_distance_from_start_m"
                    ]
                ),

                "analogue_2_distance": (
                    micro_prediction[
                        "analogue_2_distance"
                    ]
                ),

                "analogue_2_time_s": (
                    micro_prediction[
                        "analogue_2_time_s"
                    ]
                ),

                "analogue_2_activity_id": (
                    micro_prediction[
                        "analogue_2_activity_id"
                    ]
                ),

                "analogue_2_activity_name": (
                    micro_prediction[
                        "analogue_2_activity_name"
                    ]
                ),

                "analogue_2_distance_from_start_m": (
                    micro_prediction[
                        "analogue_2_distance_from_start_m"
                    ]
                ),
            }
        )

        # =====================================================================
        # ADVANCE BOTH CLOCKS
        # =====================================================================

        macro_elapsed_s = (
            macro_departure_s
        )

        micro_elapsed_s = (
            micro_departure_s
        )

    # =============================================================================
    # FINAL DATAFRAME
    # =============================================================================

    simulation_df = pd.DataFrame(
        rows
    )

    if simulation_df.empty:
        raise ValueError(
            "Simulation produced no predictions."
        )

    # =============================================================================
    # RACE SUMMARY
    # =============================================================================

    total_aid_stop_s = float(
        simulation_df[
            "aid_station_stop_s"
        ].sum()
    )

    final_row = simulation_df.iloc[
        -1
    ]

    macro_final_arrival_s = float(
        final_row[
            "macro_arrival_time_s"
        ]
    )

    micro_final_arrival_s = float(
        final_row[
            "micro_arrival_time_s"
        ]
    )

    macro_final_race_s = (
        macro_final_arrival_s
        + total_aid_stop_s
    )

    micro_final_race_s = (
        micro_final_arrival_s
        + total_aid_stop_s
    )

    clipped_mask = (
        simulation_df[
            "macro_time_clipped"
        ]
    )

    macro_clipped_segments = int(
        clipped_mask.sum()
    )

    macro_clipped_seconds = float(
        simulation_df.loc[
            clipped_mask,
            "raw_macro_predicted_time_s",
        ]
        .clip(
            upper=0.0
        )
        .sum()
    )

    race_summary = {
        "segment_length_m": (
            GPX_SEGMENT_LENGTH_M
        ),

        "segments": int(
            len(
                simulation_df
            )
        ),

        "distance_m": float(
            simulation_df[
                "distance_from_start_m"
            ].iloc[-1]
        ),

        "total_aid_stops": int(
            (
                simulation_df[
                    "aid_station_stop_s"
                ]
                > 0.0
            ).sum()
        ),

        "total_aid_stop_s": (
            total_aid_stop_s
        ),

        "total_aid_stop_min": (
            total_aid_stop_s
            / 60.0
        ),

        "macro_final_arrival_s": (
            macro_final_arrival_s
        ),

        "micro_final_arrival_s": (
            micro_final_arrival_s
        ),

        "macro_final_race_s": (
            macro_final_race_s
        ),

        "micro_final_race_s": (
            micro_final_race_s
        ),

        "macro_final_race_h": (
            macro_final_race_s
            / 3600.0
        ),

        "micro_final_race_h": (
            micro_final_race_s
            / 3600.0
        ),

        "micro_minus_macro_final_s": (
            micro_final_race_s
            - macro_final_race_s
        ),

        "micro_minus_macro_final_min": (
            (
                micro_final_race_s
                - macro_final_race_s
            )
            / 60.0
        ),

        # ---------------------------------------------------------------------
        # Macro physical-constraint diagnostics
        # ---------------------------------------------------------------------

        "macro_clipped_segments": (
            macro_clipped_segments
        ),

        "macro_clipped_seconds": (
            macro_clipped_seconds
        ),

        "macro_clipped_minutes": (
            macro_clipped_seconds
            / 60.0
        ),
    }

    return (
        simulation_df,
        race_summary,
    )


# =============================================================================
# Formatting helpers
# =============================================================================

def format_seconds(
    seconds: float,
) -> str:
    """
    Format seconds as H:MM:SS.
    """
    if not np.isfinite(
        seconds
    ):
        return "—"

    seconds = max(
        0.0,
        float(seconds),
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (
            seconds
            % 3600
        )
        // 60
    )

    remaining_seconds = int(
        round(
            seconds
            % 60
        )
    )

    if remaining_seconds == 60:

        remaining_seconds = 0
        minutes += 1

    if minutes == 60:

        minutes = 0
        hours += 1

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


def build_simulation_summary(
    simulation_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a compact summary directly from simulation output.
    """
    if (
        simulation_df is None
        or simulation_df.empty
    ):
        return {}

    final_row = simulation_df.iloc[
        -1
    ]

    total_aid_stop_s = float(
        simulation_df[
            "aid_station_stop_s"
        ].sum()
    )

    macro_race_s = (
        float(
            final_row[
                "macro_arrival_time_s"
            ]
        )
        + total_aid_stop_s
    )

    micro_race_s = (
        float(
            final_row[
                "micro_arrival_time_s"
            ]
        )
        + total_aid_stop_s
    )

    clipped_segments = int(
        simulation_df[
            "macro_time_clipped"
        ].sum()
    )

    clipped_seconds = float(
        simulation_df.loc[
            simulation_df[
                "macro_time_clipped"
            ],
            "raw_macro_predicted_time_s",
        ]
        .clip(
            upper=0.0
        )
        .sum()
    )

    return {
        "distance_m": float(
            final_row[
                "distance_from_start_m"
            ]
        ),

        "segments": int(
            len(
                simulation_df
            )
        ),

        "total_aid_stop_min": (
            total_aid_stop_s
            / 60.0
        ),

        "macro_race_time_s": (
            macro_race_s
        ),

        "micro_race_time_s": (
            micro_race_s
        ),

        "macro_race_time": (
            format_seconds(
                macro_race_s
            )
        ),

        "micro_race_time": (
            format_seconds(
                micro_race_s
            )
        ),

        "micro_minus_macro_min": (
            (
                micro_race_s
                - macro_race_s
            )
            / 60.0
        ),

        "macro_clipped_segments": (
            clipped_segments
        ),

        "macro_clipped_seconds": (
            clipped_seconds
        ),
    }


# =============================================================================
# Aid-station summary
# =============================================================================

def build_aid_station_summary(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract predicted arrival/departure information for aid stations.
    """
    if (
        simulation_df is None
        or simulation_df.empty
    ):
        return pd.DataFrame()

    if (
        "aid_station_name"
        not in simulation_df.columns
    ):
        return pd.DataFrame()

    station_rows = (
        simulation_df[
            simulation_df[
                "aid_station_name"
            ]
            .astype(str)
            .str.len()
            > 0
        ]
        .copy()
    )

    if station_rows.empty:
        return pd.DataFrame()

    return station_rows[
        [
            "aid_station_name",
            "distance_from_start_m",
            "aid_station_stop_min",
            "macro_arrival_time_s",
            "micro_arrival_time_s",
            "macro_departure_time_s",
            "micro_departure_time_s",
            "micro_minus_macro_arrival_s",
        ]
    ].reset_index(
        drop=True
    )
