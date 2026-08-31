from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aid_stations import (
    aid_station_stop_seconds,
    get_aid_station_at_profile_row,
)
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
#     1. determine the terrain state at the segment start;
#     2. predict with macro;
#     3. predict with micro;
#     4. record arrival times;
#     5. process any aid-station stop;
#     6. advance both clocks;
#     7. continue to the next segment.
#
# Macro and micro remain independent.
#
# The micro model receives the CURRENT simulated micro elapsed time because
# elapsed_time_s is one of its seven state variables.
#
# IMPORTANT:
#
# The cumulative simulation clocks already include aid-station stops.
# Therefore the final race time is the final departure clock.
#
# We must NOT add total_aid_stop_s again at the end.
#
# -----------------------------------------------------------------------------
# MACRO PHYSICAL CONSTRAINT
# -----------------------------------------------------------------------------
#
# The unconstrained polynomial macro model can occasionally produce:
#
#     M(X[k+1]) < M(X[k])
#
# which implies a negative segment duration.
#
# V0 applies:
#
#     macro_segment_time =
#         max(raw_macro_segment_time, 0)
#
# The raw value remains available for diagnostics.
# =============================================================================


# =============================================================================
# Required profile columns
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

    if (
        profile_df is None
        or profile_df.empty
    ):
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

    # -------------------------------------------------------------------------
    # First row is the end of the first segment.
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Every subsequent endpoint must advance by the configured length.
    # -------------------------------------------------------------------------

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
# Segment-start terrain state
# =============================================================================

def _get_segment_start_state(
    profile_df: pd.DataFrame,
    row_index: int,
) -> dict[str, float]:
    """
    Return terrain state at the beginning of one segment.
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
) -> tuple[
    float,
    float,
]:
    """
    Predict one GPX segment with the macro model.

    Returns:

        constrained segment time
        raw unconstrained segment time
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
    # V0 physical constraint:
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
    Predict one GPX segment with the micro analogue model.

    The micro model receives the cumulative simulated micro elapsed time at the
    beginning of this segment.

    That value includes all previously incurred aid-station stops.
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
    Simulate one GPX race.

    Each iteration represents exactly one
    GPX_SEGMENT_LENGTH_M transition.

    Macro and micro have independent cumulative clocks.

    Aid-station time is added AFTER arrival to BOTH clocks.

    Because the clocks are cumulative, the final race time is already the
    final departure clock. It must NOT receive total aid-station time a second
    time.
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
    # Independent cumulative simulation clocks.
    #
    # Both clocks include aid-station stops after they occur.
    # -------------------------------------------------------------------------

    macro_elapsed_s = 0.0
    micro_elapsed_s = 0.0

    rows: list[
        dict[str, Any]
    ] = []

    # =============================================================================
    # Segment-by-segment simulation
    # =============================================================================

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
        # SEGMENT DIFFERENCE
        # =====================================================================

        segment_difference_s = (
            micro_segment_time_s
            - macro_segment_time_s
        )

        # =====================================================================
        # AID STATION
        # =====================================================================

        aid_station = (
            get_aid_station_at_profile_row(
                end_row
            )
        )

        aid_stop_s = (
            aid_station_stop_seconds(
                aid_station
            )
        )

        if aid_station is None:

            aid_station_name = ""
            aid_station_distance_m = np.nan
            aid_stop_minutes = 0.0

        else:

            aid_station_name = (
                aid_station.name
            )

            aid_station_distance_m = (
                float(
                    aid_station.distance_from_start_m
                )
            )

            aid_stop_minutes = (
                float(
                    aid_station.stop_minutes
                )
            )

        # =====================================================================
        # DEPARTURE TIMES
        #
        # These become the cumulative clocks for the next iteration.
        # =====================================================================

        macro_departure_s = (
            macro_arrival_s
            + aid_stop_s
        )

        micro_departure_s = (
            micro_arrival_s
            + aid_stop_s
        )

        arrival_difference_s = (
            micro_arrival_s
            - macro_arrival_s
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
                # Macro prediction
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
                # Micro prediction
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

                "aid_station_distance_m": (
                    aid_station_distance_m
                ),

                "aid_station_stop_min": (
                    aid_stop_minutes
                ),

                "aid_station_stop_s": (
                    aid_stop_s
                ),

                # -----------------------------------------------------------------
                # Cumulative departure times
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
    # FINAL RACE SUMMARY
    # =============================================================================

    final_row = simulation_df.iloc[
        -1
    ]

    total_aid_stop_s = float(
        simulation_df[
            "aid_station_stop_s"
        ].sum()
    )

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

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Final arrival is NOT the final race time if an aid station occurs on the
    # final profile row. The final DEPARTURE clock is the actual race clock.
    #
    # There is NO additional + total_aid_stop_s here.
    # -------------------------------------------------------------------------

    macro_final_race_s = float(
        final_row[
            "macro_departure_time_s"
        ]
    )

    micro_final_race_s = float(
        final_row[
            "micro_departure_time_s"
        ]
    )

    # -------------------------------------------------------------------------
    # Macro clipping diagnostics
    # -------------------------------------------------------------------------

    macro_clipped_mask = (
        simulation_df[
            "macro_time_clipped"
        ]
    )

    macro_clipped_segments = int(
        macro_clipped_mask.sum()
    )

    macro_clipped_seconds = float(
        simulation_df.loc[
            macro_clipped_mask,
            "raw_macro_predicted_time_s",
        ]
        .abs()
        .sum()
    )

    # -------------------------------------------------------------------------
    # Number of actual aid-station events.
    # -------------------------------------------------------------------------

    total_aid_stops = int(
        (
            simulation_df[
                "aid_station_stop_s"
            ]
            > 0.0
        ).sum()
    )

    # -------------------------------------------------------------------------
    # Sanity check:
    #
    # Final race time must equal final arrival time plus only the stop occurring
    # on the final row, because all earlier stops are already included in the
    # cumulative arrival clock.
    # -------------------------------------------------------------------------

    final_row_stop_s = float(
        final_row[
            "aid_station_stop_s"
        ]
    )

    if not np.isclose(
        macro_final_race_s,
        macro_final_arrival_s
        + final_row_stop_s,
        atol=1e-9,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Macro cumulative clock consistency check failed."
        )

    if not np.isclose(
        micro_final_race_s,
        micro_final_arrival_s
        + final_row_stop_s,
        atol=1e-9,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Micro cumulative clock consistency check failed."
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

        "total_aid_stops": (
            total_aid_stops
        ),

        "total_aid_stop_s": (
            total_aid_stop_s
        ),

        "total_aid_stop_min": (
            total_aid_stop_s
            / 60.0
        ),

        # ---------------------------------------------------------------------
        # Arrival
        # ---------------------------------------------------------------------

        "macro_final_arrival_s": (
            macro_final_arrival_s
        ),

        "micro_final_arrival_s": (
            micro_final_arrival_s
        ),

        # ---------------------------------------------------------------------
        # Final race clocks.
        # These already include all aid-station stops.
        # ---------------------------------------------------------------------

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
        # Macro clipping
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
# Formatting helper
# =============================================================================

def format_seconds(
    seconds: float,
) -> str:
    """
    Format seconds as HH:MM:SS.
    """

    if not np.isfinite(
        seconds
    ):

        return "—"

    total_seconds = max(
        0,
        int(
            round(
                float(seconds)
            )
        ),
    )

    hours = (
        total_seconds
        // 3600
    )

    minutes = (
        total_seconds
        % 3600
    ) // 60

    remaining_seconds = (
        total_seconds
        % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


# =============================================================================
# Compact simulation summary
# =============================================================================

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

    macro_race_s = float(
        final_row[
            "macro_departure_time_s"
        ]
    )

    micro_race_s = float(
        final_row[
            "micro_departure_time_s"
        ]
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
        .abs()
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
    Extract arrival/departure predictions at aid stations.
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
            "aid_station_distance_m",
            "distance_from_start_m",
            "aid_station_stop_min",
            "aid_station_stop_s",
            "macro_arrival_time_s",
            "micro_arrival_time_s",
            "macro_departure_time_s",
            "micro_departure_time_s",
            "micro_minus_macro_arrival_s",
        ]
    ].reset_index(
        drop=True
    )
