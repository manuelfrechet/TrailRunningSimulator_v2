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
# The GPX profile contains one row per normalized segment endpoint.
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
# For each segment:
#
#     1. determine the state at the segment start;
#     2. predict the segment with the macro model;
#     3. predict the segment with the micro model;
#     4. advance the macro and micro clocks;
#     5. apply any aid-station stop to BOTH clocks;
#     6. continue to the next segment.
#
# The micro elapsed_time input is the CURRENT SIMULATED MICRO TIME.
#
# The macro model remains independent of micro.
# The micro model remains independent of macro.
#
# Aid-station time is applied to both cumulative clocks because the user
# explicitly defined aid stops as affecting both simulation paths.
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
# Validation
# =============================================================================

def _validate_profile(
    profile_df: pd.DataFrame,
) -> None:
    """
    Validate the normalized GPX profile before simulation.
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

    # -------------------------------------------------------------------------
    # The first row represents the end of the first segment.
    # -------------------------------------------------------------------------

    expected_first_endpoint = (
        GPX_SEGMENT_LENGTH_M
    )

    if not np.isclose(
        distance[0],
        expected_first_endpoint,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            "GPX profile does not match config.py. "
            f"Expected first segment endpoint at "
            f"{GPX_SEGMENT_LENGTH_M:.2f} m, "
            f"found {distance[0]:.2f} m."
        )

    # -------------------------------------------------------------------------
    # Every subsequent endpoint must advance by the configured segment length.
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
# Segment-start state
# =============================================================================

def _get_segment_start_state(
    profile_df: pd.DataFrame,
    row_index: int,
) -> dict[str, float]:
    """
    Return the terrain state at the START of one GPX segment.

    The profile stores segment END rows.

    Example:

        profile row at 100 m = segment 0 -> 100

    Therefore its start state is:

        distance = 0
        cumulative ascent = 0
        cumulative descent = 0
    """

    if row_index == 0:
        return {
            "distance_from_start_m": 0.0,
            "cumulative_ascent_m": 0.0,
            "cumulative_descent_m": 0.0,
        }

    previous_row = profile_df.iloc[
        row_index - 1
    ]

    return {
        "distance_from_start_m": float(
            previous_row[
                "distance_from_start_m"
            ]
        ),
        "cumulative_ascent_m": float(
            previous_row[
                "cumulative_ascent_m"
            ]
        ),
        "cumulative_descent_m": float(
            previous_row[
                "cumulative_descent_m"
            ]
        ),
    }


# =============================================================================
# Macro segment prediction
# =============================================================================

def _predict_macro_segment(
    macro_model: MacroModel,
    start_state: dict[str, float],
    end_row: pd.Series,
) -> float:
    """
    Predict one GPX segment with the macro model.

    The macro model predicts cumulative running time:

        M(distance, cumulative_ascent, cumulative_descent)

    The segment prediction is therefore:

        M(end)
        -
        M(start)
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

    segment_time = float(
        end_prediction[0]
        - start_prediction[0]
    )

    if not np.isfinite(
        segment_time
    ):
        raise ValueError(
            "Macro model returned an invalid segment time."
        )

    if segment_time < 0.0:
        raise ValueError(
            "Macro model returned a negative segment time "
            f"at distance {float(end_row['distance_from_start_m']):.2f} m."
        )

    return segment_time


# =============================================================================
# Micro segment prediction
# =============================================================================

def _predict_micro_segment(
    micro_model: MicroModel,
    start_state: dict[str, float],
    current_micro_elapsed_s: float,
    end_row: pd.Series,
) -> dict[str, Any]:
    """
    Predict one GPX segment with the micro analogue model.

    The elapsed_time_s input is the simulated micro elapsed time at the
    beginning of the segment.

    This is important because elapsed time is one of the seven micro
    similarity dimensions.
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

    predicted_time = float(
        prediction[
            "micro_predicted_time_s"
        ]
    )

    if not np.isfinite(
        predicted_time
    ):
        raise ValueError(
            "Micro model returned an invalid segment time."
        )

    if predicted_time < 0.0:
        raise ValueError(
            "Micro model returned a negative segment time "
            f"at distance {float(end_row['distance_from_start_m']):.2f} m."
        )

    return prediction


# =============================================================================
# Aid-station information
# =============================================================================

def _get_aid_station_info(
    row: pd.Series,
) -> tuple[str, float]:
    """
    Extract aid-station name and stop duration from one GPX profile row.

    No aid station means:

        name = ""
        duration = 0
    """

    name = ""

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
            name = str(
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
        name,
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

    The simulation advances exactly one GPX_SEGMENT_LENGTH_M at a time.

    Macro and micro maintain separate cumulative clocks.

    At an aid station:

        arrival
            ↓
        apply stop to BOTH clocks
            ↓
        next segment

    Returns:
        simulation_df
        race_summary
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

    # -------------------------------------------------------------------------
    # Work on a clean, ordered copy.
    # -------------------------------------------------------------------------

    profile = (
        gpx_profile_df
        .copy()
        .sort_values(
            "distance_from_start_m",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # Independent simulation clocks.
    #
    # These include aid-station time once a station has been visited.
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

        # ---------------------------------------------------------------------
        # Macro prediction.
        # ---------------------------------------------------------------------

        macro_segment_time_s = (
            _predict_macro_segment(
                macro_model=macro_model,
                start_state=start_state,
                end_row=end_row,
            )
        )

        # ---------------------------------------------------------------------
        # Micro prediction.
        #
        # The current MICRO clock is supplied as elapsed_time_s.
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Arrival at segment end.
        #
        # These are the predicted arrival clocks BEFORE any aid-station stop.
        # ---------------------------------------------------------------------

        macro_arrival_s = (
            macro_elapsed_s
            + macro_segment_time_s
        )

        micro_arrival_s = (
            micro_elapsed_s
            + micro_segment_time_s
        )

        # ---------------------------------------------------------------------
        # Aid station at this segment endpoint.
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Differences BEFORE the aid stop.
        #
        # This isolates the model prediction difference itself.
        # ---------------------------------------------------------------------

        segment_difference_s = (
            micro_segment_time_s
            - macro_segment_time_s
        )

        arrival_difference_s = (
            micro_arrival_s
            - macro_arrival_s
        )

        # ---------------------------------------------------------------------
        # Departure clocks AFTER the aid station.
        # ---------------------------------------------------------------------

        macro_departure_s = (
            macro_arrival_s
            + aid_stop_s
        )

        micro_departure_s = (
            micro_arrival_s
            + aid_stop_s
        )

        # ---------------------------------------------------------------------
        # Cumulative difference after the stop remains unchanged because the
        # same aid time is added to both clocks.
        # ---------------------------------------------------------------------

        post_stop_difference_s = (
            micro_departure_s
            - macro_departure_s
        )

        # ---------------------------------------------------------------------
        # Save the row.
        # ---------------------------------------------------------------------

        rows.append(
            {
                # -------------------------------------------------------------
                # Terrain state
                # -------------------------------------------------------------

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

                # -------------------------------------------------------------
                # Segment prediction
                # -------------------------------------------------------------

                "macro_predicted_time_s": (
                    macro_segment_time_s
                ),

                "micro_predicted_time_s": (
                    micro_segment_time_s
                ),

                "micro_minus_macro_s": (
                    segment_difference_s
                ),

                # -------------------------------------------------------------
                # Cumulative arrival time
                # -------------------------------------------------------------

                "macro_arrival_time_s": (
                    macro_arrival_s
                ),

                "micro_arrival_time_s": (
                    micro_arrival_s
                ),

                "micro_minus_macro_arrival_s": (
                    arrival_difference_s
                ),

                # -------------------------------------------------------------
                # Aid station
                # -------------------------------------------------------------

                "aid_station_name": (
                    aid_station_name
                ),

                "aid_station_stop_min": (
                    aid_stop_minutes
                ),

                "aid_station_stop_s": (
                    aid_stop_s
                ),

                # -------------------------------------------------------------
                # Cumulative departure time after aid station
                # -------------------------------------------------------------

                "macro_departure_time_s": (
                    macro_departure_s
                ),

                "micro_departure_time_s": (
                    micro_departure_s
                ),

                "micro_minus_macro_departure_s": (
                    post_stop_difference_s
                ),

                # -------------------------------------------------------------
                # Micro analogue diagnostics
                # -------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Update clocks.
        #
        # Aid time affects BOTH simulation paths.
        # ---------------------------------------------------------------------

        macro_elapsed_s = (
            macro_departure_s
        )

        micro_elapsed_s = (
            micro_departure_s
        )

    # -------------------------------------------------------------------------
    # Final dataframe.
    # -------------------------------------------------------------------------

    simulation_df = pd.DataFrame(
        rows
    )

    if simulation_df.empty:
        raise ValueError(
            "Simulation produced no predictions."
        )

    # -------------------------------------------------------------------------
    # Race summary.
    # -------------------------------------------------------------------------

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

    macro_final_race_s = float(
        macro_final_arrival_s
        + total_aid_stop_s
    )

    micro_final_race_s = float(
        micro_final_arrival_s
        + total_aid_stop_s
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
                ] > 0.0
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
    }

    return (
        simulation_df,
        race_summary,
    )


# =============================================================================
# Convenience helpers
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
    Build a compact summary directly from a simulation dataframe.
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
            "macro_arrival_time_s"
        ]
    ) + total_aid_stop_s

    micro_race_s = float(
        final_row[
            "micro_arrival_time_s"
        ]
    ) + total_aid_stop_s

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
        "macro_race_time": format_seconds(
            macro_race_s
        ),
        "micro_race_time": format_seconds(
            micro_race_s
        ),
        "micro_minus_macro_min": (
            (
                micro_race_s
                - macro_race_s
            )
            / 60.0
        ),
    }
