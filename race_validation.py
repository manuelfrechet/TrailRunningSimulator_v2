from __future__ import annotations

from typing import Any

import fitdecode
import numpy as np
import pandas as pd


# =============================================================================
# Temporary race validation module
# =============================================================================
#
# Purpose:
#
#   Compare an already completed race prediction against the actual FIT
#   recorded for that same race.
#
# IMPORTANT DEVELOPMENT RULE:
#
#   The actual validation FIT is introduced ONLY after the prediction has
#   already been generated.
#
# Therefore this module:
#
#   - does not participate in FIT learning;
#   - does not participate in macro_model2 fitting;
#   - does not modify the simulation;
#   - does not feed actual-race information back into any model.
#
# The actual FIT is used only for post-hoc comparison.
#
# The prediction and actual race are aligned by distance from race start.
#
# Prediction:
#
#   simulation_df
#
# Actual:
#
#   timestamp + distance extracted from the real FIT
#
# Actual cumulative elapsed time is interpolated at the normalized GPX
# segment endpoints.
#
# =============================================================================


# =============================================================================
# FIT parsing
# =============================================================================

def _extract_actual_fit_track(
    uploaded_file,
) -> pd.DataFrame:
    """
    Extract timestamp and distance from an actual race FIT file.

    Distance is normalized so that the first valid FIT point is distance 0.

    Elapsed time starts at the timestamp of the first valid FIT point.

    Stationary periods are naturally retained because elapsed time is derived
    from timestamps rather than from moving time.
    """

    uploaded_file.seek(0)

    rows: list[
        dict[str, Any]
    ] = []

    with fitdecode.FitReader(
        uploaded_file
    ) as fit:

        for frame in fit:

            if not isinstance(
                frame,
                fitdecode.records.FitDataMessage,
            ):
                continue

            if frame.name != "record":
                continue

            timestamp = None
            distance = None
            enhanced_distance = None

            for field in frame.fields:

                if field.name == "timestamp":

                    timestamp = (
                        field.value
                    )

                elif field.name == "distance":

                    distance = (
                        field.value
                    )

                elif field.name == "enhanced_distance":

                    enhanced_distance = (
                        field.value
                    )

            selected_distance = (
                distance
                if distance is not None
                else enhanced_distance
            )

            if (
                timestamp is None
                or selected_distance is None
            ):
                continue

            rows.append(
                {
                    "timestamp": timestamp,
                    "distance_m_raw": (
                        selected_distance
                    ),
                }
            )

    df = pd.DataFrame(
        rows
    )

    if df.empty:

        raise ValueError(
            "The actual FIT file contains no usable "
            "timestamp/distance records."
        )

    df[
        "timestamp"
    ] = pd.to_datetime(
        df[
            "timestamp"
        ],
        errors="coerce",
    )

    df[
        "distance_m_raw"
    ] = pd.to_numeric(
        df[
            "distance_m_raw"
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "distance_m_raw",
        ]
    ).copy()

    if df.empty:

        raise ValueError(
            "The actual FIT file contains no valid "
            "timestamp/distance records."
        )

    df = (
        df.sort_values(
            "timestamp",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # Normalize distance to race start.
    # -------------------------------------------------------------------------

    first_distance_m = float(
        df[
            "distance_m_raw"
        ].iloc[0]
    )

    first_timestamp = (
        df[
            "timestamp"
        ].iloc[0]
    )

    df[
        "distance_from_start_m"
    ] = (
        df[
            "distance_m_raw"
        ]
        - first_distance_m
    )

    df[
        "elapsed_time_s"
    ] = (
        (
            df[
                "timestamp"
            ]
            - first_timestamp
        )
        .dt.total_seconds()
    )

    # -------------------------------------------------------------------------
    # Remove invalid elapsed/distance rows.
    # -------------------------------------------------------------------------

    df = df[
        np.isfinite(
            df[
                "distance_from_start_m"
            ]
        )
        & np.isfinite(
            df[
                "elapsed_time_s"
            ]
        )
    ].copy()

    df = df[
        df[
            "distance_from_start_m"
        ]
        >= 0.0
    ].copy()

    df = df[
        df[
            "elapsed_time_s"
        ]
        >= 0.0
    ].copy()

    if df.empty:

        raise ValueError(
            "The actual FIT file contains no valid "
            "normalized race trajectory."
        )

    # -------------------------------------------------------------------------
    # FIT devices can occasionally write multiple records at the same distance
    # while time continues to advance.
    #
    # Keep the LAST timestamp at each distance so that a stationary period
    # recorded at a point is included in cumulative elapsed time.
    # -------------------------------------------------------------------------

    df = (
        df.sort_values(
            [
                "distance_from_start_m",
                "elapsed_time_s",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "distance_from_start_m"
            ],
            keep="last",
        )
        .sort_values(
            "distance_from_start_m",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # Distance must increase strictly after duplicate-distance collapse.
    # -------------------------------------------------------------------------

    if len(df) < 2:

        raise ValueError(
            "The actual FIT file does not contain enough "
            "distinct distance points."
        )

    distance = (
        df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
    )

    if np.any(
        np.diff(
            distance
        )
        <= 0.0
    ):

        raise ValueError(
            "The actual FIT distance trajectory is not strictly increasing."
        )

    return df[
        [
            "distance_from_start_m",
            "elapsed_time_s",
            "timestamp",
        ]
    ].copy()


# =============================================================================
# Actual FIT interpolation
# =============================================================================

def _interpolate_actual_elapsed_time(
    actual_df: pd.DataFrame,
    prediction_distances_m: np.ndarray,
) -> np.ndarray:
    """
    Interpolate actual cumulative elapsed time at prediction distances.
    """

    actual_distance = (
        actual_df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
    )

    actual_elapsed = (
        actual_df[
            "elapsed_time_s"
        ].to_numpy(
            dtype=float
        )
    )

    prediction_distances_m = np.asarray(
        prediction_distances_m,
        dtype=float,
    )

    actual_max_distance = float(
        actual_distance[-1]
    )

    actual_min_distance = float(
        actual_distance[0]
    )

    if np.any(
        prediction_distances_m
        < actual_min_distance
    ):

        raise ValueError(
            "Prediction contains a distance before the "
            "actual FIT starting distance."
        )

    if np.any(
        prediction_distances_m
        > actual_max_distance
    ):

        raise ValueError(
            "The actual FIT is shorter than the predicted "
            "race. Cannot compare the complete prediction."
        )

    return np.interp(
        prediction_distances_m,
        actual_distance,
        actual_elapsed,
    )


# =============================================================================
# Error metrics
# =============================================================================

def _calculate_prediction_metrics(
    actual_segment_time_s: np.ndarray,
    predicted_segment_time_s: np.ndarray,
) -> dict[str, float]:
    """
    Calculate segment-level error metrics.
    """

    error = (
        predicted_segment_time_s
        - actual_segment_time_s
    )

    absolute_error = np.abs(
        error
    )

    mae = float(
        np.mean(
            absolute_error
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    bias = float(
        np.mean(
            error
        )
    )

    return {
        "mae_s": mae,
        "rmse_s": rmse,
        "bias_s": bias,
    }


# =============================================================================
# Main comparison
# =============================================================================

def compare_prediction_to_actual_fit(
    simulation_df: pd.DataFrame,
    actual_fit_file,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Compare the completed simulation against the actual race FIT.

    Returns:

        comparison_df
        summary_df

    comparison_df:
        one row per normalized prediction segment.

    summary_df:
        one row per prediction model.
    """

    if (
        simulation_df is None
        or simulation_df.empty
    ):

        raise ValueError(
            "Simulation results are empty."
        )

    required_prediction_columns = [
        "distance_from_start_m",
        "macro_departure_time_s",
        "micro_departure_time_s",
    ]

    missing = [
        column
        for column in required_prediction_columns
        if column not in simulation_df.columns
    ]

    if missing:

        raise ValueError(
            "Simulation is missing required validation columns: "
            + ", ".join(
                missing
            )
        )

    actual_df = (
        _extract_actual_fit_track(
            actual_fit_file
        )
    )

    prediction = (
        simulation_df[
            required_prediction_columns
        ]
        .copy()
        .sort_values(
            "distance_from_start_m",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    prediction_distances = (
        prediction[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
    )

    actual_cumulative_time_s = (
        _interpolate_actual_elapsed_time(
            actual_df,
            prediction_distances,
        )
    )

    # -------------------------------------------------------------------------
    # Actual segment elapsed time.
    #
    # The first row represents the first normalized GPX segment.
    # -------------------------------------------------------------------------

    actual_segment_time_s = np.diff(
        actual_cumulative_time_s,
        prepend=0.0,
    )

    macro_cumulative_time_s = (
        prediction[
            "macro_departure_time_s"
        ].to_numpy(
            dtype=float
        )
    )

    micro_cumulative_time_s = (
        prediction[
            "micro_departure_time_s"
        ].to_numpy(
            dtype=float
        )
    )

    macro_segment_time_s = np.diff(
        macro_cumulative_time_s,
        prepend=0.0,
    )

    micro_segment_time_s = np.diff(
        micro_cumulative_time_s,
        prepend=0.0,
    )

    comparison_df = pd.DataFrame(
        {
            "distance_from_start_m": (
                prediction_distances
            ),

            "distance_km": (
                prediction_distances
                / 1000.0
            ),

            "actual_cumulative_time_s": (
                actual_cumulative_time_s
            ),

            "macro_cumulative_time_s": (
                macro_cumulative_time_s
            ),

            "micro_cumulative_time_s": (
                micro_cumulative_time_s
            ),

            "actual_segment_time_s": (
                actual_segment_time_s
            ),

            "macro_segment_time_s": (
                macro_segment_time_s
            ),

            "micro_segment_time_s": (
                micro_segment_time_s
            ),
        }
    )

    comparison_df[
        "macro_error_s"
    ] = (
        comparison_df[
            "macro_cumulative_time_s"
        ]
        - comparison_df[
            "actual_cumulative_time_s"
        ]
    )

    comparison_df[
        "micro_error_s"
    ] = (
        comparison_df[
            "micro_cumulative_time_s"
        ]
        - comparison_df[
            "actual_cumulative_time_s"
        ]
    )

    comparison_df[
        "macro_segment_error_s"
    ] = (
        comparison_df[
            "macro_segment_time_s"
        ]
        - comparison_df[
            "actual_segment_time_s"
        ]
    )

    comparison_df[
        "micro_segment_error_s"
    ] = (
        comparison_df[
            "micro_segment_time_s"
        ]
        - comparison_df[
            "actual_segment_time_s"
        ]
    )

    comparison_df[
        "macro_absolute_error_s"
    ] = comparison_df[
        "macro_error_s"
    ].abs()

    comparison_df[
        "micro_absolute_error_s"
    ] = comparison_df[
        "micro_error_s"
    ].abs()

    # -------------------------------------------------------------------------
    # Difference between model predictions.
    # -------------------------------------------------------------------------

    comparison_df[
        "micro_minus_macro_s"
    ] = (
        comparison_df[
            "micro_cumulative_time_s"
        ]
        - comparison_df[
            "macro_cumulative_time_s"
        ]
    )

    # -------------------------------------------------------------------------
    # Section assignment.
    #
    # Ten equal-distance sections for a compact diagnostic.
    # -------------------------------------------------------------------------

    race_distance_m = float(
        comparison_df[
            "distance_from_start_m"
        ].iloc[-1]
    )

    section_edges = np.linspace(
        0.0,
        race_distance_m,
        11,
    )

    comparison_df[
        "course_section"
    ] = pd.cut(
        comparison_df[
            "distance_from_start_m"
        ],
        bins=section_edges,
        labels=False,
        include_lowest=True,
    )

    # =========================================================================
    # Summary
    # =========================================================================

    macro_segment_metrics = (
        _calculate_prediction_metrics(
            actual_segment_time_s,
            macro_segment_time_s,
        )
    )

    micro_segment_metrics = (
        _calculate_prediction_metrics(
            actual_segment_time_s,
            micro_segment_time_s,
        )
    )

    actual_finish_s = float(
        actual_cumulative_time_s[-1]
    )

    macro_finish_s = float(
        macro_cumulative_time_s[-1]
    )

    micro_finish_s = float(
        micro_cumulative_time_s[-1]
    )

    summary_rows = [
        {
            "model": "macro_model",
            "predicted_finish_s": (
                macro_finish_s
            ),
            "actual_finish_s": (
                actual_finish_s
            ),
            "finish_error_s": (
                macro_finish_s
                - actual_finish_s
            ),
            "finish_error_min": (
                (
                    macro_finish_s
                    - actual_finish_s
                )
                / 60.0
            ),
            "segment_mae_s": (
                macro_segment_metrics[
                    "mae_s"
                ]
            ),
            "segment_rmse_s": (
                macro_segment_metrics[
                    "rmse_s"
                ]
            ),
            "segment_bias_s": (
                macro_segment_metrics[
                    "bias_s"
                ]
            ),
        },
        {
            "model": "micro_model",
            "predicted_finish_s": (
                micro_finish_s
            ),
            "actual_finish_s": (
                actual_finish_s
            ),
            "finish_error_s": (
                micro_finish_s
                - actual_finish_s
            ),
            "finish_error_min": (
                (
                    micro_finish_s
                    - actual_finish_s
                )
                / 60.0
            ),
            "segment_mae_s": (
                micro_segment_metrics[
                    "mae_s"
                ]
            ),
            "segment_rmse_s": (
                micro_segment_metrics[
                    "rmse_s"
                ]
            ),
            "segment_bias_s": (
                micro_segment_metrics[
                    "bias_s"
                ]
            ),
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    return (
        comparison_df,
        summary_df,
    )


# =============================================================================
# Course-section summary
# =============================================================================

def build_validation_section_summary(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate prediction errors by course section.
    """

    if (
        comparison_df is None
        or comparison_df.empty
    ):

        return pd.DataFrame()

    rows: list[
        dict[str, Any]
    ] = []

    for section, group in comparison_df.groupby(
        "course_section",
        dropna=True,
    ):

        section_number = int(
            section
        ) + 1

        rows.append(
            {
                "section": section_number,

                "start_km": float(
                    group[
                        "distance_km"
                    ].min()
                ),

                "end_km": float(
                    group[
                        "distance_km"
                    ].max()
                ),

                "actual_time_s": float(
                    group[
                        "actual_segment_time_s"
                    ].sum()
                ),

                "macro_time_s": float(
                    group[
                        "macro_segment_time_s"
                    ].sum()
                ),

                "micro_time_s": float(
                    group[
                        "micro_segment_time_s"
                    ].sum()
                ),

                "macro_bias_s": float(
                    group[
                        "macro_segment_error_s"
                    ].mean()
                ),

                "micro_bias_s": float(
                    group[
                        "micro_segment_error_s"
                    ].mean()
                ),

                "macro_cumulative_error_s": float(
                    group[
                        "macro_error_s"
                    ].iloc[-1]
                ),

                "micro_cumulative_error_s": float(
                    group[
                        "micro_error_s"
                    ].iloc[-1]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Validation summary helper
# =============================================================================

def summarize_validation(
    summary_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Convert the validation summary into a compact dictionary.
    """

    if (
        summary_df is None
        or summary_df.empty
    ):

        return {}

    result: dict[str, Any] = {}

    for _, row in summary_df.iterrows():

        model_name = str(
            row[
                "model"
            ]
        )

        result[
            model_name
        ] = {
            "predicted_finish_s": float(
                row[
                    "predicted_finish_s"
                ]
            ),
            "actual_finish_s": float(
                row[
                    "actual_finish_s"
                ]
            ),
            "finish_error_s": float(
                row[
                    "finish_error_s"
                ]
            ),
            "finish_error_min": float(
                row[
                    "finish_error_min"
                ]
            ),
            "segment_mae_s": float(
                row[
                    "segment_mae_s"
                ]
            ),
            "segment_rmse_s": float(
                row[
                    "segment_rmse_s"
                ]
            ),
            "segment_bias_s": float(
                row[
                    "segment_bias_s"
                ]
            ),
        }

    return result
  
