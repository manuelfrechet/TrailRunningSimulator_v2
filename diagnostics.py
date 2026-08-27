from __future__ import annotations

from typing import Any

import fitdecode
import numpy as np
import pandas as pd

from config import GPX_SEGMENT_LENGTH_M


# =============================================================================
# Diagnostics configuration
# =============================================================================

RANDOM_SAMPLE_SIZE = 100
EXTREME_SAMPLE_SIZE = 20
RANDOM_STATE = 42

DISTANCE_TOLERANCE_M = 0.01


# =============================================================================
# Basic learning-dataset diagnostics
# =============================================================================

def build_learning_diagnostic_sample(
    learning_df: pd.DataFrame,
    random_sample_size: int = RANDOM_SAMPLE_SIZE,
    extreme_sample_size: int = 100,
) -> pd.DataFrame:
    if learning_df is None or learning_df.empty:
        return pd.DataFrame()

    df = learning_df.copy()

    if "actual_segment_time_s" not in df.columns:
        raise ValueError(
            "Learning dataset must contain 'actual_segment_time_s'."
        )

    df["actual_segment_time_s"] = pd.to_numeric(
        df["actual_segment_time_s"],
        errors="coerce",
    )

    first_rows = df.head(100).copy()

    first_rows.insert(
        0,
        "diagnostic_group",
        "first_rows",
    )

    random_rows = df.sample(
        n=min(
            random_sample_size,
            len(df),
        ),
        random_state=RANDOM_STATE,
    ).copy()

    random_rows.insert(
        0,
        "diagnostic_group",
        "random_sample",
    )

    slow_rows = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=False,
            kind="mergesort",
        )
        .head(
            extreme_sample_size
        )
        .copy()
    )

    slow_rows.insert(
        0,
        "diagnostic_group",
        "slowest",
    )

    fast_rows = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=True,
            kind="mergesort",
        )
        .head(
            extreme_sample_size
        )
        .copy()
    )

    fast_rows.insert(
        0,
        "diagnostic_group",
        "fastest",
    )

    return pd.concat(
        [
            first_rows,
            random_rows,
            slow_rows,
            fast_rows,
        ],
        ignore_index=True,
    )


def build_activity_learning_summary(
    learning_df: pd.DataFrame,
) -> pd.DataFrame:
    if learning_df is None or learning_df.empty:
        return pd.DataFrame()

    required = [
        "activity_id",
        "activity_name",
        "distance_from_start_m",
        "actual_segment_time_s",
    ]

    missing = [
        col
        for col in required
        if col not in learning_df.columns
    ]

    if missing:
        raise ValueError(
            "Learning diagnostics missing columns: "
            + ", ".join(missing)
        )

    df = learning_df.copy()

    df["actual_segment_time_s"] = pd.to_numeric(
        df["actual_segment_time_s"],
        errors="coerce",
    )

    df["distance_from_start_m"] = pd.to_numeric(
        df["distance_from_start_m"],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []

    for (
        activity_id,
        activity_name,
    ), group in df.groupby(
        [
            "activity_id",
            "activity_name",
        ],
        sort=True,
    ):

        durations = group[
            "actual_segment_time_s"
        ].dropna()

        if durations.empty:
            continue

        fastest = float(
            durations.min()
        )

        slowest = float(
            durations.max()
        )

        fastest_speed_m_s = (
            GPX_SEGMENT_LENGTH_M / fastest
            if fastest > 0.0
            else np.nan
        )

        slowest_speed_m_s = (
            GPX_SEGMENT_LENGTH_M / slowest
            if slowest > 0.0
            else np.nan
        )

        rows.append(
            {
                "activity_id": activity_id,
                "activity_name": activity_name,
                "transitions": int(
                    len(group)
                ),
                "race_distance_m": float(
                    group[
                        "distance_from_start_m"
                    ].max()
                    + GPX_SEGMENT_LENGTH_M
                ),
                "segment_length_m": (
                    GPX_SEGMENT_LENGTH_M
                ),
                "median_segment_time_s": float(
                    durations.median()
                ),
                "mean_segment_time_s": float(
                    durations.mean()
                ),
                "fastest_segment_time_s": fastest,
                "fastest_implied_speed_m_s": (
                    fastest_speed_m_s
                ),
                "fastest_implied_speed_kmh": (
                    fastest_speed_m_s * 3.6
                ),
                "slowest_segment_time_s": slowest,
                "slowest_implied_speed_m_s": (
                    slowest_speed_m_s
                ),
                "slowest_implied_speed_kmh": (
                    slowest_speed_m_s * 3.6
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_extreme_transition_summary(
    learning_df: pd.DataFrame,
    n_each: int = EXTREME_SAMPLE_SIZE,
) -> pd.DataFrame:
    if learning_df is None or learning_df.empty:
        return pd.DataFrame()

    required = [
        "activity_id",
        "activity_name",
        "distance_from_start_m",
        "cumulative_ascent_m",
        "cumulative_descent_m",
        "elapsed_time_s",
        "segment_ascent_m",
        "segment_descent_m",
        "segment_grade_pct",
        "actual_segment_time_s",
    ]

    missing = [
        col
        for col in required
        if col not in learning_df.columns
    ]

    if missing:
        raise ValueError(
            "Extreme-transition diagnostics missing columns: "
            + ", ".join(missing)
        )

    df = learning_df[
        required
    ].copy()

    df["actual_segment_time_s"] = pd.to_numeric(
        df["actual_segment_time_s"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "actual_segment_time_s"
        ]
    )

    df = df[
        df["actual_segment_time_s"] > 0.0
    ].copy()

    df["implied_speed_m_s"] = (
        GPX_SEGMENT_LENGTH_M
        / df["actual_segment_time_s"]
    )

    df["implied_speed_kmh"] = (
        df["implied_speed_m_s"]
        * 3.6
    )

    fastest = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=True,
            kind="mergesort",
        )
        .head(
            n_each
        )
        .copy()
    )

    fastest.insert(
        0,
        "extreme_type",
        "fastest",
    )

    slowest = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=False,
            kind="mergesort",
        )
        .head(
            n_each
        )
        .copy()
    )

    slowest.insert(
        0,
        "extreme_type",
        "slowest",
    )

    return pd.concat(
        [
            fastest,
            slowest,
        ],
        ignore_index=True,
    )


# =============================================================================
# Raw FIT stationary-time diagnostics
# =============================================================================

def _extract_raw_time_distance(
    uploaded_file,
) -> pd.DataFrame:
    uploaded_file.seek(0)

    rows: list[dict[str, Any]] = []

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
                    timestamp = field.value

                elif field.name == "distance":
                    distance = field.value

                elif field.name == "enhanced_distance":
                    enhanced_distance = field.value

            selected_distance = (
                distance
                if distance is not None
                else enhanced_distance
            )

            rows.append(
                {
                    "timestamp": timestamp,
                    "distance_m": selected_distance,
                }
            )

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["distance_m"] = pd.to_numeric(
        df["distance_m"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "distance_m",
        ]
    ).copy()

    if df.empty:
        return pd.DataFrame()

    return (
        df.sort_values(
            "timestamp",
            kind="mergesort",
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


def _build_stationary_intervals(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    if len(raw_df) < 2:
        return pd.DataFrame()

    timestamps = raw_df[
        "timestamp"
    ]

    distances = raw_df[
        "distance_m"
    ].to_numpy(
        dtype=float
    )

    delta_time_s = (
        timestamps.diff()
        .dt.total_seconds()
        .to_numpy(
            dtype=float
        )
    )

    delta_distance_m = np.diff(
        distances,
        prepend=np.nan,
    )

    valid_time = (
        np.isfinite(
            delta_time_s
        )
        & (
            delta_time_s > 0.0
        )
    )

    stationary = (
        valid_time
        & np.isfinite(
            delta_distance_m
        )
        & (
            np.abs(
                delta_distance_m
            )
            <= DISTANCE_TOLERANCE_M
        )
    )

    interval_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "distance_m": distances,
            "delta_time_s": delta_time_s,
            "delta_distance_m": delta_distance_m,
            "is_stationary": stationary,
        }
    )

    interval_df[
        "stationary_time_s"
    ] = np.where(
        stationary,
        delta_time_s,
        0.0,
    )

    interval_df[
        "implied_speed_m_s"
    ] = np.where(
        valid_time,
        delta_distance_m
        / delta_time_s,
        np.nan,
    )

    return interval_df


def analyze_fit_stops(
    uploaded_file,
    activity_id: int,
    activity_name: str,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:

    raw_df = _extract_raw_time_distance(
        uploaded_file
    )

    if raw_df.empty:

        return (
            {
                "activity_id": activity_id,
                "activity_name": activity_name,
                "raw_records": 0,
                "elapsed_time_s": np.nan,
                "distance_m": np.nan,
                "stationary_intervals": 0,
                "stationary_time_s": np.nan,
                "stationary_time_min": np.nan,
                "stationary_fraction_pct": np.nan,
            },
            pd.DataFrame(),
        )

    intervals = _build_stationary_intervals(
        raw_df
    )

    elapsed_time_s = float(
        (
            raw_df[
                "timestamp"
            ].iloc[-1]
            - raw_df[
                "timestamp"
            ].iloc[0]
        ).total_seconds()
    )

    distance_m = float(
        raw_df[
            "distance_m"
        ].iloc[-1]
        - raw_df[
            "distance_m"
        ].iloc[0]
    )

    stationary_time_s = (
        float(
            intervals[
                "stationary_time_s"
            ].sum()
        )
        if not intervals.empty
        else 0.0
    )

    stationary_intervals = (
        int(
            intervals[
                "is_stationary"
            ].sum()
        )
        if not intervals.empty
        else 0
    )

    stationary_fraction = (
        stationary_time_s
        / elapsed_time_s
        if elapsed_time_s > 0.0
        else np.nan
    )

    summary = {
        "activity_id": activity_id,
        "activity_name": activity_name,
        "raw_records": int(
            len(raw_df)
        ),
        "elapsed_time_s": elapsed_time_s,
        "distance_m": distance_m,
        "stationary_intervals": (
            stationary_intervals
        ),
        "stationary_time_s": (
            stationary_time_s
        ),
        "stationary_time_min": (
            stationary_time_s
            / 60.0
        ),
        "stationary_fraction_pct": (
            stationary_fraction * 100.0
            if np.isfinite(
                stationary_fraction
            )
            else np.nan
        ),
    }

    if not intervals.empty:

        intervals.insert(
            0,
            "activity_id",
            activity_id,
        )

        intervals.insert(
            1,
            "activity_name",
            activity_name,
        )

    return (
        summary,
        intervals,
    )


def analyze_uploaded_fit_stops(
    uploaded_files,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not uploaded_files:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    summaries: list[
        dict[str, Any]
    ] = []

    interval_frames: list[
        pd.DataFrame
    ] = []

    for activity_id, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):

        activity_name = getattr(
            uploaded_file,
            "name",
            f"activity_{activity_id}",
        )

        summary, intervals = (
            analyze_fit_stops(
                uploaded_file,
                activity_id,
                activity_name,
            )
        )

        summaries.append(
            summary
        )

        if not intervals.empty:

            interval_frames.append(
                intervals
            )

    summary_df = pd.DataFrame(
        summaries
    )

    if interval_frames:

        intervals_df = pd.concat(
            interval_frames,
            ignore_index=True,
        )

    else:

        intervals_df = pd.DataFrame()

    return (
        summary_df,
        intervals_df,
    )


# =============================================================================
# Macro diagnostics
# =============================================================================

def build_activity_macro_summary(
    learning_df: pd.DataFrame,
    macro_model,
) -> pd.DataFrame:

    if (
        learning_df is None
        or learning_df.empty
    ):
        return pd.DataFrame()

    if macro_model is None:
        raise ValueError(
            "macro_model is required."
        )

    required = [
        "activity_id",
        "activity_name",
        "distance_from_start_m",
        "cumulative_ascent_m",
        "cumulative_descent_m",
        "elapsed_time_s",
    ]

    missing = [
        col
        for col in required
        if col not in learning_df.columns
    ]

    if missing:
        raise ValueError(
            "Macro diagnostics missing columns: "
            + ", ".join(missing)
        )

    rows: list[dict[str, Any]] = []

    for (
        activity_id,
        activity_name,
    ), group in learning_df.groupby(
        [
            "activity_id",
            "activity_name",
        ],
        sort=True,
    ):

        group = (
            group.copy()
            .sort_values(
                "distance_from_start_m",
                kind="mergesort",
            )
            .dropna(
                subset=[
                    "distance_from_start_m",
                    "cumulative_ascent_m",
                    "cumulative_descent_m",
                    "elapsed_time_s",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        if group.empty:
            continue

        predicted = (
            macro_model.predict_cumulative_time(
                distance_from_start_m=(
                    group[
                        "distance_from_start_m"
                    ]
                ),
                cumulative_ascent_m=(
                    group[
                        "cumulative_ascent_m"
                    ]
                ),
                cumulative_descent_m=(
                    group[
                        "cumulative_descent_m"
                    ]
                ),
            )
        )

        actual = group[
            "elapsed_time_s"
        ].to_numpy(
            dtype=float
        )

        error = (
            predicted
            - actual
        )

        abs_error = np.abs(
            error
        )

        middle_index = (
            len(group) // 2
        )

        end_index = (
            len(group) - 1
        )

        rows.append(
            {
                "activity_id": activity_id,
                "activity_name": activity_name,
                "rows": int(
                    len(group)
                ),
                "distance_m": float(
                    group[
                        "distance_from_start_m"
                    ].iloc[
                        end_index
                    ]
                ),
                "cumulative_ascent_m": float(
                    group[
                        "cumulative_ascent_m"
                    ].iloc[
                        end_index
                    ]
                ),
                "cumulative_descent_m": float(
                    group[
                        "cumulative_descent_m"
                    ].iloc[
                        end_index
                    ]
                ),
                "mean_absolute_error_s": float(
                    np.mean(
                        abs_error
                    )
                ),
                "median_absolute_error_s": float(
                    np.median(
                        abs_error
                    )
                ),
                "mean_bias_s": float(
                    np.mean(
                        error
                    )
                ),
                "start_actual_time_s": float(
                    actual[0]
                ),
                "start_macro_time_s": float(
                    predicted[0]
                ),
                "start_error_s": float(
                    error[0]
                ),
                "middle_distance_m": float(
                    group[
                        "distance_from_start_m"
                    ].iloc[
                        middle_index
                    ]
                ),
                "middle_actual_time_s": float(
                    actual[
                        middle_index
                    ]
                ),
                "middle_macro_time_s": float(
                    predicted[
                        middle_index
                    ]
                ),
                "middle_error_s": float(
                    error[
                        middle_index
                    ]
                ),
                "end_actual_time_s": float(
                    actual[
                        end_index
                    ]
                ),
                "end_macro_time_s": float(
                    predicted[
                        end_index
                    ]
                ),
                "end_error_s": float(
                    error[
                        end_index
                    ]
                ),
                "end_error_min": float(
                    error[
                        end_index
                    ]
                    / 60.0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_macro_historical_check(
    learning_df: pd.DataFrame,
    macro_model,
) -> pd.DataFrame:

    if (
        learning_df is None
        or learning_df.empty
    ):
        return pd.DataFrame()

    required = [
        "activity_id",
        "activity_name",
        "distance_from_start_m",
        "cumulative_ascent_m",
        "cumulative_descent_m",
        "elapsed_time_s",
    ]

    missing = [
        col
        for col in required
        if col not in learning_df.columns
    ]

    if missing:
        raise ValueError(
            "Macro check missing columns: "
            + ", ".join(missing)
        )

    base = learning_df[
        required
    ].copy()

    predicted = (
        macro_model.predict_cumulative_time(
            base[
                "distance_from_start_m"
            ],
            base[
                "cumulative_ascent_m"
            ],
            base[
                "cumulative_descent_m"
            ],
        )
    )

    base[
        "macro_predicted_cumulative_time_s"
    ] = predicted

    base[
        "macro_error_s"
    ] = (
        base[
            "macro_predicted_cumulative_time_s"
        ]
        - base[
            "elapsed_time_s"
        ]
    )

    base[
        "macro_error_abs_s"
    ] = base[
        "macro_error_s"
    ].abs()

    return base


def compare_stops_with_macro(
    stop_summary_df: pd.DataFrame,
    activity_macro_summary_df: pd.DataFrame,
) -> pd.DataFrame:

    if (
        stop_summary_df is None
        or stop_summary_df.empty
        or activity_macro_summary_df is None
        or activity_macro_summary_df.empty
    ):
        return pd.DataFrame()

    comparison = stop_summary_df.merge(
        activity_macro_summary_df,
        on=[
            "activity_id",
            "activity_name",
        ],
        how="inner",
    )

    if comparison.empty:
        return pd.DataFrame()

    comparison[
        "macro_underprediction_s"
    ] = (
        -comparison[
            "end_error_s"
        ]
    )

    comparison[
        "macro_underprediction_min"
    ] = (
        comparison[
            "macro_underprediction_s"
        ]
        / 60.0
    )

    comparison[
        "stationary_minus_macro_underprediction_s"
    ] = (
        comparison[
            "stationary_time_s"
        ]
        - comparison[
            "macro_underprediction_s"
        ]
    )

    comparison[
        "stationary_minus_macro_underprediction_min"
    ] = (
        comparison[
            "stationary_minus_macro_underprediction_s"
        ]
        / 60.0
    )

    return comparison


# =============================================================================
# Historical leave-one-FIT-out diagnostics
# =============================================================================

def _select_non_overlapping_test_rows(
    activity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select historical rows at multiples of GPX_SEGMENT_LENGTH_M.

    The underlying learning database remains densely sampled according to
    LEARNING_STEP_M.

    This function creates the non-overlapping validation trajectory only.
    """
    if (
        activity_df is None
        or activity_df.empty
    ):
        return pd.DataFrame()

    required = [
        "distance_from_start_m",
        "cumulative_ascent_m",
        "cumulative_descent_m",
        "elapsed_time_s",
        "segment_ascent_m",
        "segment_descent_m",
        "segment_grade_pct",
        "actual_segment_time_s",
    ]

    missing = [
        col
        for col in required
        if col not in activity_df.columns
    ]

    if missing:
        raise ValueError(
            "Held-out validation missing columns: "
            + ", ".join(missing)
        )

    df = activity_df.copy()

    df[
        "distance_from_start_m"
    ] = pd.to_numeric(
        df[
            "distance_from_start_m"
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "distance_from_start_m"
        ]
    ).copy()

    distances = (
        df[
            "distance_from_start_m"
        ].round()
    )

    is_multiple = np.isclose(
        np.mod(
            distances,
            GPX_SEGMENT_LENGTH_M,
        ),
        0.0,
        atol=1e-9,
    )

    return (
        df[
            is_multiple
        ]
        .sort_values(
            "distance_from_start_m",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


def _predict_held_out_activity_micro(
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:

    from micro_model import fit_micro_model

    model = fit_micro_model(
        training_df
    )

    query_df = test_df[
        [
            "distance_from_start_m",
            "cumulative_ascent_m",
            "cumulative_descent_m",
            "elapsed_time_s",
            "segment_ascent_m",
            "segment_descent_m",
            "segment_grade_pct",
        ]
    ].copy()

    prediction_df = model.predict(
        query_df
    )

    result = test_df[
        [
            "activity_id",
            "activity_name",
            "distance_from_start_m",
            "elapsed_time_s",
            "actual_segment_time_s",
        ]
    ].reset_index(
        drop=True
    ).copy()

    result[
        "micro_predicted_time_s"
    ] = prediction_df[
        "micro_predicted_time_s"
    ].to_numpy()

    result[
        "micro_error_s"
    ] = (
        result[
            "micro_predicted_time_s"
        ]
        - result[
            "actual_segment_time_s"
        ]
    )

    result[
        "micro_abs_error_s"
    ] = result[
        "micro_error_s"
    ].abs()

    for column in [
        "analogue_1_distance",
        "analogue_1_time_s",
        "analogue_1_activity_id",
        "analogue_1_activity_name",
        "analogue_1_distance_from_start_m",
        "analogue_2_distance",
        "analogue_2_time_s",
        "analogue_2_activity_id",
        "analogue_2_activity_name",
        "analogue_2_distance_from_start_m",
    ]:

        if column in prediction_df.columns:

            result[column] = prediction_df[
                column
            ].to_numpy()

    return result


def _predict_held_out_activity_macro(
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:

    from macro_model import fit_macro_model

    model = fit_macro_model(
        training_df
    )

    current_distance = (
        test_df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
    )

    current_ascent = (
        test_df[
            "cumulative_ascent_m"
        ].to_numpy(
            dtype=float
        )
    )

    current_descent = (
        test_df[
            "cumulative_descent_m"
        ].to_numpy(
            dtype=float
        )
    )

    future_ascent = (
        current_ascent
        + test_df[
            "segment_ascent_m"
        ].to_numpy(
            dtype=float
        )
    )

    future_descent = (
        current_descent
        + test_df[
            "segment_descent_m"
        ].to_numpy(
            dtype=float
        )
    )

    future_distance = (
        current_distance
        + GPX_SEGMENT_LENGTH_M
    )

    current_macro = (
        model.predict_cumulative_time(
            current_distance,
            current_ascent,
            current_descent,
        )
    )

    future_macro = (
        model.predict_cumulative_time(
            future_distance,
            future_ascent,
            future_descent,
        )
    )

    macro_segment = (
        future_macro
        - current_macro
    )

    result = test_df[
        [
            "activity_id",
            "activity_name",
            "distance_from_start_m",
            "actual_segment_time_s",
        ]
    ].reset_index(
        drop=True
    ).copy()

    result[
        "macro_predicted_time_s"
    ] = macro_segment

    result[
        "macro_error_s"
    ] = (
        result[
            "macro_predicted_time_s"
        ]
        - result[
            "actual_segment_time_s"
        ]
    )

    result[
        "macro_abs_error_s"
    ] = result[
        "macro_error_s"
    ].abs()

    return result


def build_leave_one_activity_out_validation(
    learning_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Development-only leave-one-FIT-out validation.

    Training:
        all dense historical transitions from every other FIT.

    Test:
        non-overlapping GPX_SEGMENT_LENGTH_M transitions from the held-out FIT.
    """
    if (
        learning_df is None
        or learning_df.empty
    ):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    required = [
        "activity_id",
        "activity_name",
        "distance_from_start_m",
        "cumulative_ascent_m",
        "cumulative_descent_m",
        "elapsed_time_s",
        "segment_ascent_m",
        "segment_descent_m",
        "segment_grade_pct",
        "actual_segment_time_s",
    ]

    missing = [
        col
        for col in required
        if col not in learning_df.columns
    ]

    if missing:

        raise ValueError(
            "Leave-one-FIT-out validation missing columns: "
            + ", ".join(missing)
        )

    activities = (
        learning_df[
            [
                "activity_id",
                "activity_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "activity_id"
        )
        .reset_index(
            drop=True
        )
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    detail_frames: list[
        pd.DataFrame
    ] = []

    for _, activity in activities.iterrows():

        held_out_id = activity[
            "activity_id"
        ]

        held_out_name = activity[
            "activity_name"
        ]

        training_df = learning_df[
            learning_df[
                "activity_id"
            ]
            != held_out_id
        ].copy()

        held_out_full = learning_df[
            learning_df[
                "activity_id"
            ]
            == held_out_id
        ].copy()

        test_df = (
            _select_non_overlapping_test_rows(
                held_out_full
            )
        )

        if test_df.empty:
            continue

        micro_result = (
            _predict_held_out_activity_micro(
                training_df,
                test_df,
            )
        )

        macro_result = (
            _predict_held_out_activity_macro(
                training_df,
                test_df,
            )
        )

        detail = test_df[
            [
                "activity_id",
                "activity_name",
                "distance_from_start_m",
                "actual_segment_time_s",
            ]
        ].reset_index(
            drop=True
        ).copy()

        detail[
            "micro_predicted_time_s"
        ] = micro_result[
            "micro_predicted_time_s"
        ].to_numpy()

        detail[
            "macro_predicted_time_s"
        ] = macro_result[
            "macro_predicted_time_s"
        ].to_numpy()

        detail[
            "micro_error_s"
        ] = (
            detail[
                "micro_predicted_time_s"
            ]
            - detail[
                "actual_segment_time_s"
            ]
        )

        detail[
            "macro_error_s"
        ] = (
            detail[
                "macro_predicted_time_s"
            ]
            - detail[
                "actual_segment_time_s"
            ]
        )

        detail[
            "micro_abs_error_s"
        ] = detail[
            "micro_error_s"
        ].abs()

        detail[
            "macro_abs_error_s"
        ] = detail[
            "macro_error_s"
        ].abs()

        detail[
            "micro_minus_macro_s"
        ] = (
            detail[
                "micro_predicted_time_s"
            ]
            - detail[
                "macro_predicted_time_s"
            ]
        )

        for column in [
            "analogue_1_distance",
            "analogue_1_time_s",
            "analogue_1_activity_id",
            "analogue_1_activity_name",
            "analogue_1_distance_from_start_m",
            "analogue_2_distance",
            "analogue_2_time_s",
            "analogue_2_activity_id",
            "analogue_2_activity_name",
            "analogue_2_distance_from_start_m",
        ]:

            if column in micro_result.columns:

                detail[column] = micro_result[
                    column
                ].to_numpy()

        detail_frames.append(
            detail
        )

        actual = detail[
            "actual_segment_time_s"
        ].to_numpy(
            dtype=float
        )

        micro_pred = detail[
            "micro_predicted_time_s"
        ].to_numpy(
            dtype=float
        )

        macro_pred = detail[
            "macro_predicted_time_s"
        ].to_numpy(
            dtype=float
        )

        micro_error = (
            micro_pred
            - actual
        )

        macro_error = (
            macro_pred
            - actual
        )

        micro_minus_macro = (
            micro_pred
            - macro_pred
        )

        actual_total = float(
            np.sum(
                actual
            )
        )

        micro_total = float(
            np.sum(
                micro_pred
            )
        )

        macro_total = float(
            np.sum(
                macro_pred
            )
        )

        actual_elapsed_at_last_test_start = float(
            test_df[
                "elapsed_time_s"
            ].iloc[-1]
        )

        actual_elapsed_after_last_test = (
            actual_elapsed_at_last_test_start
            + actual[-1]
        )

        summary_rows.append(
            {
                "activity_id": held_out_id,
                "activity_name": held_out_name,

                "segment_length_m": (
                    GPX_SEGMENT_LENGTH_M
                ),

                "training_activities": int(
                    training_df[
                        "activity_id"
                    ].nunique()
                ),

                "training_rows": int(
                    len(
                        training_df
                    )
                ),

                "test_segments": int(
                    len(
                        test_df
                    )
                ),

                "micro_mae_s": float(
                    np.mean(
                        np.abs(
                            micro_error
                        )
                    )
                ),

                "macro_mae_s": float(
                    np.mean(
                        np.abs(
                            macro_error
                        )
                    )
                ),

                "micro_median_abs_error_s": float(
                    np.median(
                        np.abs(
                            micro_error
                        )
                    )
                ),

                "macro_median_abs_error_s": float(
                    np.median(
                        np.abs(
                            macro_error
                        )
                    )
                ),

                "micro_bias_s": float(
                    np.mean(
                        micro_error
                    )
                ),

                "macro_bias_s": float(
                    np.mean(
                        macro_error
                    )
                ),

                "mean_micro_minus_macro_s": float(
                    np.mean(
                        micro_minus_macro
                    )
                ),

                "median_micro_minus_macro_s": float(
                    np.median(
                        micro_minus_macro
                    )
                ),

                "mean_abs_micro_minus_macro_s": float(
                    np.mean(
                        np.abs(
                            micro_minus_macro
                        )
                    )
                ),

                "actual_test_time_total_s": (
                    actual_total
                ),

                "micro_predicted_total_time_s": (
                    micro_total
                ),

                "macro_predicted_total_time_s": (
                    macro_total
                ),

                "micro_finish_error_s": (
                    micro_total
                    - actual_total
                ),

                "macro_finish_error_s": (
                    macro_total
                    - actual_total
                ),

                "micro_finish_error_min": (
                    micro_total
                    - actual_total
                )
                / 60.0,

                "macro_finish_error_min": (
                    macro_total
                    - actual_total
                )
                / 60.0,

                "actual_elapsed_after_last_test_s": (
                    actual_elapsed_after_last_test
                ),

                "last_test_start_distance_m": float(
                    test_df[
                        "distance_from_start_m"
                    ].iloc[-1]
                ),

                "last_test_end_distance_m": float(
                    test_df[
                        "distance_from_start_m"
                    ].iloc[-1]
                    + GPX_SEGMENT_LENGTH_M
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if detail_frames:

        detail_df = pd.concat(
            detail_frames,
            ignore_index=True,
        )

    else:

        detail_df = pd.DataFrame()

    return (
        summary_df,
        detail_df,
    )


# =============================================================================
# Simulation diagnostics
# =============================================================================

def build_simulation_divergence_diagnostic(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact diagnostic table showing where macro and micro diverge
    during the simulated race.

    This function does NOT modify the simulation.

    It exposes:

        distance
        macro cumulative time
        micro cumulative time
        micro - macro cumulative difference
        segment-level difference
    """
    if (
        simulation_df is None
        or simulation_df.empty
    ):
        return pd.DataFrame()

    required = [
        "distance_from_start_m",
        "macro_arrival_time_s",
        "micro_arrival_time_s",
    ]

    missing = [
        col
        for col in required
        if col not in simulation_df.columns
    ]

    if missing:
        raise ValueError(
            "Simulation divergence diagnostics missing columns: "
            + ", ".join(missing)
        )

    result = simulation_df[
        required
        + (
            [
                "macro_predicted_time_s",
                "micro_predicted_time_s",
                "micro_minus_macro_s",
            ]
            if all(
                column in simulation_df.columns
                for column in [
                    "macro_predicted_time_s",
                    "micro_predicted_time_s",
                    "micro_minus_macro_s",
                ]
            )
            else []
        )
        + (
            [
                "aid_station_name",
                "aid_station_stop_min",
            ]
            if all(
                column in simulation_df.columns
                for column in [
                    "aid_station_name",
                    "aid_station_stop_min",
                ]
            )
            else []
        )
    ].copy()

    result[
        "distance_km"
    ] = (
        result[
            "distance_from_start_m"
        ]
        / 1000.0
    )

    result[
        "macro_cumulative_time_h"
    ] = (
        result[
            "macro_arrival_time_s"
        ]
        / 3600.0
    )

    result[
        "micro_cumulative_time_h"
    ] = (
        result[
            "micro_arrival_time_s"
        ]
        / 3600.0
    )

    result[
        "micro_minus_macro_cumulative_min"
    ] = (
        (
            result[
                "micro_arrival_time_s"
            ]
            - result[
                "macro_arrival_time_s"
            ]
        )
        / 60.0
    )

    return result


def build_simulation_divergence_checkpoints(
    simulation_df: pd.DataFrame,
    checkpoint_distances_km: list[float] | None = None,
) -> pd.DataFrame:
    """
    Extract macro/micro cumulative divergence near selected course
    checkpoints.

    This is useful for answering:

        Is the final difference accumulating gradually,
        or is one section responsible for most of it?
    """
    diagnostic_df = (
        build_simulation_divergence_diagnostic(
            simulation_df
        )
    )

    if diagnostic_df.empty:
        return diagnostic_df

    if checkpoint_distances_km is None:

        max_distance_km = float(
            diagnostic_df[
                "distance_km"
            ].max()
        )

        checkpoint_distances_km = [
            max_distance_km * 0.20,
            max_distance_km * 0.40,
            max_distance_km * 0.60,
            max_distance_km * 0.80,
            max_distance_km,
        ]

    rows: list[
        dict[str, Any]
    ] = []

    distances = diagnostic_df[
        "distance_km"
    ].to_numpy(
        dtype=float
    )

    for checkpoint_km in checkpoint_distances_km:

        if not np.isfinite(
            checkpoint_km
        ):
            continue

        index = int(
            np.argmin(
                np.abs(
                    distances
                    - checkpoint_km
                )
            )
        )

        row = diagnostic_df.iloc[
            index
        ]

        rows.append(
            {
                "requested_checkpoint_km": (
                    float(
                        checkpoint_km
                    )
                ),
                "actual_checkpoint_km": (
                    float(
                        row[
                            "distance_km"
                        ]
                    )
                ),
                "macro_cumulative_time_h": (
                    float(
                        row[
                            "macro_cumulative_time_h"
                        ]
                    )
                ),
                "micro_cumulative_time_h": (
                    float(
                        row[
                            "micro_cumulative_time_h"
                        ]
                    )
                ),
                "micro_minus_macro_cumulative_min": (
                    float(
                        row[
                            "micro_minus_macro_cumulative_min"
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_macro_clipping_diagnostic(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Diagnose negative raw macro increments that were constrained to zero.

    This does not change the simulation.
    """
    if (
        simulation_df is None
        or simulation_df.empty
    ):
        return pd.DataFrame()

    required = [
        "distance_from_start_m",
        "raw_macro_predicted_time_s",
        "macro_predicted_time_s",
        "macro_time_clipped",
    ]

    missing = [
        column
        for column in required
        if column not in simulation_df.columns
    ]

    if missing:
        raise ValueError(
            "Macro clipping diagnostics missing columns: "
            + ", ".join(missing)
        )

    clipped = simulation_df[
        simulation_df[
            "macro_time_clipped"
        ]
    ].copy()

    if clipped.empty:
        return pd.DataFrame(
            columns=required
        )

    result = clipped[
        required
    ].copy()

    result[
        "clipped_correction_s"
    ] = (
        result[
            "macro_predicted_time_s"
        ]
        - result[
            "raw_macro_predicted_time_s"
        ]
    )

    return result


def build_simulation_section_summary(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize model differences by coarse course section.

    Sections are diagnostic only.

    Default:
        10 equal-distance sections.
    """
    diagnostic_df = (
        build_simulation_divergence_diagnostic(
            simulation_df
        )
    )

    if diagnostic_df.empty:
        return diagnostic_df

    max_distance = float(
        diagnostic_df[
            "distance_km"
        ].max()
    )

    if max_distance <= 0.0:
        return pd.DataFrame()

    section_edges = np.linspace(
        0.0,
        max_distance,
        11,
    )

    diagnostic_df[
        "section"
    ] = pd.cut(
        diagnostic_df[
            "distance_km"
        ],
        bins=section_edges,
        labels=False,
        include_lowest=True,
    )

    rows: list[
        dict[str, Any]
    ] = []

    for section, group in diagnostic_df.groupby(
        "section",
        dropna=True,
    ):

        rows.append(
            {
                "section": int(
                    section
                ) + 1,
                "start_km": float(
                    section_edges[
                        int(section)
                    ]
                ),
                "end_km": float(
                    section_edges[
                        int(section) + 1
                    ]
                ),
                "mean_segment_difference_s": (
                    float(
                        group.get(
                            "micro_minus_macro_s",
                            pd.Series(
                                dtype=float
                            ),
                        ).mean()
                    )
                    if "micro_minus_macro_s"
                    in group.columns
                    else np.nan
                ),
                "end_cumulative_difference_min": (
                    float(
                        group[
                            "micro_minus_macro_cumulative_min"
                        ].iloc[-1]
                    )
                ),
                "segments": int(
                    len(group)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Combined simulation diagnostic package
# =============================================================================

def build_simulation_diagnostics(
    simulation_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build the complete disposable diagnostics package for one simulation.

    Nothing returned here feeds back into the simulator.
    """
    divergence = (
        build_simulation_divergence_diagnostic(
            simulation_df
        )
    )

    checkpoints = (
        build_simulation_divergence_checkpoints(
            simulation_df
        )
    )

    clipping = (
        build_macro_clipping_diagnostic(
            simulation_df
        )
    )

    sections = (
        build_simulation_section_summary(
            simulation_df
        )
    )

    return {
        "simulation_divergence": divergence,
        "simulation_checkpoints": checkpoints,
        "macro_clipping": clipping,
        "simulation_sections": sections,
    }


# =============================================================================
# Combined diagnostics package
# =============================================================================

def build_all_diagnostics(
    learning_df: pd.DataFrame,
    uploaded_fit_files,
    macro_model,
) -> dict[str, pd.DataFrame]:

    diagnostic_sample = (
        build_learning_diagnostic_sample(
            learning_df
        )
    )

    activity_learning = (
        build_activity_learning_summary(
            learning_df
        )
    )

    extreme_transitions = (
        build_extreme_transition_summary(
            learning_df
        )
    )

    activity_macro = (
        build_activity_macro_summary(
            learning_df,
            macro_model,
        )
    )

    macro_check = (
        build_macro_historical_check(
            learning_df,
            macro_model,
        )
    )

    stop_summary, stop_intervals = (
        analyze_uploaded_fit_stops(
            uploaded_fit_files
        )
    )

    stop_macro_comparison = (
        compare_stops_with_macro(
            stop_summary,
            activity_macro,
        )
    )

    return {
        "learning_diagnostic_sample": (
            diagnostic_sample
        ),
        "activity_learning_summary": (
            activity_learning
        ),
        "extreme_transitions": (
            extreme_transitions
        ),
        "activity_macro_summary": (
            activity_macro
        ),
        "macro_historical_check": (
            macro_check
        ),
        "stop_summary": (
            stop_summary
        ),
        "stop_intervals": (
            stop_intervals
        ),
        "stop_macro_comparison": (
            stop_macro_comparison
        ),
    }
    
