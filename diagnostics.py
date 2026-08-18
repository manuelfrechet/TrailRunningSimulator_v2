from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import fitdecode


# -----------------------------------------------------------------------------
# Diagnostics configuration
# -----------------------------------------------------------------------------

RANDOM_SAMPLE_SIZE = 100
EXTREME_SAMPLE_SIZE = 20
RANDOM_STATE = 42

# Numerical tolerance only.
# This is NOT a physiological "stop" threshold.
DISTANCE_TOLERANCE_M = 0.01


# =============================================================================
# Learning-dataset diagnostics
# =============================================================================

def build_learning_diagnostic_sample(
    learning_df: pd.DataFrame,
    random_sample_size: int = RANDOM_SAMPLE_SIZE,
    extreme_sample_size: int = 100,
) -> pd.DataFrame:
    """
    Build a compact diagnostic extract from the complete historical dataset.

    Contains:
        - first rows
        - deterministic random sample
        - slowest transitions
        - fastest transitions
    """
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
    first_rows.insert(0, "diagnostic_group", "first_rows")

    random_rows = df.sample(
        n=min(random_sample_size, len(df)),
        random_state=RANDOM_STATE,
    ).copy()
    random_rows.insert(0, "diagnostic_group", "random_sample")

    slow_rows = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=False,
            kind="mergesort",
        )
        .head(extreme_sample_size)
        .copy()
    )
    slow_rows.insert(0, "diagnostic_group", "slowest")

    fast_rows = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=True,
            kind="mergesort",
        )
        .head(extreme_sample_size)
        .copy()
    )
    fast_rows.insert(0, "diagnostic_group", "fastest")

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
    """
    Summarize historical 50 m transitions separately by FIT activity.
    """
    if learning_df is None or learning_df.empty:
        return pd.DataFrame()

    required = [
        "activity_id",
        "activity_name",
        "distance_from_start_m",
        "actual_segment_time_s",
    ]

    missing = [
        col for col in required
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

    for (activity_id, activity_name), group in df.groupby(
        ["activity_id", "activity_name"],
        sort=True,
    ):
        durations = group["actual_segment_time_s"].dropna()

        if durations.empty:
            continue

        fastest = float(durations.min())
        slowest = float(durations.max())

        fastest_speed_m_s = (
            50.0 / fastest
            if fastest > 0.0
            else np.nan
        )

        slowest_speed_m_s = (
            50.0 / slowest
            if slowest > 0.0
            else np.nan
        )

        rows.append(
            {
                "activity_id": activity_id,
                "activity_name": activity_name,
                "transitions": int(len(group)),
                "race_distance_m": float(
                    group["distance_from_start_m"].max() + 50.0
                ),
                "median_50m_time_s": float(durations.median()),
                "mean_50m_time_s": float(durations.mean()),
                "fastest_50m_time_s": fastest,
                "fastest_implied_speed_m_s": fastest_speed_m_s,
                "fastest_implied_speed_kmh": fastest_speed_m_s * 3.6,
                "slowest_50m_time_s": slowest,
                "slowest_implied_speed_m_s": slowest_speed_m_s,
                "slowest_implied_speed_kmh": slowest_speed_m_s * 3.6,
            }
        )

    return pd.DataFrame(rows)


def build_extreme_transition_summary(
    learning_df: pd.DataFrame,
    n_each: int = EXTREME_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Return the fastest and slowest historical transitions.

    No observations are removed from the actual learning dataset.
    """
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
        col for col in required
        if col not in learning_df.columns
    ]

    if missing:
        raise ValueError(
            "Extreme-transition diagnostics missing columns: "
            + ", ".join(missing)
        )

    df = learning_df[required].copy()

    df["actual_segment_time_s"] = pd.to_numeric(
        df["actual_segment_time_s"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["actual_segment_time_s"]
    )

    df = df[
        df["actual_segment_time_s"] > 0.0
    ].copy()

    df["implied_speed_m_s"] = (
        50.0 / df["actual_segment_time_s"]
    )

    df["implied_speed_kmh"] = (
        df["implied_speed_m_s"] * 3.6
    )

    fastest = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=True,
            kind="mergesort",
        )
        .head(n_each)
        .copy()
    )
    fastest.insert(0, "extreme_type", "fastest")

    slowest = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=False,
            kind="mergesort",
        )
        .head(n_each)
        .copy()
    )
    slowest.insert(0, "extreme_type", "slowest")

    return pd.concat(
        [fastest, slowest],
        ignore_index=True,
    )


# =============================================================================
# Raw FIT stop diagnostics
# =============================================================================

def _extract_raw_time_distance(
    uploaded_file,
) -> pd.DataFrame:
    """
    Read only timestamp and distance from the original FIT records.
    """
    uploaded_file.seek(0)

    rows: list[dict[str, Any]] = []

    with fitdecode.FitReader(uploaded_file) as fit:
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

    df = pd.DataFrame(rows)

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

    # Chronological order matters for stop detection.
    df = (
        df.sort_values(
            "timestamp",
            kind="mergesort",
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def _build_stationary_intervals(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify intervals where time advances but recorded distance does not.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    if len(raw_df) < 2:
        return pd.DataFrame()

    timestamps = raw_df["timestamp"]
    distances = raw_df["distance_m"].to_numpy(dtype=float)

    delta_time_s = (
        timestamps.diff()
        .dt.total_seconds()
        .to_numpy(dtype=float)
    )

    delta_distance_m = np.diff(
        distances,
        prepend=np.nan,
    )

    valid_time = (
        np.isfinite(delta_time_s)
        & (delta_time_s > 0.0)
    )

    stationary = (
        valid_time
        & np.isfinite(delta_distance_m)
        & (
            np.abs(delta_distance_m)
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

    interval_df["stationary_time_s"] = np.where(
        stationary,
        delta_time_s,
        0.0,
    )

    interval_df["implied_speed_m_s"] = np.where(
        valid_time,
        delta_distance_m / delta_time_s,
        np.nan,
    )

    return interval_df


def analyze_fit_stops(
    uploaded_file,
    activity_id: int,
    activity_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Analyze stationary time for one FIT.

    Diagnostic only. Does not alter the learning dataset.
    """
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
            raw_df["timestamp"].iloc[-1]
            - raw_df["timestamp"].iloc[0]
        ).total_seconds()
    )

    distance_m = float(
        raw_df["distance_m"].iloc[-1]
        - raw_df["distance_m"].iloc[0]
    )

    if intervals.empty:
        stationary_time_s = 0.0
        stationary_intervals = 0
    else:
        stationary_time_s = float(
            intervals["stationary_time_s"].sum()
        )
        stationary_intervals = int(
            intervals["is_stationary"].sum()
        )

    stationary_fraction = (
        stationary_time_s / elapsed_time_s
        if elapsed_time_s > 0.0
        else np.nan
    )

    summary = {
        "activity_id": activity_id,
        "activity_name": activity_name,
        "raw_records": int(len(raw_df)),
        "elapsed_time_s": elapsed_time_s,
        "distance_m": distance_m,
        "stationary_intervals": stationary_intervals,
        "stationary_time_s": stationary_time_s,
        "stationary_time_min": stationary_time_s / 60.0,
        "stationary_fraction_pct": (
            stationary_fraction * 100.0
            if np.isfinite(stationary_fraction)
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

    return summary, intervals


def analyze_uploaded_fit_stops(
    uploaded_files,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze all uploaded FIT files for stationary time.

    Returns:
        per-activity summary
        raw interval diagnostics
    """
    if not uploaded_files:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    summaries: list[dict[str, Any]] = []
    interval_frames: list[pd.DataFrame] = []

    for activity_id, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        activity_name = getattr(
            uploaded_file,
            "name",
            f"activity_{activity_id}",
        )

        summary, intervals = analyze_fit_stops(
            uploaded_file,
            activity_id,
            activity_name,
        )

        summaries.append(summary)

        if not intervals.empty:
            interval_frames.append(intervals)

    summary_df = pd.DataFrame(
        summaries
    )

    intervals_df = (
        pd.concat(
            interval_frames,
            ignore_index=True,
        )
        if interval_frames
        else pd.DataFrame()
    )

    return summary_df, intervals_df


# =============================================================================
# Macro diagnostics
# =============================================================================

def build_activity_macro_summary(
    learning_df: pd.DataFrame,
    macro_model,
) -> pd.DataFrame:
    """
    Compare historical cumulative time to macro prediction separately
    for each FIT activity.
    """
    if learning_df is None or learning_df.empty:
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
        col for col in required
        if col not in learning_df.columns
    ]

    if missing:
        raise ValueError(
            "Macro diagnostics missing columns: "
            + ", ".join(missing)
        )

    rows: list[dict[str, Any]] = []

    for (activity_id, activity_name), group in learning_df.groupby(
        ["activity_id", "activity_name"],
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
            .reset_index(drop=True)
        )

        if group.empty:
            continue

        predicted = macro_model.predict_cumulative_time(
            distance_from_start_m=group[
                "distance_from_start_m"
            ],
            cumulative_ascent_m=group[
                "cumulative_ascent_m"
            ],
            cumulative_descent_m=group[
                "cumulative_descent_m"
            ],
        )

        actual = group["elapsed_time_s"].to_numpy(
            dtype=float
        )

        error = predicted - actual
        abs_error = np.abs(error)

        middle_index = len(group) // 2
        end_index = len(group) - 1

        rows.append(
            {
                "activity_id": activity_id,
                "activity_name": activity_name,
                "rows": int(len(group)),
                "distance_m": float(
                    group["distance_from_start_m"].iloc[end_index]
                ),
                "cumulative_ascent_m": float(
                    group["cumulative_ascent_m"].iloc[end_index]
                ),
                "cumulative_descent_m": float(
                    group["cumulative_descent_m"].iloc[end_index]
                ),
                "mean_absolute_error_s": float(
                    np.mean(abs_error)
                ),
                "median_absolute_error_s": float(
                    np.median(abs_error)
                ),
                "mean_bias_s": float(
                    np.mean(error)
                ),
                "start_actual_time_s": float(actual[0]),
                "start_macro_time_s": float(predicted[0]),
                "start_error_s": float(error[0]),
                "middle_distance_m": float(
                    group["distance_from_start_m"].iloc[middle_index]
                ),
                "middle_actual_time_s": float(
                    actual[middle_index]
                ),
                "middle_macro_time_s": float(
                    predicted[middle_index]
                ),
                "middle_error_s": float(
                    error[middle_index]
                ),
                "end_actual_time_s": float(
                    actual[end_index]
                ),
                "end_macro_time_s": float(
                    predicted[end_index]
                ),
                "end_error_s": float(
                    error[end_index]
                ),
                "end_error_min": float(
                    error[end_index] / 60.0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_macro_historical_check(
    learning_df: pd.DataFrame,
    macro_model,
) -> pd.DataFrame:
    """
    Build the full historical macro prediction/error table.
    """
    if learning_df is None or learning_df.empty:
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
        col for col in required
        if col not in learning_df.columns
    ]

    if missing:
        raise ValueError(
            "Macro check missing columns: "
            + ", ".join(missing)
        )

    base = learning_df[required].copy()

    predicted = macro_model.predict_cumulative_time(
        base["distance_from_start_m"],
        base["cumulative_ascent_m"],
        base["cumulative_descent_m"],
    )

    base["macro_predicted_cumulative_time_s"] = predicted
    base["macro_error_s"] = (
        base["macro_predicted_cumulative_time_s"]
        - base["elapsed_time_s"]
    )
    base["macro_error_abs_s"] = (
        base["macro_error_s"].abs()
    )

    return base


def compare_stops_with_macro(
    stop_summary_df: pd.DataFrame,
    activity_macro_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare stationary-time diagnostics against macro-model finish errors.

    Diagnostic only.
    """
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

    # end_error_s = macro - actual
    # Therefore positive underprediction = actual - macro = -error.
    comparison["macro_underprediction_s"] = (
        -comparison["end_error_s"]
    )

    comparison["macro_underprediction_min"] = (
        comparison["macro_underprediction_s"] / 60.0
    )

    comparison[
        "stationary_minus_macro_underprediction_s"
    ] = (
        comparison["stationary_time_s"]
        - comparison["macro_underprediction_s"]
    )

    comparison[
        "stationary_minus_macro_underprediction_min"
    ] = (
        comparison[
            "stationary_minus_macro_underprediction_s"
        ] / 60.0
    )

    return comparison


# =============================================================================
# One convenience function for the app
# =============================================================================

def build_all_diagnostics(
    learning_df: pd.DataFrame,
    uploaded_fit_files,
    macro_model,
) -> dict[str, pd.DataFrame]:
    """
    Build all diagnostics in one place.

    The production application does not depend on this function.
    """
    diagnostic_sample = build_learning_diagnostic_sample(
        learning_df
    )

    activity_learning = build_activity_learning_summary(
        learning_df
    )

    extreme_transitions = build_extreme_transition_summary(
        learning_df
    )

    activity_macro = build_activity_macro_summary(
        learning_df,
        macro_model,
    )

    macro_check = build_macro_historical_check(
        learning_df,
        macro_model,
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
        "learning_diagnostic_sample": diagnostic_sample,
        "activity_learning_summary": activity_learning,
        "extreme_transitions": extreme_transitions,
        "activity_macro_summary": activity_macro,
        "macro_historical_check": macro_check,
        "stop_summary": stop_summary,
        "stop_intervals": stop_intervals,
        "stop_macro_comparison": stop_macro_comparison,
    }
  
