from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RANDOM_SAMPLE_SIZE = 100
EXTREME_SAMPLE_SIZE = 100
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Compact diagnostic sample
# -----------------------------------------------------------------------------

def build_learning_diagnostic_sample(
    learning_df: pd.DataFrame,
    random_sample_size: int = RANDOM_SAMPLE_SIZE,
    extreme_sample_size: int = EXTREME_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Build a compact diagnostic extract from the complete historical dataset.
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

    # -------------------------------------------------------------------------
    # First rows
    # -------------------------------------------------------------------------

    first_rows = df.head(
        min(100, len(df))
    ).copy()

    first_rows.insert(
        0,
        "diagnostic_group",
        "first_rows",
    )

    # -------------------------------------------------------------------------
    # Deterministic random sample
    # -------------------------------------------------------------------------

    random_rows = df.sample(
        n=min(random_sample_size, len(df)),
        random_state=RANDOM_STATE,
    ).copy()

    random_rows.insert(
        0,
        "diagnostic_group",
        "random_sample",
    )

    # -------------------------------------------------------------------------
    # Slowest
    # -------------------------------------------------------------------------

    slow_rows = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=False,
            kind="mergesort",
        )
        .head(extreme_sample_size)
        .copy()
    )

    slow_rows.insert(
        0,
        "diagnostic_group",
        "slowest",
    )

    # -------------------------------------------------------------------------
    # Fastest
    # -------------------------------------------------------------------------

    fast_rows = (
        df.sort_values(
            "actual_segment_time_s",
            ascending=True,
            kind="mergesort",
        )
        .head(extreme_sample_size)
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


# -----------------------------------------------------------------------------
# Per-activity learning diagnostics
# -----------------------------------------------------------------------------

def build_activity_learning_summary(
    learning_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the historical 50 m transition data separately for each FIT.

    This is diagnostic only.
    No rows are filtered or modified.
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
        column
        for column in required
        if column not in learning_df.columns
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

    rows = []

    for (activity_id, activity_name), group in df.groupby(
        ["activity_id", "activity_name"],
        sort=True,
    ):

        duration = group[
            "actual_segment_time_s"
        ].dropna()

        if duration.empty:
            continue

        fastest = float(
            duration.min()
        )

        slowest = float(
            duration.max()
        )

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

                "transitions": int(
                    len(group)
                ),

                "race_distance_m": float(
                    group[
                        "distance_from_start_m"
                    ].max()
                    + 50.0
                ),

                "median_50m_time_s": float(
                    duration.median()
                ),

                "mean_50m_time_s": float(
                    duration.mean()
                ),

                "fastest_50m_time_s": fastest,

                "fastest_implied_speed_m_s": (
                    fastest_speed_m_s
                ),

                "fastest_implied_speed_kmh": (
                    fastest_speed_m_s * 3.6
                ),

                "slowest_50m_time_s": slowest,

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


# -----------------------------------------------------------------------------
# Per-activity macro diagnostics
# -----------------------------------------------------------------------------

def build_activity_macro_summary(
    learning_df: pd.DataFrame,
    macro_model,
) -> pd.DataFrame:
    """
    Compare actual cumulative elapsed time against the macro prediction
    separately for each historical FIT.

    Diagnostic only.
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
        column
        for column in required
        if column not in learning_df.columns
    ]

    if missing:
        raise ValueError(
            "Macro diagnostics missing columns: "
            + ", ".join(missing)
        )

    df = learning_df.copy()

    for column in [
        "distance_from_start_m",
        "cumulative_ascent_m",
        "cumulative_descent_m",
        "elapsed_time_s",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    rows = []

    for (activity_id, activity_name), group in df.groupby(
        ["activity_id", "activity_name"],
        sort=True,
    ):

        group = (
            group
            .dropna(
                subset=[
                    "distance_from_start_m",
                    "cumulative_ascent_m",
                    "cumulative_descent_m",
                    "elapsed_time_s",
                ]
            )
            .sort_values(
                "distance_from_start_m",
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
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

        # ---------------------------------------------------------------------
        # Beginning
        # ---------------------------------------------------------------------

        start_index = 0

        # ---------------------------------------------------------------------
        # Middle
        # ---------------------------------------------------------------------

        middle_index = (
            len(group) // 2
        )

        # ---------------------------------------------------------------------
        # End
        # ---------------------------------------------------------------------

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
                    ].iloc[end_index]
                ),

                "cumulative_ascent_m": float(
                    group[
                        "cumulative_ascent_m"
                    ].iloc[end_index]
                ),

                "cumulative_descent_m": float(
                    group[
                        "cumulative_descent_m"
                    ].iloc[end_index]
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

                # -------------------------------------------------------------
                # Start
                # -------------------------------------------------------------

                "start_actual_time_s": float(
                    actual[start_index]
                ),

                "start_macro_time_s": float(
                    predicted[start_index]
                ),

                "start_error_s": float(
                    error[start_index]
                ),

                # -------------------------------------------------------------
                # Middle
                # -------------------------------------------------------------

                "middle_distance_m": float(
                    group[
                        "distance_from_start_m"
                    ].iloc[middle_index]
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

                # -------------------------------------------------------------
                # End
                # -------------------------------------------------------------

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
                    error[end_index]
                    / 60.0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# -----------------------------------------------------------------------------
# Extreme-transition diagnostics
# -----------------------------------------------------------------------------

def build_extreme_transition_summary(
    learning_df: pd.DataFrame,
    n_each: int = 20,
) -> pd.DataFrame:
    """
    Return the fastest and slowest historical transitions with enough
    information to identify where they occurred.

    No filtering is performed.
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
        column
        for column in required
        if column not in learning_df.columns
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
    ]

    df["implied_speed_m_s"] = (
        50.0
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
        .head(n_each)
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
        .head(n_each)
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
    
