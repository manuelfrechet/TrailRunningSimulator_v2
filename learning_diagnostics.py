from __future__ import annotations

import pandas as pd


RANDOM_SAMPLE_SIZE = 100
EXTREME_SAMPLE_SIZE = 100
RANDOM_STATE = 42


def build_learning_diagnostic_sample(
    learning_df: pd.DataFrame,
    random_sample_size: int = RANDOM_SAMPLE_SIZE,
    extreme_sample_size: int = EXTREME_SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Build a compact diagnostic extract from the complete historical
    learning dataset.

    The extract contains:
        - first rows
        - deterministic random sample
        - slowest transitions
        - fastest transitions

    A 'diagnostic_group' column identifies the origin of each row.
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
    first_n = min(
        RANDOM_SAMPLE_SIZE,
        len(df),
    )

    first_rows = df.head(first_n).copy()
    first_rows.insert(
        0,
        "diagnostic_group",
        "first_rows",
    )

    # -------------------------------------------------------------------------
    # Random sample
    # -------------------------------------------------------------------------
    random_n = min(
        random_sample_size,
        len(df),
    )

    random_rows = (
        df.sample(
            n=random_n,
            random_state=RANDOM_STATE,
        )
        .copy()
    )

    random_rows.insert(
        0,
        "diagnostic_group",
        "random_sample",
    )

    # -------------------------------------------------------------------------
    # Slowest transitions
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
        "slowest_100",
    )

    # -------------------------------------------------------------------------
    # Fastest transitions
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
        "fastest_100",
    )

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------
    diagnostic_df = pd.concat(
        [
            first_rows,
            random_rows,
            slow_rows,
            fast_rows,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Add rank within extreme groups for easy inspection.
    # -------------------------------------------------------------------------
    diagnostic_df["diagnostic_rank"] = pd.NA

    slow_mask = (
        diagnostic_df["diagnostic_group"]
        == "slowest_100"
    )

    fast_mask = (
        diagnostic_df["diagnostic_group"]
        == "fastest_100"
    )

    diagnostic_df.loc[
        slow_mask,
        "diagnostic_rank",
    ] = range(
        1,
        int(slow_mask.sum()) + 1,
    )

    diagnostic_df.loc[
        fast_mask,
        "diagnostic_rank",
    ] = range(
        1,
        int(fast_mask.sum()) + 1,
    )

    return diagnostic_df
  
