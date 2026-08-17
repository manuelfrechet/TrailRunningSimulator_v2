from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import fitdecode


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ROLLING_STEP_M = 1.0
TRANSITION_LENGTH_M = 50.0


# -----------------------------------------------------------------------------
# FIT parsing
# -----------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    """
    Convert a scalar value to float, returning NaN when unavailable.
    """
    try:
        if value is None:
            return np.nan

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value).reshape(-1)
            if arr.size == 0:
                return np.nan
            value = arr[0]

        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _extract_record_messages(uploaded_file) -> pd.DataFrame:
    """
    Read FIT record messages and return a raw dataframe.

    We intentionally keep this function small. No learning happens here.
    """
    uploaded_file.seek(0)

    rows: list[dict[str, Any]] = []

    with fitdecode.FitReader(uploaded_file) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.records.FitDataMessage):
                continue

            if frame.name != "record":
                continue

            row: dict[str, Any] = {}

            for field in frame.fields:
                row[field.name] = field.value

            rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Raw FIT normalization
# -----------------------------------------------------------------------------

def _standardize_raw_fit(record_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw FIT record messages into a standardized trajectory dataframe.

    Required core information:
        timestamp
        distance
        altitude

    Other measurements are preserved when available.
    """
    if record_df is None or record_df.empty:
        return pd.DataFrame()

    df = record_df.copy()

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df = df.dropna(subset=["timestamp"]).copy()

        if not df.empty:
            df["time_from_start_s"] = (
                df["timestamp"] - df["timestamp"].iloc[0]
            ).dt.total_seconds()

    if "time_from_start_s" not in df.columns:
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Distance
    # -------------------------------------------------------------------------
    if "distance" in df.columns:
        distance = pd.to_numeric(
            df["distance"],
            errors="coerce",
        )

        # FIT distance is generally stored in metres.
        df["distance_from_start_m"] = distance

    elif "enhanced_distance" in df.columns:
        distance = pd.to_numeric(
            df["enhanced_distance"],
            errors="coerce",
        )
        df["distance_from_start_m"] = distance

    else:
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Altitude
    # -------------------------------------------------------------------------
    altitude_source = None

    if "enhanced_altitude" in df.columns:
        altitude_source = df["enhanced_altitude"]
    elif "altitude" in df.columns:
        altitude_source = df["altitude"]

    if altitude_source is not None:
        df["altitude_m"] = pd.to_numeric(
            altitude_source,
            errors="coerce",
        )
    else:
        df["altitude_m"] = np.nan

    # -------------------------------------------------------------------------
    # Rename useful FIT measurements to project names.
    # -------------------------------------------------------------------------
    rename_map = {
        "heart_rate": "heart_rate_bpm",
        "power": "power",
        "cadence": "cadence_spm",
        "enhanced_speed": "speed_m_s",
        "speed": "speed_m_s",
        "vertical_oscillation": "vertical_oscillation_mm",
        "stance_time": "stance_time_s",
        "step_length": "step_length_m",
    }

    for source_name, target_name in rename_map.items():
        if source_name in df.columns and target_name not in df.columns:
            df[target_name] = pd.to_numeric(
                df[source_name],
                errors="coerce",
            )

    # -------------------------------------------------------------------------
    # Clean ordering and duplicates.
    # -------------------------------------------------------------------------
    df = df.sort_values(
        ["distance_from_start_m", "time_from_start_s"],
        kind="mergesort",
    ).reset_index(drop=True)

    df = df.dropna(
        subset=[
            "distance_from_start_m",
            "time_from_start_s",
        ]
    ).copy()

    if df.empty:
        return pd.DataFrame()

    # Shift both axes to zero.
    df["distance_from_start_m"] -= float(
        df["distance_from_start_m"].iloc[0]
    )

    df["time_from_start_s"] -= float(
        df["time_from_start_s"].iloc[0]
    )

    # Remove duplicate distance points.
    df = df.drop_duplicates(
        subset=["distance_from_start_m"],
        keep="last",
    ).reset_index(drop=True)

    return df


# -----------------------------------------------------------------------------
# Dense 1 m trajectory
# -----------------------------------------------------------------------------

def _interpolate_numeric_column(
    source_df: pd.DataFrame,
    dense_distance_m: np.ndarray,
    column: str,
) -> np.ndarray:
    """
    Interpolate one numeric column onto the dense 1 m grid.
    """
    if column not in source_df.columns:
        return np.full(
            len(dense_distance_m),
            np.nan,
            dtype=float,
        )

    x = pd.to_numeric(
        source_df["distance_from_start_m"],
        errors="coerce",
    )

    y = pd.to_numeric(
        source_df[column],
        errors="coerce",
    )

    valid = (
        x.notna()
        & y.notna()
        & np.isfinite(x.to_numpy(dtype=float))
        & np.isfinite(y.to_numpy(dtype=float))
    )

    if valid.sum() < 2:
        return np.full(
            len(dense_distance_m),
            np.nan,
            dtype=float,
        )

    x_valid = x.loc[valid].to_numpy(dtype=float)
    y_valid = y.loc[valid].to_numpy(dtype=float)

    order = np.argsort(x_valid)

    x_valid = x_valid[order]
    y_valid = y_valid[order]

    unique_mask = np.concatenate(
        [
            np.array([True]),
            np.diff(x_valid) > 0,
        ]
    )

    x_valid = x_valid[unique_mask]
    y_valid = y_valid[unique_mask]

    if len(x_valid) < 2:
        return np.full(
            len(dense_distance_m),
            np.nan,
            dtype=float,
        )

    return np.interp(
        dense_distance_m,
        x_valid,
        y_valid,
    )


def _build_dense_trajectory(
    standardized_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert irregular FIT observations into a 1 m rolling trajectory.
    """
    if standardized_df is None or standardized_df.empty:
        return pd.DataFrame()

    max_distance_m = float(
        standardized_df["distance_from_start_m"].max()
    )

    if max_distance_m < TRANSITION_LENGTH_M:
        return pd.DataFrame()

    dense_distance = np.arange(
        0.0,
        max_distance_m + ROLLING_STEP_M * 0.5,
        ROLLING_STEP_M,
    )

    dense = pd.DataFrame(
        {
            "distance_from_start_m": dense_distance,
        }
    )

    # -------------------------------------------------------------------------
    # Time
    # -------------------------------------------------------------------------
    dense["time_from_start_s"] = _interpolate_numeric_column(
        standardized_df,
        dense_distance,
        "time_from_start_s",
    )

    # -------------------------------------------------------------------------
    # Core measurements
    # -------------------------------------------------------------------------
    for column in [
        "altitude_m",
        "heart_rate_bpm",
        "power",
        "cadence_spm",
        "speed_m_s",
        "vertical_oscillation_mm",
        "stance_time_s",
        "step_length_m",
    ]:
        dense[column] = _interpolate_numeric_column(
            standardized_df,
            dense_distance,
            column,
        )

    # -------------------------------------------------------------------------
    # Terrain quantities
    # -------------------------------------------------------------------------
    dense["altitude_delta_m"] = (
        dense["altitude_m"]
        .diff()
        .fillna(0.0)
    )

    dense["ascent_m"] = (
        dense["altitude_delta_m"]
        .clip(lower=0.0)
    )

    dense["descent_m"] = (
        -dense["altitude_delta_m"]
        .clip(upper=0.0)
    )

    dense["cumulative_ascent_m"] = (
        dense["ascent_m"]
        .cumsum()
    )

    dense["cumulative_descent_m"] = (
        dense["descent_m"]
        .cumsum()
    )

    # -------------------------------------------------------------------------
    # Mean/local grade at the 1 m scale.
    #
    # For V0, grade_pct is simply the local 1 m grade. The 50 m segment grade
    # will later be calculated as the mean of these local grades.
    # -------------------------------------------------------------------------
    dense["grade_pct"] = (
        dense["altitude_delta_m"]
        / ROLLING_STEP_M
        * 100.0
    )

    dense["grade_pct"] = dense["grade_pct"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    dense["grade_pct"] = dense["grade_pct"].fillna(0.0)

    return dense


# -----------------------------------------------------------------------------
# Rolling 50 m historical transitions
# -----------------------------------------------------------------------------

def _build_transition_rows(
    dense_df: pd.DataFrame,
    activity_id: int,
    activity_name: str,
) -> pd.DataFrame:
    """
    Build one historical transition record for every 1 m start position.

    One row represents:
        d -> d + 50 m
    """
    if dense_df is None or dense_df.empty:
        return pd.DataFrame()

    horizon_steps = int(
        round(
            TRANSITION_LENGTH_M
            / ROLLING_STEP_M
        )
    )

    if len(dense_df) <= horizon_steps:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    time_values = dense_df["time_from_start_s"].to_numpy(dtype=float)
    distance_values = dense_df["distance_from_start_m"].to_numpy(dtype=float)

    cumulative_ascent = dense_df["cumulative_ascent_m"].to_numpy(dtype=float)
    cumulative_descent = dense_df["cumulative_descent_m"].to_numpy(dtype=float)

    ascent_values = dense_df["ascent_m"].to_numpy(dtype=float)
    descent_values = dense_df["descent_m"].to_numpy(dtype=float)
    grade_values = dense_df["grade_pct"].to_numpy(dtype=float)

    for start_idx in range(
        0,
        len(dense_df) - horizon_steps,
        int(ROLLING_STEP_M),
    ):
        end_idx = start_idx + horizon_steps

        start_time = time_values[start_idx]
        end_time = time_values[end_idx]

        if not np.isfinite(start_time) or not np.isfinite(end_time):
            continue

        actual_segment_time_s = (
            end_time - start_time
        )

        if not np.isfinite(actual_segment_time_s):
            continue

        if actual_segment_time_s <= 0.0:
            continue

        segment_ascent_m = float(
            np.sum(
                ascent_values[
                    start_idx + 1 : end_idx + 1
                ]
            )
        )

        segment_descent_m = float(
            np.sum(
                descent_values[
                    start_idx + 1 : end_idx + 1
                ]
            )
        )

        segment_grade_pct = float(
            np.mean(
                grade_values[
                    start_idx + 1 : end_idx + 1
                ]
            )
        )

        rows.append(
            {
                "activity_id": activity_id,
                "activity_name": activity_name,
                "distance_from_start_m": float(
                    distance_values[start_idx]
                ),
                "cumulative_ascent_m": float(
                    cumulative_ascent[start_idx]
                ),
                "cumulative_descent_m": float(
                    cumulative_descent[start_idx]
                ),
                "elapsed_time_s": float(
                    start_time
                ),
                "segment_ascent_m": segment_ascent_m,
                "segment_descent_m": segment_descent_m,
                "segment_grade_pct": segment_grade_pct,
                "actual_segment_time_s": float(
                    actual_segment_time_s
                ),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_learning_dataset_from_fit(
    uploaded_file,
    activity_id: int,
    activity_name: str,
) -> pd.DataFrame:
    """
    Process one FIT file into its historical 1 m -> 50 m transition dataset.
    """
    record_df = _extract_record_messages(uploaded_file)

    standardized_df = _standardize_raw_fit(
        record_df
    )

    if standardized_df.empty:
        return pd.DataFrame()

    dense_df = _build_dense_trajectory(
        standardized_df
    )

    if dense_df.empty:
        return pd.DataFrame()

    return _build_transition_rows(
        dense_df,
        activity_id=activity_id,
        activity_name=activity_name,
    )


def build_learning_dataset(
    uploaded_files,
) -> pd.DataFrame:
    """
    Process all uploaded FIT files exactly once and concatenate their
    historical transition records.
    """
    if not uploaded_files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for activity_id, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        activity_name = getattr(
            uploaded_file,
            "name",
            f"activity_{activity_id}",
        )

        activity_df = build_learning_dataset_from_fit(
            uploaded_file,
            activity_id=activity_id,
            activity_name=activity_name,
        )

        if not activity_df.empty:
            frames.append(activity_df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(
        frames,
        ignore_index=True,
    )

    dataset = dataset.sort_values(
        [
            "activity_id",
            "distance_from_start_m",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return dataset


def summarize_learning_dataset(
    learning_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return a compact summary for the UI.
    """
    if learning_df is None or learning_df.empty:
        return {
            "n_activities": 0,
            "n_transitions": 0,
            "mean_segment_time_s": np.nan,
            "median_segment_time_s": np.nan,
            "min_segment_time_s": np.nan,
            "max_segment_time_s": np.nan,
        }

    duration = pd.to_numeric(
        learning_df["actual_segment_time_s"],
        errors="coerce",
    ).dropna()

    return {
        "n_activities": int(
            learning_df["activity_id"].nunique()
        ),
        "n_transitions": int(
            len(learning_df)
        ),
        "mean_segment_time_s": float(
            duration.mean()
        ) if not duration.empty else np.nan,
        "median_segment_time_s": float(
            duration.median()
        ) if not duration.empty else np.nan,
        "min_segment_time_s": float(
            duration.min()
        ) if not duration.empty else np.nan,
        "max_segment_time_s": float(
            duration.max()
        ) if not duration.empty else np.nan,
    }
  
