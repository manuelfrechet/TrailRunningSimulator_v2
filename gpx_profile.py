from __future__ import annotations

from typing import Any

import gpxpy
import numpy as np
import pandas as pd

from config import TRANSITION_LENGTH_M


# =============================================================================
# Configuration
# =============================================================================

DISTANCE_TOLERANCE_M = 0.01
EARTH_RADIUS_M = 6_371_000.0


# =============================================================================
# Raw GPX parsing
# =============================================================================

def _extract_gpx_points(
    uploaded_file,
) -> pd.DataFrame:
    """
    Read raw GPX track points in their original order.

    No terrain resampling or smoothing is performed here.
    """
    uploaded_file.seek(0)

    raw = uploaded_file.read()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    gpx = gpxpy.parse(raw)

    rows: list[dict[str, Any]] = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                rows.append(
                    {
                        "latitude": point.latitude,
                        "longitude": point.longitude,
                        "elevation_m": point.elevation,
                        "timestamp": point.time,
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =============================================================================
# Raw horizontal distance
# =============================================================================

def _calculate_cumulative_distance(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate cumulative horizontal distance from raw GPX coordinates.
    """
    if df is None or df.empty:
        return pd.Series(
            dtype="float64"
        )

    latitude = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    longitude = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if len(df) == 1:
        return pd.Series(
            [0.0],
            index=df.index,
            dtype="float64",
        )

    lat = np.radians(
        latitude
    )

    lon = np.radians(
        longitude
    )

    previous_lat = np.roll(
        lat,
        1,
    )

    previous_lon = np.roll(
        lon,
        1,
    )

    previous_lat[0] = lat[0]
    previous_lon[0] = lon[0]

    delta_lat = (
        lat
        - previous_lat
    )

    delta_lon = (
        lon
        - previous_lon
    )

    haversine_a = (
        np.sin(
            delta_lat / 2.0
        ) ** 2
        + np.cos(previous_lat)
        * np.cos(lat)
        * np.sin(
            delta_lon / 2.0
        ) ** 2
    )

    haversine_a = np.clip(
        haversine_a,
        0.0,
        1.0,
    )

    delta_distance_m = (
        2.0
        * EARTH_RADIUS_M
        * np.arcsin(
            np.sqrt(
                haversine_a
            )
        )
    )

    delta_distance_m[0] = 0.0

    return pd.Series(
        np.cumsum(
            delta_distance_m
        ),
        index=df.index,
        dtype="float64",
    )


# =============================================================================
# Raw GPX preparation
# =============================================================================

def _prepare_raw_gpx(
    uploaded_file,
) -> pd.DataFrame:
    """
    Prepare the raw GPX spatial table.

    This step only establishes:
        - raw course distance
        - raw elevation

    It does NOT calculate ascent/descent for the model.
    """
    raw_df = _extract_gpx_points(
        uploaded_file
    )

    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["elevation_m"] = pd.to_numeric(
        df["elevation_m"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    ).reset_index(
        drop=True
    )

    if df.empty:
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Raw cumulative distance
    # -------------------------------------------------------------------------

    df[
        "distance_from_start_m"
    ] = _calculate_cumulative_distance(
        df
    )

    # -------------------------------------------------------------------------
    # Elevation
    #
    # Fill missing elevations only at existing raw points.
    # We do not create new spatial points.
    # -------------------------------------------------------------------------

    distance = df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    elevation = df[
        "elevation_m"
    ].to_numpy(
        dtype=float
    )

    valid_elevation = np.isfinite(
        elevation
    )

    if valid_elevation.sum() < 2:
        raise ValueError(
            "GPX does not contain enough valid elevation data."
        )

    df[
        "elevation_m"
    ] = np.interp(
        distance,
        distance[
            valid_elevation
        ],
        elevation[
            valid_elevation
        ],
    )

    # -------------------------------------------------------------------------
    # Collapse consecutive / effectively duplicate horizontal positions.
    # -------------------------------------------------------------------------

    distance = df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    keep = np.concatenate(
        [
            np.array(
                [True]
            ),
            np.diff(
                distance
            )
            > DISTANCE_TOLERANCE_M,
        ]
    )

    df = (
        df.loc[
            keep
        ]
        .reset_index(
            drop=True
        )
    )

    distance = df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    if len(distance) < 2:
        raise ValueError(
            "GPX does not contain enough distinct spatial points."
        )

    if np.any(
        np.diff(
            distance
        )
        <= 0.0
    ):
        raise ValueError(
            "GPX cumulative distance is not strictly increasing."
        )

    return df


# =============================================================================
# Raw GPX diagnostic table
# =============================================================================

def load_raw_gpx_table(
    uploaded_file,
) -> pd.DataFrame:
    """
    Return the raw GPX table for inspection.
    """
    raw_df = _prepare_raw_gpx(
        uploaded_file
    )

    if raw_df.empty:
        return pd.DataFrame()

    return raw_df[
        [
            "distance_from_start_m",
            "latitude",
            "longitude",
            "elevation_m",
            "timestamp",
        ]
    ].copy()


# =============================================================================
# Normalized elevation interpolation
# =============================================================================

def _interpolate_boundary_elevation(
    raw_df: pd.DataFrame,
    boundary_distance_m: float,
) -> float:
    """
    Determine elevation at one normalized boundary.

    Rule:
        - exact raw GPX point -> use raw elevation
        - otherwise -> linear interpolation between the two surrounding
          raw GPX points
    """
    distance = raw_df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    elevation = raw_df[
        "elevation_m"
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------------------
    # Exact raw point
    # -------------------------------------------------------------------------

    exact_indices = np.where(
        np.abs(
            distance
            - boundary_distance_m
        )
        <= DISTANCE_TOLERANCE_M
    )[0]

    if len(
        exact_indices
    ) > 0:

        return float(
            elevation[
                exact_indices[0]
            ]
        )

    # -------------------------------------------------------------------------
    # Find two surrounding raw points
    # -------------------------------------------------------------------------

    right_index = int(
        np.searchsorted(
            distance,
            boundary_distance_m,
            side="right",
        )
    )

    left_index = (
        right_index
        - 1
    )

    if (
        left_index < 0
        or right_index >= len(distance)
    ):
        raise ValueError(
            f"Could not bracket normalized boundary "
            f"{boundary_distance_m:.2f} m."
        )

    x0 = float(
        distance[
            left_index
        ]
    )

    x1 = float(
        distance[
            right_index
        ]
    )

    y0 = float(
        elevation[
            left_index
        ]
    )

    y1 = float(
        elevation[
            right_index
        ]
    )

    if x1 <= x0:
        raise ValueError(
            "Invalid raw GPX distance interval."
        )

    fraction = (
        boundary_distance_m
        - x0
    ) / (
        x1
        - x0
    )

    return float(
        y0
        + fraction
        * (
            y1
            - y0
        )
    )


# =============================================================================
# Normalized elevation profile
# =============================================================================

def _build_normalized_elevation_profile(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the normalized elevation profile.

    Boundaries are:

        0
        TRANSITION_LENGTH_M
        2 * TRANSITION_LENGTH_M
        ...

    ONLY elevation is interpolated from the raw GPX.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    max_distance_m = float(
        raw_df[
            "distance_from_start_m"
        ].iloc[-1]
    )

    n_complete_segments = int(
        np.floor(
            max_distance_m
            / TRANSITION_LENGTH_M
        )
    )

    if n_complete_segments <= 0:
        return pd.DataFrame()

    boundary_distances = (
        np.arange(
            n_complete_segments + 1,
            dtype=float,
        )
        * TRANSITION_LENGTH_M
    )

    rows: list[dict[str, float]] = []

    for boundary_distance_m in boundary_distances:

        rows.append(
            {
                "distance_from_start_m": float(
                    boundary_distance_m
                ),
                "elevation_m": (
                    _interpolate_boundary_elevation(
                        raw_df,
                        float(
                            boundary_distance_m
                        ),
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Normalized terrain calculation
# =============================================================================

def _calculate_normalized_terrain(
    elevation_profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate ascent/descent from the normalized elevation profile ONLY.

    This is the key V0 definition.

    Raw sub-segment elevation fluctuations are deliberately NOT accumulated.

    For each normalized segment:

        delta_elevation =
            elevation_end
            - elevation_start

        ascent =
            max(delta_elevation, 0)

        descent =
            max(-delta_elevation, 0)

    Then cumulative ascent/descent are calculated from the normalized
    segments.
    """
    if (
        elevation_profile_df is None
        or elevation_profile_df.empty
    ):
        return pd.DataFrame()

    distance = elevation_profile_df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    elevation = elevation_profile_df[
        "elevation_m"
    ].to_numpy(
        dtype=float
    )

    if len(distance) < 2:
        return pd.DataFrame()

    delta_elevation = np.diff(
        elevation
    )

    segment_ascent = np.maximum(
        delta_elevation,
        0.0,
    )

    segment_descent = np.maximum(
        -delta_elevation,
        0.0,
    )

    cumulative_ascent = np.cumsum(
        segment_ascent
    )

    cumulative_descent = np.cumsum(
        segment_descent
    )

    rows: list[dict[str, Any]] = []

    for i in range(
        len(delta_elevation)
    ):
        start_distance = float(
            distance[i]
        )

        end_distance = float(
            distance[i + 1]
        )

        start_elevation = float(
            elevation[i]
        )

        end_elevation = float(
            elevation[i + 1]
        )

        grade_pct = (
            (
                end_elevation
                - start_elevation
            )
            / TRANSITION_LENGTH_M
            * 100.0
        )

        rows.append(
            {
                # Row represents the end of the segment.
                "distance_from_start_m": (
                    end_distance
                ),
                "segment_start_m": (
                    start_distance
                ),
                "segment_end_m": (
                    end_distance
                ),
                "elevation_start_m": (
                    start_elevation
                ),
                "elevation_end_m": (
                    end_elevation
                ),
                "ascent_m": float(
                    segment_ascent[i]
                ),
                "descent_m": float(
                    segment_descent[i]
                ),
                "cumulative_ascent_m": float(
                    cumulative_ascent[i]
                ),
                "cumulative_descent_m": float(
                    cumulative_descent[i]
                ),
                "grade_pct": float(
                    grade_pct
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Aid stations
# =============================================================================

def add_aid_stations(
    profile_df: pd.DataFrame,
    aid_stations: list[
        dict[str, Any]
    ] | None = None,
) -> pd.DataFrame:
    """
    Attach aid-station metadata to the nearest normalized endpoint.

    Stop duration is stored but not applied here.
    """
    if (
        profile_df is None
        or profile_df.empty
    ):
        return pd.DataFrame()

    result = profile_df.copy()

    result[
        "aid_station_name"
    ] = ""

    result[
        "aid_station_stop_min"
    ] = np.nan

    if not aid_stations:
        return result

    distances = result[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    for station in aid_stations:

        name = str(
            station.get(
                "name",
                "",
            )
        )

        station_distance = float(
            station.get(
                "distance_from_start_m",
                np.nan,
            )
        )

        stop_minutes = float(
            station.get(
                "stop_minutes",
                0.0,
            )
        )

        if not np.isfinite(
            station_distance
        ):
            continue

        nearest_index = int(
            np.argmin(
                np.abs(
                    distances
                    - station_distance
                )
            )
        )

        result.loc[
            nearest_index,
            "aid_station_name",
        ] = name

        result.loc[
            nearest_index,
            "aid_station_stop_min",
        ] = max(
            0.0,
            stop_minutes,
        )

    return result


# =============================================================================
# Public API
# =============================================================================

def build_gpx_profile(
    uploaded_file,
    aid_stations: list[
        dict[str, Any]
    ] | None = None,
) -> pd.DataFrame:
    """
    Build the normalized GPX terrain profile.

    V0 process:

        raw GPX
            ↓
        raw horizontal distance
            ↓
        normalized 50/1000/... m boundaries
            ↓
        interpolate ONLY elevation
            ↓
        calculate ascent/descent from normalized elevation
            ↓
        calculate cumulative ascent/descent
    """
    raw_df = _prepare_raw_gpx(
        uploaded_file
    )

    if raw_df.empty:
        raise ValueError(
            "No usable GPX points were found."
        )

    normalized_elevation = (
        _build_normalized_elevation_profile(
            raw_df
        )
    )

    if normalized_elevation.empty:
        raise ValueError(
            f"GPX is shorter than "
            f"{TRANSITION_LENGTH_M:.0f} m."
        )

    profile_df = (
        _calculate_normalized_terrain(
            normalized_elevation
        )
    )

    if profile_df.empty:
        raise ValueError(
            "No normalized GPX segments were created."
        )

    return add_aid_stations(
        profile_df,
        aid_stations,
    )


def summarize_gpx_profile(
    profile_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return a compact normalized GPX summary.
    """
    if (
        profile_df is None
        or profile_df.empty
    ):
        return {
            "n_segments": 0,
            "distance_m": 0.0,
            "cumulative_ascent_m": 0.0,
            "cumulative_descent_m": 0.0,
        }

    return {
        "n_segments": int(
            len(
                profile_df
            )
        ),
        "distance_m": float(
            profile_df[
                "distance_from_start_m"
            ].iloc[-1]
        ),
        "cumulative_ascent_m": float(
            profile_df[
                "cumulative_ascent_m"
            ].iloc[-1]
        ),
        "cumulative_descent_m": float(
            profile_df[
                "cumulative_descent_m"
            ].iloc[-1]
        ),
    }
    
