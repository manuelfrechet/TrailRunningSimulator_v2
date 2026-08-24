from __future__ import annotations

from typing import Any

import gpxpy
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

SEGMENT_LENGTH_M = 50.0

# Used only for deciding whether a raw point is effectively exactly on a
# normalized 50 m boundary.
DISTANCE_TOLERANCE_M = 0.01


# =============================================================================
# RAW GPX PARSING
# =============================================================================

def _extract_gpx_points(
    uploaded_file,
) -> pd.DataFrame:
    """
    Read the GPX and return the raw track points.

    No interpolation.
    No resampling.
    No smoothing.
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
# RAW DISTANCE
# =============================================================================

def _calculate_cumulative_distance(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate cumulative horizontal distance from the raw GPX coordinates.

    Uses point-to-point haversine distance.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    latitude = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    ).to_numpy(dtype=float)

    longitude = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if len(df) == 1:
        return pd.Series(
            [0.0],
            index=df.index,
            dtype="float64",
        )

    earth_radius_m = 6_371_000.0

    lat = np.radians(latitude)
    lon = np.radians(longitude)

    previous_lat = np.roll(lat, 1)
    previous_lon = np.roll(lon, 1)

    previous_lat[0] = lat[0]
    previous_lon[0] = lon[0]

    delta_lat = lat - previous_lat
    delta_lon = lon - previous_lon

    haversine_a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(previous_lat)
        * np.cos(lat)
        * np.sin(delta_lon / 2.0) ** 2
    )

    haversine_a = np.clip(
        haversine_a,
        0.0,
        1.0,
    )

    delta_distance_m = (
        2.0
        * earth_radius_m
        * np.arcsin(
            np.sqrt(haversine_a)
        )
    )

    delta_distance_m[0] = 0.0

    cumulative_distance_m = np.cumsum(
        delta_distance_m
    )

    return pd.Series(
        cumulative_distance_m,
        index=df.index,
        dtype="float64",
    )


# =============================================================================
# RAW GPX TABLE
# =============================================================================

def load_raw_gpx_table(
    uploaded_file,
) -> pd.DataFrame:
    """
    Return the raw GPX table with cumulative horizontal distance added.

    This function is diagnostic/display-oriented.

    The raw GPX points themselves are preserved.
    """
    raw_df = _extract_gpx_points(
        uploaded_file
    )

    if raw_df.empty:
        return pd.DataFrame()

    raw_df = raw_df.copy()

    raw_df[
        "distance_from_start_m"
    ] = _calculate_cumulative_distance(
        raw_df
    )

    return raw_df[
        [
            "distance_from_start_m",
            "latitude",
            "longitude",
            "elevation_m",
            "timestamp",
        ]
    ]


# =============================================================================
# RAW GPX STANDARDIZATION
# =============================================================================

def _prepare_raw_gpx(
    uploaded_file,
) -> pd.DataFrame:
    """
    Prepare raw GPX points for normalization.

    Important:
        - preserve original spatial resolution;
        - remove invalid coordinates;
        - calculate raw cumulative distance;
        - collapse exact/near duplicate-distance points;
        - do NOT interpolate the entire GPX.
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

    df[
        "distance_from_start_m"
    ] = _calculate_cumulative_distance(
        df
    )

    # We need at least two elevation measurements.
    if (
        df["elevation_m"]
        .notna()
        .sum()
        < 2
    ):
        raise ValueError(
            "GPX does not contain enough elevation data."
        )

    # Fill missing elevation values only.
    #
    # This is NOT resampling. It only fills missing values at existing raw
    # points.
    distance = df[
        "distance_from_start_m"
    ].to_numpy(dtype=float)

    elevation = df[
        "elevation_m"
    ].to_numpy(dtype=float)

    valid_elevation = np.isfinite(
        elevation
    )

    df[
        "elevation_m"
    ] = np.interp(
        distance,
        distance[valid_elevation],
        elevation[valid_elevation],
    )

    # Collapse consecutive/near-duplicate spatial positions.
    distance = df[
        "distance_from_start_m"
    ].to_numpy(dtype=float)

    keep = np.concatenate(
        [
            np.array([True]),
            np.diff(distance)
            > DISTANCE_TOLERANCE_M,
        ]
    )

    df = (
        df.loc[keep]
        .reset_index(drop=True)
    )

    distance = df[
        "distance_from_start_m"
    ].to_numpy(dtype=float)

    if len(distance) < 2:
        raise ValueError(
            "GPX does not contain enough distinct spatial points."
        )

    if np.any(
        np.diff(distance) <= 0.0
    ):
        raise ValueError(
            "GPX cumulative distance is not strictly increasing."
        )

    return df


# =============================================================================
# 50 m BOUNDARY ELEVATION
# =============================================================================

def _boundary_elevation(
    raw_df: pd.DataFrame,
    boundary_distance_m: float,
) -> float:
    """
    Return elevation at one normalized 50 m boundary.

    Rule:
        - exact raw GPX point -> use its elevation;
        - otherwise -> linear interpolation between the two surrounding
          raw GPX points.
    """
    distance = raw_df[
        "distance_from_start_m"
    ].to_numpy(dtype=float)

    elevation = raw_df[
        "elevation_m"
    ].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # Exact raw point.
    # -------------------------------------------------------------------------

    exact_indices = np.where(
        np.abs(
            distance
            - boundary_distance_m
        )
        <= DISTANCE_TOLERANCE_M
    )[0]

    if len(exact_indices) > 0:
        return float(
            elevation[
                exact_indices[0]
            ]
        )

    # -------------------------------------------------------------------------
    # Find surrounding raw points.
    # -------------------------------------------------------------------------

    right_index = int(
        np.searchsorted(
            distance,
            boundary_distance_m,
            side="right",
        )
    )

    left_index = right_index - 1

    if (
        left_index < 0
        or right_index >= len(distance)
    ):
        raise ValueError(
            f"Could not bracket normalized boundary "
            f"{boundary_distance_m:.2f} m."
        )

    x0 = float(
        distance[left_index]
    )
    x1 = float(
        distance[right_index]
    )

    y0 = float(
        elevation[left_index]
    )
    y1 = float(
        elevation[right_index]
    )

    if x1 <= x0:
        raise ValueError(
            "Invalid raw GPX distance interval."
        )

    # -------------------------------------------------------------------------
    # Linear interpolation only.
    # -------------------------------------------------------------------------

    fraction = (
        boundary_distance_m - x0
    ) / (
        x1 - x0
    )

    return float(
        y0 + fraction * (y1 - y0)
    )


# =============================================================================
# NORMALIZED 50 m BOUNDARIES
# =============================================================================

def _build_50m_boundaries(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create only the normalized 50 m boundary points:

        0
        50
        100
        150
        ...

    No intermediate points are generated.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    max_distance_m = float(
        raw_df[
            "distance_from_start_m"
        ].max()
    )

    n_complete_segments = int(
        np.floor(
            max_distance_m
            / SEGMENT_LENGTH_M
        )
    )

    if n_complete_segments <= 0:
        return pd.DataFrame()

    distances = (
        np.arange(
            n_complete_segments + 1,
            dtype=float,
        )
        * SEGMENT_LENGTH_M
    )

    elevations = np.array(
        [
            _boundary_elevation(
                raw_df,
                float(distance),
            )
            for distance in distances
        ],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "distance_from_start_m": distances,
            "elevation_m": elevations,
        }
    )


# =============================================================================
# SEGMENT TERRAIN
# =============================================================================

def _build_50m_segments(
    raw_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the normalized 50 m terrain table.

    Each row represents:

        start -> start + 50 m

    Raw GPX points inside the interval are used to calculate ascent/descent.
    The normalized boundary elevations define the segment grade.
    """
    if (
        raw_df is None
        or raw_df.empty
        or boundary_df is None
        or boundary_df.empty
    ):
        return pd.DataFrame()

    raw_distance = raw_df[
        "distance_from_start_m"
    ].to_numpy(dtype=float)

    raw_elevation = raw_df[
        "elevation_m"
    ].to_numpy(dtype=float)

    boundary_distance = boundary_df[
        "distance_from_start_m"
    ].to_numpy(dtype=float)

    boundary_elevation = boundary_df[
        "elevation_m"
    ].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []

    for i in range(
        len(boundary_distance) - 1
    ):
        start_m = float(
            boundary_distance[i]
        )

        end_m = float(
            boundary_distance[i + 1]
        )

        start_elevation_m = float(
            boundary_elevation[i]
        )

        end_elevation_m = float(
            boundary_elevation[i + 1]
        )

        # ---------------------------------------------------------------------
        # Raw points inside the 50 m segment.
        # ---------------------------------------------------------------------

        interior_mask = (
            (raw_distance > start_m)
            & (raw_distance < end_m)
        )

        interior_distance = raw_distance[
            interior_mask
        ]

        interior_elevation = raw_elevation[
            interior_mask
        ]

        # ---------------------------------------------------------------------
        # Reconstruct only this segment:
        #
        # normalized boundary
        # raw interior points
        # normalized boundary
        # ---------------------------------------------------------------------

        segment_distance = np.concatenate(
            [
                np.array(
                    [start_m],
                    dtype=float,
                ),
                interior_distance,
                np.array(
                    [end_m],
                    dtype=float,
                ),
            ]
        )

        segment_elevation = np.concatenate(
            [
                np.array(
                    [start_elevation_m],
                    dtype=float,
                ),
                interior_elevation,
                np.array(
                    [end_elevation_m],
                    dtype=float,
                ),
            ]
        )

        order = np.argsort(
            segment_distance
        )

        segment_distance = (
            segment_distance[order]
        )

        segment_elevation = (
            segment_elevation[order]
        )

        # Remove duplicate positions inside the segment.
        if len(segment_distance) >= 2:
            keep = np.concatenate(
                [
                    np.array([True]),
                    np.diff(
                        segment_distance
                    )
                    > DISTANCE_TOLERANCE_M,
                ]
            )

            segment_distance = (
                segment_distance[keep]
            )

            segment_elevation = (
                segment_elevation[keep]
            )

        elevation_delta = np.diff(
            segment_elevation
        )

        ascent_m = float(
            np.sum(
                np.maximum(
                    elevation_delta,
                    0.0,
                )
            )
        )

        descent_m = float(
            np.sum(
                np.maximum(
                    -elevation_delta,
                    0.0,
                )
            )
        )

        # ---------------------------------------------------------------------
        # Grade is the signed NET grade over the normalized 50 m.
        # ---------------------------------------------------------------------

        grade_pct = float(
            (
                end_elevation_m
                - start_elevation_m
            )
            / SEGMENT_LENGTH_M
            * 100.0
        )

        rows.append(
            {
                "distance_from_start_m": end_m,
                "ascent_m": ascent_m,
                "descent_m": descent_m,
                "cumulative_ascent_m": np.nan,
                "cumulative_descent_m": np.nan,
                "grade_pct": grade_pct,
                "elevation_start_m": start_elevation_m,
                "elevation_end_m": end_elevation_m,
                "segment_start_m": start_m,
                "segment_end_m": end_m,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    result[
        "cumulative_ascent_m"
    ] = result[
        "ascent_m"
    ].cumsum()

    result[
        "cumulative_descent_m"
    ] = result[
        "descent_m"
    ].cumsum()

    return result


# =============================================================================
# AID STATIONS
# =============================================================================

def add_aid_stations(
    profile_df: pd.DataFrame,
    aid_stations: list[
        dict[str, Any]
    ] | None = None,
) -> pd.DataFrame:
    """
    Attach aid-station information to normalized 50 m endpoints.

    Stop duration is only stored here.
    It is applied later by the simulator.
    """
    if profile_df is None or profile_df.empty:
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
    ].to_numpy(dtype=float)

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
# PUBLIC API
# =============================================================================

def build_gpx_profile(
    uploaded_file,
    aid_stations: list[
        dict[str, Any]
    ] | None = None,
) -> pd.DataFrame:
    """
    Build the normalized 50 m GPX profile.

    V0 normalization:

        raw GPX
            ↓
        cumulative raw distance
            ↓
        50 m boundaries
            ↓
        exact raw elevation OR linear interpolation
            ↓
        50 m terrain rows

    There is no 1 m GPX resampling.
    """
    raw_df = _prepare_raw_gpx(
        uploaded_file
    )

    if raw_df.empty:
        raise ValueError(
            "No usable GPX points were found."
        )

    boundary_df = _build_50m_boundaries(
        raw_df
    )

    if boundary_df.empty:
        raise ValueError(
            "GPX is shorter than 50 m."
        )

    profile_df = _build_50m_segments(
        raw_df,
        boundary_df,
    )

    if profile_df.empty:
        raise ValueError(
            "No normalized 50 m GPX segments were created."
        )

    return add_aid_stations(
        profile_df,
        aid_stations,
    )


def summarize_gpx_profile(
    profile_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return a compact summary of the normalized profile.
    """
    if profile_df is None or profile_df.empty:
        return {
            "n_segments": 0,
            "distance_m": 0.0,
            "cumulative_ascent_m": 0.0,
            "cumulative_descent_m": 0.0,
        }

    return {
        "n_segments": int(
            len(profile_df)
        ),
        "distance_m": float(
            profile_df[
                "distance_from_start_m"
            ].max()
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
