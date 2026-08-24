from __future__ import annotations

from typing import Any

import gpxpy
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# V0 configuration
# -----------------------------------------------------------------------------

SEGMENT_LENGTH_M = 50.0

# Implementation detail for boundaries that do not coincide with a raw GPX
# point. We use a local polynomial through the nearest raw points.
#
# This is deliberately isolated here so it can be changed later without
# changing the rest of the GPX pipeline.
POLYNOMIAL_DEGREE = 3
BOUNDARY_MATCH_TOLERANCE_M = 0.01


# -----------------------------------------------------------------------------
# GPX parsing
# -----------------------------------------------------------------------------

def _extract_gpx_points(uploaded_file) -> pd.DataFrame:
    """
    Extract raw GPX track points in course order.

    No resampling or interpolation is performed here.
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


# -----------------------------------------------------------------------------
# Horizontal distance
# -----------------------------------------------------------------------------

def _calculate_cumulative_distance(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate cumulative horizontal GPX distance in metres.
    """
    if df is None or df.empty:
        return pd.Series(
            dtype="float64"
        )

    lat = np.radians(
        pd.to_numeric(
            df["latitude"],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )
    )

    lon = np.radians(
        pd.to_numeric(
            df["longitude"],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )
    )

    if len(df) == 1:
        return pd.Series(
            [0.0],
            index=df.index,
            dtype="float64",
        )

    earth_radius_m = 6_371_000.0

    delta_lat = np.diff(
        lat,
        prepend=lat[0],
    )

    delta_lon = np.diff(
        lon,
        prepend=lon[0],
    )

    previous_lat = np.roll(
        lat,
        1,
    )

    previous_lat[0] = lat[0]

    haversine_a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat)
        * np.cos(previous_lat)
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


# -----------------------------------------------------------------------------
# Raw GPX preparation
# -----------------------------------------------------------------------------

def _standardize_gpx(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw GPX points while preserving their original resolution.
    """
    if raw_df is None or raw_df.empty:
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

    # We need elevation throughout the track.
    if df["elevation_m"].notna().sum() < 2:
        raise ValueError(
            "GPX does not contain enough elevation data."
        )

    valid_elevation = (
        df["elevation_m"]
        .notna()
        .to_numpy()
    )

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

    # Fill isolated missing raw elevations by interpolation in the raw
    # distance coordinate. This is only for missing source values; it is not
    # GPX resampling.
    df[
        "elevation_m"
    ] = np.interp(
        distance,
        distance[valid_elevation],
        elevation[valid_elevation],
    )

    return df


# -----------------------------------------------------------------------------
# Polynomial boundary interpolation
# -----------------------------------------------------------------------------

def _interpolate_boundary_elevation(
    raw_df: pd.DataFrame,
    boundary_distance_m: float,
) -> float:
    """
    Return elevation at one normalized 50 m boundary.

    Rule:
        - if a raw GPX point exists at the boundary, use it directly;
        - otherwise use local polynomial interpolation.

    The polynomial is local, not a global high-order fit.
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
    # Exact raw point.
    # -------------------------------------------------------------------------

    exact_indices = np.where(
        np.abs(
            distance
            - boundary_distance_m
        )
        <= BOUNDARY_MATCH_TOLERANCE_M
    )[0]

    if len(exact_indices) > 0:
        return float(
            elevation[
                exact_indices[0]
            ]
        )

    # -------------------------------------------------------------------------
    # Find local neighbouring raw points.
    # -------------------------------------------------------------------------

    insertion_index = int(
        np.searchsorted(
            distance,
            boundary_distance_m,
        )
    )

    n_points = len(distance)

    degree = min(
        POLYNOMIAL_DEGREE,
        n_points - 1,
    )

    required_points = degree + 1

    left_count = required_points // 2

    start_index = (
        insertion_index
        - left_count
    )

    start_index = max(
        0,
        start_index,
    )

    end_index = (
        start_index
        + required_points
    )

    if end_index > n_points:
        end_index = n_points
        start_index = (
            n_points
            - required_points
        )

    start_index = max(
        0,
        start_index,
    )

    local_distance = distance[
        start_index:end_index
    ]

    local_elevation = elevation[
        start_index:end_index
    ]

    # -------------------------------------------------------------------------
    # Local polynomial fit.
    # -------------------------------------------------------------------------

    if len(local_distance) < 2:
        raise ValueError(
            "Not enough GPX points to interpolate a boundary."
        )

    local_x = (
        local_distance
        - boundary_distance_m
    )

    coefficients = np.polyfit(
        local_x,
        local_elevation,
        degree,
    )

    interpolated = np.polyval(
        coefficients,
        0.0,
    )

    return float(
        interpolated
    )


def _build_normalized_boundaries(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct exactly the 50 m boundary grid.

    Example:
        0
        50
        100
        150
        ...

    Elevation is obtained at each boundary using the agreed rule.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    max_distance_m = float(
        raw_df[
            "distance_from_start_m"
        ].max()
    )

    n_segments = int(
        np.floor(
            max_distance_m
            / SEGMENT_LENGTH_M
        )
    )

    if n_segments <= 0:
        return pd.DataFrame()

    boundaries = (
        np.arange(
            0,
            n_segments + 1,
        )
        * SEGMENT_LENGTH_M
    )

    elevations = [
        _interpolate_boundary_elevation(
            raw_df,
            float(boundary),
        )
        for boundary in boundaries
    ]

    return pd.DataFrame(
        {
            "distance_from_start_m": boundaries,
            "elevation_m": elevations,
        }
    )


# -----------------------------------------------------------------------------
# Segment terrain calculation
# -----------------------------------------------------------------------------

def _calculate_segment_terrain(
    raw_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate ascent, descent and grade for each non-overlapping 50 m segment.

    Important:
        - boundary elevations define the exact start/end elevations;
        - ascent/descent use the terrain path through the raw GPX points inside
          the segment, with the normalized boundaries inserted.

    Therefore a segment can have:
        ascent > 0
        descent > 0
        grade ~= 0
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
    ].to_numpy(
        dtype=float
    )

    raw_elevation = raw_df[
        "elevation_m"
    ].to_numpy(
        dtype=float
    )

    boundary_distance = boundary_df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    boundary_elevation = boundary_df[
        "elevation_m"
    ].to_numpy(
        dtype=float
    )

    rows: list[dict[str, Any]] = []

    for segment_index in range(
        len(boundary_distance) - 1
    ):
        start_distance = float(
            boundary_distance[
                segment_index
            ]
        )

        end_distance = float(
            boundary_distance[
                segment_index + 1
            ]
        )

        start_elevation = float(
            boundary_elevation[
                segment_index
            ]
        )

        end_elevation = float(
            boundary_elevation[
                segment_index + 1
            ]
        )

        # ---------------------------------------------------------------------
        # Raw GPX points strictly inside the segment.
        # ---------------------------------------------------------------------

        inside_mask = (
            (raw_distance > start_distance)
            & (
                raw_distance
                < end_distance
            )
        )

        inside_distance = raw_distance[
            inside_mask
        ]

        inside_elevation = raw_elevation[
            inside_mask
        ]

        # ---------------------------------------------------------------------
        # Reconstruct the elevation path for this 50 m segment:
        #
        # boundary start
        # + raw interior points
        # + boundary end
        # ---------------------------------------------------------------------

        segment_distance = np.concatenate(
            [
                np.array(
                    [start_distance],
                    dtype=float,
                ),
                inside_distance,
                np.array(
                    [end_distance],
                    dtype=float,
                ),
            ]
        )

        segment_elevation = np.concatenate(
            [
                np.array(
                    [start_elevation],
                    dtype=float,
                ),
                inside_elevation,
                np.array(
                    [end_elevation],
                    dtype=float,
                ),
            ]
        )

        order = np.argsort(
            segment_distance,
            kind="mergesort",
        )

        segment_distance = (
            segment_distance[
                order
            ]
        )

        segment_elevation = (
            segment_elevation[
                order
            ]
        )

        # Remove duplicate distances if any exist at a boundary.
        unique_mask = np.concatenate(
            [
                np.array(
                    [True]
                ),
                np.diff(
                    segment_distance
                ) > 1e-9,
            ]
        )

        segment_distance = (
            segment_distance[
                unique_mask
            ]
        )

        segment_elevation = (
            segment_elevation[
                unique_mask
            ]
        )

        elevation_delta = np.diff(
            segment_elevation
        )

        segment_ascent = float(
            np.sum(
                np.maximum(
                    elevation_delta,
                    0.0,
                )
            )
        )

        segment_descent = float(
            np.sum(
                np.maximum(
                    -elevation_delta,
                    0.0,
                )
            )
        )

        # Net segment grade is based on the normalized boundary elevations.
        segment_grade_pct = float(
            (
                end_elevation
                - start_elevation
            )
            / SEGMENT_LENGTH_M
            * 100.0
        )

        rows.append(
            {
                "distance_from_start_m": end_distance,
                "ascent_m": segment_ascent,
                "descent_m": segment_descent,
                "grade_pct": segment_grade_pct,
                "elevation_start_m": start_elevation,
                "elevation_end_m": end_elevation,
                "segment_start_m": start_distance,
                "segment_end_m": end_distance,
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


# -----------------------------------------------------------------------------
# Aid stations
# -----------------------------------------------------------------------------

def add_aid_stations(
    profile_df: pd.DataFrame,
    aid_stations: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """
    Attach aid-station metadata to the nearest normalized 50 m endpoint.

    Aid-station time is NOT added here.
    The simulator handles it.
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

    profile_distances = result[
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
                    profile_distances
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


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_gpx_profile(
    uploaded_file,
    aid_stations: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Build the normalized 50 m GPX profile.

    There is NO 1 m GPX resampling.

    The only interpolation is elevation interpolation at the normalized
    50 m boundaries.
    """
    raw_df = _extract_gpx_points(
        uploaded_file
    )

    if raw_df.empty:
        raise ValueError(
            "No GPX track points were found."
        )

    standardized_df = _standardize_gpx(
        raw_df
    )

    if standardized_df.empty:
        raise ValueError(
            "GPX could not be standardized."
        )

    boundary_df = _build_normalized_boundaries(
        standardized_df
    )

    if boundary_df.empty:
        raise ValueError(
            "GPX is shorter than one complete 50 m segment."
        )

    profile_df = _calculate_segment_terrain(
        standardized_df,
        boundary_df,
    )

    if profile_df.empty:
        raise ValueError(
            "No normalized 50 m GPX segments were created."
        )

    profile_df = add_aid_stations(
        profile_df,
        aid_stations=aid_stations,
    )

    return profile_df


def summarize_gpx_profile(
    profile_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return a compact normalized GPX summary.
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
            ].max()
        ),
        "cumulative_descent_m": float(
            profile_df[
                "cumulative_descent_m"
            ].max()
        ),
    }
