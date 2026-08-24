from __future__ import annotations

from typing import Any

import gpxpy
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# V0 configuration
# -----------------------------------------------------------------------------

SEGMENT_LENGTH_M = 50.0
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
        return pd.Series(dtype="float64")

    lat_deg = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    ).to_numpy(dtype=float)

    lon_deg = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if len(df) == 1:
        return pd.Series(
            [0.0],
            index=df.index,
            dtype="float64",
        )

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

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


# -----------------------------------------------------------------------------
# Raw GPX preparation
# -----------------------------------------------------------------------------

def _standardize_gpx(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the raw GPX points without changing their spatial resolution.
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
    ).reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    df[
        "distance_from_start_m"
    ] = _calculate_cumulative_distance(
        df
    )

    # We need at least two valid elevation points.
    valid_elevation = (
        df["elevation_m"]
        .notna()
        .to_numpy()
    )

    if valid_elevation.sum() < 2:
        raise ValueError(
            "GPX does not contain enough valid elevation data."
        )

    # Fill missing elevation values only.
    #
    # This does NOT resample the GPX. It only ensures that every original
    # track point has an elevation value if the source omitted isolated values.
    distance = df[
        "distance_from_start_m"
    ].to_numpy(dtype=float)

    elevation = df[
        "elevation_m"
    ].to_numpy(dtype=float)

    df[
        "elevation_m"
    ] = np.interp(
        distance,
        distance[valid_elevation],
        elevation[valid_elevation],
    )

    # Distance must be monotonic for boundary interpolation.
    distance_diffs = np.diff(
        df[
            "distance_from_start_m"
        ].to_numpy(dtype=float)
    )

    if np.any(
        distance_diffs <= 0.0
    ):
        raise ValueError(
            "GPX cumulative distance is not strictly increasing."
        )

    return df


# -----------------------------------------------------------------------------
# 50 m boundary elevation
# -----------------------------------------------------------------------------

def _interpolate_boundary_elevation(
    raw_df: pd.DataFrame,
    boundary_distance_m: float,
) -> float:
    """
    Get the elevation at one normalized 50 m boundary.

    Rule:
        1. Use the raw GPX elevation when a raw point exists at the boundary.
        2. Otherwise linearly interpolate between the two surrounding raw
           GPX points.
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

    exact = np.where(
        np.abs(
            distance
            - boundary_distance_m
        )
        <= BOUNDARY_MATCH_TOLERANCE_M
    )[0]

    if len(exact) > 0:
        return float(
            elevation[exact[0]]
        )

    # -------------------------------------------------------------------------
    # Boundary must be inside the raw GPX range.
    # -------------------------------------------------------------------------

    if (
        boundary_distance_m < distance[0]
        or boundary_distance_m > distance[-1]
    ):
        raise ValueError(
            f"Boundary {boundary_distance_m:.2f} m "
            "is outside the GPX distance range."
        )

    # -------------------------------------------------------------------------
    # Surrounding raw points.
    # -------------------------------------------------------------------------

    right_idx = int(
        np.searchsorted(
            distance,
            boundary_distance_m,
            side="right",
        )
    )

    left_idx = right_idx - 1

    if (
        left_idx < 0
        or right_idx >= len(distance)
    ):
        raise ValueError(
            "Could not find surrounding GPX points "
            f"for boundary {boundary_distance_m:.2f} m."
        )

    x0 = distance[left_idx]
    x1 = distance[right_idx]

    y0 = elevation[left_idx]
    y1 = elevation[right_idx]

    if x1 <= x0:
        raise ValueError(
            "GPX distance is not strictly increasing "
            "around normalized boundary."
        )

    fraction = (
        boundary_distance_m - x0
    ) / (
        x1 - x0
    )

    return float(
        y0
        + fraction * (
            y1 - y0
        )
    )


def _build_normalized_boundaries(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the normalized 50 m boundary grid.

    Example:
        0
        50
        100
        150
        ...

    Only these boundaries are created.
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

    boundaries = (
        np.arange(
            n_complete_segments + 1,
            dtype=float,
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
# 50 m segment terrain
# -----------------------------------------------------------------------------

def _calculate_segment_terrain(
    raw_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate terrain quantities for each normalized 50 m segment.

    The normalized endpoints come from the boundary elevations.

    Interior raw GPX elevations are retained as the raw terrain path.

    No 1 m resampling is performed.
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
        # Keep all raw points strictly inside the segment.
        # ---------------------------------------------------------------------

        interior_mask = (
            (raw_distance > start_distance)
            & (
                raw_distance
                < end_distance
            )
        )

        interior_distance = raw_distance[
            interior_mask
        ]

        interior_elevation = raw_elevation[
            interior_mask
        ]

        # ---------------------------------------------------------------------
        # Reconstruct the actual raw-elevation path for this 50 m interval,
        # inserting only the two normalized boundary points.
        # ---------------------------------------------------------------------

        segment_distance = np.concatenate(
            [
                np.array(
                    [start_distance],
                    dtype=float,
                ),
                interior_distance,
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
                interior_elevation,
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
            segment_distance[order]
        )

        segment_elevation = (
            segment_elevation[order]
        )

        # Remove duplicated boundary distances.
        unique_mask = np.concatenate(
            [
                np.array([True]),
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

        # ---------------------------------------------------------------------
        # Cumulative ascent/descent INSIDE this 50 m path.
        #
        # These are kept separate so an undulating segment can have both:
        #
        #     ascent > 0
        #     descent > 0
        #
        # even when its net elevation change is near zero.
        # ---------------------------------------------------------------------

        segment_ascent_m = float(
            np.sum(
                np.maximum(
                    elevation_delta,
                    0.0,
                )
            )
        )

        segment_descent_m = float(
            np.sum(
                np.maximum(
                    -elevation_delta,
                    0.0,
                )
            )
        )

        # ---------------------------------------------------------------------
        # Segment grade:
        #
        # signed net elevation change over exactly 50 m.
        #
        # Example:
        #     +10 m then -10 m
        #     => ascent 10 m
        #     => descent 10 m
        #     => grade 0 %
        # ---------------------------------------------------------------------

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
                "ascent_m": segment_ascent_m,
                "descent_m": segment_descent_m,
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
    Attach aid-station information to normalized 50 m endpoints.

    Aid-station duration is NOT applied here.
    The simulator will add it to both cumulative clocks.
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

    GPX normalization is performed directly at the 50 m boundaries.

    There is:
        - no 1 m GPX resampling;
        - no polynomial interpolation;
        - no smoothing;
        - no elevation filtering.

    Boundary elevation:
        exact raw point -> raw elevation
        otherwise -> linear interpolation
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

    return add_aid_stations(
        profile_df,
        aid_stations=aid_stations,
    )


def summarize_gpx_profile(
    profile_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return a compact summary of a normalized GPX profile.
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
    
