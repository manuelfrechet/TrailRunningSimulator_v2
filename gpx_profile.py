from __future__ import annotations

from typing import Any

import gpxpy
import numpy as np
import pandas as pd

from aid_stations import (
    AidStation,
    normalize_aid_stations,
)

from config import GPX_SEGMENT_LENGTH_M


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

    if isinstance(
        raw,
        bytes,
    ):
        raw = raw.decode(
            "utf-8"
        )

    gpx = gpxpy.parse(
        raw
    )

    rows: list[
        dict[str, Any]
    ] = []

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

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Raw horizontal distance
# =============================================================================

def _calculate_cumulative_distance(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate cumulative horizontal distance from raw GPX coordinates.

    Uses point-to-point haversine distance.
    """

    if (
        df is None
        or df.empty
    ):

        return pd.Series(
            dtype="float64"
        )

    latitude = pd.to_numeric(
        df[
            "latitude"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    longitude = pd.to_numeric(
        df[
            "longitude"
        ],
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
            delta_lat
            / 2.0
        )
        ** 2
        + np.cos(
            previous_lat
        )
        * np.cos(
            lat
        )
        * np.sin(
            delta_lon
            / 2.0
        )
        ** 2
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

    cumulative_distance_m = (
        np.cumsum(
            delta_distance_m
        )
    )

    return pd.Series(
        cumulative_distance_m,
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
    Prepare the raw GPX trajectory.

    Raw spatial resolution is preserved.

    Elevation remains at raw GPX granularity until normalized boundaries
    are created.
    """

    raw_df = _extract_gpx_points(
        uploaded_file
    )

    if raw_df.empty:

        return pd.DataFrame()

    df = raw_df.copy()

    # -------------------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------------------

    df[
        "latitude"
    ] = pd.to_numeric(
        df[
            "latitude"
        ],
        errors="coerce",
    )

    df[
        "longitude"
    ] = pd.to_numeric(
        df[
            "longitude"
        ],
        errors="coerce",
    )

    df[
        "elevation_m"
    ] = pd.to_numeric(
        df[
            "elevation_m"
        ],
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
    # Raw cumulative horizontal distance.
    # -------------------------------------------------------------------------

    df[
        "distance_from_start_m"
    ] = _calculate_cumulative_distance(
        df
    )

    # -------------------------------------------------------------------------
    # Elevation.
    #
    # We only fill missing elevation values at existing raw positions.
    # We do NOT spatially resample the trajectory.
    # -------------------------------------------------------------------------

    distance = (
        df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
    )

    elevation = (
        df[
            "elevation_m"
        ].to_numpy(
            dtype=float
        )
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
    # Remove consecutive / near-duplicate horizontal positions.
    # -------------------------------------------------------------------------

    distance = (
        df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
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

    distance = (
        df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
    )

    if len(distance) < 2:

        raise ValueError(
            "GPX does not contain enough distinct spatial points."
        )

    if np.any(
        np.diff(
            distance
        ) <= 0.0
    ):

        raise ValueError(
            "GPX cumulative distance is not strictly increasing "
            "after duplicate-distance points were removed."
        )

    return df


# =============================================================================
# Raw GPX diagnostic table
# =============================================================================

def load_raw_gpx_table(
    uploaded_file,
) -> pd.DataFrame:
    """
    Return the raw GPX table with calculated cumulative distance.

    No normalized ascent/descent calculations are added.
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
# Boundary elevation interpolation
# =============================================================================

def _interpolate_boundary_elevation(
    raw_df: pd.DataFrame,
    boundary_distance_m: float,
) -> float:
    """
    Determine elevation at one normalized GPX boundary.

    Rules:

        exact raw point
            -> use raw elevation

        otherwise
            -> use the two surrounding raw points
            -> linearly interpolate elevation
    """

    distance = (
        raw_df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
    )

    elevation = (
        raw_df[
            "elevation_m"
        ].to_numpy(
            dtype=float
        )
    )

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

    if len(
        exact_indices
    ) > 0:

        return float(
            elevation[
                exact_indices[0]
            ]
        )

    # -------------------------------------------------------------------------
    # Bracket with two surrounding raw points.
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
            "Invalid raw GPX distance interval during "
            "boundary interpolation."
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
    Create normalized GPX boundaries every GPX_SEGMENT_LENGTH_M.

    Only elevation is interpolated.
    """

    if (
        raw_df is None
        or raw_df.empty
    ):

        return pd.DataFrame()

    max_distance_m = float(
        raw_df[
            "distance_from_start_m"
        ].iloc[-1]
    )

    n_complete_segments = int(
        np.floor(
            max_distance_m
            / GPX_SEGMENT_LENGTH_M
        )
    )

    if n_complete_segments <= 0:

        return pd.DataFrame()

    boundary_distances = (
        np.arange(
            n_complete_segments + 1,
            dtype=float,
        )
        * GPX_SEGMENT_LENGTH_M
    )

    rows: list[
        dict[str, float]
    ] = []

    for boundary_distance_m in (
        boundary_distances
    ):

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
    normalized_elevation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate ascent, descent, cumulative ascent, cumulative descent and grade
    FROM THE NORMALIZED ELEVATION PROFILE.
    """

    if (
        normalized_elevation_df is None
        or normalized_elevation_df.empty
    ):

        return pd.DataFrame()

    distance = (
        normalized_elevation_df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )
    )

    elevation = (
        normalized_elevation_df[
            "elevation_m"
        ].to_numpy(
            dtype=float
        )
    )

    if len(distance) < 2:

        return pd.DataFrame()

    delta_elevation = np.diff(
        elevation
    )

    ascent = np.maximum(
        delta_elevation,
        0.0,
    )

    descent = np.maximum(
        -delta_elevation,
        0.0,
    )

    cumulative_ascent = np.cumsum(
        ascent
    )

    cumulative_descent = np.cumsum(
        descent
    )

    rows: list[
        dict[str, Any]
    ] = []

    for index in range(
        len(
            delta_elevation
        )
    ):

        start_distance = float(
            distance[
                index
            ]
        )

        end_distance = float(
            distance[
                index + 1
            ]
        )

        start_elevation = float(
            elevation[
                index
            ]
        )

        end_elevation = float(
            elevation[
                index + 1
            ]
        )

        grade_pct = (
            (
                end_elevation
                - start_elevation
            )
            / GPX_SEGMENT_LENGTH_M
            * 100.0
        )

        rows.append(
            {
                # -------------------------------------------------------------
                # The row represents the END of the segment.
                # -------------------------------------------------------------

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
                    ascent[
                        index
                    ]
                ),

                "descent_m": float(
                    descent[
                        index
                    ]
                ),

                "cumulative_ascent_m": float(
                    cumulative_ascent[
                        index
                    ]
                ),

                "cumulative_descent_m": float(
                    cumulative_descent[
                        index
                    ]
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
        AidStation
        | dict[str, Any]
    ]
    | None = None,
) -> pd.DataFrame:
    """
    Attach aid-station metadata to the nearest normalized GPX endpoint.

    Accepts BOTH:

        AidStation objects
        dictionaries

    The operational application uses AidStation objects.

    Stop duration is only stored here.
    The simulator applies the stop to the elapsed clocks.
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

    result[
        "aid_station_distance_m"
    ] = np.nan

    # -------------------------------------------------------------------------
    # Normalize the public input interface.
    # -------------------------------------------------------------------------

    stations = normalize_aid_stations(
        aid_stations
    )

    if not stations:

        return result

    profile_distances = pd.to_numeric(
        result[
            "distance_from_start_m"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        profile_distances
    ).all():

        raise ValueError(
            "Profile contains invalid distances."
        )

    race_distance_m = float(
        profile_distances[-1]
    )

    # -------------------------------------------------------------------------
    # normalize_aid_stations() sorts and validates the stations.
    #
    # Validate against the actual normalized GPX course length here.
    # -------------------------------------------------------------------------

    for station in stations:

        if (
            station.distance_from_start_m
            > race_distance_m
        ):

            raise ValueError(
                f"Aid station '{station.name}' at "
                f"{station.distance_from_start_m:.1f} m is beyond "
                f"the normalized GPX distance "
                f"{race_distance_m:.1f} m."
            )

    # -------------------------------------------------------------------------
    # Map each station to its nearest normalized endpoint.
    # -------------------------------------------------------------------------

    for station in stations:

        station_distance_m = float(
            station.distance_from_start_m
        )

        nearest_index = int(
            np.argmin(
                np.abs(
                    profile_distances
                    - station_distance_m
                )
            )
        )

        result.loc[
            nearest_index,
            "aid_station_name",
        ] = station.name

        result.loc[
            nearest_index,
            "aid_station_stop_min",
        ] = station.stop_minutes

        result.loc[
            nearest_index,
            "aid_station_distance_m",
        ] = station_distance_m

    return result


# =============================================================================
# Public API
# =============================================================================

def build_gpx_profile(
    uploaded_file,
    aid_stations: list[
        AidStation
        | dict[str, Any]
    ]
    | None = None,
) -> pd.DataFrame:
    """
    Build the normalized GPX terrain profile.

    Current V0 methodology:

        raw GPX
            ↓
        raw cumulative horizontal distance
            ↓
        boundaries every GPX_SEGMENT_LENGTH_M
            ↓
        linear interpolation of elevation ONLY
            ↓
        terrain calculated from normalized elevation
            ↓
        aid stations attached to normalized endpoints
    """

    raw_df = _prepare_raw_gpx(
        uploaded_file
    )

    if raw_df.empty:

        raise ValueError(
            "No usable GPX points were found."
        )

    normalized_elevation_df = (
        _build_normalized_elevation_profile(
            raw_df
        )
    )

    if normalized_elevation_df.empty:

        raise ValueError(
            f"GPX is shorter than "
            f"{GPX_SEGMENT_LENGTH_M:.0f} m."
        )

    profile_df = (
        _calculate_normalized_terrain(
            normalized_elevation_df
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
