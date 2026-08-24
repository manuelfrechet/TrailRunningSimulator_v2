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


# =============================================================================
# Raw GPX parsing
# =============================================================================

def _extract_gpx_points(
    uploaded_file,
) -> pd.DataFrame:
    """
    Read raw GPX track points in their original order.

    No resampling.
    No smoothing.
    No interpolation.
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

    Uses point-to-point haversine distance.
    """
    if df is None or df.empty:
        return pd.Series(
            dtype="float64"
        )

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
        * earth_radius
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
    Prepare the raw GPX table.

    Raw terrain is calculated before normalization:

        raw elevation differences
        -> raw ascent/descent
        -> raw cumulative ascent/descent

    The GPX is never resampled to 1 m.
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
    # Raw cumulative horizontal distance
    # -------------------------------------------------------------------------

    df[
        "distance_from_start_m"
    ] = _calculate_cumulative_distance(
        df
    )

    # -------------------------------------------------------------------------
    # Elevation
    #
    # We only fill missing elevation values.
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
    # Collapse consecutive/near-duplicate spatial positions.
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
            "GPX cumulative distance is not strictly increasing "
            "after duplicate-distance points were removed."
        )

    # =============================================================================
    # Raw terrain calculation
    # =============================================================================

    elevation = df[
        "elevation_m"
    ].to_numpy(
        dtype=float
    )

    delta_elevation = np.diff(
        elevation,
        prepend=elevation[0],
    )

    df[
        "ascent_m"
    ] = np.maximum(
        delta_elevation,
        0.0,
    )

    df[
        "descent_m"
    ] = np.maximum(
        -delta_elevation,
        0.0,
    )

    df[
        "cumulative_ascent_m"
    ] = (
        df[
            "ascent_m"
        ]
        .cumsum()
    )

    df[
        "cumulative_descent_m"
    ] = (
        df[
            "descent_m"
        ]
        .cumsum()
    )

    return df


# =============================================================================
# Raw GPX table for diagnostics
# =============================================================================

def load_raw_gpx_table(
    uploaded_file,
) -> pd.DataFrame:
    """
    Return the raw GPX table including raw terrain calculations.
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
            "ascent_m",
            "descent_m",
            "cumulative_ascent_m",
            "cumulative_descent_m",
            "timestamp",
        ]
    ].copy()


# =============================================================================
# Normalized boundary interpolation
# =============================================================================

def _interpolate_boundary_values(
    raw_df: pd.DataFrame,
    boundary_distance_m: float,
) -> dict[str, float]:
    """
    Determine the normalized values at one boundary.

    If a raw point exists at the boundary:
        use the raw values.

    Otherwise:
        find the two surrounding raw points and linearly interpolate:

            elevation
            cumulative ascent
            cumulative descent
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

    cumulative_ascent = raw_df[
        "cumulative_ascent_m"
    ].to_numpy(
        dtype=float
    )

    cumulative_descent = raw_df[
        "cumulative_descent_m"
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

        index = exact_indices[0]

        return {
            "elevation_m": float(
                elevation[index]
            ),
            "cumulative_ascent_m": float(
                cumulative_ascent[index]
            ),
            "cumulative_descent_m": float(
                cumulative_descent[index]
            ),
        }

    # -------------------------------------------------------------------------
    # Find surrounding raw points
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

    def linear_interpolation(
        y0: float,
        y1: float,
    ) -> float:
        return float(
            y0
            + fraction
            * (
                y1
                - y0
            )
        )

    return {
        "elevation_m": linear_interpolation(
            float(
                elevation[
                    left_index
                ]
            ),
            float(
                elevation[
                    right_index
                ]
            ),
        ),
        "cumulative_ascent_m": linear_interpolation(
            float(
                cumulative_ascent[
                    left_index
                ]
            ),
            float(
                cumulative_ascent[
                    right_index
                ]
            ),
        ),
        "cumulative_descent_m": linear_interpolation(
            float(
                cumulative_descent[
                    left_index
                ]
            ),
            float(
                cumulative_descent[
                    right_index
                ]
            ),
        ),
    }


# =============================================================================
# Normalized boundaries
# =============================================================================

def _build_normalized_boundaries(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build normalized boundaries:

        0
        TRANSITION_LENGTH_M
        2 * TRANSITION_LENGTH_M
        ...

    No intermediate normalized points are generated.
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

    rows: list[dict[str, Any]] = []

    for boundary_distance_m in boundary_distances:

        values = (
            _interpolate_boundary_values(
                raw_df,
                float(
                    boundary_distance_m
                ),
            )
        )

        rows.append(
            {
                "distance_from_start_m": float(
                    boundary_distance_m
                ),
                "elevation_m": (
                    values[
                        "elevation_m"
                    ]
                ),
                "cumulative_ascent_m": (
                    values[
                        "cumulative_ascent_m"
                    ]
                ),
                "cumulative_descent_m": (
                    values[
                        "cumulative_descent_m"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Normalized segments
# =============================================================================

def _build_normalized_segments(
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build non-overlapping normalized terrain segments.

    Segment terrain:

        ascent =
            cumulative_ascent(end)
            - cumulative_ascent(start)

        descent =
            cumulative_descent(end)
            - cumulative_descent(start)

        grade =
            net elevation change
            / TRANSITION_LENGTH_M
            * 100

    This preserves ascent/descent accumulated on the raw GPX trajectory,
    including undulations inside the segment.
    """
    if (
        boundary_df is None
        or boundary_df.empty
    ):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for index in range(
        len(boundary_df) - 1
    ):

        start = boundary_df.iloc[
            index
        ]

        end = boundary_df.iloc[
            index + 1
        ]

        start_distance = float(
            start[
                "distance_from_start_m"
            ]
        )

        end_distance = float(
            end[
                "distance_from_start_m"
            ]
        )

        start_elevation = float(
            start[
                "elevation_m"
            ]
        )

        end_elevation = float(
            end[
                "elevation_m"
            ]
        )

        start_cum_ascent = float(
            start[
                "cumulative_ascent_m"
            ]
        )

        end_cum_ascent = float(
            end[
                "cumulative_ascent_m"
            ]
        )

        start_cum_descent = float(
            start[
                "cumulative_descent_m"
            ]
        )

        end_cum_descent = float(
            end[
                "cumulative_descent_m"
            ]
        )

        ascent = (
            end_cum_ascent
            - start_cum_ascent
        )

        descent = (
            end_cum_descent
            - start_cum_descent
        )

        # Numerical noise should not create negative segment quantities.
        ascent = max(
            0.0,
            float(ascent),
        )

        descent = max(
            0.0,
            float(descent),
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
                # Row represents the END of the segment.
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

                "ascent_m": ascent,

                "descent_m": descent,

                "cumulative_ascent_m": (
                    end_cum_ascent
                ),

                "cumulative_descent_m": (
                    end_cum_descent
                ),

                "grade_pct": grade_pct,
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
    ]
    | None = None,
) -> pd.DataFrame:
    """
    Attach aid-station metadata to the nearest normalized endpoint.

    Stop duration is stored but not applied.
    The simulator applies it later.
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
    ]
    | None = None,
) -> pd.DataFrame:
    """
    Build the normalized GPX profile using the project-wide
    TRANSITION_LENGTH_M.

    Processing:

        raw GPX
            ↓
        raw cumulative distance
            ↓
        raw point-to-point ascent/descent
            ↓
        raw cumulative ascent/descent
            ↓
        normalized boundaries at TRANSITION_LENGTH_M
            ↓
        linear interpolation of:
            elevation
            cumulative ascent
            cumulative descent
            ↓
        normalized segments
    """
    raw_df = _prepare_raw_gpx(
        uploaded_file
    )

    if raw_df.empty:
        raise ValueError(
            "No usable GPX points were found."
        )

    boundary_df = (
        _build_normalized_boundaries(
            raw_df
        )
    )

    if boundary_df.empty:
        raise ValueError(
            f"GPX is shorter than "
            f"{TRANSITION_LENGTH_M:.0f} m."
        )

    profile_df = (
        _build_normalized_segments(
            boundary_df
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
    
