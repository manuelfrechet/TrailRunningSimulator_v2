from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import gpxpy
import gpxpy.gpx


# -----------------------------------------------------------------------------
# V0 configuration
# -----------------------------------------------------------------------------

SEGMENT_LENGTH_M = 50.0


# -----------------------------------------------------------------------------
# GPX parsing
# -----------------------------------------------------------------------------

def _extract_gpx_points(uploaded_file) -> pd.DataFrame:
    """
    Extract GPX track points in chronological/course order.

    Returns:
        distance-compatible raw points with latitude, longitude and elevation.
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
# Distance calculation
# -----------------------------------------------------------------------------

def _calculate_cumulative_distance(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate cumulative horizontal distance in metres from GPX coordinates.
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

    dlat = np.diff(
        lat,
        prepend=lat[0],
    )

    dlon = np.diff(
        lon,
        prepend=lon[0],
    )

    lat_1 = lat

    lat_2 = np.roll(
        lat,
        1,
    )

    lat_2[0] = lat[0]

    haversine_a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_1)
        * np.cos(lat_2)
        * np.sin(dlon / 2.0) ** 2
    )

    haversine_a = np.clip(
        haversine_a,
        0.0,
        1.0,
    )

    delta_distance = (
        2.0
        * earth_radius_m
        * np.arcsin(
            np.sqrt(
                haversine_a
            )
        )
    )

    delta_distance[0] = 0.0

    cumulative_distance = np.cumsum(
        delta_distance
    )

    return pd.Series(
        cumulative_distance,
        index=df.index,
        dtype="float64",
    )


# -----------------------------------------------------------------------------
# Raw GPX normalization
# -----------------------------------------------------------------------------

def _standardize_gpx(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and normalize GPX points.
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
    ).copy()

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # Distance from start.
    # -------------------------------------------------------------------------

    df[
        "distance_from_start_m"
    ] = _calculate_cumulative_distance(
        df
    ).to_numpy()

    # -------------------------------------------------------------------------
    # Elevation cleanup.
    #
    # For V0 we do not invent a sophisticated smoothing/filtering system.
    # Interpolation onto the 50 m grid is deterministic and simple.
    # -------------------------------------------------------------------------

    if df["elevation_m"].notna().sum() >= 2:

        valid = (
            df["elevation_m"]
            .notna()
            .to_numpy()
        )

        x = df[
            "distance_from_start_m"
        ].to_numpy(
            dtype=float
        )

        y = df[
            "elevation_m"
        ].to_numpy(
            dtype=float
        )

        df[
            "elevation_m"
        ] = np.interp(
            x,
            x[valid],
            y[valid],
        )

    else:

        df[
            "elevation_m"
        ] = np.nan

    return df


# -----------------------------------------------------------------------------
# Interpolate course elevation onto a 1 m grid
# -----------------------------------------------------------------------------

def _build_1m_course(
    standardized_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an internal 1 m terrain representation.

    This is only used to calculate the agreed 50 m terrain quantities.
    Prediction still advances 50 m at a time.
    """

    if (
        standardized_df is None
        or standardized_df.empty
    ):
        return pd.DataFrame()

    max_distance_m = float(
        standardized_df[
            "distance_from_start_m"
        ].max()
    )

    if max_distance_m < SEGMENT_LENGTH_M:
        return pd.DataFrame()

    dense_distance = np.arange(
        0.0,
        max_distance_m + 1.0,
        1.0,
    )

    source_distance = standardized_df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    source_elevation = standardized_df[
        "elevation_m"
    ].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(
            source_distance
        )
        & np.isfinite(
            source_elevation
        )
    )

    if valid.sum() < 2:
        raise ValueError(
            "GPX does not contain enough valid elevation data."
        )

    elevation = np.interp(
        dense_distance,
        source_distance[valid],
        source_elevation[valid],
    )

    dense = pd.DataFrame(
        {
            "distance_from_start_m": dense_distance,
            "elevation_m": elevation,
        }
    )

    altitude_delta = np.diff(
        elevation,
        prepend=elevation[0],
    )

    dense[
        "ascent_m"
    ] = np.maximum(
        altitude_delta,
        0.0,
    )

    dense[
        "descent_m"
    ] = np.maximum(
        -altitude_delta,
        0.0,
    )

    dense[
        "cumulative_ascent_m"
    ] = dense[
        "ascent_m"
    ].cumsum()

    dense[
        "cumulative_descent_m"
    ] = dense[
        "descent_m"
    ].cumsum()

    dense[
        "grade_pct"
    ] = (
        altitude_delta
        * 100.0
    )

    dense[
        "grade_pct"
    ] = dense[
        "grade_pct"
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    return dense


# -----------------------------------------------------------------------------
# Build normalized 50 m segments
# -----------------------------------------------------------------------------

def _build_50m_profile(
    course_1m_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the internal 1 m course representation into non-overlapping
    50 m prediction segments.

    Output row distance represents the END of the 50 m segment.
    """

    if (
        course_1m_df is None
        or course_1m_df.empty
    ):
        return pd.DataFrame()

    max_distance_m = float(
        course_1m_df[
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

    rows: list[dict[str, Any]] = []

    altitude = course_1m_df[
        "elevation_m"
    ].to_numpy(
        dtype=float
    )

    ascent = course_1m_df[
        "ascent_m"
    ].to_numpy(
        dtype=float
    )

    descent = course_1m_df[
        "descent_m"
    ].to_numpy(
        dtype=float
    )

    grade = course_1m_df[
        "grade_pct"
    ].to_numpy(
        dtype=float
    )

    cumulative_ascent = course_1m_df[
        "cumulative_ascent_m"
    ].to_numpy(
        dtype=float
    )

    cumulative_descent = course_1m_df[
        "cumulative_descent_m"
    ].to_numpy(
        dtype=float
    )

    distance = course_1m_df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------------------
    # V0 profile creation.
    #
    # Each output row corresponds to:
    #
    #     start = segment_index * 50
    #     end   = start + 50
    #
    # -------------------------------------------------------------------------

    for segment_index in range(
        n_segments
    ):

        start_m = (
            segment_index
            * SEGMENT_LENGTH_M
        )

        end_m = (
            start_m
            + SEGMENT_LENGTH_M
        )

        start_idx = int(
            round(start_m)
        )

        end_idx = int(
            round(end_m)
        )

        if (
            end_idx
            >= len(course_1m_df)
        ):
            break

        segment_ascent = float(
            np.sum(
                ascent[
                    start_idx + 1
                    : end_idx + 1
                ]
            )
        )

        segment_descent = float(
            np.sum(
                descent[
                    start_idx + 1
                    : end_idx + 1
                ]
            )
        )

        segment_grade = float(
            np.mean(
                grade[
                    start_idx + 1
                    : end_idx + 1
                ]
            )
        )

        rows.append(
            {
                "distance_from_start_m": end_m,
                "ascent_m": segment_ascent,
                "descent_m": segment_descent,
                "cumulative_ascent_m": float(
                    cumulative_ascent[
                        end_idx
                    ]
                ),
                "cumulative_descent_m": float(
                    cumulative_descent[
                        end_idx
                    ]
                ),
                "grade_pct": segment_grade,
                "elevation_start_m": float(
                    altitude[start_idx]
                ),
                "elevation_end_m": float(
                    altitude[end_idx]
                ),
                "segment_start_m": start_m,
                "segment_end_m": end_m,
            }
        )

    return pd.DataFrame(
        rows
    )


# -----------------------------------------------------------------------------
# Aid stations
# -----------------------------------------------------------------------------

def add_aid_stations(
    profile_df: pd.DataFrame,
    aid_stations: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """
    Attach aid-station metadata to the nearest normalized 50 m row.

    Expected aid-station input:
        {
            "name": "Aid 1",
            "distance_from_start_m": 10000.0,
            "stop_minutes": 5.0,
        }

    No simulation time is added here.
    The simulator handles stop duration.
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

    Output:
        one row per non-overlapping 50 m prediction segment.
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

    course_1m_df = _build_1m_course(
        standardized_df
    )

    if course_1m_df.empty:
        raise ValueError(
            "GPX is shorter than one complete 50 m segment."
        )

    profile_df = _build_50m_profile(
        course_1m_df
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
    Return a compact GPX profile summary.
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
  
