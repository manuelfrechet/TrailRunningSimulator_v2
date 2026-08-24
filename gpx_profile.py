from __future__ import annotations

from typing import Any

import gpxpy
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

SEGMENT_LENGTH_M = 50.0

# Used only to identify effectively identical spatial positions.
DISTANCE_TOLERANCE_M = 0.01


# =============================================================================
# Raw GPX parsing
# =============================================================================

def _extract_gpx_points(
    uploaded_file,
) -> pd.DataFrame:
    """
    Extract raw GPX track points in their original course order.

    No resampling.
    No smoothing.
    No interpolation.
    """

    uploaded_file.seek(0)

    raw = uploaded_file.read()

    if isinstance(raw, bytes):
        raw = raw.decode(
            "utf-8"
        )

    gpx = gpxpy.parse(
        raw
    )

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

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Raw GPX cumulative horizontal distance
# =============================================================================

def _calculate_cumulative_distance(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate cumulative horizontal course distance using the raw GPX
    coordinates.

    Distance is calculated point-to-point with the haversine formula.
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

    latitude_rad = np.radians(
        latitude
    )

    longitude_rad = np.radians(
        longitude
    )

    earth_radius_m = 6_371_000.0

    previous_latitude = np.roll(
        latitude_rad,
        1,
    )

    previous_longitude = np.roll(
        longitude_rad,
        1,
    )

    previous_latitude[0] = (
        latitude_rad[0]
    )

    previous_longitude[0] = (
        longitude_rad[0]
    )

    delta_latitude = (
        latitude_rad
        - previous_latitude
    )

    delta_longitude = (
        longitude_rad
        - previous_longitude
    )

    haversine_a = (
        np.sin(
            delta_latitude / 2.0
        ) ** 2
        + np.cos(
            previous_latitude
        )
        * np.cos(
            latitude_rad
        )
        * np.sin(
            delta_longitude / 2.0
        ) ** 2
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

def _standardize_gpx(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw GPX points for 50 m normalization.

    Important:
        - raw spatial resolution is preserved;
        - duplicate/near-duplicate horizontal positions are collapsed;
        - there is no 1 m resampling;
        - there is no terrain smoothing.
    """

    if raw_df is None or raw_df.empty:

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

    # -------------------------------------------------------------------------
    # Coordinates are mandatory.
    # -------------------------------------------------------------------------

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
    # Calculate raw cumulative horizontal distance.
    # -------------------------------------------------------------------------

    df[
        "distance_from_start_m"
    ] = _calculate_cumulative_distance(
        df
    )

    # -------------------------------------------------------------------------
    # Elevation must exist.
    # -------------------------------------------------------------------------

    valid_elevation = (
        df[
            "elevation_m"
        ]
        .notna()
        .to_numpy()
    )

    if valid_elevation.sum() < 2:

        raise ValueError(
            "GPX does not contain enough valid elevation data."
        )

    # -------------------------------------------------------------------------
    # Fill only isolated missing elevation values.
    #
    # This is NOT course resampling.
    # Existing raw GPX positions remain unchanged.
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
    # Collapse consecutive raw points that do not create a new horizontal
    # course position.
    #
    # Example:
    #
    # distance:
    #   1234.50
    #   1234.50
    #   1234.50
    #   1237.20
    #
    # becomes:
    #   1234.50
    #   1237.20
    #
    # We are not inventing distance.
    # -------------------------------------------------------------------------

    distance = df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    if len(distance) >= 2:

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

    # -------------------------------------------------------------------------
    # Verify the resulting spatial coordinate.
    # -------------------------------------------------------------------------

    distance = df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    if len(distance) < 2:

        raise ValueError(
            "GPX does not contain enough distinct course positions."
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

    return df


# =============================================================================
# Elevation at normalized 50 m boundaries
# =============================================================================

def _interpolate_boundary_elevation(
    raw_df: pd.DataFrame,
    boundary_distance_m: float,
) -> float:
    """
    Return elevation at one normalized 50 m boundary.

    Rule:

        1. If a raw GPX point exists at that distance:
               use its raw elevation.

        2. Otherwise:
               use linear interpolation between the surrounding
               raw GPX points.
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
    # Exact / effectively exact raw GPX position.
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
    # Boundary must be inside the GPX range.
    # -------------------------------------------------------------------------

    if (
        boundary_distance_m
        < distance[0]
        or boundary_distance_m
        > distance[-1]
    ):

        raise ValueError(
            f"Boundary {boundary_distance_m:.2f} m "
            "is outside the GPX distance range."
        )

    # -------------------------------------------------------------------------
    # Find the two surrounding raw GPX points.
    # -------------------------------------------------------------------------

    right_index = int(
        np.searchsorted(
            distance,
            boundary_distance_m,
            side="right",
        )
    )

    left_index = (
        right_index - 1
    )

    if (
        left_index < 0
        or right_index
        >= len(distance)
    ):

        raise ValueError(
            "Could not find surrounding GPX points "
            f"for boundary {boundary_distance_m:.2f} m."
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
            "Invalid GPX distance interval during "
            "50 m boundary interpolation."
        )

    # -------------------------------------------------------------------------
    # Linear interpolation.
    # -------------------------------------------------------------------------

    fraction = (
        boundary_distance_m
        - x0
    ) / (
        x1
        - x0
    )

    boundary_elevation = (
        y0
        + fraction
        * (
            y1
            - y0
        )
    )

    return float(
        boundary_elevation
    )


# =============================================================================
# Build normalized 50 m boundaries
# =============================================================================

def _build_normalized_boundaries(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build:

        0
        50
        100
        150
        ...

    up to the final complete 50 m segment.

    No intermediate normalized points are created.
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

    boundary_distance = (
        np.arange(
            n_complete_segments
            + 1,
            dtype=float,
        )
        * SEGMENT_LENGTH_M
    )

    boundary_elevation = np.array(
        [
            _interpolate_boundary_elevation(
                raw_df,
                float(distance),
            )
            for distance
            in boundary_distance
        ],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "distance_from_start_m": (
                boundary_distance
            ),
            "elevation_m": (
                boundary_elevation
            ),
        }
    )


# =============================================================================
# Calculate terrain inside each 50 m segment
# =============================================================================

def _calculate_segment_terrain(
    raw_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate terrain for each non-overlapping 50 m segment.

    Example:

        0 -> 50
        50 -> 100
        100 -> 150

    The exact normalized start/end elevations are inserted around the raw GPX
    points that occur inside the segment.

    There is no 1 m resampling.
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
        len(
            boundary_distance
        )
        - 1
    ):

        start_distance_m = float(
            boundary_distance[
                segment_index
            ]
        )

        end_distance_m = float(
            boundary_distance[
                segment_index
                + 1
            ]
        )

        start_elevation_m = float(
            boundary_elevation[
                segment_index
            ]
        )

        end_elevation_m = float(
            boundary_elevation[
                segment_index
                + 1
            ]
        )

        # ---------------------------------------------------------------------
        # Raw points strictly inside this 50 m segment.
        # -------------------------------------------------------------------------

        interior_mask = (
            (
                raw_distance
                > start_distance_m
            )
            & (
                raw_distance
                < end_distance_m
            )
        )

        interior_distance = (
            raw_distance[
                interior_mask
            ]
        )

        interior_elevation = (
            raw_elevation[
                interior_mask
            ]
        )

        # ---------------------------------------------------------------------
        # Terrain path:
        #
        # normalized start boundary
        # raw interior GPX points
        # normalized end boundary
        # -------------------------------------------------------------------------

        segment_distance = np.concatenate(
            [
                np.array(
                    [
                        start_distance_m
                    ],
                    dtype=float,
                ),
                interior_distance,
                np.array(
                    [
                        end_distance_m
                    ],
                    dtype=float,
                ),
            ]
        )

        segment_elevation = np.concatenate(
            [
                np.array(
                    [
                        start_elevation_m
                    ],
                    dtype=float,
                ),
                interior_elevation,
                np.array(
                    [
                        end_elevation_m
                    ],
                    dtype=float,
                ),
            ]
        )

        # ---------------------------------------------------------------------
        # Sort defensively.
        # -------------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Remove any duplicate distance positions inside the reconstructed
        # segment.
        # -------------------------------------------------------------------------

        if len(
            segment_distance
        ) >= 2:

            keep = np.concatenate(
                [
                    np.array(
                        [True]
                    ),
                    np.diff(
                        segment_distance
                    )
                    > DISTANCE_TOLERANCE_M,
                ]
            )

            segment_distance = (
                segment_distance[
                    keep
                ]
            )

            segment_elevation = (
                segment_elevation[
                    keep
                ]
            )

        # ---------------------------------------------------------------------
        # Elevation transitions through the raw terrain path.
        # -------------------------------------------------------------------------

        elevation_delta = np.diff(
            segment_elevation
        )

        # ---------------------------------------------------------------------
        # Segment ascent.
        # -------------------------------------------------------------------------

        segment_ascent_m = float(
            np.sum(
                np.maximum(
                    elevation_delta,
                    0.0,
                )
            )
        )

        # ---------------------------------------------------------------------
        # Segment descent.
        # -------------------------------------------------------------------------

        segment_descent_m = float(
            np.sum(
                np.maximum(
                    -elevation_delta,
                    0.0,
                )
            )
        )

        # ---------------------------------------------------------------------
        # Grade.
        #
        # This is the NET signed elevation change over the normalized 50 m
        # segment.
        #
        # Therefore a segment may have:
        #
        # ascent = 10 m
        # descent = 10 m
        # grade = 0 %
        # -------------------------------------------------------------------------

        segment_grade_pct = float(
            (
                end_elevation_m
                - start_elevation_m
            )
            / SEGMENT_LENGTH_M
            * 100.0
        )

        rows.append(
            {
                "distance_from_start_m": (
                    end_distance_m
                ),
                "ascent_m": (
                    segment_ascent_m
                ),
                "descent_m": (
                    segment_descent_m
                ),
                "cumulative_ascent_m": np.nan,
                "cumulative_descent_m": np.nan,
                "grade_pct": (
                    segment_grade_pct
                ),
                "elevation_start_m": (
                    start_elevation_m
                ),
                "elevation_end_m": (
                    end_elevation_m
                ),
                "segment_start_m": (
                    start_distance_m
                ),
                "segment_end_m": (
                    end_distance_m
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    # -------------------------------------------------------------------------
    # Cumulative terrain.
    # -------------------------------------------------------------------------

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
# Aid stations
# =============================================================================

def add_aid_stations(
    profile_df: pd.DataFrame,
    aid_stations: list[
        dict[str, Any]
    ]
    | None,
) -> pd.DataFrame:
    """
    Attach aid-station metadata to the nearest normalized 50 m endpoint.

    Stop duration is NOT added to predicted time here.
    That belongs to the simulator.
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

    profile_distance = result[
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

        station_distance_m = float(
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
            station_distance_m
        ):

            continue

        nearest_index = int(
            np.argmin(
                np.abs(
                    profile_distance
                    - station_distance_m
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
    Build the normalized GPX profile.

    Frozen V0 rules:

        raw GPX
            ↓
        raw cumulative horizontal distance
            ↓
        normalized 50 m boundaries
            ↓
        exact raw elevation when available
        otherwise linear interpolation
            ↓
        50 m terrain table

    There is NO:
        - 1 m GPX resampling
        - polynomial interpolation
        - terrain smoothing
        - terrain filtering
    """

    raw_df = _extract_gpx_points(
        uploaded_file
    )

    if raw_df.empty:

        raise ValueError(
            "No GPX track points were found."
        )

    standardized_df = (
        _standardize_gpx(
            raw_df
        )
    )

    if standardized_df.empty:

        raise ValueError(
            "GPX could not be standardized."
        )

    boundary_df = (
        _build_normalized_boundaries(
            standardized_df
        )
    )

    if boundary_df.empty:

        raise ValueError(
            "GPX is shorter than one complete 50 m segment."
        )

    profile_df = (
        _calculate_segment_terrain(
            standardized_df,
            boundary_df,
        )
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
    Return the basic normalized GPX summary.
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
    
