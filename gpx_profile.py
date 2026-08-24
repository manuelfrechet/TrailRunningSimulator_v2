from __future__ import annotations

from typing import Any

import gpxpy
import numpy as np
import pandas as pd


DISTANCE_TOLERANCE_M = 0.01


def _extract_gpx_points(uploaded_file) -> pd.DataFrame:
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

    return pd.DataFrame(rows)


def _calculate_cumulative_distance(
    df: pd.DataFrame,
) -> pd.Series:
    latitude = np.radians(
        pd.to_numeric(
            df["latitude"],
            errors="coerce",
        ).to_numpy(dtype=float)
    )

    longitude = np.radians(
        pd.to_numeric(
            df["longitude"],
            errors="coerce",
        ).to_numpy(dtype=float)
    )

    earth_radius_m = 6_371_000.0

    previous_latitude = np.roll(
        latitude,
        1,
    )

    previous_longitude = np.roll(
        longitude,
        1,
    )

    previous_latitude[0] = latitude[0]
    previous_longitude[0] = longitude[0]

    delta_latitude = (
        latitude - previous_latitude
    )

    delta_longitude = (
        longitude - previous_longitude
    )

    a = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(previous_latitude)
        * np.cos(latitude)
        * np.sin(delta_longitude / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0,
    )

    delta_distance = (
        2.0
        * earth_radius_m
        * np.arcsin(
            np.sqrt(a)
        )
    )

    delta_distance[0] = 0.0

    return pd.Series(
        np.cumsum(delta_distance),
        index=df.index,
    )


def load_raw_gpx_table(
    uploaded_file,
) -> pd.DataFrame:

    df = _extract_gpx_points(
        uploaded_file
    )

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    df[
        "distance_from_start_m"
    ] = _calculate_cumulative_distance(
        df
    )

    return df[
        [
            "distance_from_start_m",
            "latitude",
            "longitude",
            "elevation_m",
            "timestamp",
        ]
    ]


def _prepare_raw_gpx(
    uploaded_file,
) -> pd.DataFrame:

    df = load_raw_gpx_table(
        uploaded_file
    )

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    df["elevation_m"] = pd.to_numeric(
        df["elevation_m"],
        errors="coerce",
    )

    valid = np.isfinite(
        df["elevation_m"].to_numpy(
            dtype=float
        )
    )

    if valid.sum() < 2:
        raise ValueError(
            "Not enough elevation data."
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

    df[
        "elevation_m"
    ] = np.interp(
        distance,
        distance[valid],
        elevation[valid],
    )

    # Collapse consecutive duplicate positions.
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
    ].to_numpy(
        dtype=float
    )

    elevation = df[
        "elevation_m"
    ].to_numpy(
        dtype=float
    )

    # Raw terrain quantities.
    delta_elevation = np.diff(
        elevation,
        prepend=elevation[0],
    )

    df["ascent_m"] = np.maximum(
        delta_elevation,
        0.0,
    )

    df["descent_m"] = np.maximum(
        -delta_elevation,
        0.0,
    )

    df[
        "cumulative_ascent_m"
    ] = df[
        "ascent_m"
    ].cumsum()

    df[
        "cumulative_descent_m"
    ] = df[
        "descent_m"
    ].cumsum()

    return df


def _interpolate_boundary(
    raw_df: pd.DataFrame,
    distance_m: float,
) -> dict[str, float]:

    distance = raw_df[
        "distance_from_start_m"
    ].to_numpy(
        dtype=float
    )

    columns = [
        "elevation_m",
        "cumulative_ascent_m",
        "cumulative_descent_m",
    ]

    values = {
        column: raw_df[
            column
        ].to_numpy(
            dtype=float
        )
        for column in columns
    }

    exact = np.where(
        np.abs(
            distance - distance_m
        )
        <= DISTANCE_TOLERANCE_M
    )[0]

    if len(exact) > 0:
        i = exact[0]

        return {
            column: float(
                values[column][i]
            )
            for column in columns
        }

    right = int(
        np.searchsorted(
            distance,
            distance_m,
            side="right",
        )
    )

    left = right - 1

    if (
        left < 0
        or right >= len(distance)
    ):
        raise ValueError(
            f"Cannot bracket boundary {distance_m:.2f} m."
        )

    x0 = distance[left]
    x1 = distance[right]

    fraction = (
        distance_m - x0
    ) / (
        x1 - x0
    )

    return {
        column: float(
            values[column][left]
            + fraction
            * (
                values[column][right]
                - values[column][left]
            )
        )
        for column in columns
    }


def _build_profile(
    raw_df: pd.DataFrame,
    segment_length_m: float,
) -> pd.DataFrame:

    max_distance = float(
        raw_df[
            "distance_from_start_m"
        ].iloc[-1]
    )

    n_segments = int(
        np.floor(
            max_distance
            / segment_length_m
        )
    )

    boundaries = (
        np.arange(
            n_segments + 1,
            dtype=float,
        )
        * segment_length_m
    )

    boundary_rows: list[dict[str, Any]] = []

    for boundary in boundaries:

        values = _interpolate_boundary(
            raw_df,
            float(boundary),
        )

        boundary_rows.append(
            {
                "distance_from_start_m": float(
                    boundary
                ),
                **values,
            }
        )

    boundary_df = pd.DataFrame(
        boundary_rows
    )

    rows: list[dict[str, Any]] = []

    for i in range(
        len(boundary_df) - 1
    ):

        start = boundary_df.iloc[i]
        end = boundary_df.iloc[i + 1]

        elevation_change = (
            float(
                end[
                    "elevation_m"
                ]
            )
            - float(
                start[
                    "elevation_m"
                ]
            )
        )

        rows.append(
            {
                "distance_from_start_m": float(
                    end[
                        "distance_from_start_m"
                    ]
                ),
                "segment_start_m": float(
                    start[
                        "distance_from_start_m"
                    ]
                ),
                "segment_end_m": float(
                    end[
                        "distance_from_start_m"
                    ]
                ),
                "elevation_start_m": float(
                    start["elevation_m"]
                ),
                "elevation_end_m": float(
                    end["elevation_m"]
                ),
                "ascent_m": max(
                    0.0,
                    float(
                        end[
                            "cumulative_ascent_m"
                        ]
                        - start[
                            "cumulative_ascent_m"
                        ]
                    ),
                ),
                "descent_m": max(
                    0.0,
                    float(
                        end[
                            "cumulative_descent_m"
                        ]
                        - start[
                            "cumulative_descent_m"
                        ]
                    ),
                ),
                "cumulative_ascent_m": float(
                    end[
                        "cumulative_ascent_m"
                    ]
                ),
                "cumulative_descent_m": float(
                    end[
                        "cumulative_descent_m"
                    ]
                ),
                "grade_pct": (
                    elevation_change
                    / segment_length_m
                    * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_gpx_profile(
    uploaded_file,
    segment_length_m: float = 50.0,
) -> pd.DataFrame:

    if segment_length_m <= 0:
        raise ValueError(
            "segment_length_m must be positive."
        )

    raw_df = _prepare_raw_gpx(
        uploaded_file
    )

    if raw_df.empty:
        raise ValueError(
            "No usable GPX data."
        )

    return _build_profile(
        raw_df,
        segment_length_m,
    )


def summarize_gpx_profile(
    profile_df: pd.DataFrame,
) -> dict[str, Any]:

    if profile_df.empty:
        return {
            "n_segments": 0,
            "distance_m": 0.0,
            "cumulative_ascent_m": 0.0,
            "cumulative_descent_m": 0.0,
        }

    return {
        "n_segments": len(profile_df),
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
    
