from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Aid station data model
# =============================================================================

@dataclass(frozen=True)
class AidStation:
    """
    One aid station on the race course.

    distance_from_start_m:
        Distance from race start.

    stop_minutes:
        Expected stationary time at the station.

    The stop time is NOT running time.
    """

    name: str
    distance_from_start_m: float
    stop_minutes: float


# =============================================================================
# Validation
# =============================================================================

def validate_aid_stations(
    aid_stations: list[AidStation],
    race_distance_m: float | None = None,
) -> None:
    """
    Validate an ordered list of aid stations.

    Rules:

        - name must be non-empty;
        - distance must be finite and >= 0;
        - stop duration must be finite and >= 0;
        - station distances must be strictly increasing;
        - stations must not be beyond the race distance when the race distance
          is supplied.
    """

    previous_distance = -np.inf

    for index, station in enumerate(
        aid_stations
    ):
        if not isinstance(
            station,
            AidStation,
        ):
            raise ValueError(
                f"Aid station {index + 1} is not an AidStation object."
            )

        name = station.name.strip()

        if not name:
            raise ValueError(
                f"Aid station {index + 1} has an empty name."
            )

        distance = float(
            station.distance_from_start_m
        )

        stop_minutes = float(
            station.stop_minutes
        )

        if not np.isfinite(
            distance
        ):
            raise ValueError(
                f"Aid station '{name}' has an invalid distance."
            )

        if distance < 0.0:
            raise ValueError(
                f"Aid station '{name}' has a negative distance."
            )

        if distance <= previous_distance:
            raise ValueError(
                "Aid-station distances must be strictly increasing."
            )

        if not np.isfinite(
            stop_minutes
        ):
            raise ValueError(
                f"Aid station '{name}' has an invalid stop duration."
            )

        if stop_minutes < 0.0:
            raise ValueError(
                f"Aid station '{name}' has a negative stop duration."
            )

        if (
            race_distance_m is not None
            and distance > float(
                race_distance_m
            )
        ):
            raise ValueError(
                f"Aid station '{name}' is beyond the race distance."
            )

        previous_distance = distance


# =============================================================================
# Input normalization
# =============================================================================

def normalize_aid_stations(
    aid_stations: list[
        AidStation | dict[str, Any]
    ]
    | None,
) -> list[AidStation]:
    """
    Convert flexible input dictionaries into AidStation objects.

    Accepted dictionary fields:

        name
        distance_from_start_m
        stop_minutes

    An empty/None input produces an empty station list.
    """

    if not aid_stations:
        return []

    normalized: list[
        AidStation
    ] = []

    for index, station in enumerate(
        aid_stations
    ):

        if isinstance(
            station,
            AidStation,
        ):

            normalized.append(
                station
            )
            continue

        if not isinstance(
            station,
            dict,
        ):
            raise ValueError(
                f"Aid station {index + 1} must be an AidStation "
                "or a dictionary."
            )

        name = str(
            station.get(
                "name",
                "",
            )
        ).strip()

        distance = station.get(
            "distance_from_start_m"
        )

        stop_minutes = station.get(
            "stop_minutes",
            0.0,
        )

        if distance is None:
            raise ValueError(
                f"Aid station '{name or index + 1}' "
                "is missing distance_from_start_m."
            )

        normalized.append(
            AidStation(
                name=name,
                distance_from_start_m=float(
                    distance
                ),
                stop_minutes=float(
                    stop_minutes
                ),
            )
        )

    normalized.sort(
        key=lambda station: (
            station.distance_from_start_m
        )
    )

    validate_aid_stations(
        normalized
    )

    return normalized


# =============================================================================
# Normalized profile mapping
# =============================================================================

def attach_aid_stations_to_profile(
    profile_df: pd.DataFrame,
    aid_stations: list[
        AidStation | dict[str, Any]
    ]
    | None = None,
) -> pd.DataFrame:
    """
    Attach aid-station information to the nearest normalized GPX endpoint.

    The profile rows represent segment endpoints.

    Example:

        normalized profile:
            100 m
            200 m
            300 m

        station:
            275 m

        station is attached to:
            300 m endpoint

    The original station distance is retained so the mapping remains visible.

    Stop duration is NOT added to any model prediction here.
    """

    if (
        profile_df is None
        or profile_df.empty
    ):
        return pd.DataFrame()

    result = profile_df.copy()

    # -------------------------------------------------------------------------
    # Normalize station fields on every row.
    # -------------------------------------------------------------------------

    result[
        "aid_station_name"
    ] = ""

    result[
        "aid_station_stop_min"
    ] = np.nan

    result[
        "aid_station_distance_m"
    ] = np.nan

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

    validate_aid_stations(
        stations,
        race_distance_m=race_distance_m,
    )

    # -------------------------------------------------------------------------
    # Attach each station to the nearest normalized endpoint.
    # -------------------------------------------------------------------------

    for station in stations:

        station_distance = float(
            station.distance_from_start_m
        )

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
        ] = station.name

        result.loc[
            nearest_index,
            "aid_station_stop_min",
        ] = station.stop_minutes

        result.loc[
            nearest_index,
            "aid_station_distance_m",
        ] = station_distance

    return result


# =============================================================================
# Extraction helpers for the simulator
# =============================================================================

def get_aid_station_at_profile_row(
    row: pd.Series,
) -> AidStation | None:
    """
    Return the aid station represented by a profile row.

    Returns None when there is no aid station at that endpoint.
    """

    if (
        "aid_station_name"
        not in row.index
    ):
        return None

    name = row[
        "aid_station_name"
    ]

    if (
        name is None
        or pd.isna(name)
        or str(name).strip() == ""
    ):
        return None

    if (
        "aid_station_stop_min"
        not in row.index
    ):
        stop_minutes = 0.0

    else:

        stop_value = row[
            "aid_station_stop_min"
        ]

        stop_minutes = (
            0.0
            if pd.isna(
                stop_value
            )
            else float(
                stop_value
            )
        )

    if (
        "aid_station_distance_m"
        not in row.index
    ):
        station_distance = float(
            row[
                "distance_from_start_m"
            ]
        )

    else:

        station_distance_value = row[
            "aid_station_distance_m"
        ]

        station_distance = (
            float(
                row[
                    "distance_from_start_m"
                ]
            )
            if pd.isna(
                station_distance_value
            )
            else float(
                station_distance_value
            )
        )

    return AidStation(
        name=str(name),
        distance_from_start_m=station_distance,
        stop_minutes=max(
            0.0,
            stop_minutes,
        ),
    )


def aid_station_stop_seconds(
    station: AidStation | None,
) -> float:
    """
    Convert an aid-station stop duration into seconds.
    """
    if station is None:
        return 0.0

    return max(
        0.0,
        float(
            station.stop_minutes
        )
        * 60.0,
    )


# =============================================================================
# Summary helpers
# =============================================================================

def summarize_aid_stations(
    aid_stations: list[
        AidStation | dict[str, Any]
    ]
    | None,
) -> pd.DataFrame:
    """
    Convert aid-station definitions into a simple table for the UI.
    """

    stations = normalize_aid_stations(
        aid_stations
    )

    if not stations:
        return pd.DataFrame(
            columns=[
                "name",
                "distance_from_start_m",
                "stop_minutes",
                "stop_seconds",
            ]
        )

    return pd.DataFrame(
        [
            {
                "name": station.name,
                "distance_from_start_m": (
                    station.distance_from_start_m
                ),
                "stop_minutes": (
                    station.stop_minutes
                ),
                "stop_seconds": (
                    station.stop_minutes
                    * 60.0
                ),
            }
            for station in stations
        ]
    )


def total_aid_station_stop_seconds(
    aid_stations: list[
        AidStation | dict[str, Any]
    ]
    | None,
) -> float:
    """
    Return total expected stationary time across all aid stations.
    """

    stations = normalize_aid_stations(
        aid_stations
    )

    return float(
        sum(
            aid_station_stop_seconds(
                station
            )
            for station in stations
        )
    )
