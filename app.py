from __future__ import annotations

import pandas as pd
import streamlit as st

from aid_stations import (
    AidStation,
    normalize_aid_stations,
    summarize_aid_stations,
)

from config import (
    GPX_SEGMENT_LENGTH_M,
    LEARNING_STEP_M,
)

from diagnostics import (
    build_macro_model_comparison,
    build_simulation_diagnostics,
)

from fit_learning import (
    build_learning_dataset,
    summarize_learning_dataset,
)

from macro_model import (
    fit_macro_model,
)

from micro_model import (
    fit_micro_model,
)

from gpx_profile import (
    build_gpx_profile,
    load_raw_gpx_table,
    summarize_gpx_profile,
)

from simulator import (
    build_aid_station_summary,
    format_seconds,
    simulate_race,
)


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="Trail Running Simulator V0",
    layout="wide",
)

st.title(
    "Trail Running Simulator V0"
)


# =============================================================================
# Session state
# =============================================================================

DEFAULT_STATE = {
    "learning_df": None,
    "learning_summary": None,
    "macro_model": None,
    "micro_model": None,

    "raw_gpx_df": None,
    "gpx_profile_df": None,

    "simulation_df": None,
    "race_summary": None,
    "simulation_diagnostics": None,

    "macro_model_comparison": None,

    "aid_stations": [],

    "fit_signature": None,
    "gpx_signature": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# Project configuration
# =============================================================================

st.info(
    f"FIT learning step: "
    f"{LEARNING_STEP_M:.0f} m  |  "
    f"GPX / historical transition / simulation segment: "
    f"{GPX_SEGMENT_LENGTH_M:.0f} m"
)


# =============================================================================
# 1. HISTORICAL FIT LEARNING
# =============================================================================

st.header(
    "1. Historical FIT learning"
)

uploaded_fit_files = st.file_uploader(
    "Upload historical FIT files",
    type=["fit"],
    accept_multiple_files=True,
    key="historical_fit_upload",
)


# =============================================================================
# FIT signature
# =============================================================================

def _build_fit_signature(
    uploaded_files,
):
    if not uploaded_files:
        return tuple()

    return tuple(
        (
            getattr(
                file,
                "name",
                "",
            ),
            getattr(
                file,
                "size",
                None,
            ),
        )
        for file in uploaded_files
    )


current_fit_signature = (
    _build_fit_signature(
        uploaded_fit_files
    )
)

if (
    current_fit_signature
    != st.session_state[
        "fit_signature"
    ]
):

    st.session_state[
        "fit_signature"
    ] = current_fit_signature

    st.session_state[
        "learning_df"
    ] = None

    st.session_state[
        "learning_summary"
    ] = None

    st.session_state[
        "macro_model"
    ] = None

    st.session_state[
        "micro_model"
    ] = None

    st.session_state[
        "simulation_df"
    ] = None

    st.session_state[
        "race_summary"
    ] = None

    st.session_state[
        "simulation_diagnostics"
    ] = None

    st.session_state[
        "macro_model_comparison"
    ] = None


# =============================================================================
# Build historical learning
# =============================================================================

if uploaded_fit_files:

    if st.button(
        "Build historical learning + models",
        type="primary",
        key="build_learning",
    ):

        try:

            with st.spinner(
                "Building historical learning dataset..."
            ):

                learning_df = (
                    build_learning_dataset(
                        uploaded_fit_files
                    )
                )

            if learning_df.empty:

                st.error(
                    "No historical learning transitions were created."
                )

            else:

                with st.spinner(
                    "Fitting macro model..."
                ):

                    macro_model = (
                        fit_macro_model(
                            learning_df
                        )
                    )

                with st.spinner(
                    "Building micro analogue model..."
                ):

                    micro_model = (
                        fit_micro_model(
                            learning_df
                        )
                    )

                learning_summary = (
                    summarize_learning_dataset(
                        learning_df
                    )
                )

                st.session_state[
                    "learning_df"
                ] = learning_df

                st.session_state[
                    "learning_summary"
                ] = learning_summary

                st.session_state[
                    "macro_model"
                ] = macro_model

                st.session_state[
                    "micro_model"
                ] = micro_model

                st.session_state[
                    "simulation_df"
                ] = None

                st.session_state[
                    "race_summary"
                ] = None

                st.session_state[
                    "simulation_diagnostics"
                ] = None

                st.session_state[
                    "macro_model_comparison"
                ] = None

                st.success(
                    "Historical learning, macro model and "
                    "micro model completed successfully."
                )

        except Exception as exc:

            st.error(
                f"Historical learning failed: {exc}"
            )

            st.exception(exc)


# =============================================================================
# Retrieve current model state
# =============================================================================

learning_df = st.session_state[
    "learning_df"
]

learning_summary = st.session_state[
    "learning_summary"
]

macro_model = st.session_state[
    "macro_model"
]

micro_model = st.session_state[
    "micro_model"
]


# =============================================================================
# Historical learning summary
# =============================================================================

if (
    learning_df is not None
    and learning_summary is not None
):

    st.subheader(
        "Historical learning dataset"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Activities",
            learning_summary[
                "n_activities"
            ],
        )

    with col2:

        st.metric(
            "Historical transitions",
            f"{learning_summary['n_transitions']:,}",
        )

    with col3:

        st.metric(
            "Median transition time",
            format_seconds(
                learning_summary[
                    "median_segment_time_s"
                ]
            ),
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "Mean transition time",
            format_seconds(
                learning_summary[
                    "mean_segment_time_s"
                ]
            ),
        )

    with col5:

        st.metric(
            "Fastest transition",
            format_seconds(
                learning_summary[
                    "min_segment_time_s"
                ]
            ),
        )

    with col6:

        st.metric(
            "Slowest transition",
            format_seconds(
                learning_summary[
                    "max_segment_time_s"
                ]
            ),
        )

    with st.expander(
        "Historical learning rows",
        expanded=False,
    ):

        st.dataframe(
            learning_df.head(500),
            width="stretch",
        )

        st.download_button(
            "Download historical learning dataset",
            data=learning_df.to_csv(
                index=False
            ),
            file_name=(
                "historical_learning_dataset.csv"
            ),
            mime="text/csv",
            key="download_learning",
        )


# =============================================================================
# Macro model summary
# =============================================================================

if macro_model is not None:

    macro_summary = (
        macro_model.summary()
    )

    st.subheader(
        "Macro model 1"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Training MAE",
            format_seconds(
                macro_summary[
                    "training_mae_s"
                ]
            ),
        )

    with col2:

        st.metric(
            "Training RMSE",
            format_seconds(
                macro_summary[
                    "training_rmse_s"
                ]
            ),
        )

    with col3:

        st.metric(
            "Training R²",
            f"{macro_summary['training_r2']:.4f}",
        )


# =============================================================================
# Micro model summary
# =============================================================================

if micro_model is not None:

    micro_summary = (
        micro_model.summary()
    )

    st.subheader(
        "Micro analogue model"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Historical states",
            f"{micro_summary['training_rows']:,}",
        )

    with col2:

        st.metric(
            "Activities",
            micro_summary[
                "training_activities"
            ],
        )

    with col3:

        st.metric(
            "State variables",
            micro_summary[
                "n_state_variables"
            ],
        )


# =============================================================================
# 2. FUTURE GPX
# =============================================================================

st.divider()

st.header(
    "2. Future GPX"
)

uploaded_gpx_file = st.file_uploader(
    "Upload the GPX you want to simulate",
    type=["gpx"],
    key="future_gpx_upload",
)


# =============================================================================
# GPX signature
# =============================================================================

def _build_gpx_signature(
    uploaded_file,
):
    if uploaded_file is None:
        return None

    return (
        getattr(
            uploaded_file,
            "name",
            "",
        ),
        getattr(
            uploaded_file,
            "size",
            None,
        ),
    )


current_gpx_signature = (
    _build_gpx_signature(
        uploaded_gpx_file
    )
)

if (
    current_gpx_signature
    != st.session_state[
        "gpx_signature"
    ]
):

    st.session_state[
        "gpx_signature"
    ] = current_gpx_signature

    st.session_state[
        "raw_gpx_df"
    ] = None

    st.session_state[
        "gpx_profile_df"
    ] = None

    st.session_state[
        "simulation_df"
    ] = None

    st.session_state[
        "race_summary"
    ] = None

    st.session_state[
        "simulation_diagnostics"
    ] = None

    st.session_state[
        "macro_model_comparison"
    ] = None


# =============================================================================
# Raw GPX inspection
# =============================================================================

if uploaded_gpx_file is not None:

    if st.button(
        "Load raw GPX",
        key="load_raw_gpx",
    ):

        try:

            with st.spinner(
                "Reading raw GPX..."
            ):

                raw_gpx_df = (
                    load_raw_gpx_table(
                        uploaded_gpx_file
                    )
                )

            if raw_gpx_df.empty:

                st.error(
                    "No GPX track points were found."
                )

            else:

                st.session_state[
                    "raw_gpx_df"
                ] = raw_gpx_df

                st.success(
                    "Raw GPX loaded."
                )

        except Exception as exc:

            st.error(
                f"Raw GPX loading failed: {exc}"
            )

            st.exception(exc)


raw_gpx_df = st.session_state[
    "raw_gpx_df"
]

if (
    raw_gpx_df is not None
    and not raw_gpx_df.empty
):

    with st.expander(
        "Raw GPX data",
        expanded=False,
    ):

        st.write(
            f"Raw GPX points: "
            f"{len(raw_gpx_df):,}"
        )

        st.dataframe(
            raw_gpx_df.head(500),
            width="stretch",
        )

        st.download_button(
            "Download raw GPX table",
            data=raw_gpx_df.to_csv(
                index=False
            ),
            file_name="raw_gpx_table.csv",
            mime="text/csv",
            key="download_raw_gpx",
        )


# =============================================================================
# 2A. AID STATIONS
# =============================================================================

st.subheader(
    "Aid stations"
)

st.write(
    "Enter the expected stationary time at each aid station. "
    "Distance is measured from race start."
)


current_aid_stations = (
    st.session_state[
        "aid_stations"
    ]
)

editor_rows = [
    {
        "name": station.name,
        "distance_km": (
            station.distance_from_start_m
            / 1000.0
        ),
        "stop_minutes": station.stop_minutes,
    }
    for station in current_aid_stations
]


# -----------------------------------------------------------------------------
# Aid-station form
# -----------------------------------------------------------------------------

with st.form(
    "aid_station_form",
    clear_on_submit=False,
):

    aid_station_editor_df = st.data_editor(
        pd.DataFrame(
            editor_rows,
            columns=[
                "name",
                "distance_km",
                "stop_minutes",
            ],
        ),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn(
                "Aid station",
                help="Station name",
            ),
            "distance_km": st.column_config.NumberColumn(
                "Distance (km)",
                min_value=0.0,
                step=0.1,
                format="%.1f",
            ),
            "stop_minutes": st.column_config.NumberColumn(
                "Expected stop (min)",
                min_value=0.0,
                step=1.0,
                format="%.0f",
            ),
        },
        key="aid_station_editor",
    )

    save_aid_stations = st.form_submit_button(
        "Save aid stations",
        type="primary",
    )


# =============================================================================
# Aid-station editor conversion
# =============================================================================

def _editor_to_dataframe(
    editor_value,
) -> pd.DataFrame:
    """
    Convert the Streamlit editor value into a DataFrame.
    """

    if editor_value is None:

        return pd.DataFrame(
            columns=[
                "name",
                "distance_km",
                "stop_minutes",
            ]
        )

    if isinstance(
        editor_value,
        pd.DataFrame,
    ):

        return editor_value.copy()

    if isinstance(
        editor_value,
        list,
    ):

        if not editor_value:

            return pd.DataFrame(
                columns=[
                    "name",
                    "distance_km",
                    "stop_minutes",
                ]
            )

        return pd.DataFrame(
            editor_value
        )

    if isinstance(
        editor_value,
        dict,
    ):

        return pd.DataFrame(
            editor_value
        )

    raise ValueError(
        "Unsupported aid-station editor output type: "
        f"{type(editor_value).__name__}"
    )


def _read_aid_station_editor(
    editor_value,
) -> list[AidStation]:
    """
    Convert submitted editor rows to validated AidStation objects.
    """

    editor_df = _editor_to_dataframe(
        editor_value
    )

    if editor_df.empty:

        return []

    for column in [
        "name",
        "distance_km",
        "stop_minutes",
    ]:

        if column not in editor_df.columns:

            editor_df[
                column
            ] = None

    stations: list[
        AidStation
    ] = []

    for _, row in editor_df.iterrows():

        name_value = row.get(
            "name"
        )

        distance_value = row.get(
            "distance_km"
        )

        stop_value = row.get(
            "stop_minutes"
        )

        name_empty = (
            name_value is None
            or pd.isna(
                name_value
            )
            or str(
                name_value
            ).strip() == ""
        )

        distance_empty = (
            distance_value is None
            or pd.isna(
                distance_value
            )
        )

        stop_empty = (
            stop_value is None
            or pd.isna(
                stop_value
            )
        )

        # Completely blank rows are ignored.
        if (
            name_empty
            and distance_empty
            and stop_empty
        ):
            continue

        # Partially populated rows are invalid.
        if (
            name_empty
            or distance_empty
            or stop_empty
        ):

            raise ValueError(
                "Each aid station must contain "
                "a name, distance and expected stop duration."
            )

        stations.append(
            AidStation(
                name=str(
                    name_value
                ).strip(),
                distance_from_start_m=(
                    float(
                        distance_value
                    )
                    * 1000.0
                ),
                stop_minutes=float(
                    stop_value
                ),
            )
        )

    return normalize_aid_stations(
        stations
    )


# =============================================================================
# Save aid stations
# =============================================================================

if save_aid_stations:

    try:

        new_aid_stations = (
            _read_aid_station_editor(
                aid_station_editor_df
            )
        )

        old_signature = tuple(
            (
                station.name,
                station.distance_from_start_m,
                station.stop_minutes,
            )
            for station in current_aid_stations
        )

        new_signature = tuple(
            (
                station.name,
                station.distance_from_start_m,
                station.stop_minutes,
            )
            for station in new_aid_stations
        )

        st.session_state[
            "aid_stations"
        ] = new_aid_stations

        if old_signature != new_signature:

            # -----------------------------------------------------------------
            # The normalized GPX contains the aid-station mapping, therefore
            # changing the saved configuration invalidates the profile and
            # subsequent simulation.
            # -----------------------------------------------------------------

            st.session_state[
                "gpx_profile_df"
            ] = None

            st.session_state[
                "simulation_df"
            ] = None

            st.session_state[
                "race_summary"
            ] = None

            st.session_state[
                "simulation_diagnostics"
            ] = None

        st.success(
            "Aid stations saved."
        )

    except Exception as exc:

        st.error(
            f"Aid-station input error: {exc}"
        )


# =============================================================================
# Saved aid-station summary
# =============================================================================

saved_aid_stations = (
    st.session_state[
        "aid_stations"
    ]
)

if saved_aid_stations:

    aid_summary = (
        summarize_aid_stations(
            saved_aid_stations
        )
    )

    total_stop_minutes = float(
        aid_summary[
            "stop_minutes"
        ].sum()
    )

    st.caption(
        f"{len(saved_aid_stations)} aid station(s) saved | "
        f"Total expected stationary time: "
        f"{total_stop_minutes:.0f} min"
    )

else:

    st.caption(
        "No aid stations saved."
    )


# =============================================================================
# Build normalized GPX
# =============================================================================

if uploaded_gpx_file is not None:

    if st.button(
        f"Build normalized "
        f"{GPX_SEGMENT_LENGTH_M:.0f} m GPX profile",
        key="build_gpx_profile",
    ):

        try:

            aid_stations = (
                st.session_state[
                    "aid_stations"
                ]
            )

            with st.spinner(
                "Building normalized GPX profile..."
            ):

                gpx_profile_df = (
                    build_gpx_profile(
                        uploaded_gpx_file,
                        aid_stations=(
                            aid_stations
                        ),
                    )
                )

            st.session_state[
                "gpx_profile_df"
            ] = gpx_profile_df

            st.session_state[
                "simulation_df"
            ] = None

            st.session_state[
                "race_summary"
            ] = None

            st.session_state[
                "simulation_diagnostics"
            ] = None

            st.success(
                "Normalized GPX profile created."
            )

        except Exception as exc:

            st.error(
                f"GPX normalization failed: {exc}"
            )

            st.exception(exc)


# =============================================================================
# GPX profile summary
# =============================================================================

gpx_profile_df = st.session_state[
    "gpx_profile_df"
]

if (
    gpx_profile_df is not None
    and not gpx_profile_df.empty
):

    gpx_summary = (
        summarize_gpx_profile(
            gpx_profile_df
        )
    )

    st.subheader(
        "Normalized GPX profile"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Segments",
            f"{gpx_summary['n_segments']:,}",
        )

    with col2:

        st.metric(
            "Distance",
            (
                f"{gpx_summary['distance_m'] / 1000:.2f} km"
            ),
        )

    with col3:

        st.metric(
            "Cumulative ascent",
            (
                f"{gpx_summary['cumulative_ascent_m']:.0f} m"
            ),
        )

    with col4:

        st.metric(
            "Cumulative descent",
            (
                f"{gpx_summary['cumulative_descent_m']:.0f} m"
            ),
        )

    with st.expander(
        "Normalized GPX terrain table",
        expanded=False,
    ):

        st.dataframe(
            gpx_profile_df.head(500),
            width="stretch",
        )

        st.download_button(
            "Download normalized GPX profile",
            data=gpx_profile_df.to_csv(
                index=False
            ),
            file_name=(
                "normalized_gpx_profile.csv"
            ),
            mime="text/csv",
            key="download_gpx_profile",
        )


# =============================================================================
# 3. SIMULATION
# =============================================================================

st.divider()

st.header(
    "3. Simulation"
)

models_ready = (
    macro_model is not None
    and micro_model is not None
)

profile_ready = (
    gpx_profile_df is not None
    and not gpx_profile_df.empty
)


if not models_ready:

    st.warning(
        "Build the historical FIT learning and models first."
    )

elif not profile_ready:

    st.warning(
        "Build the normalized GPX profile first."
    )

else:

    st.write(
        f"Simulation step: "
        f"{GPX_SEGMENT_LENGTH_M:.0f} m"
    )

    if st.button(
        "Run macro + micro simulation",
        type="primary",
        key="run_simulation",
    ):

        try:

            with st.spinner(
                "Simulating race..."
            ):

                (
                    simulation_df,
                    race_summary,
                ) = simulate_race(
                    gpx_profile_df=(
                        gpx_profile_df
                    ),
                    macro_model=(
                        macro_model
                    ),
                    micro_model=(
                        micro_model
                    ),
                )

            st.session_state[
                "simulation_df"
            ] = simulation_df

            st.session_state[
                "race_summary"
            ] = race_summary

            with st.spinner(
                "Building simulation diagnostics..."
            ):

                simulation_diagnostics = (
                    build_simulation_diagnostics(
                        simulation_df
                    )
                )

            st.session_state[
                "simulation_diagnostics"
            ] = simulation_diagnostics

            st.success(
                "Simulation completed."
            )

        except Exception as exc:

            st.error(
                f"Simulation failed: {exc}"
            )

            st.exception(exc)


# =============================================================================
# Retrieve simulation state
# =============================================================================

simulation_df = st.session_state[
    "simulation_df"
]

race_summary = st.session_state[
    "race_summary"
]

simulation_diagnostics = st.session_state[
    "simulation_diagnostics"
]


# =============================================================================
# Simulation results
# =============================================================================

if (
    simulation_df is not None
    and not simulation_df.empty
    and race_summary is not None
):

    st.subheader(
        "Race prediction"
    )

    # -------------------------------------------------------------------------
    # Primary results
    # -------------------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Distance",
            (
                f"{race_summary['distance_m'] / 1000:.2f} km"
            ),
        )

    with col2:

        st.metric(
            "Macro predicted race time",
            format_seconds(
                race_summary[
                    "macro_final_race_s"
                ]
            ),
        )

    with col3:

        st.metric(
            "Micro predicted race time",
            format_seconds(
                race_summary[
                    "micro_final_race_s"
                ]
            ),
        )

    # -------------------------------------------------------------------------
    # Secondary results
    # -------------------------------------------------------------------------

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "Aid-station time",
            format_seconds(
                race_summary[
                    "total_aid_stop_s"
                ]
            ),
        )

    with col5:

        st.metric(
            "Micro - Macro",
            (
                f"{race_summary['micro_minus_macro_final_min']:.1f} min"
            ),
        )

    with col6:

        st.metric(
            "Segments",
            f"{race_summary['segments']:,}",
        )

    # -------------------------------------------------------------------------
    # Aid-station predictions
    # -------------------------------------------------------------------------

    aid_station_summary = (
        build_aid_station_summary(
            simulation_df
        )
    )

    if (
        aid_station_summary is not None
        and not aid_station_summary.empty
    ):

        st.subheader(
            "Aid-station predictions"
        )

        display_aid_summary = (
            aid_station_summary.copy()
        )

        display_aid_summary[
            "distance_km"
        ] = (
            display_aid_summary[
                "aid_station_distance_m"
            ]
            / 1000.0
        )

        display_aid_summary[
            "expected_stop"
        ] = (
            display_aid_summary[
                "aid_station_stop_min"
            ]
            .astype(float)
            .map(
                lambda value:
                format_seconds(
                    value * 60.0
                )
            )
        )

        display_aid_summary[
            "macro_arrival"
        ] = (
            display_aid_summary[
                "macro_arrival_time_s"
            ]
            .map(
                format_seconds
            )
        )

        display_aid_summary[
            "micro_arrival"
        ] = (
            display_aid_summary[
                "micro_arrival_time_s"
            ]
            .map(
                format_seconds
            )
        )

        display_aid_summary[
            "macro_departure"
        ] = (
            display_aid_summary[
                "macro_departure_time_s"
            ]
            .map(
                format_seconds
            )
        )

        display_aid_summary[
            "micro_departure"
        ] = (
            display_aid_summary[
                "micro_departure_time_s"
            ]
            .map(
                format_seconds
            )
        )

        st.dataframe(
            display_aid_summary[
                [
                    "aid_station_name",
                    "distance_km",
                    "expected_stop",
                    "macro_arrival",
                    "micro_arrival",
                    "macro_departure",
                    "micro_departure",
                ]
            ],
            width="stretch",
        )

    # -------------------------------------------------------------------------
    # Macro physical constraint
    # -------------------------------------------------------------------------

    st.subheader(
        "Macro physical constraint"
    )

    col7, col8 = st.columns(2)

    with col7:

        st.metric(
            "Clipped segments",
            race_summary[
                "macro_clipped_segments"
            ],
        )

    with col8:

        st.metric(
            "Clipped time",
            format_seconds(
                race_summary[
                    "macro_clipped_seconds"
                ]
            ),
        )

    # -------------------------------------------------------------------------
    # Simulation detail
    # -------------------------------------------------------------------------

    with st.expander(
        "Simulation detail",
        expanded=False,
    ):

        st.dataframe(
            simulation_df.head(500),
            width="stretch",
        )

        st.download_button(
            "Download complete simulation",
            data=simulation_df.to_csv(
                index=False
            ),
            file_name=(
                "simulation_results.csv"
            ),
            mime="text/csv",
            key="download_simulation",
        )


# =============================================================================
# 4. TEMPORARY DIAGNOSTICS
# =============================================================================
#
# Everything below this point is disposable.
#
# Diagnostic calculations live in diagnostics.py.
# app.py only orchestrates and displays them.
# =============================================================================

if (
    learning_df is not None
):

    st.divider()

    st.header(
        "Diagnostics"
    )

    # =========================================================================
    # A. Macro Model 1 vs Macro Model 2
    # =========================================================================

    st.subheader(
        "Macro Model 1 vs Macro Model 2"
    )

    st.write(
        "Macro Model 2 is a diagnostic experiment only. "
        "It does not replace Macro Model 1 in the simulator."
    )

    if gpx_profile_df is None:

        st.info(
            "Build the normalized GPX profile before running "
            "the macro-model comparison."
        )

    else:

        if st.button(
            "Run Macro Model 1 vs Macro Model 2 comparison",
            key="run_macro_model_comparison",
        ):

            try:

                with st.spinner(
                    "Comparing Macro Model 1 and Macro Model 2..."
                ):

                    comparison = (
                        build_macro_model_comparison(
                            learning_df=(
                                learning_df
                            ),
                            gpx_profile_df=(
                                gpx_profile_df
                            ),
                        )
                    )

                st.session_state[
                    "macro_model_comparison"
                ] = comparison

                st.success(
                    "Macro Model comparison completed."
                )

            except Exception as exc:

                st.error(
                    f"Macro Model comparison failed: {exc}"
                )

                st.exception(exc)

    macro_model_comparison = (
        st.session_state[
            "macro_model_comparison"
        ]
    )

    if macro_model_comparison is not None:

        # ---------------------------------------------------------------------
        # Historical performance
        # ---------------------------------------------------------------------

        historical_comparison = (
            macro_model_comparison.get(
                "historical"
            )
        )

        if (
            historical_comparison is not None
            and not historical_comparison.empty
        ):

            st.markdown(
                "#### Historical FIT performance"
            )

            historical_display = (
                historical_comparison.copy()
            )

            historical_display[
                "MAE"
            ] = historical_display[
                "mae_s"
            ].map(
                format_seconds
            )

            historical_display[
                "RMSE"
            ] = historical_display[
                "rmse_s"
            ].map(
                format_seconds
            )

            historical_display[
                "Bias"
            ] = historical_display[
                "bias_s"
            ].map(
                format_seconds
            )

            st.dataframe(
                historical_display[
                    [
                        "model",
                        "constraint",
                        "training_rows",
                        "training_activities",
                        "MAE",
                        "RMSE",
                        "Bias",
                        "r2",
                    ]
                ],
                width="stretch",
            )

        # ---------------------------------------------------------------------
        # Coefficient comparison
        # ---------------------------------------------------------------------

        coefficient_comparison = (
            macro_model_comparison.get(
                "coefficients"
            )
        )

        if (
            coefficient_comparison is not None
            and not coefficient_comparison.empty
        ):

            with st.expander(
                "Macro coefficients",
                expanded=False,
            ):

                st.dataframe(
                    coefficient_comparison,
                    width="stretch",
                )

        # ---------------------------------------------------------------------
        # SwissPeak GPX behaviour
        # ---------------------------------------------------------------------

        gpx_summary = (
            macro_model_comparison.get(
                "gpx_summary"
            )
        )

        if (
            gpx_summary is not None
            and not gpx_summary.empty
        ):

            st.markdown(
                "#### Behaviour on current normalized GPX"
            )

            gpx_display = (
                gpx_summary.copy()
            )

            gpx_display[
                "negative_time"
            ] = gpx_display[
                "negative_time_magnitude_s"
            ].map(
                format_seconds
            )

            gpx_display[
                "final_cumulative_time"
            ] = gpx_display[
                "final_cumulative_time_s"
            ].map(
                format_seconds
            )

            st.dataframe(
                gpx_display[
                    [
                        "model",
                        "negative_segments",
                        "negative_time",
                        "final_cumulative_time",
                    ]
                ],
                width="stretch",
            )

        # ---------------------------------------------------------------------
        # Detailed GPX model comparison
        # ---------------------------------------------------------------------

        gpx_comparison = (
            macro_model_comparison.get(
                "gpx"
            )
        )

        if (
            gpx_comparison is not None
            and not gpx_comparison.empty
        ):

            with st.expander(
                "Detailed GPX Macro Model comparison",
                expanded=False,
            ):

                st.dataframe(
                    gpx_comparison.head(500),
                    width="stretch",
                )

                st.download_button(
                    "Download Macro Model comparison",
                    data=gpx_comparison.to_csv(
                        index=False
                    ),
                    file_name=(
                        "macro_model_1_vs_macro_model_2.csv"
                    ),
                    mime="text/csv",
                    key="download_macro_comparison",
                )

    # =========================================================================
    # B. Existing simulation diagnostics
    # =========================================================================

    if simulation_diagnostics is not None:

        # ---------------------------------------------------------------------
        # Cumulative macro vs micro
        # ---------------------------------------------------------------------

        divergence_df = (
            simulation_diagnostics.get(
                "simulation_divergence"
            )
        )

        if (
            divergence_df is not None
            and not divergence_df.empty
        ):

            st.subheader(
                "Cumulative macro vs micro"
            )

            chart_df = divergence_df[
                [
                    "distance_km",
                    "macro_cumulative_time_h",
                    "micro_cumulative_time_h",
                ]
            ].copy()

            st.line_chart(
                chart_df.set_index(
                    "distance_km"
                )
            )

            st.subheader(
                "Micro minus macro cumulative difference"
            )

            divergence_chart_df = (
                divergence_df[
                    [
                        "distance_km",
                        "micro_minus_macro_cumulative_min",
                    ]
                ]
                .copy()
                .set_index(
                    "distance_km"
                )
            )

            st.line_chart(
                divergence_chart_df
            )

        # ---------------------------------------------------------------------
        # Checkpoints
        # ---------------------------------------------------------------------

        checkpoints_df = (
            simulation_diagnostics.get(
                "simulation_checkpoints"
            )
        )

        if (
            checkpoints_df is not None
            and not checkpoints_df.empty
        ):

            st.subheader(
                "Prediction divergence checkpoints"
            )

            st.dataframe(
                checkpoints_df,
                width="stretch",
            )

        # ---------------------------------------------------------------------
        # Course sections
        # ---------------------------------------------------------------------

        sections_df = (
            simulation_diagnostics.get(
                "simulation_sections"
            )
        )

        if (
            sections_df is not None
            and not sections_df.empty
        ):

            st.subheader(
                "Prediction divergence by course section"
            )

            st.dataframe(
                sections_df,
                width="stretch",
            )

        # ---------------------------------------------------------------------
        # Macro clipping diagnostics
        # ---------------------------------------------------------------------

        clipping_df = (
            simulation_diagnostics.get(
                "macro_clipping"
            )
        )

        if (
            clipping_df is not None
            and not clipping_df.empty
        ):

            st.subheader(
                "Macro clipped segments"
            )

            st.dataframe(
                clipping_df,
                width="stretch",
            )

        elif (
            clipping_df is not None
            and clipping_df.empty
        ):

            st.success(
                "No macro negative-duration segments were clipped."
            )
            
