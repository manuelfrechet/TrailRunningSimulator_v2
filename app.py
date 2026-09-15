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

from fit_learning import (
    build_learning_dataset,
    summarize_learning_dataset,
)

from gpx_profile import (
    build_gpx_profile,
    load_raw_gpx_table,
    summarize_gpx_profile,
)

from macro_model import (
    fit_macro_model,
)

from micro_model import (
    fit_micro_model,
)

from race_validation import (
    build_validation_section_summary,
    compare_prediction_to_actual_fit,
    summarize_validation,
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

    "aid_stations": [],

    "fit_signature": None,
    "gpx_signature": None,

    # Temporary post-race validation state.
    "actual_race_comparison_df": None,
    "actual_race_summary_df": None,
    "actual_race_section_df": None,
    "actual_race_signature": None,
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
        "actual_race_comparison_df"
    ] = None

    st.session_state[
        "actual_race_summary_df"
    ] = None

    st.session_state[
        "actual_race_section_df"
    ] = None

    st.session_state[
        "actual_race_signature"
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

                # -------------------------------------------------------------
                # Any previous validation belongs to the previous prediction.
                # -------------------------------------------------------------

                st.session_state[
                    "actual_race_comparison_df"
                ] = None

                st.session_state[
                    "actual_race_summary_df"
                ] = None

                st.session_state[
                    "actual_race_section_df"
                ] = None

                st.session_state[
                    "actual_race_signature"
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
        "Macro model"
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
        "actual_race_comparison_df"
    ] = None

    st.session_state[
        "actual_race_summary_df"
    ] = None

    st.session_state[
        "actual_race_section_df"
    ] = None

    st.session_state[
        "actual_race_signature"
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

        if (
            name_empty
            and distance_empty
            and stop_empty
        ):
            continue

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
                "actual_race_comparison_df"
            ] = None

            st.session_state[
                "actual_race_summary_df"
            ] = None

            st.session_state[
                "actual_race_section_df"
            ] = None

            st.session_state[
                "actual_race_signature"
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
                "actual_race_comparison_df"
            ] = None

            st.session_state[
                "actual_race_summary_df"
            ] = None

            st.session_state[
                "actual_race_section_df"
            ] = None

            st.session_state[
                "actual_race_signature"
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

            # -------------------------------------------------------------
            # A new prediction invalidates any previous actual-race
            # comparison.
            # -------------------------------------------------------------

            st.session_state[
                "actual_race_comparison_df"
            ] = None

            st.session_state[
                "actual_race_summary_df"
            ] = None

            st.session_state[
                "actual_race_section_df"
            ] = None

            st.session_state[
                "actual_race_signature"
            ] = None

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
# 4. ACTUAL RACE VALIDATION
# =============================================================================
#
# This section appears ONLY after a prediction exists.
#
# The actual race FIT is deliberately isolated from:
#
#     - historical FIT learning;
#     - macro model fitting;
#     - micro model fitting;
#     - GPX normalization;
#     - race simulation.
#
# It is used only for post-prediction validation.
# =============================================================================

if (
    simulation_df is not None
    and not simulation_df.empty
):

    st.divider()

    st.header(
        "4. Actual race validation"
    )

    st.write(
        "Upload the actual FIT from the race only after the prediction "
        "has been completed."
    )

    uploaded_actual_race_fit = st.file_uploader(
        "Upload actual race FIT",
        type=["fit"],
        accept_multiple_files=False,
        key="actual_race_fit_upload",
    )

    # -------------------------------------------------------------------------
    # Actual FIT signature.
    # -------------------------------------------------------------------------

    def _build_actual_race_signature(
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

    current_actual_race_signature = (
        _build_actual_race_signature(
            uploaded_actual_race_fit
        )
    )

    if (
        current_actual_race_signature
        != st.session_state[
            "actual_race_signature"
        ]
    ):

        st.session_state[
            "actual_race_signature"
        ] = current_actual_race_signature

        st.session_state[
            "actual_race_comparison_df"
        ] = None

        st.session_state[
            "actual_race_summary_df"
        ] = None

        st.session_state[
            "actual_race_section_df"
        ] = None

    if uploaded_actual_race_fit is not None:

        if st.button(
            "Compare prediction with actual race",
            type="primary",
            key="compare_actual_race",
        ):

            try:

                with st.spinner(
                    "Comparing prediction with actual race..."
                ):

                    (
                        actual_race_comparison_df,
                        actual_race_summary_df,
                    ) = compare_prediction_to_actual_fit(
                        simulation_df=(
                            simulation_df
                        ),
                        actual_fit_file=(
                            uploaded_actual_race_fit
                        ),
                    )

                    actual_race_section_df = (
                        build_validation_section_summary(
                            actual_race_comparison_df
                        )
                    )

                st.session_state[
                    "actual_race_comparison_df"
                ] = actual_race_comparison_df

                st.session_state[
                    "actual_race_summary_df"
                ] = actual_race_summary_df

                st.session_state[
                    "actual_race_section_df"
                ] = actual_race_section_df

                st.success(
                    "Prediction / actual-race comparison completed."
                )

            except Exception as exc:

                st.error(
                    f"Actual-race validation failed: {exc}"
                )

                st.exception(exc)


# =============================================================================
# Actual race validation results
# =============================================================================

actual_race_summary_df = (
    st.session_state[
        "actual_race_summary_df"
    ]
)

actual_race_comparison_df = (
    st.session_state[
        "actual_race_comparison_df"
    ]
)

actual_race_section_df = (
    st.session_state[
        "actual_race_section_df"
    ]
)


if (
    actual_race_summary_df is not None
    and not actual_race_summary_df.empty
):

    st.subheader(
        "Prediction vs actual race"
    )

    display_summary = (
        actual_race_summary_df.copy()
    )

    display_summary[
        "predicted_finish"
    ] = (
        display_summary[
            "predicted_finish_s"
        ]
        .map(
            format_seconds
        )
    )

    display_summary[
        "actual_finish"
    ] = (
        display_summary[
            "actual_finish_s"
        ]
        .map(
            format_seconds
        )
    )

    display_summary[
        "finish_error"
    ] = (
        display_summary[
            "finish_error_s"
        ]
        .map(
            format_seconds
        )
    )

    st.dataframe(
        display_summary[
            [
                "model",
                "predicted_finish",
                "actual_finish",
                "finish_error",
                "finish_error_min",
                "segment_mae_s",
                "segment_rmse_s",
                "segment_bias_s",
            ]
        ],
        width="stretch",
    )


# =============================================================================
# Actual race section comparison
# =============================================================================

if (
    actual_race_section_df is not None
    and not actual_race_section_df.empty
):

    st.subheader(
        "Prediction error by race section"
    )

    st.dataframe(
        actual_race_section_df,
        width="stretch",
    )


# =============================================================================
# Actual race cumulative-time comparison
# =============================================================================

if (
    actual_race_comparison_df is not None
    and not actual_race_comparison_df.empty
):

    st.subheader(
        "Cumulative predicted vs actual time"
    )

    chart_df = (
        actual_race_comparison_df[
            [
                "distance_km",
                "actual_cumulative_time_s",
                "macro_cumulative_time_s",
                "micro_cumulative_time_s",
            ]
        ]
        .copy()
    )

    chart_df[
        "Actual"
    ] = (
        chart_df[
            "actual_cumulative_time_s"
        ]
        / 3600.0
    )

    chart_df[
        "Macro"
    ] = (
        chart_df[
            "macro_cumulative_time_s"
        ]
        / 3600.0
    )

    chart_df[
        "Micro"
    ] = (
        chart_df[
            "micro_cumulative_time_s"
        ]
        / 3600.0
    )

    st.line_chart(
        chart_df[
            [
                "distance_km",
                "Actual",
                "Macro",
                "Micro",
            ]
        ].set_index(
            "distance_km"
        )
    )

    st.subheader(
        "Cumulative prediction error"
    )

    error_chart_df = (
        actual_race_comparison_df[
            [
                "distance_km",
                "macro_error_s",
                "micro_error_s",
            ]
        ]
        .copy()
    )

    error_chart_df[
        "Macro error (min)"
    ] = (
        error_chart_df[
            "macro_error_s"
        ]
        / 60.0
    )

    error_chart_df[
        "Micro error (min)"
    ] = (
        error_chart_df[
            "micro_error_s"
        ]
        / 60.0
    )

    st.line_chart(
        error_chart_df[
            [
                "distance_km",
                "Macro error (min)",
                "Micro error (min)",
            ]
        ].set_index(
            "distance_km"
        )
    )

    with st.expander(
        "Detailed prediction / actual comparison",
        expanded=False,
    ):

        st.dataframe(
            actual_race_comparison_df.head(500),
            width="stretch",
        )

        st.download_button(
            "Download prediction vs actual comparison",
            data=actual_race_comparison_df.to_csv(
                index=False
            ),
            file_name=(
                "prediction_vs_actual_race.csv"
            ),
            mime="text/csv",
            key="download_actual_race_comparison",
        )

        st.download_button(
            "Download validation summary",
            data=actual_race_summary_df.to_csv(
                index=False
            ),
            file_name=(
                "prediction_vs_actual_summary.csv"
            ),
            mime="text/csv",
            key="download_actual_race_summary",
        )
        
