from __future__ import annotations

import streamlit as st

from config import (
    GPX_SEGMENT_LENGTH_M,
    LEARNING_STEP_M,
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
    simulate_race,
)


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="Trail Running Simulator V2",
    layout="wide",
)

st.title(
    "Trail Running Simulator V2"
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
    "fit_signature": None,
    "gpx_signature": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# Current project configuration
# =============================================================================

st.info(
    f"FIT learning step: "
    f"{LEARNING_STEP_M:.0f} m  |  "
    f"GPX / transition / simulation segment: "
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


# -----------------------------------------------------------------------------
# Detect change in FIT selection
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Build historical learning
# -----------------------------------------------------------------------------

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

                # New historical learning invalidates a previous simulation.
                st.session_state[
                    "simulation_df"
                ] = None

                st.session_state[
                    "race_summary"
                ] = None

                st.success(
                    "Historical learning, macro model and micro model "
                    "completed successfully."
                )

        except Exception as exc:

            st.error(
                f"Historical learning failed: {exc}"
            )

            st.exception(exc)


# =============================================================================
# Historical learning summary
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
            f"{learning_summary['median_segment_time_s']:.2f} s",
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "Mean transition time",
            f"{learning_summary['mean_segment_time_s']:.2f} s",
        )

    with col5:

        st.metric(
            "Fastest transition",
            f"{learning_summary['min_segment_time_s']:.2f} s",
        )

    with col6:

        st.metric(
            "Slowest transition",
            f"{learning_summary['max_segment_time_s']:.2f} s",
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
# Macro / Micro model summaries
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
            f"{macro_summary['training_mae_s']:.2f} s",
        )

    with col2:

        st.metric(
            "Training RMSE",
            f"{macro_summary['training_rmse_s']:.2f} s",
        )

    with col3:

        st.metric(
            "Training R²",
            f"{macro_summary['training_r2']:.4f}",
        )


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


# -----------------------------------------------------------------------------
# Detect GPX change
# -----------------------------------------------------------------------------

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
# Build normalized GPX profile
# =============================================================================

if uploaded_gpx_file is not None:

    if st.button(
        f"Build normalized "
        f"{GPX_SEGMENT_LENGTH_M:.0f} m GPX profile",
        key="build_gpx_profile",
    ):

        try:

            with st.spinner(
                "Building normalized GPX profile..."
            ):

                gpx_profile_df = (
                    build_gpx_profile(
                        uploaded_gpx_file,
                        aid_stations=None,
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

                simulation_df, race_summary = (
                    simulate_race(
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
                )

            st.session_state[
                "simulation_df"
            ] = simulation_df

            st.session_state[
                "race_summary"
            ] = race_summary

            st.success(
                "Simulation completed."
            )

        except Exception as exc:

            st.error(
                f"Simulation failed: {exc}"
            )

            st.exception(exc)


# =============================================================================
# Simulation results
# =============================================================================

simulation_df = st.session_state[
    "simulation_df"
]

race_summary = st.session_state[
    "race_summary"
]


if (
    simulation_df is not None
    and not simulation_df.empty
    and race_summary is not None
):

    st.subheader(
        "Race prediction"
    )

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
            (
                f"{race_summary['macro_final_race_h']:.2f} h"
            ),
        )

    with col3:

        st.metric(
            "Micro predicted race time",
            (
                f"{race_summary['micro_final_race_h']:.2f} h"
            ),
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "Aid-station time",
            (
                f"{race_summary['total_aid_stop_min']:.1f} min"
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

    st.subheader(
        "Simulation detail"
    )

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
    
