from __future__ import annotations

import pandas as pd
import streamlit as st

from fit_learning import (
    build_learning_dataset,
    summarize_learning_dataset,
)

from macro_model import fit_macro_model

from diagnostics import (
    build_all_diagnostics,
    build_leave_one_activity_out_validation,
)


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Trail Running Simulator V2",
    layout="wide",
)

st.title("Trail Running Simulator V2")


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

DEFAULT_STATE = {
    "learning_df": None,
    "learning_summary": None,
    "macro_model": None,
    "diagnostics": None,
    "fit_signature": None,
    "loo_summary": None,
    "loo_detail": None,
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _fit_signature(uploaded_files) -> tuple[tuple[str, int | None], ...]:
    """
    Create a lightweight signature for the currently uploaded FIT set.

    This is used only to invalidate cached session results when the user
    changes the selected FIT files.
    """
    if not uploaded_files:
        return tuple()

    return tuple(
        (
            getattr(file, "name", ""),
            getattr(file, "size", None),
        )
        for file in uploaded_files
    )


# -----------------------------------------------------------------------------
# Historical FIT input
# -----------------------------------------------------------------------------

st.header("1. Historical FIT learning")

uploaded_fit_files = st.file_uploader(
    "Upload one or more historical FIT files",
    type=["fit"],
    accept_multiple_files=True,
)

current_signature = _fit_signature(
    uploaded_fit_files
)


# -----------------------------------------------------------------------------
# Detect changed FIT selection
# -----------------------------------------------------------------------------

if (
    current_signature
    and current_signature
    != st.session_state["fit_signature"]
):
    st.session_state["learning_df"] = None
    st.session_state["learning_summary"] = None
    st.session_state["macro_model"] = None
    st.session_state["diagnostics"] = None
    st.session_state["loo_summary"] = None
    st.session_state["loo_detail"] = None


# -----------------------------------------------------------------------------
# Main combined learning
# -----------------------------------------------------------------------------

if uploaded_fit_files:

    if st.button(
        "Build historical learning + macro model",
        type="primary",
    ):

        try:

            # -----------------------------------------------------------------
            # 1. Historical learning
            # -----------------------------------------------------------------

            with st.spinner(
                "Building historical 1 m rolling / 50 m transition dataset..."
            ):
                learning_df = build_learning_dataset(
                    uploaded_fit_files
                )

            if learning_df.empty:
                st.error(
                    "No historical transitions were created."
                )
                st.stop()

            learning_summary = (
                summarize_learning_dataset(
                    learning_df
                )
            )

            # -----------------------------------------------------------------
            # 2. Macro model
            # -----------------------------------------------------------------

            with st.spinner(
                "Fitting V0 macro model..."
            ):
                macro_model = fit_macro_model(
                    learning_df
                )

            # -----------------------------------------------------------------
            # 3. Diagnostics
            #
            # Temporary research/calibration layer.
            # The production path will eventually remove this.
            # -----------------------------------------------------------------

            with st.spinner(
                "Building diagnostics..."
            ):
                diagnostics = build_all_diagnostics(
                    learning_df=learning_df,
                    uploaded_fit_files=uploaded_fit_files,
                    macro_model=macro_model,
                )

            # -----------------------------------------------------------------
            # Persist everything
            # -----------------------------------------------------------------

            st.session_state["learning_df"] = learning_df
            st.session_state["learning_summary"] = learning_summary
            st.session_state["macro_model"] = macro_model
            st.session_state["diagnostics"] = diagnostics
            st.session_state["fit_signature"] = current_signature

            # New learning run invalidates old leave-one-out results.
            st.session_state["loo_summary"] = None
            st.session_state["loo_detail"] = None

            st.success(
                "Historical learning and macro model completed successfully."
            )

        except Exception as exc:

            st.error(
                f"Learning failed: {exc}"
            )

            st.exception(exc)


# -----------------------------------------------------------------------------
# Persisted combined-learning results
# -----------------------------------------------------------------------------

learning_df = st.session_state["learning_df"]
learning_summary = st.session_state["learning_summary"]
macro_model = st.session_state["macro_model"]
diagnostics = st.session_state["diagnostics"]


if (
    learning_df is not None
    and learning_summary is not None
):

    st.subheader("Historical learning dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Activities",
            learning_summary["n_activities"],
        )

    with col2:
        st.metric(
            "Historical 50 m transitions",
            f"{learning_summary['n_transitions']:,}",
        )

    with col3:
        st.metric(
            "Median 50 m time",
            f"{learning_summary['median_segment_time_s']:.2f} s",
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Mean 50 m time",
            f"{learning_summary['mean_segment_time_s']:.2f} s",
        )

    with col5:
        st.metric(
            "Fastest 50 m",
            f"{learning_summary['min_segment_time_s']:.2f} s",
        )

    with col6:
        st.metric(
            "Slowest 50 m",
            f"{learning_summary['max_segment_time_s']:.2f} s",
        )

    with st.expander(
        "Historical transition data",
        expanded=False,
    ):
        st.dataframe(
            learning_df.head(100),
            width="stretch",
        )

    st.download_button(
        label="Download complete historical learning dataset",
        data=learning_df.to_csv(index=False),
        file_name="historical_learning_dataset_v0.csv",
        mime="text/csv",
        key="download_learning_dataset",
    )


# -----------------------------------------------------------------------------
# Macro results
# -----------------------------------------------------------------------------

if macro_model is not None:

    macro_summary = macro_model.summary()

    st.subheader("Macro model")

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

    st.write(
        f"Historical rows used: "
        f"{macro_summary['training_rows']:,}"
    )

    st.write(
        f"Activities used: "
        f"{macro_summary['training_activities']}"
    )


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

if diagnostics is not None:

    st.divider()

    st.header("Diagnostics")

    # -------------------------------------------------------------------------
    # Per-activity learning
    # -------------------------------------------------------------------------

    activity_learning = diagnostics[
        "activity_learning_summary"
    ]

    with st.expander(
        "Per-activity learning summary",
        expanded=True,
    ):

        st.dataframe(
            activity_learning,
            width="stretch",
        )

        st.download_button(
            label="Download per-activity learning summary",
            data=activity_learning.to_csv(index=False),
            file_name="activity_learning_summary_v0.csv",
            mime="text/csv",
            key="download_activity_learning",
        )

    # -------------------------------------------------------------------------
    # Per-activity macro
    # -------------------------------------------------------------------------

    activity_macro = diagnostics[
        "activity_macro_summary"
    ]

    with st.expander(
        "Per-activity macro summary",
        expanded=True,
    ):

        st.dataframe(
            activity_macro,
            width="stretch",
        )

        st.download_button(
            label="Download per-activity macro summary",
            data=activity_macro.to_csv(index=False),
            file_name="activity_macro_summary_v0.csv",
            mime="text/csv",
            key="download_activity_macro",
        )

    # -------------------------------------------------------------------------
    # Stationary-time analysis
    # -------------------------------------------------------------------------

    stop_summary = diagnostics[
        "stop_summary"
    ]

    with st.expander(
        "Stationary-time analysis",
        expanded=True,
    ):

        st.dataframe(
            stop_summary,
            width="stretch",
        )

        st.download_button(
            label="Download stationary-time summary",
            data=stop_summary.to_csv(index=False),
            file_name="stationary_time_summary_v0.csv",
            mime="text/csv",
            key="download_stop_summary",
        )

    # -------------------------------------------------------------------------
    # Stationary time versus macro error
    # -------------------------------------------------------------------------

    stop_macro_comparison = diagnostics[
        "stop_macro_comparison"
    ]

    with st.expander(
        "Stationary time vs macro error",
        expanded=True,
    ):

        st.dataframe(
            stop_macro_comparison,
            width="stretch",
        )

        st.download_button(
            label="Download stationary vs macro comparison",
            data=stop_macro_comparison.to_csv(index=False),
            file_name="stationary_vs_macro_v0.csv",
            mime="text/csv",
            key="download_stop_macro",
        )

    # -------------------------------------------------------------------------
    # Extreme transitions
    # -------------------------------------------------------------------------

    extreme_transitions = diagnostics[
        "extreme_transitions"
    ]

    with st.expander(
        "Fastest and slowest transitions",
        expanded=False,
    ):

        st.dataframe(
            extreme_transitions,
            width="stretch",
        )

        st.download_button(
            label="Download fastest / slowest transitions",
            data=extreme_transitions.to_csv(index=False),
            file_name="extreme_transitions_v0.csv",
            mime="text/csv",
            key="download_extremes",
        )

    # -------------------------------------------------------------------------
    # Compact learning sample
    # -------------------------------------------------------------------------

    diagnostic_sample = diagnostics[
        "learning_diagnostic_sample"
    ]

    with st.expander(
        "Compact learning diagnostic sample",
        expanded=False,
    ):

        st.dataframe(
            diagnostic_sample,
            width="stretch",
        )

        st.download_button(
            label="Download compact learning diagnostics",
            data=diagnostic_sample.to_csv(index=False),
            file_name="historical_learning_diagnostics_v0.csv",
            mime="text/csv",
            key="download_learning_diagnostics",
        )


# -----------------------------------------------------------------------------
# Temporary leave-one-activity-out validation
# -----------------------------------------------------------------------------
#
# This section is intentionally isolated from the operational learning path.
#
# It is a research/calibration tool only.
# It will be removed or moved entirely into diagnostics before production.
# -----------------------------------------------------------------------------

if (
    learning_df is not None
    and len(
        learning_df[
            "activity_id"
        ].unique()
    ) >= 2
):

    st.divider()

    st.header(
        "Temporary research validation"
    )

    st.write(
        "Leave-one-FIT-out validation: each FIT is predicted using a "
        "micro library built from all the other FITs."
    )

    if st.button(
        "Run leave-one-FIT-out validation",
        type="secondary",
    ):

        try:

            with st.spinner(
                "Running leave-one-FIT-out validation..."
            ):

                loo_summary, loo_detail = (
                    build_leave_one_activity_out_validation(
                        learning_df=learning_df,
                    )
                )

            st.session_state[
                "loo_summary"
            ] = loo_summary

            st.session_state[
                "loo_detail"
            ] = loo_detail

            st.success(
                "Leave-one-FIT-out validation completed."
            )

        except Exception as exc:

            st.error(
                f"Leave-one-FIT-out validation failed: {exc}"
            )

            st.exception(exc)


# -----------------------------------------------------------------------------
# Persisted leave-one-out results
# -----------------------------------------------------------------------------

loo_summary = st.session_state[
    "loo_summary"
]

loo_detail = st.session_state[
    "loo_detail"
]


if (
    loo_summary is not None
    and not loo_summary.empty
):

    st.subheader(
        "Leave-one-FIT-out results"
    )

    st.dataframe(
        loo_summary,
        width="stretch",
    )

    st.download_button(
        label="Download leave-one-FIT-out summary",
        data=loo_summary.to_csv(index=False),
        file_name="leave_one_fit_out_summary_v0.csv",
        mime="text/csv",
        key="download_loo_summary",
    )

    if (
        loo_detail is not None
        and not loo_detail.empty
    ):

        with st.expander(
            "Leave-one-FIT-out segment-level results",
            expanded=False,
        ):

            st.dataframe(
                loo_detail.head(500),
                width="stretch",
            )

            st.download_button(
                label="Download leave-one-FIT-out details",
                data=loo_detail.to_csv(index=False),
                file_name="leave_one_fit_out_details_v0.csv",
                mime="text/csv",
                key="download_loo_details",
            )
            
# =============================================================================
# TEMPORARY GPX PROFILE TEST
# =============================================================================

st.divider()
st.header("GPX 50 m profile test")

from gpx_profile import (
    build_gpx_profile,
    summarize_gpx_profile,
)

uploaded_gpx_file = st.file_uploader(
    "Upload a GPX file for normalization test",
    type=["gpx"],
    key="gpx_profile_test_upload",
)

if uploaded_gpx_file is not None:

    if st.button(
        "Build 50 m GPX profile",
        key="build_gpx_profile_test",
    ):

        try:

            with st.spinner(
                "Building normalized 50 m GPX profile..."
            ):
                gpx_profile_df = build_gpx_profile(
                    uploaded_gpx_file,
                    aid_stations=None,
                )

            st.session_state[
                "gpx_profile_test_df"
            ] = gpx_profile_df

            st.success(
                "50 m GPX profile created."
            )

        except Exception as exc:

            st.error(
                f"GPX profile failed: {exc}"
            )

            st.exception(exc)


gpx_profile_df = st.session_state.get(
    "gpx_profile_test_df"
)

if (
    gpx_profile_df is not None
    and not gpx_profile_df.empty
):

    profile_summary = summarize_gpx_profile(
        gpx_profile_df
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "50 m segments",
            f"{profile_summary['n_segments']:,}",
        )

    with col2:
        st.metric(
            "Normalized distance",
            f"{profile_summary['distance_m'] / 1000:.2f} km",
        )

    with col3:
        st.metric(
            "Cumulative ascent",
            f"{profile_summary['cumulative_ascent_m']:.0f} m",
        )

    st.write(
        f"Cumulative descent: "
        f"{profile_summary['cumulative_descent_m']:.0f} m"
    )

    st.dataframe(
        gpx_profile_df,
        width="stretch",
    )

    st.download_button(
        label="Download normalized GPX profile",
        data=gpx_profile_df.to_csv(index=False),
        file_name="normalized_gpx_profile_v0.csv",
        mime="text/csv",
        key="download_gpx_profile_test",
    )
