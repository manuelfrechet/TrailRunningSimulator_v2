from __future__ import annotations

import streamlit as st

from fit_learning import (
    build_learning_dataset,
    summarize_learning_dataset,
)

from macro_model import fit_macro_model

from diagnostics import (
    build_all_diagnostics,
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
}


for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# -----------------------------------------------------------------------------
# FIT signature
# -----------------------------------------------------------------------------

def _fit_signature(uploaded_files) -> tuple[tuple[str, int | None], ...]:
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
# Historical FIT learning
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
# New FIT selection
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


# -----------------------------------------------------------------------------
# Main execution
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
            # 2. Macro learning
            # -----------------------------------------------------------------

            with st.spinner(
                "Fitting V0 macro model..."
            ):
                macro_model = fit_macro_model(
                    learning_df
                )

            # -----------------------------------------------------------------
            # 3. Optional diagnostics
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
            # Store EVERYTHING
            # -----------------------------------------------------------------

            st.session_state["learning_df"] = learning_df
            st.session_state["learning_summary"] = learning_summary
            st.session_state["macro_model"] = macro_model
            st.session_state["diagnostics"] = diagnostics
            st.session_state["fit_signature"] = current_signature

            st.success(
                "Historical learning and macro model completed successfully."
            )

        except Exception as exc:

            st.error(
                f"Learning failed: {exc}"
            )

            st.exception(exc)


# -----------------------------------------------------------------------------
# Display persisted learning results
# -----------------------------------------------------------------------------

learning_df = st.session_state["learning_df"]
learning_summary = st.session_state["learning_summary"]
macro_model = st.session_state["macro_model"]
diagnostics = st.session_state["diagnostics"]


if learning_df is not None and learning_summary is not None:

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
    # Stop / macro comparison
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
    # Compact diagnostic sample
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

    # -------------------------------------------------------------------------
    # Raw stationary intervals
    # -------------------------------------------------------------------------

    stop_intervals = diagnostics[
        "stop_intervals"
    ]

    with st.expander(
        "Raw stationary intervals",
        expanded=False,
    ):

        st.dataframe(
            stop_intervals.head(500),
            width="stretch",
        )

        st.download_button(
            label="Download all stationary intervals",
            data=stop_intervals.to_csv(index=False),
            file_name="stationary_intervals_v0.csv",
            mime="text/csv",
            key="download_stationary_intervals",
        )

    # -------------------------------------------------------------------------
    # Full macro historical check
    # -------------------------------------------------------------------------

    macro_check = diagnostics[
        "macro_historical_check"
    ]

    with st.expander(
        "Full historical macro check",
        expanded=False,
    ):

        st.dataframe(
            macro_check.head(500),
            width="stretch",
        )

        st.download_button(
            label="Download complete macro historical check",
            data=macro_check.to_csv(index=False),
            file_name="macro_historical_check_v0.csv",
            mime="text/csv",
            key="download_macro_check",
        )
        
