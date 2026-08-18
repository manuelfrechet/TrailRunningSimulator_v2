from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from fit_learning import (
    build_learning_dataset,
    summarize_learning_dataset,
)

from macro_model import (
    fit_macro_model,
    predict_macro_profile,
)

from learning_diagnostics import (
    build_learning_diagnostic_sample,
    build_activity_learning_summary,
    build_activity_macro_summary,
    build_extreme_transition_summary,
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
# Historical FIT learning
# -----------------------------------------------------------------------------

st.header("1. Historical FIT learning")

uploaded_fit_files = st.file_uploader(
    "Upload one or more historical FIT files",
    type=["fit"],
    accept_multiple_files=True,
)

if not uploaded_fit_files:
    st.info("Upload one or more FIT files to begin.")
    st.stop()


# -----------------------------------------------------------------------------
# Run historical learning
# -----------------------------------------------------------------------------

if st.button(
    "Build historical learning + macro model",
    type="primary",
):

    try:

        # ---------------------------------------------------------------------
        # Historical transition dataset
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Store learning dataset
        # ---------------------------------------------------------------------

        st.session_state["learning_df"] = learning_df

        # ---------------------------------------------------------------------
        # Global learning summary
        # ---------------------------------------------------------------------

        summary = summarize_learning_dataset(
            learning_df
        )

        st.success(
            "Historical learning dataset created successfully."
        )

        st.subheader("Historical learning dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Activities",
                summary["n_activities"],
            )

        with col2:
            st.metric(
                "Historical 50 m transitions",
                f"{summary['n_transitions']:,}",
            )

        with col3:
            st.metric(
                "Median 50 m time",
                f"{summary['median_segment_time_s']:.2f} s",
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric(
                "Mean 50 m time",
                f"{summary['mean_segment_time_s']:.2f} s",
            )

        with col5:
            st.metric(
                "Fastest 50 m",
                f"{summary['min_segment_time_s']:.2f} s",
            )

        with col6:
            st.metric(
                "Slowest 50 m",
                f"{summary['max_segment_time_s']:.2f} s",
            )

        # ---------------------------------------------------------------------
        # Historical transition preview
        # ---------------------------------------------------------------------

        with st.expander(
            "Historical transition data",
            expanded=False,
        ):
            st.dataframe(
                learning_df.head(100),
                width="stretch",
            )

        # ---------------------------------------------------------------------
        # Complete learning dataset export
        # ---------------------------------------------------------------------

        st.download_button(
            label="Download complete historical learning dataset",
            data=learning_df.to_csv(index=False),
            file_name="historical_learning_dataset_v0.csv",
            mime="text/csv",
        )

        # ---------------------------------------------------------------------
        # Compact diagnostic sample
        # ---------------------------------------------------------------------

        diagnostic_df = build_learning_diagnostic_sample(
            learning_df
        )

        if not diagnostic_df.empty:

            st.download_button(
                label="Download compact learning diagnostics",
                data=diagnostic_df.to_csv(index=False),
                file_name="historical_learning_diagnostics_v0.csv",
                mime="text/csv",
            )

        # ---------------------------------------------------------------------
        # Per-activity learning summary
        # ---------------------------------------------------------------------

        activity_learning_summary = (
            build_activity_learning_summary(
                learning_df
            )
        )

        st.session_state[
            "activity_learning_summary"
        ] = activity_learning_summary

        st.subheader(
            "Per-activity learning summary"
        )

        if activity_learning_summary.empty:
            st.warning(
                "No per-activity learning summary could be generated."
            )
        else:
            st.dataframe(
                activity_learning_summary,
                width="stretch",
            )

            st.download_button(
                label="Download per-activity learning summary",
                data=activity_learning_summary.to_csv(
                    index=False
                ),
                file_name="activity_learning_summary_v0.csv",
                mime="text/csv",
            )

        # ---------------------------------------------------------------------
        # Extreme transitions
        # ---------------------------------------------------------------------

        extreme_df = build_extreme_transition_summary(
            learning_df,
            n_each=20,
        )

        st.session_state[
            "extreme_transition_summary"
        ] = extreme_df

        st.subheader(
            "Fastest and slowest historical transitions"
        )

        if extreme_df.empty:
            st.warning(
                "No extreme-transition diagnostics could be generated."
            )
        else:
            st.dataframe(
                extreme_df,
                width="stretch",
            )

            st.download_button(
                label="Download fastest / slowest transitions",
                data=extreme_df.to_csv(
                    index=False
                ),
                file_name="extreme_transitions_v0.csv",
                mime="text/csv",
            )

        # ---------------------------------------------------------------------
        # Macro model
        # ---------------------------------------------------------------------

        with st.spinner(
            "Fitting V0 macro model..."
        ):
            macro_model = fit_macro_model(
                learning_df
            )

        st.session_state[
            "macro_model"
        ] = macro_model

        st.success(
            "Macro model fitted successfully."
        )

        # ---------------------------------------------------------------------
        # Macro summary
        # ---------------------------------------------------------------------

        macro_summary = macro_model.summary()

        st.subheader("Macro model")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Training MAE",
                f"{macro_summary['training_mae_s']:.2f} s",
            )

        with metric2:
            st.metric(
                "Training RMSE",
                f"{macro_summary['training_rmse_s']:.2f} s",
            )

        with metric3:
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

        # ---------------------------------------------------------------------
        # Macro historical check
        # ---------------------------------------------------------------------

        macro_profile_df = predict_macro_profile(
            macro_model,
            learning_df[
                [
                    "distance_from_start_m",
                    "cumulative_ascent_m",
                    "cumulative_descent_m",
                ]
            ],
        )

        macro_check_df = learning_df[
            [
                "activity_id",
                "activity_name",
                "distance_from_start_m",
                "cumulative_ascent_m",
                "cumulative_descent_m",
                "elapsed_time_s",
            ]
        ].copy()

        macro_check_df[
            "macro_predicted_cumulative_time_s"
        ] = (
            macro_profile_df[
                "macro_predicted_cumulative_time_s"
            ]
            .to_numpy()
        )

        macro_check_df[
            "macro_error_s"
        ] = (
            macro_check_df[
                "macro_predicted_cumulative_time_s"
            ]
            - macro_check_df[
                "elapsed_time_s"
            ]
        )

        macro_check_df[
            "macro_error_abs_s"
        ] = macro_check_df[
            "macro_error_s"
        ].abs()

        # ---------------------------------------------------------------------
        # Global macro check metrics
        # ---------------------------------------------------------------------

        st.subheader(
            "Historical macro-model check"
        )

        st.write(
            "This compares the macro model against the actual cumulative "
            "elapsed time observed in the same historical FIT corpus."
        )

        check_col1, check_col2, check_col3 = st.columns(3)

        with check_col1:
            st.metric(
                "Mean absolute cumulative error",
                f"{macro_check_df['macro_error_abs_s'].mean():.1f} s",
            )

        with check_col2:
            st.metric(
                "Median absolute cumulative error",
                f"{macro_check_df['macro_error_abs_s'].median():.1f} s",
            )

        with check_col3:
            st.metric(
                "Mean cumulative bias",
                f"{macro_check_df['macro_error_s'].mean():.1f} s",
            )

        # ---------------------------------------------------------------------
        # Per-activity macro diagnostic
        # ---------------------------------------------------------------------

        activity_macro_summary = (
            build_activity_macro_summary(
                learning_df,
                macro_model,
            )
        )

        st.session_state[
            "activity_macro_summary"
        ] = activity_macro_summary

        st.subheader(
            "Per-activity macro-model summary"
        )

        if activity_macro_summary.empty:
            st.warning(
                "No per-activity macro summary could be generated."
            )
        else:
            st.dataframe(
                activity_macro_summary,
                width="stretch",
            )

            st.download_button(
                label="Download per-activity macro summary",
                data=activity_macro_summary.to_csv(
                    index=False
                ),
                file_name="activity_macro_summary_v0.csv",
                mime="text/csv",
            )

        # ---------------------------------------------------------------------
        # Historical macro check preview
        # ---------------------------------------------------------------------

        with st.expander(
            "First 100 historical macro predictions",
            expanded=True,
        ):
            st.dataframe(
                macro_check_df.head(100),
                width="stretch",
            )

        st.download_button(
            label="Download complete macro historical check",
            data=macro_check_df.to_csv(index=False),
            file_name="macro_historical_check_v0.csv",
            mime="text/csv",
        )

        # ---------------------------------------------------------------------
        # Macro residual information
        # ---------------------------------------------------------------------

        residual_col1, residual_col2, residual_col3 = st.columns(3)

        with residual_col1:
            st.metric(
                "Macro residual median",
                f"{macro_summary['residual_median_s']:.1f} s",
            )

        with residual_col2:
            st.metric(
                "Macro residual Q10",
                f"{macro_summary['residual_q10_s']:.1f} s",
            )

        with residual_col3:
            st.metric(
                "Macro residual Q90",
                f"{macro_summary['residual_q90_s']:.1f} s",
            )

    except Exception as exc:

        st.error(
            f"Learning failed: {exc}"
        )

        st.exception(exc)


# -----------------------------------------------------------------------------
# Persisted results after Streamlit rerun
# -----------------------------------------------------------------------------

if (
    "activity_learning_summary"
    in st.session_state
    and st.session_state[
        "activity_learning_summary"
    ] is not None
):

    # No additional work here.
    # This section intentionally does not recompute anything.
    pass
    
