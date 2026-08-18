from __future__ import annotations

import pandas as pd
import streamlit as st

from fit_learning import (
    build_learning_dataset,
    summarize_learning_dataset,
)

from micro_model import fit_micro_model


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Trail Running Simulator V2",
    layout="wide",
)

st.title("Trail Running Simulator V2")

st.header("V0 — Micro analog model test")


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

if "learning_df" not in st.session_state:
    st.session_state["learning_df"] = None

if "micro_model" not in st.session_state:
    st.session_state["micro_model"] = None

if "micro_test_df" not in st.session_state:
    st.session_state["micro_test_df"] = None


# -----------------------------------------------------------------------------
# FIT input
# -----------------------------------------------------------------------------

uploaded_fit_files = st.file_uploader(
    "Upload one or more historical FIT files",
    type=["fit"],
    accept_multiple_files=True,
)


# -----------------------------------------------------------------------------
# Build models
# -----------------------------------------------------------------------------

if uploaded_fit_files:

    if st.button(
        "Build historical data + test micro model",
        type="primary",
    ):

        try:
            # -----------------------------------------------------------------
            # Historical dataset
            # -----------------------------------------------------------------

            with st.spinner(
                "Building historical learning dataset..."
            ):
                learning_df = build_learning_dataset(
                    uploaded_fit_files
                )

            if learning_df.empty:
                st.error(
                    "No historical transitions were created."
                )
                st.stop()

            st.session_state["learning_df"] = learning_df

            summary = summarize_learning_dataset(
                learning_df
            )

            st.success(
                "Historical learning dataset created."
            )

            st.write(
                f"Activities: {summary['n_activities']}"
            )

            st.write(
                f"Historical transitions: "
                f"{summary['n_transitions']:,}"
            )

            # -----------------------------------------------------------------
            # Micro model
            # -----------------------------------------------------------------

            with st.spinner(
                "Building historical micro analog library..."
            ):
                micro_model = fit_micro_model(
                    learning_df
                )

            st.session_state["micro_model"] = micro_model

            micro_summary = micro_model.summary()

            st.success(
                "Micro analog model created."
            )

            st.write(
                f"Historical states: "
                f"{micro_summary['training_rows']:,}"
            )

            st.write(
                f"Activities: "
                f"{micro_summary['training_activities']}"
            )

            st.write(
                f"State variables: "
                f"{micro_summary['n_state_variables']}"
            )

            # -----------------------------------------------------------------
            # Validation sample
            # -----------------------------------------------------------------
            #
            # Important:
            # We deliberately start with a SIMPLE sanity check.
            #
            # We query historical rows using the same historical library.
            # This is NOT an out-of-sample performance evaluation.
            #
            # It is only intended to verify that:
            #
            #   1. state vectors are constructed correctly;
            #   2. nearest neighbours make sense;
            #   3. interpolation works;
            #   4. the KD-tree is functioning.
            #
            # We exclude each queried row itself from consideration by asking
            # for three neighbours and ignoring the exact self-match.
            # -----------------------------------------------------------------

            test_df = learning_df.sample(
                n=min(
                    100,
                    len(learning_df),
                ),
                random_state=42,
            ).copy()

            rows = []

            for _, row in test_df.iterrows():

                query = micro_model.build_query(
                    distance_from_start_m=float(
                        row["distance_from_start_m"]
                    ),
                    cumulative_ascent_m=float(
                        row["cumulative_ascent_m"]
                    ),
                    cumulative_descent_m=float(
                        row["cumulative_descent_m"]
                    ),
                    elapsed_time_s=float(
                        row["elapsed_time_s"]
                    ),
                    segment_ascent_m=float(
                        row["segment_ascent_m"]
                    ),
                    segment_descent_m=float(
                        row["segment_descent_m"]
                    ),
                    segment_grade_pct=float(
                        row["segment_grade_pct"]
                    ),
                )

                prediction_df = micro_model.predict(
                    query
                )

                prediction = prediction_df.iloc[0]

                rows.append(
                    {
                        "activity_id": row["activity_id"],
                        "activity_name": row["activity_name"],
                        "distance_from_start_m": row[
                            "distance_from_start_m"
                        ],
                        "actual_segment_time_s": row[
                            "actual_segment_time_s"
                        ],
                        "micro_predicted_time_s": prediction[
                            "micro_predicted_time_s"
                        ],
                        "prediction_error_s": (
                            prediction[
                                "micro_predicted_time_s"
                            ]
                            - row[
                                "actual_segment_time_s"
                            ]
                        ),
                        "analogue_1_distance": prediction[
                            "analogue_1_distance"
                        ],
                        "analogue_1_time_s": prediction[
                            "analogue_1_time_s"
                        ],
                        "analogue_1_activity_id": prediction[
                            "analogue_1_activity_id"
                        ],
                        "analogue_1_activity_name": prediction[
                            "analogue_1_activity_name"
                        ],
                        "analogue_1_distance_from_start_m": prediction[
                            "analogue_1_distance_from_start_m"
                        ],
                        "analogue_2_distance": prediction[
                            "analogue_2_distance"
                        ],
                        "analogue_2_time_s": prediction[
                            "analogue_2_time_s"
                        ],
                        "analogue_2_activity_id": prediction[
                            "analogue_2_activity_id"
                        ],
                        "analogue_2_activity_name": prediction[
                            "analogue_2_activity_name"
                        ],
                        "analogue_2_distance_from_start_m": prediction[
                            "analogue_2_distance_from_start_m"
                        ],
                    }
                )

            micro_test_df = pd.DataFrame(
                rows
            )

            st.session_state[
                "micro_test_df"
            ] = micro_test_df

        except Exception as exc:
            st.error(
                f"Micro model test failed: {exc}"
            )
            st.exception(exc)


# -----------------------------------------------------------------------------
# Display persisted micro test
# -----------------------------------------------------------------------------

micro_test_df = st.session_state.get(
    "micro_test_df"
)

micro_model = st.session_state.get(
    "micro_model"
)


if micro_test_df is not None and not micro_test_df.empty:

    st.subheader(
        "Micro analog validation sample"
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Sample size",
            f"{len(micro_test_df):,}",
        )

    with metric2:
        st.metric(
            "Mean absolute error",
            f"{micro_test_df['prediction_error_s'].abs().mean():.2f} s",
        )

    with metric3:
        st.metric(
            "Median absolute error",
            f"{micro_test_df['prediction_error_s'].abs().median():.2f} s",
        )

    st.dataframe(
        micro_test_df,
        width="stretch",
    )

    st.download_button(
        "Download micro validation sample",
        data=micro_test_df.to_csv(index=False),
        file_name="micro_validation_sample_v0.csv",
        mime="text/csv",
    )
    
