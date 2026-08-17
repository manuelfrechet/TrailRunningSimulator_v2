import streamlit as st

from fit_learning import (
    build_learning_dataset,
    summarize_learning_dataset,
)


st.title("Trail Running Simulator V2")

st.header("V0 — FIT learning test")

uploaded_fit_files = st.file_uploader(
    "Upload one or more historical FIT files",
    type=["fit"],
    accept_multiple_files=True,
)

if uploaded_fit_files:
    if st.button("Build historical learning dataset"):
        try:
            with st.spinner("Processing FIT files..."):
                learning_df = build_learning_dataset(
                    uploaded_fit_files
                )

            if learning_df.empty:
                st.error(
                    "The FIT files were read, but no historical "
                    "50 m transitions were produced."
                )

            else:
                summary = summarize_learning_dataset(
                    learning_df
                )

                st.success("Historical learning dataset created.")

                st.subheader("Summary")

                st.write(
                    f"Activities: {summary['n_activities']}"
                )
                st.write(
                    f"Historical transitions: "
                    f"{summary['n_transitions']}"
                )
                st.write(
                    f"Mean 50 m time: "
                    f"{summary['mean_segment_time_s']:.2f} s"
                )
                st.write(
                    f"Median 50 m time: "
                    f"{summary['median_segment_time_s']:.2f} s"
                )
                st.write(
                    f"Fastest 50 m: "
                    f"{summary['min_segment_time_s']:.2f} s"
                )
                st.write(
                    f"Slowest 50 m: "
                    f"{summary['max_segment_time_s']:.2f} s"
                )

                st.subheader("First 20 historical transitions")

                st.dataframe(
                    learning_df.head(20),
                    width="stretch",
                )

                st.download_button(
                    "Download historical learning dataset CSV",
                    data=learning_df.to_csv(index=False),
                    file_name="historical_learning_dataset_v0.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"FIT processing failed: {exc}")
            st.exception(exc)
          
