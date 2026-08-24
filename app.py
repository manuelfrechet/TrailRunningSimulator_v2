from __future__ import annotations

import streamlit as st

from gpx_profile import (
    build_gpx_profile,
    load_raw_gpx_table,
    summarize_gpx_profile,
)


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="Trail Running Simulator V2",
    layout="wide",
)

st.title("Trail Running Simulator V2")

st.header("Temporary GPX normalization test")


# =============================================================================
# Session state
# =============================================================================

if "raw_gpx_df" not in st.session_state:
    st.session_state["raw_gpx_df"] = None

if "gpx_profile_df" not in st.session_state:
    st.session_state["gpx_profile_df"] = None


# =============================================================================
# GPX upload
# =============================================================================

uploaded_gpx_file = st.file_uploader(
    "Upload one GPX file",
    type=["gpx"],
    key="gpx_test_upload",
)


# =============================================================================
# Build raw GPX table
# =============================================================================

if uploaded_gpx_file is not None:

    if st.button(
        "Load raw GPX table",
        key="load_raw_gpx",
    ):

        try:

            raw_gpx_df = load_raw_gpx_table(
                uploaded_gpx_file
            )

            if raw_gpx_df.empty:
                st.error(
                    "No GPX track points were found."
                )
            else:

                st.session_state[
                    "raw_gpx_df"
                ] = raw_gpx_df

                # New GPX means the normalized profile is obsolete.
                st.session_state[
                    "gpx_profile_df"
                ] = None

                st.success(
                    "Raw GPX table loaded."
                )

        except Exception as exc:

            st.error(
                f"Raw GPX loading failed: {exc}"
            )

            st.exception(exc)


# =============================================================================
# Display raw GPX table
# =============================================================================

raw_gpx_df = st.session_state[
    "raw_gpx_df"
]

if (
    raw_gpx_df is not None
    and not raw_gpx_df.empty
):

    st.subheader(
        "Raw GPX table"
    )

    st.write(
        f"Raw GPX points: {len(raw_gpx_df):,}"
    )

    st.dataframe(
        raw_gpx_df.head(500),
        width="stretch",
    )

    st.download_button(
        label="Download complete raw GPX table",
        data=raw_gpx_df.to_csv(
            index=False
        ),
        file_name="raw_gpx_table.csv",
        mime="text/csv",
        key="download_raw_gpx",
    )


# =============================================================================
# Normalize GPX
# =============================================================================

if uploaded_gpx_file is not None:

    st.subheader(
        "50 m GPX normalization"
    )

    if st.button(
        "Build normalized 50 m GPX profile",
        type="primary",
        key="build_gpx_profile",
    ):

        try:

            with st.spinner(
                "Building normalized 50 m GPX profile..."
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

            st.success(
                "Normalized 50 m GPX profile created."
            )

        except Exception as exc:

            st.error(
                f"GPX normalization failed: {exc}"
            )

            st.exception(exc)


# =============================================================================
# Display normalized profile
# =============================================================================

gpx_profile_df = st.session_state[
    "gpx_profile_df"
]

if (
    gpx_profile_df is not None
    and not gpx_profile_df.empty
):

    profile_summary = summarize_gpx_profile(
        gpx_profile_df
    )

    st.subheader(
        "Normalized 50 m GPX profile"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "50 m segments",
            f"{profile_summary['n_segments']:,}",
        )

    with col2:

        st.metric(
            "Distance",
            f"{profile_summary['distance_m'] / 1000:.2f} km",
        )

    with col3:

        st.metric(
            "Cumulative ascent",
            f"{profile_summary['cumulative_ascent_m']:.0f} m",
        )

    with col4:

        st.metric(
            "Cumulative descent",
            f"{profile_summary['cumulative_descent_m']:.0f} m",
        )

    st.write(
        "The normalized distance should progress "
        "50, 100, 150, ... metres."
    )

    st.dataframe(
        gpx_profile_df.head(500),
        width="stretch",
    )

    st.download_button(
        label="Download complete normalized GPX profile",
        data=gpx_profile_df.to_csv(
            index=False
        ),
        file_name="normalized_gpx_profile_v0.csv",
        mime="text/csv",
        key="download_normalized_gpx",
    )
    
