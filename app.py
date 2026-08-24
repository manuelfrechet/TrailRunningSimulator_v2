from __future__ import annotations

import streamlit as st

from config import (
    TRANSITION_LENGTH_M,
)

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

if "gpx_file_signature" not in st.session_state:
    st.session_state["gpx_file_signature"] = None


# =============================================================================
# GPX upload
# =============================================================================

uploaded_gpx_file = st.file_uploader(
    "Upload one GPX file",
    type=["gpx"],
    key="gpx_test_upload",
)


# =============================================================================
# Detect new GPX
# =============================================================================

if uploaded_gpx_file is not None:

    current_signature = (
        uploaded_gpx_file.name,
        getattr(
            uploaded_gpx_file,
            "size",
            None,
        ),
    )

    if (
        st.session_state[
            "gpx_file_signature"
        ]
        != current_signature
    ):
        st.session_state[
            "gpx_file_signature"
        ] = current_signature

        st.session_state[
            "raw_gpx_df"
        ] = None

        st.session_state[
            "gpx_profile_df"
        ] = None


# =============================================================================
# Display current configuration
# =============================================================================

st.info(
    f"Current learning step: "
    f"{1.0:.0f} m | "
    f"Current transition / prediction segment: "
    f"{TRANSITION_LENGTH_M:.0f} m"
)


# =============================================================================
# Load raw GPX table
# =============================================================================

if uploaded_gpx_file is not None:

    if st.button(
        "Load raw GPX table",
        key="load_raw_gpx",
    ):

        try:

            with st.spinner(
                "Reading raw GPX points..."
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

                # The GPX has been successfully re-read,
                # but the normalized profile can remain associated
                # with the same GPX. We do not rebuild it here.

                st.success(
                    "Raw GPX table loaded."
                )

        except Exception as exc:

            st.error(
                f"Raw GPX loading failed: {exc}"
            )

            st.exception(exc)


# =============================================================================
# Raw GPX table
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
        f"Raw GPX points: "
        f"{len(raw_gpx_df):,}"
    )

    st.write(
        "The table below is the raw GPX trajectory with "
        "point-to-point distance and raw terrain calculations."
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
# Normalized GPX profile
# =============================================================================

if uploaded_gpx_file is not None:

    st.subheader(
        f"{TRANSITION_LENGTH_M:.0f} m GPX normalization"
    )

    if st.button(
        f"Build normalized {TRANSITION_LENGTH_M:.0f} m GPX profile",
        type="primary",
        key="build_gpx_profile",
    ):

        try:

            with st.spinner(
                f"Building normalized "
                f"{TRANSITION_LENGTH_M:.0f} m GPX profile..."
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
                f"Normalized "
                f"{TRANSITION_LENGTH_M:.0f} m GPX profile created."
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

    profile_summary = (
        summarize_gpx_profile(
            gpx_profile_df
        )
    )

    st.subheader(
        "Normalized GPX profile"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            f"{TRANSITION_LENGTH_M:.0f} m segments",
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
        "Normalized distances:"
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
    
