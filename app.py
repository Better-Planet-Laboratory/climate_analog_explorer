from __future__ import annotations

import html
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import streamlit as st

from config import *

try:
    from climate_search_prod.search import ClimateSearchEngine
    from climate_search_prod.geography import place_label
except ImportError:
    from scripts.climate_search_prod.search import ClimateSearchEngine
    from scripts.climate_search_prod.geography import place_label

from src.geocoding import geocode_place
from src.map_utils import build_globe
from src.plots import (
    comparison_from_bio_vectors,
    BIOCLIM_DESCRIPTIONS,
    sidebar_climate_difference_plot,
    novelty_distribution_plot,
    novelty_percentile,
)


st.set_page_config(
    page_title="Climate Analog Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

HIST = HISTORICAL_LABEL


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    header[data-testid="stHeader"], footer { display:none; }
    #MainMenu { visibility:hidden; }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        overflow:hidden;
    }

    [data-testid="stMainBlockContainer"] {
        padding:0 !important;
        max-width:100% !important;
    }

    [data-testid="stSidebar"] {
        min-width:390px !important;
        max-width:390px !important;
    }

    [data-testid="stSidebarContent"] {
        overflow-y:auto !important;
    }

    .landing-wrap {
        max-width:760px;
        margin:12vh auto 0;
        text-align:center;
        padding:2rem;
    }

    .landing-title {
        font-size:3rem;
        font-weight:750;
        line-height:1.05;
        margin-bottom:1rem;
    }

    .landing-copy {
        font-size:1.08rem;
        opacity:.82;
        line-height:1.55;
        margin-bottom:1.6rem;
    }

    .globe-title {
        position:fixed;
        top:18px;
        left:430px;
        z-index:20;
        padding:8px 13px;
        border-radius:10px;
        background:rgba(10,16,28,.78);
        backdrop-filter:blur(7px);
        color:white;
        pointer-events:none;
    }

    /* Collapsible table is offset from the edges and can be minimized. */
    details.analog-overlay {
        position:fixed;
        right:34px;
        bottom:34px;
        width:min(440px,38vw);
        z-index:60;
        border-radius:12px;
        background:rgba(12,18,30,.90);
        backdrop-filter:blur(9px);
        color:white;
        box-shadow:0 4px 18px rgba(0,0,0,.30);
        pointer-events:auto;
        overflow:hidden;
    }

    details.analog-overlay summary {
        cursor:pointer;
        padding:9px 12px;
        font-size:.86rem;
        font-weight:700;
        user-select:none;
    }

    details.analog-overlay[open] summary {
        border-bottom:1px solid rgba(255,255,255,.16);
    }

    .analog-table-wrap {
        padding:7px 10px 10px 10px;
        max-height:255px;
        overflow:auto;
    }

    .analog-overlay table {
        width:100%;
        border-collapse:collapse;
        font-size:.70rem;
    }

    .analog-overlay th {
        text-align:left;
        opacity:.70;
        padding:2px 4px;
        border-bottom:1px solid rgba(255,255,255,.18);
    }

    .analog-overlay td {
        padding:2px 4px;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
        max-width:145px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Search helpers
# ============================================================

def engine_for(q, t):
    qc, tc = PERIODS[q], PERIODS[t]

    return ClimateSearchEngine(
        query_database=qc["database"],
        target_database=tc["database"],
        target_index_json=tc["index_json"],
        query_chip_h5=(
            qc["chip_h5"]
            if qc["chip_h5"].exists()
            else None
        ),
        target_chip_h5=(
            tc["chip_h5"]
            if tc["chip_h5"].exists()
            else None
        ),
    )


def objects(event):
    try:
        return dict(event.selection.objects)
    except Exception:
        try:
            return dict(event["selection"]["objects"])
        except Exception:
            return {}


def clicked_cell(o):
    picked = o.get("climate-cells", [])

    if not picked:
        return None

    x = picked[0].get(
        "properties",
        picked[0],
    )

    if (
        "center_lat" not in x
        or "center_lon" not in x
    ):
        return None

    return (
        float(x["center_lat"]),
        float(x["center_lon"]),
    )


def clicked_rank(o):
    picked = o.get(
        "analog-markers",
        [],
    )

    if not picked:
        return None

    x = picked[0].get(
        "properties",
        picked[0],
    )

    return (
        int(x["rank"])
        if "rank" in x
        else None
    )


def result_df(payload):
    return pd.DataFrame(
        [
            {
                "rank": x["rank"],
                "row": x["row"],
                "chip_id": x["chip_id"],
                "center_lat": x["lat"],
                "center_lon": x["lon"],
                "nearest_city": x.get("nearest_city", ""),
                "admin1": x.get("admin1", ""),
                "country": x.get("country", ""),
                "city_distance_km": x.get(
                    "city_distance_km",
                    np.nan,
                ),
                "embedding_similarity": x[
                    "embedding_similarity"
                ],
                "embedding_distance": x[
                    "embedding_distance"
                ],
                "climate_distance": x[
                    "climate_distance"
                ],
                "geographic_distance_km": x[
                    "geographic_distance_km"
                ],
            }
            for x in payload["results"]
        ]
    )


# ============================================================
# BIOCLIM comparison
# ============================================================

@st.cache_data(show_spinner=False, max_entries=64)
def load_bio_pair(
    query_database: str,
    target_database: str,
    query_row: int,
    analog_row: int,
):
    """
    Read the 19 BIOCLIM summaries directly from the production search DB.

    This deliberately compares the selected query against analog #1,
    independently of which marker the user has clicked.
    """

    with h5py.File(
        query_database,
        "r",
    ) as qh:

        if "bio_mean" not in qh:
            raise KeyError(
                f"{query_database} does not contain bio_mean"
            )

        query_bio = np.asarray(
            qh["bio_mean"][int(query_row)],
            dtype=np.float32,
        )

    with h5py.File(
        target_database,
        "r",
    ) as th:

        if "bio_mean" not in th:
            raise KeyError(
                f"{target_database} does not contain bio_mean"
            )

        analog_bio = np.asarray(
            th["bio_mean"][int(analog_row)],
            dtype=np.float32,
        )

    return query_bio, analog_bio


# ============================================================
# Novelty
# ============================================================

def novelty_path_for(label):
    cfg = PERIODS[label]

    return Path(
        cfg.get(
            "novelty",
            NOVELTY_DIR
            / f"{cfg['key']}_novelty.parquet",
        )
    )


@st.cache_data(show_spinner=False)
def load_novelty(label):
    path = novelty_path_for(label)

    if not path.exists():
        return pd.DataFrame(), f"Novelty file not found: {path}"

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return pd.DataFrame(), f"Could not read novelty file: {exc}"

    # Normalize common output column names.
    if "novelty" not in df.columns:
        candidates = [
            "novelty_exact",
            "exact_novelty",
            "climate_novelty",
            "novelty_distance",
            "min_exact_distance",
            "min_climate_distance",
        ]
        found = next((c for c in candidates if c in df.columns), None)
        if found is not None:
            df = df.rename(columns={found: "novelty"})

    # Most novelty outputs preserve row. If not, the parquet is expected to
    # be in search-database row order, so create it explicitly.
    if "row" not in df.columns:
        df = df.reset_index(drop=True)
        df["row"] = np.arange(len(df), dtype=np.int64)

    if "novelty" not in df.columns:
        return (
            pd.DataFrame(),
            "Novelty parquet was found, but no novelty column was recognized. "
            f"Columns: {list(df.columns)}",
        )

    df["row"] = pd.to_numeric(df["row"], errors="coerce")
    df["novelty"] = pd.to_numeric(df["novelty"], errors="coerce")

    return df, None


def novelty_value(df, row):
    if df.empty or "novelty" not in df.columns:
        return None

    hit = df.loc[
        df["row"].astype("Int64") == int(row),
        "novelty",
    ]

    if hit.empty or pd.isna(hit.iloc[0]):
        return None

    return float(hit.iloc[0])


# ============================================================
# Exact global climate distance
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=8,
)
def exact_global_distance_field(
    query_database,
    target_database,
    query_row,
    chunk_size=4096,
):
    with h5py.File(
        query_database,
        "r",
    ) as qh, h5py.File(
        target_database,
        "r",
    ) as th:

        q = np.asarray(
            qh["coarse_climate"][
                int(query_row)
            ],
            dtype=np.float32,
        )

        qv = np.asarray(
            qh["coarse_valid"][
                int(query_row)
            ],
            dtype=bool,
        )

        n = len(
            th["center_lat"]
        )

        lat = (
            th["center_lat"][:]
            .astype(np.float32)
        )

        lon = (
            th["center_lon"][:]
            .astype(np.float32)
        )

        dist = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )

        for start in range(
            0,
            n,
            chunk_size,
        ):

            stop = min(
                start + chunk_size,
                n,
            )

            x = np.asarray(
                th["coarse_climate"][
                    start:stop
                ],
                dtype=np.float32,
            )

            xv = np.asarray(
                th["coarse_valid"][
                    start:stop
                ],
                dtype=bool,
            )

            shared = (
                xv
                & qv[None, :, :]
            )

            n_shared = shared.sum(
                axis=(1, 2)
            )

            diff2 = (
                x
                - q[None, :, :, :]
            ) ** 2

            numerator = (
                diff2
                * shared[:, None, :, :]
            ).sum(
                axis=(1, 2, 3)
            )

            denominator = (
                n_shared
                * q.shape[0]
            )

            ok = denominator > 0

            d = np.full(
                stop - start,
                np.nan,
                dtype=np.float32,
            )

            d[ok] = np.sqrt(
                numerator[ok]
                / denominator[ok]
            )

            dist[
                start:stop
            ] = d

    return pd.DataFrame(
        {
            "row": np.arange(
                n,
                dtype=np.int32,
            ),
            "center_lat": lat,
            "center_lon": lon,
            "climate_distance": dist,
        }
    )


def thin_points(
    df,
    max_points,
):
    """
    Geographic thinning while preserving low-distance cells preferentially.
    """
    d = df[
        np.isfinite(
            df["climate_distance"]
        )
    ].copy()

    if len(d) <= max_points:
        return d

    factor = int(
        np.ceil(
            np.sqrt(
                len(d)
                / max_points
            )
        )
    )

    latbin = np.floor(
        (d["center_lat"] + 90)
        * 4
        / factor
    ).astype(np.int32)

    lonbin = np.floor(
        (d["center_lon"] + 180)
        * 4
        / factor
    ).astype(np.int32)

    d["_bin"] = (
        latbin.astype(str)
        + ":"
        + lonbin.astype(str)
    )

    d = (
        d.sort_values(
            "climate_distance"
        )
        .drop_duplicates("_bin")
        .drop(columns="_bin")
    )

    if len(d) > max_points:
        step = int(
            np.ceil(
                len(d)
                / max_points
            )
        )

        d = d.iloc[
            ::step
        ]

    return d.head(
        max_points
    ).copy()


# ============================================================
# Overlay
# ============================================================

def overlay_html(r):
    rows = []

    for x in r.head(
        10
    ).itertuples(
        index=False
    ):

        place = ", ".join(
            [
                p
                for p in [
                    str(x.nearest_city),
                    str(x.admin1),
                    str(x.country),
                ]
                if p
                and p != "nan"
            ]
        )

        rows.append(
            "<tr>"
            f"<td>{int(x.rank)}</td>"
            f"<td>{html.escape(place)}</td>"
            f"<td>{x.climate_distance:.3f}</td>"
            f"<td>{x.geographic_distance_km:,.0f} km</td>"
            "</tr>"
        )

    return (
        '<details class="analog-overlay" open>'
        "<summary>Top 10 climate analogs — click to minimize</summary>"
        '<div class="analog-table-wrap">'
        "<table>"
        "<thead><tr>"
        "<th>#</th>"
        "<th>Location</th>"
        "<th>Climate d.</th>"
        "<th>Distance</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody>"
        "</table>"
        "</div>"
        "</details>"
    )


# ============================================================
# Session state
# ============================================================

for key, value in {
    "entered_app": False,
    "selected_lat": 51.05,
    "selected_lon": -114.07,
    "selected_place": "",
    "query_name": "",
    "selected_analog_rank": 1,
}.items():

    st.session_state.setdefault(
        key,
        value,
    )


def simple_query_name(submitted_query: str) -> str:
    """Use the user's place name rather than the snapped climate-cell place."""
    name = submitted_query.strip().split(",")[0].strip()
    return name or submitted_query.strip()


def analog_place_text(row) -> str:
    parts = [
        str(row.nearest_city),
        str(row.admin1),
        str(row.country),
    ]
    return ", ".join(p for p in parts if p and p != "nan")


def clicked_place_name(lat: float, lon: float) -> str:
    """
    Resolve a clicked climate cell to the same nearest populated-place
    convention used by the climate-search backend.
    """
    try:
        import reverse_geocoder as rg
        import pycountry

        hit = rg.search((float(lat), float(lon)), mode=1)[0]
        city = str(hit.get("name", "")).strip()
        admin1 = str(hit.get("admin1", "")).strip()
        cc = str(hit.get("cc", "")).strip()

        country = cc
        if cc:
            c = pycountry.countries.get(alpha_2=cc)
            if c is not None:
                country = c.name

        parts = [x for x in [city, admin1, country] if x]
        if parts:
            return ", ".join(parts)
    except Exception:
        pass

    return f"{float(lat):.3f}, {float(lon):.3f}"


# ============================================================
# Landing page
# ============================================================

if not st.session_state.entered_app:

    st.markdown(
        """
        <div class="landing-wrap">
          <div class="landing-title">
            Climate Analog Explorer
          </div>
          <div class="landing-copy">
            Search for a place to see where its climate is headed.
            The explorer finds places around the world whose climate today
            most closely resembles the climate projected for your selected
            location in the future. You can compare the closest matches,
            explore climate differences, and see how unusual the projected
            climate may be.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns(
        [1, 1.35, 1]
    )

    with center:

        with st.form(
            "landing-search"
        ):

            query = st.text_input(
                "Search for a city or place",
                value="New York, New York, USA",
                placeholder="e.g. New York, New York, USA",
                label_visibility="collapsed",
            )

            go = (
                st.form_submit_button(
                    "Explore climate analogs",
                    type="primary",
                    width="stretch",
                )
            )

        if go:

            if not query.strip():
                st.warning(
                    "Enter a city or place name."
                )
                st.stop()

            with st.spinner(
                "Finding location…"
            ):
                match = geocode_place(
                    query
                )

            if match is None:
                st.error(
                    "No matching place was found."
                )
                st.stop()

            st.session_state.selected_lat = (
                match["lat"]
            )

            st.session_state.selected_lon = (
                match["lon"]
            )

            st.session_state.selected_place = (
                match["display_name"]
            )
            st.session_state.query_name = simple_query_name(query)

            st.session_state.selected_analog_rank = 1
            st.session_state.entered_app = True

            st.rerun()

    st.stop()


# ============================================================
# Sidebar controls
# ============================================================

with st.sidebar:

    st.markdown(
        "## Climate Analog Explorer"
    )

    main_tab, advanced_tab = (
        st.tabs(
            [
                "Main",
                "Advanced",
            ]
        )
    )

    with main_tab:

        with st.form(
            "sidebar-search"
        ):

            query = st.text_input(
                "Search location",
                value=(
                    st.session_state.selected_place
                ),
                placeholder=(
                    "City, region, or country"
                ),
            )

            search = (
                st.form_submit_button(
                    "Search",
                    type="primary",
                    width="stretch",
                )
            )

        if search:

            with st.spinner(
                "Finding location…"
            ):
                match = geocode_place(
                    query
                )

            if match is None:

                st.error(
                    "No matching place was found."
                )

            else:

                st.session_state.selected_lat = (
                    match["lat"]
                )

                st.session_state.selected_lon = (
                    match["lon"]
                )

                st.session_state.selected_place = (
                    match["display_name"]
                )
                st.session_state.query_name = simple_query_name(query)

                st.session_state.selected_analog_rank = 1

                st.rerun()

        climate_view = (
            st.segmented_control(
                "Climate period",
                [
                    "Present",
                    "Future",
                ],
                default="Future",
                selection_mode="single",
            )
        )

        if climate_view == "Future":

            future_period = (
                st.segmented_control(
                    "Future period",
                    [
                        "Mid-century",
                        "Late-century",
                    ],
                    default="Mid-century",
                    selection_mode="single",
                )
            )

            query_period = (
                "Mid-century (2041–2060)"
                if future_period
                == "Mid-century"
                else
                "Late-century (2081–2100)"
            )

            target_period = HIST

        else:

            query_period = HIST
            target_period = HIST

    with advanced_tab:

        k = st.slider(
            "Number of analogs",
            1,
            MAX_K,
            10,
        )

        st.markdown(
            "#### Spatial constraints"
        )

        geo = st.select_slider(
            "Minimum distance from selected location",
            options=[
                0,
                250,
                500,
                1000,
                2500,
                5000,
            ],
            value=1000,
            format_func=lambda x: (
                "None"
                if x == 0
                else f"{x:,} km"
            ),
        )

        sep = st.select_slider(
            "Minimum separation between analogs",
            options=[
                0,
                50,
                100,
                150,
                250,
                500,
            ],
            value=(
                DEFAULT_ANALOG_SEPARATION_KM
            ),
            format_func=lambda x: (
                "None"
                if x == 0
                else f"{x:,} km"
            ),
        )

        st.markdown(
            "#### Globe"
        )

        globe_zoom = st.slider(
            "Initial zoom",
            0.0,
            5.0,
            1.0,
            0.25,
        )

        max_points = (
            st.select_slider(
                "Climate-distance dots",
                options=[
                    10000,
                    20000,
                    40000,
                    60000,
                ],
                value=40000,
                format_func=lambda x: (
                    f"{x:,}"
                ),
            )
        )

        show_distance = st.toggle(
            "Climate-distance dots",
            True,
        )

        show_arcs = st.toggle(
            "Top-10 connection arcs",
            True,
        )

        if st.button(
            "Return to landing page",
            width="stretch",
        ):
            st.session_state.entered_app = False
            st.rerun()


# ============================================================
# Search
# ============================================================

with st.spinner(
    "Searching climate analogs…"
):

    with engine_for(
        query_period,
        target_period,
    ) as engine:

        payload = (
            engine.search_latlon(
                st.session_state.selected_lat,
                st.session_state.selected_lon,
                n_candidates=(
                    DEFAULT_N_CANDIDATES
                ),
                n_results=k,
                geo_exclusion_km=geo,
                min_analog_separation_km=sep,
                adaptive_candidates=True,
                max_candidates=(
                    MAX_ADAPTIVE_CANDIDATES
                ),
                include_places=True,
                include_climate_stats=True,
            )
        )


r = result_df(
    payload
)

if r.empty:
    st.warning(
        "No climate analogs returned."
    )
    st.stop()


q = payload[
    "query"
]


# ============================================================
# IMPORTANT: comparison always uses analog #1
# ============================================================

best = r.iloc[0]

try:

    query_bio, analog_bio = (
        load_bio_pair(
            str(
                PERIODS[
                    query_period
                ]["database"]
            ),
            str(
                PERIODS[
                    target_period
                ]["database"]
            ),
            int(
                q["row"]
            ),
            int(
                best["row"]
            ),
        )
    )

    comparison = (
        comparison_from_bio_vectors(
            query_bio,
            analog_bio,
        )
    )

except Exception as exc:

    comparison = pd.DataFrame()

    comparison_error = str(
        exc
    )


# ============================================================
# Sidebar analytical panels
# ============================================================

with st.sidebar:

    with main_tab:

        query_display_name = (
            st.session_state.query_name
            or st.session_state.selected_place.split(",")[0]
            or "Selected location"
        )

        st.caption(
            f"{query_display_name} · "
            f"{q['lat']:.3f}°, "
            f"{q['lon']:.3f}°"
        )

        best_place = analog_place_text(best)
        query_display_name = (
            st.session_state.query_name
            or st.session_state.selected_place.split(",")[0]
            or "The selected location"
        )

        if climate_view == "Future":
            period_phrase = (
                "by mid-century (2041–2060)"
                if query_period == "Mid-century (2041–2060)"
                else "by late century (2081–2100)"
            )
            st.markdown(
                f"**The climate of {query_display_name} {period_phrase} "
                f"most closely resembles the current climate of {best_place} "
                f"in the multi-model SSP3-7.0 ensemble.**"
            )
        else:
            st.markdown(
                f"**The current climate of {query_display_name} most closely "
                f"resembles the current climate of {best_place}.**"
            )

        st.caption(
            f"Analog #1 — {best_place}"
        )

        st.markdown(
            "#### Climate difference"
        )

        if len(
            comparison
        ):

            st.plotly_chart(
                sidebar_climate_difference_plot(
                    comparison
                ),
                width="stretch",
                config={
                    "displayModeBar": False
                },
            )

        else:

            st.error(
                "BIOCLIM comparison could "
                "not be loaded."
            )

            if "comparison_error" in locals():
                st.caption(
                    comparison_error
                )

        st.markdown(
            "#### Climate novelty"
        )

        novelty_df, novelty_error = load_novelty(
            query_period
        )

        selected_novelty = novelty_value(
            novelty_df,
            q["row"],
        )

        if (
            not novelty_df.empty
            and selected_novelty
            is not None
        ):

            selected_percentile = novelty_percentile(
                novelty_df,
                selected_novelty,
            )

            st.plotly_chart(
                novelty_distribution_plot(
                    novelty_df,
                    selected_novelty,
                ),
                width="stretch",
                config={
                    "displayModeBar": False
                },
            )

            st.markdown(
                f"**Novelty percentile: {selected_percentile:.1f}th**  "
                f"· Exact climate distance: **{selected_novelty:.3f}**"
            )
            st.caption(
                "Higher values indicate climates with fewer close analogs "
                "in the comparison period."
            )

        else:
            if novelty_error:
                st.caption(novelty_error)
            else:
                st.caption(
                    f"Novelty file loaded, but no value was found for row {int(q['row'])}."
                )

        with st.expander(
            "Selected location vs. "
            "Analog #1 climate comparison"
        ):

            if len(
                comparison
            ):

                display_comparison = comparison.copy()
                for col in [
                    "Selected",
                    "Analog",
                    "Difference",
                    "Standardized difference",
                ]:
                    display_comparison[col] = display_comparison[col].map(
                        lambda v: f"{float(v):.3f}"
                    )

                st.dataframe(
                    display_comparison,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Variable": st.column_config.TextColumn("BIOCLIM"),
                        "Description": st.column_config.TextColumn("Description"),
                    },
                )

            else:

                st.caption(
                    "Climate comparison "
                    "unavailable."
                )


# ============================================================
# Exact distance field
# ============================================================

points = pd.DataFrame()

if show_distance:

    with st.spinner(
        "Computing exact global "
        "climate-distance field…"
    ):

        points = (
            exact_global_distance_field(
                str(
                    PERIODS[
                        query_period
                    ]["database"]
                ),
                str(
                    PERIODS[
                        target_period
                    ]["database"]
                ),
                int(
                    q["row"]
                ),
            )
        )

    points = thin_points(
        points,
        max_points,
    )


# ============================================================
# Globe
# ============================================================

# Round all values exposed to deck.gl hover to three decimal places.
r["climate_distance"] = r["climate_distance"].astype(float).round(3)
r["geographic_distance_km"] = r["geographic_distance_km"].astype(float).round(0).astype(int)
r["center_lat"] = r["center_lat"].astype(float).round(3)
r["center_lon"] = r["center_lon"].astype(float).round(3)

if len(points):
    points["climate_distance"] = points["climate_distance"].astype(float).round(3)
    points["center_lat"] = points["center_lat"].astype(float).round(3)
    points["center_lon"] = points["center_lon"].astype(float).round(3)

selected = pd.Series(
    {
        "row": q["row"],
        "chip_id": q["chip_id"],
        "center_lat": q["lat"],
        "center_lon": q["lon"],
    }
)

st.markdown(
    (
        '<div class="globe-title">'
        f"<b>{html.escape(st.session_state.query_name or st.session_state.selected_place.split(',')[0] or 'Selected location')}</b>"
        "<br>"
        f"<span>{html.escape(query_period)} "
        f"→ {html.escape(target_period)}</span>"
        "</div>"
    ),
    unsafe_allow_html=True,
)


deck = build_globe(
    selected_row=selected,
    analogs=r,
    distance_points=points,
    selected_analog_rank=(
        st.session_state[
            "selected_analog_rank"
        ]
    ),
    show_distance=show_distance,
    show_arcs=show_arcs,
    map_zoom=globe_zoom,
)


event = st.pydeck_chart(
    deck,
    key=(
        f"globe-"
        f"{query_period}-"
        f"{target_period}-"
        f"{q['row']}"
    ),
    on_select="rerun",
    selection_mode="single-object",
    width="stretch",
    height=920,
)


picked = objects(
    event
)


cc = clicked_cell(
    picked
)

if (
    cc
    and (
        abs(
            cc[0]
            - st.session_state.selected_lat
        )
        > 1e-8
        or abs(
            cc[1]
            - st.session_state.selected_lon
        )
        > 1e-8
    )
):

    st.session_state.selected_lat = (
        cc[0]
    )

    st.session_state.selected_lon = (
        cc[1]
    )

    st.session_state.selected_place = ""
    st.session_state.query_name = clicked_place_name(cc[0], cc[1])
    st.session_state.selected_analog_rank = 1

    st.rerun()


clicked_analog = clicked_rank(
    picked
)

if (
    clicked_analog
    and clicked_analog
    != st.session_state[
        "selected_analog_rank"
    ]
):

    # Marker highlighting can change, but the sidebar climate
    # comparison intentionally remains fixed to analog #1.
    st.session_state[
        "selected_analog_rank"
    ] = clicked_analog

    st.rerun()


# This is the final element in the main panel.
st.markdown(
    overlay_html(r),
    unsafe_allow_html=True,
)
