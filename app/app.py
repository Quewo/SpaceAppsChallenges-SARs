"""Modern interactive Streamlit web application for Dancing with the SARs (Team W Gee)."""
from __future__ import annotations

import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

# Ensure repo root is importable when launched via `streamlit run app/app.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src.classifier import list_models
from src.config import LAYER_MODES, RESOLUTION_PRESETS_M, SURFACE_CLASSES
from src.database import (
    clear_classifications,
    get_database_stats,
    import_processed_run_to_db,
    init_database,
    list_regions,
)
from src.earthdata_pipeline import (
    OPERA_ARCHIVE_START,
    EarthdataFetchConfig,
    fetch_and_export_nasa_run,
    get_source_date_limits,
    has_earthdata_token,
    list_processed_runs,
    load_dotenv_into_environ,
    probe_earthdata_source_dates,
)
from src.i18n import LAYER_DE, localized_class_name, tr
from src.map_component import render_interactive_sar_map
from src.pipeline import (
    TimeframeBundle,
    compute_change_summary,
    compute_transition_rows,
    encode_bundle_for_frontend,
    run_aoi_timeframe_pipeline,
)

st.set_page_config(
    page_title="Dancing with the SARs · W Gee NISAR Studio",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STUDIO_CSS = """
<style>
/* --- Global Dark Observatory Canvas --- */
.stApp {
    background: radial-gradient(circle at 18% 0%, rgba(14, 165, 233, 0.08) 0%, #060913 42%, #050711 100%);
    color: #F8FAFC;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1.0rem !important;
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
    max-width: 100% !important;
}
footer, [data-testid="stDeployButton"] {
    display: none !important;
}

/* Match sidebar toggle button to dark glass theme */
[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] {
    background: rgba(11, 17, 32, 0.85) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}

/* --- Hero Header Banner --- */
.sar-hero-card {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 10px 16px;
    margin-bottom: 0.45rem;
    border-radius: 14px;
    background: linear-gradient(125deg, rgba(14, 165, 233, 0.13) 0%, rgba(15, 23, 42, 0.92) 48%, rgba(20, 184, 166, 0.10) 100%);
    border: 1px solid rgba(56, 189, 248, 0.24);
    box-shadow: 0 10px 30px rgba(2, 6, 23, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}
.sar-hero-left {
    display: flex;
    align-items: center;
    gap: 13px;
    min-width: 0;
}
.sar-hero-emblem {
    width: 40px;
    height: 40px;
    border-radius: 11px;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.24), rgba(20, 184, 166, 0.20));
    border: 1px solid rgba(56, 189, 248, 0.42);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 0 18px rgba(14, 165, 233, 0.22);
}
.sar-hero-titles {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
}
.sar-hero-row1 {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 9px;
}
.sar-hero-name {
    font-size: 1.28rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    background: linear-gradient(92deg, #FFFFFF 0%, #7DD3FC 55%, #5EEAD4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sar-brand-badge {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.34);
    color: #7DD3FC;
    white-space: nowrap;
}
.sar-hero-team {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
    font-size: 0.76rem;
    color: #94A3B8;
}
.sar-team-pill {
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 1px 8px;
    border-radius: 6px;
    background: rgba(20, 184, 166, 0.16);
    border: 1px solid rgba(45, 212, 191, 0.34);
    color: #5EEAD4;
}
.sar-hero-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 3px;
    font-size: 0.75rem;
    color: #94A3B8;
}
.sar-hero-meta strong {
    color: #E2E8F0;
    font-weight: 600;
}

/* Compact EN / DE Pill Buttons beside Hero Banner */
div[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"],
div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
    min-height: 34px !important;
    padding: 0.25rem 0.65rem !important;
    border-radius: 10px !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
}

/* Override harsh default Streamlit primary red to match SAR Cyan Accent */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0EA5E9, #38BDF8) !important;
    border: none !important;
    color: #07111F !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(14, 165, 233, 0.30) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #38BDF8, #7DD3FC) !important;
    color: #060913 !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(15, 23, 42, 0.78) !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
    color: #CBD5E1 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: rgba(56, 189, 248, 0.55) !important;
    color: #FFFFFF !important;
    background: rgba(30, 41, 59, 0.88) !important;
}

/* --- Aesthetic Overhaul for Hidden Sidebar --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1124 0%, #060913 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.16) !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.0rem !important;
}
.sar-sb-brand {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 11px;
    margin-bottom: 10px;
    border-radius: 11px;
    background: rgba(14, 165, 233, 0.10);
    border: 1px solid rgba(56, 189, 248, 0.24);
    font-size: 0.76rem;
    font-weight: 700;
    color: #7DD3FC;
    letter-spacing: 0.02em;
}
.sar-sb-section {
    margin: 12px 0 6px 0;
    padding-top: 10px;
    border-top: 1px solid rgba(148, 163, 184, 0.14);
    font-size: 0.69rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #38BDF8;
}
.sar-sb-info-box {
    padding: 8px 10px;
    margin: 6px 0 10px 0;
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.18);
    font-size: 0.73rem;
    line-height: 1.45;
    color: #94A3B8;
}
section[data-testid="stSidebar"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: rgba(15, 23, 42, 0.86) !important;
    border-color: rgba(148, 163, 184, 0.22) !important;
    border-radius: 10px !important;
    color: #F8FAFC !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.55) !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.70);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 10px;
    padding: 8px 10px;
}

/* --- Unified Dark-Glass Analytics Card Container --- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.sar-analytics-anchor) {
    margin-top: 0.75rem;
    padding: 1rem 1.25rem !important;
    border-radius: 16px !important;
    background: linear-gradient(180deg, rgba(11, 18, 36, 0.94) 0%, rgba(8, 13, 26, 0.96) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.20) !important;
    box-shadow: 0 18px 44px rgba(2, 6, 23, 0.55);
}
.sar-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 8px 0 14px 0;
}
@media (max-width: 900px) {
    .sar-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
.sar-kpi-tile {
    padding: 11px 14px;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-top: 2px solid var(--kpi-accent, #38BDF8);
}
.sar-kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.sar-kpi-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-top: 3px;
    line-height: 1.15;
}
.sar-kpi-sub {
    font-size: 0.72rem;
    color: #7DD3FC;
    margin-top: 3px;
}
.sar-trans-panel {
    background: rgba(15, 23, 42, 0.58);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    padding: 12px 14px;
}
.sar-trans-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 10px;
}
.sar-trans-row {
    margin-bottom: 9px;
}
.sar-trans-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #CBD5E1;
    margin-bottom: 4px;
}
.sar-trans-val {
    font-weight: 700;
    color: #38BDF8;
}
.sar-trans-track {
    width: 100%;
    height: 5px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.14);
    overflow: hidden;
}
.sar-trans-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0EA5E9, #14B8A6);
}
.sar-insight-foot {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    font-size: 0.72rem;
    color: #94A3B8;
}
</style>
"""

st.markdown(STUDIO_CSS, unsafe_allow_html=True)


def _init_session_state(default_region) -> None:
    st.session_state.setdefault("lang", "en")
    st.session_state.setdefault(
        "aoi_bounds",
        (default_region.south, default_region.west, default_region.north, default_region.east),
    )
    st.session_state.setdefault("aoi_label", default_region.name)
    st.session_state.setdefault("active_region_id", None)
    st.session_state.setdefault("resolution_m", 30.0)
    st.session_state.setdefault("selected_elements", [1, 2, 3])
    st.session_state.setdefault("view_mode", "elements")
    st.session_state.setdefault("sar_band", "False-Color Dual-Pol (HH/HV)")
    st.session_state.setdefault("opacity", 0.78)
    st.session_state.setdefault("model_id", "sar_mlp_v1")
    st.session_state.setdefault("start_date", None)
    st.session_state.setdefault("end_date", None)
    st.session_state.setdefault("show_graph", True)
    st.session_state.setdefault("chart_metric_mode", "area")
    st.session_state.setdefault("force_recompute_once", False)
    st.session_state.setdefault("nasa_status_msg", None)
    st.session_state.setdefault("nasa_probe_summary", None)
    st.session_state.setdefault("nasa_token_override", "")
    st.session_state.setdefault("nasa_run_name", "custom_aoi_run")
    st.session_state.setdefault("nasa_start_date", OPERA_ARCHIVE_START)
    st.session_state.setdefault("nasa_end_date", max(date.today().isoformat(), "2026-11-14"))
    st.session_state.setdefault("nasa_fetch_all_dates", False)
    st.session_state.setdefault("nasa_max_timesteps", 24)
    st.session_state.setdefault("nasa_min_valid_pct", 80.0)
    st.session_state.setdefault("inc_nasadem", True)
    st.session_state.setdefault("inc_opera_rtc", True)
    st.session_state.setdefault("inc_nisar_gcov", True)
    st.session_state.setdefault("inc_nisar_gunw", True)
    st.session_state.setdefault("inc_opera_dswx", False)
    st.session_state.setdefault("last_aoi_seq", None)
    st.session_state.setdefault("last_elem_seq", None)
    st.session_state.setdefault("last_tf_seq", None)
    st.session_state.setdefault("last_graph_seq", None)
    st.session_state.setdefault("last_mode_seq", None)
    st.session_state.setdefault("last_res_seq", None)
    st.session_state.setdefault("last_fetch_seq", None)


def _execute_nasa_fetch_for_aoi(
    south: float,
    west: float,
    north: float,
    east: float,
    resolution_m: float,
    start_date: str,
    end_date: str,
    run_name: str,
) -> None:
    """Download & align real NASA Earthdata cubes for the active square into SQLite."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", (run_name or "").strip()).strip("_") or "aoi_nasa_run"
    max_steps = None if bool(st.session_state.nasa_fetch_all_dates) else int(st.session_state.nasa_max_timesteps)
    cfg = EarthdataFetchConfig(
        run_name=safe_name,
        bbox_wgs84=(float(west), float(south), float(east), float(north)),
        start_date=str(start_date),
        end_date=str(end_date),
        target_res_m=float(resolution_m),
        min_valid_pct=float(st.session_state.nasa_min_valid_pct),
        max_timesteps_per_collection=max_steps,
        include_nisar_gcov=bool(st.session_state.inc_nisar_gcov),
        include_nisar_gunw=bool(st.session_state.inc_nisar_gunw),
        include_opera_rtc_s1=bool(st.session_state.inc_opera_rtc),
        include_nasadem=bool(st.session_state.inc_nasadem),
        include_opera_dswx_s1=bool(st.session_state.inc_opera_dswx),
    )
    token = st.session_state.nasa_token_override.strip() or None
    try:
        with st.spinner("Streaming NASA Earthdata slices (NASADEM / OPERA / NISAR)..."):
            res = fetch_and_export_nasa_run(cfg, token=token)
            rec = import_processed_run_to_db(res["out_dir"], region_name=f"NASA Earthdata · {safe_name}")
        st.session_state.active_region_id = rec.region_id
        st.session_state.aoi_bounds = (rec.south, rec.west, rec.north, rec.east)
        st.session_state.aoi_label = rec.name
        st.session_state.resolution_m = rec.resolution_m
        st.session_state.start_date = None
        st.session_state.end_date = None
        st.session_state.nasa_status_msg = (
            "success",
            f"Loaded real NASA Earthdata run '{rec.name}' ({len(res['files'])} NetCDF cubes @ {rec.resolution_m:.0f}m).",
        )
    except Exception as exc:
        st.session_state.nasa_status_msg = ("warning", f"NASA Earthdata fetch: {exc}")


def _build_timeseries_chart(
    bundle: TimeframeBundle,
    selected_class_ids: list[int],
    lang: str = "en",
    metric_mode: str = "area",
) -> alt.Chart:
    """Build a crisp, glitch-free temporal area-trend chart using true Date:T axis and clipped marks."""
    total_box_km2 = max(
        float(bundle.frames[0].class_mask.size * bundle.region.pixel_area_km2) if bundle.frames else 1.0,
        1e-4,
    )
    rows = []
    seen = set()
    for f in sorted(bundle.frames, key=lambda fr: fr.scene.acq_date):
        for cid in selected_class_ids:
            key = (f.scene.acq_date, cid)
            if key in seen:
                continue
            seen.add(key)
            cls_info = SURFACE_CLASSES[cid]
            elem_name = localized_class_name(cid, cls_info.short_label, lang)
            area_val = float(f.areas_km2.get(cid, 0.0))
            val = round((area_val / total_box_km2) * 100.0, 2) if metric_mode == "share" else round(area_val, 2)
            rows.append(
                {
                    "Date": pd.to_datetime(f.scene.acq_date),
                    "DateStr": f.scene.acq_date,
                    "Element": elem_name,
                    "Value": val,
                }
            )

    df = pd.DataFrame(rows).sort_values(["Element", "Date"]).reset_index(drop=True)
    domain = [localized_class_name(cid, SURFACE_CLASSES[cid].short_label, lang) for cid in selected_class_ids]
    range_colors = [SURFACE_CLASSES[cid].color_hex for cid in selected_class_ids]
    y_title = tr(lang, "chart_axis_share") if metric_mode == "share" else tr(lang, "chart_axis_area")
    x_title = tr(lang, "chart_axis_date")

    span_days = (df["Date"].max() - df["Date"].min()).days if len(df) >= 2 else 0
    date_fmt = "%b %Y" if span_days > 450 else "%Y-%m-%d"

    hover = alt.selection_point(
        fields=["Date"],
        nearest=True,
        on="pointerover",
        empty=False,
    )

    base = alt.Chart(df).encode(
        x=alt.X(
            "Date:T",
            title=x_title,
            axis=alt.Axis(format=date_fmt, labelAngle=-15, tickCount=8),
        ),
        color=alt.Color(
            "Element:N",
            scale=alt.Scale(domain=domain, range=range_colors),
            legend=alt.Legend(orient="top", title=None, labelFontSize=12, symbolSize=110),
        ),
    )

    areas = base.mark_area(
        interpolate="monotone",
        fillOpacity=0.11,
        strokeWidth=0,
        clip=True,
    ).encode(
        y=alt.Y("Value:Q", title=y_title, stack=None, scale=alt.Scale(zero=True, nice=True, padding=8)),
    )

    lines = base.mark_line(
        interpolate="monotone",
        strokeWidth=2.4,
        strokeJoin="round",
        strokeCap="round",
        clip=True,
    ).encode(
        y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(zero=True, nice=True, padding=8)),
    )

    points = base.mark_circle(
        size=52,
        stroke="#0B1120",
        strokeWidth=1.4,
        clip=True,
    ).encode(
        y=alt.Y("Value:Q"),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.82)),
        tooltip=[
            alt.Tooltip("DateStr:N", title=x_title),
            alt.Tooltip("Element:N", title="Element"),
            alt.Tooltip("Value:Q", title=y_title, format=".2f"),
        ],
    )

    rule = (
        alt.Chart(df)
        .mark_rule(color="rgba(56, 189, 248, 0.45)", strokeWidth=1.2, strokeDash=[4, 3])
        .encode(x="Date:T")
        .transform_filter(hover)
        .add_params(hover)
    )

    chart = (
        alt.layer(areas, lines, points, rule)
        .properties(height=285, background="transparent")
        .configure_axis(
            labelColor="#94A3B8",
            titleColor="#CBD5E1",
            titleFontSize=11.5,
            labelFontSize=11,
            gridColor="rgba(148, 163, 184, 0.12)",
            domainColor="rgba(148, 163, 184, 0.22)",
        )
        .configure_legend(labelColor="#E2E8F0", padding=6)
        .configure_view(strokeWidth=0)
    )
    return chart


def main() -> None:
    load_dotenv_into_environ()
    init_database()
    preset_regions = list_regions(presets_only=True)
    default_region = next(
        (r for r in preset_regions if r.region_id == "elbe_flood"),
        preset_regions[0],
    )
    _init_session_state(default_region)
    if st.session_state.nasa_token_override.strip():
        os.environ["EARTHDATA_TOKEN"] = st.session_state.nasa_token_override.strip()

    lang = st.session_state.lang
    models = list_models()
    model_by_id = {m.model_id: m for m in models}
    processed_runs = list_processed_runs()
    token_ready = has_earthdata_token()

    src_min_str, src_max_str, _ = get_source_date_limits(
        include_opera_rtc_s1=bool(st.session_state.inc_opera_rtc),
        include_opera_dswx_s1=bool(st.session_state.inc_opera_dswx),
        include_nisar_gcov=bool(st.session_state.inc_nisar_gcov),
        include_nisar_gunw=bool(st.session_state.inc_nisar_gunw),
    )
    src_min_d = datetime.strptime(src_min_str, "%Y-%m-%d").date()
    src_max_d = datetime.strptime(src_max_str, "%Y-%m-%d").date()

    # --- Sleek Dark-Glass Collapsible Sidebar (Collapsed by Default) ---
    with st.sidebar:
        st.markdown(f"<div class='sar-sb-brand'><span>🛰️ {tr(lang, 'sb_badge')}</span></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='sar-sb-section'>{tr(lang, 'sb_collector')}</div>", unsafe_allow_html=True)
        if token_ready:
            st.success(tr(lang, "sb_token_ok"))
        else:
            st.info(tr(lang, "sb_token_missing"))

        token_input = st.text_input(
            tr(lang, "sb_token_label"),
            value=st.session_state.nasa_token_override,
            type="password",
            key="earthdata_token_input",
        )
        if token_input != st.session_state.nasa_token_override:
            st.session_state.nasa_token_override = token_input
            if token_input.strip():
                os.environ["EARTHDATA_TOKEN"] = token_input.strip()
            token_ready = has_earthdata_token()

        res_options = list(RESOLUTION_PRESETS_M)
        cur_res = float(st.session_state.resolution_m)
        if cur_res not in res_options:
            res_options = sorted(set(res_options + [cur_res]))
        chosen_res = st.selectbox(
            tr(lang, "sb_res_label"),
            options=res_options,
            index=res_options.index(cur_res),
            format_func=lambda r: f"{int(round(r))} m",
            key="resolution_selector",
        )
        if abs(float(chosen_res) - float(st.session_state.resolution_m)) > 0.1:
            st.session_state.resolution_m = float(chosen_res)
            st.session_state.active_region_id = None
            st.rerun()

        st.session_state.nasa_run_name = st.text_input(
            tr(lang, "sb_run_name"),
            value=st.session_state.nasa_run_name,
        )

        # Full source-bounded Date Inputs (OPERA S1 from 2016-01-01 .. Present; NISAR 2025-01-01 .. Present)
        cur_nasa_s = max(src_min_d, min(src_max_d, datetime.strptime(st.session_state.nasa_start_date, "%Y-%m-%d").date()))
        cur_nasa_e = max(src_min_d, min(src_max_d, datetime.strptime(st.session_state.nasa_end_date, "%Y-%m-%d").date()))
        dc1, dc2 = st.columns(2)
        picked_s = dc1.date_input(
            tr(lang, "sb_start_date"),
            value=cur_nasa_s,
            min_value=date(2016, 1, 1),
            max_value=src_max_d,
            key="nasa_start_picker",
        )
        picked_e = dc2.date_input(
            tr(lang, "sb_end_date"),
            value=max(picked_s, cur_nasa_e),
            min_value=picked_s,
            max_value=src_max_d,
            key="nasa_end_picker",
        )
        st.session_state.nasa_start_date = picked_s.isoformat()
        st.session_state.nasa_end_date = picked_e.isoformat()

        st.markdown(f"<div class='sar-sb-info-box'>{tr(lang, 'sb_src_hint')}</div>", unsafe_allow_html=True)

        st.session_state.nasa_fetch_all_dates = st.checkbox(
            tr(lang, "sb_all_dates"),
            value=bool(st.session_state.nasa_fetch_all_dates),
        )
        kc1, kc2 = st.columns(2)
        st.session_state.nasa_max_timesteps = int(
            kc1.number_input(
                tr(lang, "sb_max_steps"),
                min_value=5,
                max_value=365,
                value=int(st.session_state.nasa_max_timesteps),
                disabled=bool(st.session_state.nasa_fetch_all_dates),
            )
        )
        st.session_state.nasa_min_valid_pct = float(
            kc2.number_input(
                tr(lang, "sb_min_valid"),
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.nasa_min_valid_pct),
                step=5.0,
            )
        )

        with st.expander(tr(lang, "sb_collections")):
            st.session_state.inc_nasadem = st.checkbox("NASADEM Elevation (static DEM)", value=st.session_state.inc_nasadem)
            st.session_state.inc_opera_rtc = st.checkbox("OPERA RTC-S1 (VV / VH dB, 2016+)", value=st.session_state.inc_opera_rtc)
            st.session_state.inc_nisar_gcov = st.checkbox("NISAR GCOV (HHHH / HVHV dB, 2025+)", value=st.session_state.inc_nisar_gcov)
            st.session_state.inc_nisar_gunw = st.checkbox("NISAR GUNW (Coherence HH/VV, 2025+)", value=st.session_state.inc_nisar_gunw)
            st.session_state.inc_opera_dswx = st.checkbox("OPERA DSWx-S1 (Water Extent)", value=st.session_state.inc_opera_dswx)

        if st.button(tr(lang, "sb_probe_btn"), width="stretch"):
            s_cur, w_cur, n_cur, e_cur = st.session_state.aoi_bounds
            try:
                with st.spinner("Querying NASA CMR granule index..."):
                    probe = probe_earthdata_source_dates(
                        bbox_wgs84=(float(w_cur), float(s_cur), float(e_cur), float(n_cur)),
                        include_opera_rtc_s1=bool(st.session_state.inc_opera_rtc),
                        include_opera_dswx_s1=bool(st.session_state.inc_opera_dswx),
                        include_nisar_gcov=bool(st.session_state.inc_nisar_gcov),
                        include_nisar_gunw=bool(st.session_state.inc_nisar_gunw),
                        start_date=st.session_state.nasa_start_date,
                        end_date=st.session_state.nasa_end_date,
                        token=st.session_state.nasa_token_override.strip() or None,
                    )
                st.session_state.nasa_probe_summary = probe
            except Exception as exc:
                st.session_state.nasa_probe_summary = {"error": str(exc)}

        if isinstance(st.session_state.nasa_probe_summary, dict):
            pr = st.session_state.nasa_probe_summary
            if "error" in pr:
                st.warning(pr["error"])
            else:
                st.success(f"Available: {pr['min_date']} → {pr['max_date']}")

        if st.button(tr(lang, "sb_download_btn"), width="stretch", type="primary"):
            s_cur, w_cur, n_cur, e_cur = st.session_state.aoi_bounds
            _execute_nasa_fetch_for_aoi(
                south=s_cur,
                west=w_cur,
                north=n_cur,
                east=e_cur,
                resolution_m=st.session_state.resolution_m,
                start_date=st.session_state.nasa_start_date,
                end_date=st.session_state.nasa_end_date,
                run_name=st.session_state.nasa_run_name,
            )
            st.rerun()

        if processed_runs:
            run_map = {r.run_name: r for r in processed_runs}
            picked_run = st.selectbox(
                tr(lang, "sb_local_runs"),
                options=[r.run_name for r in processed_runs],
                format_func=lambda rn: f"{rn} ({int(round(run_map[rn].resolution_m))}m · {len(run_map[rn].dates)} dates)",
                key="processed_run_selector",
            )
            if st.button(tr(lang, "sb_load_local_btn"), width="stretch"):
                rec = import_processed_run_to_db(run_map[picked_run].run_dir)
                st.session_state.active_region_id = rec.region_id
                st.session_state.aoi_bounds = (rec.south, rec.west, rec.north, rec.east)
                st.session_state.aoi_label = rec.name
                st.session_state.resolution_m = rec.resolution_m
                st.session_state.start_date = None
                st.session_state.end_date = None
                st.session_state.nasa_status_msg = ("success", f"Loaded NetCDF run '{picked_run}'.")
                st.rerun()

        st.markdown(f"<div class='sar-sb-section'>{tr(lang, 'sb_nn_header')}</div>", unsafe_allow_html=True)
        model_id = st.selectbox(
            tr(lang, "sb_model_label"),
            options=[m.model_id for m in models],
            index=[m.model_id for m in models].index(st.session_state.model_id),
            format_func=lambda mid: model_by_id[mid].display_name,
            key="model_selector",
        )
        st.session_state.model_id = model_id
        st.caption(model_by_id[model_id].description)

        if st.button(tr(lang, "sb_rerun_btn"), width="stretch"):
            st.session_state.force_recompute_once = True
            st.rerun()

        st.markdown(f"<div class='sar-sb-section'>{tr(lang, 'sb_radar_header')}</div>", unsafe_allow_html=True)
        st.session_state.sar_band = st.selectbox(
            tr(lang, "sb_radar_band"),
            options=["False-Color Dual-Pol (HH/HV)", "HH (dB)", "HV (dB)"],
            index=["False-Color Dual-Pol (HH/HV)", "HH (dB)", "HV (dB)"].index(
                st.session_state.sar_band
            ),
        )
        st.session_state.opacity = st.slider(
            tr(lang, "sb_opacity"),
            min_value=0.20,
            max_value=1.00,
            value=float(st.session_state.opacity),
            step=0.02,
        )

        st.markdown(f"<div class='sar-sb-section'>{tr(lang, 'sb_presets_header')}</div>", unsafe_allow_html=True)
        preset_choice = st.selectbox(
            tr(lang, "sb_preset_label"),
            options=[r.region_id for r in preset_regions],
            format_func=lambda rid: next(
                f"{r.name} ({r.country})" for r in preset_regions if r.region_id == rid
            ),
            key="preset_selector",
        )
        if st.button(tr(lang, "sb_preset_btn"), width="stretch"):
            chosen = next(r for r in preset_regions if r.region_id == preset_choice)
            st.session_state.active_region_id = None
            st.session_state.aoi_bounds = (
                chosen.south,
                chosen.west,
                chosen.north,
                chosen.east,
            )
            st.session_state.aoi_label = chosen.name
            st.session_state.start_date = None
            st.session_state.end_date = None
            st.session_state.nasa_status_msg = None
            st.rerun()

        st.markdown(f"<div class='sar-sb-section'>{tr(lang, 'sb_db_header')}</div>", unsafe_allow_html=True)
        db_stats = get_database_stats()
        s1, s2 = st.columns(2)
        s1.metric(tr(lang, "sb_db_regions"), db_stats["regions"])
        s2.metric(tr(lang, "sb_db_scenes"), db_stats["sar_scenes"])
        s3, s4 = st.columns(2)
        s3.metric(tr(lang, "sb_db_preds"), db_stats["classifications"])
        s4.metric(tr(lang, "sb_db_size"), f"{db_stats['size_kb']} KB")

        if st.button(tr(lang, "sb_clear_btn"), width="stretch"):
            deleted = clear_classifications(model_id=model_id)
            st.success(f"Cleared {deleted} cached prediction(s).")
            st.rerun()

        with st.expander(tr(lang, "sb_scenes_expander")):
            st.dataframe(
                pd.DataFrame(db_stats["scenes_table"])[
                    ["region_name", "acq_date", "sensor", "cached_models"]
                ],
                width="stretch",
                hide_index=True,
            )

    # --- Execute Backend Pipeline on Active Observation Square ---
    south, west, north, east = st.session_state.aoi_bounds
    force_recompute = bool(st.session_state.pop("force_recompute_once", False))

    bundle, avail_dates, start_date, end_date = run_aoi_timeframe_pipeline(
        south=south,
        west=west,
        north=north,
        east=east,
        start_date=st.session_state.start_date,
        end_date=st.session_state.end_date,
        model_id=st.session_state.model_id,
        resolution_m=float(st.session_state.resolution_m),
        region_id=st.session_state.active_region_id,
        label=st.session_state.aoi_label,
        force_recompute=force_recompute,
        extend_to_source_history=True,
    )
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    # --- Hero Header Strip ("Dancing with the SARs" + Team W Gee + Team Members + EN/DE Switch) ---
    active_mode_def = LAYER_MODES.get(st.session_state.view_mode, LAYER_MODES["elements"])
    mode_label = (
        LAYER_DE[active_mode_def.mode_id]["label"]
        if (lang == "de" and active_mode_def.mode_id in LAYER_DE)
        else active_mode_def.label
    )
    src_badge = tr(lang, "nasa_source") if bundle.region.source_type == "nasa_earthdata" else tr(lang, "sim_source")

    hero_col, lang_col = st.columns([11, 1.35], vertical_alignment="center")
    hero_col.markdown(
        f"""
        <div class="sar-hero-card">
            <div class="sar-hero-left">
                <div class="sar-hero-emblem" aria-hidden="true">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 12a3 3 0 1 0-3-3"/>
                        <path d="M4.93 19.07a10 10 0 0 1 0-14.14"/>
                        <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                        <path d="M7.76 16.24a6 6 0 0 1 0-8.48"/>
                        <path d="M16.24 7.76a6 6 0 0 1 0 8.48"/>
                        <circle cx="12" cy="12" r="1.5" fill="#5EEAD4" stroke="none"/>
                    </svg>
                </div>
                <div class="sar-hero-titles">
                    <div class="sar-hero-row1">
                        <span class="sar-hero-name">Dancing with the SARs</span>
                        <span class="sar-brand-badge">{src_badge} &middot; {int(round(bundle.region.resolution_m))} m</span>
                    </div>
                    <div class="sar-hero-team">
                        <span class="sar-team-pill">{tr(lang, "team_label")}</span>
                        <span>{tr(lang, "team_members")}</span>
                    </div>
                </div>
            </div>
            <div class="sar-hero-right">
                <div class="sar-hero-meta">
                    {tr(lang, "filter_prefix")}: <strong>{mode_label}</strong> &middot;
                    {tr(lang, "model_prefix")}: <strong>{model_by_id[bundle.model_id].display_name}</strong>
                </div>
                <div>
                    {len(bundle.frames)} {tr(lang, "scenes_word")}
                    ({bundle.newly_inferred} {tr(lang, "new_word")}, {bundle.cache_hits} {tr(lang, "cached_word")} &middot; {bundle.total_inference_ms:.1f} ms)
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with lang_col:
        lb1, lb2 = st.columns(2)
        if lb1.button(
            tr(lang, "lang_en"),
            key="lang_en_btn",
            type="primary" if lang == "en" else "secondary",
            width="stretch",
        ):
            if lang != "en":
                st.session_state.lang = "en"
                st.rerun()
        if lb2.button(
            tr(lang, "lang_de"),
            key="lang_de_btn",
            type="primary" if lang == "de" else "secondary",
            width="stretch",
        ):
            if lang != "de":
                st.session_state.lang = "de"
                st.rerun()

    if st.session_state.nasa_status_msg:
        lvl, msg = st.session_state.nasa_status_msg
        if lvl == "success":
            st.success(msg)
        else:
            st.warning(msg)

    # --- Full-Viewport Interactive Map Component ---
    runs_summary = [
        {"run_name": r.run_name, "resolution_m": r.resolution_m, "dates": list(r.dates)}
        for r in processed_runs
    ]
    payload = encode_bundle_for_frontend(
        bundle=bundle,
        available_dates=avail_dates,
        selected_elements=st.session_state.selected_elements,
        view_mode=st.session_state.view_mode,
        sar_band=st.session_state.sar_band,
        opacity=st.session_state.opacity,
        show_graph=st.session_state.show_graph,
        presets=preset_regions,
        start_date=start_date,
        end_date=end_date,
        processed_runs=runs_summary,
        has_earthdata_token=token_ready,
        lang=lang,
    )
    map_result = render_interactive_sar_map(payload, key="sar_studio_map")

    # Handle bidirectional triggers from the map HUD & observation square
    needs_rerun = False

    aoi_ev = getattr(map_result, "aoi_bounds", None)
    if isinstance(aoi_ev, dict) and aoi_ev.get("seq") != st.session_state.last_aoi_seq:
        st.session_state.last_aoi_seq = aoi_ev.get("seq")
        st.session_state.active_region_id = None
        st.session_state.aoi_bounds = (
            float(aoi_ev["south"]),
            float(aoi_ev["west"]),
            float(aoi_ev["north"]),
            float(aoi_ev["east"]),
        )
        st.session_state.aoi_label = aoi_ev.get("label") or None
        st.session_state.nasa_status_msg = None
        needs_rerun = True

    elem_ev = getattr(map_result, "selected_elements", None)
    if isinstance(elem_ev, dict) and elem_ev.get("seq") != st.session_state.last_elem_seq:
        st.session_state.last_elem_seq = elem_ev.get("seq")
        elems = [int(c) for c in elem_ev.get("elements", []) if int(c) in SURFACE_CLASSES]
        if elems:
            st.session_state.selected_elements = elems
            needs_rerun = True

    tf_ev = getattr(map_result, "timeframe", None)
    if isinstance(tf_ev, dict) and tf_ev.get("seq") != st.session_state.last_tf_seq:
        st.session_state.last_tf_seq = tf_ev.get("seq")
        st.session_state.start_date = tf_ev.get("start_date")
        st.session_state.end_date = tf_ev.get("end_date")
        needs_rerun = True

    graph_ev = getattr(map_result, "toggle_graph", None)
    if isinstance(graph_ev, dict) and graph_ev.get("seq") != st.session_state.last_graph_seq:
        st.session_state.last_graph_seq = graph_ev.get("seq")
        st.session_state.show_graph = bool(graph_ev.get("show", not st.session_state.show_graph))
        needs_rerun = True

    mode_ev = getattr(map_result, "view_mode", None)
    if isinstance(mode_ev, dict) and mode_ev.get("seq") != st.session_state.last_mode_seq:
        st.session_state.last_mode_seq = mode_ev.get("seq")
        req_mode = str(mode_ev.get("mode", "elements"))
        if req_mode in LAYER_MODES and req_mode != st.session_state.view_mode:
            st.session_state.view_mode = req_mode
            needs_rerun = True

    res_ev = getattr(map_result, "resolution", None)
    if isinstance(res_ev, dict) and res_ev.get("seq") != st.session_state.last_res_seq:
        st.session_state.last_res_seq = res_ev.get("seq")
        st.session_state.resolution_m = max(10.0, float(res_ev.get("resolution_m", 30.0)))
        st.session_state.active_region_id = None
        st.session_state.nasa_status_msg = None
        needs_rerun = True

    fetch_ev = getattr(map_result, "nasa_fetch", None)
    if isinstance(fetch_ev, dict) and fetch_ev.get("seq") != st.session_state.last_fetch_seq:
        st.session_state.last_fetch_seq = fetch_ev.get("seq")
        _execute_nasa_fetch_for_aoi(
            south=float(fetch_ev["south"]),
            west=float(fetch_ev["west"]),
            north=float(fetch_ev["north"]),
            east=float(fetch_ev["east"]),
            resolution_m=float(fetch_ev.get("resolution_m", st.session_state.resolution_m)),
            start_date=str(fetch_ev.get("start_date") or st.session_state.nasa_start_date),
            end_date=str(fetch_ev.get("end_date") or st.session_state.nasa_end_date),
            run_name=st.session_state.nasa_run_name,
        )
        needs_rerun = True

    if needs_rerun:
        st.rerun()

    # --- Surface Element Area Time-Frame Analytics Panel (Toggled ON by Default) ---
    if st.session_state.show_graph:
        with st.container(border=True):
            st.markdown("<div class='sar-analytics-anchor'></div>", unsafe_allow_html=True)
            h_left, h_mode, h_right = st.columns([3.0, 1.4, 0.9], vertical_alignment="center")
            h_left.markdown(f"#### {tr(lang, 'graph_title')} — {bundle.region.name}")
            chosen_metric = h_mode.radio(
                "Metric",
                options=["area", "share"],
                index=0 if st.session_state.chart_metric_mode == "area" else 1,
                format_func=lambda m: tr(lang, "chart_tab_area") if m == "area" else tr(lang, "chart_tab_share"),
                horizontal=True,
                label_visibility="collapsed",
                key="chart_metric_toggle",
            )
            st.session_state.chart_metric_mode = chosen_metric
            if h_right.button(tr(lang, "hide_graph"), width="stretch"):
                st.session_state.show_graph = False
                st.rerun()

            if len(bundle.frames) >= 2:
                base_f = bundle.frames[0]
                end_f = bundle.frames[-1]
                summary = compute_change_summary(
                    baseline_frame=base_f,
                    frame=end_f,
                    selected_class_ids=st.session_state.selected_elements,
                    pixel_area_km2=bundle.region.pixel_area_km2,
                )
                st.markdown(
                    f"""
                    <div class="sar-kpi-grid">
                        <div class="sar-kpi-tile" style="--kpi-accent:#38BDF8;">
                            <div class="sar-kpi-label">{tr(lang, 'kpi_active')} ({end_f.scene.acq_date})</div>
                            <div class="sar-kpi-value">{summary['active_km2']:.2f} km²</div>
                            <div class="sar-kpi-sub">{summary['net_km2']:+.2f} km² {tr(lang, 'kpi_vs')} {base_f.scene.acq_date}</div>
                        </div>
                        <div class="sar-kpi-tile" style="--kpi-accent:#F43F5E;">
                            <div class="sar-kpi-label">{tr(lang, 'kpi_gained')}</div>
                            <div class="sar-kpi-value">{summary['gained_km2']:.2f} km²</div>
                            <div class="sar-kpi-sub">{base_f.scene.acq_date} &rarr; {end_f.scene.acq_date}</div>
                        </div>
                        <div class="sar-kpi-tile" style="--kpi-accent:#FACC15;">
                            <div class="sar-kpi-label">{tr(lang, 'kpi_lost')}</div>
                            <div class="sar-kpi-value">{summary['lost_km2']:.2f} km²</div>
                            <div class="sar-kpi-sub">{len(bundle.frames)} {tr(lang, 'scenes_word')}</div>
                        </div>
                        <div class="sar-kpi-tile" style="--kpi-accent:#14B8A6;">
                            <div class="sar-kpi-label">{tr(lang, 'kpi_persisting')}</div>
                            <div class="sar-kpi-value">{summary['persisting_km2']:.2f} km²</div>
                            <div class="sar-kpi-sub">{int(round(bundle.region.resolution_m))} m grid</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c_chart, c_trans = st.columns([1.85, 1.0])
                chart = _build_timeseries_chart(
                    bundle,
                    st.session_state.selected_elements,
                    lang=lang,
                    metric_mode=st.session_state.chart_metric_mode,
                )
                c_chart.altair_chart(chart, width="stretch")

                transitions = compute_transition_rows(
                    baseline_frame=base_f,
                    target_frame=end_f,
                    pixel_area_km2=bundle.region.pixel_area_km2,
                )
                mean_cloud = sum(float(fr.scene.cloud_cover_optical_pct) for fr in bundle.frames) / len(bundle.frames)
                if transitions:
                    top_rows = transitions[:6]
                    max_km2 = max(float(r["Area (km²)"]) for r in top_rows) or 1.0
                    trans_items_html = []
                    for row in top_rows:
                        pct = min(100.0, max(4.0, (float(row["Area (km²)"]) / max_km2) * 100.0))
                        # Localize short names cleanly when language is German
                        trans_lbl = row["Transition"]
                        if lang == "de":
                            for cid_t, cls_t in SURFACE_CLASSES.items():
                                trans_lbl = trans_lbl.replace(
                                    cls_t.short_label,
                                    localized_class_name(cid_t, cls_t.short_label, "de"),
                                )
                        trans_items_html.append(
                            f"""
                            <div class="sar-trans-row">
                                <div class="sar-trans-head">
                                    <span>{trans_lbl}</span>
                                    <span class="sar-trans-val">{row['Area (km²)']:.2f} km²</span>
                                </div>
                                <div class="sar-trans-track">
                                    <div class="sar-trans-fill" style="width:{pct:.1f}%;"></div>
                                </div>
                            </div>
                            """
                        )
                    c_trans.markdown(
                        f"""
                        <div class="sar-trans-panel">
                            <div class="sar-trans-title">{tr(lang, 'transitions_title')}</div>
                            {''.join(trans_items_html)}
                            <div class="sar-insight-foot">{tr(lang, 'insights_caption', cloud=mean_cloud, n=len(bundle.frames))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    c_trans.info(tr(lang, "transitions_empty"))
            else:
                st.info(tr(lang, "graph_need_two"))


if __name__ == "__main__":
    main()
