"""Streamlit web application for Dancing with the SARs (NASA Space Apps)."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable when launched via `streamlit run app/app.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.classifier import list_models
from src.config import BASEMAPS, CHANGE_HEX, SELECTABLE_CLASSES, SURFACE_CLASSES
from src.database import (
    clear_classifications,
    get_database_stats,
    init_database,
    list_region_dates,
    list_regions,
)
from src.map_builder import (
    build_base_map,
    build_before_after_dual_map,
    build_dynamic_overlay_group,
)
from src.pipeline import (
    TimeframeBundle,
    compute_change_summary,
    compute_transition_rows,
    run_timeframe_pipeline,
)

st.set_page_config(
    page_title="Dancing with the SARs · NISAR Surface Change Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle aerospace styling that works cleanly in both dark and light Streamlit themes
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .sar-banner {
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(56, 189, 248, 0.28);
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.78), rgba(30, 41, 59, 0.65));
        color: #F8FAFC;
        margin-bottom: 0.9rem;
    }
    .sar-banner h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .sar-banner p {
        margin: 0.25rem 0 0 0;
        color: #CBD5E1;
        font-size: 0.92rem;
    }
    .status-strip {
        padding: 0.5rem 0.85rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background-color: rgba(15, 23, 42, 0.35);
        font-size: 0.84rem;
        margin-bottom: 0.75rem;
    }
    .legend-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        align-items: center;
        margin: 0.45rem 0 0.85rem 0;
    }
    .legend-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.22rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        font-size: 0.8rem;
        font-weight: 500;
    }
    .legend-swatch {
        width: 11px;
        height: 11px;
        border-radius: 3px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_session_state() -> None:
    st.session_state.setdefault("is_playing", False)
    st.session_state.setdefault("just_started_play", False)
    st.session_state.setdefault("play_speed", 1.1)
    st.session_state.setdefault("play_loop", True)
    st.session_state.setdefault("active_frame_idx", 0)
    st.session_state.setdefault("context_sig", None)
    st.session_state.setdefault("force_recompute_once", False)


def _render_legend_html(
    view_mode: str,
    selected_class_ids: list[int],
    sar_band: str,
) -> str:
    if view_mode == "change":
        items = [
            ("Gained (Appeared vs. Start)", CHANGE_HEX["gained"]),
            ("Lost (Disappeared vs. Start)", CHANGE_HEX["lost"]),
            ("Persisting (Unchanged)", CHANGE_HEX["persisting"]),
        ]
    elif view_mode == "backscatter":
        if sar_band == "False-Color Dual-Pol (HH/HV)":
            items = [
                ("Red: High HH (Urban / Double-Bounce)", "#EF4444"),
                ("Green: High HV (Tree Canopy Volume)", "#22C55E"),
                ("Blue: Large HH-HV (Flooded Veg / Wet Soil)", "#3B82F6"),
                ("Dark: Smooth Open Water", "#0F172A"),
            ]
        else:
            items = [
                ("Bright: Strong Backscatter (~0 dB)", "#FDE047"),
                ("Mid-Tone: Crops / Rough Soil (-12 dB)", "#3B82F6"),
                ("Dark: Specular Water (-22 dB)", "#1E293B"),
            ]
    else:
        items = [
            (SURFACE_CLASSES[cid].label, SURFACE_CLASSES[cid].color_hex)
            for cid in selected_class_ids
            if cid in SURFACE_CLASSES
        ]

    pills = "".join(
        f"<span class='legend-pill'>"
        f"<span class='legend-swatch' style='background:{color};'></span>"
        f"{label}</span>"
        for label, color in items
    )
    return f"<div class='legend-row'>{pills}</div>"


def _build_timeseries_chart(
    bundle: TimeframeBundle,
    selected_class_ids: list[int],
    active_date: str,
) -> alt.Chart:
    records = []
    for f in bundle.frames:
        for cid in selected_class_ids:
            cls_info = SURFACE_CLASSES[cid]
            records.append(
                {
                    "Date": f.scene.acq_date,
                    "Element": cls_info.short_label,
                    "Area (km²)": f.areas_km2.get(cid, 0.0),
                    "Color": cls_info.color_hex,
                }
            )

    df = pd.DataFrame(records)
    domain = [SURFACE_CLASSES[cid].short_label for cid in selected_class_ids]
    range_colors = [SURFACE_CLASSES[cid].color_hex for cid in selected_class_ids]

    base = alt.Chart(df).encode(
        x=alt.X("Date:N", title="Acquisition Date", sort=None),
        y=alt.Y("Area (km²):Q", title="Classified Area (km²)"),
        color=alt.Color(
            "Element:N",
            scale=alt.Scale(domain=domain, range=range_colors),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        tooltip=["Date", "Element", alt.Tooltip("Area (km²):Q", format=".2f")],
    )
    lines = base.mark_line(strokeWidth=2.6, point=alt.OverlayMarkDef(size=65))

    cursor_df = pd.DataFrame([{"Date": active_date}])
    cursor = (
        alt.Chart(cursor_df)
        .mark_rule(color="#F8FAFC", strokeDash=[5, 4], strokeWidth=2)
        .encode(x=alt.X("Date:N", sort=None))
    )
    return (lines + cursor).properties(height=250).interactive()


def main() -> None:
    init_database()
    _init_session_state()

    regions = list_regions()
    models = list_models()
    model_by_id = {m.model_id: m for m in models}

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Controls")

        region_labels = {r.region_id: f"{r.name} ({r.country})" for r in regions}
        selected_region_id = st.selectbox(
            "Study Area (from DB)",
            options=[r.region_id for r in regions],
            format_func=lambda rid: region_labels[rid],
            key="region_selector",
        )
        active_region = next(r for r in regions if r.region_id == selected_region_id)
        available_dates = list_region_dates(selected_region_id)

        st.subheader("Time Frame")
        if len(available_dates) >= 2:
            start_date, end_date = st.select_slider(
                "Select Date Window",
                options=available_dates,
                value=(available_dates[0], available_dates[-1]),
                key=f"timeframe_{selected_region_id}",
            )
        else:
            start_date = end_date = available_dates[0]
            st.info(f"Single scene available: {start_date}")

        st.subheader("Elements to View")
        p1, p2 = st.columns(2)
        if p1.button("Water + Trees", width="stretch"):
            for c in SELECTABLE_CLASSES:
                st.session_state[f"elem_{c.key}"] = c.id in (1, 2, 3)
            st.rerun()
        if p2.button("Show All", width="stretch"):
            for c in SELECTABLE_CLASSES:
                st.session_state[f"elem_{c.key}"] = True
            st.rerun()

        selected_class_ids: list[int] = []
        for cls_info in SELECTABLE_CLASSES:
            key = f"elem_{cls_info.key}"
            if key not in st.session_state:
                st.session_state[key] = cls_info.id in (1, 2, 3)
            checked = st.checkbox(
                cls_info.label,
                key=key,
                help=cls_info.radar_signature,
            )
            if checked:
                selected_class_ids.append(cls_info.id)

        if not selected_class_ids:
            st.warning("Select at least one surface element; defaulting to Water.")
            selected_class_ids = [1]

        st.subheader("Map View Mode")
        view_mode_label = st.radio(
            "Layer Display",
            options=[
                "Neural Network Classification",
                "Change Over Time Frame",
                "Raw SAR Backscatter (dB)",
            ],
            label_visibility="collapsed",
        )
        view_mode_map = {
            "Neural Network Classification": "elements",
            "Change Over Time Frame": "change",
            "Raw SAR Backscatter (dB)": "backscatter",
        }
        view_mode = view_mode_map[view_mode_label]

        sar_band = "False-Color Dual-Pol (HH/HV)"
        if view_mode == "backscatter":
            sar_band = st.selectbox(
                "Radar Band / Composite",
                options=["False-Color Dual-Pol (HH/HV)", "HH (dB)", "HV (dB)"],
            )

        st.subheader("Neural Network Backend")
        model_id = st.selectbox(
            "Classification Model",
            options=[m.model_id for m in models],
            format_func=lambda mid: model_by_id[mid].display_name,
            key="model_selector",
        )
        st.caption(model_by_id[model_id].description)

        if st.button("Re-run Inference on Time Frame", width="stretch"):
            st.session_state.force_recompute_once = True
            st.session_state.is_playing = False
            st.rerun()

        st.subheader("Map Appearance")
        basemap_label = st.selectbox(
            "Basemap",
            options=list(BASEMAPS.keys()),
            key="basemap_selector",
        )
        opacity = st.slider("Overlay Opacity", min_value=0.20, max_value=1.00, value=0.78, step=0.02)
        show_hotspots = st.checkbox("Show Hotspot Pins", value=True)

    # --- Header Banner ---
    st.markdown(
        f"""
        <div class="sar-banner">
            <h1>🛰️ Dancing with the SARs · Surface Change Explorer</h1>
            <p><strong>{active_region.name} ({active_region.country})</strong> — {active_region.phenomenon}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Execute Backend Pipeline (DB Query + Neural Network Inference) ---
    force_recompute = bool(st.session_state.pop("force_recompute_once", False))
    bundle = run_timeframe_pipeline(
        region_id=selected_region_id,
        start_date=start_date,
        end_date=end_date,
        model_id=model_id,
        force_recompute=force_recompute,
    )

    if not bundle.frames:
        st.error("No SAR scenes found for the selected region and time frame.")
        return

    # Reset frame index cleanly if user changes region or date window
    ctx_sig = (selected_region_id, start_date, end_date)
    if st.session_state.context_sig != ctx_sig:
        st.session_state.context_sig = ctx_sig
        st.session_state.active_frame_idx = 0
        st.session_state.is_playing = False

    st.session_state.active_frame_idx = min(
        max(st.session_state.active_frame_idx, 0),
        len(bundle.frames) - 1,
    )

    st.markdown(
        f"<div class='status-strip'>"
        f"<strong>Backend Pipeline:</strong> Pulled <strong>{len(bundle.frames)}</strong> SAR scenes from SQLite "
        f"({start_date} → {end_date}) · "
        f"<strong>{bundle.newly_inferred}</strong> newly classified by <code>{model_id}</code>, "
        f"<strong>{bundle.cache_hits}</strong> served from DB cache · "
        f"Total model compute: <strong>{bundle.total_inference_ms:.1f} ms</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    tab_map, tab_compare, tab_db, tab_science = st.tabs(
        [
            "🗺️ Interactive Map & Time-Lapse",
            "🔀 Before / After & Change Matrix",
            "🗄️ Database & Neural Net Pipeline",
            "📡 SAR Physics & Story",
        ]
    )

    # =========================================================================
    # TAB 1: Interactive Map + Play/Pause Time-Lapse
    # =========================================================================
    with tab_map:
        play_interval = (
            st.session_state.play_speed
            if st.session_state.is_playing and len(bundle.frames) > 1
            else None
        )

        @st.fragment(run_every=play_interval)
        def render_playback_workspace() -> None:
            frames = bundle.frames
            n_frames = len(frames)
            date_list = [f.scene.acq_date for f in frames]
            date_to_idx = {d: i for i, d in enumerate(date_list)}

            # Advance playback frame on timer ticks
            if st.session_state.is_playing and n_frames > 1:
                if st.session_state.just_started_play:
                    st.session_state.just_started_play = False
                else:
                    nxt = st.session_state.active_frame_idx + 1
                    if nxt >= n_frames:
                        if st.session_state.play_loop:
                            st.session_state.active_frame_idx = 0
                        else:
                            st.session_state.active_frame_idx = n_frames - 1
                            st.session_state.is_playing = False
                            st.rerun()
                    else:
                        st.session_state.active_frame_idx = nxt

            idx = st.session_state.active_frame_idx
            current_frame = frames[idx]
            baseline_frame = frames[0]

            # Playback transport controls
            c_reset, c_prev, c_play, c_next, c_speed, c_loop, c_badge = st.columns(
                [0.7, 0.7, 1.3, 0.7, 1.2, 0.9, 2.8]
            )
            if c_reset.button("⏮", help="Jump to Start Date", width="stretch"):
                st.session_state.active_frame_idx = 0
                st.session_state.is_playing = False
                st.rerun()

            if c_prev.button("◀", help="Previous Date", width="stretch"):
                st.session_state.active_frame_idx = max(0, idx - 1)
                st.session_state.is_playing = False
                st.rerun()

            play_label = "⏸ Pause" if st.session_state.is_playing else "▶ Play Changes"
            if c_play.button(
                play_label,
                type="primary",
                disabled=(n_frames <= 1),
                width="stretch",
            ):
                if st.session_state.is_playing:
                    st.session_state.is_playing = False
                else:
                    if idx >= n_frames - 1 and not st.session_state.play_loop:
                        st.session_state.active_frame_idx = 0
                    st.session_state.is_playing = True
                    st.session_state.just_started_play = True
                st.rerun()

            if c_next.button("▶", help="Next Date", width="stretch"):
                st.session_state.active_frame_idx = min(n_frames - 1, idx + 1)
                st.session_state.is_playing = False
                st.rerun()

            speed_choice = c_speed.selectbox(
                "Speed",
                options=[0.7, 1.1, 1.8],
                index=[0.7, 1.1, 1.8].index(st.session_state.play_speed),
                format_func=lambda s: f"{s:.1f}s / frame",
                label_visibility="collapsed",
            )
            if speed_choice != st.session_state.play_speed:
                st.session_state.play_speed = speed_choice
                if st.session_state.is_playing:
                    st.rerun()

            loop_val = c_loop.toggle("Loop", value=st.session_state.play_loop)
            st.session_state.play_loop = loop_val

            c_badge.markdown(
                f"<div style='padding-top:0.35rem;font-size:0.9rem;'>"
                f"📅 <strong>{current_frame.scene.acq_date}</strong> "
                f"(Frame {idx + 1}/{n_frames}) · "
                f"☁️ Optical Cloud: <strong>{current_frame.scene.cloud_cover_optical_pct:.0f}%</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Frame Scrubber synced cleanly before instantiation
            scrubber_key = f"scrubber_{selected_region_id}_{start_date}_{end_date}"
            st.session_state[scrubber_key] = current_frame.scene.acq_date

            def _on_scrub_change() -> None:
                chosen = st.session_state[scrubber_key]
                st.session_state.active_frame_idx = date_to_idx[chosen]
                st.session_state.is_playing = False

            if n_frames > 1:
                st.select_slider(
                    "Timeline Scrubber",
                    options=date_list,
                    key=scrubber_key,
                    on_change=_on_scrub_change,
                    label_visibility="collapsed",
                )

            # KPI Metrics Row
            summary = compute_change_summary(
                baseline_frame=baseline_frame,
                frame=current_frame,
                selected_class_ids=selected_class_ids,
                pixel_area_km2=active_region.pixel_area_km2,
            )
            mean_conf = float(current_frame.confidence.mean()) * 100.0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Selected Elements Area",
                f"{summary['active_km2']:.2f} km²",
                delta=f"{summary['net_km2']:+.2f} km² vs {baseline_frame.scene.acq_date}",
            )
            k2.metric("New / Gained Area", f"{summary['gained_km2']:.2f} km²")
            k3.metric("Lost / Receded Area", f"{summary['lost_km2']:.2f} km²")
            k4.metric(
                "Mean Model Confidence",
                f"{mean_conf:.1f}%",
                delta=f"{current_frame.inference_ms:.1f} ms · {'Cached' if current_frame.from_cache else 'Live NN'}",
                delta_color="off",
            )

            st.caption(
                f"📝 **Observation Note ({current_frame.scene.acq_date}):** {current_frame.scene.event_note}"
            )
            st.markdown(
                _render_legend_html(view_mode, selected_class_ids, sar_band),
                unsafe_allow_html=True,
            )

            # Interactive Map with smooth in-place overlay swap
            base_map = build_base_map(active_region, basemap_label=basemap_label)
            overlay_fg = build_dynamic_overlay_group(
                frame=current_frame,
                region=active_region,
                selected_class_ids=selected_class_ids,
                view_mode=view_mode,
                baseline_frame=baseline_frame,
                sar_band=sar_band,
                opacity=opacity,
                show_hotspots=show_hotspots,
            )

            st_folium(
                base_map,
                key=f"main_map_{selected_region_id}_{basemap_label}",
                height=520,
                use_container_width=True,
                returned_objects=[],
                feature_group_to_add=[overlay_fg],
                pixelated=True,
            )

            if n_frames > 1:
                st.markdown("#### Surface Element Area Over Time Frame")
                chart = _build_timeseries_chart(
                    bundle=bundle,
                    selected_class_ids=selected_class_ids,
                    active_date=current_frame.scene.acq_date,
                )
                st.altair_chart(chart, width="stretch")

        render_playback_workspace()

    # =========================================================================
    # TAB 2: Before / After & Land-Cover Change Matrix
    # =========================================================================
    with tab_compare:
        st.subheader("Before vs. After Surface Change Inspector")
        frames = bundle.frames
        if len(frames) < 2:
            st.info("Expand your Time Frame in the sidebar to include at least 2 dates for Before/After comparison.")
        else:
            date_opts = [f.scene.acq_date for f in frames]
            c_left, c_right, c_mode = st.columns([1.2, 1.2, 1.4])
            before_date = c_left.selectbox("Before Date (Baseline)", options=date_opts, index=0)
            after_date = c_right.selectbox(
                "After Date (Target)", options=date_opts, index=len(date_opts) - 1
            )
            right_mode = c_mode.radio(
                "Right Map Displays",
                options=["Classified Elements", "Change Mask (Gained / Lost)"],
                horizontal=True,
            )

            before_frame = next(f for f in frames if f.scene.acq_date == before_date)
            after_frame = next(f for f in frames if f.scene.acq_date == after_date)

            dual_map = build_before_after_dual_map(
                baseline_frame=before_frame,
                target_frame=after_frame,
                region=active_region,
                selected_class_ids=selected_class_ids,
                basemap_label=basemap_label,
                opacity=opacity,
                show_change_on_right=(right_mode == "Change Mask (Gained / Lost)"),
            )
            st.caption(
                f"Left: **{before_date}** (Classified) · Right: **{after_date}** ({right_mode}) — Pan & zoom stay synced."
            )
            st.iframe(dual_map._repr_html_(), height=480)

            st.markdown("#### Dominant Surface Transitions")
            transitions = compute_transition_rows(
                baseline_frame=before_frame,
                target_frame=after_frame,
                pixel_area_km2=active_region.pixel_area_km2,
            )
            if transitions:
                t_df = pd.DataFrame(transitions)
                col_tbl, col_bar = st.columns([1.1, 1.3])
                col_tbl.dataframe(t_df.head(10), width="stretch", hide_index=True)
                bar = (
                    alt.Chart(t_df.head(8))
                    .mark_bar(cornerRadiusEnd=4, color="#38BDF8")
                    .encode(
                        x=alt.X("Area (km²):Q", title="Changed Area (km²)"),
                        y=alt.Y("Transition:N", sort="-x", title=None),
                        tooltip=["Transition", alt.Tooltip("Area (km²):Q", format=".2f")],
                    )
                    .properties(height=280)
                )
                col_bar.altair_chart(bar, width="stretch")
            else:
                st.info("No major class transitions exceeded 0.05 km² between these two dates.")

    # =========================================================================
    # TAB 3: Database & Neural Network Integration Guide
    # =========================================================================
    with tab_db:
        st.subheader("SQLite Observation Database & Model Pipeline")
        stats = get_database_stats()
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Regions in DB", stats["regions"])
        d2.metric("Stored SAR Scenes", stats["sar_scenes"])
        d3.metric("Cached Classifications", stats["classifications"])
        d4.metric("Database Size", f"{stats['size_kb']} KB")

        c_btn1, c_btn2 = st.columns([1.2, 2.8])
        if c_btn1.button("Clear Cached Model Predictions", width="stretch"):
            deleted = clear_classifications(region_id=selected_region_id, model_id=model_id)
            st.success(f"Cleared {deleted} cached prediction(s) for {active_region.name}. Rerunning...")
            st.rerun()
        c_btn2.caption(f"SQLite file path: `{stats['db_path']}`")

        scenes_df = pd.DataFrame(stats["scenes_table"])
        st.dataframe(scenes_df, width="stretch", hide_index=True)

        with st.expander("How Your Team Plugs In Real NISAR Data & Your Trained Neural Network"):
            st.markdown(
                """
                **1. Ingesting Real NISAR / ASF Data into the Database (`src/database.py`)**  
                In `notebooks/01_daten_suchen_und_laden.ipynb`, download and crop your dual-pol SAR rasters (`HH` and `HV` in dB) to matching `(H, W)` NumPy arrays, then insert them directly into SQLite:
                ```python
                from src.database import insert_sar_scene

                insert_sar_scene(
                    region_id="elbe_flood",
                    acq_date="2026-06-20",
                    hh_db=hh_array_db,
                    hv_db=hv_array_db,
                    sensor="NISAR L-SAR GCOV",
                    event_note="Downloaded from ASF Vertex",
                )
                ```
                As soon as a scene is inserted, it appears in the app's **Time Frame** slider automatically.

                **2. Connecting Your Friends' PyTorch / ONNX Neural Network (`src/classifier.py`)**  
                Subclass `BaseSARClassifier` and implement a single method:
                ```python
                class TeamPyTorchClassifier(BaseSARClassifier):
                    model_id = "team_unet_v1"
                    display_name = "Team U-Net (PyTorch)"
                    description = "Trained on dual-pol NISAR patches."

                    def predict(self, hh_db: np.ndarray, hv_db: np.ndarray) -> ClassificationResult:
                        # Run your torch model -> return class_mask (uint8 0..5) & confidence (float32 0..1)
                        ...
                ```
                Register it in `get_classifier()` and `list_models()` inside `src/classifier.py`, and it immediately becomes selectable in the sidebar and caches outputs in SQLite!
                """
            )

    # =========================================================================
    # TAB 4: SAR Science & Region Story
    # =========================================================================
    with tab_science:
        st.subheader(f"Physical Story: {active_region.name}")
        st.write(active_region.story)
        st.info(f"**Data & Context Sources:** {active_region.data_sources}")

        st.markdown("#### How Radar Sees Earth's Surface Elements")
        science_rows = []
        for cid in range(6):
            c = SURFACE_CLASSES[cid]
            science_rows.append(
                {
                    "Class ID": c.id,
                    "Surface Element": c.label,
                    "Color": c.color_hex,
                    "L-Band SAR Scattering Mechanism": c.radar_signature,
                    "Physical Interpretation": c.description,
                }
            )
        st.dataframe(pd.DataFrame(science_rows), width="stretch", hide_index=True)

        st.markdown(
            """
            #### NASA & Partner Resources Used in Context
            - [NASA Earthdata Search](https://search.earthdata.nasa.gov/search) & [ASF Data Search Vertex](https://search.asf.alaska.edu/)
            - [ASF NISAR Data User Guide](https://nisar-docs.asf.alaska.edu/) & [NISAR StoryMaps](https://nisar.jpl.nasa.gov/applications/arcgis-storymaps/)
            - [CSA RADARSAT Constellation Mission](https://donnees-data.asc-csa.gc.ca/en/dataset/rcm) & [CONAE SAOCOM](https://catalogos.conae.gov.ar/catalogo/catalogoSatSaocomInfo.html)
            """
        )


if __name__ == "__main__":
    main()
