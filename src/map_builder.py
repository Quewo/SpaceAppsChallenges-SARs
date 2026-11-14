"""Folium map construction helpers for smooth time-lapse and before/after SAR comparison."""
from __future__ import annotations

from typing import Optional, Sequence

import folium
from folium.plugins import DualMap, Fullscreen, MousePosition

from src.config import BASEMAPS
from src.database import RegionRecord
from src.pipeline import ClassifiedFrame, build_rgba_overlay, extract_hotspot_markers


def build_base_map(region: RegionRecord, basemap_label: str = "Esri Satellite Imagery") -> folium.Map:
    """Create the static base Folium map for a study region (kept stable across playback frames)."""
    tiles = BASEMAPS.get(basemap_label, "Esri.WorldImagery")
    m = folium.Map(
        location=[region.center_lat, region.center_lon],
        zoom_start=region.zoom_start,
        tiles=tiles,
        control_scale=True,
    )

    # Subtle study-area boundary frame
    folium.Rectangle(
        bounds=region.bounds,
        color="#38BDF8",
        weight=1.5,
        dash_array="6, 6",
        fill=False,
        tooltip=f"SAR Study Area: {region.name}",
    ).add_to(m)

    Fullscreen(position="topleft").add_to(m)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coords:",
        num_digits=4,
    ).add_to(m)
    return m


def build_dynamic_overlay_group(
    frame: ClassifiedFrame,
    region: RegionRecord,
    selected_class_ids: Sequence[int],
    view_mode: str = "elements",
    baseline_frame: Optional[ClassifiedFrame] = None,
    sar_band: str = "False-Color Dual-Pol (HH/HV)",
    opacity: float = 0.78,
    show_hotspots: bool = True,
) -> folium.FeatureGroup:
    """
    Build a dynamic `folium.FeatureGroup` swapped in-place by `st_folium(feature_group_to_add=...)`.
    """
    fg = folium.FeatureGroup(name=f"SAR Layer ({frame.scene.acq_date})")

    rgba = build_rgba_overlay(
        frame=frame,
        selected_class_ids=selected_class_ids,
        view_mode=view_mode,
        baseline_frame=baseline_frame,
        sar_band=sar_band,
    )
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=region.bounds,
        opacity=float(opacity),
        origin="upper",
        mercator_project=True,
        pixelated=True,
        interactive=True,
        name=f"Overlay {frame.scene.acq_date}",
    ).add_to(fg)

    if show_hotspots and view_mode != "backscatter":
        hotspots = extract_hotspot_markers(
            frame=frame,
            baseline_frame=baseline_frame,
            region=region,
            selected_class_ids=selected_class_ids,
        )
        for spot in hotspots:
            popup_html = (
                f"<div style='font-family:sans-serif;min-width:190px;'>"
                f"<strong style='color:{spot['color']};'>{spot['title']}</strong><br>"
                f"<span style='font-size:12px;'>{spot['detail']}</span><br>"
                f"<span style='font-size:11px;color:#64748B;'>Date: {frame.scene.acq_date}</span>"
                f"</div>"
            )
            folium.CircleMarker(
                location=[spot["lat"], spot["lon"]],
                radius=7,
                color="#FFFFFF",
                weight=2,
                fill=True,
                fill_color=spot["color"],
                fill_opacity=0.92,
                tooltip=f"{spot['title']} ({frame.scene.acq_date})",
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(fg)

    return fg


def build_before_after_dual_map(
    baseline_frame: ClassifiedFrame,
    target_frame: ClassifiedFrame,
    region: RegionRecord,
    selected_class_ids: Sequence[int],
    basemap_label: str = "Esri Satellite Imagery",
    opacity: float = 0.80,
    show_change_on_right: bool = False,
) -> DualMap:
    """Create a synced side-by-side map comparing Start Date (left) vs End Date (right)."""
    tiles = BASEMAPS.get(basemap_label, "Esri.WorldImagery")
    dual = DualMap(
        location=[region.center_lat, region.center_lon],
        zoom_start=region.zoom_start,
        tiles=tiles,
    )

    left_rgba = build_rgba_overlay(
        frame=baseline_frame,
        selected_class_ids=selected_class_ids,
        view_mode="elements",
    )
    right_rgba = build_rgba_overlay(
        frame=target_frame,
        selected_class_ids=selected_class_ids,
        view_mode="change" if show_change_on_right else "elements",
        baseline_frame=baseline_frame,
    )

    folium.raster_layers.ImageOverlay(
        image=left_rgba,
        bounds=region.bounds,
        opacity=float(opacity),
        origin="upper",
        mercator_project=True,
        pixelated=True,
    ).add_to(dual.m1)

    folium.raster_layers.ImageOverlay(
        image=right_rgba,
        bounds=region.bounds,
        opacity=float(opacity),
        origin="upper",
        mercator_project=True,
        pixelated=True,
    ).add_to(dual.m2)

    for m_side in (dual.m1, dual.m2):
        folium.Rectangle(
            bounds=region.bounds,
            color="#38BDF8",
            weight=1.2,
            dash_array="5, 5",
            fill=False,
        ).add_to(m_side)

    return dual
