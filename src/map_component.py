"""Modern bidirectional Leaflet geospatial canvas using Streamlit Components v2."""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

_COMPONENT_HTML = """
<div class="sar-studio-root" id="sar-studio-root">
  <div class="sar-map-canvas" id="sar-leaflet-map"></div>

  <!-- Top Floating Control Bar -->
  <div class="sar-hud-top">
    <!-- Left: Area Search -->
    <div class="sar-glass sar-search-wrap">
      <svg class="sar-icon-sm sar-muted" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="11" cy="11" r="8"></circle>
        <path d="m21 21-4.3-4.3"></path>
      </svg>
      <input
        type="text"
        id="sar-search-input"
        class="sar-search-input"
        placeholder="Search any city, river, or lat, lon..."
        autocomplete="off"
      />
      <button type="button" id="sar-search-clear" class="sar-icon-clear" title="Clear">&times;</button>
      <div class="sar-search-dropdown" id="sar-search-dropdown"></div>
    </div>

    <!-- Center: Element Icon Toggles + Quick AI Switch + Data Modes Filter -->
    <div class="sar-glass sar-center-bar">
      <div class="sar-elem-group" id="sar-elem-group">
        <button type="button" class="sar-elem-btn" data-cid="1" data-color="#0EA5E9" data-tooltip="Toggle Water &amp; Flood Extent">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
          </svg>
        </button>
        <button type="button" class="sar-elem-btn" data-cid="2" data-color="#22C55E" data-tooltip="Toggle Tree Lines &amp; Forest Canopy">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="m12 2-6 8h4l-3 6h10l-3-6h4Z"></path>
            <path d="M12 16v6"></path>
          </svg>
        </button>
        <button type="button" class="sar-elem-btn" data-cid="3" data-color="#14B8A6" data-tooltip="Toggle Wetlands &amp; Flooded Vegetation">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 16c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"></path>
            <path d="M2 20c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"></path>
            <path d="M9 12V5c0 0 3 1 3 5"></path>
            <path d="M15 12V4c0 0-3 1.5-3 6"></path>
          </svg>
        </button>
        <button type="button" class="sar-elem-btn" data-cid="4" data-color="#EAB308" data-tooltip="Toggle Cropland &amp; Soil Moisture">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="3"></rect>
            <path d="M3 12h18"></path>
            <path d="M12 3v18"></path>
          </svg>
        </button>
        <button type="button" class="sar-elem-btn" data-cid="5" data-color="#F97316" data-tooltip="Toggle Urban &amp; Infrastructure">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 21h18"></path>
            <path d="M5 21V7l6-4v18"></path>
            <path d="M11 21V11l8-3v13"></path>
            <path d="M8 9h.01"></path>
            <path d="M8 13h.01"></path>
            <path d="M15 13h.01"></path>
          </svg>
        </button>
      </div>

      <div class="sar-divider"></div>

      <div class="sar-seg-group" id="sar-mode-group">
        <button type="button" class="sar-seg-btn" data-mode="elements" data-tooltip="Classified Surface Elements">Elements</button>
        <button type="button" class="sar-seg-btn" data-mode="change" data-tooltip="Changes vs. Start of Time Frame">Changes</button>
      </div>

      <div class="sar-pop-wrap" id="sar-mode-wrap">
        <button type="button" class="sar-filter-btn" id="sar-mode-filter-btn" data-tooltip="Filter Individual Data Modes (Elevation, Slope, OPERA S1, NISAR GCOV, Coherence)">
          <svg class="sar-icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
          </svg>
          <span id="sar-mode-filter-label">Data Modes</span>
          <span class="sar-caret">&#9662;</span>
        </button>
        <div class="sar-glass sar-layer-popover" id="sar-mode-popover"></div>
      </div>
    </div>

    <!-- Right: Resolution/Area, Timeframe, Graph Toggle & Advanced Sidebar -->
    <div class="sar-hud-right">
      <div class="sar-pop-wrap" id="sar-res-wrap">
        <button type="button" class="sar-glass sar-pill-btn" id="sar-res-btn" data-tooltip="Adjust Resolution, Square Size &amp; Fetch NASA Data">
          <svg class="sar-icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="7" height="7"></rect>
            <rect x="14" y="3" width="7" height="7"></rect>
            <rect x="14" y="14" width="7" height="7"></rect>
            <rect x="3" y="14" width="7" height="7"></rect>
          </svg>
          <span id="sar-res-label">30 m</span>
        </button>
        <div class="sar-glass sar-res-popover" id="sar-res-popover">
          <div class="sar-pop-title">Resolution &amp; Collected Area</div>
          <label class="sar-pop-label">Target Resolution (TARGET_RES_M)</label>
          <div class="sar-chip-row" id="sar-res-chips">
            <button type="button" class="sar-mini-chip" data-res="15">15 m</button>
            <button type="button" class="sar-mini-chip" data-res="30">30 m</button>
            <button type="button" class="sar-mini-chip" data-res="60">60 m</button>
            <button type="button" class="sar-mini-chip" data-res="100">100 m</button>
            <button type="button" class="sar-mini-chip" data-res="150">150 m</button>
          </div>
          <label class="sar-pop-label">Observation Square Size</label>
          <div class="sar-chip-row" id="sar-size-chips">
            <button type="button" class="sar-mini-chip" data-km="8">8&times;8 km</button>
            <button type="button" class="sar-mini-chip" data-km="16">16&times;16 km</button>
            <button type="button" class="sar-mini-chip" data-km="28">28&times;28 km</button>
          </div>
          <div class="sar-pop-meta" id="sar-res-meta">Grid: -- &middot; Drag corners for custom box</div>
          <div class="sar-pop-divider"></div>
          <button type="button" id="sar-nasa-fetch-btn" class="sar-btn-primary sar-btn-full">
            Fetch Real NASA Data for Square
          </button>
          <div class="sar-pop-meta" id="sar-token-hint">Uses Earthdata AOI slicing (`NASADEM`, `OPERA`, `NISAR`)</div>
        </div>
      </div>

      <div class="sar-tf-wrap">
        <button type="button" class="sar-glass sar-pill-btn" id="sar-tf-btn" data-tooltip="Choose Time Frame">
          <svg class="sar-icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="4" width="18" height="18" rx="2"></rect>
            <path d="M16 2v4M8 2v4M3 10h18"></path>
          </svg>
          <span id="sar-tf-label">--</span>
        </button>
        <div class="sar-glass sar-tf-popover" id="sar-tf-popover">
          <div class="sar-pop-title" id="sar-tf-title">Analysis Time Frame</div>
          <div class="sar-pop-meta" id="sar-tf-span-hint">Source window: 2016-01-01 &rarr; Present</div>
          <div class="sar-chip-row" id="sar-tf-presets">
            <button type="button" class="sar-mini-chip" data-tf-preset="max">Max</button>
            <button type="button" class="sar-mini-chip" data-tf-preset="5y">5Y</button>
            <button type="button" class="sar-mini-chip" data-tf-preset="1y">1Y</button>
            <button type="button" class="sar-mini-chip" data-tf-preset="6m">6M</button>
          </div>
          <label class="sar-pop-label" id="sar-tf-start-lbl">Start Date</label>
          <input type="date" id="sar-tf-start" class="sar-select" />
          <label class="sar-pop-label" id="sar-tf-end-lbl">End Date</label>
          <input type="date" id="sar-tf-end" class="sar-select" />
          <div class="sar-pop-actions">
            <button type="button" id="sar-tf-full" class="sar-btn-ghost">Full Range</button>
            <button type="button" id="sar-tf-apply" class="sar-btn-primary">Apply</button>
          </div>
        </div>
      </div>

      <button type="button" class="sar-glass sar-pill-btn" id="sar-graph-btn" data-tooltip="Toggle Area Time-Frame Graph">
        <svg class="sar-icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 3v18h18"></path>
          <path d="m19 9-5 5-4-4-3 3"></path>
        </svg>
        <span id="sar-graph-label">Graph</span>
      </button>

      <button type="button" class="sar-glass sar-pill-btn" id="sar-sidebar-btn" data-tooltip="Toggle NASA Collector &amp; Model Sidebar">
        <svg class="sar-icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>
          <circle cx="12" cy="12" r="3"></circle>
        </svg>
        <span id="sar-sidebar-label">Controls</span>
      </button>

      <button type="button" class="sar-glass sar-pill-btn sar-icon-only-btn" id="sar-fullscreen-btn" data-tooltip="Fullscreen Map">
        <svg id="sar-fs-enter-icon" class="sar-icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="15 3 21 3 21 9"></polyline>
          <polyline points="9 21 3 21 3 15"></polyline>
          <line x1="21" y1="3" x2="14" y2="10"></line>
          <line x1="3" y1="21" x2="10" y2="14"></line>
        </svg>
        <svg id="sar-fs-exit-icon" class="sar-icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none;">
          <polyline points="4 14 10 14 10 20"></polyline>
          <polyline points="20 10 14 10 14 4"></polyline>
          <line x1="14" y1="10" x2="21" y2="3"></line>
          <line x1="3" y1="21" x2="10" y2="14"></line>
        </svg>
      </button>
    </div>
  </div>

  <!-- AOI Status & Drag Hint Chip (Top-Left under search) -->
  <div class="sar-glass sar-aoi-status" id="sar-aoi-status">
    <span class="sar-dot-live" id="sar-dot-live"></span>
    <span id="sar-region-title">Loading region...</span>
    <span class="sar-sep">&middot;</span>
    <span id="sar-aoi-dims">-- km&sup2;</span>
    <span class="sar-sep">&middot;</span>
    <span class="sar-hint-muted" id="sar-drag-hint">Drag or resize square to analyze any area</span>
  </div>

  <!-- Bottom-Right Modern Map Controls -->
  <div class="sar-map-controls">
    <button type="button" class="sar-glass sar-basemap-btn" id="sar-basemap-btn" data-tooltip="Switch Map Style">Satellite</button>
    <div class="sar-glass sar-zoom-stack">
      <button type="button" class="sar-ctrl-btn" id="sar-center-btn" data-tooltip="Center on Observation Square">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M12 2v4M12 18v4M2 12h4M18 12h4"></path>
        </svg>
      </button>
      <button type="button" class="sar-ctrl-btn" id="sar-zoom-in" data-tooltip="Zoom In">+</button>
      <button type="button" class="sar-ctrl-btn" id="sar-zoom-out" data-tooltip="Zoom Out">&minus;</button>
    </div>
  </div>

  <!-- Bottom Floating Timeline & Playback Dock -->
  <div class="sar-hud-bottom">
    <div class="sar-glass sar-legend-strip" id="sar-legend-strip"></div>
    <div class="sar-glass sar-play-dock">
      <div class="sar-transport">
        <button type="button" class="sar-tbtn" id="sar-btn-reset" data-tooltip="First Date">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zm3.5 6 8.5 6V6z"/></svg>
        </button>
        <button type="button" class="sar-tbtn" id="sar-btn-prev" data-tooltip="Previous Date">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="m15.41 7.41-1.41-1.41-6 6 6 6 1.41-1.41L10.83 12z"/></svg>
        </button>
        <button type="button" class="sar-play-btn" id="sar-btn-play" data-tooltip="Play / Pause Time-Lapse">
          <svg id="sar-icon-play" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
          <svg id="sar-icon-pause" viewBox="0 0 24 24" fill="currentColor" style="display:none;"><path d="M6 19h4V5H6zm8-14v14h4V5z"/></svg>
        </button>
        <button type="button" class="sar-tbtn" id="sar-btn-next" data-tooltip="Next Date">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="m10 6-1.41 1.41L13.17 12l-4.58 4.59L10 18l6-6z"/></svg>
        </button>
      </div>

      <div class="sar-scrubber-wrap">
        <input type="range" id="sar-scrubber" class="sar-scrubber" min="0" max="0" value="0" step="1" />
        <div class="sar-frame-meta">
          <span class="sar-date-pill" id="sar-active-date">--</span>
          <span class="sar-meta-text" id="sar-frame-info">--</span>
        </div>
      </div>

      <div class="sar-dock-right">
        <div class="sar-pop-wrap" id="sar-speed-wrap">
          <button type="button" class="sar-chip-btn" id="sar-speed-btn" data-tooltip="Playback Step">1mo / fr &#9662;</button>
          <div class="sar-glass sar-speed-popover" id="sar-speed-popover"></div>
        </div>
        <button type="button" class="sar-chip-btn active" id="sar-loop-btn" data-tooltip="Loop Playback">Loop</button>
        <div class="sar-area-readout">
          <span class="sar-area-val" id="sar-area-val">-- km&sup2;</span>
          <span class="sar-area-sub" id="sar-area-sub">selected cover</span>
        </div>
      </div>
    </div>
  </div>
</div>
"""

_COMPONENT_CSS = """
.sar-studio-root {
    position: relative;
    width: 100%;
    height: calc(100vh - 118px);
    min-height: 680px;
    border-radius: 16px;
    overflow: hidden;
    background: #070B14;
    border: 1px solid rgba(148, 163, 184, 0.22);
    box-shadow: 0 22px 54px rgba(2, 6, 23, 0.62);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #F8FAFC;
    user-select: none;
}

.sar-studio-root.sar-is-fullscreen {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: none !important;
    border-radius: 0 !important;
    border: none !important;
    z-index: 999999 !important;
}

.sar-studio-root:fullscreen {
    width: 100vw !important;
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: none !important;
    border-radius: 0 !important;
    border: none !important;
}

.sar-icon-only-btn {
    padding: 8px 9px !important;
}

.sar-speed-popover {
    position: absolute;
    bottom: calc(100% + 10px);
    right: 0;
    width: 215px;
    padding: 10px;
    border-radius: 13px;
    display: none;
    flex-direction: column;
    gap: 5px;
    z-index: 1260;
}
.sar-speed-popover.open {
    display: flex;
}
.sar-speed-opt {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 6px 9px;
    border-radius: 8px;
    border: 1px solid transparent;
    background: transparent;
    color: #CBD5E1;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    text-align: left;
}
.sar-speed-opt:hover {
    background: rgba(56, 189, 248, 0.14);
    color: #FFFFFF;
}
.sar-speed-opt.active {
    background: rgba(14, 165, 233, 0.24);
    border-color: rgba(56, 189, 248, 0.55);
    color: #38BDF8;
}

.sar-map-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
    background: #070B14;
}

.sar-studio-root.sar-dark-tiles .leaflet-tile-pane {
    filter: invert(96%) hue-rotate(180deg) brightness(88%) contrast(108%) saturate(45%);
}

/* Modernize Leaflet attribution & clean pixel rendering */
.sar-studio-root .leaflet-image-layer {
    image-rendering: pixelated;
    image-rendering: crisp-edges;
}

.sar-studio-root .leaflet-control-attribution {
    background: rgba(8, 13, 25, 0.55) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: rgba(148, 163, 184, 0.72) !important;
    font-size: 10px !important;
    padding: 2px 8px !important;
    border-radius: 8px 0 0 0 !important;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    border-left: 1px solid rgba(148, 163, 184, 0.12);
    margin: 0 !important;
}

.sar-studio-root .leaflet-control-attribution a {
    color: rgba(148, 163, 184, 0.85) !important;
    text-decoration: none !important;
}

/* Observation Square handles & badge */
.sar-aoi-rect {
    cursor: grab !important;
}
.sar-aoi-rect:active {
    cursor: grabbing !important;
}

.sar-corner-handle {
    width: 14px !important;
    height: 14px !important;
    border-radius: 4px;
    background: #F8FAFC;
    border: 2.5px solid #0EA5E9;
    box-shadow: 0 2px 10px rgba(14, 165, 233, 0.65);
    transition: transform 0.12s ease, background 0.12s ease;
}
.sar-corner-handle:hover {
    transform: scale(1.25);
    background: #38BDF8;
}
.sar-corner-nw, .sar-corner-se { cursor: nwse-resize !important; }
.sar-corner-ne, .sar-corner-sw { cursor: nesw-resize !important; }

.sar-box-badge-wrap {
    background: transparent !important;
    border: none !important;
}
.sar-box-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    padding: 3px 9px;
    border-radius: 999px;
    background: rgba(9, 14, 28, 0.85);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(56, 189, 248, 0.55);
    color: #E2E8F0;
    font-size: 11px;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    transform: translate(-50%, -100%);
    pointer-events: none;
}

/* Frosted Glass Surface */
.sar-glass {
    background: rgba(11, 17, 32, 0.78);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 10px 28px rgba(2, 6, 23, 0.45);
}

/* Top HUD Layout */
.sar-hud-top {
    position: absolute;
    top: 14px;
    left: 14px;
    right: 14px;
    z-index: 900;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    pointer-events: none;
}

.sar-hud-top > * {
    pointer-events: auto;
}

/* Search Bar */
.sar-search-wrap {
    position: relative;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 11px;
    border-radius: 12px;
    width: 228px;
}

.sar-icon-sm {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
}

.sar-muted {
    color: #94A3B8;
}

.sar-search-input {
    width: 100%;
    background: transparent;
    border: none;
    outline: none;
    color: #F8FAFC;
    font-size: 13px;
}
.sar-search-input::placeholder {
    color: #64748B;
}

.sar-icon-clear {
    background: transparent;
    border: none;
    color: #64748B;
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
    padding: 0 2px;
    display: none;
}
.sar-icon-clear:hover {
    color: #F8FAFC;
}

.sar-search-dropdown {
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    right: 0;
    border-radius: 12px;
    background: rgba(11, 17, 32, 0.94);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 16px 36px rgba(2, 6, 23, 0.65);
    overflow: hidden;
    display: none;
    max-height: 290px;
    overflow-y: auto;
    z-index: 1200;
}

.sar-search-section {
    padding: 6px 12px 4px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748B;
}

.sar-search-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
    width: 100%;
    padding: 8px 12px;
    text-align: left;
    background: transparent;
    border: none;
    color: #E2E8F0;
    cursor: pointer;
    font-size: 12.5px;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
}
.sar-search-item:hover {
    background: rgba(56, 189, 248, 0.14);
}
.sar-search-sub {
    font-size: 11px;
    color: #94A3B8;
}

/* Center Element Icons + View Mode */
.sar-center-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px;
    border-radius: 14px;
}

.sar-elem-group {
    display: flex;
    align-items: center;
    gap: 5px;
}

.sar-elem-btn {
    position: relative;
    width: 34px;
    height: 34px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(30, 41, 59, 0.55);
    color: #64748B;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.16s ease;
}

.sar-elem-btn svg {
    width: 17px;
    height: 17px;
}

.sar-elem-btn:hover {
    color: #F8FAFC;
    transform: translateY(-1px);
    background: rgba(51, 65, 85, 0.78);
}

.sar-elem-btn.active {
    color: #FFFFFF;
    border-color: var(--elem-color, #38BDF8);
    background: color-mix(in srgb, var(--elem-color, #38BDF8) 26%, rgba(15, 23, 42, 0.85));
    box-shadow: 0 0 14px color-mix(in srgb, var(--elem-color, #38BDF8) 38%, transparent);
}

.sar-divider {
    width: 1px;
    height: 22px;
    background: rgba(255, 255, 255, 0.12);
    margin: 0 2px;
}

.sar-seg-group {
    display: flex;
    align-items: center;
    gap: 2px;
    padding: 2px;
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.65);
}

.sar-seg-btn {
    position: relative;
    border: none;
    background: transparent;
    color: #94A3B8;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 10px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
}
.sar-seg-btn:hover {
    color: #F8FAFC;
}
.sar-seg-btn.active {
    background: rgba(56, 189, 248, 0.20);
    color: #38BDF8;
    box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.35);
}

/* Right Controls */
.sar-hud-right {
    display: flex;
    align-items: center;
    gap: 8px;
}

.sar-pill-btn {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    border-radius: 12px;
    color: #E2E8F0;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
}
.sar-pill-btn:hover {
    border-color: rgba(56, 189, 248, 0.45);
    color: #FFFFFF;
}
.sar-pill-btn.active {
    background: rgba(14, 165, 233, 0.25);
    border-color: rgba(56, 189, 248, 0.55);
    color: #38BDF8;
}

.sar-tf-wrap, .sar-pop-wrap {
    position: relative;
}
.sar-filter-btn {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    background: rgba(15, 23, 42, 0.72);
    color: #CBD5E1;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
}
.sar-filter-btn:hover {
    color: #FFFFFF;
    border-color: rgba(56, 189, 248, 0.45);
}
.sar-filter-btn.active {
    background: rgba(14, 165, 233, 0.24);
    border-color: rgba(56, 189, 248, 0.6);
    color: #38BDF8;
}
.sar-caret {
    font-size: 10px;
    opacity: 0.75;
}
.sar-layer-popover {
    position: absolute;
    top: calc(100% + 10px);
    left: 50%;
    transform: translateX(-50%);
    width: 315px;
    max-height: 430px;
    overflow-y: auto;
    padding: 10px;
    border-radius: 14px;
    display: none;
    flex-direction: column;
    gap: 4px;
    z-index: 1250;
}
.sar-layer-popover.open {
    display: flex;
}
.sar-layer-group-title {
    padding: 6px 8px 3px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748B;
}
.sar-layer-opt {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    width: 100%;
    padding: 7px 9px;
    border-radius: 9px;
    border: 1px solid transparent;
    background: transparent;
    color: #E2E8F0;
    text-align: left;
    cursor: pointer;
    transition: background 0.12s ease, border-color 0.12s ease;
}
.sar-layer-opt:hover:not(:disabled) {
    background: rgba(56, 189, 248, 0.12);
    border-color: rgba(56, 189, 248, 0.25);
}
.sar-layer-opt.active {
    background: rgba(14, 165, 233, 0.20);
    border-color: rgba(56, 189, 248, 0.50);
    color: #FFFFFF;
}
.sar-layer-opt:disabled {
    opacity: 0.42;
    cursor: not-allowed;
}
.sar-layer-opt-main {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
}
.sar-layer-swatch {
    width: 14px;
    height: 14px;
    border-radius: 4px;
    flex-shrink: 0;
    border: 1px solid rgba(255, 255, 255, 0.25);
}
.sar-layer-name {
    font-size: 12px;
    font-weight: 600;
    line-height: 1.2;
}
.sar-layer-sub {
    font-size: 10.5px;
    color: #94A3B8;
    line-height: 1.2;
    margin-top: 2px;
}
.sar-layer-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.14);
    color: #94A3B8;
    white-space: nowrap;
}
.sar-res-popover {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    width: 280px;
    padding: 12px;
    border-radius: 14px;
    display: none;
    flex-direction: column;
    gap: 8px;
    z-index: 1250;
}
.sar-res-popover.open {
    display: flex;
}
.sar-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}
.sar-mini-chip {
    padding: 5px 9px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(15, 23, 42, 0.72);
    color: #CBD5E1;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
}
.sar-mini-chip:hover {
    border-color: rgba(56, 189, 248, 0.5);
    color: #FFFFFF;
}
.sar-mini-chip.active {
    background: rgba(14, 165, 233, 0.25);
    border-color: rgba(56, 189, 248, 0.65);
    color: #38BDF8;
}
.sar-pop-meta {
    font-size: 11px;
    color: #94A3B8;
    line-height: 1.3;
}
.sar-pop-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.10);
    margin: 2px 0;
}
.sar-btn-full {
    width: 100%;
    padding: 8px 12px !important;
}
.sar-colorbar {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: #E2E8F0;
}
.sar-colorbar-ramp {
    width: 115px;
    height: 8px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.22);
}
.sar-tf-popover {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    width: 268px;
    padding: 12px;
    border-radius: 14px;
    display: none;
    flex-direction: column;
    gap: 8px;
    z-index: 1200;
}
input[type="date"].sar-select {
    color-scheme: dark;
}
.sar-tf-popover.open {
    display: flex;
}
.sar-pop-title {
    font-size: 12px;
    font-weight: 700;
    color: #F8FAFC;
}
.sar-pop-label {
    font-size: 11px;
    color: #94A3B8;
    margin-bottom: -4px;
}
.sar-select {
    width: 100%;
    padding: 6px 8px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(15, 23, 42, 0.9);
    color: #F8FAFC;
    font-size: 12.5px;
    outline: none;
}
.sar-pop-actions {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-top: 4px;
}
.sar-btn-ghost {
    flex: 1;
    padding: 6px 10px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: transparent;
    color: #CBD5E1;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
}
.sar-btn-primary {
    flex: 1;
    padding: 6px 10px;
    border-radius: 8px;
    border: none;
    background: #0EA5E9;
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
}
.sar-btn-primary:hover {
    background: #38BDF8;
}

/* AOI Status Chip */
.sar-aoi-status {
    position: absolute;
    top: 66px;
    left: 14px;
    z-index: 800;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 11.5px;
    color: #E2E8F0;
    pointer-events: none;
}
.sar-dot-live {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22C55E;
    box-shadow: 0 0 8px #22C55E;
}
.sar-dot-live.busy {
    background: #38BDF8;
    box-shadow: 0 0 10px #38BDF8;
}
.sar-sep {
    color: #475569;
}
.sar-hint-muted {
    color: #94A3B8;
}

/* Bottom-Right Controls */
.sar-map-controls {
    position: absolute;
    right: 14px;
    bottom: 28px;
    z-index: 850;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 8px;
}

.sar-basemap-btn {
    padding: 6px 11px;
    border-radius: 10px;
    color: #E2E8F0;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    position: relative;
}
.sar-basemap-btn:hover {
    border-color: rgba(56, 189, 248, 0.45);
}

.sar-zoom-stack {
    display: flex;
    flex-direction: column;
    border-radius: 12px;
    overflow: hidden;
}

.sar-ctrl-btn {
    position: relative;
    width: 34px;
    height: 34px;
    border: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    background: transparent;
    color: #E2E8F0;
    font-size: 16px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
}
.sar-ctrl-btn:last-child {
    border-bottom: none;
}
.sar-ctrl-btn:hover {
    background: rgba(56, 189, 248, 0.18);
    color: #FFFFFF;
}
.sar-ctrl-btn svg {
    width: 15px;
    height: 15px;
}

/* Bottom Timeline & Playback Dock */
.sar-hud-bottom {
    position: absolute;
    left: 50%;
    bottom: 14px;
    transform: translateX(-50%);
    z-index: 900;
    width: min(760px, calc(100% - 170px));
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    pointer-events: none;
}
.sar-hud-bottom > * {
    pointer-events: auto;
}

.sar-legend-strip {
    display: inline-flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11px;
    color: #CBD5E1;
}
.sar-leg-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.sar-leg-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    display: inline-block;
}

.sar-play-dock {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 14px;
    border-radius: 16px;
}

.sar-transport {
    display: flex;
    align-items: center;
    gap: 4px;
}

.sar-tbtn {
    position: relative;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: #94A3B8;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
}
.sar-tbtn:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #F8FAFC;
}
.sar-tbtn svg {
    width: 15px;
    height: 15px;
}

.sar-play-btn {
    position: relative;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    border: none;
    background: linear-gradient(135deg, #0EA5E9, #38BDF8);
    color: #07111F;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.45);
    transition: transform 0.12s ease;
}
.sar-play-btn:hover {
    transform: scale(1.07);
}
.sar-play-btn svg {
    width: 17px;
    height: 17px;
}

.sar-scrubber-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 140px;
}

.sar-scrubber {
    width: 100%;
    height: 4px;
    border-radius: 999px;
    appearance: none;
    background: rgba(148, 163, 184, 0.28);
    outline: none;
    cursor: pointer;
}
.sar-scrubber::-webkit-slider-thumb {
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #38BDF8;
    border: 2px solid #F8FAFC;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.7);
}

.sar-frame-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11px;
}
.sar-date-pill {
    font-weight: 700;
    color: #F8FAFC;
}
.sar-meta-text {
    color: #94A3B8;
}

.sar-dock-right {
    display: flex;
    align-items: center;
    gap: 6px;
}
.sar-chip-btn {
    position: relative;
    padding: 4px 8px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(15, 23, 42, 0.65);
    color: #94A3B8;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
}
.sar-chip-btn.active {
    color: #38BDF8;
    border-color: rgba(56, 189, 248, 0.45);
    background: rgba(14, 165, 233, 0.15);
}

.sar-area-readout {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    padding-left: 6px;
    border-left: 1px solid rgba(255, 255, 255, 0.10);
}
.sar-area-val {
    font-size: 12.5px;
    font-weight: 700;
    color: #38BDF8;
    line-height: 1.1;
}
.sar-area-sub {
    font-size: 10px;
    color: #64748B;
}

/* Crisp custom hover tooltips */
[data-tooltip]:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    left: 50%;
    top: calc(100% + 8px);
    transform: translateX(-50%);
    white-space: nowrap;
    padding: 5px 9px;
    border-radius: 7px;
    background: rgba(2, 6, 23, 0.94);
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: #F8FAFC;
    font-size: 11px;
    font-weight: 500;
    z-index: 2000;
    pointer-events: none;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
}

.sar-hud-bottom [data-tooltip]:hover::after,
.sar-map-controls [data-tooltip]:hover::after {
    top: auto;
    bottom: calc(100% + 8px);
}

.sar-map-controls [data-tooltip]:hover::after {
    left: auto;
    right: 0;
    transform: none;
}

@media (max-width: 960px) {
    .sar-hud-bottom {
        width: calc(100% - 28px);
    }
    .sar-map-controls {
        bottom: 118px;
    }
    .sar-aoi-status {
        display: none;
    }
}
"""

_COMPONENT_JS = """
const CLASS_RGBA = {
    1: [14, 165, 233, 210],   // Water
    2: [34, 197, 94, 205],    // Tree Lines
    3: [20, 184, 166, 200],   // Wetlands
    4: [234, 179, 8, 175],    // Cropland
    5: [249, 115, 22, 215],   // Urban
};

const CLASS_META = {
    1: { name: 'Water', color: '#0EA5E9' },
    2: { name: 'Tree Lines', color: '#22C55E' },
    3: { name: 'Wetlands', color: '#14B8A6' },
    4: { name: 'Cropland', color: '#EAB308' },
    5: { name: 'Urban', color: '#F97316' },
};

const CMAP_STOPS = {
    terrain: [[51, 51, 153], [0, 152, 254], [0, 204, 102], [254, 254, 153], [128, 92, 84], [248, 248, 250]],
    gray: [[14, 14, 18], [132, 132, 136], [248, 248, 250]],
    viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    magma: [[8, 6, 26], [89, 22, 110], [183, 55, 121], [251, 136, 97], [252, 253, 191]],
    twilight: [[226, 217, 226], [100, 127, 173], [47, 20, 54], [174, 83, 73], [226, 217, 226]],
    coolwarm: [[59, 76, 192], [141, 176, 254], [221, 221, 221], [244, 154, 123], [180, 4, 38]],
};

function buildLut(stops) {
    const lut = new Uint8Array(256 * 3);
    const m = stops.length - 1;
    for (let i = 0; i < 256; i++) {
        const t = (i / 255) * m;
        const idx = Math.min(m - 1, Math.floor(t));
        const frac = t - idx;
        const a = stops[idx], b = stops[idx + 1];
        lut[i * 3] = Math.round(a[0] + (b[0] - a[0]) * frac);
        lut[i * 3 + 1] = Math.round(a[1] + (b[1] - a[1]) * frac);
        lut[i * 3 + 2] = Math.round(a[2] + (b[2] - a[2]) * frac);
    }
    return lut;
}

const CMAP_LUTS = Object.fromEntries(
    Object.entries(CMAP_STOPS).map(([k, stops]) => [k, buildLut(stops)])
);

function cmapCssGradient(cmap) {
    if (cmap === 'classes') return 'linear-gradient(90deg, #0EA5E9, #22C55E, #14B8A6, #EAB308, #F97316)';
    if (cmap === 'change') return 'linear-gradient(90deg, #F43F5E, #38BDF8, #FACC15)';
    if (cmap === 'rgb') return 'linear-gradient(90deg, #EF4444, #22C55E, #0EA5E9)';
    if (cmap === 'dswx') return 'linear-gradient(90deg, #0284C7, #14B8A6, #38BDF8)';
    const stops = CMAP_STOPS[cmap] || CMAP_STOPS.gray;
    return `linear-gradient(90deg, ${stops.map((c) => `rgb(${c[0]},${c[1]},${c[2]})`).join(', ')})`;
}

const BASEMAP_DEFS = [
    {
        id: 'satellite',
        labelKey: 'map_bm_sat',
        label: 'Satellite',
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr: 'Tiles &copy; Esri &amp; GIS Community',
        darkFilter: false,
    },
    {
        id: 'dark',
        labelKey: 'map_bm_dark',
        label: 'Dark',
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        attr: 'Tiles &copy; Esri',
        darkFilter: true,
    },
    {
        id: 'map',
        labelKey: 'map_bm_street',
        label: 'Street',
        url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr: '&copy; OpenStreetMap contributors',
        darkFilter: false,
    },
    {
        id: 'topo',
        labelKey: 'map_bm_topo',
        label: 'Terrain',
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr: 'Tiles &copy; Esri',
        darkFilter: false,
    },
];

const SPEED_STEPS = [
    { id: 'pass', stepDays: 0, intervalMs: 850, labelKey: 'map_speed_pass', shortKey: 'map_speed_pass_short', minSpan: 0, maxSpan: 999999 },
    { id: '1d', stepDays: 1, intervalMs: 650, labelKey: 'map_speed_1d', shortKey: 'map_speed_1d_short', minSpan: 2, maxSpan: 120 },
    { id: '1w', stepDays: 7, intervalMs: 680, labelKey: 'map_speed_1w', shortKey: 'map_speed_1w_short', minSpan: 10, maxSpan: 999999 },
    { id: '2w', stepDays: 14, intervalMs: 680, labelKey: 'map_speed_2w', shortKey: 'map_speed_2w_short', minSpan: 28, maxSpan: 999999 },
    { id: '1m', stepDays: 30, intervalMs: 680, labelKey: 'map_speed_1m', shortKey: 'map_speed_1m_short', minSpan: 45, maxSpan: 999999 },
    { id: '3m', stepDays: 91, intervalMs: 680, labelKey: 'map_speed_3m', shortKey: 'map_speed_3m_short', minSpan: 150, maxSpan: 999999 },
    { id: '6m', stepDays: 182, intervalMs: 680, labelKey: 'map_speed_6m', shortKey: 'map_speed_6m_short', minSpan: 300, maxSpan: 999999 },
    { id: '1y', stepDays: 365, intervalMs: 720, labelKey: 'map_speed_1y', shortKey: 'map_speed_1y_short', minSpan: 500, maxSpan: 999999 },
    { id: '2y', stepDays: 730, intervalMs: 750, labelKey: 'map_speed_2y', shortKey: 'map_speed_2y_short', minSpan: 1100, maxSpan: 999999 },
];

function parseIsoMs(isoStr) {
    if (!isoStr) return 0;
    const ms = Date.parse(`${String(isoStr).slice(0, 10)}T00:00:00Z`);
    return Number.isFinite(ms) ? ms : 0;
}

function fmtIsoDate(ms) {
    return new Date(ms).toISOString().slice(0, 10);
}

function shiftIsoDateDays(isoStr, deltaDays) {
    const ms = parseIsoMs(isoStr);
    return ms ? fmtIsoDate(ms + deltaDays * 86400000) : isoStr;
}

function getAllowedSpeeds(spanDays) {
    return SPEED_STEPS.filter((s) => {
        if (s.stepDays === 0) return true;
        return spanDays >= s.minSpan && spanDays <= s.maxSpan && s.stepDays <= Math.max(1, spanDays * 0.72);
    });
}

function decodeB64Bytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}

function ensureLeafletLoaded() {
    if (window.L) return Promise.resolve(window.L);
    if (window.__sarLeafletPromise) return window.__sarLeafletPromise;

    window.__sarLeafletPromise = new Promise((resolve, reject) => {
        if (!document.getElementById('sar-leaflet-css')) {
            const link = document.createElement('link');
            link.id = 'sar-leaflet-css';
            link.rel = 'stylesheet';
            link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
            document.head.appendChild(link);
        }
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
        script.async = true;
        script.onload = () => resolve(window.L);
        script.onerror = (err) => reject(err);
        document.head.appendChild(script);
    });
    return window.__sarLeafletPromise;
}

function computeBoxKm(bounds) {
    const s = bounds.south, w = bounds.west, n = bounds.north, e = bounds.east;
    const latMid = 0.5 * (s + n);
    const hKm = Math.abs(n - s) * 111.32;
    const wKm = Math.abs(e - w) * 111.32 * Math.max(Math.cos(latMid * Math.PI / 180), 0.15);
    return { wKm, hKm, areaKm2: wKm * hKm };
}

function normalizeBounds(s, w, n, e) {
    let south = Math.min(s, n), north = Math.max(s, n);
    let west = Math.min(w, e), east = Math.max(w, e);
    south = Math.max(-83, Math.min(83, south));
    north = Math.max(-83, Math.min(83, north));
    west = Math.max(-179.8, Math.min(179.8, west));
    east = Math.max(-179.8, Math.min(179.8, east));
    if (north - south < 0.012) north = Math.min(83, south + 0.012);
    if (east - west < 0.012) east = Math.min(179.8, west + 0.012);
    if (north - south > 1.5) north = south + 1.5;
    if (east - west > 1.5) east = west + 1.5;
    return { south, west, north, east };
}

export default function(component) {
    const { data, parentElement, setTriggerValue } = component;
    if (!data) return;

    ensureLeafletLoaded().then((L) => {
        const root = parentElement.querySelector('#sar-studio-root');
        if (!root) return;

        let st = parentElement._sarState;
        if (!st) {
            st = {
                map: null,
                tileLayer: null,
                basemapIdx: 0,
                overlay: null,
                aoiRect: null,
                handles: {},
                boxBadgeMarker: null,
                canvas: document.createElement('canvas'),
                ctx: null,
                frames: [],
                decoded: [],
                frameIdx: 0,
                playing: false,
                playTimer: null,
                speedId: '1m',
                allowedSpeeds: [SPEED_STEPS[0]],
                virtualDateMs: null,
                spanDays: 0,
                loop: true,
                isFullscreen: false,
                lang: data.lang || 'en',
                i18n: data.i18n || {},
                sourceLimits: data.source_limits || { min_date: '2016-01-01', max_date: '2026-11-14' },
                selectedElements: new Set(data.selected_elements || [1, 2, 3]),
                viewMode: data.view_mode || 'elements',
                sarBand: data.sar_band || 'False-Color Dual-Pol (HH/HV)',
                opacity: data.opacity ?? 0.78,
                showGraph: Boolean(data.show_graph),
                activeLayer: data.active_layer || null,
                staticLayer: null,
                resolutionM: (data.region && data.region.resolution_m) || 30,
                aoiBounds: null,
                lastRegionId: null,
                lastModeId: null,
                seq: 1,
                searchTimer: null,
                interacting: false,
            };
            st.t = (key, fallback = '') => (st.i18n && st.i18n[key]) || fallback || key;
            st.ctx = st.canvas.getContext('2d');
            parentElement._sarState = st;

            const mapEl = root.querySelector('#sar-leaflet-map');
            const reg = data.region;
            st.map = L.map(mapEl, {
                center: [reg.center_lat, reg.center_lon],
                zoom: 12,
                zoomControl: false,
                attributionControl: true,
            });

            const setBasemap = (idx) => {
                st.basemapIdx = idx % BASEMAP_DEFS.length;
                const def = BASEMAP_DEFS[st.basemapIdx];
                if (st.tileLayer) st.map.removeLayer(st.tileLayer);
                st.tileLayer = L.tileLayer(def.url, { attribution: def.attr, maxZoom: 19 }).addTo(st.map);
                root.classList.toggle('sar-dark-tiles', def.darkFilter);
                root.querySelector('#sar-basemap-btn').textContent = st.t(def.labelKey, def.label);
            };
            setBasemap(0);

            const emitAoi = (b, label = null) => {
                const dot = root.querySelector('#sar-dot-live');
                if (dot) dot.classList.add('busy');
                setTriggerValue('aoi_bounds', {
                    south: Number(b.south.toFixed(4)),
                    west: Number(b.west.toFixed(4)),
                    north: Number(b.north.toFixed(4)),
                    east: Number(b.east.toFixed(4)),
                    label: label,
                    seq: ++st.seq,
                });
            };

            const syncBoxVisuals = (b, updateOverlayBounds = true) => {
                st.aoiBounds = b;
                const lBounds = [[b.south, b.west], [b.north, b.east]];
                st.aoiRect.setBounds(lBounds);
                if (updateOverlayBounds && st.overlay) {
                    st.overlay.setBounds(lBounds);
                }
                st.handles.nw.setLatLng([b.north, b.west]);
                st.handles.ne.setLatLng([b.north, b.east]);
                st.handles.sw.setLatLng([b.south, b.west]);
                st.handles.se.setLatLng([b.south, b.east]);

                const { wKm, hKm, areaKm2 } = computeBoxKm(b);
                const text = `${wKm.toFixed(1)} × ${hKm.toFixed(1)} km · ${areaKm2.toFixed(0)} km²`;
                root.querySelector('#sar-aoi-dims').textContent = text;
                if (st.boxBadgeMarker) {
                    st.boxBadgeMarker.setLatLng([b.north, 0.5 * (b.west + b.east)]);
                    const el = st.boxBadgeMarker.getElement();
                    if (el) {
                        const span = el.querySelector('.sar-box-badge-text');
                        if (span) span.textContent = text;
                    }
                }
            };

            // Create Observation Square + 4 Corner Resize Handles
            const initB = { south: reg.south, west: reg.west, north: reg.north, east: reg.east };
            st.aoiRect = L.rectangle([[initB.south, initB.west], [initB.north, initB.east]], {
                color: '#38BDF8',
                weight: 2,
                dashArray: '6 4',
                fillColor: '#38BDF8',
                fillOpacity: 0.04,
                interactive: true,
                className: 'sar-aoi-rect',
            }).addTo(st.map);

            st.boxBadgeMarker = L.marker([initB.north, 0.5 * (initB.west + initB.east)], {
                interactive: false,
                icon: L.divIcon({
                    className: 'sar-box-badge-wrap',
                    html: `<div class="sar-box-badge"><span class="sar-box-badge-text">--</span></div>`,
                    iconSize: [0, 0],
                }),
            }).addTo(st.map);

            const cornerDefs = [
                ['nw', (b, ll) => normalizeBounds(b.south, ll.lng, ll.lat, b.east)],
                ['ne', (b, ll) => normalizeBounds(b.south, b.west, ll.lat, ll.lng)],
                ['sw', (b, ll) => normalizeBounds(ll.lat, ll.lng, b.north, b.east)],
                ['se', (b, ll) => normalizeBounds(ll.lat, b.west, b.north, ll.lng)],
            ];

            cornerDefs.forEach(([key, resizeFn]) => {
                const mk = L.marker([initB.north, initB.west], {
                    draggable: true,
                    icon: L.divIcon({
                        className: `sar-corner-handle sar-corner-${key}`,
                        iconSize: [14, 14],
                        iconAnchor: [7, 7],
                    }),
                    zIndexOffset: 500,
                }).addTo(st.map);

                mk.on('dragstart', () => {
                    st.interacting = true;
                    st._anchorBounds = { ...st.aoiBounds };
                });
                mk.on('drag', (ev) => {
                    const nextB = resizeFn(st._anchorBounds, ev.target.getLatLng());
                    syncBoxVisuals(nextB, true);
                });
                mk.on('dragend', () => {
                    st.interacting = false;
                    emitAoi(st.aoiBounds, null);
                });
                st.handles[key] = mk;
            });

            // Drag inside the observation square to move it anywhere
            st.aoiRect.on('mousedown', (e) => {
                L.DomEvent.stopPropagation(e);
                st.interacting = true;
                st.map.dragging.disable();
                const startLatLng = e.latlng;
                const origB = { ...st.aoiBounds };
                let moved = false;

                const onMove = (me) => {
                    const dLat = me.latlng.lat - startLatLng.lat;
                    const dLng = me.latlng.lng - startLatLng.lng;
                    if (Math.abs(dLat) > 0.0004 || Math.abs(dLng) > 0.0004) moved = true;
                    const nextB = normalizeBounds(
                        origB.south + dLat,
                        origB.west + dLng,
                        origB.north + dLat,
                        origB.east + dLng
                    );
                    syncBoxVisuals(nextB, true);
                };
                const onUp = () => {
                    st.map.off('mousemove', onMove);
                    st.map.off('mouseup', onUp);
                    window.removeEventListener('mouseup', onUp);
                    st.map.dragging.enable();
                    st.interacting = false;
                    if (moved) emitAoi(st.aoiBounds, null);
                };
                st.map.on('mousemove', onMove);
                st.map.on('mouseup', onUp);
                window.addEventListener('mouseup', onUp);
            });

            st.syncBoxVisuals = syncBoxVisuals;
            st.emitAoi = emitAoi;

            // --- Frame Canvas Renderer ---
            st.renderFrame = () => {
                if (!st.decoded.length) return;
                const idx = Math.max(0, Math.min(st.frameIdx, st.decoded.length - 1));
                st.frameIdx = idx;
                const frame = st.decoded[idx];
                const baseFrame = st.decoded[0];
                const w = frame.cols, h = frame.rows;
                if (st.canvas.width !== w || st.canvas.height !== h) {
                    st.canvas.width = w;
                    st.canvas.height = h;
                }

                const imgData = st.ctx.createImageData(w, h);
                const px = imgData.data;
                const n = w * h;
                const mask = frame.mask;
                const baseMask = baseFrame.mask;
                const hh = frame.hh;
                const hv = frame.hv;
                const sel = st.selectedElements;
                const act = st.activeLayer || { mode_id: st.viewMode, colormap: 'classes', vmin: 0, vmax: 1, unit: '', is_static: false };

                const isBuiltIn = st.viewMode === 'elements' || st.viewMode === 'change' || st.viewMode === 'backscatter';

                if (st.viewMode === 'backscatter') {
                    for (let i = 0; i < n; i++) {
                        const o = i * 4;
                        if (st.sarBand === 'HH (dB)') {
                            const v = hh[i];
                            px[o] = v; px[o + 1] = v; px[o + 2] = Math.min(255, v + 30); px[o + 3] = 225;
                        } else if (st.sarBand === 'HV (dB)') {
                            const v = hv[i];
                            px[o] = Math.floor(v * 0.45); px[o + 1] = v; px[o + 2] = Math.floor(v * 0.75); px[o + 3] = 225;
                        } else {
                            const r = hh[i];
                            const g = hv[i];
                            const b = Math.max(0, Math.min(255, Math.floor((r - g * 0.65) + 60)));
                            px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = 225;
                        }
                    }
                } else if (st.viewMode === 'change') {
                    for (let i = 0; i < n; i++) {
                        const o = i * 4;
                        const wasActive = sel.has(baseMask[i]);
                        const isActive = sel.has(mask[i]);
                        if (!wasActive && isActive) {
                            px[o] = 244; px[o + 1] = 63; px[o + 2] = 94; px[o + 3] = 225;
                        } else if (wasActive && !isActive) {
                            px[o] = 250; px[o + 1] = 204; px[o + 2] = 21; px[o + 3] = 225;
                        } else if (wasActive && isActive) {
                            px[o] = 56; px[o + 1] = 189; px[o + 2] = 248; px[o + 3] = 115;
                        } else {
                            px[o + 3] = 0;
                        }
                    }
                } else if (st.viewMode === 'elements') {
                    for (let i = 0; i < n; i++) {
                        const cid = mask[i];
                        const o = i * 4;
                        if (sel.has(cid) && CLASS_RGBA[cid]) {
                            const c = CLASS_RGBA[cid];
                            px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = c[3];
                        } else {
                            px[o + 3] = 0;
                        }
                    }
                } else {
                    // Auxiliary NASA / Terrain / Multi-Sensor Raster Layer
                    const srcBytes = act.is_static ? st.staticLayer : frame.layer;
                    if (srcBytes && srcBytes.length === n) {
                        if (act.colormap === 'dswx') {
                            for (let i = 0; i < n; i++) {
                                const v = srcBytes[i];
                                const o = i * 4;
                                if (v === 0) { px[o + 3] = 0; continue; }
                                const clsVal = Math.round(((v - 1) / 254.0) * (act.vmax - act.vmin) + act.vmin);
                                if (clsVal <= 0) { px[o + 3] = 0; }
                                else if (clsVal === 1) { px[o] = 2; px[o + 1] = 132; px[o + 2] = 199; px[o + 3] = 220; }
                                else if (clsVal === 2) { px[o] = 20; px[o + 1] = 184; px[o + 2] = 166; px[o + 3] = 210; }
                                else { px[o] = 226; px[o + 1] = 232; px[o + 2] = 240; px[o + 3] = 195; }
                            }
                        } else {
                            const lut = CMAP_LUTS[act.colormap] || CMAP_LUTS.gray;
                            for (let i = 0; i < n; i++) {
                                const v = srcBytes[i];
                                const o = i * 4;
                                if (v === 0) {
                                    px[o + 3] = 0;
                                } else {
                                    const li = Math.min(255, Math.max(0, v - 1)) * 3;
                                    px[o] = lut[li];
                                    px[o + 1] = lut[li + 1];
                                    px[o + 2] = lut[li + 2];
                                    px[o + 3] = 230;
                                }
                            }
                        }
                    }
                }

                st.ctx.putImageData(imgData, 0, 0);
                const dataUrl = st.canvas.toDataURL();
                const lBounds = [[st.aoiBounds.south, st.aoiBounds.west], [st.aoiBounds.north, st.aoiBounds.east]];
                if (!st.overlay) {
                    st.overlay = L.imageOverlay(dataUrl, lBounds, { opacity: st.opacity, interactive: false }).addTo(st.map);
                } else {
                    st.overlay.setUrl(dataUrl);
                    st.overlay.setBounds(lBounds);
                    st.overlay.setOpacity(st.opacity);
                }

                // Update Timeline Dock Metadata
                const scrubber = root.querySelector('#sar-scrubber');
                scrubber.max = Math.max(0, st.decoded.length - 1);
                scrubber.value = String(idx);
                const activeSpd = (st.allowedSpeeds || []).find((s) => s.id === st.speedId) || SPEED_STEPS[0];
                const clockTag = (st.playing && activeSpd.stepDays > 0 && st.virtualDateMs)
                    ? `${fmtIsoDate(st.virtualDateMs)} · ${frame.date}`
                    : frame.date;
                root.querySelector('#sar-active-date').textContent = act.is_static
                    ? `${act.short_label} (${st.t('map_static_tag', 'Static')})`
                    : clockTag;
                const fw = st.t('map_frame_word', 'Frame');
                if (!isBuiltIn && !act.is_static && !frame.has_layer) {
                    root.querySelector('#sar-frame-info').textContent =
                        `${fw} ${idx + 1}/${st.decoded.length} · ${st.t('map_no_pass_word', 'No acquisition on')} ${frame.date}`;
                } else {
                    root.querySelector('#sar-frame-info').textContent =
                        `${fw} ${idx + 1}/${st.decoded.length} · ${frame.cloud_pct}% ${st.t('map_cloud_word', 'optical cloud')} · ${frame.mean_conf_pct}% ${st.t('map_conf_word', 'conf')}`;
                }

                let activeKm2 = 0;
                st.selectedElements.forEach((cid) => {
                    activeKm2 += Number(frame.areas_km2[String(cid)] || 0);
                });
                root.querySelector('#sar-area-val').textContent = `${activeKm2.toFixed(1)} km²`;
                root.querySelector('#sar-area-sub').textContent =
                    st.viewMode === 'change' ? `${st.t('kpi_vs', 'vs')} ${baseFrame.date}` : st.t('map_area_selected', 'selected cover');

                // Update legend strip
                const leg = root.querySelector('#sar-legend-strip');
                if (st.viewMode === 'change') {
                    leg.innerHTML =
                        `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:#F43F5E;"></span>${st.t('map_legend_gained', 'Gained')}</span>` +
                        `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:#FACC15;"></span>${st.t('map_legend_lost', 'Lost')}</span>` +
                        `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:#38BDF8;"></span>${st.t('map_legend_stable', 'Stable')}</span>`;
                } else if (st.viewMode === 'backscatter') {
                    leg.innerHTML =
                        `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:#EF4444;"></span>HH Double-Bounce</span>` +
                        `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:#22C55E;"></span>HV Canopy Volume</span>` +
                        `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:#0EA5E9;"></span>Low Specular Water</span>`;
                } else if (!isBuiltIn) {
                    if (act.colormap === 'dswx') {
                        leg.innerHTML =
                            `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:#0284C7;"></span>${st.t('map_tip_water', 'Open Water')}</span>` +
                            `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:#14B8A6;"></span>${st.t('map_tip_wetland', 'Flooded Veg')}</span>` +
                            `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:#94A3B8;"></span>${act.label}</span>`;
                    } else {
                        const unitStr = act.unit ? ` ${act.unit}` : '';
                        leg.innerHTML =
                            `<div class="sar-colorbar">` +
                            `<span>${Number(act.vmin).toFixed(1)}${unitStr}</span>` +
                            `<span class="sar-colorbar-ramp" style="background:${cmapCssGradient(act.colormap)};"></span>` +
                            `<span>${Number(act.vmax).toFixed(1)}${unitStr}</span>` +
                            `<span class="sar-sep">&middot;</span>` +
                            `<strong>${act.label}</strong>` +
                            `</div>`;
                    }
                } else {
                    const clsNames = {
                        1: st.lang === 'de' ? 'Wasser' : 'Water',
                        2: st.lang === 'de' ? 'Baumreihen' : 'Tree Lines',
                        3: st.lang === 'de' ? 'Feuchtgebiet' : 'Wetlands',
                        4: st.lang === 'de' ? 'Ackerland' : 'Cropland',
                        5: st.lang === 'de' ? 'Siedlung' : 'Urban',
                    };
                    const items = Array.from(st.selectedElements).sort().map((cid) => {
                        const m = CLASS_META[cid];
                        const nm = clsNames[cid] || m.name;
                        return `<span class="sar-leg-item"><span class="sar-leg-dot" style="background:${m.color};"></span>${nm}</span>`;
                    });
                    leg.innerHTML = items.length ? items.join('') : `<span class="sar-hint-muted">${st.t('map_legend_none', 'No elements selected')}</span>`;
                }
            };

            const selectViewMode = (nextMode) => {
                const modeFilterPop = root.querySelector('#sar-mode-popover');
                if (modeFilterPop) modeFilterPop.classList.remove('open');
                st.viewMode = nextMode;
                root.querySelectorAll('.sar-seg-btn').forEach((b) =>
                    b.classList.toggle('active', b.getAttribute('data-mode') === nextMode)
                );
                const filterBtn = root.querySelector('#sar-mode-filter-btn');
                if (filterBtn) {
                    const isCustomFilter = nextMode !== 'elements' && nextMode !== 'change';
                    filterBtn.classList.toggle('active', isCustomFilter);
                }
                if (nextMode === 'elements' || nextMode === 'change' || nextMode === 'backscatter') {
                    st.renderFrame();
                } else {
                    const dot = root.querySelector('#sar-dot-live');
                    if (dot) dot.classList.add('busy');
                }
                setTriggerValue('view_mode', { mode: nextMode, seq: ++st.seq });
            };
            st.selectViewMode = selectViewMode;

            // --- Element Toggle Buttons ---
            root.querySelectorAll('.sar-elem-btn').forEach((btn) => {
                btn.style.setProperty('--elem-color', btn.getAttribute('data-color'));
                btn.onclick = () => {
                    const cid = Number(btn.getAttribute('data-cid'));
                    if (st.selectedElements.has(cid)) {
                        if (st.selectedElements.size > 1) st.selectedElements.delete(cid);
                    } else {
                        st.selectedElements.add(cid);
                    }
                    syncElementButtons();
                    if (st.viewMode !== 'elements' && st.viewMode !== 'change') {
                        st.selectViewMode('elements');
                    } else {
                        st.renderFrame();
                    }
                    setTriggerValue('selected_elements', {
                        elements: Array.from(st.selectedElements).sort(),
                        seq: ++st.seq,
                    });
                };
            });

            function syncElementButtons() {
                root.querySelectorAll('.sar-elem-btn').forEach((btn) => {
                    const cid = Number(btn.getAttribute('data-cid'));
                    btn.classList.toggle('active', st.selectedElements.has(cid));
                });
            }
            st.syncElementButtons = syncElementButtons;

            // --- Quick AI Segmented Switch ---
            root.querySelectorAll('.sar-seg-btn').forEach((btn) => {
                btn.onclick = () => st.selectViewMode(btn.getAttribute('data-mode'));
            });

            // --- Data Mode Filter Popover Button ---
            const modeFilterBtn = root.querySelector('#sar-mode-filter-btn');
            const modeFilterPop = root.querySelector('#sar-mode-popover');
            modeFilterBtn.onclick = (e) => {
                e.stopPropagation();
                root.querySelector('#sar-tf-popover').classList.remove('open');
                root.querySelector('#sar-res-popover').classList.remove('open');
                modeFilterPop.classList.toggle('open');
            };

            // --- Resolution, Area Size & NASA Fetch Popover ---
            const resBtn = root.querySelector('#sar-res-btn');
            const resPop = root.querySelector('#sar-res-popover');
            resBtn.onclick = (e) => {
                e.stopPropagation();
                root.querySelector('#sar-tf-popover').classList.remove('open');
                modeFilterPop.classList.remove('open');
                resPop.classList.toggle('open');
            };

            root.querySelectorAll('#sar-res-chips .sar-mini-chip').forEach((chip) => {
                chip.onclick = () => {
                    const resVal = Number(chip.getAttribute('data-res'));
                    resPop.classList.remove('open');
                    const dot = root.querySelector('#sar-dot-live');
                    if (dot) dot.classList.add('busy');
                    setTriggerValue('resolution', { resolution_m: resVal, seq: ++st.seq });
                };
            });

            root.querySelectorAll('#sar-size-chips .sar-mini-chip').forEach((chip) => {
                chip.onclick = () => {
                    const km = Number(chip.getAttribute('data-km'));
                    if (!st.aoiBounds) return;
                    const cLat = 0.5 * (st.aoiBounds.south + st.aoiBounds.north);
                    const cLon = 0.5 * (st.aoiBounds.west + st.aoiBounds.east);
                    const dLat = (km * 0.5) / 111.32;
                    const dLon = (km * 0.5) / (111.32 * Math.max(Math.cos(cLat * Math.PI / 180), 0.15));
                    const nextB = normalizeBounds(cLat - dLat, cLon - dLon, cLat + dLat, cLon + dLon);
                    resPop.classList.remove('open');
                    syncBoxVisuals(nextB, true);
                    st.map.fitBounds([[nextB.south, nextB.west], [nextB.north, nextB.east]], { padding: [90, 90], maxZoom: 12 });
                    emitAoi(nextB, null);
                };
            });

            root.querySelector('#sar-nasa-fetch-btn').onclick = () => {
                if (!st.aoiBounds) return;
                resPop.classList.remove('open');
                const dot = root.querySelector('#sar-dot-live');
                if (dot) dot.classList.add('busy');
                setTriggerValue('nasa_fetch', {
                    south: Number(st.aoiBounds.south.toFixed(4)),
                    west: Number(st.aoiBounds.west.toFixed(4)),
                    north: Number(st.aoiBounds.north.toFixed(4)),
                    east: Number(st.aoiBounds.east.toFixed(4)),
                    resolution_m: st.resolutionM || 30,
                    start_date: root.querySelector('#sar-tf-start').value || '2025-01-01',
                    end_date: root.querySelector('#sar-tf-end').value || '2026-11-14',
                    seq: ++st.seq,
                });
            };

            // --- Dynamic Time-Step Playback Controls ---
            const findValidIdx = (fromIdx, step) => {
                const len = st.decoded.length;
                if (!len) return 0;
                const act = st.activeLayer;
                const filterSparse = act && !act.is_static && act.mode_id !== 'elements' && act.mode_id !== 'change' && act.mode_id !== 'backscatter';
                if (!filterSparse) {
                    const nxt = fromIdx + step;
                    if (nxt >= len) return st.loop ? 0 : len - 1;
                    if (nxt < 0) return len - 1;
                    return nxt;
                }
                for (let k = 1; k <= len; k++) {
                    const cand = (fromIdx + step * k + len * 2) % len;
                    if (st.decoded[cand] && st.decoded[cand].has_layer) return cand;
                }
                return fromIdx;
            };

            const findClosestValidFrameForMs = (targetMs) => {
                const len = st.decoded.length;
                if (!len) return 0;
                const act = st.activeLayer;
                const filterSparse = act && !act.is_static && act.mode_id !== 'elements' && act.mode_id !== 'change' && act.mode_id !== 'backscatter';
                let bestIdx = -1;
                let bestDiff = Infinity;
                for (let i = 0; i < len; i++) {
                    const f = st.decoded[i];
                    if (filterSparse && !f.has_layer) continue;
                    const diff = Math.abs(parseIsoMs(f.date) - targetMs);
                    if (diff < bestDiff) {
                        bestDiff = diff;
                        bestIdx = i;
                    }
                }
                return bestIdx >= 0 ? bestIdx : 0;
            };

            const syncSpeedUI = () => {
                const allowed = st.allowedSpeeds || [SPEED_STEPS[0]];
                let cur = allowed.find((s) => s.id === st.speedId);
                if (!cur) {
                    cur = allowed[allowed.length - 1];
                    st.speedId = cur.id;
                }
                const speedBtn = root.querySelector('#sar-speed-btn');
                if (speedBtn) {
                    speedBtn.innerHTML = `${st.t(cur.shortKey, cur.id)} &#9662;`;
                }
                const speedPop = root.querySelector('#sar-speed-popover');
                if (speedPop) {
                    const maxSpd = allowed[allowed.length - 1];
                    let html = `<div class="sar-pop-title">${st.t('map_speed_title', 'Playback Step per Frame')}</div>`;
                    html += `<div class="sar-pop-meta">${st.t('map_speed_max', 'Max step')}: <strong>${st.t(maxSpd.shortKey, maxSpd.id)}</strong> (${st.spanDays}d)</div>`;
                    allowed.forEach((spd) => {
                        html += `<button type="button" class="sar-speed-opt ${spd.id === st.speedId ? 'active' : ''}" data-spd="${spd.id}">
                            <span>${st.t(spd.labelKey, spd.id)}</span>
                            <span class="sar-layer-tag">${st.t(spd.shortKey, spd.id)}</span>
                        </button>`;
                    });
                    speedPop.innerHTML = html;
                    speedPop.querySelectorAll('.sar-speed-opt').forEach((btn) => {
                        btn.onclick = (ev) => {
                            ev.stopPropagation();
                            st.speedId = btn.getAttribute('data-spd');
                            speedPop.classList.remove('open');
                            syncSpeedUI();
                            if (st.playing) startPlayback();
                            else st.renderFrame();
                        };
                    });
                }
            };
            st.syncSpeedUI = syncSpeedUI;

            const stopPlayback = () => {
                st.playing = false;
                if (st.playTimer) {
                    clearInterval(st.playTimer);
                    st.playTimer = null;
                }
                root.querySelector('#sar-icon-play').style.display = 'block';
                root.querySelector('#sar-icon-pause').style.display = 'none';
            };

            const startPlayback = () => {
                if (st.decoded.length <= 1) return;
                if (st.frameIdx >= st.decoded.length - 1 && !st.loop) st.frameIdx = 0;
                st.playing = true;
                root.querySelector('#sar-icon-play').style.display = 'none';
                root.querySelector('#sar-icon-pause').style.display = 'block';
                if (st.playTimer) clearInterval(st.playTimer);

                const curSpeed = (st.allowedSpeeds || []).find((s) => s.id === st.speedId) || SPEED_STEPS[0];
                const firstMs = parseIsoMs(st.decoded[0].date);
                const lastMs = parseIsoMs(st.decoded[st.decoded.length - 1].date);
                st.virtualDateMs = parseIsoMs(st.decoded[st.frameIdx].date) || firstMs;

                st.playTimer = setInterval(() => {
                    if (curSpeed.stepDays === 0) {
                        const nextIdx = findValidIdx(st.frameIdx, 1);
                        if (!st.loop && nextIdx <= st.frameIdx) {
                            stopPlayback();
                            return;
                        }
                        st.frameIdx = nextIdx;
                        st.virtualDateMs = parseIsoMs(st.decoded[st.frameIdx].date);
                    } else {
                        const nextMs = st.virtualDateMs + curSpeed.stepDays * 86400000;
                        if (nextMs > lastMs) {
                            if (st.loop) {
                                st.virtualDateMs = firstMs;
                                st.frameIdx = findClosestValidFrameForMs(firstMs);
                            } else {
                                st.virtualDateMs = lastMs;
                                st.frameIdx = findClosestValidFrameForMs(lastMs);
                                st.renderFrame();
                                stopPlayback();
                                return;
                            }
                        } else {
                            st.virtualDateMs = nextMs;
                            st.frameIdx = findClosestValidFrameForMs(nextMs);
                        }
                    }
                    st.renderFrame();
                }, curSpeed.intervalMs);
            };
            st.stopPlayback = stopPlayback;

            root.querySelector('#sar-btn-play').onclick = () => {
                if (st.playing) stopPlayback();
                else startPlayback();
            };
            root.querySelector('#sar-btn-reset').onclick = () => {
                stopPlayback();
                st.frameIdx = 0;
                if (st.decoded[0] && !st.decoded[0].has_layer) {
                    st.frameIdx = findValidIdx(0, 1);
                }
                if (st.decoded[st.frameIdx]) st.virtualDateMs = parseIsoMs(st.decoded[st.frameIdx].date);
                st.renderFrame();
            };
            root.querySelector('#sar-btn-prev').onclick = () => {
                stopPlayback();
                st.frameIdx = Math.max(0, findValidIdx(st.frameIdx, -1));
                if (st.decoded[st.frameIdx]) st.virtualDateMs = parseIsoMs(st.decoded[st.frameIdx].date);
                st.renderFrame();
            };
            root.querySelector('#sar-btn-next').onclick = () => {
                stopPlayback();
                st.frameIdx = Math.min(st.decoded.length - 1, findValidIdx(st.frameIdx, 1));
                if (st.decoded[st.frameIdx]) st.virtualDateMs = parseIsoMs(st.decoded[st.frameIdx].date);
                st.renderFrame();
            };
            root.querySelector('#sar-scrubber').oninput = (e) => {
                stopPlayback();
                st.frameIdx = Number(e.target.value);
                if (st.decoded[st.frameIdx]) st.virtualDateMs = parseIsoMs(st.decoded[st.frameIdx].date);
                st.renderFrame();
            };
            root.querySelector('#sar-speed-btn').onclick = (e) => {
                e.stopPropagation();
                const spPop = root.querySelector('#sar-speed-popover');
                if (spPop) spPop.classList.toggle('open');
            };
            root.querySelector('#sar-loop-btn').onclick = (e) => {
                st.loop = !st.loop;
                e.currentTarget.classList.toggle('active', st.loop);
            };

            // --- Fullscreen Map Toggle ---
            const syncFullscreenUI = () => {
                const fsBtn = root.querySelector('#sar-fullscreen-btn');
                const enterIc = root.querySelector('#sar-fs-enter-icon');
                const exitIc = root.querySelector('#sar-fs-exit-icon');
                root.classList.toggle('sar-is-fullscreen', st.isFullscreen);
                if (fsBtn) {
                    fsBtn.classList.toggle('active', st.isFullscreen);
                    fsBtn.setAttribute('data-tooltip', st.isFullscreen ? st.t('map_fs_exit', 'Exit Fullscreen') : st.t('map_fs_tip', 'Fullscreen Map'));
                }
                if (enterIc) enterIc.style.display = st.isFullscreen ? 'none' : 'block';
                if (exitIc) exitIc.style.display = st.isFullscreen ? 'block' : 'none';
                setTimeout(() => st.map && st.map.invalidateSize(), 110);
                setTimeout(() => st.map && st.map.invalidateSize(), 260);
            };

            root.querySelector('#sar-fullscreen-btn').onclick = () => {
                st.isFullscreen = !st.isFullscreen;
                if (st.isFullscreen) {
                    if (root.requestFullscreen) {
                        root.requestFullscreen().catch(() => {});
                    }
                } else {
                    if (document.fullscreenElement === root && document.exitFullscreen) {
                        document.exitFullscreen().catch(() => {});
                    }
                }
                syncFullscreenUI();
            };

            document.addEventListener('fullscreenchange', () => {
                if (!document.fullscreenElement && st.isFullscreen) {
                    st.isFullscreen = false;
                    syncFullscreenUI();
                }
            });
            document.addEventListener('keydown', (ev) => {
                if (ev.key === 'Escape' && st.isFullscreen && !document.fullscreenElement) {
                    st.isFullscreen = false;
                    syncFullscreenUI();
                }
            });

            // --- Map Controls (Zoom, Center, Basemap) ---
            root.querySelector('#sar-zoom-in').onclick = () => st.map.zoomIn();
            root.querySelector('#sar-zoom-out').onclick = () => st.map.zoomOut();
            root.querySelector('#sar-center-btn').onclick = () => {
                if (st.aoiBounds) {
                    st.map.fitBounds(
                        [[st.aoiBounds.south, st.aoiBounds.west], [st.aoiBounds.north, st.aoiBounds.east]],
                        { padding: [85, 85], maxZoom: 13 }
                    );
                }
            };
            root.querySelector('#sar-basemap-btn').onclick = () => setBasemap(st.basemapIdx + 1);

            // --- Timeframe Popover ---
            const tfBtn = root.querySelector('#sar-tf-btn');
            const tfPop = root.querySelector('#sar-tf-popover');
            tfBtn.onclick = (e) => {
                e.stopPropagation();
                root.querySelector('#sar-mode-popover').classList.remove('open');
                root.querySelector('#sar-res-popover').classList.remove('open');
                tfPop.classList.toggle('open');
            };
            root.querySelectorAll('#sar-tf-presets .sar-mini-chip').forEach((chip) => {
                chip.onclick = () => {
                    const kind = chip.getAttribute('data-tf-preset');
                    const minD = (st.sourceLimits && st.sourceLimits.min_date) || '2016-01-01';
                    const maxD = (st.sourceLimits && st.sourceLimits.max_date) || '2026-11-14';
                    const sEl = root.querySelector('#sar-tf-start');
                    const eEl = root.querySelector('#sar-tf-end');
                    let targetStart = minD;
                    if (kind === '5y') targetStart = shiftIsoDateDays(maxD, -1825);
                    else if (kind === '1y') targetStart = shiftIsoDateDays(maxD, -365);
                    else if (kind === '6m') targetStart = shiftIsoDateDays(maxD, -183);
                    if (targetStart < minD) targetStart = minD;
                    sEl.value = targetStart;
                    eEl.value = maxD;
                };
            });
            root.querySelector('#sar-tf-full').onclick = () => {
                const minD = (st.sourceLimits && st.sourceLimits.min_date) || '2016-01-01';
                const maxD = (st.sourceLimits && st.sourceLimits.max_date) || '2026-11-14';
                root.querySelector('#sar-tf-start').value = minD;
                root.querySelector('#sar-tf-end').value = maxD;
            };
            root.querySelector('#sar-tf-apply').onclick = () => {
                let startVal = root.querySelector('#sar-tf-start').value || st.sourceLimits.min_date;
                let endVal = root.querySelector('#sar-tf-end').value || st.sourceLimits.max_date;
                if (startVal > endVal) {
                    const tmp = startVal;
                    startVal = endVal;
                    endVal = tmp;
                }
                tfPop.classList.remove('open');
                stopPlayback();
                setTriggerValue('timeframe', { start_date: startVal, end_date: endVal, seq: ++st.seq });
            };

            // --- Toggle Graph & Sidebar ---
            root.querySelector('#sar-graph-btn').onclick = () => {
                st.showGraph = !st.showGraph;
                root.querySelector('#sar-graph-btn').classList.toggle('active', st.showGraph);
                setTriggerValue('toggle_graph', { show: st.showGraph, seq: ++st.seq });
            };

            root.querySelector('#sar-sidebar-btn').onclick = () => {
                const sidebarToggle = document.querySelector(
                    '[data-testid="stExpandSidebarButton"] button, [data-testid="collapsedControl"] button, button[aria-label*="sidebar" i], [data-testid="stSidebarCollapseButton"] button'
                );
                if (sidebarToggle) sidebarToggle.click();
            };

            // --- Search Any Area (Presets + Coordinate Parsing + Live Nominatim) ---
            const searchIn = root.querySelector('#sar-search-input');
            const searchDrop = root.querySelector('#sar-search-dropdown');
            const searchClear = root.querySelector('#sar-search-clear');

            const jumpToLocation = (south, west, north, east, label) => {
                searchDrop.style.display = 'none';
                stopPlayback();
                const b = normalizeBounds(south, west, north, east);
                syncBoxVisuals(b, true);
                st.map.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [90, 90], maxZoom: 12 });
                emitAoi(b, label);
            };

            const renderSearchDropdown = (query, remoteResults = []) => {
                const q = query.trim().toLowerCase();
                let html = '';

                // Direct coordinate check: "lat, lon"
                const coordMatch = query.trim().match(/^(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)$/);
                if (coordMatch) {
                    const lat = parseFloat(coordMatch[1]), lon = parseFloat(coordMatch[2]);
                    if (Math.abs(lat) <= 84 && Math.abs(lon) <= 180) {
                        const dLat = 0.07, dLon = 0.09 / Math.max(Math.cos(lat * Math.PI / 180), 0.25);
                        html += `<div class="sar-search-section">${st.t('map_search_coords', 'Coordinates')}</div>`;
                        html += `<button type="button" class="sar-search-item" data-s="${lat - dLat}" data-w="${lon - dLon}" data-n="${lat + dLat}" data-e="${lon + dLon}" data-label="Custom (${lat.toFixed(2)}, ${lon.toFixed(2)})">
                            <strong>${lat.toFixed(4)}, ${lon.toFixed(4)}</strong>
                            <span class="sar-search-sub">${st.t('map_search_goto', 'Place observation square & run neural net')}</span>
                        </button>`;
                    }
                }

                // Matching presets
                const presets = (st.presets || []).filter((p) =>
                    !q || p.name.toLowerCase().includes(q) || p.country.toLowerCase().includes(q)
                );
                if (presets.length) {
                    html += `<div class="sar-search-section">${st.t('map_search_presets', 'Featured NASA Study Areas')}</div>`;
                    presets.forEach((p) => {
                        html += `<button type="button" class="sar-search-item" data-s="${p.south}" data-w="${p.west}" data-n="${p.north}" data-e="${p.east}" data-label="${p.name}">
                            <strong>${p.name}</strong>
                            <span class="sar-search-sub">${p.country} · Multi-date SAR archive</span>
                        </button>`;
                    });
                }

                if (remoteResults.length) {
                    html += `<div class="sar-search-section">${st.t('map_search_global', 'Global Places')}</div>`;
                    remoteResults.forEach((r) => {
                        const lat = parseFloat(r.lat), lon = parseFloat(r.lon);
                        const dLat = 0.075, dLon = 0.095 / Math.max(Math.cos(lat * Math.PI / 180), 0.25);
                        const shortName = r.display_name.split(',').slice(0, 2).join(', ');
                        html += `<button type="button" class="sar-search-item" data-s="${lat - dLat}" data-w="${lon - dLon}" data-n="${lat + dLat}" data-e="${lon + dLon}" data-label="${shortName.replace(/"/g, '&quot;')}">
                            <strong>${shortName}</strong>
                            <span class="sar-search-sub">${r.display_name}</span>
                        </button>`;
                    });
                }

                searchDrop.innerHTML = html;
                searchDrop.style.display = html ? 'block' : 'none';

                searchDrop.querySelectorAll('.sar-search-item').forEach((item) => {
                    item.onclick = () => {
                        jumpToLocation(
                            parseFloat(item.dataset.s),
                            parseFloat(item.dataset.w),
                            parseFloat(item.dataset.n),
                            parseFloat(item.dataset.e),
                            item.dataset.label
                        );
                        searchIn.value = item.dataset.label;
                        searchClear.style.display = 'inline';
                    };
                });
            };

            const triggerRemoteSearch = (q) => {
                if (q.trim().length < 2) {
                    renderSearchDropdown(q, []);
                    return;
                }
                fetch(`https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(q.trim())}`)
                    .then((res) => res.json())
                    .then((rows) => renderSearchDropdown(q, Array.isArray(rows) ? rows : []))
                    .catch(() => renderSearchDropdown(q, []));
            };

            searchIn.onfocus = () => renderSearchDropdown(searchIn.value, []);
            searchIn.oninput = () => {
                searchClear.style.display = searchIn.value ? 'inline' : 'none';
                renderSearchDropdown(searchIn.value, []);
                if (st.searchTimer) clearTimeout(st.searchTimer);
                st.searchTimer = setTimeout(() => triggerRemoteSearch(searchIn.value), 350);
            };
            searchIn.onkeydown = (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    if (st.searchTimer) clearTimeout(st.searchTimer);
                    triggerRemoteSearch(searchIn.value);
                }
            };
            searchClear.onclick = () => {
                searchIn.value = '';
                searchClear.style.display = 'none';
                renderSearchDropdown('', []);
                searchIn.focus();
            };

            document.addEventListener('click', (ev) => {
                const modePop = root.querySelector('#sar-mode-popover');
                const resPopEl = root.querySelector('#sar-res-popover');
                const spPop = root.querySelector('#sar-speed-popover');
                if (!root.contains(ev.target)) {
                    tfPop.classList.remove('open');
                    if (modePop) modePop.classList.remove('open');
                    if (resPopEl) resPopEl.classList.remove('open');
                    if (spPop) spPop.classList.remove('open');
                    searchDrop.style.display = 'none';
                } else {
                    if (!ev.target.closest('.sar-tf-wrap')) tfPop.classList.remove('open');
                    if (modePop && !ev.target.closest('#sar-mode-wrap')) modePop.classList.remove('open');
                    if (resPopEl && !ev.target.closest('#sar-res-wrap')) resPopEl.classList.remove('open');
                    if (spPop && !ev.target.closest('#sar-speed-wrap')) spPop.classList.remove('open');
                    if (!ev.target.closest('.sar-search-wrap')) searchDrop.style.display = 'none';
                }
            });
        }

        // --- Sync Latest Python Data Payload into Persistent Map Component ---
        const reg = data.region;
        st.lang = data.lang || 'en';
        st.i18n = data.i18n || {};
        st.sourceLimits = data.source_limits || { min_date: '2016-01-01', max_date: '2026-11-14' };
        st.presets = data.presets || [];
        st.opacity = data.opacity ?? 0.78;
        st.sarBand = data.sar_band || 'False-Color Dual-Pol (HH/HV)';
        st.showGraph = Boolean(data.show_graph);
        st.viewMode = data.view_mode || st.viewMode || 'elements';
        st.activeLayer = data.active_layer || null;
        st.staticLayer = data.static_layer_b64 ? decodeB64Bytes(data.static_layer_b64) : null;
        st.resolutionM = Number(reg.resolution_m || 30);

        // Apply localized labels & tooltips across static HUD elements
        const setTip = (sel, key, fallback) => {
            const el = root.querySelector(sel);
            if (el) el.setAttribute('data-tooltip', st.t(key, fallback));
        };
        const setTxt = (sel, key, fallback) => {
            const el = root.querySelector(sel);
            if (el) el.textContent = st.t(key, fallback);
        };
        const srchInput = root.querySelector('#sar-search-input');
        if (srchInput) srchInput.placeholder = st.t('map_search_ph', 'Search city, river, or lat, lon...');
        const srchClr = root.querySelector('#sar-search-clear');
        if (srchClr) srchClr.title = st.t('map_clear', 'Clear');
        setTip('.sar-elem-btn[data-cid="1"]', 'map_tip_water', 'Toggle Water');
        setTip('.sar-elem-btn[data-cid="2"]', 'map_tip_tree', 'Toggle Tree Lines');
        setTip('.sar-elem-btn[data-cid="3"]', 'map_tip_wetland', 'Toggle Wetlands');
        setTip('.sar-elem-btn[data-cid="4"]', 'map_tip_crop', 'Toggle Cropland');
        setTip('.sar-elem-btn[data-cid="5"]', 'map_tip_urban', 'Toggle Urban');
        const segEl = root.querySelector('.sar-seg-btn[data-mode="elements"]');
        if (segEl) { segEl.textContent = st.t('map_tab_elements', 'Elements'); segEl.setAttribute('data-tooltip', st.t('map_tab_elements_tip', 'Classified Surface Elements')); }
        const segCh = root.querySelector('.sar-seg-btn[data-mode="change"]');
        if (segCh) { segCh.textContent = st.t('map_tab_changes', 'Changes'); segCh.setAttribute('data-tooltip', st.t('map_tab_changes_tip', 'Temporal Changes')); }
        setTip('#sar-mode-filter-btn', 'map_modes_tip', 'Filter Individual Data Modes');
        setTip('#sar-res-btn', 'map_res_tip', 'Adjust Resolution & Collected Area');
        setTxt('.sar-res-popover .sar-pop-title', 'map_res_title', 'Resolution & Collected Area');
        setTxt('.sar-res-popover label:nth-of-type(1)', 'map_res_target', 'Target Resolution (TARGET_RES_M)');
        setTxt('.sar-res-popover label:nth-of-type(2)', 'map_res_size', 'Observation Square Size');
        setTxt('#sar-nasa-fetch-btn', 'map_res_fetch', 'Fetch Real NASA Data for Square');
        setTip('#sar-tf-btn', 'map_tf_tip', 'Choose Time Frame');
        setTxt('#sar-tf-title', 'map_tf_title', 'Analysis Time Frame');
        setTxt('#sar-tf-start-lbl', 'map_tf_start', 'Start Date');
        setTxt('#sar-tf-end-lbl', 'map_tf_end', 'End Date');
        setTxt('#sar-tf-full', 'map_tf_full', 'Max Source Range');
        setTxt('#sar-tf-apply', 'map_tf_apply', 'Apply');
        setTip('#sar-graph-btn', 'map_graph_tip', 'Toggle Area Time-Frame Graph');
        setTxt('#sar-graph-label', 'map_graph_btn', 'Graph');
        setTip('#sar-sidebar-btn', 'map_model_tip', 'Toggle NASA Collector & Model Sidebar');
        setTxt('#sar-sidebar-label', 'map_model_btn', 'Controls');
        setTip('#sar-fullscreen-btn', st.isFullscreen ? 'map_fs_exit' : 'map_fs_tip', 'Fullscreen Map');
        setTxt('#sar-drag-hint', 'map_drag_hint', 'Drag or resize square to analyze any area');
        setTip('#sar-basemap-btn', 'map_style_tip', 'Switch Map Style');
        root.querySelector('#sar-basemap-btn').textContent = st.t(BASEMAP_DEFS[st.basemapIdx].labelKey, BASEMAP_DEFS[st.basemapIdx].label);
        setTip('#sar-center-btn', 'map_center_tip', 'Center on Observation Square');
        setTip('#sar-zoom-in', 'map_zoom_in', 'Zoom In');
        setTip('#sar-zoom-out', 'map_zoom_out', 'Zoom Out');
        setTip('#sar-btn-reset', 'map_first_tip', 'First Date');
        setTip('#sar-btn-prev', 'map_prev_tip', 'Previous Date');
        setTip('#sar-btn-play', 'map_play_tip', 'Play / Pause Time-Lapse');
        setTip('#sar-btn-next', 'map_next_tip', 'Next Date');
        setTip('#sar-speed-btn', 'map_speed_tip', 'Playback Step');
        setTxt('#sar-loop-btn', 'map_loop_btn', 'Loop');
        setTip('#sar-loop-btn', 'map_loop_tip', 'Loop Playback');

        root.querySelector('#sar-dot-live').classList.remove('busy');
        root.querySelector('#sar-region-title').textContent = reg.name;
        root.querySelector('#sar-graph-btn').classList.toggle('active', st.showGraph);
        root.querySelector('#sar-tf-label').textContent = `${data.start_date} → ${data.end_date}`;
        root.querySelector('#sar-res-label').textContent = `${Math.round(st.resolutionM)} m`;
        const srcTag = reg.source_type === 'nasa_earthdata'
            ? st.t('map_src_nasa', 'NASA Earthdata')
            : st.t('map_src_sim', 'Multi-Sensor Sim');
        root.querySelector('#sar-res-meta').textContent =
            `Grid: ${reg.grid_rows}×${reg.grid_cols} px · ${Math.round(st.resolutionM)} m · ${srcTag} · ${st.t('map_res_drag_hint', 'Drag corners')}`;

        const tokenHint = root.querySelector('#sar-token-hint');
        if (tokenHint) {
            tokenHint.textContent = data.has_earthdata_token
                ? st.t('map_tok_ok', 'EARTHDATA_TOKEN detected · Click to slice real NASA granules')
                : st.t('map_tok_no', 'No EARTHDATA_TOKEN found · Add in .env or Controls sidebar');
        }

        // Highlight active resolution chip
        root.querySelectorAll('#sar-res-chips .sar-mini-chip').forEach((chip) => {
            chip.classList.toggle('active', Math.abs(Number(chip.getAttribute('data-res')) - st.resolutionM) < 1);
        });

        // Build grouped Mode Filter dropdown menu dynamically from data.available_layers
        const modePop = root.querySelector('#sar-mode-popover');
        const filterBtn = root.querySelector('#sar-mode-filter-btn');
        const filterLbl = root.querySelector('#sar-mode-filter-label');
        const layers = data.available_layers || [];
        const groups = {};
        layers.forEach((l) => {
            if (!groups[l.category]) groups[l.category] = [];
            groups[l.category].push(l);
        });
        let menuHtml = '';
        Object.entries(groups).forEach(([cat, items]) => {
            menuHtml += `<div class="sar-layer-group-title">${cat}</div>`;
            items.forEach((item) => {
                const isActive = item.mode_id === st.viewMode;
                const tag = !item.available
                    ? st.t('map_na_tag', 'N/A')
                    : (item.is_static ? st.t('map_static_tag', 'Static') : item.unit);
                menuHtml += `
                    <button type="button" class="sar-layer-opt ${isActive ? 'active' : ''}" data-layer-mode="${item.mode_id}" ${item.available ? '' : 'disabled'}>
                        <div class="sar-layer-opt-main">
                            <span class="sar-layer-swatch" style="background:${cmapCssGradient(item.colormap)};"></span>
                            <div>
                                <div class="sar-layer-name">${item.label}</div>
                                <div class="sar-layer-sub">${item.description}</div>
                            </div>
                        </div>
                        <span class="sar-layer-tag">${tag}</span>
                    </button>`;
            });
        });
        modePop.innerHTML = menuHtml;
        modePop.querySelectorAll('.sar-layer-opt').forEach((optBtn) => {
            optBtn.onclick = () => st.selectViewMode(optBtn.getAttribute('data-layer-mode'));
        });

        const isAux = st.viewMode !== 'elements' && st.viewMode !== 'change';
        filterBtn.classList.toggle('active', isAux);
        filterLbl.textContent = (isAux && st.activeLayer) ? st.activeLayer.short_label : st.t('map_modes_btn', 'Data Modes');

        // Configure Time Frame date inputs bounded by full allowed source range
        const sIn = root.querySelector('#sar-tf-start');
        const eIn = root.querySelector('#sar-tf-end');
        const minLim = st.sourceLimits.min_date || '2016-01-01';
        const maxLim = st.sourceLimits.max_date || '2026-11-14';
        sIn.min = minLim;
        sIn.max = maxLim;
        eIn.min = minLim;
        eIn.max = maxLim;
        sIn.value = data.start_date;
        eIn.value = data.end_date;
        const spanHint = root.querySelector('#sar-tf-span-hint');
        if (spanHint) {
            spanHint.textContent = `${st.t('map_tf_span', 'Source window')}: ${minLim} → ${maxLim}`;
        }

        // Sync selected elements & mode button highlight
        if (!st.interacting) {
            st.selectedElements = new Set(data.selected_elements || [1, 2, 3]);
            st.syncElementButtons();
            root.querySelectorAll('.sar-seg-btn').forEach((b) =>
                b.classList.toggle('active', b.getAttribute('data-mode') === st.viewMode)
            );
        }

        // Decode frame arrays cleanly
        st.decoded = (data.frames || []).map((f) => ({
            ...f,
            mask: decodeB64Bytes(f.mask_b64),
            hh: decodeB64Bytes(f.hh_b64),
            hv: decodeB64Bytes(f.hv_b64),
            layer: f.layer_b64 ? decodeB64Bytes(f.layer_b64) : null,
        }));
        if (st.frameIdx >= st.decoded.length) st.frameIdx = 0;

        // Recompute allowed playback speeds strictly from loaded date span
        if (st.decoded.length >= 2) {
            const firstMs = parseIsoMs(st.decoded[0].date);
            const lastMs = parseIsoMs(st.decoded[st.decoded.length - 1].date);
            st.spanDays = Math.max(1, Math.round((lastMs - firstMs) / 86400000));
        } else {
            st.spanDays = 0;
        }
        st.allowedSpeeds = getAllowedSpeeds(st.spanDays);
        if (!st.allowedSpeeds.some((s) => s.id === st.speedId)) {
            if (st.allowedSpeeds.some((s) => s.id === '1m')) st.speedId = '1m';
            else if (st.allowedSpeeds.some((s) => s.id === '1w')) st.speedId = '1w';
            else st.speedId = st.allowedSpeeds[st.allowedSpeeds.length - 1].id;
        }
        st.syncSpeedUI();

        // If entering a sparse NASA pass layer (e.g. NISAR GCOV or GUNW) and current index has no pass, snap to first available pass
        if (st.lastModeId !== st.viewMode) {
            st.lastModeId = st.viewMode;
            if (st.decoded.length && st.activeLayer && !st.activeLayer.is_static && !st.decoded[st.frameIdx].has_layer) {
                const firstPass = st.decoded.findIndex((f) => f.has_layer);
                if (firstPass >= 0) st.frameIdx = firstPass;
            }
        }

        // Sync observation square bounds and center if region changed
        const newB = { south: reg.south, west: reg.west, north: reg.north, east: reg.east };
        if (!st.interacting) {
            st.syncBoxVisuals(newB, false);
        }
        if (st.lastRegionId !== reg.id) {
            st.lastRegionId = reg.id;
            st.frameIdx = 0;
            if (st.decoded.length && st.activeLayer && !st.activeLayer.is_static && !st.decoded[0].has_layer) {
                const firstPass = st.decoded.findIndex((f) => f.has_layer);
                if (firstPass >= 0) st.frameIdx = firstPass;
            }
            st.map.fitBounds([[newB.south, newB.west], [newB.north, newB.east]], { padding: [90, 90], maxZoom: 12 });
        }

        st.renderFrame();
        setTimeout(() => st.map && st.map.invalidateSize(), 120);
    });
}
"""

sar_map_component = st.components.v2.component(
    "sar_map_explorer",
    html=_COMPONENT_HTML,
    css=_COMPONENT_CSS,
    js=_COMPONENT_JS,
    isolate_styles=False,
)


def render_interactive_sar_map(payload: Dict[str, Any], key: str = "sar_studio_map"):
    """Mount the modern interactive SAR map component and return bidirectional state/triggers."""
    return sar_map_component(
        key=key,
        data=payload,
        width="stretch",
        height="content",
        on_aoi_bounds_change=lambda: None,
        on_selected_elements_change=lambda: None,
        on_timeframe_change=lambda: None,
        on_toggle_graph_change=lambda: None,
        on_view_mode_change=lambda: None,
        on_resolution_change=lambda: None,
        on_nasa_fetch_change=lambda: None,
    )
