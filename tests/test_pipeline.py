"""End-to-end verification tests for database, neural network classifier, pipeline, and Streamlit app."""
from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier import SARNeuralNetClassifier, train_reference_sar_mlp
from src.database import (
    clear_classifications,
    get_database_stats,
    get_layer_grids_for_region,
    import_processed_run_to_db,
    init_database,
    insert_sar_scene,
    list_region_dates,
    list_regions,
)
from src.earthdata_pipeline import list_processed_runs
from src.flood import otsu_water_mask
from src.map_builder import (
    build_base_map,
    build_before_after_dual_map,
    build_dynamic_overlay_group,
)
from src.pipeline import (
    build_rgba_overlay,
    compute_change_summary,
    compute_transition_rows,
    encode_bundle_for_frontend,
    extract_hotspot_markers,
    run_aoi_timeframe_pipeline,
    run_timeframe_pipeline,
)


class SARBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test_sar.db"
        init_database(self.db_path, seed_if_empty=True)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_regions_and_dates_seeded(self) -> None:
        regions = list_regions(self.db_path)
        self.assertEqual(len(regions), 4)
        for region in regions:
            dates = list_region_dates(region.region_id, self.db_path)
            self.assertGreaterEqual(len(dates), 7)
            self.assertEqual(dates, sorted(dates))

    def test_mlp_accuracy_on_physical_signatures(self) -> None:
        weights_path = Path(self._tmpdir.name) / "test_mlp.npz"
        train_reference_sar_mlp(save_path=weights_path, seed=42, epochs=280)
        clf = SARNeuralNetClassifier(weights_path=weights_path)

        rng = np.random.default_rng(123)
        # Build a 6-strip synthetic SAR benchmark scene (one strip per class 0..5)
        h, w = 120, 60
        gt = np.zeros((h, w), dtype=np.uint8)
        hh = np.zeros((h, w), dtype=np.float32)
        hv = np.zeros((h, w), dtype=np.float32)
        specs = {
            0: (-13.0, 0.55, -22.2, 0.65, 0.25),
            1: (-18.5, 0.85, -25.0, 0.80, 0.20),
            2: (-6.8, 0.65, -11.2, 0.65, 0.70),
            3: (-4.2, 0.65, -13.4, 0.65, 0.50),
            4: (-9.8, 0.65, -16.8, 0.65, 0.35),
            5: (-0.4, 0.90, -11.5, 0.95, 1.95),
        }
        for cid, (hh_m, hh_s, hv_m, hv_s, tex) in specs.items():
            r0, r1 = cid * 20, (cid + 1) * 20
            gt[r0:r1, :] = cid
            hh[r0:r1, :] = rng.normal(hh_m, hh_s, size=(20, w))
            hv[r0:r1, :] = rng.normal(hv_m, hv_s, size=(20, w))
            if tex > 0.4:
                spikes = rng.normal(0.0, tex, size=(20, w))
                hh[r0:r1, :] += spikes
                hv[r0:r1, :] += 0.45 * spikes

        res = clf.predict(hh, hv)
        self.assertEqual(res.class_mask.shape, (h, w))
        self.assertEqual(res.confidence.shape, (h, w))
        # Evaluate interior rows away from strip borders
        acc = float(np.mean(res.class_mask[2:-2, 2:-2] == gt[2:-2, 2:-2]))
        self.assertGreater(acc, 0.90)

    def test_pipeline_inference_caching_and_maps(self) -> None:
        dates = list_region_dates("elbe_flood", self.db_path)
        start_d, end_d = dates[0], dates[-1]

        # First run: classifies all scenes via neural net and caches in SQLite
        b1 = run_timeframe_pipeline(
            region_id="elbe_flood",
            start_date=start_d,
            end_date=end_d,
            model_id="sar_mlp_v1",
            db_path=self.db_path,
        )
        self.assertEqual(b1.newly_inferred, len(dates))
        self.assertEqual(b1.cache_hits, 0)

        # Second run: served cleanly from SQLite cache
        b2 = run_timeframe_pipeline(
            region_id="elbe_flood",
            start_date=start_d,
            end_date=end_d,
            model_id="sar_mlp_v1",
            db_path=self.db_path,
        )
        self.assertEqual(b2.newly_inferred, 0)
        self.assertEqual(b2.cache_hits, len(dates))

        # Check overlays, change summary, hotspots, and Folium rendering
        base_f, peak_f = b2.frames[0], b2.frames[3]
        for mode in ("elements", "change", "backscatter"):
            rgba = build_rgba_overlay(
                peak_f,
                selected_class_ids=[1, 2, 3],
                view_mode=mode,
                baseline_frame=base_f,
            )
            self.assertEqual(rgba.shape, (144, 144, 4))
            self.assertEqual(rgba.dtype, np.uint8)

        summary = compute_change_summary(base_f, peak_f, [1, 3], b2.region.pixel_area_km2)
        self.assertGreater(summary["gained_km2"], 0.5)

        transitions = compute_transition_rows(base_f, peak_f, b2.region.pixel_area_km2)
        self.assertGreater(len(transitions), 0)

        hotspots = extract_hotspot_markers(peak_f, base_f, b2.region, [1, 2])
        self.assertGreater(len(hotspots), 0)

        base_map = build_base_map(b2.region)
        fg = build_dynamic_overlay_group(peak_f, b2.region, [1, 2, 3], "elements", base_f)
        self.assertIsNotNone(base_map._repr_html_())
        self.assertGreater(len(fg._children), 0)

        dual = build_before_after_dual_map(base_f, peak_f, b2.region, [1, 2], show_change_on_right=True)
        self.assertIn("leaflet", dual._repr_html_().lower())

        # Otsu classical water sanity check
        water_mask = otsu_water_mask(peak_f.scene.hh_db)
        self.assertGreater(int(water_mask.sum()), 100)

    def test_custom_scene_ingest_invalidates_and_clears(self) -> None:
        run_timeframe_pipeline("amazon_deforestation", "2025-11-10", "2026-08-20", db_path=self.db_path)
        hh = np.full((144, 144), -18.0, dtype=np.float32)
        hv = np.full((144, 144), -24.5, dtype=np.float32)
        scene_id = insert_sar_scene(
            region_id="amazon_deforestation",
            acq_date="2026-09-01",
            hh_db=hh,
            hv_db=hv,
            db_path=self.db_path,
        )
        self.assertIn("amazon_deforestation", scene_id)
        stats = get_database_stats(self.db_path)
        self.assertEqual(stats["sar_scenes"], 31)

        cleared = clear_classifications(region_id="amazon_deforestation", db_path=self.db_path)
        self.assertGreaterEqual(cleared, 8)

    def test_custom_aoi_drag_pipeline_and_presets(self) -> None:
        # Arbitrary dragged observation box (e.g., Seine / Paris corridor)
        bundle_1, avail, s_date, e_date = run_aoi_timeframe_pipeline(
            south=48.78,
            west=2.24,
            north=48.94,
            east=2.48,
            label="Custom AOI · Paris",
            db_path=self.db_path,
        )
        self.assertGreaterEqual(len(avail), 4)
        self.assertEqual(s_date, avail[0])
        self.assertEqual(e_date, avail[-1])
        self.assertEqual(bundle_1.newly_inferred, len(avail))
        self.assertEqual(bundle_1.cache_hits, 0)
        self.assertGreater(bundle_1.region.pixel_area_km2, 0.0)

        # Presets filter must still return only the 4 curated study areas, excluding dragged AOIs
        presets = list_regions(self.db_path, presets_only=True)
        all_regions = list_regions(self.db_path, presets_only=False)
        self.assertEqual(len(presets), 4)
        self.assertEqual(len(all_regions), 5)

        # Second call with identical bounds hits SQLite cache cleanly
        bundle_2, _, _, _ = run_aoi_timeframe_pipeline(
            south=48.78,
            west=2.24,
            north=48.94,
            east=2.48,
            label="Custom AOI · Paris",
            db_path=self.db_path,
        )
        self.assertEqual(bundle_2.region.region_id, bundle_1.region.region_id)
        self.assertEqual(bundle_2.newly_inferred, 0)
        self.assertEqual(bundle_2.cache_hits, len(avail))

        # World-coordinate consistency: shifting box east by half width keeps overlapping terrain coherent
        bundle_shifted, _, _, _ = run_aoi_timeframe_pipeline(
            south=48.78,
            west=2.36,
            north=48.94,
            east=2.60,
            label="Custom AOI · Paris East",
            db_path=self.db_path,
        )
        hh_a = bundle_1.frames[0].scene.hh_db
        hh_b = bundle_shifted.frames[0].scene.hh_db
        half_cols = min(hh_a.shape[1], hh_b.shape[1]) // 2
        corr = float(np.corrcoef(hh_a[:, -half_cols:].ravel(), hh_b[:, :half_cols].ravel())[0, 1])
        self.assertGreater(corr, 0.7)

    def test_encode_bundle_for_frontend(self) -> None:
        presets = list_regions(self.db_path, presets_only=True)
        preset = presets[0]
        bundle, avail, s_date, e_date = run_aoi_timeframe_pipeline(
            south=preset.south,
            west=preset.west,
            north=preset.north,
            east=preset.east,
            label=preset.name,
            db_path=self.db_path,
        )
        payload = encode_bundle_for_frontend(
            bundle=bundle,
            available_dates=avail,
            selected_elements=[1, 2],
            view_mode="elements",
            sar_band="HH",
            opacity=0.78,
            show_graph=False,
            presets=presets,
            start_date=s_date,
            end_date=e_date,
            db_path=self.db_path,
        )
        self.assertEqual(len(payload["presets"]), 4)
        self.assertEqual(payload["selected_elements"], [1, 2])
        self.assertFalse(payload["show_graph"])
        self.assertGreater(payload["region"]["width_km"], 0.0)
        self.assertGreater(payload["region"]["height_km"], 0.0)
        self.assertAlmostEqual(payload["region"]["south"], preset.south, places=4)
        self.assertAlmostEqual(payload["region"]["east"], preset.east, places=4)
        self.assertEqual(len(payload["frames"]), len(bundle.frames))

        first_frame = payload["frames"][0]
        expected_len = first_frame["rows"] * first_frame["cols"]
        self.assertEqual(len(base64.b64decode(first_frame["mask_b64"])), expected_len)
        self.assertEqual(len(base64.b64decode(first_frame["hh_b64"])), expected_len)
        self.assertEqual(len(base64.b64decode(first_frame["hv_b64"])), expected_len)

    def test_resolution_scaling_and_elevation_mode(self) -> None:
        b_fine, _, _, _ = run_aoi_timeframe_pipeline(
            south=48.95,
            west=8.35,
            north=49.10,
            east=8.55,
            resolution_m=30.0,
            label="AOI Fine",
            db_path=self.db_path,
        )
        b_coarse, avail_c, s_d, e_d = run_aoi_timeframe_pipeline(
            south=48.95,
            west=8.35,
            north=49.10,
            east=8.55,
            resolution_m=150.0,
            label="AOI Coarse",
            db_path=self.db_path,
        )
        self.assertGreater(
            b_fine.frames[0].scene.hh_db.shape[0],
            b_coarse.frames[0].scene.hh_db.shape[0],
        )
        self.assertGreater(b_coarse.region.pixel_area_km2, b_fine.region.pixel_area_km2)

        presets = list_regions(self.db_path, presets_only=True)
        elev_payload = encode_bundle_for_frontend(
            bundle=b_coarse,
            available_dates=avail_c,
            selected_elements=[1, 2],
            view_mode="elevation_m",
            sar_band="False-Color Dual-Pol (HH/HV)",
            opacity=0.8,
            show_graph=False,
            presets=presets,
            start_date=s_d,
            end_date=e_d,
            db_path=self.db_path,
        )
        self.assertIsNotNone(elev_payload["static_layer_b64"])
        self.assertEqual(elev_payload["active_layer"]["mode_id"], "elevation_m")
        self.assertEqual(elev_payload["region"]["resolution_m"], 150.0)
        static_bytes = base64.b64decode(elev_payload["static_layer_b64"])
        expected_px = elev_payload["region"]["grid_rows"] * elev_payload["region"]["grid_cols"]
        self.assertEqual(len(static_bytes), expected_px)

        elev_map = get_layer_grids_for_region(b_coarse.region.region_id, "elevation_m", db_path=self.db_path)
        rgba = build_rgba_overlay(
            b_coarse.frames[0], [1, 2], view_mode="elevation_m", layer_grid=elev_map["static"]
        )
        self.assertEqual(rgba.shape[2], 4)

    def test_import_processed_nasa_netcdf_run(self) -> None:
        import pandas as pd
        import rioxarray  # noqa: F401
        import xarray as xr

        processed_root = Path(self._tmpdir.name) / "processed"
        run_dir = processed_root / "oberrhein_unit"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create 60x60 UTM 32632 grid at 60m resolution in Upper Rhine Valley
        res_m = 60.0
        xs = 450000.0 + np.arange(60, dtype=np.float64) * res_m + res_m * 0.5
        ys = 5445000.0 - np.arange(60, dtype=np.float64) * res_m - res_m * 0.5
        yy, xx = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60), indexing="ij")

        def _geo_da(arr: np.ndarray, time_val=None):
            if time_val is not None:
                da = xr.DataArray(
                    arr.astype("float32"),
                    coords={"time": [pd.Timestamp(time_val)], "y": ys, "x": xs},
                    dims=("time", "y", "x"),
                )
            else:
                da = xr.DataArray(arr.astype("float32"), coords={"y": ys, "x": xs}, dims=("y", "x"))
            return da.rio.write_crs("EPSG:32632").rio.set_spatial_dims(x_dim="x", y_dim="y")

        elev = (85.0 + 140.0 * xx + 25.0 * np.sin(yy * 4.0)).astype("float32")
        slope = np.full_like(elev, 6.5)
        aspect = np.full_like(elev, 180.0)
        xr.Dataset(
            {"elevation_m": _geo_da(elev), "slope_deg": _geo_da(slope), "aspect_deg": _geo_da(aspect)}
        ).to_netcdf(run_dir / "nasadem_static.nc", engine="h5netcdf")

        vv = np.full((1, 60, 60), -11.5, dtype="float32")
        vv[0, :, 25:30] = -20.5  # River channel
        vh = vv - 6.5
        xr.Dataset(
            {"VV_dB": _geo_da(vv, "2025-01-02"), "VH_dB": _geo_da(vh, "2025-01-02")}
        ).to_netcdf(run_dir / "opera_rtc_s1.nc", engine="h5netcdf")

        hhhh = np.full((1, 60, 60), -9.0, dtype="float32")
        hhhh[0, :, 25:30] = -19.5
        hvhv = hhhh - 5.5
        xr.Dataset(
            {"HHHH_dB": _geo_da(hhhh, "2026-06-17"), "HVHV_dB": _geo_da(hvhv, "2026-06-17")}
        ).to_netcdf(run_dir / "nisar_gcov.nc", engine="h5netcdf")

        coh = np.full((1, 60, 60), 0.68, dtype="float32")
        coh[0, :, 25:30] = 0.14
        xr.Dataset(
            {"coherence_HH": _geo_da(coh, "2026-06-24")}
        ).to_netcdf(run_dir / "nisar_gunw.nc", engine="h5netcdf")

        runs = list_processed_runs(processed_root)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0].resolution_m, 60.0, places=1)
        self.assertEqual(list(runs[0].dates), ["2025-01-02", "2026-06-17", "2026-06-24"])

        rec = import_processed_run_to_db(run_dir, db_path=self.db_path)
        self.assertEqual(rec.source_type, "nasa_earthdata")
        self.assertAlmostEqual(rec.resolution_m, 60.0, places=1)

        bundle, avail, s_d, e_d = run_aoi_timeframe_pipeline(
            south=rec.south,
            west=rec.west,
            north=rec.north,
            east=rec.east,
            region_id=rec.region_id,
            db_path=self.db_path,
        )
        self.assertEqual(avail, ["2025-01-02", "2026-06-17", "2026-06-24"])

        presets = list_regions(self.db_path, presets_only=True)
        coh_payload = encode_bundle_for_frontend(
            bundle=bundle,
            available_dates=avail,
            selected_elements=[1, 2],
            view_mode="coherence_hh",
            sar_band="False-Color Dual-Pol (HH/HV)",
            opacity=0.8,
            show_graph=False,
            presets=presets,
            start_date=s_d,
            end_date=e_d,
            db_path=self.db_path,
        )
        # Coherence was only on 2026-06-24 (frame index 2)
        self.assertEqual([f["has_layer"] for f in coh_payload["frames"]], [False, False, True])

    def test_multiyear_source_range_and_german_localization(self) -> None:
        from src.data_source import DEFAULT_WORLD_TIMELINE
        from src.earthdata_pipeline import get_source_date_limits

        min_d, max_d, per_src = get_source_date_limits()
        self.assertEqual(min_d, "2016-01-01")
        self.assertGreaterEqual(max_d, "2026-01-01")
        self.assertIn("OPERA_L2_RTC-S1_V1", per_src)
        self.assertIn("NISAR_L2_GCOV_V1", per_src)
        self.assertGreaterEqual(len(DEFAULT_WORLD_TIMELINE), 12)
        self.assertTrue(DEFAULT_WORLD_TIMELINE[0][0].startswith("2016-"))
        self.assertTrue(DEFAULT_WORLD_TIMELINE[-1][0].startswith("2026-"))

        bundle, avail, s_d, e_d = run_aoi_timeframe_pipeline(
            south=12.40,
            west=13.55,
            north=13.50,
            east=14.95,
            region_id="lake_chad_shrinking",
            extend_to_source_history=True,
            db_path=self.db_path,
        )
        self.assertGreaterEqual(len(avail), 12)
        self.assertTrue(avail[0].startswith("2016-"))
        self.assertGreaterEqual(len(bundle.frames), 12)

        presets = list_regions(self.db_path, presets_only=True)
        de_payload = encode_bundle_for_frontend(
            bundle=bundle,
            available_dates=avail,
            selected_elements=[1, 2],
            view_mode="elevation_m",
            sar_band="False-Color Dual-Pol (HH/HV)",
            opacity=0.8,
            show_graph=True,
            presets=presets,
            start_date=s_d,
            end_date=e_d,
            lang="de",
            db_path=self.db_path,
        )
        self.assertEqual(de_payload["lang"], "de")
        self.assertEqual(de_payload["source_limits"]["min_date"], "2016-01-01")
        self.assertIn("i18n", de_payload)
        self.assertEqual(de_payload["active_layer"]["label"], "NASADEM Höhe (m)")


if __name__ == "__main__":
    unittest.main()
