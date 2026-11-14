"""End-to-end verification tests for database, neural network classifier, pipeline, and Streamlit app."""
from __future__ import annotations

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
    init_database,
    insert_sar_scene,
    list_region_dates,
    list_regions,
)
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
    extract_hotspot_markers,
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


if __name__ == "__main__":
    unittest.main()
