"""Headless Streamlit AppTest verification for redesigned app/app.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class StreamlitAppTests(unittest.TestCase):
    def test_app_boots_graph_toggle_presets_and_models(self) -> None:
        at = AppTest.from_file(str(REPO_ROOT / "app" / "app.py"), default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App raised exception on boot: {at.exception}")

        # Analytics graph is toggled on by default and renders Hide Graph cleanly
        self.assertTrue(bool(at.session_state["show_graph"]))
        hide_btns = [b for b in at.button if "Hide Graph" in b.label]
        self.assertEqual(len(hide_btns), 1)

        # Clicking Hide Graph toggles it off
        hide_btns[0].click().run()
        self.assertFalse(at.exception, f"App raised exception clicking Hide Graph: {at.exception}")
        self.assertFalse(bool(at.session_state["show_graph"]))

        # Re-enable analytics graph and test metric toggle + EN/DE language switching
        at.session_state["show_graph"] = True
        at.run()
        self.assertFalse(at.exception, f"App raised exception re-enabling show_graph: {at.exception}")

        metric_radios = [r for r in at.radio if r.key == "chart_metric_toggle"]
        if metric_radios:
            metric_radios[0].set_value("share").run()
            self.assertFalse(at.exception, f"App raised exception switching chart metric: {at.exception}")

        de_btns = [b for b in at.button if b.key == "lang_de_btn"]
        self.assertEqual(len(de_btns), 1)
        de_btns[0].click().run()
        self.assertFalse(at.exception, f"App raised exception switching language to DE: {at.exception}")
        self.assertEqual(at.session_state["lang"], "de")
        de_hide_btns = [b for b in at.button if "Diagramm ausblenden" in b.label]
        self.assertEqual(len(de_hide_btns), 1)

        en_btns = [b for b in at.button if b.key == "lang_en_btn"]
        self.assertEqual(len(en_btns), 1)
        en_btns[0].click().run()
        self.assertFalse(at.exception, f"App raised exception switching language back to EN: {at.exception}")
        self.assertEqual(at.session_state["lang"], "en")

        # Switch classifier backend in sidebar
        at.sidebar.selectbox(key="model_selector").select(
            "SAR Physics Baseline (Lee + Dual-Pol Rules)"
        ).run()
        self.assertFalse(at.exception, f"App raised exception switching model: {at.exception}")

        # Jump to a curated study preset via sidebar
        at.sidebar.selectbox(key="preset_selector").select(
            "Tapajós / Santarém Frontier (Brazil)"
        ).run()
        load_btns = [b for b in at.sidebar.button if "Load Selected Study Area" in b.label]
        self.assertEqual(len(load_btns), 1)
        load_btns[0].click().run()
        self.assertFalse(at.exception, f"App raised exception loading preset: {at.exception}")
        self.assertLess(float(at.session_state["aoi_bounds"][0]), 0.0)

        # Trigger explicit re-inference on the active observation square
        rerun_btns = [b for b in at.sidebar.button if "Re-run Model on Active Square" in b.label]
        self.assertEqual(len(rerun_btns), 1)
        rerun_btns[0].click().run()
        self.assertFalse(at.exception, f"App raised exception on Re-run Model: {at.exception}")

        # Test filter modes ported from teammate's NASA pipeline (Elevation, OPERA VV, NISAR GCOV, Coherence)
        for mode_id in ("elevation_m", "slope_deg", "opera_vv_db", "nisar_hhhh_db", "coherence_hh", "elements"):
            at.session_state["view_mode"] = mode_id
            at.run()
            self.assertFalse(at.exception, f"App raised exception in mode '{mode_id}': {at.exception}")

        # Test changing TARGET_RES_M resolution via sidebar selector
        at.sidebar.selectbox(key="resolution_selector").select(100.0).run()
        self.assertFalse(at.exception, f"App raised exception switching resolution to 100m: {at.exception}")
        self.assertAlmostEqual(float(at.session_state["resolution_m"]), 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
