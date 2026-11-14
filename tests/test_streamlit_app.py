"""Headless Streamlit AppTest verification for app/app.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class StreamlitAppTests(unittest.TestCase):
    def test_app_boots_and_interacts_cleanly(self) -> None:
        at = AppTest.from_file(str(REPO_ROOT / "app" / "app.py"), default_timeout=25)
        at.run()
        self.assertFalse(at.exception, f"App raised exception on boot: {at.exception}")

        # Toggle Play -> Pause
        play_buttons = [b for b in at.button if "Play Changes" in b.label]
        self.assertEqual(len(play_buttons), 1)
        play_buttons[0].click().run()
        self.assertFalse(at.exception, f"App raised exception on Play: {at.exception}")

        pause_buttons = [b for b in at.button if "Pause" in b.label]
        self.assertEqual(len(pause_buttons), 1)
        pause_buttons[0].click().run()
        self.assertFalse(at.exception, f"App raised exception on Pause: {at.exception}")

        # Step forward / backward
        next_buttons = [b for b in at.button if b.label == "▶"]
        self.assertEqual(len(next_buttons), 1)
        next_buttons[0].click().run()
        self.assertFalse(at.exception, f"App raised exception stepping forward: {at.exception}")

        # Switch map view modes
        at.sidebar.radio[0].set_value("Change Over Time Frame").run()
        self.assertFalse(at.exception, f"App raised exception in Change view: {at.exception}")

        at.sidebar.radio[0].set_value("Raw SAR Backscatter (dB)").run()
        self.assertFalse(at.exception, f"App raised exception in Backscatter view: {at.exception}")

        # Switch region and model using widget keys
        at.sidebar.selectbox(key="region_selector").select(
            "Tapajós / Santarém Frontier (Brazil)"
        ).run()
        self.assertFalse(at.exception, f"App raised exception switching region: {at.exception}")

        at.sidebar.selectbox(key="model_selector").select(
            "SAR Physics Baseline (Lee + Dual-Pol Rules)"
        ).run()
        self.assertFalse(at.exception, f"App raised exception switching model: {at.exception}")

        # Toggle Show All elements + Re-run inference
        show_all = [b for b in at.sidebar.button if b.label == "Show All"]
        self.assertEqual(len(show_all), 1)
        show_all[0].click().run()
        self.assertFalse(at.exception, f"App raised exception on Show All: {at.exception}")

        rerun_btn = [b for b in at.sidebar.button if "Re-run Inference" in b.label]
        self.assertEqual(len(rerun_btn), 1)
        rerun_btn[0].click().run()
        self.assertFalse(at.exception, f"App raised exception on Re-run Inference: {at.exception}")


if __name__ == "__main__":
    unittest.main()
