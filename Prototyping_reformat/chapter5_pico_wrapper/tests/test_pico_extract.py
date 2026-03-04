"""PICO extractor tests for Part 3."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from pico_wrapper.pico_extract import extract_pico_baseline


class TestPicoExtract(unittest.TestCase):
    def test_extract_pico_baseline(self) -> None:
        q = (
            "In adults with ulcerative colitis and Mayo 2 disease, does biologic therapy "
            "versus placebo improve clinical remission at 12 weeks in outpatient setting without steroids?"
        )
        p = extract_pico_baseline(q)
        self.assertTrue(any("adult" in x.lower() for x in p.population))
        self.assertTrue(any("ulcerative colitis" in x.lower() or x.lower() == "uc" for x in p.population))
        self.assertTrue(any("biologic" in x.lower() for x in p.intervention))
        self.assertTrue(any("placebo" in x.lower() for x in p.comparator))
        self.assertTrue(any("remission" in x.lower() for x in p.outcomes))
        self.assertTrue(any("mes 2" in x.lower() for x in p.severity_anchors))
        self.assertIsNotNone(p.timeframe)
        self.assertEqual((p.setting or "").lower(), "outpatient")
        self.assertTrue(any("without steroids" in x.lower() for x in p.constraints))


if __name__ == "__main__":
    unittest.main()
