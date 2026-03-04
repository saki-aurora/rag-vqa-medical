"""Safety helper tests for Part 1."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from pico_wrapper.safety import contains_dosing_request, make_insufficient_evidence_message, should_refuse_dosing


class TestSafety(unittest.TestCase):
    def test_dosing_detection(self) -> None:
        self.assertTrue(contains_dosing_request("What dose in mg should I prescribe?"))
        self.assertTrue(should_refuse_dosing("How much should I titrate this treatment?"))
        self.assertFalse(contains_dosing_request("What outcomes are expected in 8 weeks?"))

    def test_insufficient_evidence_message(self) -> None:
        msg = make_insufficient_evidence_message()
        self.assertIn("Insufficient evidence", msg)


if __name__ == "__main__":
    unittest.main()
