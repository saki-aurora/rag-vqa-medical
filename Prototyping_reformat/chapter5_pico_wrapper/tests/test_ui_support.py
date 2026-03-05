"""Tests for UI support helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from pico_wrapper.ui_support import build_markdown_report, build_safety_alert


class TestUiSupport(unittest.TestCase):
    def test_build_safety_alert_refusal(self) -> None:
        alert = build_safety_alert(
            run_info={"refusal_triggered": True, "abstained_low_evidence": False},
            wrapper_output={"refusal": True, "limitations": []},
        )
        self.assertIsNotNone(alert)
        assert alert is not None
        self.assertEqual(alert["level"], "danger")

    def test_build_safety_alert_low_evidence(self) -> None:
        alert = build_safety_alert(
            run_info={
                "refusal_triggered": False,
                "abstained_low_evidence": True,
                "abstain_reason": "top_score<0.180",
            },
            wrapper_output={"refusal": False, "limitations": []},
        )
        self.assertIsNotNone(alert)
        assert alert is not None
        self.assertEqual(alert["level"], "warning")
        self.assertIn("top_score<0.180", alert["message"])

    def test_build_markdown_report(self) -> None:
        md = build_markdown_report(
            run_id="run_001",
            request_payload={"query": "Does biologic therapy improve remission?", "mode": "baseline", "retrieval_k": 5},
            run_info={"used_mode": "baseline", "retrieval_backend": "hybrid", "retrieval_k": 5},
            wrapper_output={
                "pico": {
                    "population": ["adults with UC"],
                    "intervention": ["biologic therapy"],
                    "comparator": ["standard care"],
                    "outcomes": ["clinical remission"],
                    "severity_anchors": ["mayo 2"],
                    "timeframe": "12 weeks",
                    "setting": "outpatient",
                    "constraints": [],
                },
                "claims": [{"text": "Biologics improve remission in selected cohorts.", "citation_ids": ["c1"]}],
                "evidence": [{"chunk_id": "c1", "doc_id": "docA", "text": "Biologics improved remission outcomes."}],
                "uncertainty": "Moderate certainty due to limited sample size.",
                "limitations": ["Small evidence base."],
                "disclaimer": "Decision-support only.",
            },
            safety_alert={"level": "warning", "title": "Low-Evidence Warning", "message": "Low-evidence abstention triggered."},
            generated_utc="2026-03-05T00:00:00+00:00",
        )
        self.assertIn("Chapter 5 Wrapper Report", md)
        self.assertIn("## Safety Alert", md)
        self.assertIn("## Claims", md)
        self.assertIn("## Evidence (Top Retrieved)", md)
        self.assertIn("Decision-support only.", md)


if __name__ == "__main__":
    unittest.main()
