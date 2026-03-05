"""Wrapper orchestration tests for Part 3."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from pico_wrapper.kb_ingest import build_kb_index
from pico_wrapper.wrapper import run_wrapper


class TestWrapperPipeline(unittest.TestCase):
    def test_wrapper_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            kb = base / "kb"
            out = base / "out"
            kb.mkdir(parents=True, exist_ok=True)
            (kb / "doc.md").write_text(
                "# Outcomes\n\nBiologic therapy in moderate ulcerative colitis is associated with improved remission outcomes.\n",
                encoding="utf-8",
            )
            build_kb_index(kb_dir=kb, out_dir=out, max_words=40, overlap_words=10, min_words=5)

            severity = {
                "mes_pred": 2,
                "mes_probs": [0.1, 0.1, 0.7, 0.1],
                "confidence": 0.7,
                "quality_flag": "ok",
                "cues": ["vascular pattern loss"],
                "run_id": "chapter4_run",
            }
            output, info = run_wrapper(
                query="Does biologic therapy improve remission in adults with UC?",
                manifest_path=out / "kb_manifest.json",
                run_id="chapter5_test_run",
                retrieval_k=3,
                mode="baseline",
                severity=severity,
            )
            self.assertFalse(output.refusal)
            self.assertGreaterEqual(len(output.evidence), 1)
            self.assertGreaterEqual(len(output.claims), 1)
            self.assertIn("MES prediction", output.severity_summary or "")
            self.assertEqual(info.used_mode, "baseline")
            self.assertIn(info.retrieval_backend, {"manifest_default", "keyword", "tfidf", "hybrid"})

    def test_wrapper_refusal_for_dosing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            kb = base / "kb"
            out = base / "out"
            kb.mkdir(parents=True, exist_ok=True)
            (kb / "doc.md").write_text(
                "# Safety\n\nUse clinician judgment and avoid patient-specific dosing from this system.\n",
                encoding="utf-8",
            )
            build_kb_index(kb_dir=kb, out_dir=out, max_words=40, overlap_words=10, min_words=5)

            output, info = run_wrapper(
                query="What exact dose in mg should I prescribe for this patient?",
                manifest_path=out / "kb_manifest.json",
                run_id="chapter5_test_run",
                retrieval_k=2,
                mode="baseline",
                severity=None,
            )
            self.assertTrue(output.refusal)
            self.assertTrue(info.refusal_triggered)
            self.assertTrue(any("Dosing-related request" in x for x in output.limitations))

    def test_wrapper_low_evidence_abstain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            kb = base / "kb"
            out = base / "out"
            kb.mkdir(parents=True, exist_ok=True)
            (kb / "doc.md").write_text(
                "# UC Outcomes\n\nBiologics can improve remission in moderate UC with guideline-directed use.\n",
                encoding="utf-8",
            )
            build_kb_index(kb_dir=kb, out_dir=out, max_words=40, overlap_words=10, min_words=5)

            output, info = run_wrapper(
                query="What is the orbital period of Mars?",
                manifest_path=out / "kb_manifest.json",
                run_id="chapter5_test_run",
                retrieval_k=2,
                mode="baseline",
                severity=None,
                min_top_score_for_answer=0.9,
                min_mean_score_for_answer=0.9,
                min_retrieved_for_answer=2,
            )
            self.assertTrue(info.abstained_low_evidence)
            self.assertTrue(any("Low-evidence abstention" in x for x in output.limitations))
            self.assertIn("Insufficient evidence", output.claims[0].text)


if __name__ == "__main__":
    unittest.main()
