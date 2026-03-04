"""Schema contract tests for Part 1."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from pico_wrapper.schemas import (
    Citation,
    Claim,
    EvidenceChunk,
    PicoFrame,
    SeverityResult,
    WrapperInput,
    WrapperOutput,
)


class TestSchemas(unittest.TestCase):
    def test_severity_result_validation(self) -> None:
        with self.assertRaises(ValueError):
            SeverityResult(mes_pred=9)

        with self.assertRaises(ValueError):
            SeverityResult(mes_pred=2, mes_probs=[0.2, 0.2, 0.2])

        result = SeverityResult(
            mes_pred=2,
            mes_probs=[0.1, 0.2, 0.6, 0.1],
            confidence=0.6,
            quality_flag="ok",
            cues=["friability", "ulceration"],
            model_version="chapter4_lora",
            run_id="r1",
            timestamp="2026-03-03T00:00:00Z",
        )
        self.assertEqual(result.mes_pred, 2)
        self.assertEqual(len(result.cues), 2)

    def test_wrapper_output_roundtrip(self) -> None:
        pico = PicoFrame(
            population=["adult UC"],
            intervention=["biologic therapy"],
            comparator=["standard care"],
            outcomes=["clinical remission"],
            severity_anchors=["MES 2"],
            timeframe="8-12 weeks",
            setting="outpatient",
            constraints=["no dosing advice"],
        )
        severity = SeverityResult(
            mes_pred=2,
            mes_probs=[0.1, 0.1, 0.7, 0.1],
            confidence=0.7,
            quality_flag="ok",
            cues=["vascular pattern loss"],
            model_version="chapter4_lora",
            run_id="vlm_lora_finetune_mayo_balanced_full_20260303",
        )
        _ = WrapperInput(query="How should I compare biologics in moderate UC?", severity=severity, retrieval_k=5)

        out = WrapperOutput(
            run_id="chapter5_run_01",
            query="How should I compare biologics in moderate UC?",
            pico=pico,
            severity_summary="MES 2 predicted with moderate confidence.",
            evidence=[
                EvidenceChunk(
                    doc_id="doc_uc_1",
                    chunk_id="doc_uc_1#c0001",
                    source_path="data/kb/sample_docs/doc1.md",
                    text="Biologics may improve remission rates in moderate disease.",
                    section="Results",
                    start_offset=0,
                    end_offset=64,
                )
            ],
            claims=[Claim(text="Biologics are associated with improved remission.", citation_ids=["doc_uc_1#c0001"])],
            citations=[Citation(chunk_id="doc_uc_1#c0001", doc_id="doc_uc_1")],
            uncertainty="Moderate evidence strength.",
            limitations=["Single chunk cited."],
            disclaimer="Decision-support only.",
            refusal=False,
        )
        payload = out.to_dict()
        restored = WrapperOutput.from_dict(payload)
        self.assertEqual(restored.run_id, out.run_id)
        self.assertEqual(restored.pico.intervention[0], "biologic therapy")
        self.assertEqual(restored.claims[0].citation_ids[0], "doc_uc_1#c0001")


if __name__ == "__main__":
    unittest.main()
