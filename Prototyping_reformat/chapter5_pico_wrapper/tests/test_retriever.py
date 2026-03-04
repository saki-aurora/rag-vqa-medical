"""Retriever tests for Part 2."""

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
from pico_wrapper.retriever import retrieve_evidence
from pico_wrapper.schemas import PicoFrame


class TestRetriever(unittest.TestCase):
    def test_retrieve_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            kb = base / "kb"
            out = base / "out"
            kb.mkdir(parents=True, exist_ok=True)

            (kb / "biologics.md").write_text(
                "# Biologic outcomes\n\nAdults with moderate UC may show improved remission "
                "with biologic therapy compared with placebo in selected study cohorts.\n",
                encoding="utf-8",
            )
            (kb / "monitoring.md").write_text(
                "# Monitoring\n\nMES 2 to 3 indicates active inflammation and requires "
                "clinical review with endoscopic context.\n",
                encoding="utf-8",
            )

            build_kb_index(
                kb_dir=kb,
                out_dir=out,
                max_words=40,
                overlap_words=10,
                min_words=5,
            )

            pico = PicoFrame(
                population=["adults with UC"],
                intervention=["biologic therapy"],
                comparator=["placebo"],
                outcomes=["remission"],
            )
            results = retrieve_evidence(
                query="What evidence supports biologics for remission in moderate UC?",
                manifest_path=out / "kb_manifest.json",
                pico=pico,
                k=3,
            )
            self.assertGreaterEqual(len(results), 1)
            top = results[0]
            self.assertGreater(top.score, 0.0)
            self.assertIn("uc", top.chunk.text.lower())


if __name__ == "__main__":
    unittest.main()
