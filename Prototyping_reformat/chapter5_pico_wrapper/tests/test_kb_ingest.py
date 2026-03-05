"""KB ingestion tests for Part 2."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from pico_wrapper.kb_ingest import build_kb_index


class TestKbIngest(unittest.TestCase):
    def test_build_kb_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            kb = base / "kb"
            out = base / "out"
            kb.mkdir(parents=True, exist_ok=True)

            (kb / "doc1.md").write_text(
                "# Title\n\nAdults with UC and MES 2 may require therapy review.\n"
                "Outcomes include remission and endoscopic improvement.\n",
                encoding="utf-8",
            )
            (kb / "doc2.txt").write_text(
                "Comparator options include placebo or standard care in trial summaries.\n",
                encoding="utf-8",
            )

            summary = build_kb_index(
                kb_dir=kb,
                out_dir=out,
                max_words=40,
                overlap_words=10,
                min_words=5,
            )
            self.assertGreater(summary["n_chunks"], 0)
            self.assertTrue((out / "kb_manifest.json").exists())
            self.assertTrue((out / "chunks.jsonl").exists())

            manifest = json.loads((out / "kb_manifest.json").read_text(encoding="utf-8"))
            self.assertIn(manifest["index_backend"], {"hybrid", "tfidf", "keyword"})
            self.assertEqual(manifest["n_source_files"], 2)
            self.assertIn("available_backends", manifest)
            self.assertIn("index_files", manifest)


if __name__ == "__main__":
    unittest.main()
