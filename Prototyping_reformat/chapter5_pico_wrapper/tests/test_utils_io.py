"""I/O utility tests for Part 1."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from pico_wrapper.utils_io import freeze_environment, generate_run_id, read_json, write_json, write_jsonl


class TestUtilsIo(unittest.TestCase):
    def test_generate_run_id(self) -> None:
        rid = generate_run_id("chapter5")
        self.assertTrue(rid.startswith("chapter5_"))
        self.assertGreaterEqual(len(rid), 20)

    def test_json_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            p_json = base / "a" / "x.json"
            p_jsonl = base / "a" / "x.jsonl"
            p_env = base / "a" / "environment.txt"

            write_json(p_json, {"a": 1, "b": "ok"})
            self.assertEqual(read_json(p_json)["a"], 1)

            write_jsonl(p_jsonl, [{"x": 1}, {"x": 2}])
            lines = p_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            freeze_environment(p_env)
            text = p_env.read_text(encoding="utf-8")
            self.assertIn("python=", text)


if __name__ == "__main__":
    unittest.main()
