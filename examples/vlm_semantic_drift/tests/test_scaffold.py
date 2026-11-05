"""Verify the scaffold contract without loading production services."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SAMPLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SAMPLE_DIR.parents[1]


class ScaffoldTests(unittest.TestCase):
    """Check the published example and isolated package imports."""

    def setUp(self) -> None:
        with (SAMPLE_DIR / "config.example.json").open(encoding="utf-8") as stream:
            self.config = json.load(stream)

    def test_configuration_structure(self) -> None:
        self.assertEqual(set(self.config), {"schema_version", "vlm", "aggregation"})
        self.assertIs(type(self.config["schema_version"]), int)
        self.assertEqual(self.config["schema_version"], 1)
        self.assertEqual(self.config["vlm"], {"provider": "mock"})
        self.assertEqual(set(self.config["aggregation"]), {
            "window_size", "min_valid_samples", "mismatch_ratio_threshold",
            "consecutive_mismatches",
        })

    def test_demonstration_defaults(self) -> None:
        self.assertEqual(self.config["aggregation"], {
            "window_size": 10,
            "min_valid_samples": 5,
            "mismatch_ratio_threshold": 0.6,
            "consecutive_mismatches": 3,
        })

    def test_threshold_ranges_and_relationships(self) -> None:
        values = self.config["aggregation"]
        for key in ("window_size", "min_valid_samples", "consecutive_mismatches"):
            with self.subTest(field=key):
                self.assertIs(type(values[key]), int)
                self.assertGreater(values[key], 0)
        self.assertLessEqual(values["min_valid_samples"], values["window_size"])
        self.assertGreaterEqual(values["consecutive_mismatches"], 2)
        self.assertLessEqual(values["consecutive_mismatches"], values["min_valid_samples"])
        ratio = values["mismatch_ratio_threshold"]
        self.assertIn(type(ratio), (int, float))
        self.assertGreater(ratio, 0)
        self.assertLessEqual(ratio, 1)

    def test_import_without_dependencies_or_side_effects(self) -> None:
        # The audit hook rejects writes even outside the temporary working directory.
        script = r'''
import os
import sys

sys.path.insert(0, sys.argv[1])
blocked = {
    "vision_analysis", "cron", "retrain", "dbmanager", "monitor",
    "util", "logger_config", "torch", "ultralytics", "cv2", "PIL",
    "psycopg2", "requests", "fastapi", "yaml", "apscheduler",
}

class BlockServices:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("Unexpected dependency: " + fullname)
        return None

def audit(event, args):
    if event == "open":
        mode, flags = args[1], args[2]
        writing = isinstance(mode, str) and any(c in mode for c in "wax+")
        writing = writing or (isinstance(flags, int) and flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        ))
        if writing:
            raise AssertionError("Import attempted a file write")
        if str(args[0]).endswith(("config.yaml", "config.example.json")):
            raise AssertionError("Import attempted configuration loading")
    if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.chdir",
                 "os.system", "subprocess.Popen"} or event.startswith("socket."):
        raise AssertionError("Import side effect: " + event)

sys.meta_path.insert(0, BlockServices())
sys.addaudithook(audit)
import examples
import examples.vlm_semantic_drift
assert not (blocked & {name.split(".")[0] for name in sys.modules})
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", script, str(REPO_ROOT)],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
