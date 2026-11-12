"""Exercise actual demo modules, CLI contracts and offline resource handling."""

from contextlib import ExitStack, redirect_stdout, redirect_stderr
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from examples.vlm_semantic_drift import demo_pipeline as demo


class DemoPipelineTests(unittest.TestCase):
    def test_candidate(self):
        report = demo.run_demo("candidate")
        self.assertEqual(report["decision"]["status"], "CANDIDATE")
        self.assertTrue(report["decision"]["is_candidate"])
        self.assertEqual(report["window"]["valid_samples"], 5)
        self.assertEqual(report["window"]["mismatch_ratio"], 0.6)
        self.assertEqual(report["decision"]["consecutive_mismatches"], 3)
        self.assertEqual(report["human_feedback"][0]["verdict"], "CONFIRMED")

    def test_no_candidate(self):
        report = demo.run_demo("no_candidate")
        self.assertEqual(report["decision"]["status"], "NO_CANDIDATE")
        self.assertFalse(report["decision"]["is_candidate"])
        self.assertEqual(report["window"]["valid_samples"], 5)
        self.assertEqual(report["window"]["mismatch_ratio"], 0.6)
        self.assertEqual(report["decision"]["consecutive_mismatches"], 0)
        self.assertEqual(report["human_feedback"], [])

    def test_insufficient_data(self):
        report = demo.run_demo("insufficient_data")
        window = report["window"]
        self.assertEqual(report["decision"]["status"], "INSUFFICIENT_DATA")
        self.assertFalse(report["decision"]["is_candidate"])
        self.assertEqual([window[k] for k in ("total_entries", "yes_count", "no_count",
                                             "unknown_count", "invalid_count", "valid_samples")],
                         [4, 1, 1, 1, 1, 2])
        self.assertEqual(window["mismatch_ratio"], 0.5)
        self.assertEqual(report["human_feedback"], [])

    def test_actual_image_quality_prompt_and_resource_lifecycle(self):
        images, rois, events = [], [], []
        extract = demo.extract_roi
        quality_fn = demo.analyze_image_quality
        prompt_fn = demo.build_hypothesis_prompts

        def capture_roi(image, event):
            before = image.tobytes()
            result = extract(image, event)
            self.assertEqual(result.bbox, (0, 4, 28, 32))
            self.assertEqual(result.size, (28, 28))
            self.assertEqual(image.tobytes(), before)
            images.append((image, before))
            rois.append(result.image)
            events.append((event, asdict(event)))
            return result

        def capture_quality(image):
            before = image.tobytes()
            result = quality_fn(image)
            self.assertEqual(image.tobytes(), before)
            self.assertTrue(result.assessable)
            self.assertGreater(result.edge_score, 0)
            return result

        def capture_prompt(event, quality=None):
            before = asdict(quality)
            result = prompt_fn(event, quality=quality)
            self.assertEqual(asdict(quality), before)
            data = json.loads(result.data_json)
            self.assertEqual(data["class_label"], "person")
            self.assertEqual(data["quality"]["mean_brightness"], quality.mean_brightness)
            self.assertIn("H0:", result.h0)
            self.assertIn("H1:", result.h1)
            self.assertIn("YES, NO, or UNKNOWN", result.response_contract)
            self.assertIn("UNTRUSTED DATA", result.prompt)
            return result

        with patch.object(demo, "extract_roi", side_effect=capture_roi) as crop, \
                patch.object(demo, "analyze_image_quality", side_effect=capture_quality) as quality, \
                patch.object(demo, "build_hypothesis_prompts", side_effect=capture_prompt) as prompt:
            demo.run_demo()
        self.assertEqual((crop.call_count, quality.call_count, prompt.call_count), (5, 5, 5))
        for event, before in events:
            self.assertEqual(asdict(event), before)
        for image, _ in images:
            with self.assertRaises(ValueError):
                image.tobytes()
        for image in rois:
            with self.assertRaises(ValueError):
                image.tobytes()

    def test_source_unchanged_through_whole_event(self):
        snapshots = []
        factory = demo._synthetic_image
        add = demo.SlidingWindow.add

        def image_factory():
            image = factory()
            snapshots.append((image, image.tobytes()))
            return image

        def capture_add(window, event, parsed):
            image, pixels = snapshots[-1]
            self.assertEqual(image.tobytes(), pixels)
            before = (asdict(event), asdict(parsed))
            add(window, event, parsed)
            self.assertEqual((asdict(event), asdict(parsed)), before)

        with patch.object(demo, "_synthetic_image", side_effect=image_factory), \
                patch.object(demo.SlidingWindow, "add", new=capture_add):
            demo.run_demo()

    def test_unassessable_quality_does_not_gate(self):
        with patch.object(demo, "_synthetic_image", side_effect=lambda: Image.new("RGB", (32, 32), 0)), \
                patch.object(demo, "analyze_image_quality", wraps=demo.analyze_image_quality) as quality:
            report = demo.run_demo("candidate")
        self.assertEqual(quality.call_count, 5)
        self.assertEqual(report["decision"]["status"], "CANDIDATE")
        self.assertEqual(report["window"]["valid_samples"], 5)

    def test_actual_mock_parser_and_invalid_fixture(self):
        seen = []
        parse = demo.parse_vlm_response
        generate = demo.MockVLMAdapter.generate

        def capture_parse(raw):
            result = parse(raw)
            self.assertEqual(result.raw_response, raw.raw_response)
            self.assertEqual((result.provider, result.is_mock), ("mock", True))
            seen.append(result)
            return result

        def capture_generate(adapter, prompt, roi=None):
            self.assertIsNotNone(roi)
            self.assertEqual(roi.size, (28, 28))
            return generate(adapter, prompt, roi)

        with patch.object(demo, "parse_vlm_response", side_effect=capture_parse), \
                patch.object(demo.MockVLMAdapter, "generate", autospec=True,
                             side_effect=capture_generate) as mock:
            demo.run_demo("insufficient_data")
        self.assertEqual(mock.call_count, 3)
        self.assertEqual([r.status for r in seen], ["YES", "NO", "UNKNOWN", "INVALID"])
        self.assertIsNone(seen[-1].supports_h1)
        self.assertFalse(seen[-1].valid)

    def test_decision_feedback_report_are_actually_connected(self):
        with ExitStack() as stack:
            spies = {name: stack.enter_context(patch.object(demo, name, wraps=getattr(demo, name)))
                     for name in ("evaluate_drift", "create_feedback", "build_drift_report")}
            report = demo.run_demo()
        self.assertTrue(all(spy.call_count == 1 for spy in spies.values()))
        args = spies["build_drift_report"].call_args.args
        stats, decision, records = args
        self.assertEqual(stats.valid_count, 5)
        self.assertEqual(decision.status, "CANDIDATE")
        self.assertEqual(records[0].decision_status, decision.status)
        self.assertEqual(records[0].is_mock, decision.is_mock)
        self.assertEqual(report["human_feedback"][0]["feedback_id"], records[0].feedback_id)

    def test_no_feedback_creation_for_other_scenarios(self):
        with patch.object(demo, "create_feedback", side_effect=AssertionError("unexpected feedback")):
            demo.run_demo("no_candidate")
            demo.run_demo("insufficient_data")

    def test_time_metadata_schema_and_isolation(self):
        with patch.object(demo, "build_drift_report", wraps=demo.build_drift_report) as builder:
            first = demo.run_demo("candidate")
            second = demo.run_demo("insufficient_data")
        stats = builder.call_args_list[0].args[0]
        self.assertEqual([e.timestamp for e in stats.entries],
                         ["2026-01-01T00:00:0{}Z".format(i) for i in range(5)])
        self.assertEqual(len({e.event_id for e in stats.entries}), 5)
        self.assertEqual(first["generated_at"], "2026-01-01T00:02:00Z")
        self.assertEqual(first["source"], {"provider": "mock", "is_mock": True})
        self.assertEqual(first["schema_version"], "1.0")
        self.assertEqual(second["window"]["total_entries"], 4)
        self.assertEqual(first, demo.run_demo("candidate"))

    def test_reproducible_json_and_independent_reports(self):
        first = demo.run_demo()
        self.assertEqual(demo.serialize_drift_report(first), demo.serialize_drift_report(demo.run_demo()))
        first["human_feedback"].clear()
        self.assertEqual(len(demo.run_demo()["human_feedback"]), 1)

    def test_invalid_scenario(self):
        for value in ("", "CANDIDATE", "other"):
            with self.assertRaises(ValueError):
                demo.run_demo(value)
        for value in (None, [], True):
            with self.assertRaises(TypeError):
                demo.run_demo(value)

    def test_cleanup_on_failure(self):
        image = demo._synthetic_image()
        rois = []
        extract = demo.extract_roi

        def capture(image, event):
            roi = extract(image, event)
            rois.append(roi.image)
            return roi

        with patch.object(demo, "_synthetic_image", return_value=image), \
                patch.object(demo, "extract_roi", side_effect=capture), \
                patch.object(demo, "build_hypothesis_prompts", side_effect=ValueError("synthetic failure")):
            with self.assertRaises(ValueError):
                demo.run_demo()
        for closed in (image, rois[0]):
            with self.assertRaises(ValueError):
                closed.tobytes()

    def test_main_execution_error(self):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(demo, "run_demo", side_effect=ImportError("Pillow missing")), \
                redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(demo.main([]), 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("Pillow missing", err.getvalue())

    def test_cli_scenarios_default_and_bad_argument(self):
        for scenario, expected in (("candidate", "CANDIDATE"), ("no_candidate", "NO_CANDIDATE"),
                                   ("insufficient_data", "INSUFFICIENT_DATA"), (None, "CANDIDATE")):
            args = [] if scenario is None else ["--scenario", scenario]
            result = subprocess.run([sys.executable, "-B", "-m", demo.__name__] + args,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertEqual(json.loads(result.stdout)["decision"]["status"], expected)
        result = subprocess.run([sys.executable, "-B", "-m", demo.__name__, "--scenario", "bad"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("invalid choice", result.stderr)

    def test_offline_no_files_created(self):
        script = '''
import os, sys
def audit(name, args):
    if name.startswith(("socket.", "subprocess.")):
        raise AssertionError(name)
    if name == "open" and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
        raise AssertionError("file write")
sys.addaudithook(audit)
from examples.vlm_semantic_drift.demo_pipeline import main
assert main(["--scenario", "candidate"]) == 0
assert not any(name in sys.modules for name in ("torch", "psycopg2", "requests"))
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-B", "-c", script], cwd=folder, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(folder).iterdir()), [])
            self.assertEqual(json.loads(result.stdout)["decision"]["status"], "CANDIDATE")

    def test_import_without_pillow_or_execution(self):
        script = '''
import sys
from examples.vlm_semantic_drift import demo_pipeline
assert "PIL" not in sys.modules
assert callable(demo_pipeline.run_demo)
'''
        result = subprocess.run([sys.executable, "-B", "-S", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
