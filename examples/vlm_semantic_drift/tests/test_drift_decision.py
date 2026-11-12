"""Synthetic boundary cases for demonstration candidate decisions."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import json
import subprocess
import sys
import unittest

from examples.vlm_semantic_drift.drift_decision import DriftDecisionConfig, evaluate_drift
from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.prompt_builder import build_hypothesis_prompts
from examples.vlm_semantic_drift.response_parser import parse_vlm_response
from examples.vlm_semantic_drift.sliding_window import SlidingWindow
from examples.vlm_semantic_drift.vlm_adapter import MockVLMAdapter, VLMRawResponse


def snapshot(tokens=(), size=10, minimum=5, provider="mock", is_mock=True):
    window = SlidingWindow(size, minimum)
    for index, token in enumerate(tokens):
        event = DetectionEvent(
            event_id="synthetic-decision-" + str(index),
            timestamp="2026-01-01T00:00:{:02d}Z".format(index),
            camera_id="synthetic-camera", class_label="person",
            confidence=0.8, bbox=(0, 0, 8, 8),
        )
        if token in ("YES", "NO", "UNKNOWN") and provider == "mock" and is_mock:
            raw = MockVLMAdapter(token).generate(build_hypothesis_prompts(event))
        else:
            raw = VLMRawResponse(provider, is_mock, token)
        window.add(event, parse_vlm_response(raw))
    return window.snapshot()


class DriftDecisionTests(unittest.TestCase):
    def test_empty_window(self):
        result = evaluate_drift(snapshot())
        self.assertEqual(result.status, "INSUFFICIENT_DATA")
        self.assertFalse(result.is_candidate)
        self.assertIsNone(result.mismatch_ratio)
        self.assertIsNone(result.provider)
        self.assertIsNone(result.is_mock)
        self.assertIsNone(result.camera_id)
        self.assertEqual(result.consecutive_mismatches, 0)

    def test_minimum_not_met(self):
        result = evaluate_drift(snapshot(["NO"] * 4))
        self.assertEqual((result.status, result.valid_samples), ("INSUFFICIENT_DATA", 4))
        self.assertEqual((result.mismatch_ratio, result.consecutive_mismatches), (1.0, 4))

    def test_exact_three_boundaries(self):
        result = evaluate_drift(snapshot(["YES", "YES", "NO", "NO", "NO"]))
        self.assertEqual((result.valid_samples, result.mismatch_ratio, result.consecutive_mismatches),
                         (5, 0.6, 3))
        self.assertEqual(result.status, "CANDIDATE")
        self.assertTrue(result.is_candidate)
        self.assertTrue(result.reason)

    def test_ratio_below_threshold_streak_only(self):
        result = evaluate_drift(snapshot(["YES"] * 3 + ["NO"] * 3))
        self.assertEqual(result.mismatch_ratio, 0.5)
        self.assertEqual(result.consecutive_mismatches, 3)
        self.assertEqual(result.status, "NO_CANDIDATE")

    def test_ratio_above_and_four_recent_no(self):
        result = evaluate_drift(snapshot(["YES"] + ["NO"] * 4))
        self.assertEqual((result.mismatch_ratio, result.consecutive_mismatches), (0.8, 4))
        self.assertTrue(result.is_candidate)

    def test_ratio_only_recent_two_no(self):
        result = evaluate_drift(snapshot(["NO", "YES", "YES", "NO", "NO"]))
        self.assertEqual((result.mismatch_ratio, result.consecutive_mismatches), (0.6, 2))
        self.assertEqual(result.status, "NO_CANDIDATE")

    def test_latest_yes_breaks_streak(self):
        result = evaluate_drift(snapshot(["NO", "NO", "NO", "YES", "YES"]))
        self.assertEqual((result.mismatch_ratio, result.consecutive_mismatches), (0.6, 0))
        self.assertEqual(result.status, "NO_CANDIDATE")

    def test_latest_unknown_breaks_streak(self):
        result = evaluate_drift(snapshot(["NO"] * 5 + ["UNKNOWN"]))
        self.assertEqual((result.valid_samples, result.consecutive_mismatches), (5, 0))
        self.assertEqual(result.status, "NO_CANDIDATE")

    def test_latest_invalid_breaks_streak(self):
        result = evaluate_drift(snapshot(["NO"] * 5 + ["YES because..."]))
        self.assertEqual((result.valid_samples, result.consecutive_mismatches), (5, 0))
        self.assertEqual(result.status, "NO_CANDIDATE")

    def test_unknown_insufficient_example(self):
        result = evaluate_drift(snapshot(["YES", "NO", "NO", "NO", "UNKNOWN"]))
        self.assertEqual((result.valid_samples, result.mismatch_ratio, result.consecutive_mismatches),
                         (4, 0.75, 0))
        self.assertEqual(result.status, "INSUFFICIENT_DATA")

    def test_no_valid_samples(self):
        result = evaluate_drift(snapshot(["UNKNOWN", "bad"]))
        self.assertIsNone(result.mismatch_ratio)
        self.assertEqual(result.valid_samples, 0)
        self.assertFalse(result.is_candidate)

    def test_internal_breaks_limit_recent_run(self):
        for breaker in ("YES", "UNKNOWN", "bad"):
            with self.subTest(breaker=breaker):
                result = evaluate_drift(snapshot(["NO"] * 3 + [breaker] + ["NO"] * 2))
                self.assertEqual(result.consecutive_mismatches, 2)
                self.assertEqual(result.status, "NO_CANDIDATE")

    def test_eviction_bounds_streak_and_ratio(self):
        result = evaluate_drift(snapshot(["NO"] * 8, size=5))
        self.assertEqual((result.valid_samples, result.consecutive_mismatches), (5, 5))
        self.assertTrue(result.is_candidate)
        result = evaluate_drift(snapshot(["NO"] * 5 + ["YES"] * 5, size=5))
        self.assertEqual((result.mismatch_ratio, result.consecutive_mismatches), (0.0, 0))
        self.assertEqual(result.status, "NO_CANDIDATE")

    def test_custom_config_and_window_readiness_independent(self):
        config = DriftDecisionConfig(3, 1.0, 3)
        stats = snapshot(["NO"] * 3)
        self.assertFalse(stats.ready)
        self.assertTrue(evaluate_drift(stats, config).is_candidate)
        stats = snapshot(["NO"] * 3, minimum=2)
        self.assertTrue(stats.ready)
        self.assertEqual(evaluate_drift(stats).status, "INSUFFICIENT_DATA")
        self.assertEqual(evaluate_drift(snapshot(["NO"] * 3, size=3, minimum=2)).status,
                         "INSUFFICIENT_DATA")

    def test_custom_ratio_boundary(self):
        stats = snapshot(["YES", "YES", "NO", "NO", "NO"])
        self.assertTrue(evaluate_drift(stats, DriftDecisionConfig(5, 0.6, 3)).is_candidate)
        self.assertFalse(evaluate_drift(stats, DriftDecisionConfig(5, 0.6000001, 3)).is_candidate)
        self.assertTrue(evaluate_drift(stats, DriftDecisionConfig(5, 0.5999999, 3)).is_candidate)

    def test_invalid_config_values(self):
        for field, values in (
            ("min_valid_samples", (True, 0, 1, -1, 5.0, "5", None)),
            ("consecutive_mismatches", (True, 0, 1, -1, 6, 3.0, "3", None)),
            ("mismatch_ratio_threshold", (True, 0, -0.1, 1.1, float("nan"),
                                          float("inf"), "0.6", None)),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    DriftDecisionConfig(**{field: value})

    def test_wrong_types(self):
        for wrong in (None, {}, [], "stats", 1):
            with self.assertRaises(TypeError):
                evaluate_drift(wrong)
        for wrong in ({}, False, 5, "config"):
            with self.assertRaises(TypeError):
                evaluate_drift(snapshot(), wrong)

    def test_snapshot_count_and_ratio_validation(self):
        stats = snapshot(["NO"] * 5)
        for changes in ({"valid_count": True}, {"yes_count": 1}, {"total_count": 4},
                        {"window_size": 4}, {"mismatch_ratio": None}, {"mismatch_ratio": 0.6},
                        {"mismatch_ratio": float("nan")}, {"ready": 1}, {"ready": False},
                        {"entries": list(stats.entries)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate_drift(replace(stats, **changes))
        with self.assertRaises(ValueError):
            evaluate_drift(replace(snapshot(), mismatch_ratio=0.0))

    def test_mixed_stream_and_corrupted_entry_rejected(self):
        stats = snapshot(["NO"] * 5)
        first = stats.entries[0]
        for entry in (
            replace(first, camera_id="other"), replace(first, class_label="car"),
            replace(first, response=replace(first.response, provider="other")),
            replace(first, response=replace(first.response, is_mock=False)),
            replace(first, response=replace(first.response, supports_h1=False)),
            replace(first, timestamp=stats.entries[-1].timestamp),
            replace(first, event_id=stats.entries[-1].event_id),
        ):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                evaluate_drift(replace(stats, entries=(entry,) + stats.entries[1:]))

    def test_metadata_and_mock_pipeline(self):
        for provider, is_mock in (("mock", True), ("synthetic-provider", False)):
            result = evaluate_drift(snapshot(["YES", "YES", "NO", "NO", "NO"],
                                             provider=provider, is_mock=is_mock))
            self.assertEqual((result.provider, result.is_mock), (provider, is_mock))
            self.assertEqual((result.camera_id, result.class_label), ("synthetic-camera", "person"))
            self.assertTrue(result.is_candidate)

    def test_immutable_repeatable_and_input_preserved(self):
        stats = snapshot(["NO"] * 5)
        before = replace(stats)
        config = DriftDecisionConfig()
        result = evaluate_drift(stats, config)
        self.assertEqual(stats, before)
        self.assertEqual(result, evaluate_drift(stats, config))
        with self.assertRaises(FrozenInstanceError):
            result.status = "NO_CANDIDATE"
        with self.assertRaises(FrozenInstanceError):
            config.min_valid_samples = 2

    def test_defaults_match_existing_json(self):
        data = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
        config = DriftDecisionConfig()
        for name in ("min_valid_samples", "mismatch_ratio_threshold", "consecutive_mismatches"):
            self.assertEqual(getattr(config, name), data["aggregation"][name])

    def test_offline_standard_library_execution(self):
        script = '''
import sys
def audit(name, args):
    if name.startswith(("socket.", "subprocess.")):
        raise AssertionError(name)
    if name == "open" and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
        raise AssertionError("file write")
sys.addaudithook(audit)
from examples.vlm_semantic_drift.drift_decision import evaluate_drift
from examples.vlm_semantic_drift.sliding_window import SlidingWindow
assert evaluate_drift(SlidingWindow().snapshot()).status == "INSUFFICIENT_DATA"
assert not any(name in sys.modules for name in ("PIL", "torch", "psycopg2", "requests"))
'''
        result = subprocess.run([sys.executable, "-B", "-S", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
