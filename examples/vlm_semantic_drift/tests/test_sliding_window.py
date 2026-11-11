"""Synthetic single-stream aggregation and compatibility tests."""

from dataclasses import FrozenInstanceError, replace
import subprocess
import sys
import unittest

from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.prompt_builder import build_hypothesis_prompts
from examples.vlm_semantic_drift.response_parser import parse_vlm_response
from examples.vlm_semantic_drift.sliding_window import SlidingWindow
from examples.vlm_semantic_drift.vlm_adapter import MockVLMAdapter, VLMRawResponse


def event(index=0, **changes):
    value = DetectionEvent(
        event_id="synthetic-" + str(index), timestamp="2026-01-01T00:00:{:02d}Z".format(index),
        camera_id="synthetic-camera", class_label="person", confidence=0.8, bbox=(0, 0, 8, 8),
    )
    return replace(value, **changes)


def response(token, provider="mock", is_mock=True):
    return parse_vlm_response(VLMRawResponse(provider, is_mock, token))


class SlidingWindowTests(unittest.TestCase):
    def test_empty_defaults(self):
        stats = SlidingWindow().snapshot()
        self.assertEqual((stats.window_size, stats.min_valid_samples), (10, 5))
        self.assertEqual((stats.total_count, stats.valid_count, stats.entries), (0, 0, ()))
        self.assertIsNone(stats.mismatch_ratio)
        self.assertFalse(stats.ready)

    def test_yes(self):
        window = SlidingWindow(1, 1)
        window.add(event(), response("YES"))
        stats = window.snapshot()
        self.assertEqual((stats.yes_count, stats.no_count, stats.valid_count), (1, 0, 1))
        self.assertEqual(stats.mismatch_ratio, 0.0)
        self.assertTrue(stats.ready)

    def test_no(self):
        window = SlidingWindow(1, 1)
        window.add(event(), response("NO"))
        self.assertEqual(window.snapshot().no_count, 1)
        self.assertEqual(window.snapshot().mismatch_ratio, 1.0)

    def test_unknown(self):
        window = SlidingWindow()
        window.add(event(), response("UNKNOWN"))
        stats = window.snapshot()
        self.assertEqual((stats.total_count, stats.unknown_count, stats.valid_count), (1, 1, 0))
        self.assertIsNone(stats.mismatch_ratio)

    def test_invalid(self):
        window = SlidingWindow()
        window.add(event(), response("yes because..."))
        stats = window.snapshot()
        self.assertEqual((stats.total_count, stats.invalid_count, stats.valid_count), (1, 1, 0))
        self.assertIsNone(stats.mismatch_ratio)

    def test_mixed_counts_ratio_and_readiness(self):
        window = SlidingWindow(10, 3)
        for index, token in enumerate(("YES", "NO", "UNKNOWN", "bad", "NO")):
            window.add(event(index), response(token))
        stats = window.snapshot()
        self.assertEqual((stats.yes_count, stats.no_count, stats.unknown_count, stats.invalid_count),
                         (1, 2, 1, 1))
        self.assertEqual((stats.total_count, stats.valid_count), (5, 3))
        self.assertAlmostEqual(stats.mismatch_ratio, 2 / 3)
        self.assertTrue(stats.ready)

    def test_eviction_and_readiness_can_drop(self):
        window = SlidingWindow(2, 2)
        window.add(event(0), response("YES"))
        self.assertFalse(window.snapshot().ready)
        window.add(event(1), response("NO"))
        self.assertTrue(window.snapshot().ready)
        window.add(event(2), response("UNKNOWN"))
        stats = window.snapshot()
        self.assertEqual([entry.event_id for entry in stats.entries], ["synthetic-1", "synthetic-2"])
        self.assertEqual((stats.total_count, stats.valid_count, stats.mismatch_ratio), (2, 1, 1.0))
        self.assertFalse(stats.ready)
        window.add(event(3), response("bad"))
        self.assertIsNone(window.snapshot().mismatch_ratio)

    def test_many_observations_bounded(self):
        window = SlidingWindow(3, 2)
        for index in range(50):
            window.add(event(index), response("NO"))
            self.assertLessEqual(window.snapshot().total_count, 3)
        self.assertEqual(window.snapshot().entries[0].event_id, "synthetic-47")

    def test_time_order_and_equal_time_rejection_atomic(self):
        window = SlidingWindow()
        window.add(event(2), response("YES"))
        before = window.snapshot()
        for candidate in (event(1), event(3, timestamp=event(2).timestamp)):
            with self.assertRaises(ValueError):
                window.add(candidate, response("NO"))
            self.assertEqual(window.snapshot(), before)
        window.add(event(3), response("YES"))

    def test_fractional_utc_time_compared_as_datetime(self):
        window = SlidingWindow()
        for index, stamp in enumerate(("2026-01-01T00:00:00+00:00",
                                       "2026-01-01T00:00:00.1Z", "2026-01-01T00:00:01Z")):
            window.add(event(index, timestamp=stamp), response("YES"))
        self.assertEqual(window.snapshot().total_count, 3)

    def test_duplicate_retained_id_rejected(self):
        window = SlidingWindow()
        window.add(event(), response("YES"))
        before = window.snapshot()
        with self.assertRaises(ValueError):
            window.add(event(1, event_id=event().event_id), response("NO"))
        self.assertEqual(window.snapshot(), before)

    def test_evicted_id_policy(self):
        window = SlidingWindow(1, 1)
        window.add(event(0), response("YES"))
        window.add(event(1), response("YES"))
        with self.assertRaises(ValueError):
            window.add(event(0), response("NO"))
        window.add(event(2, event_id=event(0).event_id), response("NO"))
        self.assertEqual(window.snapshot().no_count, 1)

    def test_stream_separation(self):
        for candidate, parsed in (
            (event(1, camera_id="other"), response("NO")),
            (event(1, class_label="car"), response("NO")),
            (event(1), response("NO", provider="other")),
            (event(1), response("NO", is_mock=False)),
        ):
            with self.subTest(candidate=candidate, parsed=parsed):
                window = SlidingWindow(1, 1)
                window.add(event(), response("YES"))
                before = window.snapshot()
                with self.assertRaises(ValueError):
                    window.add(candidate, parsed)
                self.assertEqual(window.snapshot(), before)

    def test_metadata_and_original_inputs_preserved(self):
        original_event, parsed = event(), response(" NO \n", provider="synthetic-provider", is_mock=False)
        before = (original_event.to_dict(), replace(parsed))
        window = SlidingWindow()
        window.add(original_event, parsed)
        entry = window.snapshot().entries[0]
        self.assertEqual(entry.response, parsed)
        self.assertIsNot(entry.response, parsed)
        self.assertEqual((entry.camera_id, entry.class_label), (original_event.camera_id, original_event.class_label))
        self.assertEqual((original_event.to_dict(), parsed), before)

    def test_invalid_settings(self):
        for size, minimum in ((0, 1), (-1, 1), (True, 1), (2.0, 1), ("2", 1),
                              (2, 0), (2, -1), (2, 3), (2, True), (2, 1.5), (2, None)):
            with self.subTest(size=size, minimum=minimum), self.assertRaises(ValueError):
                SlidingWindow(size, minimum)

    def test_invalid_input_types(self):
        window = SlidingWindow()
        for wrong in (None, {}, "YES", 1):
            with self.assertRaises(TypeError):
                window.add(wrong, response("YES"))
            with self.assertRaises(TypeError):
                window.add(event(), wrong)
        self.assertEqual(window.snapshot().total_count, 0)

    def test_inconsistent_parsed_fields_rejected(self):
        parsed = response("YES")
        for changes in ({"status": "NO"}, {"label_match": 1}, {"supports_h1": True},
                        {"valid": 1}, {"normalized_response": None}, {"is_mock": 1},
                        {"provider": ""}, {"raw_response": None}):
            with self.subTest(changes=changes):
                window = SlidingWindow()
                with self.assertRaises(ValueError):
                    window.add(event(), replace(parsed, **changes))
                self.assertEqual(window.snapshot().total_count, 0)
                window.add(event(camera_id="other"), parsed)

    def test_corrupted_event_rejected(self):
        candidate = event()
        object.__setattr__(candidate, "timestamp", "invalid")
        window = SlidingWindow()
        with self.assertRaises(ValueError):
            window.add(candidate, response("YES"))
        self.assertEqual(window.snapshot().total_count, 0)

    def test_snapshot_immutable_and_stable(self):
        window = SlidingWindow(1, 1)
        window.add(event(), response("YES"))
        before = window.snapshot()
        for obj, name, value in ((before, "ready", False), (before.entries[0], "event_id", "x"),
                                 (before.entries[0].response, "status", "NO")):
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, name, value)
        self.assertEqual(before, window.snapshot())
        window.add(event(1), response("NO"))
        self.assertEqual(before.yes_count, 1)
        self.assertEqual(before.entries[0].event_id, event().event_id)

    def test_mock_pipeline_and_new_instance_reset(self):
        window = SlidingWindow()
        for index, token in enumerate(("YES", "NO", "UNKNOWN")):
            candidate = event(index)
            raw = MockVLMAdapter(token).generate(build_hypothesis_prompts(candidate))
            window.add(candidate, parse_vlm_response(raw))
        self.assertEqual(window.snapshot().mismatch_ratio, 0.5)
        self.assertTrue(all(entry.response.is_mock for entry in window.snapshot().entries))
        reset = SlidingWindow()
        reset.add(event(camera_id="other"), response("UNKNOWN"))
        self.assertEqual(reset.snapshot().total_count, 1)

    def test_offline_standard_library_execution(self):
        script = '''
import sys
def audit(name, args):
    if name.startswith(("socket.", "subprocess.")):
        raise AssertionError(name)
    if name == "open" and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
        raise AssertionError("file write")
sys.addaudithook(audit)
from examples.vlm_semantic_drift.sliding_window import SlidingWindow
assert SlidingWindow().snapshot().mismatch_ratio is None
assert not any(name in sys.modules for name in ("PIL", "torch", "psycopg2", "requests"))
'''
        result = subprocess.run([sys.executable, "-B", "-S", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
