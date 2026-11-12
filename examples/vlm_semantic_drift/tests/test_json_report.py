"""Synthetic report contracts, metadata checks and offline integration."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from examples.vlm_semantic_drift.drift_decision import DriftDecisionConfig, evaluate_drift
from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.human_feedback import HumanFeedbackLog, create_feedback
from examples.vlm_semantic_drift.json_report import build_drift_report, serialize_drift_report
from examples.vlm_semantic_drift.prompt_builder import build_hypothesis_prompts
from examples.vlm_semantic_drift.response_parser import parse_vlm_response
from examples.vlm_semantic_drift.sliding_window import SlidingWindow
from examples.vlm_semantic_drift.vlm_adapter import MockVLMAdapter, VLMRawResponse


NOW = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)


def inputs(tokens=("YES", "YES", "NO", "NO", "NO"), minimum=5, config=None):
    window = SlidingWindow(min_valid_samples=minimum)
    for index, token in enumerate(tokens):
        event = DetectionEvent(
            event_id="synthetic-report-" + str(index),
            timestamp="2026-01-01T00:00:{:02d}Z".format(index),
            camera_id="synthetic-camera", class_label="person", confidence=0.8, bbox=(0, 0, 8, 8),
        )
        raw = (MockVLMAdapter(token).generate(build_hypothesis_prompts(event))
               if token in ("YES", "NO", "UNKNOWN") else VLMRawResponse("mock", True, token))
        window.add(event, parse_vlm_response(raw))
    stats = window.snapshot()
    return stats, evaluate_drift(stats, config)


def review(decision, verdict="CONFIRMED", identifier="f1", **changes):
    record = create_feedback(
        decision, decision_ref="decision_demo_01", feedback_id=identifier,
        reviewer_id="reviewer_demo_" + identifier, verdict=verdict,
        reviewed_at="2026-01-01T00:01:00Z", comment="합성 자료 검토 의견",
    )
    return replace(record, **changes)


def report(stats, decision, records=(), **changes):
    args = dict(decision_ref="decision_demo_01", generated_at=NOW)
    args.update(changes)
    return build_drift_report(stats, decision, records, **args)


class JSONReportTests(unittest.TestCase):
    def test_candidate_schema(self):
        value = report(*inputs())
        self.assertEqual(set(value), {"schema_version", "decision_ref", "generated_at", "source",
                                     "stream", "window", "decision", "human_feedback", "limitations"})
        self.assertEqual(value["schema_version"], "1.0")
        self.assertEqual(value["decision"]["status"], "CANDIDATE")
        self.assertIs(value["decision"]["is_candidate"], True)
        self.assertTrue(value["limitations"]["demo_only"])

    def test_no_candidate_without_feedback(self):
        value = report(*inputs(("NO", "NO", "NO", "YES", "YES")))
        self.assertEqual(value["decision"]["status"], "NO_CANDIDATE")
        self.assertEqual(value["human_feedback"], [])

    def test_insufficient_and_empty_metadata(self):
        value = report(*inputs(()))
        self.assertEqual(value["decision"]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(value["source"], {"provider": None, "is_mock": None})
        self.assertEqual(value["stream"], {"camera_id": None, "class_label": None})
        self.assertIsNone(value["limitations"]["is_mock"])

    def test_no_feedback_default(self):
        self.assertEqual(report(*inputs())["human_feedback"], [])

    def test_confirmed(self):
        stats, decision = inputs()
        value = report(stats, decision, [review(decision)])
        self.assertEqual(value["human_feedback"][0]["verdict"], "CONFIRMED")
        self.assertEqual(value["decision"]["status"], "CANDIDATE")

    def test_false_positive(self):
        stats, decision = inputs()
        value = report(stats, decision, [review(decision, "FALSE_POSITIVE")])
        self.assertEqual(value["human_feedback"][0]["verdict"], "FALSE_POSITIVE")
        self.assertTrue(value["decision"]["is_candidate"])

    def test_deferred(self):
        stats, decision = inputs()
        value = report(stats, decision, [review(decision, "DEFERRED")])
        self.assertEqual(value["human_feedback"][0]["verdict"], "DEFERRED")

    def test_conflicting_reviews_preserve_log_order(self):
        stats, decision = inputs()
        log = HumanFeedbackLog()
        for index, verdict in enumerate(("DEFERRED", "FALSE_POSITIVE", "CONFIRMED")):
            log.add(review(decision, verdict, "f" + str(index)))
        value = report(stats, decision, log.list_for_decision("decision_demo_01"))
        self.assertEqual([r["verdict"] for r in value["human_feedback"]],
                         ["DEFERRED", "FALSE_POSITIVE", "CONFIRMED"])
        self.assertEqual(len(value["human_feedback"]), 3)
        self.assertEqual(value["decision"]["status"], "CANDIDATE")

    def test_reference_mismatch_and_duplicate_id(self):
        stats, decision = inputs()
        record = review(decision)
        for records in ((replace(record, decision_ref="other"),), (record, record)):
            with self.assertRaises(ValueError):
                report(stats, decision, records)

    def test_feedback_metadata_conflicts(self):
        stats, decision = inputs()
        record = review(decision)
        for name, value in (("camera_id", "other"), ("class_label", "car"),
                            ("provider", "other"), ("is_mock", False)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                report(stats, decision, [replace(record, **{name: value})])
        with self.assertRaises(ValueError):
            report(*inputs(("YES",) * 5), [record])

    def test_wrong_stats_and_decision_types(self):
        stats, decision = inputs()
        for wrong in (None, {}, "value", True):
            with self.assertRaises(TypeError):
                report(wrong, decision)
            with self.assertRaises(TypeError):
                report(stats, wrong)

    def test_wrong_feedback_container_and_item(self):
        stats, decision = inputs()
        for wrong in (None, {}, "", set(), iter(()), [None], [decision]):
            with self.subTest(wrong=wrong), self.assertRaises(TypeError):
                report(stats, decision, wrong)

    def test_reference_validation(self):
        for wrong in ("", " ", " padded", "bad\nref", "x" * 129):
            with self.assertRaises(ValueError):
                report(*inputs(), decision_ref=wrong)
        with self.assertRaises(TypeError):
            report(*inputs(), decision_ref=True)

    def test_generated_at_validation(self):
        for wrong in (None, "2026-01-01T00:00:00Z", True, 1):
            with self.assertRaises(TypeError):
                report(*inputs(), generated_at=wrong)
        for wrong in (datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=9)))):
            with self.assertRaises(ValueError):
                report(*inputs(), generated_at=wrong)

    def test_utc_conversion(self):
        value = report(*inputs(), generated_at=NOW.replace(microsecond=123))
        self.assertEqual(value["generated_at"], "2026-01-01T00:02:00.000123Z")
        self.assertEqual(report(*inputs())["generated_at"], "2026-01-01T00:02:00Z")

    def test_source_and_stream_preserved(self):
        value = report(*inputs())
        self.assertEqual(value["source"], {"provider": "mock", "is_mock": True})
        self.assertEqual(value["stream"], {"camera_id": "synthetic-camera", "class_label": "person"})
        self.assertIs(value["limitations"]["is_mock"], True)

    def test_window_counts_and_null(self):
        value = report(*inputs(("YES", "NO", "UNKNOWN", "bad")))
        window = value["window"]
        self.assertEqual([window[n] for n in ("total_entries", "yes_count", "no_count",
                                             "unknown_count", "invalid_count", "valid_samples")],
                         [4, 1, 1, 1, 1, 2])
        self.assertEqual(window["mismatch_ratio"], 0.5)
        text = serialize_drift_report(report(*inputs(("UNKNOWN", "bad"))))
        self.assertIn('"mismatch_ratio": null', text)
        self.assertIsNone(json.loads(text)["window"]["mismatch_ratio"])

    def test_thresholds_and_ready_are_preserved(self):
        stats, decision = inputs(("NO",) * 3, config=DriftDecisionConfig(3, 0.75, 2))
        value = report(stats, decision)
        self.assertFalse(value["window"]["ready"])
        self.assertEqual(value["window"]["min_valid_samples"], 5)
        self.assertEqual(value["decision"]["min_valid_samples"], 3)
        self.assertEqual(value["decision"]["mismatch_ratio_threshold"], 0.75)
        self.assertEqual(value["decision"]["consecutive_mismatches_threshold"], 2)
        self.assertEqual(value["decision"]["status"], "CANDIDATE")

    def test_no_decision_recalculation(self):
        stats, decision = inputs()
        with patch("examples.vlm_semantic_drift.drift_decision.evaluate_drift",
                   side_effect=AssertionError("must not evaluate")):
            value = report(stats, decision)
        self.assertEqual(value["decision"]["reason"], decision.reason)

    def test_inconsistent_shared_facts_and_metadata(self):
        stats, decision = inputs()
        for changes in ({"valid_samples": 4}, {"valid_samples": True}, {"mismatch_ratio": 0.5},
                        {"is_candidate": 1}, {"camera_id": "other"}, {"class_label": "car"},
                        {"provider": "other"}, {"is_mock": 1}, {"status": "bad"},
                        {"consecutive_mismatches": True}, {"reason": object()}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                report(stats, replace(decision, **changes))
        with self.assertRaises(ValueError):
            report(replace(stats, no_count=0), decision)

    def test_json_roundtrip_unicode_determinism(self):
        stats, decision = inputs()
        record = review(decision)
        value = report(stats, decision, [record])
        text = serialize_drift_report(value)
        self.assertEqual(json.loads(text), value)
        self.assertIn("합성 자료 검토 의견", text)
        self.assertIn('\n  "decision":', text)
        self.assertEqual(text, serialize_drift_report(report(stats, decision, [record])))
        self.assertEqual(text, serialize_drift_report(dict(reversed(list(value.items())))))

    def test_report_mutation_does_not_change_inputs(self):
        stats, decision = inputs()
        record = review(decision)
        before = (replace(stats), replace(decision), replace(record))
        value = report(stats, decision, [record])
        value["window"]["no_count"] = 0
        value["decision"]["status"] = "changed"
        value["source"]["is_mock"] = False
        value["human_feedback"][0]["comment"] = "changed"
        self.assertEqual((stats, decision, record), before)
        self.assertEqual(report(stats, decision, [record])["human_feedback"][0]["comment"], record.comment)

    def test_serializer_invalid_values(self):
        for wrong in (object(), NOW, {1, 2}, b"bytes"):
            with self.assertRaises(TypeError):
                serialize_drift_report({"value": wrong})
        for wrong in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                serialize_drift_report({"nested": [wrong]})
        with self.assertRaises(TypeError):
            serialize_drift_report({1: "ambiguous"})
        with self.assertRaises(TypeError):
            serialize_drift_report([])
        cycle = {}
        cycle["self"] = cycle
        with self.assertRaises(ValueError):
            serialize_drift_report(cycle)

    def test_tuple_boolean_null_serialization(self):
        value = {"values": (True, False, None)}
        self.assertEqual(json.loads(serialize_drift_report(value)), {"values": [True, False, None]})
        self.assertIsInstance(value["values"], tuple)

    def test_offline_standard_library_pipeline(self):
        script = '''
import sys
def audit(name, args):
    if name.startswith(("socket.", "subprocess.")):
        raise AssertionError(name)
    if name == "open" and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
        raise AssertionError("file write")
sys.addaudithook(audit)
from examples.vlm_semantic_drift.tests.test_json_report import inputs, review, report
from examples.vlm_semantic_drift.json_report import serialize_drift_report
import json
stats, decision = inputs()
value = json.loads(serialize_drift_report(report(stats, decision, [review(decision)])))
assert value["source"]["is_mock"] is True
assert not any(name in sys.modules for name in ("PIL", "torch", "psycopg2", "requests"))
'''
        result = subprocess.run([sys.executable, "-B", "-S", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
