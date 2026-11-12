"""Synthetic reviews and offline compatibility for the human feedback sample."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import subprocess
import sys
import unittest

from examples.vlm_semantic_drift.drift_decision import evaluate_drift
from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.human_feedback import HumanFeedbackLog, create_feedback
from examples.vlm_semantic_drift.prompt_builder import build_hypothesis_prompts
from examples.vlm_semantic_drift.response_parser import parse_vlm_response
from examples.vlm_semantic_drift.sliding_window import SlidingWindow
from examples.vlm_semantic_drift.vlm_adapter import MockVLMAdapter


def decision(tokens=("YES", "YES", "NO", "NO", "NO")):
    window = SlidingWindow()
    for index, token in enumerate(tokens):
        event = DetectionEvent(
            event_id="synthetic-feedback-" + str(index),
            timestamp="2026-01-01T00:00:0{}Z".format(index),
            camera_id="synthetic-camera", class_label="person", confidence=0.8, bbox=(0, 0, 8, 8),
        )
        raw = MockVLMAdapter(token).generate(build_hypothesis_prompts(event))
        window.add(event, parse_vlm_response(raw))
    return evaluate_drift(window.snapshot())


def feedback(source=None, **changes):
    args = dict(decision_ref="decision_demo_01", feedback_id="feedback_demo_01",
                reviewer_id="reviewer_demo_01", verdict="CONFIRMED",
                reviewed_at="2026-01-01T00:01:00Z")
    args.update(changes)
    return create_feedback(decision() if source is None else source, **args)


class HumanFeedbackTests(unittest.TestCase):
    def test_confirmed(self):
        record = feedback()
        self.assertEqual((record.verdict, record.decision_status), ("CONFIRMED", "CANDIDATE"))
        self.assertEqual(record.comment, "")

    def test_false_positive_does_not_change_decision(self):
        source = decision()
        before = replace(source)
        record = feedback(source, verdict="FALSE_POSITIVE")
        self.assertEqual(record.verdict, "FALSE_POSITIVE")
        self.assertEqual(source, before)
        self.assertTrue(source.is_candidate)

    def test_deferred_stays_deferred(self):
        log = HumanFeedbackLog()
        record = feedback(verdict="DEFERRED")
        log.add(record)
        self.assertEqual(log.get(record.feedback_id).verdict, "DEFERRED")

    def test_required_identifiers(self):
        for name in ("feedback_id", "decision_ref", "reviewer_id"):
            for value in ("", " ", " padded", "padded ", "line\nbreak", "x" * 129):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    feedback(**{name: value})
            self.assertEqual(getattr(feedback(**{name: "x" * 128}), name), "x" * 128)

    def test_invalid_verdict(self):
        for value in ("confirmed", "YES", "", " CONFIRMED", "NO_CANDIDATE"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                feedback(verdict=value)

    def test_wrong_input_types(self):
        for name in ("feedback_id", "decision_ref", "reviewer_id", "verdict", "reviewed_at", "comment"):
            for value in (None, True, 1, [], {}):
                with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                    feedback(**{name: value})
        for value in (None, {}, "CANDIDATE", True):
            with self.assertRaises(TypeError):
                create_feedback(value, decision_ref="d", feedback_id="f", reviewer_id="r",
                                verdict="CONFIRMED", reviewed_at="2026-01-01T00:01:00Z")

    def test_non_candidates_rejected(self):
        for source in (decision(()), decision(("YES",) * 5)):
            with self.subTest(status=source.status), self.assertRaises(ValueError):
                feedback(source)
        for value in (False, 1, None):
            with self.assertRaises(ValueError):
                feedback(replace(decision(), is_candidate=value))

    def test_utc_normalization(self):
        first = feedback(reviewed_at="2026-01-01T00:01:00.123+00:00")
        second = feedback(reviewed_at="2026-01-01T00:01:00.123000Z")
        self.assertEqual(first, second)
        self.assertEqual(first.reviewed_at, "2026-01-01T00:01:00.123000Z")

    def test_invalid_timestamps(self):
        for value in ("2026-01-01T00:01:00", "2026-01-01", "2026-01-01T09:01:00+09:00",
                      "2026-01-01T00:01:00-00:00", "2026-02-30T00:01:00Z",
                      "2026-01-01T00:01:60Z", "2026-01-01T00:01:00.1234567Z", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                feedback(reviewed_at=value)
        with self.assertRaises(TypeError):
            feedback(reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_comment_limits(self):
        text = "Synthetic review\n\tNo personal information."
        self.assertEqual(feedback(comment=text).comment, text)
        self.assertEqual(len(feedback(comment="x" * 1000).comment), 1000)
        for text in ("x" * 1001, "bad\x00text", "bad\rtext"):
            with self.assertRaises(ValueError):
                feedback(comment=text)

    def test_immutable_record(self):
        record = feedback()
        with self.assertRaises(FrozenInstanceError):
            record.verdict = "FALSE_POSITIVE"
        with self.assertRaises(ValueError):
            replace(record, verdict="INVALID")

    def test_metadata_preserved(self):
        source = decision()
        record = feedback(source)
        for field in ("camera_id", "class_label", "provider", "is_mock"):
            self.assertEqual(getattr(source, field), getattr(record, field))
        self.assertIs(record.is_mock, True)
        other = feedback(replace(source, provider="synthetic-provider", is_mock=False))
        self.assertEqual((other.provider, other.is_mock), ("synthetic-provider", False))

    def test_invalid_metadata(self):
        for field in ("camera_id", "class_label", "provider"):
            with self.assertRaises(TypeError):
                feedback(replace(decision(), **{field: None}))
            with self.assertRaises(ValueError):
                feedback(replace(decision(), **{field: ""}))
        for value in (None, 0, 1, "true"):
            with self.assertRaises(TypeError):
                feedback(replace(decision(), is_mock=value))

    def test_add_get_and_missing_id(self):
        log = HumanFeedbackLog()
        record = feedback()
        self.assertIsNone(log.add(record))
        self.assertEqual(log.get(record.feedback_id), record)
        with self.assertRaises(KeyError):
            log.get("missing")
        self.assertEqual(log.list_for_decision("missing"), ())

    def test_duplicate_id_rejected_without_overwrite(self):
        log = HumanFeedbackLog()
        record = feedback()
        log.add(record)
        for duplicate in (record, replace(record, decision_ref="other", verdict="FALSE_POSITIVE")):
            with self.assertRaises(ValueError):
                log.add(duplicate)
        self.assertEqual(log.get(record.feedback_id), record)
        self.assertEqual(log.list_for_decision("other"), ())

    def test_multiple_reviewers_conflicts_and_registration_order(self):
        log = HumanFeedbackLog()
        records = (
            feedback(feedback_id="f1", reviewed_at="2026-01-02T00:00:00Z"),
            feedback(feedback_id="f2", reviewer_id="reviewer_demo_02", verdict="FALSE_POSITIVE"),
            feedback(feedback_id="f3", verdict="DEFERRED"),
        )
        log.add(records[0])
        log.add(feedback(feedback_id="unrelated", decision_ref="other"))
        for record in records[1:]:
            log.add(record)
        self.assertEqual(log.list_for_decision("decision_demo_01"), records)
        self.assertEqual([r.verdict for r in log.list_for_decision("decision_demo_01")],
                         ["CONFIRMED", "FALSE_POSITIVE", "DEFERRED"])

    def test_queries_return_independent_immutable_snapshots(self):
        log = HumanFeedbackLog()
        record = feedback()
        log.add(record)
        old = log.list_for_decision(record.decision_ref)
        self.assertIsInstance(old, tuple)
        self.assertIsNot(log.get(record.feedback_id), record)
        # Even deliberate frozen-dataclass bypass on caller copies cannot alter storage.
        object.__setattr__(record, "verdict", "DEFERRED")
        returned = log.get(record.feedback_id)
        object.__setattr__(returned, "verdict", "FALSE_POSITIVE")
        self.assertEqual(log.get(record.feedback_id).verdict, "CONFIRMED")
        log.add(feedback(feedback_id="f2"))
        self.assertEqual(len(old), 1)

    def test_log_validates_inputs_and_corrupted_records(self):
        log = HumanFeedbackLog()
        for value in (None, {}, "record"):
            with self.assertRaises(TypeError):
                log.add(value)
        for method in (log.get, log.list_for_decision):
            with self.assertRaises(ValueError):
                method("")
            with self.assertRaises(TypeError):
                method(True)
        record = feedback()
        object.__setattr__(record, "verdict", "bad")
        with self.assertRaises(ValueError):
            log.add(record)
        self.assertEqual(log.list_for_decision(record.decision_ref), ())

    def test_instances_are_independent(self):
        first, second = HumanFeedbackLog(), HumanFeedbackLog()
        first.add(feedback())
        self.assertEqual(second.list_for_decision("decision_demo_01"), ())

    def test_offline_standard_library_pipeline(self):
        script = '''
import sys
def audit(name, args):
    if name.startswith(("socket.", "subprocess.")):
        raise AssertionError(name)
    if name == "open" and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
        raise AssertionError("file write")
sys.addaudithook(audit)
from examples.vlm_semantic_drift.tests.test_human_feedback import feedback
from examples.vlm_semantic_drift.human_feedback import HumanFeedbackLog
record = feedback()
log = HumanFeedbackLog()
log.add(record)
assert log.get(record.feedback_id).is_mock is True
assert not any(name in sys.modules for name in ("PIL", "torch", "psycopg2", "requests"))
'''
        result = subprocess.run([sys.executable, "-B", "-S", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
