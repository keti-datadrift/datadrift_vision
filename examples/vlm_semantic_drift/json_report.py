"""Deterministic in-memory reports; no decision evaluation or persistence."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import math
from typing import Any, Dict, Sequence

from .drift_decision import DriftDecisionConfig, DriftDecisionResult, _validate_stats
from .human_feedback import HumanFeedbackRecord
from .sliding_window import WindowStats


def _reference(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("decision_ref must be a string")
    if (not value.strip() or value != value.strip() or len(value) > 128
            or any(ord(c) < 32 for c in value)):
        raise ValueError("decision_ref must be unpadded, non-empty and at most 128 characters")


def _generated_at(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("generated_at must be a UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise ValueError("generated_at must have timezone information and UTC offset zero")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_pair(stats: WindowStats, decision: DriftDecisionResult) -> None:
    """Check structure and shared facts, without re-running candidate rules."""
    if not isinstance(decision, DriftDecisionResult):
        raise TypeError("decision must be DriftDecisionResult")
    _validate_stats(stats)
    DriftDecisionConfig(decision.min_valid_samples, decision.mismatch_ratio_threshold,
                        decision.consecutive_mismatches_threshold)
    if decision.status not in ("CANDIDATE", "NO_CANDIDATE", "INSUFFICIENT_DATA"):
        raise ValueError("unknown decision status")
    if type(decision.is_candidate) is not bool or decision.is_candidate != (decision.status == "CANDIDATE"):
        raise ValueError("is_candidate contradicts status")
    if type(decision.valid_samples) is not int or decision.valid_samples != stats.valid_count:
        raise ValueError("decision valid_samples conflicts with window")
    if (type(decision.mismatch_ratio) is not type(stats.mismatch_ratio)
            or decision.mismatch_ratio != stats.mismatch_ratio):
        raise ValueError("decision mismatch_ratio conflicts with window")
    if (type(decision.consecutive_mismatches) is not int
            or not 0 <= decision.consecutive_mismatches <= stats.no_count):
        raise ValueError("invalid consecutive_mismatches")
    if not isinstance(decision.reason, str) or not decision.reason.strip():
        raise ValueError("reason must be a non-empty string")
    for name in ("camera_id", "class_label", "provider", "is_mock"):
        expected = None
        if stats.entries:
            entry = stats.entries[0]
            expected = getattr(entry.response if name in ("provider", "is_mock") else entry, name)
        value = getattr(decision, name)
        if type(value) is not type(expected) or value != expected:
            raise ValueError("decision metadata conflicts with window: " + name)


def build_drift_report(
    stats: WindowStats, decision: DriftDecisionResult,
    feedback_records: Sequence[HumanFeedbackRecord] = (), *,
    decision_ref: str, generated_at: datetime,
) -> Dict[str, Any]:
    """Copy supplied statistics, decision and ordered reviews into schema 1.0.

    Inputs must refer to the same decision; references are caller-managed.
    The caller must supply synthetic identifiers and non-personal comments.
    """
    _reference(decision_ref)
    timestamp = _generated_at(generated_at)
    _validate_pair(stats, decision)
    if not isinstance(feedback_records, (list, tuple)):
        raise TypeError("feedback_records must be a list or tuple of HumanFeedbackRecord")
    feedback = []
    seen = set()
    for record in feedback_records:
        if not isinstance(record, HumanFeedbackRecord):
            raise TypeError("feedback item must be HumanFeedbackRecord")
        checked = replace(record)  # Revalidate without altering the caller's record.
        if checked.decision_ref != decision_ref:
            raise ValueError("feedback decision_ref conflicts with report")
        if checked.feedback_id in seen:
            raise ValueError("duplicate feedback_id")
        seen.add(checked.feedback_id)
        for name in ("camera_id", "class_label", "provider", "is_mock", "decision_status"):
            expected = decision.status if name == "decision_status" else getattr(decision, name)
            if getattr(checked, name) != expected:
                raise ValueError("feedback metadata conflicts with decision: " + name)
        feedback.append({name: getattr(checked, name) for name in (
            "feedback_id", "decision_ref", "reviewer_id", "reviewed_at", "verdict", "comment",
        )})
    report = {
        "schema_version": "1.0",
        "decision_ref": decision_ref,
        "generated_at": timestamp,
        "source": {"provider": decision.provider, "is_mock": decision.is_mock},
        "stream": {"camera_id": decision.camera_id, "class_label": decision.class_label},
        "window": {
            "total_entries": stats.total_count, "yes_count": stats.yes_count,
            "no_count": stats.no_count, "unknown_count": stats.unknown_count,
            "invalid_count": stats.invalid_count, "valid_samples": stats.valid_count,
            "mismatch_ratio": stats.mismatch_ratio, "ready": stats.ready,
            "min_valid_samples": stats.min_valid_samples,
        },
        "decision": {name: getattr(decision, name) for name in (
            "status", "is_candidate", "consecutive_mismatches", "mismatch_ratio_threshold",
            "consecutive_mismatches_threshold", "min_valid_samples", "reason",
        )},
        "human_feedback": feedback,
        "limitations": {
            "demo_only": True, "is_mock": decision.is_mock,
            "description": "Demonstration only. Candidate and confirmed reviews do not establish actual "
                           "data drift; no-candidate does not prove its absence. Mock statistics are "
                           "not actual model performance metrics.",
        },
    }
    return report


def _check_json(value: Any, active: set) -> None:
    """Reject unsupported values, lossy non-string keys and circular containers."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("NaN and Infinity are not allowed")
        return
    if type(value) not in (dict, list, tuple):
        raise TypeError("unsupported JSON value: " + type(value).__name__)
    if id(value) in active:
        raise ValueError("circular JSON data")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("JSON object keys must be strings")
                _check_json(item, active)
        else:
            for item in value:
                _check_json(item, active)
    finally:
        active.remove(id(value))


def serialize_drift_report(report: Dict[str, Any]) -> str:
    """Serialize JSON-compatible report data; do not infer or repair its schema."""
    if type(report) is not dict:
        raise TypeError("report must be a dictionary")
    _check_json(report, set())
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
