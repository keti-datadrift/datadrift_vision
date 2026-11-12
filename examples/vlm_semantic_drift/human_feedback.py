"""In-memory human reviews kept separate from automatic candidate decisions."""

from dataclasses import dataclass, replace
from datetime import datetime
import re
from typing import Dict, Tuple

from .drift_decision import DriftDecisionResult


def _text(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(name + " must be a string")
    if not value.strip() or value != value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError(name + " must be non-empty without padding or control characters")


def _identifier(value: str, name: str) -> None:
    _text(value, name)
    if len(value) > 128:
        raise ValueError(name + " must contain at most 128 characters")


def _utc(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("reviewed_at must be a UTC ISO 8601 string")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", value):
        raise ValueError("reviewed_at must use UTC with Z or +00:00")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class HumanFeedbackRecord:
    """One review, not scientific confirmation or an authenticated identity."""

    feedback_id: str
    decision_ref: str
    reviewer_id: str
    reviewed_at: str
    verdict: str
    comment: str
    decision_status: str
    camera_id: str
    class_label: str
    provider: str
    is_mock: bool

    def __post_init__(self) -> None:
        for name in ("feedback_id", "decision_ref", "reviewer_id"):
            _identifier(getattr(self, name), name)
        for name in ("camera_id", "class_label", "provider"):
            _text(getattr(self, name), name)
        if not isinstance(self.verdict, str):
            raise TypeError("verdict must be a string")
        if self.verdict not in ("CONFIRMED", "FALSE_POSITIVE", "DEFERRED"):
            raise ValueError("verdict must be CONFIRMED, FALSE_POSITIVE or DEFERRED")
        if self.decision_status != "CANDIDATE":
            raise ValueError("only CANDIDATE decisions can be reviewed")
        if type(self.is_mock) is not bool:
            raise TypeError("is_mock must be bool")
        if not isinstance(self.comment, str):
            raise TypeError("comment must be a string")
        if len(self.comment) > 1000:
            raise ValueError("comment must contain at most 1000 characters")
        if any(ord(c) < 32 and c not in "\n\t" for c in self.comment):
            raise ValueError("comment contains unsupported control characters")
        object.__setattr__(self, "reviewed_at", _utc(self.reviewed_at))


def create_feedback(
    decision: DriftDecisionResult, *, decision_ref: str, feedback_id: str,
    reviewer_id: str, verdict: str, reviewed_at: str, comment: str = "",
) -> HumanFeedbackRecord:
    """Copy candidate metadata; caller supplies reference and review evidence.

    No decision is recalculated, authenticated or overwritten. Use synthetic
    reviewer IDs and comments only; personal information is not automatically detected.
    """
    if not isinstance(decision, DriftDecisionResult):
        raise TypeError("decision must be DriftDecisionResult")
    if decision.status != "CANDIDATE" or decision.is_candidate is not True:
        raise ValueError("only CANDIDATE decisions with is_candidate=True can be reviewed")
    return HumanFeedbackRecord(
        feedback_id=feedback_id, decision_ref=decision_ref, reviewer_id=reviewer_id,
        reviewed_at=reviewed_at, verdict=verdict, comment=comment,
        decision_status=decision.status, camera_id=decision.camera_id,
        class_label=decision.class_label, provider=decision.provider, is_mock=decision.is_mock,
    )


class HumanFeedbackLog:
    """Append-only in-memory records in registration order, without consensus."""

    def __init__(self) -> None:
        self._records: Dict[str, HumanFeedbackRecord] = {}

    def add(self, record: HumanFeedbackRecord) -> None:
        """Validate and copy a record; reject duplicate IDs without overwriting."""
        if not isinstance(record, HumanFeedbackRecord):
            raise TypeError("record must be HumanFeedbackRecord")
        checked = replace(record)
        if checked.feedback_id in self._records:
            raise ValueError("feedback_id already exists")
        self._records[checked.feedback_id] = checked

    def get(self, feedback_id: str) -> HumanFeedbackRecord:
        """Return an immutable copy; a missing ID raises KeyError."""
        _identifier(feedback_id, "feedback_id")
        return replace(self._records[feedback_id])

    def list_for_decision(self, decision_ref: str) -> Tuple[HumanFeedbackRecord, ...]:
        """Return immutable copies in registration order, or an empty tuple."""
        _identifier(decision_ref, "decision_ref")
        return tuple(replace(record) for record in self._records.values()
                     if record.decision_ref == decision_ref)
