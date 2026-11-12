"""Demonstration candidate rules, not a statistical drift test."""

from dataclasses import dataclass, fields
from datetime import datetime
import math
from typing import Optional

from .response_parser import ParsedVLMResponse, parse_vlm_response
from .sliding_window import WindowEntry, WindowStats
from .vlm_adapter import VLMRawResponse


@dataclass(frozen=True)
class DriftDecisionConfig:
    """Uncalibrated demonstration thresholds; no configuration file is read."""

    min_valid_samples: int = 5
    mismatch_ratio_threshold: float = 0.6
    consecutive_mismatches: int = 3

    def __post_init__(self) -> None:
        if type(self.min_valid_samples) is not int or self.min_valid_samples < 2:
            raise ValueError("min_valid_samples must be an integer >= 2")
        if (type(self.consecutive_mismatches) is not int
                or not 2 <= self.consecutive_mismatches <= self.min_valid_samples):
            raise ValueError("consecutive_mismatches must be an integer from 2 to min_valid_samples")
        value = self.mismatch_ratio_threshold
        if type(value) not in (int, float) or not 0 < value <= 1 or not math.isfinite(value):
            raise ValueError("mismatch_ratio_threshold must be finite and in (0, 1]")


@dataclass(frozen=True)
class DriftDecisionResult:
    """Rule outcome and provenance, never a claim of model accuracy."""

    status: str
    is_candidate: bool
    valid_samples: int
    mismatch_ratio: Optional[float]
    consecutive_mismatches: int
    min_valid_samples: int
    mismatch_ratio_threshold: float
    consecutive_mismatches_threshold: int
    reason: str
    provider: Optional[str]
    is_mock: Optional[bool]
    camera_id: Optional[str]
    class_label: Optional[str]


def _validate_stats(stats: WindowStats) -> None:
    """Reject inconsistent public dataclasses instead of making false candidates."""
    if not isinstance(stats, WindowStats):
        raise TypeError("stats must be WindowStats")
    if type(stats.entries) is not tuple:
        raise ValueError("entries must be an immutable tuple")
    for name in ("window_size", "min_valid_samples", "total_count", "yes_count",
                 "no_count", "unknown_count", "invalid_count", "valid_count"):
        if type(getattr(stats, name)) is not int or getattr(stats, name) < 0:
            raise ValueError(name + " must be a non-negative integer")
    if not 1 <= stats.min_valid_samples <= stats.window_size:
        raise ValueError("invalid window configuration")
    if stats.total_count != len(stats.entries) or stats.total_count > stats.window_size:
        raise ValueError("total_count contradicts entries or window capacity")
    counts = dict.fromkeys(("YES", "NO", "UNKNOWN", "INVALID"), 0)
    stream = None
    last_time = None
    seen = set()
    for entry in stats.entries:
        if not isinstance(entry, WindowEntry) or not isinstance(entry.response, ParsedVLMResponse):
            raise ValueError("invalid window entry or parsed response")
        for value in (entry.event_id, entry.camera_id, entry.class_label):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("entry identity must contain non-empty strings")
        if not isinstance(entry.timestamp, str) or not entry.timestamp.endswith(("Z", "+00:00")):
            raise ValueError("entry timestamp must be UTC")
        stamp = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
        if entry.event_id in seen or (last_time is not None and stamp <= last_time):
            raise ValueError("entries must have unique IDs and increasing timestamps")
        seen.add(entry.event_id)
        last_time = stamp
        response = entry.response
        checked = parse_vlm_response(VLMRawResponse(
            response.provider, response.is_mock, response.raw_response,
        ))
        for field in fields(ParsedVLMResponse):
            value, expected = getattr(response, field.name), getattr(checked, field.name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError("response contradicts parser contract")
        key = (entry.camera_id, entry.class_label, response.provider, response.is_mock)
        if stream is not None and key != stream:
            raise ValueError("mixed streams are not allowed")
        stream = key
        counts[response.status] += 1
    for status, name in (("YES", "yes_count"), ("NO", "no_count"),
                         ("UNKNOWN", "unknown_count"), ("INVALID", "invalid_count")):
        if getattr(stats, name) != counts[status]:
            raise ValueError(name + " contradicts entries")
    if stats.valid_count != stats.yes_count + stats.no_count:
        raise ValueError("valid_count must equal YES + NO")
    if stats.valid_count == 0:
        if stats.mismatch_ratio is not None:
            raise ValueError("mismatch_ratio must be None without valid samples")
    elif (type(stats.mismatch_ratio) is not float
          or stats.mismatch_ratio != stats.no_count / stats.valid_count):
        raise ValueError("mismatch_ratio contradicts counts")
    if type(stats.ready) is not bool or stats.ready != (stats.valid_count >= stats.min_valid_samples):
        raise ValueError("ready contradicts window configuration")


def evaluate_drift(
    stats: WindowStats, config: Optional[DriftDecisionConfig] = None,
) -> DriftDecisionResult:
    """Apply all three inclusive thresholds without changing the snapshot."""
    if config is None:
        config = DriftDecisionConfig()
    if not isinstance(config, DriftDecisionConfig):
        raise TypeError("config must be DriftDecisionConfig")
    DriftDecisionConfig(config.min_valid_samples, config.mismatch_ratio_threshold,
                        config.consecutive_mismatches)
    _validate_stats(stats)
    streak = 0
    for entry in reversed(stats.entries):
        if entry.response.status != "NO":
            break
        streak += 1
    if stats.valid_count < config.min_valid_samples:
        status = "INSUFFICIENT_DATA"
        reason = "Valid sample count is below the configured minimum."
    else:
        failures = []
        if stats.mismatch_ratio is None or stats.mismatch_ratio < config.mismatch_ratio_threshold:
            failures.append("Mismatch ratio is below the threshold or unavailable.")
        if streak < config.consecutive_mismatches:
            failures.append("Recent consecutive NO count is below the threshold.")
        status = "NO_CANDIDATE" if failures else "CANDIDATE"
        reason = " ".join(failures) if failures else "All three demonstration thresholds are met."
    latest = stats.entries[-1] if stats.entries else None
    return DriftDecisionResult(
        status=status, is_candidate=status == "CANDIDATE", valid_samples=stats.valid_count,
        mismatch_ratio=stats.mismatch_ratio, consecutive_mismatches=streak,
        min_valid_samples=config.min_valid_samples,
        mismatch_ratio_threshold=config.mismatch_ratio_threshold,
        consecutive_mismatches_threshold=config.consecutive_mismatches, reason=reason,
        provider=latest.response.provider if latest else None,
        is_mock=latest.response.is_mock if latest else None,
        camera_id=latest.camera_id if latest else None,
        class_label=latest.class_label if latest else None,
    )
