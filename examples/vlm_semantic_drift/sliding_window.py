"""Bounded single-stream response statistics, without drift decisions."""

from collections import deque
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Deque, Optional, Tuple

from .event_schema import DetectionEvent
from .response_parser import ParsedVLMResponse, parse_vlm_response
from .vlm_adapter import VLMRawResponse


@dataclass(frozen=True)
class WindowEntry:
    """Independent immutable event identity and parsed response."""

    event_id: str
    timestamp: str
    camera_id: str
    class_label: str
    response: ParsedVLMResponse


@dataclass(frozen=True)
class WindowStats:
    """A point-in-time snapshot; readiness is not a drift decision."""

    entries: Tuple[WindowEntry, ...]
    window_size: int
    min_valid_samples: int
    total_count: int
    yes_count: int
    no_count: int
    unknown_count: int
    invalid_count: int
    valid_count: int
    mismatch_ratio: Optional[float]
    ready: bool


class SlidingWindow:
    """Keep recent N observations in strictly increasing UTC event time.

    The first entry binds camera, label, provider and is_mock for this instance.
    Duplicate IDs are rejected within retained entries only. Create a new
    instance to reset history and stream binding. Not thread-safe.
    """

    def __init__(self, window_size: int = 10, min_valid_samples: int = 5) -> None:
        if type(window_size) is not int or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if type(min_valid_samples) is not int or not 1 <= min_valid_samples <= window_size:
            raise ValueError("min_valid_samples must be an integer from 1 to window_size")
        self._window_size = window_size
        self._min_valid_samples = min_valid_samples
        self._entries: Deque[WindowEntry] = deque(maxlen=window_size)
        self._stream: Optional[Tuple[str, str, str, bool]] = None
        self._last_time: Optional[datetime] = None

    def add(self, event: DetectionEvent, parsed_response: ParsedVLMResponse) -> None:
        """Validate before mutation; wrong types/values raise TypeError/ValueError."""
        if not isinstance(event, DetectionEvent):
            raise TypeError("event must be DetectionEvent")
        if not isinstance(parsed_response, ParsedVLMResponse):
            raise TypeError("parsed_response must be ParsedVLMResponse")
        checked_event = DetectionEvent.from_dict(event.to_dict())
        checked_response = parse_vlm_response(VLMRawResponse(
            parsed_response.provider, parsed_response.is_mock, parsed_response.raw_response,
        ))
        # Dataclass equality alone would accept 1 in place of True.
        for field in fields(ParsedVLMResponse):
            value = getattr(parsed_response, field.name)
            expected = getattr(checked_response, field.name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError("parsed response contradicts the parser contract: " + field.name)
        stream = (checked_event.camera_id, checked_event.class_label,
                  checked_response.provider, checked_response.is_mock)
        timestamp = datetime.fromisoformat(checked_event.timestamp.replace("Z", "+00:00"))
        if self._stream is not None and stream != self._stream:
            raise ValueError("camera, class, provider and is_mock must match this stream")
        if any(entry.event_id == checked_event.event_id for entry in self._entries):
            raise ValueError("duplicate event_id in the retained window")
        if self._last_time is not None and timestamp <= self._last_time:
            raise ValueError("event timestamp must be strictly increasing")
        self._entries.append(WindowEntry(
            checked_event.event_id, checked_event.timestamp, checked_event.camera_id,
            checked_event.class_label, checked_response,
        ))
        self._stream = stream
        self._last_time = timestamp

    def snapshot(self) -> WindowStats:
        """Return immutable entries and counts without changing the window."""
        entries = tuple(self._entries)
        counts = dict.fromkeys(("YES", "NO", "UNKNOWN", "INVALID"), 0)
        for entry in entries:
            counts[entry.response.status] += 1
        valid_count = counts["YES"] + counts["NO"]
        return WindowStats(
            entries=entries, window_size=self._window_size,
            min_valid_samples=self._min_valid_samples, total_count=len(entries),
            yes_count=counts["YES"], no_count=counts["NO"],
            unknown_count=counts["UNKNOWN"], invalid_count=counts["INVALID"],
            valid_count=valid_count,
            mismatch_ratio=counts["NO"] / valid_count if valid_count else None,
            ready=valid_count >= self._min_valid_samples,
        )
