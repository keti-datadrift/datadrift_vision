"""Validated, immutable detection events for the standalone sample."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import re
from typing import Any, Dict, Optional, Tuple


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a non-empty string")
    if value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(name + " must not contain padding or control characters")


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(name + " must be a finite number (not bool)")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(name + " must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _unique_object(pairs: Any) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field: " + key)
        result[key] = value
    return result


@dataclass(frozen=True)
class DetectionEvent:
    """One detector claim; it is not a verified label or a drift decision."""

    event_id: str
    timestamp: str
    camera_id: str
    class_label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    image_path: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be integer 1")
        for name in ("event_id", "camera_id", "class_label"):
            _text(getattr(self, name), name)
        if not isinstance(self.timestamp, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)",
            self.timestamp,
        ):
            raise ValueError("timestamp must be a UTC ISO 8601 string with Z or +00:00")
        try:
            parsed = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp contains an invalid date or time") from exc
        object.__setattr__(self, "timestamp", parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"))
        confidence = _number(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1 inclusive")
        object.__setattr__(self, "confidence", confidence)
        if not isinstance(self.bbox, (list, tuple)) or len(self.bbox) != 4:
            raise ValueError("bbox must contain four xyxy pixel coordinates")
        bbox = tuple(_number(value, "bbox coordinate") for value in self.bbox)
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("bbox must have positive width and height")
        object.__setattr__(self, "bbox", bbox)
        if self.image_path is not None:
            _text(self.image_path, "image_path")
            parts = self.image_path.split("/")
            if "\\" in self.image_path or ":" in self.image_path or any(
                part in ("", ".", "..") for part in parts
            ):
                raise ValueError("image_path must be a relative POSIX path without traversal")

    def to_dict(self) -> Dict[str, Any]:
        """Return an independent JSON-compatible object."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "class_label": self.class_label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "image_path": self.image_path,
        }

    def to_json(self) -> str:
        """Serialize deterministically without accessing images or services."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionEvent":
        """Validate versioned input; reject missing and unexpected fields."""
        if not isinstance(data, dict):
            raise ValueError("event must be a JSON object")
        required = {"schema_version", "event_id", "timestamp", "camera_id", "class_label", "confidence", "bbox"}
        if not required.issubset(data) or set(data) - required - {"image_path"}:
            raise ValueError("event has missing or unexpected fields")
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "DetectionEvent":
        """Parse one event, rejecting ambiguous duplicate JSON keys."""
        if not isinstance(text, str):
            raise ValueError("event JSON must be a string")
        return cls.from_dict(json.loads(text, object_pairs_hook=_unique_object))
