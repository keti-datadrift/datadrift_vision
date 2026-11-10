"""Offline adapter contract and deterministic, explicitly simulated responses."""

from dataclasses import dataclass, fields
import json
from typing import Any, Dict, Optional, Protocol

from .prompt_builder import PromptResult
from .roi_extractor import ROIResult


@dataclass(frozen=True)
class VLMRawResponse:
    """Raw provider output, with no label polarity or drift interpretation."""

    provider: str
    is_mock: bool
    raw_response: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if type(self.is_mock) is not bool:
            raise ValueError("is_mock must be bool")
        if not isinstance(self.raw_response, str):
            raise ValueError("raw_response must be a string")


class VLMAdapter(Protocol):
    """Common provider interface; ROI ownership stays with the caller."""

    def generate(self, prompt: PromptResult, roi: Optional[ROIResult] = None) -> VLMRawResponse:
        """Return raw output without parsing or interpreting it."""
        ...


def _unique_object(pairs: Any) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate prompt data key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant: " + value)


def _validate_prompt(prompt: PromptResult) -> None:
    if not isinstance(prompt, PromptResult):
        raise TypeError("prompt must be PromptResult")
    for field in fields(PromptResult):
        value = getattr(prompt, field.name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(field.name + " must be a non-empty string")
    data = json.loads(prompt.data_json, object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)
    if not isinstance(data, dict) or set(data) != {"class_label", "quality"}:
        raise ValueError("prompt data must contain class_label and quality")
    if not isinstance(data["class_label"], str) or not data["class_label"].strip():
        raise ValueError("prompt class_label must be a non-empty string")
    if data["quality"] is not None and not isinstance(data["quality"], dict):
        raise ValueError("prompt quality must be a JSON object or null")


@dataclass(frozen=True)
class MockVLMAdapter:
    """Return the configured token regardless of label, quality or ROI content."""

    response: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if not isinstance(self.response, str) or self.response not in ("YES", "NO", "UNKNOWN"):
            raise ValueError("response must be exactly YES, NO, or UNKNOWN")

    def generate(self, prompt: PromptResult, roi: Optional[ROIResult] = None) -> VLMRawResponse:
        """Validate input shape, then reproduce a fixture token without inference.

        ROI is type-checked only: no pixels, dimensions, files or services are
        accessed. Prompt validation is structural, not semantic verification.
        """
        _validate_prompt(prompt)
        if roi is not None and not isinstance(roi, ROIResult):
            raise TypeError("roi must be ROIResult or None")
        return VLMRawResponse(provider="mock", is_mock=True, raw_response=self.response)
