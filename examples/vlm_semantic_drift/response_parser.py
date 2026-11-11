"""Strict single-response normalization, without inference or drift decisions."""

from dataclasses import dataclass
from typing import Optional

from .vlm_adapter import VLMRawResponse


@dataclass(frozen=True)
class ParsedVLMResponse:
    """Normalized polarity and unchanged provenance of a raw response."""

    provider: str
    is_mock: bool
    raw_response: str
    normalized_response: Optional[str]
    status: str
    label_match: Optional[bool]
    supports_h1: Optional[bool]
    valid: bool


def parse_vlm_response(response: VLMRawResponse) -> ParsedVLMResponse:
    """Accept only exact uppercase tokens after str.strip().

    Unrecognized strings produce INVALID. Wrong object types raise TypeError;
    corrupted fields raise ValueError under the existing raw-response contract.
    UNKNOWN is valid abstention; INVALID is a format failure, never H1 evidence.
    """
    if not isinstance(response, VLMRawResponse):
        raise TypeError("response must be VLMRawResponse")
    # Revalidate public fields without changing the existing interface or input.
    VLMRawResponse(response.provider, response.is_mock, response.raw_response)
    token = response.raw_response.strip()
    valid = token in ("YES", "NO", "UNKNOWN")
    label_match = True if token == "YES" else False if token == "NO" else None
    supports_h1 = False if token == "YES" else True if token == "NO" else None
    return ParsedVLMResponse(
        provider=response.provider,
        is_mock=response.is_mock,
        raw_response=response.raw_response,
        normalized_response=token if valid else None,
        status=token if valid else "INVALID",
        label_match=label_match,
        supports_h1=supports_h1,
        valid=valid,
    )
