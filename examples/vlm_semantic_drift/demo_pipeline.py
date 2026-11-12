"""Offline synthetic end-to-end demo; MockVLM never interprets image pixels."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import sys
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

from .event_schema import DetectionEvent
from .roi_extractor import extract_roi
from .image_quality import analyze_image_quality
from .prompt_builder import build_hypothesis_prompts
from .vlm_adapter import MockVLMAdapter, VLMRawResponse
from .response_parser import parse_vlm_response
from .sliding_window import SlidingWindow
from .drift_decision import DriftDecisionConfig, evaluate_drift
from .human_feedback import create_feedback, HumanFeedbackLog
from .json_report import build_drift_report, serialize_drift_report

if TYPE_CHECKING:
    from PIL import Image


def _synthetic_image() -> "Image.Image":
    """Create an RGB checkerboard in memory, without a detector or file access."""
    from PIL import Image

    image = Image.new("RGB", (32, 32))
    image.putdata([(level, level, level) for y in range(32) for x in range(32)
                   for level in (64 if (x // 4 + y // 4) % 2 == 0 else 192,)])
    return image


def run_demo(scenario: str = "candidate") -> Dict[str, Any]:
    """Run existing modules with fixed fixtures and return a schema 1.0 report."""
    scenarios = {
        "candidate": ("YES", "YES", "NO", "NO", "NO"),
        "no_candidate": ("NO", "NO", "NO", "YES", "YES"),
        "insufficient_data": ("YES", "NO", "UNKNOWN", "INVALID"),
    }
    if not isinstance(scenario, str):
        raise TypeError("scenario must be a string")
    if scenario not in scenarios:
        raise ValueError("scenario must be candidate, no_candidate or insufficient_data")
    window = SlidingWindow(window_size=10, min_valid_samples=5)
    for index, token in enumerate(scenarios[scenario]):
        event = DetectionEvent(
            event_id="synthetic-" + scenario + "-" + str(index),
            timestamp="2026-01-01T00:00:0{}Z".format(index),
            camera_id="synthetic-camera", class_label="person", confidence=0.8,
            bbox=(-2, 4, 28, 36),
        )
        with closing(_synthetic_image()) as image:
            roi = extract_roi(image, event)
            try:
                quality = analyze_image_quality(roi.image)
                prompt = build_hypothesis_prompts(event, quality=quality)
                if token == "INVALID":
                    # A malformed fixture tests the parser; Mock's contract stays unchanged.
                    raw = VLMRawResponse("mock", True, "YES because this is a synthetic malformed response")
                else:
                    raw = MockVLMAdapter(token).generate(prompt, roi=roi)
                parsed = parse_vlm_response(raw)
                window.add(event, parsed)
            finally:
                roi.image.close()
    stats = window.snapshot()
    decision = evaluate_drift(stats, DriftDecisionConfig())
    decision_ref = "decision_demo_" + scenario
    log = HumanFeedbackLog()
    if scenario == "candidate":
        log.add(create_feedback(
            decision, decision_ref=decision_ref, feedback_id="feedback_demo_01",
            reviewer_id="reviewer_demo_01", verdict="CONFIRMED",
            reviewed_at="2026-01-01T00:01:00Z", comment="Synthetic review only; not proof of drift.",
        ))
    return build_drift_report(
        stats, decision, log.list_for_decision(decision_ref), decision_ref=decision_ref,
        generated_at=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print only JSON on success; report execution errors on stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("candidate", "no_candidate", "insufficient_data"),
                        default="candidate", help="fixed synthetic response scenario (default: candidate)")
    args = parser.parse_args(argv)
    try:
        output = serialize_drift_report(run_demo(args.scenario))
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        print("Demo failed: " + str(exc), file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
