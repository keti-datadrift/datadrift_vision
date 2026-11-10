"""Synthetic adapter contract tests without a model or remote service."""

from dataclasses import FrozenInstanceError, asdict, fields, replace
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.prompt_builder import PromptResult, build_hypothesis_prompts
from examples.vlm_semantic_drift.roi_extractor import ROIResult
from examples.vlm_semantic_drift.vlm_adapter import MockVLMAdapter, VLMAdapter, VLMRawResponse


class VLMAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = DetectionEvent("synthetic-mock", "2026-01-01T00:00:00Z",
                                    "synthetic-camera", "person", 0.8, (0, 0, 8, 8))
        self.prompt = build_hypothesis_prompts(self.event)

    def test_default_unknown_and_interface(self) -> None:
        adapter: VLMAdapter = MockVLMAdapter()
        self.assertEqual(adapter.generate(self.prompt), VLMRawResponse("mock", True, "UNKNOWN"))

    def test_yes(self) -> None:
        self.assertEqual(MockVLMAdapter("YES").generate(self.prompt).raw_response, "YES")

    def test_no(self) -> None:
        self.assertEqual(MockVLMAdapter("NO").generate(self.prompt).raw_response, "NO")

    def test_explicit_unknown(self) -> None:
        self.assertEqual(MockVLMAdapter("UNKNOWN").generate(self.prompt).raw_response, "UNKNOWN")

    def test_repeatability_and_label_independence(self) -> None:
        adapter = MockVLMAdapter("NO")
        expected = adapter.generate(self.prompt)
        for _ in range(3):
            self.assertEqual(adapter.generate(self.prompt), expected)
        other = build_hypothesis_prompts(replace(self.event, class_label="car"))
        self.assertEqual(adapter.generate(other), expected)
        self.assertEqual(MockVLMAdapter("NO").generate(self.prompt), expected)

    def test_invalid_settings(self) -> None:
        for value in (None, True, 1, "", "yes", " NO", "UNKNOWN\n", "MAYBE"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MockVLMAdapter(value)

    def test_invalid_prompt_types_and_fields(self) -> None:
        for value in (None, "YES", {}, self.event):
            with self.subTest(value=value), self.assertRaises(TypeError):
                MockVLMAdapter().generate(value)
        for field in fields(PromptResult):
            for value in (None, 1, "", " \n "):
                with self.subTest(field=field.name, value=value), self.assertRaises(ValueError):
                    MockVLMAdapter().generate(replace(self.prompt, **{field.name: value}))

    def test_invalid_prompt_json(self) -> None:
        for data in ("{", "[]", "null", "{}", '{"class_label":"person","quality":NaN}',
                     '{"class_label":"person","class_label":"car","quality":null}',
                     json.dumps({"class_label": 1, "quality": None}),
                     json.dumps({"class_label": "person", "quality": []}),
                     json.dumps({"class_label": "person", "quality": None, "extra": True})):
            with self.subTest(data=data), self.assertRaises(ValueError):
                MockVLMAdapter().generate(replace(self.prompt, data_json=data))

    def test_metadata_without_interpretation(self) -> None:
        result = MockVLMAdapter("YES").generate(self.prompt)
        self.assertEqual(asdict(result), {"provider": "mock", "is_mock": True, "raw_response": "YES"})
        self.assertFalse(hasattr(result, "label_match"))
        self.assertFalse(hasattr(result, "supports_h1"))

    def test_input_and_result_immutability(self) -> None:
        before = asdict(self.prompt)
        adapter = MockVLMAdapter()
        result = adapter.generate(self.prompt)
        self.assertEqual(asdict(self.prompt), before)
        with self.assertRaises(FrozenInstanceError):
            result.is_mock = False
        with self.assertRaises(FrozenInstanceError):
            adapter.response = "YES"

    def test_raw_response_validation_does_not_parse(self) -> None:
        result = VLMRawResponse("future-provider", False, "unparsed response")
        self.assertEqual(result.raw_response, "unparsed response")
        for args in (("", True, "YES"), ("mock", 1, "YES"), ("mock", True, None)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                VLMRawResponse(*args)

    def test_roi_type_and_no_pixel_access(self) -> None:
        for roi in ("image.png", {}, 1):
            with self.subTest(roi=roi), self.assertRaises(TypeError):
                MockVLMAdapter().generate(self.prompt, roi)
        class UnreadableImage:
            def __getattribute__(self, name):
                raise AssertionError("Mock must not inspect image contents")
        roi = ROIResult(UnreadableImage(), (0, 0, 8, 8))
        self.assertEqual(MockVLMAdapter("YES").generate(self.prompt, roi).raw_response, "YES")

    def test_existing_modules_integration(self) -> None:
        from PIL import Image
        from examples.vlm_semantic_drift.roi_extractor import extract_roi
        from examples.vlm_semantic_drift.image_quality import analyze_image_quality
        with Image.new("RGB", (8, 8), (128, 128, 128)) as image:
            roi = extract_roi(image, self.event)
            try:
                quality = analyze_image_quality(roi.image)
                prompt = build_hypothesis_prompts(self.event, quality)
                before = roi.image.tobytes()
                with patch.object(Image, "open", side_effect=AssertionError("No file access")):
                    result = MockVLMAdapter("NO").generate(prompt, roi)
                self.assertEqual(result.raw_response, "NO")
                self.assertFalse(quality.assessable)
                self.assertEqual(roi.image.tobytes(), before)
                self.assertEqual(roi.bbox, (0, 0, 8, 8))
            finally:
                roi.image.close()

    def test_offline_standard_library_execution(self) -> None:
        root = Path(__file__).resolve().parents[3]
        script = r'''
import os
import sys
sys.path.insert(0, sys.argv[1])
def audit(event, args):
    if event.startswith("socket.") or event in {"subprocess.Popen", "os.system", "os.mkdir", "os.remove", "os.rename"}:
        raise AssertionError("External side effect: " + event)
    if event == "open":
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        ):
            raise AssertionError("File write")
sys.addaudithook(audit)
from examples.vlm_semantic_drift.vlm_adapter import MockVLMAdapter
from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.prompt_builder import build_hypothesis_prompts
event = DetectionEvent('synthetic', '2026-01-01T00:00:00Z', 'synthetic', 'person', 0.8, (0,0,8,8))
assert MockVLMAdapter().generate(build_hypothesis_prompts(event)).raw_response == 'UNKNOWN'
assert not {'PIL', 'torch', 'psycopg2', 'requests', 'vision_analysis', 'dbmanager'} & set(sys.modules)
'''
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(root)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
