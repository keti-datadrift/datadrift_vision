"""Exercise ROI geometry on deterministic in-memory synthetic images."""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from PIL import Image

from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.roi_extractor import extract_roi


def event_for(bbox: tuple) -> DetectionEvent:
    """Build a synthetic detection without reading an image reference."""
    return DetectionEvent(
        event_id="synthetic-roi-01", timestamp="2026-01-01T00:00:00Z",
        camera_id="synthetic-camera", class_label="synthetic-object",
        confidence=0.8, bbox=bbox, image_path="synthetic/nonexistent.png",
    )


class ROIExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (10, 8))
        self.image.putdata([(x, y, x + y) for y in range(8) for x in range(10)])
        self.addCleanup(self.image.close)

    def test_normal_crop_pixels_and_size(self) -> None:
        result = extract_roi(self.image, event_for((2, 1, 7, 5)))
        self.assertEqual(result.bbox, (2, 1, 7, 5))
        self.assertEqual(result.size, (5, 4))
        self.assertEqual(result.image.getpixel((0, 0)), (2, 1, 3))
        self.assertEqual(result.image.getpixel((4, 3)), (6, 4, 10))

    def test_full_image(self) -> None:
        result = extract_roi(self.image, event_for((0, 0, 10, 8)))
        self.assertEqual(result.bbox, (0, 0, 10, 8))
        self.assertEqual(result.size, self.image.size)
        self.assertEqual(result.image.tobytes(), self.image.tobytes())
        self.assertIsNot(result.image, self.image)

    def test_right_bottom_clamping(self) -> None:
        result = extract_roi(self.image, event_for((7, 5, 20, 30)))
        self.assertEqual(result.bbox, (7, 5, 10, 8))
        self.assertEqual(result.size, (3, 3))

    def test_negative_and_all_sides_clamping(self) -> None:
        for bbox, expected, size in (
            ((-4, -3, 3, 2), (0, 0, 3, 2), (3, 2)),
            ((-100, -100, 100, 100), (0, 0, 10, 8), (10, 8)),
        ):
            with self.subTest(bbox=bbox):
                result = extract_roi(self.image, event_for(bbox))
                self.assertEqual(result.bbox, expected)
                self.assertEqual(result.size, size)

    def test_outside_and_edge_only_boxes(self) -> None:
        for bbox in ((-5, 1, -1, 3), (11, 1, 13, 3), (1, -5, 3, -1),
                     (1, 9, 3, 11), (-2, 1, 0, 3), (10, 1, 12, 3),
                     (1, -2, 3, 0), (1, 8, 3, 10), (-0.9, 1, -0.1, 3)):
            with self.subTest(bbox=bbox), self.assertRaisesRegex(ValueError, "intersection"):
                extract_roi(self.image, event_for(bbox))

    def test_fractional_rounding(self) -> None:
        result = extract_roi(self.image, event_for((1.9, 2.1, 6.2, 5.9)))
        self.assertEqual(result.bbox, (1, 2, 7, 6))
        self.assertEqual(result.size, (6, 4))
        result = extract_roi(self.image, event_for((9.8, 7.8, 10.2, 8.2)))
        self.assertEqual(result.bbox, (9, 7, 10, 8))
        self.assertEqual(result.size, (1, 1))

    def test_subpixel_box_covers_one_pixel(self) -> None:
        result = extract_roi(self.image, event_for((2.1, 3.1, 2.2, 3.2)))
        self.assertEqual(result.bbox, (2, 3, 3, 4))
        self.assertEqual(result.size, (1, 1))

    def test_invalid_image_and_event_types(self) -> None:
        for value in (None, "image.png", b"pixels", [[1, 2]]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                extract_roi(value, event_for((0, 0, 1, 1)))
        with self.assertRaises(TypeError):
            extract_roi(self.image, {"bbox": [0, 0, 1, 1]})

    def test_empty_and_closed_images(self) -> None:
        for size in ((0, 8), (10, 0), (0, 0)):
            with self.subTest(size=size), Image.new("RGB", size) as empty:
                with self.assertRaisesRegex(ValueError, "positive"):
                    extract_roi(empty, event_for((0, 0, 1, 1)))
        closed = Image.new("RGB", (10, 8))
        closed.close()
        with self.assertRaisesRegex(ValueError, "open and usable"):
            extract_roi(closed, event_for((0, 0, 1, 1)))

    def test_source_unchanged_and_independent(self) -> None:
        event = event_for((-1, 1, 7, 5))
        before = self.image.tobytes()
        event_before = event.to_json()
        result = extract_roi(self.image, event)
        result.image.putpixel((0, 0), (255, 255, 255))
        self.assertEqual(self.image.tobytes(), before)
        self.assertEqual(event.to_json(), event_before)
        pixel = result.image.getpixel((1, 0))
        self.image.putpixel((1, 1), (255, 0, 0))
        self.assertEqual(result.image.getpixel((1, 0)), pixel)

    def test_modes_preserved(self) -> None:
        for mode, color in (("L", 42), ("RGBA", (1, 2, 3, 4))):
            with self.subTest(mode=mode), Image.new(mode, (3, 3), color) as source:
                result = extract_roi(source, event_for((1, 1, 3, 3)))
                self.assertEqual(result.image.mode, mode)
                self.assertEqual(result.image.getpixel((0, 0)), color)

    def test_json_event_and_no_path_access(self) -> None:
        event = DetectionEvent.from_json(event_for((1, 1, 3, 4)).to_json())
        with patch.object(Image, "open", side_effect=AssertionError("No path access")):
            result = extract_roi(self.image, event)
        self.assertEqual(result.size, (2, 3))

    def test_module_import_without_pillow(self) -> None:
        root = Path(__file__).resolve().parents[3]
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import examples.vlm_semantic_drift.roi_extractor; "
            "assert not any(n == 'PIL' or n.startswith('PIL.') for n in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", script, str(root)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
