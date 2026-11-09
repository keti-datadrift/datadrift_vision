"""Synthetic quality examples; no real images or performance claims."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from PIL import Image, ImageFilter

from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.roi_extractor import extract_roi
from examples.vlm_semantic_drift.image_quality import (
    ImageQualityConfig, analyze_image_quality,
)


def patterned(size: int = 16) -> Image.Image:
    image = Image.new("L", (size, size))
    image.putdata([80 if (x + y) % 2 else 176 for y in range(size) for x in range(size)])
    return image


class ImageQualityTests(unittest.TestCase):
    def test_normal_brightness_and_edges(self) -> None:
        with patterned() as image:
            result = analyze_image_quality(image)
        self.assertEqual(result.mean_brightness, 128)
        self.assertEqual((result.dark_pixel_ratio, result.bright_pixel_ratio), (0, 0))
        self.assertAlmostEqual(result.edge_score, 96 / 255)
        self.assertTrue(result.assessable)
        self.assertEqual(result.issues, ())

    def test_dark_and_bright(self) -> None:
        for level, issue in ((0, "underexposure_candidate"), (255, "overexposure_candidate")):
            with self.subTest(level=level), Image.new("L", (16, 16), level) as image:
                result = analyze_image_quality(image)
                self.assertEqual(result.mean_brightness, level)
                self.assertIn(issue, result.issues)
                self.assertIn("low_texture", result.issues)
                self.assertNotIn("blur_candidate", result.issues)
                self.assertFalse(result.assessable)

    def test_uniform_is_not_proven_blur(self) -> None:
        with Image.new("RGB", (16, 16), (128, 128, 128)) as image:
            result = analyze_image_quality(image)
        self.assertEqual(result.edge_score, 0)
        self.assertEqual(result.issues, ("low_texture",))
        self.assertFalse(result.assessable)

    def test_low_edge_gradient_is_only_candidate(self) -> None:
        with Image.new("L", (16, 16)) as image:
            image.putdata([80 + 4 * x for y in range(16) for x in range(16)])
            result = analyze_image_quality(image)
        self.assertAlmostEqual(result.edge_score, 2 / 255)
        self.assertEqual(result.issues, ("blur_candidate",))
        self.assertTrue(result.assessable)

    def test_synthetic_blurring_reduces_score(self) -> None:
        with patterned(32) as image, image.filter(ImageFilter.GaussianBlur(1)) as blurred:
            self.assertLess(analyze_image_quality(blurred).edge_score,
                            analyze_image_quality(image).edge_score)

    def test_small_image_and_single_pixel(self) -> None:
        with patterned(4) as image:
            result = analyze_image_quality(image)
            self.assertEqual(result.issues, ("insufficient_size",))
            self.assertFalse(result.assessable)
        with Image.new("L", (1, 1), 128) as image:
            result = analyze_image_quality(image)
            self.assertIsNone(result.edge_score)
            self.assertEqual(result.mean_brightness, 128)
            self.assertFalse(result.assessable)

    def test_invalid_images(self) -> None:
        for image in (None, "image.png", b"image", []):
            with self.subTest(image=image), self.assertRaises(TypeError):
                analyze_image_quality(image)
        with Image.new("L", (0, 2)) as image, self.assertRaises(ValueError):
            analyze_image_quality(image)
        image = Image.new("L", (16, 16))
        image.close()
        with self.assertRaises(ValueError):
            analyze_image_quality(image)
        with Image.new("RGBA", (16, 16)) as image, self.assertRaises(ValueError):
            analyze_image_quality(image)

    def test_exposure_threshold_inclusive(self) -> None:
        config = ImageQualityConfig(exposure_ratio_threshold=0.5)
        with Image.new("L", (8, 8)) as image:
            image.putdata([32] * 32 + [223] * 32)
            result = analyze_image_quality(image, config)
            self.assertEqual((result.dark_pixel_ratio, result.bright_pixel_ratio), (0.5, 0.5))
            self.assertIn("underexposure_candidate", result.issues)
            self.assertIn("overexposure_candidate", result.issues)
            result = analyze_image_quality(image, replace(config, exposure_ratio_threshold=0.51))
            self.assertNotIn("underexposure_candidate", result.issues)
            self.assertNotIn("overexposure_candidate", result.issues)

    def test_size_and_texture_thresholds(self) -> None:
        with patterned(8) as image:
            self.assertTrue(analyze_image_quality(image).assessable)
            self.assertFalse(analyze_image_quality(image, ImageQualityConfig(min_dimension=9)).assessable)
            self.assertFalse(analyze_image_quality(image, ImageQualityConfig(min_luminance_stddev=48)).assessable)

    def test_blur_threshold_strict(self) -> None:
        with patterned() as image:
            score = analyze_image_quality(image).edge_score
            self.assertNotIn("blur_candidate", analyze_image_quality(
                image, ImageQualityConfig(blur_score_threshold=score)).issues)
            self.assertIn("blur_candidate", analyze_image_quality(
                image, ImageQualityConfig(blur_score_threshold=score + 0.01)).issues)

    def test_invalid_config(self) -> None:
        bad = {"dark_pixel_threshold": [-1, 256, True, 1.5, "32"],
               "bright_pixel_threshold": [0, 256, False],
               "min_dimension": [1, True, 8.0],
               "exposure_ratio_threshold": [0, -1, 1.1, float("nan"), float("inf"), True, "0.8"],
               "blur_score_threshold": [-1, 1.1, float("nan"), None],
               "min_luminance_stddev": [-1, 128, float("inf")]}
        for key, values in bad.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    ImageQualityConfig(**{key: value})
        with self.assertRaises(ValueError):
            ImageQualityConfig(dark_pixel_threshold=223)
        with patterned() as image, self.assertRaises(TypeError):
            analyze_image_quality(image, {})

    def test_immutable_results_and_config(self) -> None:
        config = ImageQualityConfig()
        with patterned() as image:
            result = analyze_image_quality(image, config)
        with self.assertRaises(FrozenInstanceError):
            result.assessable = False
        with self.assertRaises(FrozenInstanceError):
            config.min_dimension = 2
        self.assertIsInstance(result.issues, tuple)

    def test_source_unchanged_and_repeatable(self) -> None:
        with patterned() as image:
            before = image.tobytes()
            image.info["synthetic"] = True
            with patch.object(Image, "open", side_effect=AssertionError("No file reads")):
                first = analyze_image_quality(image)
                second = analyze_image_quality(image)
            self.assertEqual(first, second)
            self.assertEqual(image.tobytes(), before)
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.info, {"synthetic": True})

    def test_roi_integration(self) -> None:
        event = DetectionEvent("synthetic-quality", "2026-01-01T00:00:00Z",
                               "synthetic-camera", "object", 0.8, (0, 0, 8, 8))
        with patterned() as image:
            roi = extract_roi(image, event)
            try:
                result = analyze_image_quality(roi.image)
                self.assertTrue(result.assessable)
                self.assertEqual(result.mean_brightness, 128)
            finally:
                roi.image.close()

    def test_import_without_pillow_or_services(self) -> None:
        root = Path(__file__).resolve().parents[3]
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import examples.vlm_semantic_drift.image_quality; "
            "assert not {'PIL', 'numpy', 'cv2', 'vision_analysis', 'dbmanager'} & set(sys.modules)"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(root)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
