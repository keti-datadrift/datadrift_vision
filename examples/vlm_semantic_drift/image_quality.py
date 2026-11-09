"""Demonstration quality heuristics, independent of semantic drift decisions."""

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class ImageQualityConfig:
    """Uncalibrated demonstration thresholds for 8-bit luminance."""

    dark_pixel_threshold: int = 32
    bright_pixel_threshold: int = 223
    exposure_ratio_threshold: float = 0.8
    min_dimension: int = 8
    min_luminance_stddev: float = 2.0
    blur_score_threshold: float = 0.03

    def __post_init__(self) -> None:
        for name in ("dark_pixel_threshold", "bright_pixel_threshold"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError(name + " must be an integer in [0, 255]")
        if self.dark_pixel_threshold >= self.bright_pixel_threshold:
            raise ValueError("dark threshold must be below bright threshold")
        if type(self.min_dimension) is not int or self.min_dimension < 2:
            raise ValueError("min_dimension must be an integer >= 2")
        for name, upper in (("exposure_ratio_threshold", 1),
                            ("min_luminance_stddev", 127.5),
                            ("blur_score_threshold", 1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value <= upper or not math.isfinite(value):
                raise ValueError(name + " must be finite and in [0, " + str(upper) + "]")
        if self.exposure_ratio_threshold == 0:
            raise ValueError("exposure_ratio_threshold must be > 0")


@dataclass(frozen=True)
class ImageQualityResult:
    """Metrics and candidate issues; assessable refers to blur eligibility."""

    mean_brightness: float
    dark_pixel_ratio: float
    bright_pixel_ratio: float
    luminance_stddev: float
    edge_score: Optional[float]
    issues: Tuple[str, ...]
    assessable: bool


def analyze_image_quality(
    image: "Image.Image", config: Optional[ImageQualityConfig] = None,
) -> ImageQualityResult:
    """Measure a loaded L/RGB image without altering it or opening paths.

    Low edge energy is only a blur candidate when size and contrast suffice.
    Exposure metrics remain available even when blur cannot be assessed.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a loaded Pillow Image.Image")
    if config is None:
        config = ImageQualityConfig()
    if not isinstance(config, ImageQualityConfig):
        raise TypeError("config must be ImageQualityConfig or None")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image must have positive width and height")
    if image.mode not in ("L", "RGB"):
        raise ValueError("image mode must be L or RGB; convert explicitly before analysis")
    try:
        with image.convert("L") as gray:
            histogram = gray.histogram()
            pixels = gray.tobytes()
    except ValueError as exc:
        raise ValueError("image must be open and usable") from exc
    count = width * height
    mean = sum(level * number for level, number in enumerate(histogram)) / count
    variance = sum((level - mean) ** 2 * number for level, number in enumerate(histogram)) / count
    stddev = math.sqrt(variance)
    dark = sum(histogram[:config.dark_pixel_threshold + 1]) / count
    bright = sum(histogram[config.bright_pixel_threshold:]) / count
    pairs = (width - 1) * height + width * (height - 1)
    total = 0
    for y in range(height):
        for x in range(width):
            index = y * width + x
            if x + 1 < width:
                total += abs(pixels[index] - pixels[index + 1])
            if y + 1 < height:
                total += abs(pixels[index] - pixels[index + width])
    score = total / (255 * pairs) if pairs else None
    issues = []
    if dark >= config.exposure_ratio_threshold:
        issues.append("underexposure_candidate")
    if bright >= config.exposure_ratio_threshold:
        issues.append("overexposure_candidate")
    small = min(width, height) < config.min_dimension
    low_texture = stddev <= config.min_luminance_stddev
    if small:
        issues.append("insufficient_size")
    if low_texture:
        issues.append("low_texture")
    assessable = not small and not low_texture
    if assessable and score is not None and score < config.blur_score_threshold:
        issues.append("blur_candidate")
    return ImageQualityResult(mean, dark, bright, stddev, score, tuple(issues), assessable)
