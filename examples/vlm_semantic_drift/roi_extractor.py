"""Extract bounded ROIs from an in-memory image and a detection event."""

from dataclasses import dataclass
from math import ceil, floor
from typing import TYPE_CHECKING, Tuple

from .event_schema import DetectionEvent

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class ROIResult:
    """Independent crop and its integer, right/bottom-exclusive bounds."""

    image: "Image.Image"
    bbox: Tuple[int, int, int, int]

    @property
    def size(self) -> Tuple[int, int]:
        """Return the crop's (width, height)."""
        return self.image.size


def extract_roi(image: "Image.Image", event: DetectionEvent) -> ROIResult:
    """Floor left/top, ceil right/bottom, clamp, and copy the valid area.

    The caller supplies a loaded Pillow image in the bbox coordinate system.
    No image path is opened and no orientation or color conversion is applied.
    Raises TypeError for wrong input types and ValueError for empty or closed
    images, or a bbox with no positive-area intersection with the image.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a loaded Pillow Image.Image")
    if not isinstance(event, DetectionEvent):
        raise TypeError("event must be a DetectionEvent")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image must have positive width and height")

    x1, y1, x2, y2 = event.bbox
    # Check continuous overlap before rounding: touching an edge is not an ROI.
    if max(0, x1) >= min(width, x2) or max(0, y1) >= min(height, y2):
        raise ValueError("bbox has no positive-area intersection with image")
    bounds = (
        max(0, min(width, floor(x1))),
        max(0, min(height, floor(y1))),
        max(0, min(width, ceil(x2))),
        max(0, min(height, ceil(y2))),
    )
    # Pillow crop returns independent pixel storage and preserves image mode.
    try:
        cropped = image.crop(bounds)
    except ValueError as exc:
        raise ValueError("image must be open and usable for cropping") from exc
    return ROIResult(image=cropped, bbox=bounds)
