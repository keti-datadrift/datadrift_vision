"""Synthetic event contracts and malformed input boundaries."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from examples.vlm_semantic_drift.event_schema import DetectionEvent


class EventSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {
            "schema_version": 1, "event_id": "synthetic-001",
            "timestamp": "2026-01-01T00:00:00Z", "camera_id": "synthetic-camera",
            "class_label": "person", "confidence": 0.8, "bbox": [1, 2, 10, 20],
        }

    def test_json_roundtrip(self) -> None:
        self.data["class_label"] = "합성 객체"
        event = DetectionEvent.from_dict(self.data)
        self.assertEqual(DetectionEvent.from_json(event.to_json()), event)
        self.assertEqual(json.loads(event.to_json()), event.to_dict())
        self.assertIsNone(event.image_path)
        self.assertEqual(event.to_json(), event.to_json())

    def test_required_and_unknown_fields(self) -> None:
        for key in self.data:
            with self.subTest(missing=key), self.assertRaises(ValueError):
                DetectionEvent.from_dict({k: v for k, v in self.data.items() if k != key})
        with self.assertRaises(ValueError):
            DetectionEvent.from_dict(dict(self.data, unexpected=True))
        for value in (None, [], "event"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DetectionEvent.from_dict(value)

    def test_invalid_text_and_version(self) -> None:
        for key in ("event_id", "camera_id", "class_label"):
            for value in (None, 1, "", "  ", " padded", "line\nbreak"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    DetectionEvent.from_dict(dict(self.data, **{key: value}))
        for value in (True, 1.0, "1", 2):
            with self.subTest(version=value), self.assertRaises(ValueError):
                DetectionEvent.from_dict(dict(self.data, schema_version=value))

    def test_utc_timestamp(self) -> None:
        event = DetectionEvent.from_dict(dict(self.data, timestamp="2026-01-01T00:00:00.123+00:00"))
        self.assertEqual(event.timestamp, "2026-01-01T00:00:00.123000Z")
        for value in (None, 1, "2026-01-01", "2026-01-01T00:00:00",
                      "2026-01-01T09:00:00+09:00", "2026-02-30T00:00:00Z",
                      "2026-01-01T24:00:00Z", "2026-01-01T00:00:00.1234567Z"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DetectionEvent.from_dict(dict(self.data, timestamp=value))

    def test_confidence_boundaries(self) -> None:
        for value in (0, 1):
            self.assertEqual(DetectionEvent.from_dict(dict(self.data, confidence=value)).confidence, value)
        for value in (-0.01, 1.01, True, "0.5", None, float("nan"), float("inf"), 10**400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DetectionEvent.from_dict(dict(self.data, confidence=value))

    def test_bbox_valid_and_invalid(self) -> None:
        event = DetectionEvent.from_dict(dict(self.data, bbox=[-5.5, -1, 2000, 1000]))
        self.assertEqual(event.bbox, (-5.5, -1.0, 2000.0, 1000.0))
        for value in (None, [], [1, 2, 3], [1, 2, 3, 4, 5], "1234",
                      [1, 2, 1, 4], [3, 2, 1, 4], [1, 4, 3, 2],
                      [True, 0, 3, 4], [0, 0, float("inf"), 4], [0, 0, float("nan"), 4]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DetectionEvent.from_dict(dict(self.data, bbox=value))

    def test_image_reference(self) -> None:
        path = "synthetic/not-created.png"
        self.assertEqual(DetectionEvent.from_dict(dict(self.data, image_path=path)).image_path, path)
        for value in ("", 42, "/image.png", "../image.png", "a/../b", "a//b",
                      "./a", "C:/image.png", "a\\b", "https://example.invalid/a"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DetectionEvent.from_dict(dict(self.data, image_path=value))

    def test_immutable_and_no_aliasing(self) -> None:
        event = DetectionEvent.from_dict(self.data)
        self.data["bbox"][0] = 99
        exported = event.to_dict()
        exported["bbox"][0] = 88
        self.assertEqual(event.bbox[0], 1)
        with self.assertRaises(FrozenInstanceError):
            event.confidence = 2
        with self.assertRaises(ValueError):
            DetectionEvent(**dict(self.data, confidence=2))

    def test_malformed_json(self) -> None:
        for value in ("{", "null", "[]", '{"event_id":"a","event_id":"b"}', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DetectionEvent.from_json(value)
        raw = json.dumps(dict(self.data, confidence=float("nan")))
        with self.assertRaises(ValueError):
            DetectionEvent.from_json(raw)

    def test_synthetic_fixture(self) -> None:
        path = Path(__file__).resolve().parents[1] / "sample_data" / "events.json"
        with path.open(encoding="utf-8") as stream:
            events = [DetectionEvent.from_dict(item) for item in json.load(stream)]
        self.assertEqual(len(events), 2)
        self.assertEqual(len({event.event_id for event in events}), 2)
        for event in events:
            self.assertTrue(event.camera_id.startswith("synthetic-"))
            self.assertEqual(DetectionEvent.from_json(event.to_json()), event)


if __name__ == "__main__":
    unittest.main()
