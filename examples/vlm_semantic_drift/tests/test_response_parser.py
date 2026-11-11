"""Strict token polarity, provenance and offline MockVLM integration."""

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
import subprocess
import sys
import unittest

from examples.vlm_semantic_drift.event_schema import DetectionEvent
from examples.vlm_semantic_drift.prompt_builder import build_hypothesis_prompts
from examples.vlm_semantic_drift.vlm_adapter import MockVLMAdapter, VLMRawResponse
from examples.vlm_semantic_drift.response_parser import parse_vlm_response


class ResponseParserTests(unittest.TestCase):
    def assert_token(self, raw: str, token: str, match, h1) -> None:
        result = parse_vlm_response(VLMRawResponse("mock", True, raw))
        self.assertEqual(result.status, token)
        self.assertEqual(result.normalized_response, token)
        self.assertIs(result.label_match, match)
        self.assertIs(result.supports_h1, h1)
        self.assertIs(result.valid, True)
        self.assertEqual(result.raw_response, raw)

    def assert_invalid(self, raw: str) -> None:
        result = parse_vlm_response(VLMRawResponse("mock", True, raw))
        self.assertEqual(result.status, "INVALID")
        self.assertIsNone(result.normalized_response)
        self.assertIsNone(result.label_match)
        self.assertIsNone(result.supports_h1)
        self.assertIs(result.valid, False)
        self.assertEqual(result.raw_response, raw)

    def test_yes(self) -> None:
        self.assert_token("YES", "YES", True, False)

    def test_no(self) -> None:
        self.assert_token("NO", "NO", False, True)

    def test_unknown(self) -> None:
        self.assert_token("UNKNOWN", "UNKNOWN", None, None)

    def test_outer_whitespace_and_newlines(self) -> None:
        for token, match, h1 in (("YES", True, False), ("NO", False, True), ("UNKNOWN", None, None)):
            for padding in (" ", "\t", "\r\n", "\u2003"):
                with self.subTest(token=token, padding=padding):
                    self.assert_token(padding + token + padding, token, match, h1)

    def test_lowercase_and_mixed_case(self) -> None:
        for raw in ("yes", "no", "unknown", "Yes", "No", "Unknown"):
            with self.subTest(raw=raw):
                self.assert_invalid(raw)

    def test_multiple_tokens_and_inner_whitespace(self) -> None:
        for raw in ("YES NO", "NO YES", "YES\nNO", "UNKNOWN YES", "Y ES", "N\tO"):
            with self.subTest(raw=raw):
                self.assert_invalid(raw)

    def test_explanations_and_punctuation(self) -> None:
        for raw in ("YES because...", "NO: no person", "Answer: YES", "YES.", "UNKNOWN!", "NOT YES"):
            with self.subTest(raw=raw):
                self.assert_invalid(raw)

    def test_json_markdown_and_unicode_are_not_decoded(self) -> None:
        for raw in ('"YES"', '{"answer":"YES"}', "`NO`", "```\nYES\n```",
                    "ＹＥＳ", "YES\x00", "\ufeffYES", "\u200bYES"):
            with self.subTest(raw=raw):
                self.assert_invalid(raw)

    def test_empty_and_unrecognized_strings(self) -> None:
        for raw in ("", " \n\t", "MAYBE", "INVALID", "NONE"):
            with self.subTest(raw=raw):
                self.assert_invalid(raw)

    def test_unknown_and_invalid_are_distinct(self) -> None:
        unknown = parse_vlm_response(VLMRawResponse("mock", True, "UNKNOWN"))
        invalid = parse_vlm_response(VLMRawResponse("mock", True, "unknown"))
        self.assertNotEqual(unknown.status, invalid.status)
        self.assertTrue(unknown.valid)
        self.assertFalse(invalid.valid)
        self.assertIsNone(unknown.supports_h1)
        self.assertIsNone(invalid.supports_h1)

    def test_wrong_input_types(self) -> None:
        for value in (None, "YES", b"YES", {}, 1):
            with self.subTest(value=value), self.assertRaises(TypeError):
                parse_vlm_response(value)

    def test_corrupted_raw_fields(self) -> None:
        for field, value in (("provider", ""), ("is_mock", 1), ("raw_response", None)):
            raw = VLMRawResponse("mock", True, "YES")
            object.__setattr__(raw, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_vlm_response(raw)

    def test_metadata_and_exact_raw_preservation(self) -> None:
        for provider, is_mock in (("mock", True), (" synthetic-provider ", False)):
            for raw in (" YES\r\n", "invalid text"):
                with self.subTest(provider=provider, raw=raw):
                    result = parse_vlm_response(VLMRawResponse(provider, is_mock, raw))
                    self.assertEqual(result.provider, provider)
                    self.assertIs(result.is_mock, is_mock)
                    self.assertEqual(result.raw_response, raw)

    def test_immutable_repeatable_and_input_preserved(self) -> None:
        raw = VLMRawResponse("mock", True, " NO ")
        before = asdict(raw)
        result = parse_vlm_response(raw)
        self.assertEqual(result, parse_vlm_response(raw))
        self.assertEqual(asdict(raw), before)
        with self.assertRaises(FrozenInstanceError):
            result.status = "YES"

    def test_mock_pipeline(self) -> None:
        event = DetectionEvent("synthetic-parser", "2026-01-01T00:00:00Z",
                               "synthetic-camera", "person", 0.8, (0, 0, 8, 8))
        prompt = build_hypothesis_prompts(event)
        for token in ("YES", "NO", "UNKNOWN"):
            result = parse_vlm_response(MockVLMAdapter(token).generate(prompt))
            self.assertEqual(result.status, token)
            self.assertEqual(result.provider, "mock")
            self.assertTrue(result.is_mock)
        self.assertEqual(parse_vlm_response(MockVLMAdapter().generate(prompt)).status, "UNKNOWN")

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
from examples.vlm_semantic_drift.response_parser import parse_vlm_response
from examples.vlm_semantic_drift.vlm_adapter import VLMRawResponse
assert parse_vlm_response(VLMRawResponse('mock', True, 'YES')).label_match is True
assert not {'PIL', 'torch', 'psycopg2', 'requests', 'vision_analysis', 'dbmanager'} & set(sys.modules)
'''
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(root)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
