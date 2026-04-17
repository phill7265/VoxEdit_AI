"""
src/utils/test_transcriber_integration.py

Integration tests for the Transcriber skill using the REAL Whisper model.

Requirements
------------
    pip install openai-whisper
    # model 'base' (~139 MB) auto-downloaded on first run

Audio fixture
-------------
tests/sample_audio.wav — 3 s synthetic WAV (sine waves with a 0.5 s silence gap).
Whisper will produce short output (music notes, noise tokens) — that is expected
for sine waves.  What these tests verify is that the full pipeline runs correctly
end-to-end with no mock: model loads, transcribes, writes transcript.json, and
commits a Handover Record to harness memory.

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_transcriber_integration.py -v -s

Note: first run downloads the base model (~139 MB) and may take ~30 s.
      Subsequent runs use the cached model and finish in a few seconds.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.transcriber.logic import (
    RealWhisperBackend,
    TranscribeResult,
    transcribe,
)
from skills.transcriber.main import run as transcriber_run
from harness.memory.manager import MemoryManager
import harness.memory.manager as mgr_mod

SAMPLE_WAV = PROJECT_ROOT / "tests" / "sample_audio.wav"


@unittest.skipUnless(SAMPLE_WAV.exists(), f"Sample WAV not found: {SAMPLE_WAV}")
class TestRealWhisperBackend(unittest.TestCase):
    """RealWhisperBackend — model load and transcribe contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = RealWhisperBackend()
        cls.backend.load("base")

    def test_load_does_not_raise(self) -> None:
        # Already loaded in setUpClass — calling again must be a no-op
        self.backend.load("base")

    def test_transcribe_returns_dict(self) -> None:
        result = self.backend.transcribe(str(SAMPLE_WAV), word_timestamps=True)
        self.assertIsInstance(result, dict)

    def test_result_has_text_key(self) -> None:
        result = self.backend.transcribe(str(SAMPLE_WAV))
        self.assertIn("text", result)

    def test_result_has_segments_key(self) -> None:
        result = self.backend.transcribe(str(SAMPLE_WAV))
        self.assertIn("segments", result)
        self.assertIsInstance(result["segments"], list)

    def test_result_has_language_key(self) -> None:
        result = self.backend.transcribe(str(SAMPLE_WAV))
        self.assertIn("language", result)
        self.assertIsInstance(result["language"], str)


@unittest.skipUnless(SAMPLE_WAV.exists(), f"Sample WAV not found: {SAMPLE_WAV}")
class TestTranscribeWithRealModel(unittest.TestCase):
    """transcribe() with RealWhisperBackend — result structure validation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = RealWhisperBackend()
        cls.result: TranscribeResult = transcribe(
            str(SAMPLE_WAV),
            model_name="base",
            backend=cls.backend,
        )

    # ── Result type ───────────────────────────────────────────────────────

    def test_returns_transcribe_result(self) -> None:
        self.assertIsInstance(self.result, TranscribeResult)

    def test_model_name_recorded(self) -> None:
        self.assertEqual(self.result.model_name, "base")

    def test_language_detected(self) -> None:
        self.assertIsInstance(self.result.language, str)
        self.assertGreater(len(self.result.language), 0)

    # ── Duration ──────────────────────────────────────────────────────────

    def test_duration_approximately_correct(self) -> None:
        """Sample WAV is 3.0 s; Whisper's end timestamp should be within 1 s."""
        self.assertAlmostEqual(self.result.duration_s, 3.0, delta=1.0)

    # ── Words ─────────────────────────────────────────────────────────────

    def test_words_is_list(self) -> None:
        self.assertIsInstance(self.result.words, list)

    def test_word_entries_have_required_fields(self) -> None:
        for w in self.result.words:
            self.assertIsInstance(w.word, str)
            self.assertIsInstance(w.start, float)
            self.assertIsInstance(w.end, float)
            self.assertIsInstance(w.probability, float)

    def test_word_timestamps_are_ordered(self) -> None:
        for i in range(len(self.result.words) - 1):
            self.assertLessEqual(
                self.result.words[i].end,
                self.result.words[i + 1].start + 0.1,  # 100 ms tolerance
                f"Word order violated at index {i}",
            )

    def test_word_probabilities_in_range(self) -> None:
        for w in self.result.words:
            self.assertGreaterEqual(w.probability, 0.0)
            self.assertLessEqual(w.probability, 1.0)

    def test_words_as_dicts_has_start_ms(self) -> None:
        for d in self.result.words_as_dicts():
            self.assertIn("start_ms", d)
            self.assertIn("end_ms", d)
            self.assertIn("confidence", d)
            self.assertIsInstance(d["start_ms"], int)

    # ── VAD segments ──────────────────────────────────────────────────────

    def test_vad_segments_is_list(self) -> None:
        self.assertIsInstance(self.result.vad_segments, list)

    def test_vad_segments_have_required_fields(self) -> None:
        for s in self.result.vad_segments:
            self.assertIsInstance(s.start, float)
            self.assertIsInstance(s.end, float)
            self.assertIsInstance(s.is_voice, bool)
            self.assertIsInstance(s.avg_probability, float)

    def test_vad_as_dicts_has_confidence_alias(self) -> None:
        for d in self.result.vad_as_dicts():
            self.assertIn("confidence", d)
            self.assertIn("start_s", d)
            self.assertIn("end_s", d)

    def test_vad_segments_cover_entire_duration(self) -> None:
        """All words must fall inside some VAD segment."""
        for word in self.result.words:
            mid = (word.start + word.end) / 2.0
            covered = any(s.start <= mid <= s.end for s in self.result.vad_segments)
            self.assertTrue(covered,
                            f"Word '{word.word}' at {mid:.2f}s not covered by any VAD segment")

    # ── avg_confidence ────────────────────────────────────────────────────

    def test_avg_confidence_in_range(self) -> None:
        if self.result.words:
            self.assertGreaterEqual(self.result.avg_confidence, 0.0)
            self.assertLessEqual(self.result.avg_confidence, 1.0)


@unittest.skipUnless(SAMPLE_WAV.exists(), f"Sample WAV not found: {SAMPLE_WAV}")
class TestTranscriberMainRealModel(unittest.TestCase):
    """main.run() end-to-end with RealWhisperBackend — harness integration."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.staging_dir = self.tmp_path / "staging" / "job_real_001"
        self.jobs_root = self.tmp_path / "jobs"
        self._patcher = patch.object(mgr_mod, "JOBS_ROOT", self.jobs_root)
        self._patcher.start()
        # Share model across tests in this class to avoid re-loading
        self._backend = RealWhisperBackend()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    def _run(self, job_id: str = "job_real_001"):
        return transcriber_run(
            job_id=job_id,
            source_file=str(SAMPLE_WAV),
            model_name="base",
            backend=self._backend,
            staging_dir=self.staging_dir,
        )

    # ── Status ────────────────────────────────────────────────────────────

    def test_record_status_is_success(self) -> None:
        record = self._run()
        self.assertEqual(record.status, "success",
                         f"Expected success, got failed: {record.error}")

    def test_record_skill_name(self) -> None:
        record = self._run()
        self.assertEqual(record.skill, "transcriber")

    # ── Transcript file ───────────────────────────────────────────────────

    def test_transcript_json_exists(self) -> None:
        self._run()
        self.assertTrue((self.staging_dir / "transcript.json").exists())

    def test_transcript_json_valid(self) -> None:
        self._run()
        raw = json.loads((self.staging_dir / "transcript.json").read_text())
        for key in ("full_text", "words", "vad_segments", "metadata"):
            self.assertIn(key, raw)

    def test_transcript_words_have_timestamps(self) -> None:
        self._run()
        raw = json.loads((self.staging_dir / "transcript.json").read_text())
        for w in raw["words"]:
            self.assertIn("start_ms", w)
            self.assertIn("end_ms", w)
            self.assertIsInstance(w["start_ms"], int)

    def test_transcript_metadata_model(self) -> None:
        self._run()
        raw = json.loads((self.staging_dir / "transcript.json").read_text())
        self.assertEqual(raw["metadata"]["model"], "base")

    def test_transcript_metadata_duration_approx(self) -> None:
        self._run()
        raw = json.loads((self.staging_dir / "transcript.json").read_text())
        self.assertAlmostEqual(raw["metadata"]["duration_s"], 3.0, delta=1.0)

    def test_transcript_vad_segments_present(self) -> None:
        self._run()
        raw = json.loads((self.staging_dir / "transcript.json").read_text())
        self.assertIsInstance(raw["vad_segments"], list)

    # ── Handover record ───────────────────────────────────────────────────

    def test_handover_record_written_to_memory(self) -> None:
        self._run()
        records = list((self.jobs_root / "job_real_001").glob("*.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].name, "001_transcriber.json")

    def test_handover_record_output_path(self) -> None:
        record = self._run()
        self.assertTrue(record.output_path.endswith("transcript.json"))
        self.assertTrue(Path(record.output_path).exists())

    def test_handover_cursor_end_non_zero(self) -> None:
        record = self._run()
        self.assertNotEqual(record.cursor_end, "00:00:00.000")

    def test_handover_payload_output_block(self) -> None:
        record = self._run()
        output = record.payload.get("output", {})
        self.assertIn("full_text", output)
        self.assertIn("words", output)
        self.assertIn("vad_segments", output)

    def test_handover_payload_metadata_block(self) -> None:
        record = self._run()
        meta = record.payload.get("metadata", {})
        self.assertIn("model", meta)
        self.assertIn("duration", meta)
        self.assertIn("word_count", meta)
        self.assertIn("avg_confidence", meta)

    # ── MemoryManager resume ──────────────────────────────────────────────

    def test_resume_point_advances_to_cutter(self) -> None:
        self._run()
        mgr = MemoryManager("job_real_001")
        resume = mgr.find_resume_point()
        self.assertEqual(resume.next_skill, "cutter")
        self.assertIn("transcriber", resume.completed)

    def test_resume_prior_output_is_transcript_path(self) -> None:
        self._run()
        mgr = MemoryManager("job_real_001")
        resume = mgr.find_resume_point()
        output = resume.prior_output("transcriber")
        self.assertIsNotNone(output)
        self.assertTrue(str(output).endswith("transcript.json"))
        self.assertTrue(Path(output).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
