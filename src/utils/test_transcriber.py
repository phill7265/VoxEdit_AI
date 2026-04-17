"""
src/utils/test_transcriber.py

Unit tests for skills/transcriber/logic.py and skills/transcriber/main.py.

Strategy
--------
openai-whisper requires PyTorch and a model download (~150 MB for 'base').
Tests use MockWhisperBackend — a deterministic stub that returns a fixed
Whisper-format dict — so the full pipeline can be verified without any
ML dependency.

The mock fixture mimics a real 8-second Korean interview clip with:
  · 5 words, all high-confidence
  · 1 low-confidence word (simulates noise) to test VAD thresholding
  · A 1.2-second silence gap between word 3 and word 4 (for silence detection)

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_transcriber.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.transcriber.logic import (
    DEFAULT_MODEL,
    VAD_VOICE_PROBABILITY_THRESHOLD,
    TranscribeResult,
    VadSegment,
    WhisperBackend,
    WordEntry,
    _derive_vad_segments,
    _parse_whisper_output,
    transcribe,
)
from skills.transcriber.main import run as transcriber_run
from harness.memory.manager import MemoryManager, SKILL_ORDER
import harness.memory.manager as mgr_mod


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixture
# ═══════════════════════════════════════════════════════════════════════════

# Whisper-format result dict representing an 8-second audio clip.
# Segment 1 : words 0–2  (high confidence, tight timing)
# Gap        : 1.2 s silence between word 2 (end=3.2) and word 3 (start=4.4)
# Segment 2  : words 3–4  (word 3 high-conf, word 4 low-conf → noise)
MOCK_WHISPER_RESULT: dict[str, Any] = {
    "text": "안녕하세요 저는 개발자 입니다 좋아요",
    "language": "ko",
    "segments": [
        {
            "start": 0.0,
            "end": 3.2,
            "text": "안녕하세요 저는 개발자",
            "words": [
                {"word": "안녕하세요", "start": 0.0,  "end": 1.0,  "probability": 0.97},
                {"word": "저는",      "start": 1.1,  "end": 1.8,  "probability": 0.95},
                {"word": "개발자",    "start": 2.0,  "end": 3.2,  "probability": 0.92},
            ],
        },
        {
            "start": 4.4,   # 1.2 s gap after previous segment
            "end": 8.0,
            "text": "입니다 좋아요",
            "words": [
                {"word": "입니다",  "start": 4.4, "end": 5.5, "probability": 0.91},
                {"word": "좋아요",  "start": 6.0, "end": 8.0, "probability": 0.30},  # noise
            ],
        },
    ],
}


class MockWhisperBackend:
    """Deterministic Whisper stub — no torch, no model download."""

    def __init__(self, result: Optional[dict] = None) -> None:
        self._result = result or MOCK_WHISPER_RESULT
        self.load_called_with: list[str] = []
        self.transcribe_called_with: list[tuple] = []

    def load(self, model_name: str) -> None:
        self.load_called_with.append(model_name)

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = True,
    ) -> dict[str, Any]:
        self.transcribe_called_with.append((audio_path, language))
        return self._result


class ErrorWhisperBackend:
    """Backend that always raises — simulates model crash."""

    def load(self, model_name: str) -> None:
        pass

    def transcribe(self, audio_path: str, **_: Any) -> dict[str, Any]:
        raise RuntimeError("GPU out of memory")


# ═══════════════════════════════════════════════════════════════════════════
# logic.py — _parse_whisper_output
# ═══════════════════════════════════════════════════════════════════════════

class TestParseWhisperOutput(unittest.TestCase):

    def setUp(self) -> None:
        self.words, self.duration = _parse_whisper_output(MOCK_WHISPER_RESULT)

    def test_word_count(self) -> None:
        self.assertEqual(len(self.words), 5)

    def test_word_text(self) -> None:
        texts = [w.word for w in self.words]
        self.assertIn("안녕하세요", texts)
        self.assertIn("좋아요", texts)

    def test_word_start_end_order(self) -> None:
        for w in self.words:
            self.assertLessEqual(w.start, w.end,
                                 f"Word '{w.word}': start > end")

    def test_probability_range(self) -> None:
        for w in self.words:
            self.assertGreaterEqual(w.probability, 0.0)
            self.assertLessEqual(w.probability, 1.0)

    def test_duration_equals_last_segment_end(self) -> None:
        self.assertAlmostEqual(self.duration, 8.0)

    def test_word_dict_has_required_fields(self) -> None:
        """to_dict() must expose fields that context_manager._slice_transcript expects."""
        d = self.words[0].to_dict()
        for key in ("word", "start", "end", "probability", "start_ms", "end_ms", "confidence"):
            self.assertIn(key, d, f"Missing field '{key}' in WordEntry.to_dict()")

    def test_start_ms_is_milliseconds(self) -> None:
        # start=0.0 s → start_ms=0, start=1.1 s → start_ms=1100
        second_word = self.words[1]
        self.assertEqual(second_word.to_dict()["start_ms"], 1100)

    def test_log_probability_converted_to_linear(self) -> None:
        """If Whisper emits a negative log-prob, parser must convert to [0,1]."""
        import math
        raw = {
            "text": "test",
            "language": "en",
            "segments": [{
                "start": 0.0, "end": 1.0, "text": "test",
                "words": [{"word": "test", "start": 0.0, "end": 1.0,
                           "probability": math.log(0.75)}],   # negative log-prob
            }],
        }
        words, _ = _parse_whisper_output(raw)
        self.assertAlmostEqual(words[0].probability, 0.75, places=4)

    def test_segment_without_word_timestamps_is_synthesised(self) -> None:
        """Segments with no 'words' key must be split token-by-token."""
        raw = {
            "text": "hello world",
            "language": "en",
            "segments": [{
                "start": 0.0, "end": 2.0, "text": "hello world",
                # No "words" key
            }],
        }
        words, dur = _parse_whisper_output(raw)
        self.assertEqual(len(words), 2)
        self.assertAlmostEqual(dur, 2.0)
        self.assertEqual(words[0].probability, 0.5)   # neutral fallback

    def test_empty_result(self) -> None:
        words, dur = _parse_whisper_output({"segments": []})
        self.assertEqual(words, [])
        self.assertAlmostEqual(dur, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# logic.py — _derive_vad_segments
# ═══════════════════════════════════════════════════════════════════════════

class TestDeriveVadSegments(unittest.TestCase):

    def setUp(self) -> None:
        words, _ = _parse_whisper_output(MOCK_WHISPER_RESULT)
        self.words = words
        self.vad = _derive_vad_segments(words)

    # ── Structural ────────────────────────────────────────────────────────

    def test_returns_list(self) -> None:
        self.assertIsInstance(self.vad, list)

    def test_all_words_are_covered(self) -> None:
        """Every word's midpoint must fall in exactly one VAD segment."""
        for word in self.words:
            mid = (word.start + word.end) / 2
            covering = [
                s for s in self.vad
                if s.start <= mid <= s.end
            ]
            self.assertEqual(len(covering), 1,
                             f"Word '{word.word}' at {mid:.2f}s covered by {len(covering)} segments")

    def test_segments_non_overlapping(self) -> None:
        for i in range(len(self.vad) - 1):
            self.assertLessEqual(
                self.vad[i].end, self.vad[i + 1].start,
                "VAD segments must not overlap",
            )

    # ── Voice / silence classification ───────────────────────────────────

    def test_high_confidence_words_produce_voice_segment(self) -> None:
        """Words 0–3 all have probability ≥ 0.85 → at least one voice segment."""
        voice_segs = [s for s in self.vad if s.is_voice]
        self.assertGreater(len(voice_segs), 0)

    def test_low_confidence_word_produces_non_voice_segment(self) -> None:
        """Word 4 ('좋아요') has probability 0.30 → must appear in a non-voice segment."""
        low_word = self.words[-1]   # 좋아요, probability=0.30
        non_voice = [
            s for s in self.vad
            if not s.is_voice and s.start <= low_word.start <= s.end
        ]
        self.assertEqual(len(non_voice), 1,
                         "Low-confidence word must be in a non-voice VAD segment")

    def test_vad_dict_has_confidence_alias(self) -> None:
        """audio/master.py reads .confidence; verify alias is present."""
        d = self.vad[0].to_dict()
        self.assertIn("confidence", d)
        self.assertIn("start_s", d)
        self.assertIn("end_s", d)

    # ── Merge gap ─────────────────────────────────────────────────────────

    def test_small_gap_voice_segments_are_merged(self) -> None:
        """Two voice segments with a 0.2 s gap (< merge threshold 0.3 s) must merge."""
        words = [
            WordEntry("a", 0.0, 1.0, 0.95),
            WordEntry("b", 1.2, 2.0, 0.93),   # gap = 0.2 s
        ]
        vad = _derive_vad_segments(words)
        voice = [s for s in vad if s.is_voice]
        self.assertEqual(len(voice), 1, "Adjacent voice segments with small gap must merge")

    def test_large_gap_voice_segments_stay_separate(self) -> None:
        """Two voice segments with a 1.2 s gap (> merge threshold) must stay separate."""
        words = [
            WordEntry("a", 0.0, 1.0, 0.95),
            WordEntry("b", 2.5, 3.5, 0.93),   # gap = 1.5 s
        ]
        vad = _derive_vad_segments(words)
        voice = [s for s in vad if s.is_voice]
        self.assertEqual(len(voice), 2)

    def test_empty_word_list_returns_empty_vad(self) -> None:
        self.assertEqual(_derive_vad_segments([]), [])


# ═══════════════════════════════════════════════════════════════════════════
# logic.py — transcribe() (public entry point)
# ═══════════════════════════════════════════════════════════════════════════

class TestTranscribeFunction(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # Create a dummy audio file (content irrelevant — MockBackend ignores it)
        self.audio_path = Path(self._tmp.name) / "sample.mp3"
        self.audio_path.write_bytes(b"\x00" * 64)
        self.backend = MockWhisperBackend()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, **kwargs: Any) -> TranscribeResult:
        return transcribe(str(self.audio_path), backend=self.backend, **kwargs)

    # ── Happy path ────────────────────────────────────────────────────────

    def test_returns_transcribe_result(self) -> None:
        result = self._run()
        self.assertIsInstance(result, TranscribeResult)

    def test_full_text_populated(self) -> None:
        result = self._run()
        self.assertGreater(len(result.full_text), 0)

    def test_words_populated(self) -> None:
        result = self._run()
        self.assertEqual(result.word_count, 5)

    def test_vad_segments_populated(self) -> None:
        result = self._run()
        self.assertGreater(len(result.vad_segments), 0)

    def test_duration_from_last_segment(self) -> None:
        result = self._run()
        self.assertAlmostEqual(result.duration_s, 8.0)

    def test_language_passed_to_backend(self) -> None:
        self._run(language="ko")
        _, lang_used = self.backend.transcribe_called_with[0]
        self.assertEqual(lang_used, "ko")

    def test_model_name_passed_to_backend(self) -> None:
        self._run(model_name="small")
        self.assertIn("small", self.backend.load_called_with)

    def test_avg_confidence_in_range(self) -> None:
        result = self._run()
        self.assertGreaterEqual(result.avg_confidence, 0.0)
        self.assertLessEqual(result.avg_confidence, 1.0)

    def test_words_as_dicts_keys(self) -> None:
        result = self._run()
        for d in result.words_as_dicts():
            for key in ("word", "start", "end", "probability", "start_ms", "end_ms"):
                self.assertIn(key, d)

    def test_vad_as_dicts_keys(self) -> None:
        result = self._run()
        for d in result.vad_as_dicts():
            for key in ("start", "end", "is_voice", "confidence", "start_s", "end_s"):
                self.assertIn(key, d)

    # ── Error handling ────────────────────────────────────────────────────

    def test_missing_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            transcribe("/nonexistent/audio.wav", backend=self.backend)

    def test_backend_error_propagates(self) -> None:
        with self.assertRaises(RuntimeError):
            transcribe(str(self.audio_path), backend=ErrorWhisperBackend())


# ═══════════════════════════════════════════════════════════════════════════
# main.py — run() harness integration
# ═══════════════════════════════════════════════════════════════════════════

class TestTranscriberMain(unittest.TestCase):
    """Verify that main.run() correctly integrates logic ↔ MemoryManager."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

        # Dummy audio file
        self.audio_path = self.tmp_path / "interview.mp3"
        self.audio_path.write_bytes(b"\x00" * 64)

        # Redirect JOBS_ROOT so tests don't touch the real harness directory
        self.jobs_root = self.tmp_path / "jobs"
        self._patcher = patch.object(mgr_mod, "JOBS_ROOT", self.jobs_root)
        self._patcher.start()

        self.staging_dir = self.tmp_path / "staging" / "job_001"
        self.backend = MockWhisperBackend()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    def _run(self, job_id: str = "job_001", **kwargs: Any):
        return transcriber_run(
            job_id=job_id,
            source_file=str(self.audio_path),
            backend=self.backend,
            staging_dir=self.staging_dir,
            **kwargs,
        )

    # ── Transcript file written to staging ────────────────────────────────

    def test_transcript_json_created(self) -> None:
        self._run()
        self.assertTrue((self.staging_dir / "transcript.json").exists())

    def test_transcript_json_structure(self) -> None:
        self._run()
        data = json.loads((self.staging_dir / "transcript.json").read_text())
        for key in ("full_text", "words", "vad_segments", "metadata"):
            self.assertIn(key, data, f"transcript.json missing '{key}'")

    def test_transcript_words_have_timestamps(self) -> None:
        self._run()
        data = json.loads((self.staging_dir / "transcript.json").read_text())
        for word in data["words"]:
            self.assertIn("start_ms", word)
            self.assertIn("end_ms", word)

    def test_transcript_metadata_model_field(self) -> None:
        self._run(model_name="small")
        data = json.loads((self.staging_dir / "transcript.json").read_text())
        self.assertEqual(data["metadata"]["model"], "small")

    def test_transcript_metadata_duration(self) -> None:
        self._run()
        data = json.loads((self.staging_dir / "transcript.json").read_text())
        self.assertAlmostEqual(data["metadata"]["duration_s"], 8.0)

    # ── Handover Record written to harness memory ─────────────────────────

    def test_handover_record_file_created(self) -> None:
        self._run()
        job_dir = self.jobs_root / "job_001"
        records = list(job_dir.glob("*.json"))
        self.assertEqual(len(records), 1)

    def test_handover_record_skill_name(self) -> None:
        record = self._run()
        self.assertEqual(record.skill, "transcriber")

    def test_handover_record_status_success(self) -> None:
        record = self._run()
        self.assertEqual(record.status, "success")

    def test_handover_record_output_path_points_to_transcript(self) -> None:
        record = self._run()
        self.assertTrue(record.output_path.endswith("transcript.json"))

    def test_handover_record_cursor_end_from_duration(self) -> None:
        """cursor_end must represent the audio duration (8 s → 00:00:08.000)."""
        record = self._run()
        self.assertEqual(record.cursor_end, "00:00:08.000")

    def test_handover_record_payload_output_block(self) -> None:
        """Payload must contain the output block with full_text and words."""
        record = self._run()
        self.assertIn("output", record.payload)
        self.assertIn("full_text", record.payload["output"])
        self.assertIn("words", record.payload["output"])
        self.assertIn("vad_segments", record.payload["output"])

    def test_handover_record_payload_metadata_block(self) -> None:
        record = self._run()
        meta = record.payload.get("metadata", {})
        self.assertIn("model", meta)
        self.assertIn("duration", meta)
        self.assertIn("word_count", meta)

    def test_handover_record_filename_format(self) -> None:
        """Filename must follow 001_transcriber.json convention."""
        self._run()
        job_dir = self.jobs_root / "job_001"
        names = [p.name for p in job_dir.glob("*.json")]
        self.assertIn("001_transcriber.json", names)

    # ── Retry index ───────────────────────────────────────────────────────

    def test_retry_index_reflected_in_record(self) -> None:
        record = self._run(retry_index=2)
        self.assertEqual(record.retry_index, 2)

    def test_retry_filename_has_suffix(self) -> None:
        self._run(retry_index=1)
        job_dir = self.jobs_root / "job_001"
        names = [p.name for p in job_dir.glob("*.json")]
        self.assertTrue(any("retry1" in n for n in names))

    # ── Failure path ──────────────────────────────────────────────────────

    def test_backend_failure_writes_failed_record(self) -> None:
        record = transcriber_run(
            job_id="job_fail",
            source_file=str(self.audio_path),
            backend=ErrorWhisperBackend(),
            staging_dir=self.staging_dir,
        )
        self.assertEqual(record.status, "failed")
        self.assertIsNotNone(record.error)

    def test_backend_failure_still_writes_to_memory(self) -> None:
        """Even on failure, a record must be committed to harness memory."""
        transcriber_run(
            job_id="job_fail2",
            source_file=str(self.audio_path),
            backend=ErrorWhisperBackend(),
            staging_dir=self.staging_dir,
        )
        job_dir = self.jobs_root / "job_fail2"
        records = list(job_dir.glob("*.json"))
        self.assertEqual(len(records), 1)

    def test_missing_source_file_writes_failed_record(self) -> None:
        record = transcriber_run(
            job_id="job_nofile",
            source_file="/does/not/exist.mp3",
            backend=self.backend,
            staging_dir=self.staging_dir,
        )
        self.assertEqual(record.status, "failed")

    # ── Stateless: no cross-job contamination ────────────────────────────

    def test_two_jobs_write_separate_records(self) -> None:
        """Each job_id must produce records in its own directory."""
        self._run(job_id="job_alpha")
        self._run(job_id="job_beta")

        alpha_dir = self.jobs_root / "job_alpha"
        beta_dir = self.jobs_root / "job_beta"
        self.assertTrue(alpha_dir.exists())
        self.assertTrue(beta_dir.exists())
        self.assertEqual(len(list(alpha_dir.glob("*.json"))), 1)
        self.assertEqual(len(list(beta_dir.glob("*.json"))), 1)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: MemoryManager.find_resume_point after transcriber runs
# ═══════════════════════════════════════════════════════════════════════════

class TestTranscriberHarnessIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.audio_path = self.tmp_path / "sample.wav"
        self.audio_path.write_bytes(b"\x00" * 64)

        self.jobs_root = self.tmp_path / "jobs"
        self._patcher = patch.object(mgr_mod, "JOBS_ROOT", self.jobs_root)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    def test_resume_point_next_skill_is_cutter_after_transcriber(self) -> None:
        """After transcriber succeeds, resume point must advance to cutter."""
        transcriber_run(
            job_id="job_int_001",
            source_file=str(self.audio_path),
            backend=MockWhisperBackend(),
            staging_dir=self.tmp_path / "staging" / "job_int_001",
        )
        mgr = MemoryManager("job_int_001")
        resume = mgr.find_resume_point()

        self.assertEqual(resume.next_skill, "cutter")
        self.assertIn("transcriber", resume.completed)

    def test_resume_cursor_is_audio_duration(self) -> None:
        """The cursor passed to Cutter must equal the audio duration timecode."""
        transcriber_run(
            job_id="job_int_002",
            source_file=str(self.audio_path),
            backend=MockWhisperBackend(),
            staging_dir=self.tmp_path / "staging" / "job_int_002",
        )
        mgr = MemoryManager("job_int_002")
        resume = mgr.find_resume_point()
        # MockWhisperBackend returns duration 8.0 s → 00:00:08.000
        self.assertEqual(resume.cursor, "00:00:08.000")

    def test_prior_output_returns_transcript_path(self) -> None:
        """ResumePoint.prior_output('transcriber') must return the transcript path."""
        staging = self.tmp_path / "staging" / "job_int_003"
        transcriber_run(
            job_id="job_int_003",
            source_file=str(self.audio_path),
            backend=MockWhisperBackend(),
            staging_dir=staging,
        )
        mgr = MemoryManager("job_int_003")
        resume = mgr.find_resume_point()
        output = resume.prior_output("transcriber")
        self.assertIsNotNone(output)
        self.assertTrue(str(output).endswith("transcript.json"))


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
