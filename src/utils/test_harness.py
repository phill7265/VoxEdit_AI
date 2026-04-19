"""
src/utils/test_harness.py

Unit tests for:
  · harness/memory/manager.py   — MemoryManager, SkillRecord, ResumePoint
  · src/pipeline/context_manager.py — build_skill_context, timecode helpers,
                                      transcript/timeline windowing

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_harness.py -v

No external dependencies beyond the standard library and pytest.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Project root on sys.path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.memory.manager import (
    SKILL_ORDER,
    MemoryManager,
    ResumePoint,
    SkillRecord,
)
from src.pipeline.context_manager import (
    CutterContext,
    DesignerContext,
    ExporterContext,
    TranscriberContext,
    _seconds_to_tc,
    _tc_to_seconds,
    build_skill_context,
)
import src.pipeline.context_manager as ctx_mod


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_record(
    job_id: str,
    skill: str,
    status: str = "success",
    cursor_start: str = "00:00:00.000",
    cursor_end: str = "00:01:00.000",
    output_path: str = "staging/out.json",
    retry_index: int = 0,
    payload: dict | None = None,
) -> SkillRecord:
    return SkillRecord(
        job_id=job_id,
        skill=skill,
        status=status,
        output_path=output_path,
        cursor_start=cursor_start,
        cursor_end=cursor_end,
        payload=payload or {},
        retry_index=retry_index,
    )


def _transcript_words(timestamps_ms: list[tuple[int, int, str]]) -> list[dict]:
    """Build a word-list transcript from (start_ms, end_ms, word) tuples."""
    return [
        {"word": w, "start_ms": s, "end_ms": e, "confidence": 0.95}
        for s, e, w in timestamps_ms
    ]


# ═══════════════════════════════════════════════════════════════════════════
# MemoryManager tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryManagerWrite(unittest.TestCase):
    """MemoryManager.write() — file creation, naming, and no-overwrite contract."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_jobs = Path(self._tmp.name)
        # Redirect JOBS_ROOT so tests never write to the real harness directory
        self._patcher = patch("harness.memory.manager.JOBS_ROOT", self.tmp_jobs)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    def _mgr(self, job_id: str = "job_test_001") -> MemoryManager:
        return MemoryManager(job_id)

    # ── Correct filename format ───────────────────────────────────────────

    def test_filename_uses_skill_order_sequence(self) -> None:
        mgr = self._mgr()
        path = mgr.write(_make_record("job_test_001", "transcriber"))
        self.assertEqual(path.name, "001_transcriber.json")

    def test_cutter_gets_sequence_002(self) -> None:
        mgr = self._mgr()
        path = mgr.write(_make_record("job_test_001", "cutter"))
        self.assertEqual(path.name, "002_cutter.json")

    def test_retry_suffix_appended(self) -> None:
        mgr = self._mgr()
        rec = _make_record("job_test_001", "cutter", retry_index=1)
        path = mgr.write(rec)
        self.assertEqual(path.name, "002_cutter_retry1.json")

    def test_file_content_is_valid_json(self) -> None:
        mgr = self._mgr()
        rec = _make_record("job_test_001", "transcriber", payload={"word_count": 120})
        path = mgr.write(rec)
        data = json.loads(path.read_text())
        self.assertEqual(data["skill"], "transcriber")
        self.assertEqual(data["payload"]["word_count"], 120)

    # ── No-overwrite contract ────────────────────────────────────────────

    def test_write_does_not_overwrite_existing_file(self) -> None:
        mgr = self._mgr()
        mgr.write(_make_record("job_test_001", "transcriber"))
        # Write a second record for the same skill — must not clobber the first
        path2 = mgr.write(_make_record("job_test_001", "transcriber"))
        self.assertIn("conflict", path2.name)

    # ── Job directory created automatically ──────────────────────────────

    def test_job_directory_created_on_init(self) -> None:
        job_dir = self.tmp_jobs / "job_new_999"
        self.assertFalse(job_dir.exists())
        MemoryManager("job_new_999")
        self.assertTrue(job_dir.exists())

    # ── sequence assigned to record ──────────────────────────────────────

    def test_sequence_set_on_record_after_write(self) -> None:
        mgr = self._mgr()
        rec = _make_record("job_test_001", "designer")
        mgr.write(rec)
        self.assertEqual(rec.sequence, 3)   # designer is index 2 → seq 3


class TestMemoryManagerLoad(unittest.TestCase):
    """MemoryManager.load_all() and load_successful()."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_jobs = Path(self._tmp.name)
        self._patcher = patch("harness.memory.manager.JOBS_ROOT", self.tmp_jobs)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    def _mgr(self) -> MemoryManager:
        return MemoryManager("job_load_001")

    def _write_all(self, mgr: MemoryManager, specs: list[tuple]) -> None:
        """Write multiple records from (skill, status) tuples."""
        for skill, status in specs:
            mgr.write(_make_record("job_load_001", skill, status=status))

    def test_load_all_returns_all_records(self) -> None:
        mgr = self._mgr()
        self._write_all(mgr, [("transcriber", "success"), ("cutter", "failed")])
        records = mgr.load_all()
        self.assertEqual(len(records), 2)

    def test_load_all_sorted_by_filename(self) -> None:
        mgr = self._mgr()
        self._write_all(mgr, [
            ("transcriber", "success"),
            ("cutter", "success"),
            ("designer", "success"),
        ])
        records = mgr.load_all()
        skills = [r.skill for r in records]
        self.assertEqual(skills, ["transcriber", "cutter", "designer"])

    def test_load_successful_excludes_failed(self) -> None:
        mgr = self._mgr()
        self._write_all(mgr, [
            ("transcriber", "success"),
            ("cutter", "failed"),
            ("designer", "success"),
        ])
        successes = mgr.load_successful()
        self.assertEqual(len(successes), 2)
        skills = {r.skill for r in successes}
        self.assertNotIn("cutter", skills)

    def test_load_empty_job_returns_empty_list(self) -> None:
        mgr = self._mgr()
        self.assertEqual(mgr.load_all(), [])
        self.assertEqual(mgr.load_successful(), [])


class TestFindResumePoint(unittest.TestCase):
    """MemoryManager.find_resume_point() — the core session-resume logic."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_jobs = Path(self._tmp.name)
        self._patcher = patch("harness.memory.manager.JOBS_ROOT", self.tmp_jobs)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    def _mgr(self, job_id: str = "job_resume_001") -> MemoryManager:
        return MemoryManager(job_id)

    # ── Fresh job ─────────────────────────────────────────────────────────

    def test_fresh_job_starts_at_transcriber(self) -> None:
        mgr = self._mgr()
        rp = mgr.find_resume_point()
        self.assertEqual(rp.next_skill, "transcriber")
        self.assertEqual(rp.cursor, "00:00:00.000")
        self.assertEqual(rp.completed, [])

    # ── Partial completion ─────────────────────────────────────────────────

    def test_resumes_after_last_success(self) -> None:
        """transcriber succeeded, cutter failed → resume at cutter."""
        mgr = self._mgr()
        mgr.write(_make_record(
            "job_resume_001", "transcriber",
            status="success", cursor_end="00:00:30.000",
        ))
        mgr.write(_make_record(
            "job_resume_001", "cutter",
            status="failed", cursor_end="00:00:00.000",
        ))
        rp = mgr.find_resume_point()
        self.assertEqual(rp.next_skill, "cutter")

    def test_cursor_comes_from_last_successful_skill(self) -> None:
        mgr = self._mgr()
        mgr.write(_make_record(
            "job_resume_001", "transcriber",
            status="success", cursor_end="00:02:00.000",
        ))
        rp = mgr.find_resume_point()
        self.assertEqual(rp.cursor, "00:02:00.000")

    def test_completed_list_contains_only_successes(self) -> None:
        mgr = self._mgr()
        mgr.write(_make_record("job_resume_001", "transcriber", status="success"))
        mgr.write(_make_record("job_resume_001", "cutter", status="failed"))
        rp = mgr.find_resume_point()
        self.assertIn("transcriber", rp.completed)
        self.assertNotIn("cutter", rp.completed)

    # ── Full completion ────────────────────────────────────────────────────

    def test_complete_job_has_no_next_skill(self) -> None:
        mgr = self._mgr()
        for skill in SKILL_ORDER:
            mgr.write(_make_record("job_resume_001", skill, status="success"))
        rp = mgr.find_resume_point()
        self.assertIsNone(rp.next_skill)
        self.assertTrue(rp.is_complete)

    def test_complete_job_completed_list_matches_skill_order(self) -> None:
        mgr = self._mgr()
        for skill in SKILL_ORDER:
            mgr.write(_make_record("job_resume_001", skill, status="success"))
        rp = mgr.find_resume_point()
        self.assertEqual(rp.completed, list(SKILL_ORDER))

    # ── prior_output helper ────────────────────────────────────────────────

    def test_prior_output_returns_path_of_completed_skill(self) -> None:
        mgr = self._mgr()
        mgr.write(_make_record(
            "job_resume_001", "transcriber",
            status="success",
            output_path="staging/transcript.json",
        ))
        rp = mgr.find_resume_point()
        self.assertEqual(rp.prior_output("transcriber"), "staging/transcript.json")

    def test_prior_output_returns_none_for_incomplete_skill(self) -> None:
        mgr = self._mgr()
        rp = mgr.find_resume_point()
        self.assertIsNone(rp.prior_output("cutter"))

    # ── Retry selects highest-sequence record ──────────────────────────────

    def test_retry_record_supersedes_original_failure(self) -> None:
        """First cutter attempt failed; retry succeeded. Resume must advance past cutter."""
        mgr = self._mgr()
        mgr.write(_make_record("job_resume_001", "transcriber", status="success",
                               cursor_end="00:01:00.000"))
        mgr.write(_make_record("job_resume_001", "cutter", status="failed",
                               retry_index=0, cursor_end="00:00:00.000"))
        mgr.write(_make_record("job_resume_001", "cutter", status="success",
                               retry_index=1, cursor_end="00:02:00.000"))
        rp = mgr.find_resume_point()
        # cutter has a success now → next should be designer
        self.assertEqual(rp.next_skill, "designer")


# ═══════════════════════════════════════════════════════════════════════════
# Timecode helper tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTimecodeHelpers(unittest.TestCase):
    """_tc_to_seconds and _seconds_to_tc round-trip correctness."""

    def test_hms_to_seconds(self) -> None:
        self.assertAlmostEqual(_tc_to_seconds("00:01:30.000"), 90.0)

    def test_one_hour(self) -> None:
        self.assertAlmostEqual(_tc_to_seconds("01:00:00.000"), 3600.0)

    def test_fractional_seconds(self) -> None:
        self.assertAlmostEqual(_tc_to_seconds("00:00:02.500"), 2.5)

    def test_seconds_to_tc_format(self) -> None:
        self.assertEqual(_seconds_to_tc(90.0), "00:01:30.000")

    def test_seconds_to_tc_zero(self) -> None:
        self.assertEqual(_seconds_to_tc(0.0), "00:00:00.000")

    def test_round_trip(self) -> None:
        for tc in ["00:00:00.000", "00:01:30.500", "01:00:00.000", "00:02:15.000"]:
            self.assertAlmostEqual(_tc_to_seconds(_seconds_to_tc(_tc_to_seconds(tc))),
                                   _tc_to_seconds(tc), places=3)

    def test_negative_clamped_to_zero(self) -> None:
        self.assertEqual(_seconds_to_tc(-5.0), "00:00:00.000")


# ═══════════════════════════════════════════════════════════════════════════
# build_skill_context — per-skill contract tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildSkillContextTranscriber(unittest.TestCase):

    def test_returns_transcriber_context_type(self) -> None:
        ctx = build_skill_context(
            "transcriber", "job_001",
            source_file="/raw/video.mp4",
        )
        self.assertIsInstance(ctx, TranscriberContext)

    def test_source_file_is_passed_through(self) -> None:
        ctx = build_skill_context(
            "transcriber", "job_001",
            source_file="/raw/interview.mp4",
        )
        self.assertEqual(ctx.source_audio_path, "/raw/interview.mp4")

    def test_language_hint_default_is_ko(self) -> None:
        ctx = build_skill_context("transcriber", "job_001", source_file="x.mp4")
        self.assertEqual(ctx.language_hint, "ko")

    def test_language_hint_override(self) -> None:
        ctx = build_skill_context(
            "transcriber", "job_001",
            source_file="x.mp4", language_hint="en",
        )
        self.assertEqual(ctx.language_hint, "en")

    def test_no_job_config_or_history_fields(self) -> None:
        """TranscriberContext must expose only the two allow-listed fields."""
        ctx = build_skill_context("transcriber", "job_001", source_file="x.mp4")
        allowed = {"source_audio_path", "language_hint"}
        actual = set(ctx.__dataclass_fields__.keys())
        self.assertEqual(actual, allowed,
                         f"Unexpected fields in TranscriberContext: {actual - allowed}")


class TestBuildSkillContextCutter(unittest.TestCase):
    """Context Firewall: Cutter receives only the ±30 s transcript window."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        # Clear spec cache so tests don't depend on real spec file content
        ctx_mod._spec_cache.clear()

    def tearDown(self) -> None:
        ctx_mod._spec_cache.clear()
        self._tmp.cleanup()

    def _write_transcript(self, words: list[dict]) -> str:
        path = self.tmp_dir / "transcript.json"
        path.write_text(json.dumps(words))
        return str(path)

    # ── Window filtering ──────────────────────────────────────────────────

    def test_words_inside_window_are_included(self) -> None:
        """Cursor=60s, window=[30,90]s. Words at 40s and 80s must be included."""
        words = _transcript_words([
            (40_000, 41_000, "hello"),    # midpoint 40.5 s — inside window
            (80_000, 81_000, "world"),    # midpoint 80.5 s — inside window
        ])
        path = self._write_transcript(words)
        ctx = build_skill_context(
            "cutter", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            transcript_path=path,
        )
        self.assertIsInstance(ctx, CutterContext)
        self.assertEqual(len(ctx.transcript_window), 2)

    def test_cutter_sees_full_transcript(self) -> None:
        """Cutter always receives the full transcript regardless of cursor position.

        Unlike Designer/Exporter which use a 30s sliding window, Cutter needs all
        words to make global cut decisions. The window is set to [0, total_duration].
        """
        words = _transcript_words([
            (10_000, 11_000, "early"),     # midpoint 10.5 s
            (50_000, 51_000, "middle"),    # midpoint 50.5 s
            (120_000, 121_000, "late"),    # midpoint 120.5 s
        ])
        path = self._write_transcript(words)
        ctx = build_skill_context(
            "cutter", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            transcript_path=path,
        )
        window_words = [w["word"] for w in ctx.transcript_window]
        self.assertIn("early", window_words)
        self.assertIn("middle", window_words)
        self.assertIn("late", window_words)

    def test_window_exactly_30s_boundary(self) -> None:
        """All words within total duration are included for cutter."""
        words = _transcript_words([
            (60_000, 61_000, "boundary"),  # at 60s within 10-minute video
        ])
        path = self._write_transcript(words)
        ctx = build_skill_context(
            "cutter", "job_001",
            cursor="00:01:30.000",
            total_duration="00:10:00.000",
            transcript_path=path,
        )
        self.assertEqual(len(ctx.transcript_window), 1)

    # ── Window clamping ───────────────────────────────────────────────────

    def test_window_start_clamped_to_zero(self) -> None:
        """Cursor at 10s → window start must clamp to 0 (not −20 s)."""
        path = self._write_transcript([])
        ctx = build_skill_context(
            "cutter", "job_001",
            cursor="00:00:10.000",
            total_duration="00:05:00.000",
            transcript_path=path,
        )
        self.assertEqual(ctx.window_start, "00:00:00.000")

    def test_window_end_clamped_to_total_duration(self) -> None:
        """Cursor at total_duration − 5 s → window end must not exceed total_duration."""
        path = self._write_transcript([])
        ctx = build_skill_context(
            "cutter", "job_001",
            cursor="00:04:55.000",
            total_duration="00:05:00.000",
            transcript_path=path,
        )
        self.assertEqual(ctx.window_end, "00:05:00.000")

    # ── Spec values injected ───────────────────────────────────────────────

    def test_spec_silence_threshold_is_injected(self) -> None:
        path = self._write_transcript([])
        ctx = build_skill_context(
            "cutter", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            transcript_path=path,
        )
        self.assertAlmostEqual(ctx.silence_threshold_s, 0.5)

    def test_spec_min_clip_duration_is_injected(self) -> None:
        path = self._write_transcript([])
        ctx = build_skill_context(
            "cutter", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            transcript_path=path,
        )
        self.assertAlmostEqual(ctx.min_clip_duration_s, 0.5)

    # ── Firewall: full transcript NOT exposed ────────────────────────────

    def test_cutter_context_has_no_full_transcript_field(self) -> None:
        """The full transcript must never appear as a field on CutterContext."""
        path = self._write_transcript([])
        ctx = build_skill_context(
            "cutter", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            transcript_path=path,
        )
        fields = set(ctx.__dataclass_fields__.keys())
        self.assertNotIn("full_transcript", fields)
        self.assertNotIn("transcript_path", fields)
        self.assertNotIn("export_profile", fields)

    # ── Missing transcript file ────────────────────────────────────────────

    def test_missing_transcript_returns_empty_window(self) -> None:
        ctx = build_skill_context(
            "cutter", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            transcript_path="/nonexistent/transcript.json",
        )
        self.assertEqual(ctx.transcript_window, [])


class TestBuildSkillContextDesigner(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        ctx_mod._spec_cache.clear()

    def tearDown(self) -> None:
        ctx_mod._spec_cache.clear()
        self._tmp.cleanup()

    def _write_timeline(self, clips: list[dict]) -> str:
        path = self.tmp_dir / "timeline.json"
        path.write_text(json.dumps(clips))
        return str(path)

    def test_returns_designer_context_type(self) -> None:
        path = self._write_timeline([])
        ctx = build_skill_context(
            "designer", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            timeline_path=path,
        )
        self.assertIsInstance(ctx, DesignerContext)

    def test_clips_outside_window_are_excluded(self) -> None:
        """Cursor=60s, window=[30,90]s. Clip at 150–160s must not appear."""
        clips = [
            {"start": "00:00:45.000", "end": "00:00:55.000", "id": "inside"},
            {"start": "00:02:30.000", "end": "00:02:40.000", "id": "outside"},
        ]
        path = self._write_timeline(clips)
        ctx = build_skill_context(
            "designer", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            timeline_path=path,
        )
        ids = [c["id"] for c in ctx.timeline_segment]
        self.assertIn("inside", ids)
        self.assertNotIn("outside", ids)

    def test_spec_zoom_factor_injected(self) -> None:
        path = self._write_timeline([])
        ctx = build_skill_context(
            "designer", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            timeline_path=path,
        )
        self.assertAlmostEqual(ctx.jump_cut_zoom_factor, 1.1)

    def test_spec_duck_db_injected(self) -> None:
        path = self._write_timeline([])
        ctx = build_skill_context(
            "designer", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            timeline_path=path,
        )
        self.assertAlmostEqual(ctx.audio_duck_db, -20.0)

    def test_brand_kit_passed_through(self) -> None:
        path = self._write_timeline([])
        kit = {"font": "Pretendard", "primary_color": "#FF0000"}
        ctx = build_skill_context(
            "designer", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            timeline_path=path,
            brand_kit=kit,
        )
        self.assertEqual(ctx.brand_kit["font"], "Pretendard")

    def test_firewall_cutter_internals_not_exposed(self) -> None:
        path = self._write_timeline([])
        ctx = build_skill_context(
            "designer", "job_001",
            cursor="00:01:00.000",
            total_duration="00:05:00.000",
            timeline_path=path,
        )
        fields = set(ctx.__dataclass_fields__.keys())
        for forbidden in ("cut_list", "silence_candidates", "export_profile",
                          "transcript_path", "ffmpeg_filter"):
            self.assertNotIn(forbidden, fields,
                             f"DesignerContext must not expose '{forbidden}'")


class TestBuildSkillContextExporter(unittest.TestCase):

    def test_returns_exporter_context_type(self) -> None:
        ctx = build_skill_context(
            "exporter", "job_001",
            timeline_path="staging/timeline.json",
        )
        self.assertIsInstance(ctx, ExporterContext)

    def test_timeline_path_passed_through(self) -> None:
        ctx = build_skill_context(
            "exporter", "job_001",
            timeline_path="staging/annotated.json",
        )
        self.assertEqual(ctx.annotated_timeline_path, "staging/annotated.json")

    def test_default_export_profile_is_youtube(self) -> None:
        ctx = build_skill_context(
            "exporter", "job_001",
            timeline_path="staging/timeline.json",
        )
        self.assertEqual(ctx.export_profile.get("platform"), "youtube")

    def test_custom_export_profile_accepted(self) -> None:
        profile = {"platform": "instagram", "resolution": "1080x1080"}
        ctx = build_skill_context(
            "exporter", "job_001",
            timeline_path="staging/timeline.json",
            export_profile=profile,
        )
        self.assertEqual(ctx.export_profile["platform"], "instagram")

    def test_firewall_upstream_history_not_exposed(self) -> None:
        ctx = build_skill_context(
            "exporter", "job_001",
            timeline_path="staging/timeline.json",
        )
        fields = set(ctx.__dataclass_fields__.keys())
        for forbidden in ("transcript_path", "brand_kit", "silence_candidates",
                          "cut_list", "handover_records"):
            self.assertNotIn(forbidden, fields,
                             f"ExporterContext must not expose '{forbidden}'")


class TestBuildSkillContextErrors(unittest.TestCase):

    def test_unknown_skill_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as cm:
            build_skill_context("unknown_skill", "job_001")
        self.assertIn("unknown_skill", str(cm.exception))

    def test_error_message_lists_valid_skills(self) -> None:
        with self.assertRaises(ValueError) as cm:
            build_skill_context("mixer", "job_001")
        msg = str(cm.exception)
        for skill in ("transcriber", "cutter", "designer", "exporter"):
            self.assertIn(skill, msg)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: MemoryManager ↔ build_skill_context
# ═══════════════════════════════════════════════════════════════════════════

class TestHarnessIntegration(unittest.TestCase):
    """Verify that the resume cursor from MemoryManager flows correctly into
    build_skill_context's window calculation."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_jobs = Path(self._tmp.name)
        self._patcher = patch("harness.memory.manager.JOBS_ROOT", self.tmp_jobs)
        self._patcher.start()
        ctx_mod._spec_cache.clear()
        self.transcript_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._patcher.stop()
        ctx_mod._spec_cache.clear()
        self._tmp.cleanup()

    def _write_transcript(self, words: list[dict]) -> str:
        path = self.transcript_dir / "transcript.json"
        path.write_text(json.dumps(words))
        return str(path)

    def test_resume_cursor_becomes_cutter_window_center(self) -> None:
        """
        Scenario:
          1. transcriber succeeded, cursor_end = 00:02:00.000 (transcript end)
          2. cutter failed
          → resume_point.cursor = 00:02:00.000
          → CutterContext sees full transcript [0, total_duration] regardless of
            cursor position, because Cutter needs all words for global cut decisions.
        """
        mgr = MemoryManager("job_integration_001")
        mgr.write(_make_record(
            "job_integration_001", "transcriber",
            status="success", cursor_end="00:02:00.000",
        ))
        mgr.write(_make_record(
            "job_integration_001", "cutter",
            status="failed",
        ))

        resume = mgr.find_resume_point()
        self.assertEqual(resume.next_skill, "cutter")
        self.assertEqual(resume.cursor, "00:02:00.000")

        words = _transcript_words([
            (60_000, 61_000, "early"),     # 60.5 s
            (100_000, 101_000, "middle"),  # 100.5 s
            (200_000, 201_000, "late"),    # 200.5 s
        ])
        transcript_path = self._write_transcript(words)

        ctx = build_skill_context(
            "cutter", "job_integration_001",
            cursor=resume.cursor,
            total_duration="00:10:00.000",
            transcript_path=transcript_path,
        )

        window_words = [w["word"] for w in ctx.transcript_window]
        self.assertIn("early", window_words)
        self.assertIn("middle", window_words)
        self.assertIn("late", window_words)

    def test_complete_job_produces_no_next_skill(self) -> None:
        mgr = MemoryManager("job_integration_002")
        for skill in SKILL_ORDER:
            mgr.write(_make_record("job_integration_002", skill, status="success"))
        resume = mgr.find_resume_point()
        self.assertTrue(resume.is_complete)
        self.assertIsNone(resume.next_skill)


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
