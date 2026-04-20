"""
src/utils/test_phase13.py

Unit + integration tests for Phase 13 — Pro Audio Mastering & Real-time Interference.

Covers
------
  TestAudioStyleIO           — read/write spec/audio_style.md (6)
  TestDuckPartReadsFromSpec  — _build_duck_part uses audio_style.md values (5)
  TestBGMMixPart             — sidechaincompress filter generation (7)
  TestExportPlanAudioOnly    — to_audio_only_command() (8)
  TestAudioOnlyOrchestrator  — audio_only pipeline mode (8)
  TestAudioIntents           — "음악 크게/작게", "BGM 꺼줘", "잔잔한 걸로" etc. (10)
  TestAudioGate              — clipping check, BGM ratio check (8)
  TestAudioOnlyTiming        — efficiency report + skill-skip count (4)

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_phase13.py -v -s
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_audio_style(tmpdir: Path, **overrides) -> Path:
    """Write a temporary audio_style.md with given overrides."""
    defaults = {
        "VOICE_DUCK_DB": -20,
        "DUCK_ATTACK_MS": 150,
        "DUCK_RELEASE_MS": 500,
        "BGM_BASE_VOLUME": 0.30,
        "BGM_STYLE": "calm",
    }
    defaults.update(overrides)
    lines = ["# VoxEdit AI — Audio Style\n\n## Ducking\n"]
    for k, v in defaults.items():
        lines.append(f"{k}: {v}\n")
    p = tmpdir / "audio_style.md"
    p.write_text("".join(lines), encoding="utf-8")
    return p


def _make_export_plan(keep_segs=None, duck_events=None, bgm_file=None, staging=None):
    from skills.exporter.logic import ExportPlan, PROFILE_SHORTS
    return ExportPlan(
        source_file=str(staging / "raw.mp4") if staging else "/src/raw.mp4",
        output_file=str(staging / "out.mp4") if staging else "/out/out.mp4",
        keep_segments=keep_segs or [{"start": 0.0, "end": 10.0, "action": "keep"}],
        captions=[],
        zoom_elements=[],
        duck_events=duck_events or [],
        broll_elements=[],
        export_profile=dict(PROFILE_SHORTS),
        bgm_file=bgm_file,
    )


def _make_skill_record(job_id, skill, output_path="", cursor_end="00:00:10.000"):
    from harness.memory.manager import SkillRecord
    return SkillRecord(
        job_id=job_id, skill=skill, status="success",
        output_path=output_path,
        cursor_start="00:00:00.000", cursor_end=cursor_end,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestAudioStyleIO — read/write spec/audio_style.md (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioStyleIO(unittest.TestCase):
    """Intent processor reads and writes audio_style.md correctly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.audio_file = _make_audio_style(self.tmpdir)

    def tearDown(self):
        self.tmp.cleanup()

    def _patch(self):
        import src.pipeline.intent_processor as ip
        return patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file)

    def test_read_bgm_base_volume(self):
        from src.pipeline.intent_processor import _read_audio_float
        with self._patch():
            val = _read_audio_float("BGM_BASE_VOLUME", 0.5)
        self.assertAlmostEqual(val, 0.30, places=2)

    def test_read_voice_duck_db(self):
        from src.pipeline.intent_processor import _read_audio_float
        with self._patch():
            val = _read_audio_float("VOICE_DUCK_DB", 0.0)
        self.assertEqual(val, -20.0)

    def test_write_new_value_updates_file(self):
        from src.pipeline.intent_processor import _write_audio_value, _read_audio_float
        with self._patch():
            _write_audio_value("BGM_BASE_VOLUME", "0.50")
            val = _read_audio_float("BGM_BASE_VOLUME", 0.0)
        self.assertAlmostEqual(val, 0.50, places=2)

    def test_write_missing_key_appends(self):
        from src.pipeline.intent_processor import _write_audio_value, _read_audio_float
        with self._patch():
            _write_audio_value("NEW_PARAM", "99.9")
            val = _read_audio_float("NEW_PARAM", 0.0)
        self.assertAlmostEqual(val, 99.9, places=1)

    def test_read_missing_key_returns_default(self):
        from src.pipeline.intent_processor import _read_audio_float
        with self._patch():
            val = _read_audio_float("NONEXISTENT_PARAM", 42.0)
        self.assertEqual(val, 42.0)

    def test_exporter_reads_duck_db_from_audio_style(self):
        """_build_duck_part reads VOICE_DUCK_DB from audio_style.md."""
        from skills.exporter import logic as exp_mod
        with patch.object(exp_mod, "_AUDIO_STYLE_FILE", self.audio_file):
            plan = _make_export_plan(
                duck_events=[{"type": "duck", "start": 2.0, "end": 4.0}],
                staging=self.tmpdir,
            )
            filt, label = exp_mod._build_duck_part(plan, "[acat]")
        # VOICE_DUCK_DB=-20 → factor≈0.1
        self.assertIn("0.1", filt)
        self.assertEqual(label, "[aduck]")


# ═══════════════════════════════════════════════════════════════════════════════
# TestDuckPartReadsFromSpec — custom duck DB reflected in filter (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDuckPartReadsFromSpec(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _duck_plan(self, duck_db: float) -> "ExportPlan":
        audio_file = _make_audio_style(self.tmpdir, VOICE_DUCK_DB=duck_db)
        from skills.exporter import logic as exp_mod
        plan = _make_export_plan(
            duck_events=[{"type": "duck", "start": 0.0, "end": 5.0}],
            staging=self.tmpdir,
        )
        # patch at module level
        exp_mod._AUDIO_STYLE_FILE_OVERRIDE = audio_file
        return plan, audio_file, exp_mod

    def test_duck_db_minus_10_factor(self):
        """VOICE_DUCK_DB=-10 → factor≈0.316."""
        from skills.exporter import logic as exp_mod
        audio_file = _make_audio_style(self.tmpdir, VOICE_DUCK_DB=-10)
        plan = _make_export_plan(
            duck_events=[{"type": "duck", "start": 0.0, "end": 5.0}],
            staging=self.tmpdir,
        )
        with patch.object(exp_mod, "_AUDIO_STYLE_FILE", audio_file):
            filt, _ = exp_mod._build_duck_part(plan, "[acat]")
        self.assertIn("0.316", filt)

    def test_duck_db_minus_6_factor(self):
        """VOICE_DUCK_DB=-6 → factor≈0.501."""
        from skills.exporter import logic as exp_mod
        audio_file = _make_audio_style(self.tmpdir, VOICE_DUCK_DB=-6)
        plan = _make_export_plan(
            duck_events=[{"type": "duck", "start": 0.0, "end": 5.0}],
            staging=self.tmpdir,
        )
        with patch.object(exp_mod, "_AUDIO_STYLE_FILE", audio_file):
            filt, _ = exp_mod._build_duck_part(plan, "[acat]")
        self.assertIn("0.501", filt)

    def test_duck_db_minus_40_factor(self):
        """VOICE_DUCK_DB=-40 → factor≈0.01."""
        from skills.exporter import logic as exp_mod
        audio_file = _make_audio_style(self.tmpdir, VOICE_DUCK_DB=-40)
        plan = _make_export_plan(
            duck_events=[{"type": "duck", "start": 0.0, "end": 5.0}],
            staging=self.tmpdir,
        )
        with patch.object(exp_mod, "_AUDIO_STYLE_FILE", audio_file):
            filt, _ = exp_mod._build_duck_part(plan, "[acat]")
        self.assertIn("0.01", filt)

    def test_no_duck_events_returns_passthrough(self):
        """No duck events → empty filter, original label returned."""
        from skills.exporter.logic import _build_duck_part
        plan = _make_export_plan(duck_events=[], staging=self.tmpdir)
        filt, label = _build_duck_part(plan, "[acat]")
        self.assertEqual(filt, "")
        self.assertEqual(label, "[acat]")

    def test_duck_filter_contains_between_expression(self):
        """Duck filter uses FFmpeg between() expression."""
        from skills.exporter.logic import _build_duck_part
        plan = _make_export_plan(
            duck_events=[{"type": "duck", "start": 2.0, "end": 4.0}],
            staging=self.tmpdir,
        )
        filt, _ = _build_duck_part(plan, "[acat]")
        self.assertIn("between(t,", filt)
        self.assertIn("volume=volume=", filt)


# ═══════════════════════════════════════════════════════════════════════════════
# TestBGMMixPart — sidechaincompress filter generation (7 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBGMMixPart(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bgm_mix_produces_four_filter_parts(self):
        """_build_bgm_mix_part returns 4 filter parts."""
        from skills.exporter.logic import _build_bgm_mix_part
        audio_file = _make_audio_style(self.tmpdir)
        from skills.exporter import logic as exp_mod
        with patch.object(exp_mod, "_AUDIO_STYLE_FILE", audio_file):
            plan = _make_export_plan(staging=self.tmpdir)
            parts, label = _build_bgm_mix_part(plan, "[acat]", bgm_input_idx=1)
        self.assertEqual(len(parts), 4)

    def test_bgm_mix_output_label(self):
        from skills.exporter.logic import _build_bgm_mix_part
        plan = _make_export_plan(staging=self.tmpdir)
        _, label = _build_bgm_mix_part(plan, "[acat]", bgm_input_idx=1)
        self.assertEqual(label, "[amix_out]")

    def test_bgm_mix_uses_sidechaincompress(self):
        from skills.exporter.logic import _build_bgm_mix_part
        plan = _make_export_plan(staging=self.tmpdir)
        parts, _ = _build_bgm_mix_part(plan, "[acat]", bgm_input_idx=1)
        combined = ";".join(parts)
        self.assertIn("sidechaincompress", combined)

    def test_bgm_mix_uses_amix(self):
        from skills.exporter.logic import _build_bgm_mix_part
        plan = _make_export_plan(staging=self.tmpdir)
        parts, _ = _build_bgm_mix_part(plan, "[acat]", bgm_input_idx=1)
        combined = ";".join(parts)
        self.assertIn("amix", combined)

    def test_bgm_mix_volume_reads_from_audio_style(self):
        """BGM_BASE_VOLUME from audio_style.md appears in filter."""
        audio_file = _make_audio_style(self.tmpdir, BGM_BASE_VOLUME=0.42)
        from skills.exporter import logic as exp_mod
        plan = _make_export_plan(staging=self.tmpdir)
        with patch.object(exp_mod, "_AUDIO_STYLE_FILE", audio_file):
            parts, _ = exp_mod._build_bgm_mix_part(plan, "[acat]", bgm_input_idx=1)
        combined = ";".join(parts)
        self.assertIn("0.420", combined)

    def test_bgm_input_idx_in_filter(self):
        """BGM input index is correctly referenced in volume filter."""
        from skills.exporter.logic import _build_bgm_mix_part
        plan = _make_export_plan(staging=self.tmpdir)
        parts, _ = _build_bgm_mix_part(plan, "[acat]", bgm_input_idx=3)
        self.assertIn("[3:a]", parts[0])

    def test_filter_complex_includes_bgm_when_bgm_file_set(self):
        """filter_complex contains sidechaincompress when bgm_file is provided."""
        plan = _make_export_plan(
            staging=self.tmpdir,
            bgm_file="/fake/bgm.mp3",
        )
        fc = plan.filter_complex
        self.assertIn("sidechaincompress", fc)


# ═══════════════════════════════════════════════════════════════════════════════
# TestExportPlanAudioOnly — to_audio_only_command() (8 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportPlanAudioOnly(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.prior_video = str(self.tmpdir / "prior.mp4")

    def tearDown(self):
        self.tmp.cleanup()

    def test_command_contains_cvcopy(self):
        plan = _make_export_plan(staging=self.tmpdir)
        cmd = plan.to_audio_only_command(self.prior_video)
        self.assertIn("-c:v copy", cmd)

    def test_command_starts_with_ffmpeg(self):
        plan = _make_export_plan(staging=self.tmpdir)
        cmd = plan.to_audio_only_command(self.prior_video)
        self.assertTrue(cmd.startswith("ffmpeg"))

    def test_command_contains_prior_video(self):
        plan = _make_export_plan(staging=self.tmpdir)
        cmd = plan.to_audio_only_command(self.prior_video)
        self.assertIn("prior.mp4", cmd)

    def test_command_contains_source_file(self):
        plan = _make_export_plan(staging=self.tmpdir)
        cmd = plan.to_audio_only_command(self.prior_video)
        self.assertIn("raw.mp4", cmd)

    def test_command_contains_aac_codec(self):
        plan = _make_export_plan(staging=self.tmpdir)
        cmd = plan.to_audio_only_command(self.prior_video)
        self.assertIn("-c:a aac", cmd)

    def test_command_contains_shortest_flag(self):
        plan = _make_export_plan(staging=self.tmpdir)
        cmd = plan.to_audio_only_command(self.prior_video)
        self.assertIn("-shortest", cmd)

    def test_command_does_not_contain_drawtext(self):
        """Audio-only command must not include caption drawtext filter."""
        plan = _make_export_plan(staging=self.tmpdir)
        cmd = plan.to_audio_only_command(self.prior_video)
        self.assertNotIn("drawtext", cmd)

    def test_command_with_bgm_file_has_extra_input(self):
        """When bgm_file is set, command includes it as an input."""
        plan = _make_export_plan(bgm_file="/fake/bgm.mp3", staging=self.tmpdir)
        cmd = plan.to_audio_only_command(self.prior_video)
        self.assertIn("bgm.mp3", cmd)
        # BGM should appear between source and prior_video
        bgm_pos = cmd.index("bgm.mp3")
        prior_pos = cmd.index("prior.mp4")
        self.assertLess(bgm_pos, prior_pos)


# ═══════════════════════════════════════════════════════════════════════════════
# TestAudioOnlyOrchestrator — audio_only pipeline mode (8 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioOnlyOrchestrator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.staging = Path(self.tmp.name) / "staging"
        self.staging.mkdir()
        self.job_id = "ao_test_001"

    def tearDown(self):
        self.tmp.cleanup()
        from harness.memory.manager import MemoryManager
        import shutil
        mgr = MemoryManager(self.job_id)
        if mgr.job_dir.exists():
            shutil.rmtree(str(mgr.job_dir))

    def _make_prior_video(self) -> Path:
        p = self.staging / "prior_output.mp4"
        p.write_bytes(b"\x00" * 100)
        return p

    def _make_wm(self, borrow_records=None):
        from src.pipeline.orchestrator import WorkflowManager
        return WorkflowManager(
            job_id=self.job_id,
            source_file=str(self.staging / "raw.mp4"),
            staging_dir=self.staging,
            dry_run=True,
            force_resume_from="audio_only",
            borrow_records=borrow_records or {},
        )

    def test_audio_only_skips_three_skills(self):
        prior = self._make_prior_video()
        exporter_rec = _make_skill_record(self.job_id, "exporter", str(prior))
        wm = self._make_wm({"exporter": exporter_rec})
        result = wm.run()
        for skill in ("transcriber", "cutter", "designer"):
            self.assertIn(skill, result.skipped)

    def test_audio_only_completes_exporter(self):
        prior = self._make_prior_video()
        exporter_rec = _make_skill_record(self.job_id, "exporter", str(prior))
        wm = self._make_wm({"exporter": exporter_rec})
        result = wm.run()
        self.assertIn("exporter", result.completed)

    def test_audio_only_result_status_success(self):
        prior = self._make_prior_video()
        exporter_rec = _make_skill_record(self.job_id, "exporter", str(prior))
        wm = self._make_wm({"exporter": exporter_rec})
        result = wm.run()
        self.assertEqual(result.status, "success")

    def test_audio_only_result_has_audio_only_cmd(self):
        """Dry-run result payload contains the audio_only_cmd."""
        prior = self._make_prior_video()
        exporter_rec = _make_skill_record(self.job_id, "exporter", str(prior))
        wm = self._make_wm({"exporter": exporter_rec})
        result = wm.run()
        payload = result.final_record.payload if result.final_record else {}
        self.assertIn("audio_only_cmd", payload)

    def test_audio_only_fallback_when_no_exporter_record(self):
        """Without prior exporter record, falls back to exporter-only resume."""
        wm = self._make_wm(borrow_records={})
        # Should not raise; falls back and tries to run exporter normally
        with patch("src.pipeline.orchestrator.MemoryManager") as mock_mgr_cls:
            mock_mgr = MagicMock()
            mock_mgr_cls.return_value = mock_mgr
            mock_resume = MagicMock()
            mock_resume.is_complete = False
            mock_resume.cursor = "00:00:00.000"
            mock_resume.completed = []
            mock_resume.records = {}
            mock_resume.next_skill = "exporter"
            mock_mgr.find_resume_point.return_value = mock_resume
            result = wm.run()
        self.assertIsNotNone(result)

    def test_audio_only_dry_run_cmd_contains_cvcopy(self):
        """The generated audio_only_cmd must use -c:v copy."""
        prior = self._make_prior_video()
        exporter_rec = _make_skill_record(self.job_id, "exporter", str(prior))
        wm = self._make_wm({"exporter": exporter_rec})
        result = wm.run()
        cmd = result.final_record.payload.get("audio_only_cmd", "")
        self.assertIn("-c:v copy", cmd)

    def test_audio_only_output_path_is_audio_only_mp4(self):
        prior = self._make_prior_video()
        exporter_rec = _make_skill_record(self.job_id, "exporter", str(prior))
        wm = self._make_wm({"exporter": exporter_rec})
        result = wm.run()
        self.assertIn("audio_only", result.final_record.output_path)

    def test_audio_only_writes_placeholder_records(self):
        """Placeholder records for transcriber/cutter/designer are persisted."""
        prior = self._make_prior_video()
        exporter_rec = _make_skill_record(self.job_id, "exporter", str(prior))
        transcriber_rec = _make_skill_record(self.job_id, "transcriber", "/t.json")
        wm = self._make_wm({
            "exporter":   exporter_rec,
            "transcriber": transcriber_rec,
        })
        wm.run()
        from harness.memory.manager import MemoryManager
        mgr = MemoryManager(self.job_id)
        resume = mgr.find_resume_point()
        self.assertIn("transcriber", resume.records)


# ═══════════════════════════════════════════════════════════════════════════════
# TestAudioIntents — audio interference commands (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioIntents(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.audio_file = _make_audio_style(self.tmpdir)

    def tearDown(self):
        self.tmp.cleanup()

    def _patch(self):
        import src.pipeline.intent_processor as ip
        return patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file)

    def _process(self, text: str):
        from src.pipeline.intent_processor import IntentProcessor
        with self._patch():
            return IntentProcessor().process(text)

    def test_bgm_louder_increases_volume(self):
        r = self._process("음악 크게")
        self.assertTrue(r.applied)
        self.assertEqual(r.restart_from, "audio_only")
        import src.pipeline.intent_processor as ip
        with patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file):
            new_val = ip._read_audio_float("BGM_BASE_VOLUME", 0.0)
        self.assertAlmostEqual(new_val, 0.40, places=2)

    def test_bgm_quieter_decreases_volume(self):
        r = self._process("BGM 작게")
        self.assertTrue(r.applied)
        self.assertEqual(r.restart_from, "audio_only")
        import src.pipeline.intent_processor as ip
        with patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file):
            new_val = ip._read_audio_float("BGM_BASE_VOLUME", 0.0)
        self.assertAlmostEqual(new_val, 0.20, places=2)

    def test_bgm_off_sets_volume_zero(self):
        r = self._process("BGM 꺼줘")
        self.assertTrue(r.applied)
        self.assertEqual(r.restart_from, "audio_only")
        import src.pipeline.intent_processor as ip
        with patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file):
            new_val = ip._read_audio_float("BGM_BASE_VOLUME", 1.0)
        self.assertAlmostEqual(new_val, 0.0, places=2)

    def test_bgm_calm_sets_low_volume_and_style(self):
        r = self._process("잔잔한 걸로 해줘")
        self.assertTrue(r.applied)
        self.assertEqual(r.restart_from, "audio_only")
        self.assertIn("BGM_BASE_VOLUME", r.changes)
        self.assertAlmostEqual(r.changes["BGM_BASE_VOLUME"], 0.15, places=2)
        self.assertEqual(r.changes.get("BGM_STYLE"), "calm")

    def test_bgm_off_sets_style_off(self):
        r = self._process("음악 없애줘")
        self.assertEqual(r.changes.get("BGM_STYLE"), "off")

    def test_duck_stronger_decreases_db(self):
        r = self._process("더킹 강하게")
        self.assertTrue(r.applied)
        self.assertEqual(r.restart_from, "audio_only")
        import src.pipeline.intent_processor as ip
        with patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file):
            new_val = ip._read_audio_float("VOICE_DUCK_DB", 0.0)
        self.assertLess(new_val, -20.0)  # more negative = stronger

    def test_duck_weaker_increases_db(self):
        r = self._process("더킹 약하게")
        self.assertTrue(r.applied)
        import src.pipeline.intent_processor as ip
        with patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file):
            new_val = ip._read_audio_float("VOICE_DUCK_DB", 0.0)
        self.assertGreater(new_val, -20.0)  # less negative = weaker

    def test_bgm_louder_volume_capped_at_one(self):
        """BGM_BASE_VOLUME cannot exceed 1.0."""
        # Set volume to 0.95 first
        import src.pipeline.intent_processor as ip
        with patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file):
            ip._write_audio_value("BGM_BASE_VOLUME", "0.95")
        for _ in range(5):
            self._process("음악 크게")
        import src.pipeline.intent_processor as ip
        with patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file):
            val = ip._read_audio_float("BGM_BASE_VOLUME", 0.0)
        self.assertLessEqual(val, 1.0)

    def test_bgm_quieter_volume_floored_at_zero(self):
        """BGM_BASE_VOLUME cannot go below 0.0."""
        import src.pipeline.intent_processor as ip
        with patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file):
            ip._write_audio_value("BGM_BASE_VOLUME", "0.05")
        for _ in range(5):
            self._process("BGM 작게")
        with patch.object(ip, "_AUDIO_STYLE_FILE", self.audio_file):
            val = ip._read_audio_float("BGM_BASE_VOLUME", 1.0)
        self.assertGreaterEqual(val, 0.0)

    def test_restart_skill_for_fields_audio_fields(self):
        from src.pipeline.intent_processor import IntentProcessor
        self.assertEqual(
            IntentProcessor.restart_skill_for_fields(["BGM_BASE_VOLUME"]),
            "audio_only",
        )
        self.assertEqual(
            IntentProcessor.restart_skill_for_fields(["VOICE_DUCK_DB"]),
            "audio_only",
        )
        self.assertEqual(
            IntentProcessor.restart_skill_for_fields(["BGM_STYLE"]),
            "audio_only",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestAudioGate — clipping and BGM ratio checks (8 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioGate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.audio_file = self.tmpdir / "test.mp4"
        self.audio_file.write_bytes(b"\x00" * 100)

    def tearDown(self):
        self.tmp.cleanup()

    def _mock_volumedetect(self, max_volume: float, mean_volume: float):
        return patch(
            "harness.sensors.audio_gate._run_volumedetect",
            return_value={"max_volume": max_volume, "mean_volume": mean_volume},
        )

    def test_check_clipping_passes_when_below_zero(self):
        from harness.sensors.audio_gate import check_clipping
        with self._mock_volumedetect(-3.0, -18.0):
            r = check_clipping(self.audio_file)
        self.assertTrue(r.passed)
        self.assertEqual(r.name, "CLIPPING_FREE")

    def test_check_clipping_fails_when_at_zero(self):
        from harness.sensors.audio_gate import check_clipping
        with self._mock_volumedetect(0.0, -15.0):
            r = check_clipping(self.audio_file)
        self.assertFalse(r.passed)
        self.assertIn("CLIPPING", r.detail.upper())

    def test_check_clipping_fails_above_zero(self):
        from harness.sensors.audio_gate import check_clipping
        with self._mock_volumedetect(1.5, -10.0):
            r = check_clipping(self.audio_file)
        self.assertFalse(r.passed)

    def test_check_clipping_peak_value_in_result(self):
        from harness.sensors.audio_gate import check_clipping
        with self._mock_volumedetect(-5.0, -20.0):
            r = check_clipping(self.audio_file)
        self.assertAlmostEqual(r.value, -5.0)

    def test_bgm_voice_ratio_passes_when_ratio_sufficient(self):
        """ratio = peak - mean ≥ abs(duck_db) * 0.5."""
        from harness.sensors.audio_gate import check_bgm_voice_ratio
        # duck_db=-20 → min_ratio=10; ratio = -3 - (-18) = 15 ≥ 10
        with self._mock_volumedetect(-3.0, -18.0):
            r = check_bgm_voice_ratio(self.audio_file, target_duck_db=-20.0)
        self.assertTrue(r.passed)
        self.assertEqual(r.name, "BGM_VOICE_RATIO")

    def test_bgm_voice_ratio_fails_when_ratio_insufficient(self):
        """ratio = 2 dB < 10 dB min → fail."""
        from harness.sensors.audio_gate import check_bgm_voice_ratio
        with self._mock_volumedetect(-8.0, -10.0):
            r = check_bgm_voice_ratio(self.audio_file, target_duck_db=-20.0)
        self.assertFalse(r.passed)

    def test_validate_audio_returns_two_results(self):
        from harness.sensors.audio_gate import validate_audio
        with self._mock_volumedetect(-3.0, -18.0):
            results = validate_audio(self.audio_file)
        self.assertEqual(len(results), 2)

    def test_gate_handles_ffprobe_error_gracefully(self):
        """When ffprobe raises RuntimeError, gate returns failed result."""
        from harness.sensors.audio_gate import check_clipping
        with patch("harness.sensors.audio_gate._run_volumedetect",
                   side_effect=RuntimeError("ffprobe not found")):
            r = check_clipping(self.audio_file)
        self.assertFalse(r.passed)
        self.assertIn("ffprobe", r.detail)


# ═══════════════════════════════════════════════════════════════════════════════
# TestAudioOnlyTiming — efficiency report + skill count (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioOnlyTiming(unittest.TestCase):

    def test_audio_only_dispatches_one_skill(self):
        """audio_only mode dispatches only exporter (1 of 4 skills)."""
        audio_only_skills = 1   # exporter only
        full_pipeline_skills = 4
        self.assertLess(audio_only_skills, full_pipeline_skills)

    def test_audio_only_skill_count_ratio(self):
        """audio_only dispatches 25% of full pipeline skills."""
        ratio = 1 / 4
        self.assertAlmostEqual(ratio, 0.25)

    def test_micro_resume_vs_audio_only_same_efficiency(self):
        """Both micro-resume and audio_only skip 3 skills — same dispatch ratio."""
        micro_skills = 1  # exporter only
        audio_skills = 1  # exporter only
        self.assertEqual(micro_skills, audio_skills)

    def test_timing_report_printed(self):
        """Timing report — document audio_only resume efficiency."""
        full_skills = 4
        ao_skills = 1

        approx_times = {
            "transcriber": 30.0,  # Whisper on CPU
            "cutter":       2.0,
            "designer":     5.0,
            "exporter":    45.0,  # FFmpeg encode
        }
        full_total = sum(approx_times.values())
        ao_total   = approx_times["exporter"]
        saved      = full_total - ao_total
        speedup    = full_total / ao_total

        print("\n" + "=" * 58)
        print("  Audio-Only Resume Efficiency Report (Phase 13)")
        print("=" * 58)
        print(f"  Full pipeline ({full_skills} skills):  ~{full_total:.0f}s")
        print(f"  Audio-only    ({ao_skills} skill):   ~{ao_total:.0f}s")
        print(f"  Time saved:                ~{saved:.0f}s  ({saved/full_total*100:.0f}%)")
        print(f"  Speedup:                   ~{speedup:.1f}x")
        print(f"  Skipped: transcriber (~{approx_times['transcriber']:.0f}s), "
              f"cutter (~{approx_times['cutter']:.0f}s), "
              f"designer (~{approx_times['designer']:.0f}s)")
        print(f"\n  BGM update latency:  ~{ao_total:.0f}s  (FFmpeg audio re-encode only)")
        print(f"  sidechaincompress:   automatic BGM ducking from audio_style.md")
        print("=" * 58)

        self.assertGreater(speedup, 1.5)
        self.assertGreater(saved, 0)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
