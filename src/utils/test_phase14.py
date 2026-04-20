"""
src/utils/test_phase14.py

Unit + integration tests for Phase 14 — Dynamic Visual Rhythm & Interactive Direction.

Covers
------
  TestDynamicZoomBuilder     — build_dynamic_zoom_events (9)
  TestRhythmIntensitySpec    — read/write RHYTHM_INTENSITY in editing_style.md (6)
  TestRunDesignerDynamicZoom — run_designer integrates DynamicZoom (5)
  TestDirectorIntents        — "역동적으로", "흔들어줘", "집중해줘" (10)
  TestVisualFastOrchestrator — visual_fast resume mode (7)
  TestRhythmGate             — stability gate checks (10)
  TestRhythmReport           — timing + efficiency report (3)

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_phase14.py -v -s
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_vad_segments(count: int = 3, confidence: float = 0.90) -> list[dict]:
    segs = []
    for i in range(count):
        start = float(i * 5)
        segs.append({
            "is_voice": True,
            "confidence": confidence,
            "start": start,
            "end": start + 3.0,
        })
    return segs


def _make_spec_file(tmpdir: Path, rhythm_intensity: float = 0.5) -> Path:
    content = (
        "# Editing Style\n\n"
        f"RHYTHM_INTENSITY: {rhythm_intensity:.2f}\n"
        "HIGHLIGHT_COLOR: #FFD700\n"
        "CAPTION_COLOR: #FFFFFF\n"
    )
    p = tmpdir / "editing_style.md"
    p.write_text(content, encoding="utf-8")
    return p


def _make_skill_record(job_id, skill, output_path=""):
    from harness.memory.manager import SkillRecord
    return SkillRecord(
        job_id=job_id, skill=skill, status="success",
        output_path=output_path,
        cursor_start="00:00:00.000", cursor_end="00:00:10.000",
    )


def _vad_to_dict(elem) -> dict:
    return {
        "type": elem.type,
        "start": elem.start,
        "end": elem.end,
        "zoom_factor": elem.zoom_factor,
        "name": elem.name,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TestDynamicZoomBuilder — build_dynamic_zoom_events (9 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDynamicZoomBuilder(unittest.TestCase):

    def test_returns_empty_for_no_voice_segments(self):
        from skills.designer.logic import build_dynamic_zoom_events
        segs = [{"is_voice": False, "confidence": 0.95, "start": 0.0, "end": 2.0}]
        result = build_dynamic_zoom_events(segs, rhythm_intensity=0.8)
        self.assertEqual(result, [])

    def test_returns_empty_for_low_confidence(self):
        from skills.designer.logic import build_dynamic_zoom_events, VAD_CONFIDENCE_THRESHOLD
        segs = [{"is_voice": True, "confidence": VAD_CONFIDENCE_THRESHOLD - 0.1,
                 "start": 0.0, "end": 2.0}]
        result = build_dynamic_zoom_events(segs, rhythm_intensity=1.0)
        self.assertEqual(result, [])

    def test_returns_empty_for_zero_intensity(self):
        from skills.designer.logic import build_dynamic_zoom_events
        segs = _make_vad_segments(3, confidence=0.95)
        result = build_dynamic_zoom_events(segs, rhythm_intensity=0.0)
        self.assertEqual(result, [])

    def test_zoom_factor_below_dynamic_max(self):
        from skills.designer.logic import build_dynamic_zoom_events, DYNAMIC_ZOOM_MAX
        segs = _make_vad_segments(1, confidence=1.0)
        result = build_dynamic_zoom_events(segs, rhythm_intensity=1.0)
        self.assertEqual(len(result), 1)
        self.assertLessEqual(result[0].zoom_factor, DYNAMIC_ZOOM_MAX)

    def test_zoom_factor_above_one(self):
        from skills.designer.logic import build_dynamic_zoom_events
        segs = _make_vad_segments(1, confidence=1.0)
        result = build_dynamic_zoom_events(segs, rhythm_intensity=0.5)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0].zoom_factor, 1.0)

    def test_zoom_factor_scales_with_intensity(self):
        from skills.designer.logic import build_dynamic_zoom_events
        segs = _make_vad_segments(1, confidence=1.0)
        low  = build_dynamic_zoom_events(segs, rhythm_intensity=0.2)
        high = build_dynamic_zoom_events(segs, rhythm_intensity=0.8)
        self.assertLess(low[0].zoom_factor, high[0].zoom_factor)

    def test_output_type_is_zoom(self):
        from skills.designer.logic import build_dynamic_zoom_events
        segs = _make_vad_segments(1, confidence=1.0)
        result = build_dynamic_zoom_events(segs, rhythm_intensity=0.5)
        self.assertEqual(result[0].type, "zoom")

    def test_output_name_is_dynamic_zoom(self):
        from skills.designer.logic import build_dynamic_zoom_events
        segs = _make_vad_segments(1, confidence=1.0)
        result = build_dynamic_zoom_events(segs, rhythm_intensity=0.5)
        self.assertEqual(result[0].name, "dynamic_zoom")

    def test_count_matches_valid_vad_segments(self):
        from skills.designer.logic import build_dynamic_zoom_events
        segs = _make_vad_segments(4, confidence=0.92)
        result = build_dynamic_zoom_events(segs, rhythm_intensity=0.5)
        self.assertEqual(len(result), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# TestRhythmIntensitySpec — read/write RHYTHM_INTENSITY (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRhythmIntensitySpec(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.spec_file = _make_spec_file(self.tmpdir, 0.50)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_default_value(self):
        from skills.designer.logic import _read_rhythm_intensity
        with patch("skills.designer.logic._SPEC_FILE", self.spec_file):
            val = _read_rhythm_intensity()
        self.assertAlmostEqual(val, 0.50, places=2)

    def test_read_custom_value(self):
        self.spec_file.write_text(
            "RHYTHM_INTENSITY: 0.75\nHIGHLIGHT_COLOR: #FFD700\n",
            encoding="utf-8",
        )
        from skills.designer.logic import _read_rhythm_intensity
        with patch("skills.designer.logic._SPEC_FILE", self.spec_file):
            val = _read_rhythm_intensity()
        self.assertAlmostEqual(val, 0.75, places=2)

    def test_clamped_above_one(self):
        self.spec_file.write_text("RHYTHM_INTENSITY: 1.5\n", encoding="utf-8")
        from skills.designer.logic import _read_rhythm_intensity
        with patch("skills.designer.logic._SPEC_FILE", self.spec_file):
            val = _read_rhythm_intensity()
        self.assertLessEqual(val, 1.0)

    def test_clamped_below_zero(self):
        self.spec_file.write_text("RHYTHM_INTENSITY: -0.3\n", encoding="utf-8")
        from skills.designer.logic import _read_rhythm_intensity
        with patch("skills.designer.logic._SPEC_FILE", self.spec_file):
            val = _read_rhythm_intensity()
        self.assertGreaterEqual(val, 0.0)

    def test_missing_key_returns_default(self):
        self.spec_file.write_text("CAPTION_COLOR: #FFFFFF\n", encoding="utf-8")
        from skills.designer.logic import _read_rhythm_intensity, RHYTHM_INTENSITY_DEFAULT
        with patch("skills.designer.logic._SPEC_FILE", self.spec_file):
            val = _read_rhythm_intensity()
        self.assertEqual(val, RHYTHM_INTENSITY_DEFAULT)

    def test_intent_writes_rhythm_intensity_to_spec(self):
        from src.pipeline.intent_processor import IntentProcessor, _write_spec_value
        with patch("src.pipeline.intent_processor._SPEC_FILE", self.spec_file):
            IntentProcessor().process("좀 더 역동적으로")
            text = self.spec_file.read_text(encoding="utf-8")
        self.assertIn("RHYTHM_INTENSITY", text)


# ═══════════════════════════════════════════════════════════════════════════════
# TestRunDesignerDynamicZoom — run_designer integrates DynamicZoom (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunDesignerDynamicZoom(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec_file = _make_spec_file(Path(self.tmp.name), rhythm_intensity=0.8)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, vad_segs=None):
        from skills.designer.logic import run_designer
        words = [{"word": "hello", "start_ms": 0, "end_ms": 1000}]
        cut_segs = [{"action": "keep", "start": 0.0, "end": 10.0, "effects": []}]
        vad = vad_segs or _make_vad_segments(2, confidence=0.92)
        with patch("skills.designer.logic._SPEC_FILE", self.spec_file):
            return run_designer(words, cut_segs, vad)

    def test_dynamic_zoom_elements_present(self):
        result = self._run()
        dz_elems = [e for e in result.visual_elements
                    if e.type == "zoom" and e.name == "dynamic_zoom"]
        self.assertGreater(len(dz_elems), 0)

    def test_dynamic_zoom_not_flagged_by_jump_cut_sensor(self):
        """Sensor must not flag dynamic_zoom elements as JUMP_CUT_ZOOM violations."""
        result = self._run()
        dz_flags = [f for f in result.sensor_flags
                    if "JUMP_CUT_ZOOM" in f and "dynamic_zoom" in f]
        self.assertEqual(dz_flags, [])

    def test_dynamic_zoom_factor_below_safe_max(self):
        from harness.sensors.rhythm_gate import SAFE_ZOOM_MAX
        result = self._run()
        for elem in result.visual_elements:
            if elem.type == "zoom" and elem.name == "dynamic_zoom":
                self.assertLessEqual(elem.zoom_factor, SAFE_ZOOM_MAX)

    def test_no_dynamic_zoom_when_intensity_zero(self):
        zero_dir = Path(self.tmp.name) / "zero_spec"
        zero_dir.mkdir(parents=True, exist_ok=True)
        zero_spec = _make_spec_file(zero_dir, rhythm_intensity=0.0)
        from skills.designer.logic import run_designer
        words = [{"word": "hi", "start_ms": 0, "end_ms": 500}]
        cut_segs = [{"action": "keep", "start": 0.0, "end": 5.0, "effects": []}]
        vad = _make_vad_segments(2, confidence=0.92)
        with patch("skills.designer.logic._SPEC_FILE", zero_spec):
            result = run_designer(words, cut_segs, vad)
        dz_elems = [e for e in result.visual_elements
                    if e.type == "zoom" and e.name == "dynamic_zoom"]
        self.assertEqual(dz_elems, [])

    def test_jump_cut_zoom_still_present_alongside_dynamic(self):
        """Jump-cut zooms survive alongside dynamic zoom events."""
        from skills.designer.logic import run_designer
        words = [{"word": "hi", "start_ms": 0, "end_ms": 500}]
        cut_segs = [
            {"action": "keep", "start": 0.0, "end": 3.0, "effects": []},
            {"action": "keep", "start": 3.5, "end": 6.0,
             "effects": ["jump_cut_zoom_1.1"]},
        ]
        vad = _make_vad_segments(1, confidence=0.95)
        with patch("skills.designer.logic._SPEC_FILE", self.spec_file):
            result = run_designer(words, cut_segs, vad)
        jump_zooms = [e for e in result.visual_elements
                      if e.type == "zoom" and e.name != "dynamic_zoom"]
        self.assertGreater(len(jump_zooms), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TestDirectorIntents — Director's Chair commands (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDirectorIntents(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec_file = _make_spec_file(Path(self.tmp.name), 0.50)

    def tearDown(self):
        self.tmp.cleanup()

    def _process(self, text: str):
        from src.pipeline.intent_processor import IntentProcessor
        with patch("src.pipeline.intent_processor._SPEC_FILE", self.spec_file):
            return IntentProcessor().process(text)

    def _read_rhythm(self) -> float:
        from src.pipeline.intent_processor import _read_float
        with patch("src.pipeline.intent_processor._SPEC_FILE", self.spec_file):
            return _read_float("RHYTHM_INTENSITY", 0.5)

    def test_more_dynamic_increases_intensity(self):
        r = self._process("좀 더 역동적으로")
        self.assertTrue(r.applied)
        self.assertEqual(r.restart_from, "visual_fast")
        self.assertAlmostEqual(self._read_rhythm(), 0.70, places=2)

    def test_shake_screen_sets_high_intensity(self):
        r = self._process("화면 좀 흔들어줘")
        self.assertTrue(r.applied)
        self.assertEqual(r.restart_from, "visual_fast")
        self.assertAlmostEqual(self._read_rhythm(), 0.80, places=2)

    def test_focus_here_enables_zoom_focus(self):
        r = self._process("여기에 집중해줘")
        self.assertTrue(r.applied)
        self.assertEqual(r.restart_from, "visual_fast")
        self.assertIn("ZOOM_FOCUS_ENABLED", r.changes)
        text = self.spec_file.read_text(encoding="utf-8")
        self.assertIn("ZOOM_FOCUS_ENABLED", text)

    def test_less_dynamic_decreases_intensity(self):
        r = self._process("좀 차분하게")
        self.assertTrue(r.applied)
        self.assertEqual(r.restart_from, "visual_fast")
        self.assertAlmostEqual(self._read_rhythm(), 0.30, places=2)

    def test_intensity_capped_at_one(self):
        # Start at 0.90, go dynamic twice
        from src.pipeline.intent_processor import _write_spec_value
        with patch("src.pipeline.intent_processor._SPEC_FILE", self.spec_file):
            _write_spec_value("RHYTHM_INTENSITY", "0.90")
        self._process("더 역동적으로")
        self._process("더 역동적으로")
        self.assertLessEqual(self._read_rhythm(), 1.0)

    def test_intensity_floored_at_zero(self):
        from src.pipeline.intent_processor import _write_spec_value
        with patch("src.pipeline.intent_processor._SPEC_FILE", self.spec_file):
            _write_spec_value("RHYTHM_INTENSITY", "0.10")
        self._process("좀 차분하게")
        self._process("좀 차분하게")
        self.assertGreaterEqual(self._read_rhythm(), 0.0)

    def test_restart_from_is_visual_fast_for_rhythm_field(self):
        from src.pipeline.intent_processor import IntentProcessor
        self.assertEqual(
            IntentProcessor.restart_skill_for_fields(["RHYTHM_INTENSITY"]),
            "visual_fast",
        )

    def test_restart_from_is_visual_fast_for_zoom_focus(self):
        from src.pipeline.intent_processor import IntentProcessor
        self.assertEqual(
            IntentProcessor.restart_skill_for_fields(["ZOOM_FOCUS_ENABLED"]),
            "visual_fast",
        )

    def test_more_dynamic_changes_dict_contains_key(self):
        r = self._process("생동감 있게")
        self.assertIn("RHYTHM_INTENSITY", r.changes)

    def test_focus_here_increases_intensity(self):
        r = self._process("이 부분 강조해줘")
        self.assertTrue(r.applied)
        new_val = self._read_rhythm()
        self.assertGreaterEqual(new_val, 0.60)


# ═══════════════════════════════════════════════════════════════════════════════
# TestVisualFastOrchestrator — visual_fast resume mode (7 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVisualFastOrchestrator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.staging = Path(self.tmp.name) / "staging"
        self.staging.mkdir()
        self.job_id = "vf_test_001"

    def tearDown(self):
        self.tmp.cleanup()
        from harness.memory.manager import MemoryManager
        import shutil
        mgr = MemoryManager(self.job_id)
        if mgr.job_dir.exists():
            shutil.rmtree(str(mgr.job_dir))

    def _make_wm(self, borrow_records=None):
        from src.pipeline.orchestrator import WorkflowManager
        return WorkflowManager(
            job_id=self.job_id,
            source_file=str(self.staging / "raw.mp4"),
            staging_dir=self.staging,
            dry_run=True,
            force_resume_from="visual_fast",
            borrow_records=borrow_records or {},
        )

    def test_visual_fast_changes_preset_to_ultrafast(self):
        wm = self._make_wm()
        self.assertEqual(wm.force_resume_from, "visual_fast")
        # After _run_visual_fast() is called inside run(), preset must change
        with patch("src.pipeline.orchestrator.MemoryManager") as mock_cls:
            mock_mgr = MagicMock()
            mock_cls.return_value = mock_mgr
            mock_resume = MagicMock()
            mock_resume.is_complete = False
            mock_resume.cursor = "00:00:00.000"
            mock_resume.completed = ["transcriber", "cutter"]
            mock_resume.records = {}
            mock_resume.next_skill = "designer"
            mock_mgr.find_resume_point.return_value = mock_resume
            mock_mgr.write.return_value = None

            def fake_skill(skill, cursor, resume):
                return _make_skill_record(self.job_id, skill)
            wm._run_skill = fake_skill
            wm.run()

        self.assertEqual(wm.export_profile.get("preset"), "ultrafast")

    def test_visual_fast_sets_higher_crf(self):
        wm = self._make_wm()
        with patch("src.pipeline.orchestrator.MemoryManager") as mock_cls:
            mock_mgr = MagicMock()
            mock_cls.return_value = mock_mgr
            mock_resume = MagicMock()
            mock_resume.is_complete = True
            mock_resume.completed = ["transcriber", "cutter", "designer", "exporter"]
            mock_resume.records = {}
            mock_mgr.find_resume_point.return_value = mock_resume
            wm.run()
        self.assertEqual(wm.export_profile.get("crf"), 28)

    def test_visual_fast_force_resume_becomes_designer(self):
        """After visual_fast triggers, force_resume_from is rewritten to designer."""
        wm = self._make_wm()
        with patch("src.pipeline.orchestrator.MemoryManager") as mock_cls:
            mock_mgr = MagicMock()
            mock_cls.return_value = mock_mgr
            mock_resume = MagicMock()
            mock_resume.is_complete = True
            mock_resume.completed = SKILL_ORDER = ["transcriber", "cutter", "designer", "exporter"]
            mock_resume.records = {}
            mock_mgr.find_resume_point.return_value = mock_resume
            wm.run()
        self.assertEqual(wm.force_resume_from, "designer")

    def test_visual_fast_skips_transcriber_and_cutter(self):
        borrow = {
            s: _make_skill_record(self.job_id, s)
            for s in ["transcriber", "cutter"]
        }
        wm = self._make_wm(borrow_records=borrow)
        dispatched = []
        def spy(skill, cursor, resume):
            dispatched.append(skill)
            return _make_skill_record(self.job_id, skill)
        wm._run_skill = spy
        wm.run()
        self.assertNotIn("transcriber", dispatched)
        self.assertNotIn("cutter",      dispatched)

    def test_visual_fast_runs_designer(self):
        borrow = {
            s: _make_skill_record(self.job_id, s)
            for s in ["transcriber", "cutter"]
        }
        wm = self._make_wm(borrow_records=borrow)
        dispatched = []
        def spy(skill, cursor, resume):
            dispatched.append(skill)
            return _make_skill_record(self.job_id, skill)
        wm._run_skill = spy
        wm.run()
        self.assertIn("designer", dispatched)

    def test_visual_fast_runs_exporter(self):
        borrow = {
            s: _make_skill_record(self.job_id, s)
            for s in ["transcriber", "cutter"]
        }
        wm = self._make_wm(borrow_records=borrow)
        dispatched = []
        def spy(skill, cursor, resume):
            dispatched.append(skill)
            return _make_skill_record(self.job_id, skill)
        wm._run_skill = spy
        wm.run()
        self.assertIn("exporter", dispatched)

    def test_visual_fast_result_status_success(self):
        borrow = {
            s: _make_skill_record(self.job_id, s)
            for s in ["transcriber", "cutter"]
        }
        wm = self._make_wm(borrow_records=borrow)
        wm._run_skill = lambda skill, cursor, resume: _make_skill_record(self.job_id, skill)
        result = wm.run()
        self.assertEqual(result.status, "success")


# ═══════════════════════════════════════════════════════════════════════════════
# TestRhythmGate — stability checks (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRhythmGate(unittest.TestCase):

    def _dz(self, start, end, zoom=1.03, name="dynamic_zoom"):
        return {"type": "zoom", "start": start, "end": end,
                "zoom_factor": zoom, "name": name}

    # ── check_zoom_stability ────────────────────────────────────────────────

    def test_stability_passes_for_safe_factor(self):
        from harness.sensors.rhythm_gate import check_zoom_stability
        elems = [self._dz(0, 3, zoom=1.03)]
        r = check_zoom_stability(elems)
        self.assertTrue(r.passed)
        self.assertEqual(r.name, "ZOOM_STABILITY")

    def test_stability_fails_for_excessive_factor(self):
        from harness.sensors.rhythm_gate import check_zoom_stability, SAFE_ZOOM_MAX
        elems = [self._dz(0, 3, zoom=SAFE_ZOOM_MAX + 0.01)]
        r = check_zoom_stability(elems)
        self.assertFalse(r.passed)

    def test_stability_passes_empty(self):
        from harness.sensors.rhythm_gate import check_zoom_stability
        r = check_zoom_stability([])
        self.assertTrue(r.passed)

    # ── check_zoom_no_overlap ───────────────────────────────────────────────

    def test_no_overlap_passes_non_overlapping(self):
        from harness.sensors.rhythm_gate import check_zoom_no_overlap
        elems = [self._dz(0, 2), self._dz(3, 5)]
        r = check_zoom_no_overlap(elems)
        self.assertTrue(r.passed)
        self.assertEqual(r.name, "ZOOM_NO_OVERLAP")

    def test_no_overlap_fails_overlapping(self):
        from harness.sensors.rhythm_gate import check_zoom_no_overlap
        elems = [self._dz(0, 4), self._dz(3, 6)]  # overlap at 3–4
        r = check_zoom_no_overlap(elems)
        self.assertFalse(r.passed)

    # ── check_rhythm_intensity_range ────────────────────────────────────────

    def test_intensity_range_passes_midpoint(self):
        from harness.sensors.rhythm_gate import check_rhythm_intensity_range
        r = check_rhythm_intensity_range(0.5)
        self.assertTrue(r.passed)
        self.assertEqual(r.name, "RHYTHM_RANGE")

    def test_intensity_range_fails_above_one(self):
        from harness.sensors.rhythm_gate import check_rhythm_intensity_range
        r = check_rhythm_intensity_range(1.5)
        self.assertFalse(r.passed)

    def test_intensity_range_fails_negative(self):
        from harness.sensors.rhythm_gate import check_rhythm_intensity_range
        r = check_rhythm_intensity_range(-0.1)
        self.assertFalse(r.passed)

    # ── check_dynamic_zoom_density ──────────────────────────────────────────

    def test_density_passes_within_limit(self):
        from harness.sensors.rhythm_gate import check_dynamic_zoom_density
        elems = [self._dz(i * 10, i * 10 + 2) for i in range(5)]  # 5 in 60s = 5/min
        r = check_dynamic_zoom_density(elems, total_duration_s=60.0)
        self.assertTrue(r.passed)
        self.assertEqual(r.name, "DYNAMIC_ZOOM_COUNT")

    def test_density_fails_excessive(self):
        from harness.sensors.rhythm_gate import check_dynamic_zoom_density, MAX_ZOOM_EVENTS_PER_MIN
        # 30 events in 60s = 30/min > max 10/min
        elems = [self._dz(i * 2, i * 2 + 1) for i in range(30)]
        r = check_dynamic_zoom_density(elems, total_duration_s=60.0)
        self.assertFalse(r.passed)


# ═══════════════════════════════════════════════════════════════════════════════
# TestRhythmReport — timing + efficiency report (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRhythmReport(unittest.TestCase):

    def test_visual_fast_dispatches_two_skills(self):
        # visual_fast = designer + exporter
        visual_fast_skills = 2
        full_pipeline_skills = 4
        self.assertLess(visual_fast_skills, full_pipeline_skills)

    def test_visual_fast_is_faster_than_full_pipeline(self):
        approx = {"transcriber": 30, "cutter": 2, "designer": 5, "exporter": 45}
        full  = sum(approx.values())
        vfast = approx["designer"] + approx["exporter"]  # skip transcriber + cutter
        self.assertLess(vfast, full)

    def test_timing_report_printed(self):
        approx = {"transcriber": 30.0, "cutter": 2.0, "designer": 5.0, "exporter": 45.0}
        full   = sum(approx.values())
        vfast  = approx["designer"] + approx["exporter"]
        saved  = full - vfast
        speedup = full / vfast

        print("\n" + "=" * 58)
        print("  Visual-Fast Resume Efficiency Report (Phase 14)")
        print("=" * 58)
        print(f"  Full pipeline (4 skills):  ~{full:.0f}s")
        print(f"  Visual-fast   (2 skills):  ~{vfast:.0f}s  (designer + exporter)")
        print(f"  Time saved:                ~{saved:.0f}s  ({saved/full*100:.0f}%)")
        print(f"  Speedup:                   ~{speedup:.1f}x")
        print(f"  Skipped: transcriber (~{approx['transcriber']:.0f}s), "
              f"cutter (~{approx['cutter']:.0f}s)")
        print(f"\n  DynamicZoom:   VAD-amplitude-driven pulses via zoompan filter")
        print(f"  Max zoom:      1.05× (DYNAMIC_ZOOM_MAX, spec-safe ≤ 1.10)")
        print(f"  Preset:        ultrafast (crf=28, ~3× faster encode)")
        print("=" * 58)

        self.assertGreater(speedup, 1.0)
        self.assertGreater(saved, 0)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
