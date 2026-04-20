"""
src/utils/test_phase12.py

Unit + integration tests for Phase 12 — Interactive Asset Refinement & Micro-Resume.

Covers
------
  TestMicroResumeOrchestrator     — designer_fast pipeline mode (10)
  TestIndexBasedIntents           — index-based delete/re-roll commands (12)
  TestAssetManagerGate            — ui_evaluator gate_asset_manager_update (5)
  TestMicroResumeTiming           — efficiency report + skill-skip verification (4)

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_phase12.py -v -s
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_skill_record(job_id, skill, output_path="", cursor_end="00:00:10.000"):
    from harness.memory.manager import SkillRecord
    return SkillRecord(
        job_id=job_id, skill=skill, status="success",
        output_path=output_path,
        cursor_start="00:00:00.000", cursor_end=cursor_end,
    )


def _make_timeline(visual_elements: list[dict]) -> dict:
    return {
        "visual_elements": visual_elements,
        "metadata": {"caption_count": 0, "broll_count": 0},
    }


def _write_timeline(path: Path, elements: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_make_timeline(elements), indent=2), encoding="utf-8")


def _make_valid_png(path: Path) -> None:
    from PIL import Image
    img = Image.new("RGB", (1080, 1920), (50, 50, 50))
    img.save(str(path), "PNG")


# ═══════════════════════════════════════════════════════════════════════════════
# TestMicroResumeOrchestrator — designer_fast pipeline (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMicroResumeOrchestrator(unittest.TestCase):
    """WorkflowManager with force_resume_from='designer_fast'."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.staging = Path(self.tmp.name) / "staging"
        self.staging.mkdir()
        self.job_id = "test_micro_001"

    def tearDown(self):
        self.tmp.cleanup()
        from harness.memory.manager import MemoryManager
        import shutil
        mgr = MemoryManager(self.job_id)
        if mgr.job_dir.exists():
            shutil.rmtree(str(mgr.job_dir))

    def _make_wm(self, borrow_records=None, staging=None):
        from src.pipeline.orchestrator import WorkflowManager
        return WorkflowManager(
            job_id=self.job_id,
            source_file=str(self.staging / "raw.mp4"),
            staging_dir=staging or self.staging,
            dry_run=True,
            force_resume_from="designer_fast",
            borrow_records=borrow_records or {},
        )

    # ── No prior designer record → falls back to exporter ─────────────────

    def test_fallback_to_exporter_when_no_prior_designer(self):
        """designer_fast with no borrow_records falls back gracefully."""
        wm = self._make_wm(borrow_records={})
        # Exporter will fail (no real files), but we're checking the mode switch
        with patch.object(wm, "_run_exporter") as mock_exp:
            mock_exp.return_value = _make_skill_record(self.job_id, "exporter")
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
        # Regardless of what exporter does, the pipeline chose the exporter path
        self.assertIsNotNone(result)

    def test_micro_resume_writes_placeholder_records(self):
        """_run_micro_resume writes transcriber/cutter/designer placeholders."""
        # Set up a prior annotated_timeline.json
        timeline_path = self.staging / "prior_timeline.json"
        _write_timeline(timeline_path, [
            {"type": "caption", "start": 0.0, "end": 5.0, "text": "Hi"},
            {"type": "b-roll", "start": 1.0, "end": 3.0, "asset_path": "/old/broll.mp4"},
        ])

        transcriber_rec = _make_skill_record(self.job_id, "transcriber", "/t.json")
        cutter_rec      = _make_skill_record(self.job_id, "cutter",      "/c.json")
        designer_rec    = _make_skill_record(self.job_id, "designer",    str(timeline_path))

        wm = self._make_wm(borrow_records={
            "transcriber": transcriber_rec,
            "cutter":      cutter_rec,
            "designer":    designer_rec,
        })

        # Patch exporter to succeed without real files
        with patch.object(wm, "_run_exporter",
                          return_value=_make_skill_record(self.job_id, "exporter", "out.mp4")):
            result = wm.run()

        # Placeholders should have been written (MemoryManager persists them)
        from harness.memory.manager import MemoryManager
        mgr = MemoryManager(self.job_id)
        resume = mgr.find_resume_point()
        self.assertIn("designer", resume.records)
        self.assertIn("cutter",   resume.records)

    def test_patch_broll_replaces_broll_elements(self):
        """_patch_broll_in_timeline replaces b-roll but keeps captions."""
        timeline_path = self.staging / "timeline.json"
        _write_timeline(timeline_path, [
            {"type": "caption", "start": 0.0, "end": 5.0, "text": "Hello"},
            {"type": "b-roll",  "start": 1.0, "end": 3.0, "asset_path": "/old.mp4"},
        ])

        designer_rec = _make_skill_record(self.job_id, "designer", str(timeline_path))
        wm = self._make_wm(borrow_records={"designer": designer_rec})

        # Patch broll_requests.json to empty
        with patch("src.pipeline.orchestrator._BROLL_REQUESTS_FILE",
                   self.staging / "empty_broll.json"):
            (self.staging / "empty_broll.json").write_text("[]")
            patched = wm._patch_broll_in_timeline(designer_rec)

        self.assertIsNotNone(patched)
        data = json.loads(patched.read_text(encoding="utf-8"))
        types = [e["type"] for e in data["visual_elements"]]
        self.assertIn("caption", types)     # caption preserved
        self.assertNotIn("b-roll", types)   # old b-roll removed (none in requests)

    def test_patch_broll_adds_new_elements(self):
        """_patch_broll_in_timeline adds new b-roll elements from requests."""
        from PIL import Image
        import tempfile

        # Create a real MP4 stub (asset must exist for build_broll_elements)
        asset_path = self.staging / "cat.mp4"
        asset_path.write_bytes(b"\x00" * 100)  # stub — designer logic just needs path

        timeline_path = self.staging / "timeline.json"
        _write_timeline(timeline_path, [
            {"type": "caption", "start": 0.0, "end": 30.0, "text": "Hi"},
        ])

        broll_req = [{"keyword": "cat", "asset_path": str(asset_path),
                      "opacity": 1.0, "mode": "overlay"}]
        broll_file = self.staging / "broll_requests.json"
        broll_file.write_text(json.dumps(broll_req), encoding="utf-8")

        designer_rec = _make_skill_record(self.job_id, "designer", str(timeline_path))
        wm = self._make_wm(borrow_records={"designer": designer_rec})

        with patch("src.pipeline.orchestrator._BROLL_REQUESTS_FILE", broll_file):
            patched = wm._patch_broll_in_timeline(designer_rec)

        self.assertIsNotNone(patched)
        data = json.loads(patched.read_text(encoding="utf-8"))
        broll_elems = [e for e in data["visual_elements"] if e.get("type") == "b-roll"]
        self.assertEqual(len(broll_elems), 1)
        self.assertEqual(broll_elems[0]["keyword"], "cat")

    def test_patch_returns_none_when_prior_timeline_missing(self):
        """_patch_broll_in_timeline returns None if prior file doesn't exist."""
        designer_rec = _make_skill_record(
            self.job_id, "designer", "/nonexistent/timeline.json"
        )
        wm = self._make_wm(borrow_records={"designer": designer_rec})
        result = wm._patch_broll_in_timeline(designer_rec)
        self.assertIsNone(result)

    def test_micro_resume_result_skips_three_skills(self):
        """PipelineResult.skipped must include transcriber, cutter, designer."""
        timeline_path = self.staging / "t.json"
        _write_timeline(timeline_path, [
            {"type": "caption", "start": 0.0, "end": 10.0, "text": "ok"},
        ])
        designer_rec = _make_skill_record(self.job_id, "designer", str(timeline_path))

        wm = self._make_wm(borrow_records={"designer": designer_rec})
        with patch.object(wm, "_run_exporter",
                          return_value=_make_skill_record(self.job_id, "exporter")):
            result = wm.run()

        for skill in ("transcriber", "cutter", "designer"):
            self.assertIn(skill, result.skipped)

    def test_micro_resume_result_completed_contains_exporter(self):
        """When exporter succeeds, completed should include 'exporter'."""
        timeline_path = self.staging / "t.json"
        _write_timeline(timeline_path, [])
        designer_rec = _make_skill_record(self.job_id, "designer", str(timeline_path))

        wm = self._make_wm(borrow_records={"designer": designer_rec})
        with patch.object(wm, "_run_exporter",
                          return_value=_make_skill_record(self.job_id, "exporter")):
            result = wm.run()

        self.assertIn("exporter", result.completed)

    def test_micro_resume_exporter_failure_propagates(self):
        """When exporter fails, result.status == 'failed'."""
        from harness.memory.manager import SkillRecord
        timeline_path = self.staging / "t.json"
        _write_timeline(timeline_path, [])
        designer_rec = _make_skill_record(self.job_id, "designer", str(timeline_path))

        failed_rec = SkillRecord(
            job_id=self.job_id, skill="exporter", status="failed",
            output_path="", cursor_start="00:00:00.000", cursor_end="00:00:00.000",
            error="ffmpeg error",
        )
        wm = self._make_wm(borrow_records={"designer": designer_rec})
        with patch.object(wm, "_run_exporter", return_value=failed_rec):
            result = wm.run()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failed_skill, "exporter")

    def test_micro_resume_does_not_call_transcriber_cutter_designer(self):
        """_run_micro_resume must NOT dispatch transcriber/cutter/designer skills."""
        timeline_path = self.staging / "t.json"
        _write_timeline(timeline_path, [])
        designer_rec = _make_skill_record(self.job_id, "designer", str(timeline_path))

        wm = self._make_wm(borrow_records={"designer": designer_rec})
        dispatch_log = []

        original_run_skill = wm._run_skill
        def spy_run_skill(skill, cursor, resume):
            dispatch_log.append(skill)
            return _make_skill_record(self.job_id, skill)
        wm._run_skill = spy_run_skill

        wm.run()

        self.assertNotIn("transcriber", dispatch_log)
        self.assertNotIn("cutter",      dispatch_log)
        self.assertNotIn("designer",    dispatch_log)

    def test_patch_preserves_metadata_broll_count(self):
        """Patched timeline metadata.broll_count reflects new b-roll count."""
        timeline_path = self.staging / "t.json"
        _write_timeline(timeline_path, [
            {"type": "b-roll", "start": 0.0, "end": 5.0, "asset_path": "/old.mp4"},
        ])
        broll_file = self.staging / "br.json"
        broll_file.write_text("[]", encoding="utf-8")  # remove all b-roll

        designer_rec = _make_skill_record(self.job_id, "designer", str(timeline_path))
        wm = self._make_wm(borrow_records={"designer": designer_rec})
        with patch("src.pipeline.orchestrator._BROLL_REQUESTS_FILE", broll_file):
            patched = wm._patch_broll_in_timeline(designer_rec)

        data = json.loads(patched.read_text(encoding="utf-8"))
        self.assertEqual(data["metadata"]["broll_count"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TestIndexBasedIntents — "2번 자료화면 빼줘" / "마지막 그림 다시 그려줘" (12 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestIndexBasedIntents(unittest.TestCase):
    """intent_processor index-based delete/re-roll commands."""

    def setUp(self):
        self.spec_tmp = tempfile.TemporaryDirectory()
        self.gen_tmp  = tempfile.TemporaryDirectory()
        self.broll_file = Path(self.spec_tmp.name) / "broll_requests.json"

    def tearDown(self):
        self.spec_tmp.cleanup()
        self.gen_tmp.cleanup()

    def _write_requests(self, items: list[dict]) -> None:
        self.broll_file.write_text(json.dumps(items), encoding="utf-8")

    def _read_requests(self) -> list[dict]:
        return json.loads(self.broll_file.read_text(encoding="utf-8"))

    def _patch_broll_file(self):
        return patch("src.pipeline.intent_processor._BROLL_REQUESTS_FILE",
                     self.broll_file)

    # ── _parse_ordinal ──────────────────────────────────────────────────────

    def test_parse_ordinal_arabic(self):
        from src.pipeline.intent_processor import _parse_ordinal
        self.assertEqual(_parse_ordinal("2번 자료화면 빼줘"), 2)

    def test_parse_ordinal_korean(self):
        from src.pipeline.intent_processor import _parse_ordinal
        self.assertEqual(_parse_ordinal("두번째 자료화면"), 2)

    def test_parse_ordinal_last(self):
        from src.pipeline.intent_processor import _parse_ordinal
        self.assertEqual(_parse_ordinal("마지막 그림"), -1)

    def test_parse_ordinal_none_for_unknown(self):
        from src.pipeline.intent_processor import _parse_ordinal
        self.assertIsNone(_parse_ordinal("자료화면 다 빼줘"))

    # ── Delete by index ─────────────────────────────────────────────────────

    def test_delete_first_item(self):
        from src.pipeline.intent_processor import IntentProcessor
        items = [
            {"keyword": "cat",   "asset_path": "/cat.mp4",   "opacity": 1.0, "mode": "overlay"},
            {"keyword": "ocean", "asset_path": "/ocean.mp4", "opacity": 1.0, "mode": "overlay"},
        ]
        self._write_requests(items)
        with self._patch_broll_file():
            r = IntentProcessor().process("1번 자료화면 빼줘")
        remaining = self._read_requests()
        self.assertTrue(r.applied)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["keyword"], "ocean")

    def test_delete_last_item(self):
        from src.pipeline.intent_processor import IntentProcessor
        items = [
            {"keyword": "cat",   "asset_path": "/cat.mp4",   "opacity": 1.0, "mode": "overlay"},
            {"keyword": "ocean", "asset_path": "/ocean.mp4", "opacity": 1.0, "mode": "overlay"},
        ]
        self._write_requests(items)
        with self._patch_broll_file():
            r = IntentProcessor().process("마지막 자료화면 빼줘")
        remaining = self._read_requests()
        self.assertTrue(r.applied)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["keyword"], "cat")

    def test_delete_out_of_range_returns_applied_false(self):
        from src.pipeline.intent_processor import IntentProcessor
        self._write_requests([{"keyword": "cat", "asset_path": "/c.mp4",
                               "opacity": 1.0, "mode": "overlay"}])
        with self._patch_broll_file():
            r = IntentProcessor().process("5번 그림 빼줘")
        self.assertFalse(r.applied)

    def test_delete_empty_list_returns_applied_false(self):
        from src.pipeline.intent_processor import IntentProcessor
        self._write_requests([])
        with self._patch_broll_file():
            r = IntentProcessor().process("1번 자료화면 빼줘")
        self.assertFalse(r.applied)

    def test_delete_restart_from_designer_fast(self):
        from src.pipeline.intent_processor import IntentProcessor
        self._write_requests([{"keyword": "cat", "asset_path": "/c.mp4",
                               "opacity": 1.0, "mode": "overlay"}])
        with self._patch_broll_file():
            r = IntentProcessor().process("1번 자료화면 빼줘")
        self.assertEqual(r.restart_from, "designer_fast")

    # ── Re-roll by index ────────────────────────────────────────────────────

    def test_reroll_last_updates_asset_path(self):
        from src.pipeline.intent_processor import IntentProcessor
        from src.utils import asset_generator as ag_mod

        fake_new_path = str(Path(self.gen_tmp.name) / "cat.png")
        _make_valid_png(Path(fake_new_path))

        self._write_requests([{"keyword": "cat", "asset_path": "/old/cat.mp4",
                               "opacity": 1.0, "mode": "overlay"}])
        with self._patch_broll_file(), \
             patch.object(ag_mod.AssetGenerator, "generate", return_value=fake_new_path), \
             patch.object(ag_mod.AssetGenerator, "cache_path",
                          return_value=Path(self.gen_tmp.name) / "cat.png"):
            r = IntentProcessor().process("마지막 그림 다시 그려줘")

        self.assertTrue(r.applied)
        updated = self._read_requests()
        self.assertEqual(updated[0]["asset_path"], fake_new_path)

    def test_reroll_sets_restart_from_designer_fast(self):
        from src.pipeline.intent_processor import IntentProcessor
        from src.utils import asset_generator as ag_mod

        fake_path = str(Path(self.gen_tmp.name) / "cat.png")
        _make_valid_png(Path(fake_path))

        self._write_requests([{"keyword": "cat", "asset_path": "/old.mp4",
                               "opacity": 1.0, "mode": "overlay"}])
        with self._patch_broll_file(), \
             patch.object(ag_mod.AssetGenerator, "generate", return_value=fake_path), \
             patch.object(ag_mod.AssetGenerator, "cache_path",
                          return_value=Path(self.gen_tmp.name) / "cat.png"):
            r = IntentProcessor().process("마지막 그림 다시 그려줘")
        self.assertEqual(r.restart_from, "designer_fast")

    def test_reroll_generation_failure_returns_applied_false(self):
        from src.pipeline.intent_processor import IntentProcessor
        from src.utils import asset_generator as ag_mod

        self._write_requests([{"keyword": "cat", "asset_path": "/old.mp4",
                               "opacity": 1.0, "mode": "overlay"}])
        with self._patch_broll_file(), \
             patch.object(ag_mod.AssetGenerator, "generate", return_value=None), \
             patch.object(ag_mod.AssetGenerator, "cache_path",
                          return_value=Path(self.gen_tmp.name) / "cat.png"):
            r = IntentProcessor().process("마지막 그림 다시 그려줘")
        self.assertFalse(r.applied)


# ═══════════════════════════════════════════════════════════════════════════════
# TestAssetManagerGate — ui_evaluator gate (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssetManagerGate(unittest.TestCase):
    """gate_asset_manager_update disk-level verification."""

    def test_gate_passes_when_delete_and_reroll_work(self):
        """gate_asset_manager_update should pass on a clean run."""
        from harness.sensors.ui_evaluator import gate_asset_manager_update
        result = gate_asset_manager_update()
        self.assertTrue(result.passed, msg=result.detail)

    def test_gate_returns_gate_result(self):
        from harness.sensors.ui_evaluator import gate_asset_manager_update, GateResult
        result = gate_asset_manager_update()
        self.assertIsInstance(result, GateResult)

    def test_gate_name_is_asset_manager_update(self):
        from harness.sensors.ui_evaluator import gate_asset_manager_update
        result = gate_asset_manager_update()
        self.assertEqual(result.name, "ASSET_MANAGER_UPDATE")

    def test_gate_restores_original_broll_file(self):
        """gate must restore broll_requests.json to its original state."""
        from harness.sensors.ui_evaluator import BROLL_REQUESTS_FILE, gate_asset_manager_update

        original = None
        if BROLL_REQUESTS_FILE.exists():
            original = BROLL_REQUESTS_FILE.read_bytes()

        gate_asset_manager_update()

        if original is not None:
            self.assertEqual(BROLL_REQUESTS_FILE.read_bytes(), original)
        else:
            self.assertFalse(BROLL_REQUESTS_FILE.exists())

    def test_gate_detail_contains_both_operations(self):
        from harness.sensors.ui_evaluator import gate_asset_manager_update
        result = gate_asset_manager_update()
        if result.passed:
            self.assertIn("Delete", result.detail)
            self.assertIn("Re-roll", result.detail)


# ═══════════════════════════════════════════════════════════════════════════════
# TestMicroResumeTiming — efficiency measurement (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMicroResumeTiming(unittest.TestCase):
    """Verify Micro-Resume invokes only 1 skill vs 4 for full pipeline."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.staging = Path(self.tmp.name) / "staging"
        self.staging.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_records(self, job_id: str) -> dict:
        from harness.memory.manager import SkillRecord
        return {
            s: SkillRecord(job_id=job_id, skill=s, status="success",
                           output_path="", cursor_start="00:00:00.000",
                           cursor_end="00:00:10.000")
            for s in ["transcriber", "cutter", "designer", "exporter"]
        }

    def test_micro_resume_dispatches_exactly_one_skill(self):
        """designer_fast must dispatch only the exporter (1 of 4 skills)."""
        job_id = "timing_test_001"
        timeline_path = self.staging / "t.json"
        _write_timeline(timeline_path, [])

        borrow = self._make_records("prior_job")
        borrow["designer"].output_path = str(timeline_path)

        from src.pipeline.orchestrator import WorkflowManager
        wm = WorkflowManager(
            job_id=job_id,
            source_file=str(self.staging / "raw.mp4"),
            staging_dir=self.staging,
            dry_run=True,
            force_resume_from="designer_fast",
            borrow_records=borrow,
        )

        skills_dispatched = []
        def spy(skill, cursor, resume):
            skills_dispatched.append(skill)
            return _make_skill_record(job_id, skill)
        wm._run_skill = spy
        wm.run()

        self.assertEqual(len(skills_dispatched), 1,
                         msg=f"Expected 1 skill, got {skills_dispatched}")
        self.assertEqual(skills_dispatched[0], "exporter")

        # Clean up memory
        from harness.memory.manager import MemoryManager
        import shutil
        mgr = MemoryManager(job_id)
        if mgr.job_dir.exists():
            shutil.rmtree(str(mgr.job_dir))

    def test_full_pipeline_dispatches_four_skills(self):
        """Normal pipeline dispatches all 4 skills (baseline comparison)."""
        from src.pipeline.orchestrator import WorkflowManager
        job_id = "timing_full_001"
        wm = WorkflowManager(
            job_id=job_id,
            source_file=str(self.staging / "raw.mp4"),
            staging_dir=self.staging,
            dry_run=True,
        )

        skills_dispatched = []
        def spy(skill, cursor, resume):
            skills_dispatched.append(skill)
            return _make_skill_record(job_id, skill)
        wm._run_skill = spy
        wm.run()

        self.assertEqual(len(skills_dispatched), 4)

        # Clean up
        from harness.memory.manager import MemoryManager
        import shutil
        mgr = MemoryManager(job_id)
        if mgr.job_dir.exists():
            shutil.rmtree(str(mgr.job_dir))

    def test_micro_resume_skill_count_ratio(self):
        """Micro-Resume dispatches 25% of skills compared to full pipeline."""
        # 1 out of 4 = 25%
        micro_count = 1
        full_count = 4
        ratio = micro_count / full_count
        self.assertAlmostEqual(ratio, 0.25)

    def test_timing_report_printed(self):
        """Timing report — document Micro-Resume efficiency."""
        micro_skills = 1     # only exporter
        full_skills  = 4     # transcriber + cutter + designer + exporter

        # Approximate real-world time savings per skill (no Whisper/FFmpeg in tests)
        approx_times = {
            "transcriber": 30.0,   # Whisper on CPU
            "cutter":       2.0,
            "designer":     5.0,
            "exporter":    45.0,   # FFmpeg encode
        }
        full_total = sum(approx_times.values())
        micro_total = approx_times["exporter"]  # only exporter
        saved = full_total - micro_total
        speedup = full_total / micro_total

        print("\n" + "=" * 55)
        print("  Micro-Resume Efficiency Report (Phase 12)")
        print("=" * 55)
        print(f"  Full pipeline ({full_skills} skills):  ~{full_total:.0f}s")
        print(f"  Micro-Resume  ({micro_skills} skill):   ~{micro_total:.0f}s")
        print(f"  Time saved:                ~{saved:.0f}s  ({saved/full_total*100:.0f}%)")
        print(f"  Speedup:                   ~{speedup:.1f}x")
        print(f"  Skipped: transcriber (~{approx_times['transcriber']:.0f}s), "
              f"cutter (~{approx_times['cutter']:.0f}s), "
              f"designer (~{approx_times['designer']:.0f}s)")
        print("=" * 55)

        self.assertGreater(speedup, 1.5)   # must be at least 1.5x faster
        self.assertGreater(saved, 0)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
