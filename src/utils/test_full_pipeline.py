"""
src/utils/test_full_pipeline.py

Integration and E2E tests for the full VoxEdit AI pipeline.

Covers
------
  TestExportValidator         — aspect ratio and audio clipping quality gates
  TestBuildFilterComplex      — FFmpeg filtergraph builder correctness
  TestExportCommandStructure  — to_command() output validation
  TestExportPlanCompute       — time remapping compute_output_time()
  TestExporterMain            — harness integration (SkillRecord)
  TestWorkflowManagerResume   — orchestrator resume / skip behaviour
  TestWorkflowManagerDispatch — per-skill dispatch with overrides
  TestFullPipelineE2E         — end-to-end using mock skills (dry_run=True)

All tests are mock-based — no FFmpeg, no Whisper, no audio files required.

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_full_pipeline.py -v
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

from harness.sensors.export_validator import (
    GateReport,
    GateStatus,
    check_aspect_ratio,
    check_audio_clipping,
    check_output_exists,
    validate,
)
from skills.exporter.logic import (
    DEFAULT_PROFILE,
    ExportPlan,
    build_export_plan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_segments(*pairs):
    """Return keep-segment dicts from (start, end) pairs."""
    return [{"start": s, "end": e, "action": "keep"} for s, e in pairs]


def _make_cut_segments(*pairs):
    """Return cut_list dicts (action=keep) from (start, end) pairs."""
    return [{"start": s, "end": e, "action": "keep"} for s, e in pairs]


def _make_plan(
    keep_pairs=None,
    captions=None,
    zooms=None,
    ducks=None,
    source="source.mp4",
    output="staging/output.mp4",
):
    keep_segs = _make_segments(*(keep_pairs or [(0.0, 5.0)]))
    return ExportPlan(
        source_file=source,
        output_file=output,
        keep_segments=keep_segs,
        captions=captions or [],
        zoom_elements=zooms or [],
        duck_events=ducks or [],
        export_profile=dict(DEFAULT_PROFILE),
    )


def _make_success_record(job_id, skill, output_path):
    from harness.memory.manager import SkillRecord
    return SkillRecord(
        job_id=job_id,
        skill=skill,
        status="success",
        output_path=output_path,
        cursor_start="00:00:00.000",
        cursor_end="00:00:05.000",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestExportValidator — quality gates
# ═══════════════════════════════════════════════════════════════════════════════

class TestAspectRatioGate(unittest.TestCase):
    """check_aspect_ratio — 14 tests"""

    def test_portrait_1080x1920_passes(self):
        r = check_aspect_ratio(1080, 1920)
        self.assertEqual(r.status, GateStatus.PASS)

    def test_portrait_720x1280_passes(self):
        r = check_aspect_ratio(720, 1280)
        self.assertEqual(r.status, GateStatus.PASS)

    def test_portrait_540x960_passes(self):
        r = check_aspect_ratio(540, 960)
        self.assertEqual(r.status, GateStatus.PASS)

    def test_landscape_1920x1080_fails(self):
        r = check_aspect_ratio(1920, 1080)
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_square_fails(self):
        r = check_aspect_ratio(1080, 1080)
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_zero_width_fails(self):
        r = check_aspect_ratio(0, 1920)
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_zero_height_fails(self):
        r = check_aspect_ratio(1080, 0)
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_custom_ratio_passes(self):
        # 16:9 landscape
        r = check_aspect_ratio(1920, 1080, expected_ratio="16:9")
        self.assertEqual(r.status, GateStatus.PASS)

    def test_custom_ratio_fail(self):
        r = check_aspect_ratio(1080, 1920, expected_ratio="16:9")
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_measured_ratio_in_result(self):
        r = check_aspect_ratio(1080, 1920)
        self.assertIsNotNone(r.measured)
        self.assertAlmostEqual(r.measured, 1080 / 1920, places=4)

    def test_fail_includes_detail(self):
        r = check_aspect_ratio(1920, 1080)
        self.assertIn("1920×1080", r.detail)

    def test_fail_includes_action(self):
        r = check_aspect_ratio(1920, 1080)
        self.assertIn("scale", r.fail_action)

    def test_tolerance_edge_pass(self):
        # Within 1% tolerance
        r = check_aspect_ratio(1082, 1920, tolerance=0.01)
        self.assertEqual(r.status, GateStatus.PASS)

    def test_tolerance_edge_fail(self):
        # 10% off should fail with default tolerance
        r = check_aspect_ratio(1200, 1920)
        self.assertEqual(r.status, GateStatus.FAIL)


class TestAudioClippingGate(unittest.TestCase):
    """check_audio_clipping — 10 tests"""

    def test_clean_samples_pass(self):
        samples = [0.1, -0.5, 0.8, -0.3]
        r = check_audio_clipping(samples)
        self.assertEqual(r.status, GateStatus.PASS)

    def test_clipped_sample_fails(self):
        samples = [0.1, 0.999, -0.5]
        r = check_audio_clipping(samples)
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_full_scale_fails(self):
        samples = [1.0, -1.0]
        r = check_audio_clipping(samples)
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_empty_samples_skipped(self):
        r = check_audio_clipping([])
        self.assertEqual(r.status, GateStatus.SKIP)

    def test_near_clip_passes(self):
        samples = [0.998, -0.998]
        r = check_audio_clipping(samples)
        self.assertEqual(r.status, GateStatus.PASS)

    def test_custom_threshold(self):
        samples = [0.9]
        r = check_audio_clipping(samples, threshold=0.85)
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_fail_reports_peak(self):
        samples = [0.999, 0.5]
        r = check_audio_clipping(samples)
        self.assertIsNotNone(r.measured)
        self.assertAlmostEqual(r.measured, 0.999, places=4)

    def test_fail_includes_action(self):
        r = check_audio_clipping([1.0])
        self.assertIn("alimiter", r.fail_action)

    def test_negative_clip_detected(self):
        samples = [-0.999, 0.0]
        r = check_audio_clipping(samples)
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_pass_measured_equals_peak(self):
        samples = [0.5, -0.7, 0.3]
        r = check_audio_clipping(samples)
        self.assertAlmostEqual(r.measured, 0.7, places=4)


class TestOutputExistsGate(unittest.TestCase):
    """check_output_exists — 6 tests"""

    def test_existing_file_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"x" * 2048)
            path = f.name
        r = check_output_exists(path)
        self.assertEqual(r.status, GateStatus.PASS)
        Path(path).unlink()

    def test_missing_file_fails(self):
        r = check_output_exists("/nonexistent/path/output.mp4")
        self.assertEqual(r.status, GateStatus.FAIL)

    def test_empty_file_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        r = check_output_exists(path)
        self.assertEqual(r.status, GateStatus.FAIL)
        Path(path).unlink()

    def test_small_file_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"x" * 100)
            path = f.name
        r = check_output_exists(path)
        self.assertEqual(r.status, GateStatus.FAIL)
        Path(path).unlink()

    def test_fail_detail_mentions_path(self):
        r = check_output_exists("/missing/video.mp4")
        self.assertIn("/missing/video.mp4", r.detail)

    def test_pass_measured_is_size(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"x" * 2048)
            path = f.name
        r = check_output_exists(path)
        self.assertGreaterEqual(r.measured, 2048.0)
        Path(path).unlink()


class TestValidateEntryPoint(unittest.TestCase):
    """validate() public API — 5 tests"""

    def test_metadata_only_passes_for_shorts(self):
        report = validate(width=1080, height=1920)
        self.assertTrue(report.passed)

    def test_metadata_only_fails_bad_ratio(self):
        report = validate(width=1920, height=1080)
        self.assertFalse(report.passed)

    def test_no_dimensions_skips_aspect_ratio(self):
        report = validate(width=0, height=0)
        from harness.sensors.export_validator import GateStatus
        aspect = next(g for g in report.gates if g.gate == "ASPECT_RATIO")
        self.assertEqual(aspect.status, GateStatus.SKIP)

    def test_no_output_path_skips_output_exists(self):
        report = validate(width=1080, height=1920, output_path=None)
        exists = next(g for g in report.gates if g.gate == "OUTPUT_EXISTS")
        self.assertEqual(exists.status, GateStatus.SKIP)

    def test_summary_passes(self):
        report = validate(width=1080, height=1920)
        self.assertIn("EXPORT OK", report.summary)


# ═══════════════════════════════════════════════════════════════════════════════
# TestBuildFilterComplex — filtergraph builder
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFilterComplex(unittest.TestCase):
    """ExportPlan.filter_complex — 12 tests"""

    def test_single_segment_has_trim(self):
        plan = _make_plan([(0.0, 5.0)])
        fc = plan.filter_complex
        self.assertIn("trim=start=0.000:end=5.000", fc)

    def test_single_segment_has_null(self):
        # Single segment → null/anull instead of concat
        plan = _make_plan([(0.0, 5.0)])
        fc = plan.filter_complex
        self.assertIn("null", fc)

    def test_multi_segment_has_concat(self):
        plan = _make_plan([(0.0, 3.0), (5.0, 8.0)])
        fc = plan.filter_complex
        self.assertIn("concat=n=2", fc)

    def test_multi_segment_audio_trimmed(self):
        plan = _make_plan([(0.0, 3.0), (5.0, 8.0)])
        fc = plan.filter_complex
        self.assertIn("atrim", fc)
        self.assertIn("asetpts", fc)

    def test_no_segments_returns_passthrough(self):
        plan = _make_plan([])
        plan.keep_segments = []
        fc = plan.filter_complex
        self.assertIn("null", fc)

    def test_caption_produces_drawtext(self):
        cap = {"type": "caption", "start": 1.0, "end": 3.0, "text": "Hello", "style": "shorts_default"}
        plan = _make_plan([(0.0, 5.0)], captions=[cap])
        fc = plan.filter_complex
        self.assertIn("drawtext", fc)
        self.assertIn("Hello", fc)

    def test_caption_uses_enable_expression(self):
        cap = {"type": "caption", "start": 1.0, "end": 3.0, "text": "Hi", "style": "shorts_default"}
        plan = _make_plan([(0.0, 5.0)], captions=[cap])
        fc = plan.filter_complex
        self.assertIn("enable=", fc)
        self.assertIn("between(t,", fc)

    def test_duck_produces_volume_filter(self):
        duck = {"type": "duck", "start": 1.0, "end": 3.0}
        plan = _make_plan([(0.0, 5.0)], ducks=[duck])
        fc = plan.filter_complex
        self.assertIn("volume=", fc)
        self.assertIn("eval=frame", fc)

    def test_zoom_produces_zoompan(self):
        zoom = {
            "type": "zoom", "start": 0.0, "end": 5.0,
            "zoom_factor": 1.1, "anchor_x": 0.5, "anchor_y": 0.5,
        }
        plan = _make_plan([(0.0, 5.0)], zooms=[zoom])
        fc = plan.filter_complex
        self.assertIn("zoompan", fc)

    def test_zoom_specifies_dimensions(self):
        zoom = {
            "type": "zoom", "start": 0.0, "end": 5.0,
            "zoom_factor": 1.1, "anchor_x": 0.5, "anchor_y": 0.5,
        }
        plan = _make_plan([(0.0, 5.0)], zooms=[zoom])
        fc = plan.filter_complex
        self.assertIn("s=1080x1920", fc)

    def test_caption_deleted_segment_skipped(self):
        # Caption whose source timestamps fall entirely in a deleted segment
        cap = {"type": "caption", "start": 10.0, "end": 12.0, "text": "Ghost", "style": "shorts_default"}
        plan = _make_plan([(0.0, 5.0)], captions=[cap])
        fc = plan.filter_complex
        # Ghost caption falls outside any keep segment — should be omitted
        self.assertNotIn("Ghost", fc)

    def test_parts_joined_with_semicolon(self):
        plan = _make_plan([(0.0, 3.0), (5.0, 8.0)])
        fc = plan.filter_complex
        self.assertIn(";", fc)


# ═══════════════════════════════════════════════════════════════════════════════
# TestExportCommandStructure — to_command()
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportCommandStructure(unittest.TestCase):
    """ExportPlan.to_command() — 8 tests"""

    def _plan(self):
        return _make_plan([(0.0, 5.0)])

    def test_starts_with_ffmpeg(self):
        cmd = self._plan().to_command()
        self.assertTrue(cmd.startswith("ffmpeg"))

    def test_includes_input_flag(self):
        cmd = self._plan().to_command()
        self.assertIn("-i", cmd)

    def test_includes_filter_complex(self):
        cmd = self._plan().to_command()
        self.assertIn("-filter_complex", cmd)

    def test_includes_map_flags(self):
        cmd = self._plan().to_command()
        self.assertIn("-map", cmd)

    def test_includes_video_codec(self):
        cmd = self._plan().to_command()
        self.assertIn("libx264", cmd)

    def test_includes_audio_codec(self):
        cmd = self._plan().to_command()
        self.assertIn("aac", cmd)

    def test_includes_output_dimensions(self):
        cmd = self._plan().to_command()
        self.assertIn("1080x1920", cmd)

    def test_output_file_in_command(self):
        plan = _make_plan([(0.0, 5.0)], output="staging/job/output.mp4")
        cmd = plan.to_command()
        self.assertIn("staging/job/output.mp4", cmd)


# ═══════════════════════════════════════════════════════════════════════════════
# TestExportPlanCompute — time remapping
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportPlanCompute(unittest.TestCase):
    """ExportPlan.compute_output_time() and total_output_duration_s — 8 tests"""

    def _plan(self):
        # Keep [0,3] and [5,8]; delete [3,5]
        return _make_plan([(0.0, 3.0), (5.0, 8.0)])

    def test_start_of_first_segment(self):
        self.assertAlmostEqual(self._plan().compute_output_time(0.0), 0.0)

    def test_middle_of_first_segment(self):
        self.assertAlmostEqual(self._plan().compute_output_time(1.5), 1.5)

    def test_end_of_first_segment(self):
        self.assertAlmostEqual(self._plan().compute_output_time(3.0), 3.0)

    def test_start_of_second_segment(self):
        # source 5.0 → output 3.0 (offset from first segment)
        self.assertAlmostEqual(self._plan().compute_output_time(5.0), 3.0)

    def test_middle_of_second_segment(self):
        # source 6.5 → output 4.5
        self.assertAlmostEqual(self._plan().compute_output_time(6.5), 4.5)

    def test_deleted_segment_returns_none(self):
        result = self._plan().compute_output_time(4.0)
        self.assertIsNone(result)

    def test_total_duration_two_segments(self):
        self.assertAlmostEqual(self._plan().total_output_duration_s, 6.0)

    def test_total_duration_single_segment(self):
        plan = _make_plan([(0.0, 10.0)])
        self.assertAlmostEqual(plan.total_output_duration_s, 10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TestExporterMain — harness integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestExporterMain(unittest.TestCase):
    """skills/exporter/main.run() — 8 tests"""

    def _make_cut_list(self, tmp_dir: Path) -> str:
        data = {
            "cut_segments": [
                {"start": 0.0, "end": 5.0, "action": "keep", "effects": []},
            ]
        }
        p = tmp_dir / "cut_list.json"
        p.write_text(json.dumps(data))
        return str(p)

    def _make_timeline(self, tmp_dir: Path) -> str:
        data = {"visual_elements": []}
        p = tmp_dir / "annotated_timeline.json"
        p.write_text(json.dumps(data))
        return str(p)

    def _run_dry(self, tmp_dir: Path, **extra):
        import skills.exporter.main as exporter_mod
        import harness.sandbox.executor as executor_mod

        cut = self._make_cut_list(tmp_dir)
        timeline = self._make_timeline(tmp_dir)

        with patch.object(executor_mod, "STAGING_ROOT", tmp_dir):
            return exporter_mod.run(
                job_id="test_job",
                source_file="",
                staging_dir=tmp_dir,
                cut_list_path=cut,
                annotated_timeline_path=timeline,
                dry_run=True,
                **extra,
            )

    def test_dry_run_returns_success(self):
        with tempfile.TemporaryDirectory() as td:
            record = self._run_dry(Path(td))
            self.assertEqual(record.status, "success")

    def test_dry_run_skill_is_exporter(self):
        with tempfile.TemporaryDirectory() as td:
            record = self._run_dry(Path(td))
            self.assertEqual(record.skill, "exporter")

    def test_dry_run_output_path_set(self):
        with tempfile.TemporaryDirectory() as td:
            record = self._run_dry(Path(td))
            self.assertTrue(record.output_path.endswith("output.mp4"))

    def test_dry_run_payload_has_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            record = self._run_dry(Path(td))
            self.assertIn("metadata", record.payload)

    def test_dry_run_payload_ffmpeg_command(self):
        with tempfile.TemporaryDirectory() as td:
            record = self._run_dry(Path(td))
            self.assertIn("ffmpeg_command", record.payload["metadata"])

    def test_dry_run_true_in_payload(self):
        with tempfile.TemporaryDirectory() as td:
            record = self._run_dry(Path(td))
            self.assertTrue(record.payload["output"]["dry_run"])

    def test_missing_cut_list_returns_failed(self):
        import skills.exporter.main as exporter_mod
        with tempfile.TemporaryDirectory() as td:
            # No cut_list_path and no memory entry
            record = exporter_mod.run(
                job_id="no_cutter_job",
                source_file="",
                staging_dir=Path(td),
                dry_run=True,
            )
            self.assertEqual(record.status, "failed")

    def test_manifest_json_written(self):
        with tempfile.TemporaryDirectory() as td:
            self._run_dry(Path(td))
            manifest = Path(td) / "export_manifest.json"
            self.assertTrue(manifest.exists())


# ═══════════════════════════════════════════════════════════════════════════════
# TestWorkflowManagerResume — orchestrator resume logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkflowManagerResume(unittest.TestCase):
    """WorkflowManager resume behaviour — 10 tests"""

    def _mgr(self, job_id, staging):
        from src.pipeline.orchestrator import WorkflowManager
        return WorkflowManager(
            job_id=job_id,
            source_file="dummy.mp4",
            staging_dir=staging,
            dry_run=True,
        )

    def test_complete_job_returns_success_immediately(self):
        """If all 4 skills already succeeded, run() returns without re-executing."""
        from src.pipeline.orchestrator import WorkflowManager
        from harness.memory.manager import SkillRecord

        with tempfile.TemporaryDirectory() as td:
            job_id = "resume_complete"
            wm = WorkflowManager(
                job_id=job_id,
                source_file="dummy.mp4",
                staging_dir=Path(td),
                dry_run=True,
            )
            mock_resume = MagicMock()
            mock_resume.is_complete = True
            mock_resume.completed = ["transcriber", "cutter", "designer", "exporter"]
            mock_resume.records = {
                s: _make_success_record(job_id, s, f"/staging/{s}_out") for s in mock_resume.completed
            }
            mock_resume.next_skill = None

            with patch("src.pipeline.orchestrator.MemoryManager") as MockMgr:
                MockMgr.return_value.find_resume_point.return_value = mock_resume
                result = wm.run()

            self.assertEqual(result.status, "success")
            self.assertEqual(result.skipped, mock_resume.completed)

    def test_resume_from_cutter(self):
        """If transcriber succeeded, pipeline resumes from cutter."""
        from src.pipeline.orchestrator import WorkflowManager

        with tempfile.TemporaryDirectory() as td:
            job_id = "resume_cutter"
            wm = WorkflowManager(
                job_id=job_id,
                source_file="dummy.mp4",
                staging_dir=Path(td),
                dry_run=True,
            )

            mock_resume = MagicMock()
            mock_resume.is_complete = False
            mock_resume.completed = ["transcriber"]
            mock_resume.next_skill = "cutter"
            mock_resume.cursor = "00:00:05.000"
            mock_resume.records = {
                "transcriber": _make_success_record(job_id, "transcriber", "/staging/transcript.json")
            }
            mock_resume.prior_output = lambda s: mock_resume.records.get(s, MagicMock()).output_path if s in mock_resume.records else None

            executed = []

            def fake_cutter_run(**kwargs):
                executed.append("cutter")
                return _make_success_record(job_id, "cutter", "/staging/cut_list.json")

            def fake_designer_run(**kwargs):
                executed.append("designer")
                return _make_success_record(job_id, "designer", "/staging/annotated_timeline.json")

            def fake_exporter_run(**kwargs):
                executed.append("exporter")
                return _make_success_record(job_id, "exporter", "/staging/output.mp4")

            cutter_mock = MagicMock()
            cutter_mock.run = fake_cutter_run
            designer_mock = MagicMock()
            designer_mock.run = fake_designer_run
            exporter_mock = MagicMock()
            exporter_mock.run = fake_exporter_run

            with patch("src.pipeline.orchestrator.MemoryManager") as MockMgr, \
                 patch("importlib.import_module") as mock_import:
                MockMgr.return_value.find_resume_point.return_value = mock_resume

                def side_effect(name):
                    if "cutter" in name:
                        return cutter_mock
                    if "designer" in name:
                        return designer_mock
                    if "exporter" in name:
                        return exporter_mock
                    return MagicMock()

                mock_import.side_effect = side_effect
                result = wm.run()

            self.assertEqual(result.skipped, ["transcriber"])
            self.assertIn("cutter", executed)
            self.assertIn("designer", executed)
            self.assertIn("exporter", executed)
            self.assertNotIn("transcriber", executed)

    def test_failed_skill_stops_pipeline(self):
        """Pipeline stops at the first failed skill."""
        from src.pipeline.orchestrator import WorkflowManager
        from harness.memory.manager import SkillRecord

        with tempfile.TemporaryDirectory() as td:
            job_id = "resume_fail"
            wm = WorkflowManager(
                job_id=job_id,
                source_file="dummy.mp4",
                staging_dir=Path(td),
                dry_run=True,
            )

            mock_resume = MagicMock()
            mock_resume.is_complete = False
            mock_resume.completed = []
            mock_resume.next_skill = "transcriber"
            mock_resume.cursor = "00:00:00.000"
            mock_resume.records = {}
            mock_resume.prior_output = lambda s: None

            failed = SkillRecord(
                job_id=job_id, skill="transcriber", status="failed",
                output_path="", cursor_start="00:00:00.000", cursor_end="00:00:00.000",
                error="Whisper not available",
            )

            transcriber_mock = MagicMock()
            transcriber_mock.run.return_value = failed

            with patch("src.pipeline.orchestrator.MemoryManager") as MockMgr, \
                 patch("importlib.import_module", return_value=transcriber_mock):
                MockMgr.return_value.find_resume_point.return_value = mock_resume
                result = wm.run()

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.failed_skill, "transcriber")

    def test_result_skipped_matches_completed_from_memory(self):
        """skipped list equals the completed list from the resume point."""
        from src.pipeline.orchestrator import WorkflowManager

        with tempfile.TemporaryDirectory() as td:
            job_id = "skipped_check"
            wm = WorkflowManager(
                job_id=job_id,
                source_file="dummy.mp4",
                staging_dir=Path(td),
                dry_run=True,
            )

            mock_resume = MagicMock()
            mock_resume.is_complete = True
            mock_resume.completed = ["transcriber", "cutter"]
            mock_resume.records = {
                s: _make_success_record(job_id, s, f"/out/{s}") for s in mock_resume.completed
            }

            with patch("src.pipeline.orchestrator.MemoryManager") as MockMgr:
                MockMgr.return_value.find_resume_point.return_value = mock_resume
                result = wm.run()

            self.assertEqual(sorted(result.skipped), sorted(["transcriber", "cutter"]))

    def test_result_succeeded_property(self):
        from src.pipeline.orchestrator import PipelineResult
        r = PipelineResult(job_id="j", status="success")
        self.assertTrue(r.succeeded)

    def test_result_not_succeeded_on_fail(self):
        from src.pipeline.orchestrator import PipelineResult
        r = PipelineResult(job_id="j", status="failed", failed_skill="cutter")
        self.assertFalse(r.succeeded)


# ═══════════════════════════════════════════════════════════════════════════════
# TestBuildExportPlan — build_export_plan() filtering
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildExportPlan(unittest.TestCase):
    """build_export_plan() — 7 tests"""

    def _segs(self):
        return [
            {"start": 0.0, "end": 3.0, "action": "keep", "effects": []},
            {"start": 3.0, "end": 5.0, "action": "delete", "effects": []},
            {"start": 5.0, "end": 8.0, "action": "keep", "effects": []},
        ]

    def test_keep_segments_filtered(self):
        plan = build_export_plan(self._segs(), [], source_file="s", output_file="o")
        self.assertEqual(len(plan.keep_segments), 2)
        self.assertTrue(all(s["action"] == "keep" for s in plan.keep_segments))

    def test_delete_segments_excluded(self):
        plan = build_export_plan(self._segs(), [], source_file="s", output_file="o")
        for seg in plan.keep_segments:
            self.assertNotEqual(seg["action"], "delete")

    def test_captions_extracted(self):
        ve = [{"type": "caption", "start": 0.0, "end": 1.0, "text": "Hi"}]
        plan = build_export_plan(self._segs(), ve, source_file="s", output_file="o")
        self.assertEqual(len(plan.captions), 1)

    def test_zoom_extracted(self):
        ve = [{"type": "zoom", "start": 0.0, "end": 3.0, "zoom_factor": 1.1}]
        plan = build_export_plan(self._segs(), ve, source_file="s", output_file="o")
        self.assertEqual(len(plan.zoom_elements), 1)

    def test_duck_extracted(self):
        # Duck must be in the deleted/silence region (3.0-5.0) to pass the
        # "don't duck speech" filter; overlapping a keep segment is filtered out.
        ve = [{"type": "duck", "start": 3.2, "end": 4.8}]
        plan = build_export_plan(self._segs(), ve, source_file="s", output_file="o")
        self.assertEqual(len(plan.duck_events), 1)

    def test_default_profile_is_shorts(self):
        plan = build_export_plan(self._segs(), [], source_file="s", output_file="o")
        self.assertEqual(plan.export_profile["platform"], "youtube_shorts")

    def test_custom_profile_applied(self):
        profile = dict(DEFAULT_PROFILE)
        profile["crf"] = 23
        plan = build_export_plan(self._segs(), [], source_file="s", output_file="o",
                                 export_profile=profile)
        self.assertEqual(plan.export_profile["crf"], 23)


# ═══════════════════════════════════════════════════════════════════════════════
# TestFullPipelineE2E — end-to-end with mock skills
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineE2E(unittest.TestCase):
    """Full pipeline end-to-end using all real skill modules (dry_run=True) — 8 tests"""

    def _write_cut_list(self, staging: Path) -> str:
        data = {
            "cut_segments": [
                {"start": 0.0, "end": 5.0, "action": "keep", "effects": []},
                {"start": 5.0, "end": 7.0, "action": "delete", "effects": []},
                {"start": 7.0, "end": 12.0, "action": "keep", "effects": ["jump_cut_zoom_1.1"]},
            ]
        }
        p = staging / "cut_list.json"
        staging.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
        return str(p)

    def _write_transcript(self, staging: Path) -> str:
        data = {
            "words": [
                {"word": "중요", "start_ms": 500, "end_ms": 1000, "confidence": 0.95, "speaker_id": "A"},
                {"word": "테스트", "start_ms": 1200, "end_ms": 1800, "confidence": 0.90, "speaker_id": "A"},
            ],
            "vad_segments": [
                {"start": 0.0, "end": 3.0, "is_voice": True, "confidence": 0.92},
            ],
            "metadata": {"duration_s": 12.0, "model": "base", "language": "ko",
                          "word_count": 2, "avg_confidence": 0.92},
        }
        p = staging / "transcript.json"
        p.write_text(json.dumps(data))
        return str(p)

    def _run_designer_stage(self, staging: Path, cut_path: str, transcript_path: str):
        import skills.designer.main as designer_mod
        return designer_mod.run(
            job_id="e2e_job",
            staging_dir=staging,
            cut_list_path=cut_path,
            transcript_path=transcript_path,
        )

    def _run_exporter_stage(self, staging: Path, cut_path: str, timeline_path: str):
        import skills.exporter.main as exporter_mod
        import harness.sandbox.executor as executor_mod
        with patch.object(executor_mod, "STAGING_ROOT", staging):
            return exporter_mod.run(
                job_id="e2e_job",
                source_file="",
                staging_dir=staging,
                cut_list_path=cut_path,
                annotated_timeline_path=timeline_path,
                dry_run=True,
            )

    def test_designer_stage_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            cut = self._write_cut_list(staging)
            tx = self._write_transcript(staging)
            record = self._run_designer_stage(staging, cut, tx)
            self.assertEqual(record.status, "success")

    def test_designer_writes_annotated_timeline(self):
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            cut = self._write_cut_list(staging)
            tx = self._write_transcript(staging)
            self._run_designer_stage(staging, cut, tx)
            self.assertTrue((staging / "annotated_timeline.json").exists())

    def test_exporter_stage_succeeds_after_designer(self):
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            cut = self._write_cut_list(staging)
            tx = self._write_transcript(staging)
            d_record = self._run_designer_stage(staging, cut, tx)
            e_record = self._run_exporter_stage(staging, cut, d_record.output_path)
            self.assertEqual(e_record.status, "success")

    def test_exporter_dry_run_flag_in_payload(self):
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            cut = self._write_cut_list(staging)
            tx = self._write_transcript(staging)
            d_record = self._run_designer_stage(staging, cut, tx)
            e_record = self._run_exporter_stage(staging, cut, d_record.output_path)
            self.assertTrue(e_record.payload["output"]["dry_run"])

    def test_exporter_quality_gates_reported(self):
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            cut = self._write_cut_list(staging)
            tx = self._write_transcript(staging)
            d_record = self._run_designer_stage(staging, cut, tx)
            e_record = self._run_exporter_stage(staging, cut, d_record.output_path)
            self.assertIn("quality_gates", e_record.payload["metadata"])

    def test_exporter_ffmpeg_command_starts_with_ffmpeg(self):
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            cut = self._write_cut_list(staging)
            tx = self._write_transcript(staging)
            d_record = self._run_designer_stage(staging, cut, tx)
            e_record = self._run_exporter_stage(staging, cut, d_record.output_path)
            cmd = e_record.payload["metadata"]["ffmpeg_command"]
            self.assertTrue(cmd.startswith("ffmpeg"))

    def test_manifest_written_by_exporter(self):
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            cut = self._write_cut_list(staging)
            tx = self._write_transcript(staging)
            d_record = self._run_designer_stage(staging, cut, tx)
            self._run_exporter_stage(staging, cut, d_record.output_path)
            manifest = staging / "export_manifest.json"
            self.assertTrue(manifest.exists())

    def test_manifest_contains_expected_keys(self):
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            cut = self._write_cut_list(staging)
            tx = self._write_transcript(staging)
            d_record = self._run_designer_stage(staging, cut, tx)
            self._run_exporter_stage(staging, cut, d_record.output_path)
            manifest = json.loads((staging / "export_manifest.json").read_text())
            for key in ("output_file", "keep_segment_count", "total_duration_s",
                        "ffmpeg_command", "quality_gate_summary"):
                self.assertIn(key, manifest)


if __name__ == "__main__":
    unittest.main()
