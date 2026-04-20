"""
src/pipeline/orchestrator.py

WorkflowManager — coordinates the Transcriber → Cutter → Designer → Exporter pipeline.

Responsibilities
----------------
1. Accept a job configuration (source file, job_id, optional overrides).
2. Consult MemoryManager to find the resume point (skip already-successful skills).
3. Execute each skill in order, passing outputs forward via SkillRecord.output_path.
4. On failure, stop the pipeline and return the failed SkillRecord.
5. On success, return the final exporter SkillRecord.

Resume logic
------------
- Skills already recorded as "success" in harness/memory/jobs/{job_id}/ are skipped.
- The pipeline resumes from the first skill that has no successful record.
- Output paths for skipped skills are read from their existing SkillRecord.

Stateless guarantee
-------------------
WorkflowManager holds no cross-run state. Everything is derived from MemoryManager
on each call to run().
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness.memory.manager import MemoryManager, SkillRecord, SKILL_ORDER

_ROOT = Path(__file__).resolve().parents[2]
_BROLL_REQUESTS_FILE = _ROOT / "spec" / "broll_requests.json"

logger = logging.getLogger(__name__)

# Lazy imports for skills to allow mocking in tests
_SKILL_MODULES: dict[str, str] = {
    "transcriber": "skills.transcriber.main",
    "cutter":      "skills.cutter.main",
    "designer":    "skills.designer.main",
    "exporter":    "skills.exporter.main",
}


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Summary returned after a full or partial pipeline run.

    Attributes
    ----------
    job_id       : The job that was executed.
    status       : "success" if all skills passed; "failed" if any skill failed.
    completed    : List of skill names that succeeded in this (or prior) run.
    skipped      : List of skill names that were skipped (already succeeded).
    failed_skill : Name of the skill that failed, or None on full success.
    records      : Mapping skill → SkillRecord for every skill that was run.
    final_record : The last SkillRecord produced (success or failure).
    """

    job_id: str
    status: str
    completed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed_skill: Optional[str] = None
    records: dict[str, SkillRecord] = field(default_factory=dict)
    final_record: Optional[SkillRecord] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


# ── WorkflowManager ───────────────────────────────────────────────────────────

class WorkflowManager:
    """Coordinate the VoxEdit AI pipeline for a single job.

    Usage
    -----
    >>> wm = WorkflowManager(job_id="job_20260418_001", source_file="/raw/video.mp4")
    >>> result = wm.run()
    >>> print(result.status)

    Parameters
    ----------
    job_id          : Unique job identifier.
    source_file     : Absolute path to the raw source video/audio.
    staging_dir     : Override staging root (default: project-level staging/{job_id}).
    language        : Language hint for Transcriber (None = auto).
    model_name      : Whisper model name (default "base").
    export_profile  : Override export settings dict passed to Exporter.
    dry_run           : If True, Exporter validates FFmpeg command but does not render.
    skill_overrides   : Dict of skill → kwarg dict for low-level test overrides.
                        E.g. {"transcriber": {"backend": mock_backend}}
    force_resume_from : Smart Resume — skip all skills BEFORE this one even if they
                        have no records in this job, borrowing outputs from
                        borrow_records instead.  E.g. "exporter" re-renders only.
    borrow_records    : SkillRecords from a prior job to use as prior_output() sources
                        for any skill before force_resume_from.
    """

    def __init__(
        self,
        *,
        job_id: str,
        source_file: str,
        staging_dir: Optional[Path] = None,
        language: Optional[str] = None,
        model_name: str = "base",
        export_profile: Optional[dict] = None,
        dry_run: bool = False,
        skill_overrides: Optional[dict[str, dict]] = None,
        force_resume_from: Optional[str] = None,
        borrow_records: Optional[dict[str, "SkillRecord"]] = None,
    ) -> None:
        self.job_id = job_id
        self.source_file = source_file
        self.staging_dir = staging_dir
        self.language = language
        self.model_name = model_name
        self.export_profile = export_profile
        self.dry_run = dry_run
        self.skill_overrides = skill_overrides or {}
        self.force_resume_from = force_resume_from
        self.borrow_records: dict[str, SkillRecord] = borrow_records or {}

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> PipelineResult:
        """Execute the pipeline, resuming from the last successful skill.

        Returns
        -------
        PipelineResult with status "success" or "failed".
        """
        # ── Micro-Resume fast paths ────────────────────────────────────────
        if self.force_resume_from == "designer_fast":
            return self._run_micro_resume()
        if self.force_resume_from == "audio_only":
            return self._run_audio_only()
        if self.force_resume_from == "visual_fast":
            return self._run_visual_fast()

        mgr = MemoryManager(self.job_id)
        resume = mgr.find_resume_point()

        if self.force_resume_from and self.force_resume_from in SKILL_ORDER:
            resume = self._apply_force_resume(resume, mgr)

        result = PipelineResult(
            job_id=self.job_id,
            status="success",
            skipped=list(resume.completed),
            records=dict(resume.records),
        )

        if resume.is_complete:
            logger.info(
                "WorkflowManager: job '%s' already complete — nothing to run", self.job_id
            )
            result.completed = list(resume.completed)
            result.final_record = resume.records.get(SKILL_ORDER[-1])
            return result

        logger.info(
            "WorkflowManager: starting job '%s' from skill '%s'  cursor=%s",
            self.job_id, resume.next_skill, resume.cursor,
        )

        current_cursor = resume.cursor

        for skill in SKILL_ORDER:
            # Skip already-completed skills
            if skill in resume.completed:
                logger.info("WorkflowManager: skipping '%s' (already succeeded)", skill)
                continue

            logger.info("WorkflowManager: executing skill '%s'", skill)
            record = self._run_skill(skill, current_cursor, resume)

            result.records[skill] = record
            result.final_record = record

            if record.status != "success":
                result.status = "failed"
                result.failed_skill = skill
                logger.error(
                    "WorkflowManager: skill '%s' failed — stopping pipeline. error=%s",
                    skill, record.error,
                )
                return result

            result.completed.append(skill)
            current_cursor = record.cursor_end
            # Update resume so later skills can find this record's output_path
            resume.records[skill] = record

        logger.info(
            "WorkflowManager: pipeline complete — job='%s'  skills=%s",
            self.job_id, result.completed,
        )
        return result

    # ── Smart Resume ────────────────────────────────────────────────────────

    def _apply_force_resume(self, resume, mgr: MemoryManager):
        """Inject borrow_records for skills before force_resume_from.

        Writes placeholder SkillRecords to this job's memory dir so that
        MEMORY_CONSISTENCY checks see a full 4-skill success set.
        Then forces resume.next_skill = force_resume_from.
        """
        from dataclasses import asdict
        forced_idx = SKILL_ORDER.index(self.force_resume_from)

        for skill in SKILL_ORDER[:forced_idx]:
            if skill in resume.completed:
                continue  # already in this job — keep it
            src = self.borrow_records.get(skill)
            if src is None:
                logger.warning(
                    "WorkflowManager: force_resume_from=%s but no borrow_record for '%s' — "
                    "pipeline will run from transcriber instead",
                    self.force_resume_from, skill,
                )
                return resume  # fall back to normal resume

            placeholder = SkillRecord(
                job_id=self.job_id,
                skill=skill,
                status="success",
                output_path=src.output_path,
                cursor_start=src.cursor_start,
                cursor_end=src.cursor_end,
                payload={"borrowed_from_job": src.job_id},
            )
            mgr.write(placeholder)
            resume.records[skill] = placeholder

        resume.completed = [s for s in SKILL_ORDER[:forced_idx] if s in resume.records]
        resume.next_skill = self.force_resume_from
        if resume.completed:
            resume.cursor = resume.records[resume.completed[-1]].cursor_end
        else:
            resume.cursor = "00:00:00.000"

        logger.info(
            "WorkflowManager: smart resume — force_from=%s  borrowed=%s  cursor=%s",
            self.force_resume_from, list(resume.completed), resume.cursor,
        )
        return resume

    # ── Micro-Resume (designer_fast) ────────────────────────────────────────

    def _run_micro_resume(self) -> PipelineResult:
        """Micro-Resume: patch b-roll elements only, skip to Exporter.

        Flow
        ----
        1. Load prior annotated_timeline.json from borrow_records['designer'].
        2. Replace ALL b-roll elements with fresh ones from broll_requests.json.
        3. Write the patched timeline to this job's staging directory.
        4. Write placeholder SkillRecords for transcriber, cutter, designer.
        5. Run ONLY the Exporter skill.

        Falls back to force_resume_from='exporter' if prior records are missing.
        """
        designer_rec = self.borrow_records.get("designer")
        if designer_rec is None:
            logger.warning(
                "Micro-resume: no prior designer record for job '%s' — "
                "falling back to exporter-only resume",
                self.job_id,
            )
            self.force_resume_from = "exporter"
            return self.run()

        patched_path = self._patch_broll_in_timeline(designer_rec)
        if patched_path is None:
            logger.warning(
                "Micro-resume: timeline patch failed for job '%s' — "
                "falling back to exporter-only resume",
                self.job_id,
            )
            self.force_resume_from = "exporter"
            return self.run()

        # Update borrow_records so exporter sees the patched timeline
        patched_designer = SkillRecord(
            job_id=self.job_id,
            skill="designer",
            status="success",
            output_path=str(patched_path),
            cursor_start=designer_rec.cursor_start,
            cursor_end=designer_rec.cursor_end,
            payload={"fast_resume": True},
        )
        self.borrow_records["designer"] = patched_designer

        # Write borrowed placeholder records to this job's memory dir
        mgr = MemoryManager(self.job_id)
        for skill in ("transcriber", "cutter", "designer"):
            src = self.borrow_records.get(skill)
            if src is None:
                continue
            placeholder = SkillRecord(
                job_id=self.job_id,
                skill=skill,
                status="success",
                output_path=src.output_path,
                cursor_start=src.cursor_start,
                cursor_end=src.cursor_end,
                payload={"borrowed_from_job": src.job_id, "fast_resume": True},
            )
            mgr.write(placeholder)

        # Resume from exporter
        resume = mgr.find_resume_point()
        result = PipelineResult(
            job_id=self.job_id,
            status="success",
            skipped=["transcriber", "cutter", "designer"],
            records=dict(resume.records),
        )

        if resume.is_complete:
            result.completed = list(resume.completed)
            result.final_record = resume.records.get(SKILL_ORDER[-1])
            return result

        current_cursor = resume.cursor
        record = self._run_skill("exporter", current_cursor, resume)
        result.records["exporter"] = record
        result.final_record = record

        if record.status != "success":
            result.status = "failed"
            result.failed_skill = "exporter"
            logger.error(
                "Micro-resume: exporter failed — job='%s'  error=%s",
                self.job_id, record.error,
            )
        else:
            result.completed = ["exporter"]
            logger.info(
                "Micro-resume complete — job='%s'  skipped=3 skills  "
                "patched_timeline=%s",
                self.job_id, patched_path.name,
            )
        return result

    def _patch_broll_in_timeline(self, designer_rec: SkillRecord) -> Optional[Path]:
        """Replace b-roll elements in the prior annotated_timeline.json.

        Reads the current broll_requests.json, rebuilds VisualElement objects,
        and writes a patched timeline to this job's staging directory.
        Returns the path of the patched file, or None on any error.
        """
        try:
            prior_path = Path(designer_rec.output_path)
            if not prior_path.exists():
                logger.error(
                    "Micro-resume: prior timeline not found: %s", prior_path
                )
                return None

            timeline = json.loads(prior_path.read_text(encoding="utf-8"))
            existing = timeline.get("visual_elements", [])

            # Total duration from existing elements (needed by build_broll_elements)
            total_s = max(
                (e.get("end", 0.0) for e in existing), default=30.0
            )

            # Load current broll requests
            broll_requests: list[dict] = []
            if _BROLL_REQUESTS_FILE.exists():
                broll_requests = json.loads(
                    _BROLL_REQUESTS_FILE.read_text(encoding="utf-8")
                )

            from skills.designer.logic import build_broll_elements
            new_brolls = build_broll_elements(
                broll_requests, total_duration_s=total_s
            )

            # Keep everything except b-roll, then append fresh b-roll elements
            non_broll = [e for e in existing if e.get("type") != "b-roll"]
            timeline["visual_elements"] = non_broll + [
                b.to_dict() for b in new_brolls
            ]
            if "metadata" in timeline:
                timeline["metadata"]["broll_count"] = len(new_brolls)

            # Write patched timeline to this job's staging directory
            staging = self.staging_dir
            if staging is None:
                staging = _ROOT / "staging" / self.job_id
            staging.mkdir(parents=True, exist_ok=True)

            out_path = staging / "annotated_timeline.json"
            out_path.write_text(
                json.dumps(timeline, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                "Micro-resume: patched timeline → %s  "
                "non_broll=%d  new_brolls=%d",
                out_path.name, len(non_broll), len(new_brolls),
            )
            return out_path

        except Exception as exc:
            logger.error("Micro-resume: _patch_broll_in_timeline failed — %s", exc)
            return None

    # ── Visual-Fast Resume (designer + exporter, ultrafast encoding) ────────

    def _run_visual_fast(self) -> PipelineResult:
        """Visual-fast resume: re-run Designer + Exporter with ultrafast preset.

        Use case
        --------
        Only visual parameters changed (RHYTHM_INTENSITY, zoom style, etc.).
        Transcriber and Cutter outputs are unchanged — borrow them.

        Flow
        ----
        1. Override export_profile with ``preset="ultrafast"`` and ``crf=28``
           for ~3× faster encoding (preview quality).
        2. Fall through to standard ``force_resume_from="designer"`` which
           calls ``_apply_force_resume`` and borrows transcriber + cutter.

        Falls back gracefully when borrow_records is empty (runs from scratch).
        """
        from skills.exporter.logic import DEFAULT_PROFILE
        if self.export_profile is None:
            self.export_profile = dict(DEFAULT_PROFILE)
        self.export_profile["preset"] = "ultrafast"
        self.export_profile["crf"] = 28

        self.force_resume_from = "designer"
        return self.run()

    # ── Audio-Only Resume ───────────────────────────────────────────────────

    def _run_audio_only(self) -> PipelineResult:
        """Audio-only resume: skip video encoding, re-mux audio with -c:v copy.

        Flow
        ----
        1. Load prior exporter output (rendered video) from borrow_records['exporter'].
        2. Load keep_segments from borrow_records['cutter'] and duck_events from
           borrow_records['designer'].
        3. Build an ExportPlan carrying only audio filters (no captions, no b-roll).
        4. Call plan.to_audio_only_command(prior_video) which uses -c:v copy.
        5. Run exporter skill with the pre-built command (or validate in dry_run mode).

        Falls back to force_resume_from='exporter' if prior records are missing.
        Approximately 3–5× faster than a full render.
        """
        exporter_rec = self.borrow_records.get("exporter")
        cutter_rec   = self.borrow_records.get("cutter")
        designer_rec = self.borrow_records.get("designer")

        if exporter_rec is None or not exporter_rec.output_path:
            logger.warning(
                "audio_only: no prior exporter record for job '%s' — "
                "falling back to exporter-only resume",
                self.job_id,
            )
            self.force_resume_from = "exporter"
            return self.run()

        prior_video = Path(exporter_rec.output_path)
        if not prior_video.exists() and not self.dry_run:
            logger.warning(
                "audio_only: prior video not found '%s' for job '%s' — "
                "falling back to exporter-only resume",
                prior_video, self.job_id,
            )
            self.force_resume_from = "exporter"
            return self.run()

        # Derive staging directory
        staging = self.staging_dir
        if staging is None:
            staging = _ROOT / "staging" / self.job_id
        staging = Path(staging)
        staging.mkdir(parents=True, exist_ok=True)

        output_path = staging / "output_audio_only.mp4"

        # Load keep_segments from cutter record
        keep_segments: list[dict] = []
        if cutter_rec and cutter_rec.output_path:
            try:
                data = json.loads(Path(cutter_rec.output_path).read_text(encoding="utf-8"))
                keep_segments = [s for s in data.get("segments", []) if s.get("action") == "keep"]
            except Exception as exc:
                logger.warning("audio_only: could not load cut_list from cutter record — %s", exc)

        # Load duck_events from designer record
        duck_events: list[dict] = []
        if designer_rec and designer_rec.output_path:
            try:
                data = json.loads(Path(designer_rec.output_path).read_text(encoding="utf-8"))
                duck_events = [e for e in data.get("visual_elements", []) if e.get("type") == "duck"]
            except Exception as exc:
                logger.warning("audio_only: could not load duck_events from designer record — %s", exc)

        from skills.exporter.logic import ExportPlan, DEFAULT_PROFILE
        plan = ExportPlan(
            source_file=self.source_file,
            output_file=str(output_path),
            keep_segments=keep_segments,
            captions=[],
            zoom_elements=[],
            duck_events=duck_events,
            broll_elements=[],
            export_profile=self.export_profile or dict(DEFAULT_PROFILE),
        )

        cmd = plan.to_audio_only_command(str(prior_video))

        # Write placeholder records for skipped skills
        mgr = MemoryManager(self.job_id)
        for skill in ("transcriber", "cutter", "designer"):
            src = self.borrow_records.get(skill)
            if src is None:
                continue
            placeholder = SkillRecord(
                job_id=self.job_id,
                skill=skill,
                status="success",
                output_path=src.output_path,
                cursor_start=src.cursor_start,
                cursor_end=src.cursor_end,
                payload={"borrowed_from_job": src.job_id, "audio_only": True},
            )
            mgr.write(placeholder)

        if self.dry_run:
            # Validate only — return a synthetic success record
            exporter_out = SkillRecord(
                job_id=self.job_id,
                skill="exporter",
                status="success",
                output_path=str(output_path),
                cursor_start="00:00:00.000",
                cursor_end="00:00:00.000",
                payload={"audio_only_cmd": cmd, "dry_run": True},
            )
            mgr.write(exporter_out)
        else:
            import importlib
            mod = importlib.import_module(_SKILL_MODULES["exporter"])
            resume = mgr.find_resume_point()
            exporter_out = mod.run(
                job_id=self.job_id,
                source_file=self.source_file,
                cursor=resume.cursor,
                staging_dir=staging,
                dry_run=False,
                audio_only_cmd=cmd,
                **self.skill_overrides.get("exporter", {}),
            )

        result = PipelineResult(
            job_id=self.job_id,
            status="success" if exporter_out.status == "success" else "failed",
            skipped=["transcriber", "cutter", "designer"],
            completed=["exporter"] if exporter_out.status == "success" else [],
            failed_skill="exporter" if exporter_out.status != "success" else None,
            records={"exporter": exporter_out},
            final_record=exporter_out,
        )

        logger.info(
            "audio_only resume complete — job='%s'  output=%s  skipped=3 skills",
            self.job_id, output_path.name,
        )
        return result

    # ── Skill dispatch ──────────────────────────────────────────────────────

    def _run_skill(
        self,
        skill: str,
        cursor: str,
        resume,
    ) -> SkillRecord:
        """Dispatch to the appropriate skill module and return a SkillRecord."""
        overrides = self.skill_overrides.get(skill, {})

        if skill == "transcriber":
            return self._run_transcriber(cursor, overrides)
        elif skill == "cutter":
            return self._run_cutter(cursor, resume, overrides)
        elif skill == "designer":
            return self._run_designer(cursor, resume, overrides)
        elif skill == "exporter":
            return self._run_exporter(cursor, resume, overrides)
        else:
            raise ValueError(f"WorkflowManager: unknown skill '{skill}'")

    def _run_transcriber(self, cursor: str, overrides: dict) -> SkillRecord:
        import importlib
        mod = importlib.import_module(_SKILL_MODULES["transcriber"])
        return mod.run(
            job_id=self.job_id,
            source_file=self.source_file,
            language=self.language,
            model_name=self.model_name,
            staging_dir=self.staging_dir / "transcriber" if self.staging_dir else None,
            **overrides,
        )

    def _run_cutter(self, cursor: str, resume, overrides: dict) -> SkillRecord:
        import importlib
        mod = importlib.import_module(_SKILL_MODULES["cutter"])
        transcript_path = resume.prior_output("transcriber")
        kwargs = dict(
            job_id=self.job_id,
            cursor=cursor,
            staging_dir=self.staging_dir,
        )
        if transcript_path:
            kwargs["transcript_path"] = transcript_path
        kwargs.update(overrides)
        return mod.run(**kwargs)

    def _run_designer(self, cursor: str, resume, overrides: dict) -> SkillRecord:
        import importlib
        mod = importlib.import_module(_SKILL_MODULES["designer"])
        cut_list_path = resume.prior_output("cutter")
        transcript_path = resume.prior_output("transcriber")
        kwargs = dict(
            job_id=self.job_id,
            cursor=cursor,
            staging_dir=self.staging_dir,
        )
        if cut_list_path:
            kwargs["cut_list_path"] = cut_list_path
        if transcript_path:
            kwargs["transcript_path"] = transcript_path
        kwargs.update(overrides)
        return mod.run(**kwargs)

    def _run_exporter(self, cursor: str, resume, overrides: dict) -> SkillRecord:
        import importlib
        mod = importlib.import_module(_SKILL_MODULES["exporter"])
        cut_list_path = resume.prior_output("cutter")
        annotated_timeline_path = resume.prior_output("designer")
        kwargs = dict(
            job_id=self.job_id,
            source_file=self.source_file,
            cursor=cursor,
            staging_dir=self.staging_dir,
            dry_run=self.dry_run,
        )
        if self.export_profile:
            kwargs["export_profile"] = self.export_profile
        if cut_list_path:
            kwargs["cut_list_path"] = cut_list_path
        if annotated_timeline_path:
            kwargs["annotated_timeline_path"] = annotated_timeline_path
        kwargs.update(overrides)
        return mod.run(**kwargs)
