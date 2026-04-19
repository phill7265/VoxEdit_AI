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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness.memory.manager import MemoryManager, SkillRecord, SKILL_ORDER

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
