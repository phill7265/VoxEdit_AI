"""
harness/memory/manager.py

Handover & Resume manager for VoxEdit AI.

Responsibilities
----------------
- Write a typed Record to  harness/memory/jobs/{job_id}/{seq:03d}_{skill}.json
  after each skill execution (success OR failure).
- On session resume, scan the job directory and return the last successful
  record so the pipeline can start from exactly the right point.
- Enforce the "fail-forward" principle: a failed skill's record is preserved
  so a human or retry loop can inspect it without touching earlier successes.

Directory layout produced
-------------------------
harness/memory/
  jobs/
    {job_id}/
      001_transcriber.json
      002_cutter.json          ← may have status "failed"
      002_cutter_retry1.json   ← retry writes a new file, never overwrites
      003_designer.json
      ...
  handover_note.json           ← session-level summary (human-readable)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MEMORY_ROOT = Path(__file__).resolve().parent
JOBS_ROOT = MEMORY_ROOT / "jobs"

# Canonical pipeline skill order (used to derive sequence numbers)
SKILL_ORDER: list[str] = ["transcriber", "cutter", "designer", "exporter"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

SkillStatus = Literal["success", "failed", "skipped"]


@dataclass
class SkillRecord:
    """Immutable result written after every skill execution.

    Fields
    ------
    job_id       : Unique job identifier.
    skill        : Skill name (must be in SKILL_ORDER).
    status       : "success" | "failed" | "skipped"
    output_path  : Path to the skill's primary output file (staging-relative).
    cursor_start : Timecode where this skill's work segment began (HH:MM:SS.mmm).
    cursor_end   : Timecode where this skill's work segment ended.
    payload      : Skill-specific result data (cuts applied, word count, etc.).
    error        : Error message if status == "failed".
    recorded_at  : UTC ISO-8601 timestamp of when the record was written.
    sequence     : Auto-assigned execution order index (1-based).
    retry_index  : 0 for first attempt; increments on each retry.
    """

    job_id: str
    skill: str
    status: SkillStatus
    output_path: str
    cursor_start: str
    cursor_end: str
    payload: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    recorded_at: str = field(default_factory=lambda: _utcnow())
    sequence: int = 0          # set by MemoryManager.write()
    retry_index: int = 0


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Read/write handover records for a single job.

    Usage
    -----
    >>> mgr = MemoryManager("job_20260418_001")
    >>> mgr.write(record)                    # after skill completes
    >>> resume = mgr.find_resume_point()     # on session start
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.job_dir = JOBS_ROOT / job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, record: SkillRecord) -> Path:
        """Persist a SkillRecord to disk.  Never overwrites an existing file.

        The filename encodes execution order so lexicographic sort == time order:
          001_transcriber.json
          002_cutter_retry1.json   (retry_index > 0 appends suffix)
        """
        record.sequence = self._next_sequence(record.skill)
        suffix = f"_retry{record.retry_index}" if record.retry_index > 0 else ""
        filename = f"{record.sequence:03d}_{record.skill}{suffix}.json"
        path = self.job_dir / filename

        # Safety: never silently overwrite — append a conflict suffix instead
        if path.exists():
            path = self.job_dir / f"{record.sequence:03d}_{record.skill}{suffix}_conflict.json"
            logger.warning("Record file already exists; writing to %s", path.name)

        path.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False))
        logger.info("MemoryManager: wrote %s [%s]", path.name, record.status)
        return path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_all(self) -> list[SkillRecord]:
        """Return all records for this job, sorted by filename (= time order)."""
        records: list[SkillRecord] = []
        for path in sorted(self.job_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                records.append(SkillRecord(**data))
            except Exception as exc:
                logger.error("MemoryManager: failed to load %s — %s", path.name, exc)
        return records

    def load_successful(self) -> list[SkillRecord]:
        """Return only records with status == 'success'."""
        return [r for r in self.load_all() if r.status == "success"]

    def find_resume_point(self) -> ResumePoint:
        """Scan stored records and return where the pipeline should restart.

        Algorithm
        ---------
        1. Walk SKILL_ORDER.
        2. The first skill that has no successful record is the resume target.
        3. The resume cursor is the cursor_end of the last successful record
           (or "00:00:00.000" if nothing has succeeded yet).

        Returns a ResumePoint with:
          - next_skill : name of the skill to run next (None = job complete)
          - cursor     : timecode to resume from
          - completed  : list of skill names already done
          - records    : mapping of skill -> last successful SkillRecord
        """
        successful = self.load_successful()
        completed_map: dict[str, SkillRecord] = {}
        for rec in successful:
            # Keep the highest-sequence record per skill (= latest retry)
            if rec.skill not in completed_map or rec.sequence > completed_map[rec.skill].sequence:
                completed_map[rec.skill] = rec

        completed_skills = [s for s in SKILL_ORDER if s in completed_map]
        next_skill: Optional[str] = None
        for skill in SKILL_ORDER:
            if skill not in completed_map:
                next_skill = skill
                break

        # Resume cursor = cursor_end of the last completed skill in order
        resume_cursor = "00:00:00.000"
        for skill in reversed(SKILL_ORDER):
            if skill in completed_map:
                resume_cursor = completed_map[skill].cursor_end
                break

        point = ResumePoint(
            job_id=self.job_id,
            next_skill=next_skill,
            cursor=resume_cursor,
            completed=completed_skills,
            records=completed_map,
        )
        logger.info(
            "MemoryManager: resume point — next_skill=%s  cursor=%s  completed=%s",
            point.next_skill, point.cursor, point.completed,
        )
        return point

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_sequence(self, skill: str) -> int:
        """Derive the canonical sequence number for a skill from SKILL_ORDER.

        Falls back to max(existing) + 1 for unknown skills.
        """
        if skill in SKILL_ORDER:
            return SKILL_ORDER.index(skill) + 1
        existing = list(self.job_dir.glob("*.json"))
        if not existing:
            return len(SKILL_ORDER) + 1
        nums = []
        for p in existing:
            m = re.match(r"^(\d+)_", p.name)
            if m:
                nums.append(int(m.group(1)))
        return (max(nums) + 1) if nums else len(SKILL_ORDER) + 1


# ---------------------------------------------------------------------------
# Resume point (returned to pipeline runner)
# ---------------------------------------------------------------------------

@dataclass
class ResumePoint:
    """Describes where the pipeline should restart after a session break.

    Attributes
    ----------
    job_id       : The job being resumed.
    next_skill   : Skill to run next; None means the job is complete.
    cursor       : Timecode to pass as the starting position.
    completed    : Ordered list of skills already successfully executed.
    records      : Mapping of skill name -> its most recent successful SkillRecord.
                   Pipeline uses this to resolve output_paths without re-running.
    """

    job_id: str
    next_skill: Optional[str]
    cursor: str
    completed: list[str]
    records: dict[str, SkillRecord]

    @property
    def is_complete(self) -> bool:
        return self.next_skill is None

    def prior_output(self, skill: str) -> Optional[str]:
        """Return the output_path of a previously completed skill, or None."""
        rec = self.records.get(skill)
        return rec.output_path if rec else None
