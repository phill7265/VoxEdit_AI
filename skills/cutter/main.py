"""
skills/cutter/main.py

Harness integration layer for the Cutter skill.

Responsibilities
----------------
1. Locate the Transcriber Handover Record via MemoryManager.
2. Load transcript.json from the path recorded in that handover.
3. Build a CutterContext (±30 s window) via src/pipeline/context_manager.
4. Run skills/cutter/logic.run_cutter() — transcript-based silence and jump-cut.
5. Write cut_list.json to the staging area.
6. Write a SkillRecord (Handover Note) to harness/memory via MemoryManager.
7. Return the SkillRecord so the pipeline runner can proceed to Designer.

Stateless guarantee
-------------------
This module holds no instance state between calls.
All persistent state lives in harness/memory/jobs/{job_id}/.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Optional

from harness.memory.manager import MemoryManager, SkillRecord
from skills.cutter.logic import CutSegment, CutterResult, run_cutter
from src.pipeline.context_manager import (
    CutterContext,
    _tc_to_seconds,
    build_skill_context,
)

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────
STAGING_ROOT: Path = Path(__file__).resolve().parents[2] / "staging"


# ── Helpers ───────────────────────────────────────────────────────────────

def _seconds_to_tc(secs: float) -> str:
    secs = max(0.0, secs)
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _load_transcript(transcript_path: str) -> tuple[list[dict], float]:
    """Read transcript.json and return (words, duration_s).

    Returns ([], 0.0) on any read/parse error.
    """
    path = Path(transcript_path)
    if not path.exists():
        logger.error("Cutter: transcript not found at %s", transcript_path)
        return [], 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        words: list[dict] = data.get("words", [])
        duration_s: float = data.get("metadata", {}).get("duration_s", 0.0)
        return words, duration_s
    except Exception as exc:
        logger.error("Cutter: failed to parse transcript — %s", exc)
        return [], 0.0


def _write_cut_list(
    result: CutterResult,
    output_path: Path,
    window_start_s: float,
    window_end_s: float,
) -> None:
    """Serialise the cut segment list to the canonical cut_list.json format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cut_segments": [seg.to_dict() for seg in result.segments],
        "metadata": {
            "window_start_s": round(window_start_s, 3),
            "window_end_s": round(window_end_s, 3),
            "keep_count": sum(1 for s in result.segments if s.action == "keep"),
            "delete_count": sum(1 for s in result.segments if s.action == "delete"),
            "sensor_flags": result.sensor_flags,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "Cut list written: %s (%d segments)",
        output_path.name,
        len(result.segments),
    )


# ── Public entry point ────────────────────────────────────────────────────

def run(
    *,
    job_id: str,
    staging_dir: Optional[Path] = None,
    retry_index: int = 0,
    cursor: str = "00:00:00.000",
    total_duration: str = "00:00:00.000",
    transcript_path: Optional[str] = None,
) -> SkillRecord:
    """Execute the Cutter skill and commit a Handover Record.

    Parameters
    ----------
    job_id           : Pipeline job identifier (used for memory path).
    staging_dir      : Override staging directory (used in tests).
    retry_index      : Incremented by the pipeline on retry runs.
    cursor           : Current edit position as HH:MM:SS[.mmm].
    total_duration   : Full video duration as HH:MM:SS[.mmm].
    transcript_path  : Direct path to transcript.json; skips MemoryManager
                       lookup when provided (useful for tests).

    Returns
    -------
    SkillRecord committed to harness memory.
    """
    staging = staging_dir or (STAGING_ROOT / job_id)
    output_path = staging / "cut_list.json"

    mgr = MemoryManager(job_id)

    # ── Resolve transcript path ───────────────────────────────────────────
    resolved_transcript = transcript_path
    if resolved_transcript is None:
        resume = mgr.find_resume_point()
        resolved_transcript = resume.prior_output("transcriber")
        if resolved_transcript is None:
            error_msg = (
                f"Cutter: no transcriber output found in memory for job '{job_id}'. "
                "Run the Transcriber skill first."
            )
            logger.error(error_msg)
            failed_record = SkillRecord(
                job_id=job_id,
                skill="cutter",
                status="failed",
                output_path="",
                cursor_start=cursor,
                cursor_end=cursor,
                error=error_msg,
                retry_index=retry_index,
            )
            mgr.write(failed_record)
            return failed_record

    # ── Load transcript ───────────────────────────────────────────────────
    try:
        all_words, duration_s = _load_transcript(resolved_transcript)
    except Exception as exc:
        error_detail = traceback.format_exc()
        logger.error("Cutter: transcript load failed — %s", exc)
        failed_record = SkillRecord(
            job_id=job_id,
            skill="cutter",
            status="failed",
            output_path="",
            cursor_start=cursor,
            cursor_end=cursor,
            error=str(exc),
            payload={"traceback": error_detail},
            retry_index=retry_index,
        )
        mgr.write(failed_record)
        return failed_record

    # ── Build context (Context Firewall) ──────────────────────────────────
    effective_duration = total_duration
    if _tc_to_seconds(total_duration) == 0.0 and duration_s > 0:
        effective_duration = _seconds_to_tc(duration_s)

    try:
        ctx: CutterContext = build_skill_context(  # type: ignore[assignment]
            "cutter",
            job_id,
            cursor=cursor,
            total_duration=effective_duration,
            transcript_path=resolved_transcript,
        )
    except Exception as exc:
        error_detail = traceback.format_exc()
        logger.error("Cutter: context build failed — %s", exc)
        failed_record = SkillRecord(
            job_id=job_id,
            skill="cutter",
            status="failed",
            output_path="",
            cursor_start=cursor,
            cursor_end=cursor,
            error=str(exc),
            payload={"traceback": error_detail},
            retry_index=retry_index,
        )
        mgr.write(failed_record)
        return failed_record

    window_start_s = _tc_to_seconds(ctx.window_start)
    window_end_s = _tc_to_seconds(ctx.window_end)

    # ── Execute cutter logic ──────────────────────────────────────────────
    try:
        result: CutterResult = run_cutter(
            ctx.transcript_window,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            threshold_s=ctx.silence_threshold_s,
        )
    except Exception as exc:
        error_detail = traceback.format_exc()
        logger.error("Cutter: logic failed — %s", exc)
        failed_record = SkillRecord(
            job_id=job_id,
            skill="cutter",
            status="failed",
            output_path="",
            cursor_start=cursor,
            cursor_end=ctx.window_end,
            error=str(exc),
            payload={"traceback": error_detail},
            retry_index=retry_index,
        )
        mgr.write(failed_record)
        return failed_record

    # ── Write cut_list.json to staging ────────────────────────────────────
    _write_cut_list(result, output_path, window_start_s, window_end_s)

    # ── Build Handover Record ─────────────────────────────────────────────
    keep_segs = [s for s in result.segments if s.action == "keep"]
    delete_segs = [s for s in result.segments if s.action == "delete"]
    jump_cuts = [s for s in keep_segs if "jump_cut_zoom_1.1" in s.effects]

    record = SkillRecord(
        job_id=job_id,
        skill="cutter",
        status="success",
        output_path=str(output_path),
        cursor_start=cursor,
        cursor_end=ctx.window_end,
        payload={
            "output": {
                "cut_segments": [seg.to_dict() for seg in result.segments],
            },
            "metadata": {
                "window_start_s": round(window_start_s, 3),
                "window_end_s": round(window_end_s, 3),
                "keep_count": len(keep_segs),
                "delete_count": len(delete_segs),
                "jump_cut_count": len(jump_cuts),
                "sensor_flags": result.sensor_flags,
            },
        },
        retry_index=retry_index,
    )

    mgr.write(record)
    logger.info(
        "Cutter handover committed — job=%s  keep=%d  delete=%d  jump_cuts=%d",
        job_id, len(keep_segs), len(delete_segs), len(jump_cuts),
    )
    return record
