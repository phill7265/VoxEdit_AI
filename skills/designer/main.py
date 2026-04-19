"""
skills/designer/main.py

Harness integration layer for the Designer skill.

Responsibilities
----------------
1. Locate the Cutter Handover Record via MemoryManager to get cut_list.json.
2. Locate the Transcriber Handover Record to get transcript.json (words + VAD).
3. Run skills/designer/logic.run_designer() — captions, zoom, duck events.
4. Write annotated_timeline.json to the staging area.
5. Write a SkillRecord (Handover Note) to harness/memory via MemoryManager.
6. Return the SkillRecord so the pipeline runner can proceed to Exporter.

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
from skills.designer.logic import (
    DEFAULT_KEYWORDS,
    DesignerResult,
    run_designer,
)

_BROLL_REQUESTS_FILE = Path(__file__).resolve().parents[2] / "spec" / "broll_requests.json"

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


def _load_cut_list(path: str) -> list[dict]:
    """Load cut_segments from cut_list.json.  Returns [] on failure."""
    p = Path(path)
    if not p.exists():
        logger.error("Designer: cut_list not found at %s", path)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("cut_segments", [])
    except Exception as exc:
        logger.error("Designer: failed to parse cut_list — %s", exc)
        return []


def _load_transcript(path: str) -> tuple[list[dict], list[dict], float]:
    """Load (words, vad_segments, duration_s) from transcript.json.

    Returns ([], [], 0.0) on failure.
    """
    p = Path(path)
    if not p.exists():
        logger.error("Designer: transcript not found at %s", path)
        return [], [], 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        words: list[dict] = data.get("words", [])
        vad_segs: list[dict] = data.get("vad_segments", [])
        duration_s: float = data.get("metadata", {}).get("duration_s", 0.0)
        return words, vad_segs, duration_s
    except Exception as exc:
        logger.error("Designer: failed to parse transcript — %s", exc)
        return [], [], 0.0


def _write_annotated_timeline(
    result: DesignerResult,
    output_path: Path,
) -> None:
    """Serialise the designer result to annotated_timeline.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "visual_elements": [e.to_dict() for e in result.visual_elements],
        "metadata": {
            "caption_count": len(result.captions),
            "zoom_count": len(result.zooms),
            "duck_event_count": len(result.duck_events),
            "highlight_count": len(result.highlights),
            "overlay_count": len(result.overlays),
            "sensor_flags": result.sensor_flags,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "Annotated timeline written: %s (%d elements)",
        output_path.name,
        len(result.visual_elements),
    )


# ── Public entry point ────────────────────────────────────────────────────

def run(
    *,
    job_id: str,
    staging_dir: Optional[Path] = None,
    retry_index: int = 0,
    cursor: str = "00:00:00.000",
    keywords: Optional[frozenset] = None,
    # Test overrides — skip MemoryManager lookup when provided
    cut_list_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
) -> SkillRecord:
    """Execute the Designer skill and commit a Handover Record.

    Parameters
    ----------
    job_id           : Pipeline job identifier (used for memory path).
    staging_dir      : Override staging directory (used in tests).
    retry_index      : Incremented by the pipeline on retry runs.
    cursor           : Current edit position as HH:MM:SS[.mmm].
    keywords         : Override keyword highlight set (default: DEFAULT_KEYWORDS).
    cut_list_path    : Direct path to cut_list.json; skips MemoryManager lookup.
    transcript_path  : Direct path to transcript.json; skips MemoryManager lookup.

    Returns
    -------
    SkillRecord committed to harness memory.
    """
    staging = staging_dir or (STAGING_ROOT / job_id)
    output_path = staging / "annotated_timeline.json"
    kw = keywords if keywords is not None else DEFAULT_KEYWORDS

    mgr = MemoryManager(job_id)

    # ── Resolve cut_list path ─────────────────────────────────────────────
    resolved_cut_list = cut_list_path
    if resolved_cut_list is None:
        resume = mgr.find_resume_point()
        resolved_cut_list = resume.prior_output("cutter")
        if resolved_cut_list is None:
            error_msg = (
                f"Designer: no cutter output found in memory for job '{job_id}'. "
                "Run the Cutter skill first."
            )
            logger.error(error_msg)
            failed = SkillRecord(
                job_id=job_id,
                skill="designer",
                status="failed",
                output_path="",
                cursor_start=cursor,
                cursor_end=cursor,
                error=error_msg,
                retry_index=retry_index,
            )
            mgr.write(failed)
            return failed

    # ── Resolve transcript path ───────────────────────────────────────────
    resolved_transcript = transcript_path
    if resolved_transcript is None:
        resume = mgr.find_resume_point()
        resolved_transcript = resume.prior_output("transcriber")
        # Transcript is optional for designer; log a warning if missing
        if resolved_transcript is None:
            logger.warning(
                "Designer: no transcriber output for job '%s' — "
                "captions and VAD ducking will be skipped.", job_id
            )

    # ── Load inputs ───────────────────────────────────────────────────────
    try:
        cut_segments = _load_cut_list(resolved_cut_list)
    except Exception as exc:
        error_detail = traceback.format_exc()
        logger.error("Designer: cut_list load failed — %s", exc)
        failed = SkillRecord(
            job_id=job_id,
            skill="designer",
            status="failed",
            output_path="",
            cursor_start=cursor,
            cursor_end=cursor,
            error=str(exc),
            payload={"traceback": error_detail},
            retry_index=retry_index,
        )
        mgr.write(failed)
        return failed

    words: list[dict] = []
    vad_segments: list[dict] = []
    duration_s: float = 0.0

    if resolved_transcript:
        try:
            words, vad_segments, duration_s = _load_transcript(resolved_transcript)
        except Exception as exc:
            logger.warning("Designer: transcript load failed (%s) — continuing without captions", exc)

    # ── Load b-roll requests ──────────────────────────────────────────────
    broll_requests: list[dict] = []
    if _BROLL_REQUESTS_FILE.exists():
        try:
            broll_requests = json.loads(_BROLL_REQUESTS_FILE.read_text(encoding="utf-8"))
            logger.info("Designer: loaded %d b-roll request(s)", len(broll_requests))
        except Exception as exc:
            logger.warning("Designer: failed to load broll_requests.json — %s", exc)

    # ── Execute designer logic ────────────────────────────────────────────
    try:
        result: DesignerResult = run_designer(
            words,
            cut_segments,
            vad_segments,
            keywords=kw,
            broll_requests=broll_requests if broll_requests else None,
        )
    except Exception as exc:
        error_detail = traceback.format_exc()
        logger.error("Designer: logic failed — %s", exc)
        failed = SkillRecord(
            job_id=job_id,
            skill="designer",
            status="failed",
            output_path="",
            cursor_start=cursor,
            cursor_end=cursor,
            error=str(exc),
            payload={"traceback": error_detail},
            retry_index=retry_index,
        )
        mgr.write(failed)
        return failed

    # ── Write annotated_timeline.json ─────────────────────────────────────
    _write_annotated_timeline(result, output_path)

    # ── Compute cursor_end ────────────────────────────────────────────────
    # Advance cursor to end of the last visual element, or keep at duration
    cursor_end_s = duration_s
    if result.visual_elements:
        cursor_end_s = max(cursor_end_s, result.visual_elements[-1].end)

    # ── Build Handover Record ─────────────────────────────────────────────
    record = SkillRecord(
        job_id=job_id,
        skill="designer",
        status="success",
        output_path=str(output_path),
        cursor_start=cursor,
        cursor_end=_seconds_to_tc(cursor_end_s),
        payload={
            "output": {
                "visual_elements": [e.to_dict() for e in result.visual_elements],
            },
            "metadata": {
                "caption_count": len(result.captions),
                "zoom_count": len(result.zooms),
                "duck_event_count": len(result.duck_events),
                "highlight_count": len(result.highlights),
                "overlay_count": len(result.overlays),
                "broll_count": len(result.brolls),
                "sensor_flags": result.sensor_flags,
            },
        },
        retry_index=retry_index,
    )

    mgr.write(record)
    logger.info(
        "Designer handover committed — job=%s  captions=%d  zooms=%d  "
        "ducks=%d  highlights=%d  brolls=%d",
        job_id,
        len(result.captions),
        len(result.zooms),
        len(result.duck_events),
        len(result.highlights),
        len(result.brolls),
    )
    return record
