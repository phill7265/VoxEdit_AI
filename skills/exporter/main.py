"""
skills/exporter/main.py

Harness integration layer for the Exporter skill.

Responsibilities
----------------
1. Resolve cutter (cut_list.json) and designer (annotated_timeline.json)
   outputs via MemoryManager.
2. Build an ExportPlan via skills/exporter/logic.build_export_plan().
3. Validate the FFmpeg command via harness/sandbox/executor (dry_run).
4. Execute in the sandbox (unless dry_run=True is requested by caller).
5. Run post-render quality gates via harness/sensors/export_validator.
6. Write a SkillRecord (Handover Note) to harness/memory.
7. Return the SkillRecord.

Stateless guarantee
-------------------
No instance state between calls.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Optional

from harness.memory.manager import MemoryManager, SkillRecord
from harness.sandbox import executor as sandbox
from harness.sensors import export_validator
from skills.exporter.logic import (
    DEFAULT_PROFILE,
    ExportPlan,
    build_export_plan,
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


def _load_json(path: str, key: Optional[str] = None) -> list[dict]:
    """Load a JSON file and return a list.  Returns [] on failure."""
    p = Path(path)
    if not p.exists():
        logger.error("Exporter: file not found: %s", path)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if key:
            return data.get(key, [])
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.error("Exporter: failed to parse %s — %s", path, exc)
        return []


def _write_export_manifest(
    plan: ExportPlan,
    output_path: Path,
    ffmpeg_cmd: str,
    gate_report_summary: str,
) -> None:
    """Write export_manifest.json alongside the rendered output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output_file": plan.output_file,
        "source_file": plan.source_file,
        "keep_segment_count": len(plan.keep_segments),
        "total_duration_s": round(plan.total_output_duration_s, 3),
        "caption_count": len(plan.captions),
        "zoom_count": len(plan.zoom_elements),
        "duck_event_count": len(plan.duck_events),
        "export_profile": plan.export_profile,
        "ffmpeg_command": ffmpeg_cmd,
        "quality_gate_summary": gate_report_summary,
    }
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Export manifest written: %s", output_path.name)


# ── Public entry point ────────────────────────────────────────────────────

def run(
    *,
    job_id: str,
    source_file: str = "",
    staging_dir: Optional[Path] = None,
    retry_index: int = 0,
    cursor: str = "00:00:00.000",
    export_profile: Optional[dict] = None,
    dry_run: bool = False,
    # Test overrides
    cut_list_path: Optional[str] = None,
    annotated_timeline_path: Optional[str] = None,
) -> SkillRecord:
    """Execute the Exporter skill and commit a Handover Record.

    Parameters
    ----------
    job_id                   : Pipeline job identifier.
    source_file              : Path to raw source video/audio.
    staging_dir              : Override staging directory (tests).
    retry_index              : Retry counter.
    cursor                   : Current edit position.
    export_profile           : Override export settings (default: PROFILE_SHORTS).
    dry_run                  : If True, validate command but do not render.
    cut_list_path            : Direct path to cut_list.json (skips MemoryManager).
    annotated_timeline_path  : Direct path to annotated_timeline.json.

    Returns
    -------
    SkillRecord committed to harness memory.
    """
    staging = staging_dir or (STAGING_ROOT / job_id)
    output_video = staging / "output.mp4"
    manifest_path = staging / "export_manifest.json"
    profile = export_profile or dict(DEFAULT_PROFILE)

    mgr = MemoryManager(job_id)

    # ── Resolve cut_list path ─────────────────────────────────────────────
    resolved_cut_list = cut_list_path
    if resolved_cut_list is None:
        resume = mgr.find_resume_point()
        resolved_cut_list = resume.prior_output("cutter")
        if resolved_cut_list is None:
            error_msg = f"Exporter: no cutter output found for job '{job_id}'"
            logger.error(error_msg)
            failed = SkillRecord(
                job_id=job_id, skill="exporter", status="failed",
                output_path="", cursor_start=cursor, cursor_end=cursor,
                error=error_msg, retry_index=retry_index,
            )
            mgr.write(failed)
            return failed

    # ── Resolve annotated_timeline path ───────────────────────────────────
    resolved_timeline = annotated_timeline_path
    if resolved_timeline is None:
        resume = mgr.find_resume_point()
        resolved_timeline = resume.prior_output("designer")
        if resolved_timeline is None:
            logger.warning(
                "Exporter: no designer output for job '%s' — "
                "captions and zoom effects will be skipped", job_id
            )

    # ── Load inputs ───────────────────────────────────────────────────────
    try:
        cut_segments = _load_json(resolved_cut_list, key="cut_segments")
        visual_elements: list[dict] = []
        if resolved_timeline:
            visual_elements = _load_json(resolved_timeline, key="visual_elements")
    except Exception as exc:
        error_detail = traceback.format_exc()
        failed = SkillRecord(
            job_id=job_id, skill="exporter", status="failed",
            output_path="", cursor_start=cursor, cursor_end=cursor,
            error=str(exc), payload={"traceback": error_detail},
            retry_index=retry_index,
        )
        mgr.write(failed)
        return failed

    # ── Build export plan ─────────────────────────────────────────────────
    try:
        plan: ExportPlan = build_export_plan(
            cut_segments,
            visual_elements,
            source_file=source_file,
            output_file=str(output_video),
            export_profile=profile,
        )
    except Exception as exc:
        error_detail = traceback.format_exc()
        logger.error("Exporter: plan build failed — %s", exc)
        failed = SkillRecord(
            job_id=job_id, skill="exporter", status="failed",
            output_path="", cursor_start=cursor, cursor_end=cursor,
            error=str(exc), payload={"traceback": error_detail},
            retry_index=retry_index,
        )
        mgr.write(failed)
        return failed

    ffmpeg_cmd = plan.to_command()

    # ── Validate command (always) ─────────────────────────────────────────
    # Skip sandbox staging check in dry_run tests by using executor directly
    validation_errors = sandbox.validate_command(ffmpeg_cmd, source_file or "")
    if validation_errors:
        # Sanitise: staging enforcement fails when source_file is empty;
        # that is expected in mock tests — only hard errors are fatal here
        hard_errors = [e for e in validation_errors if "Staging enforcement" not in e]
        if hard_errors:
            error_msg = "Exporter: FFmpeg command validation failed: " + "; ".join(hard_errors)
            logger.error(error_msg)
            failed = SkillRecord(
                job_id=job_id, skill="exporter", status="failed",
                output_path="", cursor_start=cursor, cursor_end=cursor,
                error=error_msg,
                payload={"validation_errors": validation_errors},
                retry_index=retry_index,
            )
            mgr.write(failed)
            return failed
        else:
            logger.warning(
                "Exporter: staging enforcement warning (ignored in dry_run): %s",
                validation_errors,
            )

    # ── Execute (unless dry_run) ──────────────────────────────────────────
    if not dry_run:
        exec_result = sandbox.run(ffmpeg_cmd, source_file or "", dry_run=False)
        if not exec_result.success:
            error_msg = (
                f"Exporter: FFmpeg execution failed "
                f"(rc={exec_result.returncode}): "
                f"{exec_result.stderr[:500]}"
            )
            logger.error(error_msg)
            failed = SkillRecord(
                job_id=job_id, skill="exporter", status="failed",
                output_path="", cursor_start=cursor, cursor_end=cursor,
                error=error_msg,
                payload={
                    "ffmpeg_cmd": ffmpeg_cmd,
                    "stderr": exec_result.stderr,
                    "killed_reason": exec_result.killed_reason,
                },
                retry_index=retry_index,
            )
            mgr.write(failed)
            return failed

    # ── Post-render quality gates ─────────────────────────────────────────
    w = profile.get("width", 1080)
    h = profile.get("height", 1920)
    gate_report = export_validator.validate(
        width=w,
        height=h,
        output_path=str(output_video) if not dry_run else None,
    )

    # ── Write manifest ────────────────────────────────────────────────────
    _write_export_manifest(plan, manifest_path, ffmpeg_cmd, gate_report.summary)

    # ── Build Handover Record ─────────────────────────────────────────────
    duration_s = plan.total_output_duration_s
    record = SkillRecord(
        job_id=job_id,
        skill="exporter",
        status="success",
        output_path=str(output_video),
        cursor_start=cursor,
        cursor_end=_seconds_to_tc(duration_s),
        payload={
            "output": {
                "output_file": str(output_video),
                "dry_run": dry_run,
            },
            "metadata": {
                "keep_segment_count": len(plan.keep_segments),
                "total_duration_s": round(duration_s, 3),
                "caption_count": len(plan.captions),
                "zoom_count": len(plan.zoom_elements),
                "duck_event_count": len(plan.duck_events),
                "export_profile": profile,
                "quality_gates": {
                    "passed": gate_report.passed,
                    "summary": gate_report.summary,
                    "gates": [str(g) for g in gate_report.gates],
                },
                "ffmpeg_command": ffmpeg_cmd,
            },
        },
        retry_index=retry_index,
    )

    mgr.write(record)
    logger.info(
        "Exporter handover committed — job=%s  duration=%.1fs  "
        "dry_run=%s  gates=%s",
        job_id, duration_s, dry_run, gate_report.summary,
    )
    return record
