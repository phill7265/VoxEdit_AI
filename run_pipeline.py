"""
run_pipeline.py — End-to-end pipeline runner for VoxEdit AI.

Usage:
    python run_pipeline.py

Runs Transcriber → Cutter → Designer → Exporter on row_video.mp4.
After each skill, prints the Handover Note from MemoryManager.
After render, probes the output file and runs quality gates.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

# ── Paths ────────────────────────────────────────────────────────────────
SOURCE_FILE = r"C:\Users\rearl\Documents\work\sales\VoxEdit_AI\tests\Input\raw_video1.mp4"
JOB_ID = f"job_{time.strftime('%Y%m%d_%H%M%S')}"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STAGING_DIR = ROOT / "staging" / JOB_ID


# ── Helpers ───────────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _print_handover(job_id: str, skill: str) -> None:
    """Read and pretty-print the SkillRecord written by a skill."""
    from harness.memory.manager import MemoryManager
    mgr = MemoryManager(job_id)
    resume = mgr.find_resume_point()
    rec = resume.records.get(skill)
    if rec is None:
        print(f"  [!] No SkillRecord found for '{skill}'")
        return
    print(f"  skill    : {rec.skill}")
    print(f"  status   : {rec.status}")
    print(f"  cursor   : {rec.cursor_start} → {rec.cursor_end}")
    print(f"  output   : {rec.output_path}")
    if rec.error:
        print(f"  ERROR    : {rec.error}")
    if rec.payload:
        meta = rec.payload.get("metadata", {})
        if meta:
            for k, v in meta.items():
                if k == "quality_gates":
                    qg = v
                    print(f"  quality  : {qg.get('summary','')}")
                    for g in qg.get("gates", []):
                        print(f"             {g}")
                elif k == "ffmpeg_command":
                    print(f"  cmd      : {str(v)[:120]}…")
                else:
                    print(f"  {k:<12}: {v}")


def _ffprobe_output(path: str) -> dict:
    """Return ffprobe stream info for the rendered output file."""
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except Exception as exc:
        return {"error": str(exc)}


def _quality_gate_report(output_path: str, profile_w: int, profile_h: int) -> None:
    """Run export_validator + ffprobe-based analysis on the rendered file."""
    from harness.sensors import export_validator

    _sep("QUALITY GATES")

    # ── ffprobe: measure actual rendered dimensions ───────────────────────
    probe = _ffprobe_output(output_path)
    actual_w, actual_h = profile_w, profile_h
    actual_duration = None
    actual_audio_channels = None

    if probe and "streams" in probe:
        for stream in probe["streams"]:
            if stream.get("codec_type") == "video":
                actual_w = stream.get("width", profile_w)
                actual_h = stream.get("height", profile_h)
                dur = stream.get("duration")
                if dur:
                    actual_duration = float(dur)
            elif stream.get("codec_type") == "audio":
                actual_audio_channels = stream.get("channels")
    if actual_duration is None and probe and "format" in probe:
        dur = probe["format"].get("duration")
        if dur:
            actual_duration = float(dur)

    dur_str = f"{actual_duration:.1f}s" if actual_duration is not None else "unknown"
    print(f"  ffprobe  : {actual_w}×{actual_h}  "
          f"duration={dur_str}  "
          f"audio_ch={actual_audio_channels}")

    # ── Aspect ratio gate (ffprobe-measured dimensions) ───────────────────
    report = export_validator.validate(
        width=actual_w,
        height=actual_h,
        output_path=output_path,
    )

    print(f"\n  {report.summary}")
    for gate in report.gates:
        print(f"  {gate}")

    if not report.passed:
        print("\n  [!] One or more quality gates FAILED — see detail above.")
    else:
        print("\n  [✓] All gates passed — output is ready for upload.")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    from src.pipeline.orchestrator import WorkflowManager, SKILL_ORDER

    _sep(f"VoxEdit AI Pipeline  |  job={JOB_ID}")
    print(f"  source   : {SOURCE_FILE}")
    print(f"  staging  : {STAGING_DIR}")
    print(f"  dry_run  : False (real render)")

    wm = WorkflowManager(
        job_id=JOB_ID,
        source_file=SOURCE_FILE,
        staging_dir=STAGING_DIR,
        model_name="base",
        dry_run=False,
    )

    # ── Run pipeline, monitoring each skill ───────────────────────────────
    from harness.memory.manager import MemoryManager, SKILL_ORDER as SKILLS

    t0 = time.monotonic()
    result = wm.run()
    elapsed = time.monotonic() - t0

    # ── Print handover notes ───────────────────────────────────────────────
    for skill in SKILLS:
        if skill in result.records or skill in result.skipped:
            _sep(f"Handover Note — {skill.upper()}")
            _print_handover(JOB_ID, skill)

    # ── Pipeline summary ──────────────────────────────────────────────────
    _sep("PIPELINE SUMMARY")
    print(f"  status    : {result.status.upper()}")
    print(f"  completed : {result.completed}")
    print(f"  skipped   : {result.skipped}")
    if result.failed_skill:
        print(f"  failed at : {result.failed_skill}")
        if result.final_record and result.final_record.error:
            print(f"  error     : {result.final_record.error}")
        print(f"\n  Resume possible: run the same script again — "
              f"MemoryManager will skip {result.completed} and restart from '{result.failed_skill}'.")
        return

    print(f"  elapsed   : {elapsed:.1f}s")

    # ── Quality gates on actual output ────────────────────────────────────
    output_mp4 = str(STAGING_DIR / "output.mp4")
    if Path(output_mp4).exists():
        _quality_gate_report(output_mp4, profile_w=1080, profile_h=1920)
    else:
        print(f"\n  [!] output.mp4 not found at {output_mp4}")

    _sep("DONE")
    print(f"  Output: {output_mp4}")
    print(f"  Memory: {ROOT / 'harness' / 'memory' / 'jobs' / JOB_ID}")


if __name__ == "__main__":
    main()
