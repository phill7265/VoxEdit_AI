"""
skills/transcriber/main.py

Harness integration layer for the Transcriber skill.

Responsibilities
----------------
1. Accept a TranscriberContext (from src/pipeline/context_manager).
2. Call skills/transcriber/logic.py — the only place that touches audio files.
3. Write the transcript JSON to the staging area.
4. Write a SkillRecord (Handover Note) to harness/memory via MemoryManager.
5. Return the SkillRecord so the pipeline runner can proceed.

Stateless guarantee
-------------------
This module holds no instance state between calls.
All persistent state lives in harness/memory/jobs/{job_id}/.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from harness.memory.manager import MemoryManager, SkillRecord
from skills.transcriber.logic import (
    TranscribeResult,
    WhisperBackend,
    transcribe,
)

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_MODEL: str = "base"
STAGING_ROOT: Path = Path(__file__).resolve().parents[2] / "staging"


# ── Helpers ───────────────────────────────────────────────────────────────

def _seconds_to_tc(secs: float) -> str:
    secs = max(0.0, secs)
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _write_transcript_json(result: TranscribeResult, output_path: Path) -> None:
    """Serialise the transcript to the canonical JSON format consumed by Cutter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "full_text": result.full_text,
        "words": result.words_as_dicts(),
        "vad_segments": result.vad_as_dicts(),
        "metadata": {
            "model": result.model_name,
            "language": result.language,
            "duration_s": round(result.duration_s, 3),
            "word_count": result.word_count,
            "avg_confidence": round(result.avg_confidence, 4),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Transcript written: %s (%d words)", output_path.name, result.word_count)


# ── Public entry point ────────────────────────────────────────────────────

def run(
    *,
    job_id: str,
    source_file: str,
    language: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    staging_dir: Optional[Path] = None,
    retry_index: int = 0,
    backend: Optional[WhisperBackend] = None,
) -> SkillRecord:
    """Execute the Transcriber skill and commit a Handover Record.

    Parameters
    ----------
    job_id       : Pipeline job identifier (used for memory path).
    source_file  : Path to the raw audio / video file.
    language     : ISO-639-1 language hint (None = auto-detect).
    model_name   : Whisper model size.
    staging_dir  : Override staging directory (used in tests).
    retry_index  : Incremented by the pipeline on retry runs.
    backend      : WhisperBackend override (injected in tests).

    Returns
    -------
    SkillRecord committed to harness memory.
    The caller (pipeline runner) uses record.output_path to locate the
    transcript for the next skill (Cutter).
    """
    staging = staging_dir or (STAGING_ROOT / job_id)
    output_path = staging / "transcript.json"

    mgr = MemoryManager(job_id)

    # ── Execute ───────────────────────────────────────────────────────────
    try:
        result: TranscribeResult = transcribe(
            source_file,
            language=language,
            model_name=model_name,
            backend=backend,
        )
    except Exception as exc:
        error_detail = traceback.format_exc()
        logger.error("Transcriber failed: %s", exc)

        failed_record = SkillRecord(
            job_id=job_id,
            skill="transcriber",
            status="failed",
            output_path="",
            cursor_start="00:00:00.000",
            cursor_end="00:00:00.000",
            error=str(exc),
            payload={"traceback": error_detail},
            retry_index=retry_index,
        )
        mgr.write(failed_record)
        return failed_record

    # ── Write transcript JSON to staging ──────────────────────────────────
    _write_transcript_json(result, output_path)

    # ── Build Handover Record ─────────────────────────────────────────────
    #
    # Payload format matches the spec in docs/architecture.md §Handover Notes:
    #   { status, output_path, word_count, duration_ms }
    # Extended here with the full output block for downstream skills.
    #
    record = SkillRecord(
        job_id=job_id,
        skill="transcriber",
        status="success",
        output_path=str(output_path),
        cursor_start="00:00:00.000",
        cursor_end=_seconds_to_tc(result.duration_s),
        payload={
            "output": {
                "full_text": result.full_text,
                "words": result.words_as_dicts(),
                "vad_segments": result.vad_as_dicts(),
            },
            "metadata": {
                "model": result.model_name,
                "language": result.language,
                "duration": round(result.duration_s, 3),
                "word_count": result.word_count,
                "avg_confidence": round(result.avg_confidence, 4),
            },
        },
        retry_index=retry_index,
    )

    mgr.write(record)
    logger.info(
        "Transcriber handover committed — job=%s  words=%d  duration=%.1fs",
        job_id, result.word_count, result.duration_s,
    )
    return record
