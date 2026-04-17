"""
src/pipeline/context_manager.py

Context Firewall for VoxEdit AI.

Every skill receives exactly the context slice it needs — nothing more.
This module is the single choke-point between the full job state stored in
harness/memory/ and the stateless skills in skills/.

Design principle: Context Rot prevention
-----------------------------------------
Passing the full accumulated history into each skill causes two failure modes:
  1. Skills make decisions on stale data from earlier stages.
  2. LLM-backed skills drift toward earlier instructions as context grows.

The firewall enforces a per-skill allow-list at the *data* level (not just
at the prompt level), so the corruption cannot occur even if the orchestrator
is re-written.

Skill context contracts
-----------------------
  Transcriber : { source_audio_path, language_hint }
  Cutter      : { transcript_window (±30s), style_rules, cursor }
  Designer    : { timeline_segment, brand_kit, cursor }
  Exporter    : { annotated_timeline_path, export_profile }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SPEC_ROOT = Path(__file__).resolve().parents[2] / "spec"
MEMORY_JOBS_ROOT = Path(__file__).resolve().parents[2] / "harness" / "memory" / "jobs"

# Half-width of the Cutter transcript window in seconds
CUTTER_WINDOW_S: float = 30.0


# ---------------------------------------------------------------------------
# Timecode helpers
# ---------------------------------------------------------------------------

def _tc_to_seconds(tc: str) -> float:
    """Convert HH:MM:SS[.mmm] to total seconds."""
    tc = tc.strip()
    parts = tc.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def _seconds_to_tc(secs: float) -> str:
    """Convert total seconds to HH:MM:SS.mmm."""
    secs = max(0.0, secs)
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Per-skill context dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TranscriberContext:
    """Minimum context needed by the Transcriber skill."""
    source_audio_path: str
    language_hint: str = "ko"
    # Withheld: job config, handover records, style spec


@dataclass
class CutterContext:
    """Minimum context needed by the Cutter skill.

    transcript_window contains only the words whose midpoint falls within
    [cursor - 30s, cursor + 30s].  The full transcript is never passed.
    """
    cursor: str                           # current edit position (HH:MM:SS.mmm)
    window_start: str                     # cursor − 30 s (clamped to 0)
    window_end: str                       # cursor + 30 s (clamped to total_duration)
    transcript_window: list[dict]         # [{word, start_ms, end_ms, confidence}]
    silence_threshold_s: float            # from spec/editing_style.md
    min_clip_duration_s: float            # from spec/quality_gates.md
    # Withheld: full transcript, designer config, export profile, prior handover records


@dataclass
class DesignerContext:
    """Minimum context needed by the Designer skill."""
    cursor: str
    timeline_segment: list[dict]          # clips in [cursor, cursor+window]
    brand_kit: dict[str, Any]
    jump_cut_zoom_factor: float           # from spec (1.1)
    audio_duck_db: float                  # from spec (-20.0)
    audio_duck_attack_ms: float           # from spec (150)
    audio_duck_release_ms: float          # from spec (500)
    # Withheld: cutter internals, export profile, full timeline


@dataclass
class ExporterContext:
    """Minimum context needed by the Exporter skill."""
    annotated_timeline_path: str
    export_profile: dict[str, Any]
    # Withheld: all upstream processing history, brand kit internals, transcript


# ---------------------------------------------------------------------------
# Spec loader (cached per process)
# ---------------------------------------------------------------------------

_spec_cache: dict[str, Any] = {}


def _load_spec() -> dict[str, Any]:
    """Return parsed editing rules from spec/editing_style.md.

    Reads a small set of numeric rules; the rest of the markdown is ignored.
    Values are hard-coded here as fallbacks — spec takes precedence when parseable.
    """
    if _spec_cache:
        return _spec_cache

    defaults: dict[str, Any] = {
        "silence_threshold_s": 0.5,
        "min_clip_duration_s": 0.5,
        "jump_cut_zoom_factor": 1.1,
        "audio_duck_db": -20.0,
        "audio_duck_attack_ms": 150.0,
        "audio_duck_release_ms": 500.0,
        "target_loudness_lufs": -14.0,
    }

    spec_path = SPEC_ROOT / "editing_style.md"
    if spec_path.exists():
        text = spec_path.read_text(encoding="utf-8")
        _parse_spec_into(text, defaults)

    _spec_cache.update(defaults)
    logger.debug("Spec loaded: %s", _spec_cache)
    return _spec_cache


def _parse_spec_into(text: str, out: dict[str, Any]) -> None:
    """Extract numeric values from editing_style.md into `out`."""
    import re

    patterns = {
        "silence_threshold_s": r"Detection threshold.*?(\d+\.?\d*)\s*s",
        "jump_cut_zoom_factor": r"(\d+\.\d+)×\s*zoom",
        "audio_duck_db": r"Background music adjustment.*?([−\-]\d+)\s*dB",
        "audio_duck_attack_ms": r"Attack time.*?(\d+)\s*ms",
        "audio_duck_release_ms": r"Release time.*?(\d+)\s*ms",
        "min_clip_duration_s": r"All clips.*?≥\s*(\d+\.?\d*)\s*s",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            raw = m.group(1).replace("−", "-")
            try:
                out[key] = float(raw)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Transcript windowing
# ---------------------------------------------------------------------------

def _slice_transcript(
    transcript_path: str,
    center_s: float,
    half_window_s: float,
    total_duration_s: float,
) -> tuple[list[dict], str, str]:
    """Load a transcript JSON and return only words within the time window.

    Returns
    -------
    (words_in_window, window_start_tc, window_end_tc)
    """
    win_start_s = max(0.0, center_s - half_window_s)
    win_end_s = min(total_duration_s, center_s + half_window_s)

    words_in_window: list[dict] = []

    path = Path(transcript_path)
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            # Support both a bare word list and the canonical transcript.json format
            # { "full_text": ..., "words": [...], "vad_segments": [...], "metadata": {...} }
            all_words: list[dict] = raw if isinstance(raw, list) else raw.get("words", [])
            for w in all_words:
                mid_s = (w.get("start_ms", 0) + w.get("end_ms", 0)) / 2000.0
                if win_start_s <= mid_s <= win_end_s:
                    words_in_window.append(w)
        except Exception as exc:
            logger.error("context_manager: failed to slice transcript — %s", exc)
    else:
        logger.warning("context_manager: transcript not found at %s", transcript_path)

    return (
        words_in_window,
        _seconds_to_tc(win_start_s),
        _seconds_to_tc(win_end_s),
    )


# ---------------------------------------------------------------------------
# Timeline segment windowing
# ---------------------------------------------------------------------------

def _slice_timeline(
    timeline_path: str,
    center_s: float,
    half_window_s: float,
    total_duration_s: float,
) -> list[dict]:
    """Return timeline clips whose midpoint falls within the window."""
    win_start_s = max(0.0, center_s - half_window_s)
    win_end_s = min(total_duration_s, center_s + half_window_s)

    path = Path(timeline_path)
    if not path.exists():
        logger.warning("context_manager: timeline not found at %s", timeline_path)
        return []

    try:
        clips: list[dict] = json.loads(path.read_text())
        return [
            c for c in clips
            if win_start_s
            <= (_tc_to_seconds(c.get("start", "0")) + _tc_to_seconds(c.get("end", "0"))) / 2
            <= win_end_s
        ]
    except Exception as exc:
        logger.error("context_manager: failed to slice timeline — %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_skill_context(
    skill_name: str,
    job_id: str,
    *,
    cursor: str = "00:00:00.000",
    total_duration: str = "00:00:00.000",
    source_file: str = "",
    transcript_path: str = "",
    timeline_path: str = "",
    export_profile: Optional[dict] = None,
    brand_kit: Optional[dict] = None,
    language_hint: str = "ko",
) -> TranscriberContext | CutterContext | DesignerContext | ExporterContext:
    """Build a skill-specific context object — the Context Firewall entry point.

    The pipeline runner calls this function once per skill invocation.
    It reads only what is necessary for the named skill and returns a typed
    context object whose fields cannot carry more than the allow-list permits.

    Parameters
    ----------
    skill_name       : One of "transcriber", "cutter", "designer", "exporter".
    job_id           : Identifies the job's memory directory.
    cursor           : Current edit position as HH:MM:SS[.mmm].
    total_duration   : Full video duration as HH:MM:SS[.mmm].
    source_file      : Path to the raw source video/audio (Transcriber only).
    transcript_path  : Path to transcript.json (Cutter only).
    timeline_path    : Path to annotated timeline JSON (Designer, Exporter).
    export_profile   : Dict of export settings (Exporter only).
    brand_kit        : Dict of brand assets (Designer only).
    language_hint    : Language code for transcription (Transcriber only).

    Returns
    -------
    A typed context dataclass for the requested skill.

    Raises
    ------
    ValueError if skill_name is not recognised.
    """
    spec = _load_spec()
    cursor_s = _tc_to_seconds(cursor)
    total_s = _tc_to_seconds(total_duration) or 1.0  # guard against zero

    if skill_name == "transcriber":
        return TranscriberContext(
            source_audio_path=source_file,
            language_hint=language_hint,
        )

    elif skill_name == "cutter":
        words, win_start, win_end = _slice_transcript(
            transcript_path, cursor_s, CUTTER_WINDOW_S, total_s
        )
        ctx = CutterContext(
            cursor=cursor,
            window_start=win_start,
            window_end=win_end,
            transcript_window=words,
            silence_threshold_s=spec["silence_threshold_s"],
            min_clip_duration_s=spec["min_clip_duration_s"],
        )
        logger.info(
            "Context firewall [cutter]: window %s–%s, %d words injected "
            "(full transcript withheld)",
            win_start, win_end, len(words),
        )
        return ctx

    elif skill_name == "designer":
        segment = _slice_timeline(timeline_path, cursor_s, CUTTER_WINDOW_S, total_s)
        ctx = DesignerContext(
            cursor=cursor,
            timeline_segment=segment,
            brand_kit=brand_kit or {},
            jump_cut_zoom_factor=spec["jump_cut_zoom_factor"],
            audio_duck_db=spec["audio_duck_db"],
            audio_duck_attack_ms=spec["audio_duck_attack_ms"],
            audio_duck_release_ms=spec["audio_duck_release_ms"],
        )
        logger.info(
            "Context firewall [designer]: %d timeline clips injected "
            "(cutter internals + export profile withheld)",
            len(segment),
        )
        return ctx

    elif skill_name == "exporter":
        ctx = ExporterContext(
            annotated_timeline_path=timeline_path,
            export_profile=export_profile or {"platform": "youtube", "resolution": "1080p"},
        )
        logger.info(
            "Context firewall [exporter]: timeline path + export profile injected "
            "(all upstream processing history withheld)"
        )
        return ctx

    else:
        raise ValueError(
            f"build_skill_context: unknown skill '{skill_name}'. "
            f"Valid skills: transcriber, cutter, designer, exporter"
        )
