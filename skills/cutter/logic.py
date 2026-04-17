"""
skills/cutter/logic.py

Cutter skill — Silence Removal and Jump Cut detection.

Spec source: spec/editing_style.md  v0.2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rule §2  Silence Removal
  · Primary: word gap ≥ 0.5 s in transcript → delete segment
  · Secondary: RMS < -40 dBFS sustained ≥ 0.5 s (PCM audio path)
  · Head/tail silence ≥ 0.2 s → always remove (auto_remove)

Rule §3  Jump Cut
  · Trigger: two consecutive keep segments on the same speaker track
             with no B-roll between them
  · Action:  add "jump_cut_zoom_1.1" to the second segment's effects list
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Context Firewall
----------------
The primary entry point ``run_cutter()`` consumes only the
transcript_window injected by src/pipeline/context_manager.py (±30 s),
never the full transcript.

This module is STATELESS.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ── Spec constants (mirrors context_manager._load_spec defaults) ──────────
RMS_SILENCE_DBFS: float = -40.0       # §2  level threshold
SILENCE_MIN_DURATION_S: float = 0.50  # §2  minimum silence duration
HEAD_TAIL_SILENCE_S: float = 0.20     # §2  edge-silence auto-remove threshold
JUMP_CUT_ZOOM: float = 1.1            # §3  zoom factor
MIN_CLIP_DURATION_S: float = 0.50     # §1  quality gate floor

# Derived effect string so it stays in sync with JUMP_CUT_ZOOM
JUMP_CUT_EFFECT: str = f"jump_cut_zoom_{JUMP_CUT_ZOOM}"  # "jump_cut_zoom_1.1"


# ── Helpers ───────────────────────────────────────────────────────────────

def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)

def _linear_to_db(linear: float) -> float:
    if linear <= 0:
        return -math.inf
    return 20.0 * math.log10(linear)

def _rms_db(samples: list[float]) -> float:
    """Return RMS level in dBFS for a list of normalised float samples [-1, 1]."""
    if not samples:
        return -math.inf
    mean_sq = sum(s * s for s in samples) / len(samples)
    return _linear_to_db(math.sqrt(mean_sq))


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class CutSegment:
    """A single timeline segment produced by the transcript-based Cutter path.

    Attributes
    ----------
    start_s    : Segment start in seconds.
    end_s      : Segment end in seconds.
    action     : "keep" — include in output; "delete" — remove from output.
    reason     : Why the segment is deleted ("silence", "head_silence",
                 "tail_silence").  None for keep segments.
    effects    : Post-processing flags to apply on render (e.g.
                 "jump_cut_zoom_1.1").
    speaker_id : Speaker label from transcript words, if available.
                 Used for jump-cut detection.
    is_broll   : True if this segment is B-roll; suppresses jump-cut
                 detection on this and adjacent segments.
    """

    start_s: float
    end_s: float
    action: Literal["keep", "delete"]
    reason: Optional[str] = None
    effects: list[str] = field(default_factory=list)
    speaker_id: Optional[str] = None
    is_broll: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        """Serialise to the canonical cut_list.json segment format."""
        d: dict = {
            "start": round(self.start_s, 3),
            "end": round(self.end_s, 3),
            "action": self.action,
            "effects": list(self.effects),
        }
        if self.reason is not None:
            d["reason"] = self.reason
        return d


@dataclass
class SilenceCandidate:
    """Legacy PCM-path: a potential silence gap flagged for removal.

    Attributes
    ----------
    start_s        : Gap start in seconds.
    end_s          : Gap end in seconds.
    duration_s     : end_s - start_s.
    rms_db         : Measured RMS level during this gap.
    auto_remove    : True for head/tail silences (bypass review queue).
    keep           : User override — set True to preserve intentional pauses.
    speaker_id     : Speaker label from transcript, if available.
    """
    start_s: float
    end_s: float
    duration_s: float
    rms_db: float
    auto_remove: bool = False
    keep: bool = False
    speaker_id: Optional[str] = None

    def to_timecode(self, val: float) -> str:
        h = int(val // 3600)
        m = int((val % 3600) // 60)
        s = val % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    @property
    def start_tc(self) -> str:
        return self.to_timecode(self.start_s)

    @property
    def end_tc(self) -> str:
        return self.to_timecode(self.end_s)


@dataclass
class Cut:
    """Legacy PCM-path: a single in/out cut point on the timeline.

    Attributes
    ----------
    in_s           : Cut-in position in seconds.
    out_s          : Cut-out position in seconds.
    speaker_id     : Speaker label; None means B-roll or unknown.
    is_broll       : True if this clip is B-roll (suppresses jump-cut detection).
    zoom_factor    : 1.0 = no zoom.  Set to JUMP_CUT_ZOOM for jump cuts.
    zoom_anchor_x  : Normalised X anchor [0–1]; 0.5 = centre.
    zoom_anchor_y  : Normalised Y anchor [0–1]; 0.5 = centre.
    ffmpeg_filter  : Populated by build_zoom_filter(); consumed by executor.
    """
    in_s: float
    out_s: float
    speaker_id: Optional[str] = None
    is_broll: bool = False
    zoom_factor: float = 1.0
    zoom_anchor_x: float = 0.5
    zoom_anchor_y: float = 0.5
    ffmpeg_filter: str = ""

    @property
    def duration_s(self) -> float:
        return self.out_s - self.in_s


@dataclass
class CutterResult:
    """Output produced by the Cutter skill for one transcript window.

    segments          : New transcript-based CutSegment list.
    cuts              : Legacy PCM-based cut list (backwards compat).
    silence_candidates: Legacy PCM-based silence candidate list.
    sensor_flags      : Advisory sensor flags from run_sensors / _run_cutter_sensors.
    """
    segments: list[CutSegment] = field(default_factory=list)
    cuts: list[Cut] = field(default_factory=list)
    silence_candidates: list[SilenceCandidate] = field(default_factory=list)
    sensor_flags: list[str] = field(default_factory=list)


# ── Transcript-based silence detection (primary path) ────────────────────

def detect_silence_from_transcript(
    words: list[dict],
    *,
    window_start_s: float = 0.0,
    window_end_s: float,
    threshold_s: float = SILENCE_MIN_DURATION_S,
    head_tail_s: float = HEAD_TAIL_SILENCE_S,
) -> list[CutSegment]:
    """Build a keep/delete segment list from transcript word timestamps.

    Primary detection mechanism (spec §2):
      · Gap between word[i].end_ms and word[i+1].start_ms ≥ threshold_s → delete
      · Head silence (window_start → first word) ≥ head_tail_s → delete
      · Tail silence (last word → window_end) ≥ head_tail_s → delete
      · Gaps below their respective thresholds are absorbed into adjacent keeps.

    Segments are contiguous and together cover [window_start_s, window_end_s].

    Parameters
    ----------
    words           : List of word dicts with start_ms, end_ms, and optional
                      speaker_id.  Must be within the ±30 s context window.
    window_start_s  : Start of the context window in seconds.
    window_end_s    : End of the context window in seconds.
    threshold_s     : Minimum inter-word gap to flag as silence.
    head_tail_s     : Minimum head/tail silence to auto-remove.

    Returns
    -------
    Ordered, non-overlapping list of CutSegment objects.
    """
    if not words:
        duration = window_end_s - window_start_s
        if duration > 0:
            return [CutSegment(
                start_s=window_start_s,
                end_s=window_end_s,
                action="delete",
                reason="silence",
            )]
        return []

    # Sort by start_ms ascending
    sorted_words = sorted(words, key=lambda w: w.get("start_ms", 0))

    # Convert to seconds; extract speaker_id if present
    spans: list[tuple[float, float, Optional[str]]] = [
        (
            w.get("start_ms", 0) / 1000.0,
            w.get("end_ms", 0) / 1000.0,
            w.get("speaker_id"),
        )
        for w in sorted_words
    ]

    # ── Group consecutive words into keep blocks ──────────────────────────
    # A new block starts whenever the inter-word gap >= threshold_s.
    # speaker_id of a block = speaker_id of its first word.
    keep_blocks: list[tuple[float, float, Optional[str]]] = []
    blk_start, blk_end, blk_speaker = spans[0]

    for ws, we, wspeaker in spans[1:]:
        gap = ws - blk_end
        if gap >= threshold_s:
            keep_blocks.append((blk_start, blk_end, blk_speaker))
            blk_start, blk_end, blk_speaker = ws, we, wspeaker
        else:
            blk_end = max(blk_end, we)
            # speaker_id stays as the first word of the block

    keep_blocks.append((blk_start, blk_end, blk_speaker))

    # ── Build contiguous segment list ─────────────────────────────────────
    segments: list[CutSegment] = []
    cur = window_start_s

    for i, (kb_s, kb_e, kb_spk) in enumerate(keep_blocks):
        gap = kb_s - cur
        if gap > 0:
            is_head = (cur == window_start_s)
            min_gap = head_tail_s if is_head else threshold_s
            if gap >= min_gap:
                reason = "head_silence" if is_head else "silence"
                segments.append(CutSegment(
                    start_s=cur,
                    end_s=kb_s,
                    action="delete",
                    reason=reason,
                ))
                cur = kb_s
            # else: gap is too small to flag — absorb into the keep (cur unchanged)

        keep_start = cur  # either kb_s (delete was inserted) or window_start_s (absorbed)
        segments.append(CutSegment(
            start_s=keep_start,
            end_s=kb_e,
            action="keep",
            speaker_id=kb_spk,
        ))
        cur = kb_e

    # ── Tail gap ──────────────────────────────────────────────────────────
    tail_gap = window_end_s - cur
    if tail_gap > 0:
        if tail_gap >= head_tail_s:
            segments.append(CutSegment(
                start_s=cur,
                end_s=window_end_s,
                action="delete",
                reason="tail_silence",
            ))
        else:
            # Absorb small tail into last keep segment
            if segments and segments[-1].action == "keep":
                last = segments[-1]
                segments[-1] = CutSegment(
                    start_s=last.start_s,
                    end_s=window_end_s,
                    action="keep",
                    speaker_id=last.speaker_id,
                    effects=list(last.effects),
                    is_broll=last.is_broll,
                )

    logger.info(
        "detect_silence_from_transcript: %d segments (%d delete, %d keep) "
        "from %d words in window [%.3f–%.3f s]",
        len(segments),
        sum(1 for s in segments if s.action == "delete"),
        sum(1 for s in segments if s.action == "keep"),
        len(words),
        window_start_s,
        window_end_s,
    )
    return segments


# ── Jump cut detection (transcript path) ─────────────────────────────────

def apply_jump_cuts(segments: list[CutSegment]) -> list[CutSegment]:
    """Mark jump cuts in a segment list (in-place) and return the same list.

    A jump cut is triggered when:
      · Two consecutive non-B-roll keep segments share the same non-None speaker_id.
      · "Consecutive" means no other keep segment (B-roll or otherwise) falls
        between them in the keep-segment sub-list.

    The *second* keep segment of each jump-cut pair receives
    JUMP_CUT_EFFECT appended to its ``effects`` list.

    Modifies ``segments`` in-place and returns it.
    """
    # Collect indices of non-broll keep segments in order
    keep_indices = [
        i for i, s in enumerate(segments)
        if s.action == "keep" and not s.is_broll
    ]

    jump_cut_count = 0
    for pair in range(1, len(keep_indices)):
        prev = segments[keep_indices[pair - 1]]
        curr = segments[keep_indices[pair]]

        same_speaker = (
            prev.speaker_id is not None
            and curr.speaker_id is not None
            and prev.speaker_id == curr.speaker_id
        )

        if same_speaker:
            if JUMP_CUT_EFFECT not in curr.effects:
                curr.effects.append(JUMP_CUT_EFFECT)
            jump_cut_count += 1
            logger.debug(
                "jump cut: keep at %.3f–%.3f s (speaker=%s) flagged → %s",
                curr.start_s, curr.end_s, curr.speaker_id, JUMP_CUT_EFFECT,
            )

    logger.info("apply_jump_cuts: %d jump cuts marked", jump_cut_count)
    return segments


# ── Transcript-path sensor validation ────────────────────────────────────

def _run_cutter_sensors(result: CutterResult) -> CutterResult:
    """Advisory quality gates for the transcript-based path.

    Gates checked:
      · DURATION : no keep segment shorter than MIN_CLIP_DURATION_S
    """
    flags: list[str] = []

    for i, seg in enumerate(result.segments):
        if seg.action == "keep" and seg.duration_s < MIN_CLIP_DURATION_S:
            flags.append(
                f"DURATION: segment[{i}] keep is {seg.duration_s:.3f}s "
                f"(< {MIN_CLIP_DURATION_S}s minimum)"
            )

    if flags:
        logger.warning(
            "Cutter sensors (transcript path) raised %d flag(s):\n  %s",
            len(flags), "\n  ".join(flags),
        )
    else:
        logger.info("Cutter sensors (transcript path): all gates clean")

    result.sensor_flags.extend(flags)
    return result


# ── Transcript-path public entry point ───────────────────────────────────

def run_cutter(
    words: list[dict],
    *,
    window_start_s: float = 0.0,
    window_end_s: float,
    threshold_s: float = SILENCE_MIN_DURATION_S,
    head_tail_s: float = HEAD_TAIL_SILENCE_S,
) -> CutterResult:
    """Full transcript-based Cutter pipeline for one context window.

    Parameters
    ----------
    words           : Words from CutterContext.transcript_window.
    window_start_s  : Context window start (seconds).
    window_end_s    : Context window end (seconds).
    threshold_s     : Minimum inter-word gap to flag as silence.
    head_tail_s     : Minimum head/tail silence to auto-remove.

    Returns
    -------
    CutterResult with ``segments`` populated and sensor_flags set.
    """
    result = CutterResult()

    result.segments = detect_silence_from_transcript(
        words,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        threshold_s=threshold_s,
        head_tail_s=head_tail_s,
    )

    result.segments = apply_jump_cuts(result.segments)

    result = _run_cutter_sensors(result)

    logger.info(
        "run_cutter complete: %d segments (%d keep / %d delete), %d sensor flags",
        len(result.segments),
        sum(1 for s in result.segments if s.action == "keep"),
        sum(1 for s in result.segments if s.action == "delete"),
        len(result.sensor_flags),
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Legacy PCM audio path (kept for backwards compatibility)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_silence(
    audio_frames: list[dict],
    *,
    total_duration_s: float,
    threshold_db: float = RMS_SILENCE_DBFS,
    min_duration_s: float = SILENCE_MIN_DURATION_S,
    head_tail_s: float = HEAD_TAIL_SILENCE_S,
) -> list[SilenceCandidate]:
    """Detect silence gaps in an audio frame sequence (PCM path).

    Parameters
    ----------
    audio_frames : List of dicts:
                     { start_s: float, end_s: float,
                       samples: list[float],   # normalised PCM [-1,1]
                       speaker_id: str | None }
                   Frames should be contiguous and sorted by start_s.
    total_duration_s : Full clip duration (needed for head/tail detection).
    threshold_db     : RMS level below which a frame is considered silent.
    min_duration_s   : Minimum continuous silence to flag.
    head_tail_s      : Edge silence length that triggers auto-remove.

    Returns
    -------
    Ordered list of SilenceCandidate objects.  Does NOT modify audio_frames.
    """
    candidates: list[SilenceCandidate] = []

    # Pass 1: tag each frame as silent or not
    tagged: list[tuple[dict, bool]] = []
    for frame in audio_frames:
        rms = _rms_db(frame.get("samples", []))
        tagged.append((frame, rms < threshold_db))

    # Pass 2: merge adjacent silent frames into gaps
    i = 0
    while i < len(tagged):
        frame, is_silent = tagged[i]
        if not is_silent:
            i += 1
            continue

        gap_start = frame["start_s"]
        gap_end = frame["end_s"]
        gap_samples: list[float] = list(frame.get("samples", []))
        speaker = frame.get("speaker_id")
        j = i + 1
        while j < len(tagged) and tagged[j][1]:
            gap_end = tagged[j][0]["end_s"]
            gap_samples.extend(tagged[j][0].get("samples", []))
            j += 1

        duration = gap_end - gap_start

        if duration >= min_duration_s:
            rms = _rms_db(gap_samples)
            auto = (
                gap_start <= head_tail_s
                or gap_end >= total_duration_s - head_tail_s
            )
            candidates.append(SilenceCandidate(
                start_s=gap_start,
                end_s=gap_end,
                duration_s=duration,
                rms_db=rms,
                auto_remove=auto,
                speaker_id=speaker,
            ))
            logger.debug(
                "silence detected  %.3f–%.3f s  (%.2f s, %.1f dBFS)%s",
                gap_start, gap_end, duration, rms,
                "  [auto-remove]" if auto else "",
            )

        i = j

    logger.info(
        "detect_silence: %d candidates found (%d auto-remove)",
        len(candidates),
        sum(1 for c in candidates if c.auto_remove),
    )
    return candidates


def detect_jump_cuts(cuts: list[Cut]) -> list[Cut]:
    """Scan a cut list and mark jump cuts for zoompan treatment (PCM path).

    A jump cut is defined as two consecutive cuts where:
      · Both clips share the same non-None speaker_id
      · Neither clip is B-roll
      · There is no B-roll clip between them

    The *second* clip of each jump-cut pair receives:
      · zoom_factor = JUMP_CUT_ZOOM  (1.1×)
      · ffmpeg_filter built by build_zoom_filter()

    Modifies cuts in-place and returns the same list.
    """
    jump_cut_count = 0

    for i in range(1, len(cuts)):
        prev = cuts[i - 1]
        curr = cuts[i]

        is_same_speaker = (
            curr.speaker_id is not None
            and prev.speaker_id is not None
            and curr.speaker_id == prev.speaker_id
        )
        no_broll_between = not prev.is_broll and not curr.is_broll

        if is_same_speaker and no_broll_between:
            curr.zoom_factor = JUMP_CUT_ZOOM
            curr.ffmpeg_filter = build_zoom_filter(
                zoom=JUMP_CUT_ZOOM,
                anchor_x=curr.zoom_anchor_x,
                anchor_y=curr.zoom_anchor_y,
                duration_s=curr.duration_s,
            )
            jump_cut_count += 1
            logger.debug(
                "jump cut detected: clip %d (%.3f–%.3f s, speaker=%s) → zoom %.1f×",
                i, curr.in_s, curr.out_s, curr.speaker_id, JUMP_CUT_ZOOM,
            )

    logger.info("detect_jump_cuts: %d jump cuts marked", jump_cut_count)
    return cuts


def build_zoom_filter(
    zoom: float,
    anchor_x: float,
    anchor_y: float,
    duration_s: float,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """Build an FFmpeg zoompan filter string for a static zoom baked into a clip.

    Returns
    -------
    FFmpeg filter string, e.g.:
      zoompan=z='1.1000':x='(iw-iw/zoom)*0.5000':y='(ih-ih/zoom)*0.5000':d=90:s=1920x1080:fps=30
    """
    frame_count = max(1, round(duration_s * fps))
    x_expr = f"(iw-iw/zoom)*{anchor_x:.4f}"
    y_expr = f"(ih-ih/zoom)*{anchor_y:.4f}"

    return (
        f"zoompan="
        f"z='{zoom:.4f}':"
        f"x='{x_expr}':"
        f"y='{y_expr}':"
        f"d={frame_count}:"
        f"s={width}x{height}:"
        f"fps={fps}"
    )


def run_sensors(result: CutterResult) -> CutterResult:
    """Post-process sensor checks for the PCM audio path (spec §5 quality gates).

    Gates checked:
      · JUMP_CUT_ZOOM : every jump cut must have a non-1.0 zoom_factor
      · DURATION      : no cut shorter than MIN_CLIP_DURATION_S
      · SILENCE       : no un-flagged silence remains
    """
    flags: list[str] = []

    for i, cut in enumerate(result.cuts):
        if cut.zoom_factor == JUMP_CUT_ZOOM and not cut.ffmpeg_filter:
            flags.append(
                f"JUMP_CUT_ZOOM: cut[{i}] marked for zoom but ffmpeg_filter is empty"
            )

    short_clips = [c for c in result.cuts if c.duration_s < MIN_CLIP_DURATION_S]
    for c in short_clips:
        flags.append(
            f"DURATION: clip {c.in_s:.3f}–{c.out_s:.3f}s is "
            f"{c.duration_s:.3f}s (< {MIN_CLIP_DURATION_S}s minimum)"
        )

    total_cut_duration = sum(c.duration_s for c in result.cuts)
    if total_cut_duration > 5.0 and not result.silence_candidates:
        flags.append(
            "SILENCE: no silence candidates generated — verify detector ran on audio"
        )

    if flags:
        logger.warning("Cutter sensors raised %d flag(s):\n  %s", len(flags), "\n  ".join(flags))
    else:
        logger.info("Cutter sensors: all gates clean")

    result.sensor_flags = flags
    return result


def run(
    audio_frames: list[dict],
    initial_cuts: list[Cut],
    *,
    total_duration_s: float,
    threshold_db: float = RMS_SILENCE_DBFS,
    min_silence_s: float = SILENCE_MIN_DURATION_S,
) -> CutterResult:
    """Full PCM-based Cutter pipeline for one context window (legacy path).

    Parameters
    ----------
    audio_frames     : PCM frame list from the transcript window.
    initial_cuts     : Rough cut list produced from the transcript word boundaries.
    total_duration_s : Full clip duration (for head/tail edge detection).
    threshold_db     : Silence RMS threshold in dBFS.
    min_silence_s    : Minimum silence duration to flag.

    Returns
    -------
    CutterResult with silence_candidates, zoom-marked cuts, and sensor flags.
    """
    result = CutterResult()

    result.silence_candidates = detect_silence(
        audio_frames,
        total_duration_s=total_duration_s,
        threshold_db=threshold_db,
        min_duration_s=min_silence_s,
    )

    result.cuts = detect_jump_cuts(list(initial_cuts))

    result = run_sensors(result)

    logger.info(
        "Cutter run complete (PCM path): %d cuts, %d silence candidates, "
        "%d sensor flags",
        len(result.cuts),
        len(result.silence_candidates),
        len(result.sensor_flags),
    )
    return result
