"""
skills/exporter/logic.py

Exporter skill — FFmpeg complex filtergraph builder.

Spec source: spec/editing_style.md  v0.2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Responsibilities
  · Read the cutter's keep/delete segment list → trim source video
  · Read the designer's visual elements → apply to the output
      captions   → drawtext filter
      zoom       → zoompan filter (baked per-segment, before concat)
      duck       → volume filter (envelope expression on audio)
  · Produce a validated FFmpeg command for the sandbox executor
  · Validate via harness/sandbox/executor.validate_command (dry-run)

Output profile
  · YouTube Shorts: 1080×1920, 9:16, 30 fps, H.264/AAC
  · YouTube landscape: 1920×1080, 16:9, 30 fps, H.264/AAC (fallback)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module is STATELESS.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Export profiles ────────────────────────────────────────────────────────
PROFILE_SHORTS: dict = {
    "platform": "youtube_shorts",
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "video_codec": "libx264",
    "audio_codec": "aac",
    "crf": 18,
    "preset": "fast",
    "aspect_ratio": "9:16",
}

PROFILE_LANDSCAPE: dict = {
    "platform": "youtube",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "video_codec": "libx264",
    "audio_codec": "aac",
    "crf": 18,
    "preset": "fast",
    "aspect_ratio": "16:9",
}

DEFAULT_PROFILE = PROFILE_SHORTS

# ── Audio ducking ──────────────────────────────────────────────────────────
DUCK_DB: float = -20.0
DUCK_FACTOR: float = 10 ** (DUCK_DB / 20.0)  # ≈ 0.100


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class ExportPlan:
    """Blueprint for a complete FFmpeg render command.

    Attributes
    ----------
    source_file     : Absolute path to the source video/audio file.
    output_file     : Absolute path to the rendered output file (in staging).
    keep_segments   : List of {start, end} dicts from cut_list.json.
    captions        : Caption VisualElement dicts (type=="caption").
    zoom_elements   : Zoom VisualElement dicts (type=="zoom").
    duck_events     : Duck VisualElement dicts (type=="duck").
    export_profile  : Dict from PROFILE_SHORTS / PROFILE_LANDSCAPE.
    """
    source_file: str
    output_file: str
    keep_segments: list[dict]
    captions: list[dict]
    zoom_elements: list[dict]
    duck_events: list[dict]
    export_profile: dict = field(default_factory=lambda: dict(DEFAULT_PROFILE))

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def total_output_duration_s(self) -> float:
        return sum(seg["end"] - seg["start"] for seg in self.keep_segments)

    def compute_output_time(self, source_t: float) -> Optional[float]:
        """Map a source timestamp to the post-edit output timestamp.

        Returns None if the source_t falls inside a deleted segment.
        """
        offset = 0.0
        for seg in self.keep_segments:
            if seg["start"] <= source_t <= seg["end"]:
                return offset + (source_t - seg["start"])
            offset += seg["end"] - seg["start"]
        return None

    @property
    def filter_complex(self) -> str:
        """Build and return the complete -filter_complex string."""
        return _build_filter_complex(self)

    @property
    def video_out_label(self) -> str:
        """Final video stream label after all filters."""
        return _compute_labels(self)[0]

    @property
    def audio_out_label(self) -> str:
        """Final audio stream label after all filters."""
        return _compute_labels(self)[1]

    def to_command(self) -> str:
        """Produce the full FFmpeg command string for this export plan.

        The command is safe to pass to shlex.split() and harness/sandbox/executor.run().
        """
        p = self.export_profile
        width = p.get("width", 1080)
        height = p.get("height", 1920)
        vcodec = p.get("video_codec", "libx264")
        acodec = p.get("audio_codec", "aac")
        crf = p.get("crf", 18)
        preset = p.get("preset", "fast")

        fc = self.filter_complex
        vout = self.video_out_label
        aout = self.audio_out_label

        # Build command. Double-quote paths and filter_complex to survive shlex.split().
        # The filter_complex may contain single quotes (FFmpeg expressions) which are
        # safe inside double-quoted strings.
        parts = [
            "ffmpeg",
            f'-i "{self.source_file}"',
            f'-filter_complex "{fc}"',
            f'-map "{vout}"',
            f'-map "{aout}"',
            f"-c:v {vcodec}",
            f"-crf {crf}",
            f"-preset {preset}",
            f"-c:a {acodec}",
            f"-s {width}x{height}",
            f'"{self.output_file}"',
        ]
        return " ".join(parts)


# ── Filtergraph builder ───────────────────────────────────────────────────

def _escape_drawtext(text: str) -> str:
    """Escape special chars for FFmpeg drawtext ``text`` option."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    return text


def _find_zoom(seg: dict, zoom_elements: list[dict]) -> Optional[dict]:
    """Return the zoom element whose time range overlaps this segment, or None."""
    s, e = seg.get("start", 0.0), seg.get("end", 0.0)
    for z in zoom_elements:
        # Match if centre of zoom window falls inside the segment
        zm = (z.get("start", 0.0) + z.get("end", 0.0)) / 2.0
        if s <= zm <= e:
            return z
        # Also match on exact bounds
        if abs(z.get("start", 0.0) - s) < 0.05 and abs(z.get("end", 0.0) - e) < 0.05:
            return z
    return None


def _build_video_trim_parts(
    plan: ExportPlan,
) -> tuple[list[str], list[str]]:
    """Build per-segment video trim (+ optional zoompan) filter parts.

    Returns (filter_parts, output_labels).
    """
    parts: list[str] = []
    labels: list[str] = []
    fps = plan.export_profile.get("fps", 30)
    w = plan.export_profile.get("width", 1080)
    h = plan.export_profile.get("height", 1920)

    for i, seg in enumerate(plan.keep_segments):
        s, e = seg["start"], seg["end"]
        raw = f"[v{i}]"
        parts.append(
            f"[0:v]trim=start={s:.3f}:end={e:.3f},"
            f"setpts=PTS-STARTPTS{raw}"
        )

        zoom = _find_zoom(seg, plan.zoom_elements)
        if zoom:
            zf = zoom.get("zoom_factor", 1.1)
            ax = zoom.get("anchor_x", 0.5)
            ay = zoom.get("anchor_y", 0.5)
            frames = max(1, round((e - s) * fps))
            z_label = f"[vz{i}]"
            zp = (
                f"zoompan=z='{zf:.4f}':"
                f"x='(iw-iw/zoom)*{ax:.4f}':"
                f"y='(ih-ih/zoom)*{ay:.4f}':"
                f"d={frames}:s={w}x{h}:fps={fps}"
            )
            parts.append(f"{raw}{zp}{z_label}")
            labels.append(z_label)
        else:
            labels.append(raw)

    return parts, labels


def _build_audio_trim_parts(plan: ExportPlan) -> tuple[list[str], list[str]]:
    """Build per-segment audio trim filter parts.

    Returns (filter_parts, output_labels).
    """
    parts: list[str] = []
    labels: list[str] = []
    for i, seg in enumerate(plan.keep_segments):
        s, e = seg["start"], seg["end"]
        label = f"[a{i}]"
        parts.append(
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},"
            f"asetpts=PTS-STARTPTS{label}"
        )
        labels.append(label)
    return parts, labels


def _build_concat_part(
    video_labels: list[str],
    audio_labels: list[str],
) -> tuple[str, str, str]:
    """Build concat filter.

    Returns (filter_string, video_out_label, audio_out_label).
    """
    n = len(video_labels)
    if n == 0:
        return "", "[0:v]", "[0:a]"
    if n == 1:
        # Single segment — no concat needed, just rename via null filter
        v_in = video_labels[0]
        a_in = audio_labels[0]
        return (
            f"{v_in}null[vcat];{a_in}anull[acat]",
            "[vcat]",
            "[acat]",
        )
    inputs = "".join(video_labels) + "".join(audio_labels)
    filt = f"{inputs}concat=n={n}:v=1:a=1[vcat][acat]"
    return filt, "[vcat]", "[acat]"


def _build_caption_parts(
    plan: ExportPlan,
    video_in: str,
) -> tuple[list[str], str]:
    """Build a chain of drawtext filters for captions.

    Maps source timestamps to output timeline before building.

    Returns (filter_parts, final_video_label).
    """
    from skills.designer.logic import STYLE_GOLD, HIGHLIGHT_COLOR, CAPTION_COLOR

    if not plan.captions:
        return [], video_in

    parts: list[str] = []
    current = video_in

    for i, cap in enumerate(plan.captions):
        src_start = cap.get("start", 0.0)
        src_end = cap.get("end", 0.0)

        # Remap to output timeline
        out_start = plan.compute_output_time(src_start)
        out_end = plan.compute_output_time(src_end)

        # Skip captions that fall entirely inside deleted segments
        if out_start is None and out_end is None:
            continue
        out_start = out_start if out_start is not None else 0.0
        out_end = out_end if out_end is not None else plan.total_output_duration_s

        text = _escape_drawtext(cap.get("text", ""))
        style = cap.get("style", "shorts_default")
        color = HIGHLIGHT_COLOR if style == STYLE_GOLD else CAPTION_COLOR
        # Remove # prefix — FFmpeg uses 0xRRGGBB or colour names
        color_ff = color.lstrip("#")
        y_px = int(cap.get("position_y_px", 864))
        x_px = int(cap.get("position_x_px", 960))
        font_size = cap.get("font_size_pt", 72)

        out_label = f"[vc{i}]"
        # Use half-width centre alignment: x = (w - text_w) / 2 ≈ x_px
        filt = (
            f"{current}drawtext="
            f"text='{text}':"
            f"x={x_px}:"
            f"y={y_px}:"
            f"fontsize={font_size}:"
            f"fontcolor=0x{color_ff}:"
            f"enable='between(t,{out_start:.3f},{out_end:.3f})'"
            f"{out_label}"
        )
        parts.append(filt)
        current = out_label

    return parts, current


def _build_duck_part(
    plan: ExportPlan,
    audio_in: str,
) -> tuple[str, str]:
    """Build a volume filter expression for audio ducking.

    Returns (filter_string, audio_out_label). Returns ("", audio_in) if no duck events.
    """
    if not plan.duck_events:
        return "", audio_in

    # Build nested if-expression: if(between(t,S,E),FACTOR,if(between(t,...
    conditions: list[str] = []
    for ev in plan.duck_events:
        s = plan.compute_output_time(ev.get("start", 0.0)) or 0.0
        e = plan.compute_output_time(ev.get("end", 0.0)) or 0.0
        factor = round(DUCK_FACTOR, 4)
        conditions.append((s, e, factor))

    if not conditions:
        return "", audio_in

    # Innermost value is 1 (no ducking outside all windows)
    expr = "1"
    for s, e, f in reversed(conditions):
        expr = f"if(between(t,{s:.3f},{e:.3f}),{f},{expr})"

    out_label = "[aduck]"
    filt = f"{audio_in}volume=volume='{expr}':eval=frame{out_label}"
    return filt, out_label


def _compute_labels(plan: ExportPlan) -> tuple[str, str]:
    """Return (final_video_label, final_audio_label) without building the full graph."""
    _, v_labels = _build_video_trim_parts(plan)
    _, a_labels = _build_audio_trim_parts(plan)
    _, vcat, acat = _build_concat_part(v_labels, a_labels)

    cap_parts, v_after_caps = _build_caption_parts(plan, vcat)
    duck_part, a_after_duck = _build_duck_part(plan, acat)

    return v_after_caps, a_after_duck


def _build_filter_complex(plan: ExportPlan) -> str:
    """Assemble the complete FFmpeg -filter_complex string for the plan."""
    if not plan.keep_segments:
        # No editing — pass through (null/anull)
        return "[0:v]null[vout];[0:a]anull[aout]"

    all_parts: list[str] = []

    # 1. Video trim + optional zoom
    v_trim_parts, v_labels = _build_video_trim_parts(plan)
    all_parts.extend(v_trim_parts)

    # 2. Audio trim
    a_trim_parts, a_labels = _build_audio_trim_parts(plan)
    all_parts.extend(a_trim_parts)

    # 3. Concat
    concat_filt, vcat, acat = _build_concat_part(v_labels, a_labels)
    if concat_filt:
        all_parts.append(concat_filt)

    # 4. Caption drawtext chain
    cap_parts, v_final = _build_caption_parts(plan, vcat)
    all_parts.extend(cap_parts)

    # 5. Audio duck volume filter
    duck_filt, a_final = _build_duck_part(plan, acat)
    if duck_filt:
        all_parts.append(duck_filt)

    logger.info(
        "filter_complex: %d parts | v_out=%s a_out=%s",
        len(all_parts), v_final, a_final,
    )
    return ";".join(all_parts)


# ── Plan builder ──────────────────────────────────────────────────────────

def build_export_plan(
    cut_segments: list[dict],
    visual_elements: list[dict],
    *,
    source_file: str,
    output_file: str,
    export_profile: Optional[dict] = None,
) -> ExportPlan:
    """Build an ExportPlan from cutter and designer outputs.

    Parameters
    ----------
    cut_segments    : CutSegment dicts from cut_list.json.
    visual_elements : VisualElement dicts from annotated_timeline.json.
    source_file     : Path to raw source video/audio.
    output_file     : Path for rendered output (must be inside staging).
    export_profile  : Export settings dict; defaults to PROFILE_SHORTS.

    Returns
    -------
    ExportPlan ready for ``to_command()`` and sandbox execution.
    """
    profile = export_profile or dict(DEFAULT_PROFILE)

    keep_segs = [
        seg for seg in cut_segments
        if seg.get("action") == "keep"
    ]

    captions = [e for e in visual_elements if e.get("type") == "caption"]
    zoom_elems = [e for e in visual_elements if e.get("type") == "zoom"]
    duck_evts = [e for e in visual_elements if e.get("type") == "duck"]

    plan = ExportPlan(
        source_file=source_file,
        output_file=output_file,
        keep_segments=keep_segs,
        captions=captions,
        zoom_elements=zoom_elems,
        duck_events=duck_evts,
        export_profile=profile,
    )

    logger.info(
        "build_export_plan: %d keep segs, %d captions, "
        "%d zooms, %d duck events",
        len(keep_segs), len(captions), len(zoom_elems), len(duck_evts),
    )
    return plan
