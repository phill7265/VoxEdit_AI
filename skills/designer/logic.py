"""
skills/designer/logic.py

Designer skill — Caption placement, keyword highlighting, B-roll/zoom overlays,
and audio-duck event generation.

Spec source: spec/editing_style.md  v0.2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rule §3  Jump Cut visual treatment
  · 1.1× zoom-in applied to the second clip of each jump-cut pair
  · Anchor: face bbox centre if detected; fallback to frame centre

Rule §4  Audio Ducking
  · Trigger: VAD confidence ≥ 0.85
  · Duck: −20 dB relative to current level, 150 ms attack, 500 ms release
  · Floor: −40 dBFS minimum (never absolute silence)

Visual constants (from project spec + Shorts best-practice):
  · Canvas : 1920 × 1080 px
  · Caption zone: bottom 20 % of frame  (y ≥ 864 px from top)
  · Caption font : 72 pt sans-serif
  · Normal text  : #FFFFFF (white)
  · Keyword highlight : #FFD700 (Gold)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module is STATELESS.  It consumes data injected by
src/pipeline/context_manager.py and returns typed result objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Canvas ─────────────────────────────────────────────────────────────────
CANVAS_WIDTH_PX: int = 1920
CANVAS_HEIGHT_PX: int = 1080

# ── Caption spec ───────────────────────────────────────────────────────────
CAPTION_FONT_SIZE_PT: int = 72
CAPTION_FONT_FAMILY: str = "sans-serif"

# Bottom 20 % rule: captions must sit at or below y = 0.80 × frame height.
CAPTION_BOTTOM_PCT: float = 0.20                          # 20 % of frame height
CAPTION_ZONE_TOP_PCT: float = 1.0 - CAPTION_BOTTOM_PCT   # 0.80
CAPTION_Y_PX: float = CANVAS_HEIGHT_PX * CAPTION_ZONE_TOP_PCT  # 864.0

CAPTION_X_PX: float = CANVAS_WIDTH_PX / 2.0              # 960.0 (centre)
CAPTION_ALIGN: str = "center"

CAPTION_COLOR: str = "#FFFFFF"          # normal text
CAPTION_BG_COLOR: str = "#00000080"     # semi-transparent black background
HIGHLIGHT_COLOR: str = "#FFD700"        # Gold — keyword highlight

# Caption grouping heuristics
CAPTION_MAX_WORDS: int = 5              # max words per caption block
CAPTION_MAX_DURATION_S: float = 3.0    # max seconds per caption block
CAPTION_GAP_SPLIT_S: float = 0.5       # inter-word gap that forces a new block

# ── Caption style names ────────────────────────────────────────────────────
STYLE_DEFAULT: str = "shorts_default"
STYLE_GOLD: str = "shorts_gold"

STYLE_COLORS: dict[str, str] = {
    STYLE_DEFAULT: CAPTION_COLOR,
    STYLE_GOLD: HIGHLIGHT_COLOR,
}

# ── Keyword highlight triggers (default set) ───────────────────────────────
# Any of these strings appearing in a caption block → apply Gold style.
DEFAULT_KEYWORDS: frozenset[str] = frozenset([
    "중요", "핵심", "주의", "포인트",
    "핵심적", "중요한", "주목", "결론",
])

# ── Audio ducking ──────────────────────────────────────────────────────────
VAD_CONFIDENCE_THRESHOLD: float = 0.85   # §4 trigger
DUCK_DB: float = -20.0                   # §4 duck depth (relative)
DUCK_ATTACK_MS: float = 150.0            # §4 attack
DUCK_RELEASE_MS: float = 500.0           # §4 release
DUCK_FLOOR_DBFS: float = -40.0           # §4 absolute floor

# ── Jump cut ───────────────────────────────────────────────────────────────
JUMP_CUT_ZOOM: float = 1.1
JUMP_CUT_EFFECT_KEY: str = "jump_cut_zoom_1.1"


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class CaptionStyle:
    """Describes the visual appearance of a caption block."""
    name: str = STYLE_DEFAULT
    font_size_pt: int = CAPTION_FONT_SIZE_PT
    font_family: str = CAPTION_FONT_FAMILY
    color: str = CAPTION_COLOR
    background_color: str = CAPTION_BG_COLOR
    position_y_px: float = CAPTION_Y_PX
    position_x_px: float = CAPTION_X_PX
    align: str = CAPTION_ALIGN


# Pre-built style instances used throughout the module
CAPTION_STYLE_DEFAULT = CaptionStyle(name=STYLE_DEFAULT, color=CAPTION_COLOR)
CAPTION_STYLE_GOLD = CaptionStyle(name=STYLE_GOLD, color=HIGHLIGHT_COLOR)


@dataclass
class VisualElement:
    """A single visual or audio event on the output timeline.

    ``type`` selects the semantic role:
      "caption"  — on-screen subtitle block
      "overlay"  — composited graphic asset (subscribe CTA, logo, etc.)
      "zoom"     — frame-level scale transform (jump-cut 1.1×)
      "duck"     — audio ducking envelope instruction
    """

    type: str                            # "caption" | "overlay" | "zoom" | "duck"
    start: float                         # seconds (absolute timeline)
    end: float                           # seconds

    # caption ──────────────────────────────────────────────────────────────
    text: str = ""
    style: str = STYLE_DEFAULT
    position_x_px: float = CAPTION_X_PX
    position_y_px: float = CAPTION_Y_PX
    font_size_pt: int = CAPTION_FONT_SIZE_PT
    color: str = CAPTION_COLOR
    background_color: str = CAPTION_BG_COLOR

    # overlay ──────────────────────────────────────────────────────────────
    name: str = ""

    # zoom ─────────────────────────────────────────────────────────────────
    zoom_factor: float = 1.0
    anchor_x: float = 0.5              # normalised [0–1]; 0.5 = centre
    anchor_y: float = 0.5

    # duck ─────────────────────────────────────────────────────────────────
    duck_db: float = DUCK_DB
    attack_ms: float = DUCK_ATTACK_MS
    release_ms: float = DUCK_RELEASE_MS

    # b-roll ────────────────────────────────────────────────────────────────
    asset_path: str = ""        # absolute path to b-roll video file
    opacity: float = 1.0        # 0.0–1.0; <1.0 blends over main; 1.0 replaces
    broll_mode: str = "overlay" # "overlay" | "replace"
    keyword: str = ""           # source keyword used to select this asset

    @property
    def duration_s(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        """Serialise to the canonical annotated_timeline.json element format."""
        d: dict = {
            "type": self.type,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }
        if self.type == "caption":
            d.update({
                "text": self.text,
                "style": self.style,
                "position_x_px": round(self.position_x_px, 1),
                "position_y_px": round(self.position_y_px, 1),
                "font_size_pt": self.font_size_pt,
                "color": self.color,
            })
        elif self.type == "overlay":
            d["name"] = self.name
        elif self.type == "zoom":
            d.update({
                "zoom_factor": round(self.zoom_factor, 4),
                "anchor_x": round(self.anchor_x, 4),
                "anchor_y": round(self.anchor_y, 4),
            })
        elif self.type == "duck":
            d.update({
                "duck_db": self.duck_db,
                "attack_ms": self.attack_ms,
                "release_ms": self.release_ms,
            })
        elif self.type == "b-roll":
            d.update({
                "asset_path": self.asset_path,
                "opacity": round(self.opacity, 3),
                "broll_mode": self.broll_mode,
                "keyword": self.keyword,
            })
        return d


@dataclass
class DesignerResult:
    """All visual and audio events produced for one context window."""
    visual_elements: list[VisualElement] = field(default_factory=list)
    sensor_flags: list[str] = field(default_factory=list)

    # ── Typed accessors ───────────────────────────────────────────────────
    @property
    def captions(self) -> list[VisualElement]:
        return [e for e in self.visual_elements if e.type == "caption"]

    @property
    def overlays(self) -> list[VisualElement]:
        return [e for e in self.visual_elements if e.type == "overlay"]

    @property
    def zooms(self) -> list[VisualElement]:
        return [e for e in self.visual_elements if e.type == "zoom"]

    @property
    def duck_events(self) -> list[VisualElement]:
        return [e for e in self.visual_elements if e.type == "duck"]

    @property
    def brolls(self) -> list[VisualElement]:
        return [e for e in self.visual_elements if e.type == "b-roll"]

    @property
    def highlights(self) -> list[VisualElement]:
        return [e for e in self.captions if e.style == STYLE_GOLD]


# ── Keyword detection ─────────────────────────────────────────────────────

def detect_keyword_style(
    text: str,
    keywords: frozenset[str] = DEFAULT_KEYWORDS,
) -> str:
    """Return the caption style name appropriate for ``text``.

    Returns ``STYLE_GOLD`` if any keyword appears in the text as a substring;
    otherwise returns ``STYLE_DEFAULT``.

    Parameters
    ----------
    text     : Caption block text (space-separated words).
    keywords : Set of trigger strings.

    Returns
    -------
    "shorts_gold" | "shorts_default"
    """
    for kw in keywords:
        if kw in text:
            return STYLE_GOLD
    return STYLE_DEFAULT


# ── Caption generation ────────────────────────────────────────────────────

def generate_captions(
    words: list[dict],
    *,
    keywords: frozenset[str] = DEFAULT_KEYWORDS,
    canvas_height_px: int = CANVAS_HEIGHT_PX,
    canvas_width_px: int = CANVAS_WIDTH_PX,
    caption_bottom_pct: float = CAPTION_BOTTOM_PCT,
    font_size_pt: int = CAPTION_FONT_SIZE_PT,
    max_words: int = CAPTION_MAX_WORDS,
    max_duration_s: float = CAPTION_MAX_DURATION_S,
    gap_split_s: float = CAPTION_GAP_SPLIT_S,
) -> list[VisualElement]:
    """Generate caption VisualElements from word-level transcript data.

    Words are grouped into caption blocks using three split triggers:
      1. Inter-word gap ≥ gap_split_s (natural pause / sentence boundary).
      2. Block would exceed max_words words.
      3. Block would exceed max_duration_s seconds.

    Each block is placed at:
      · x = canvas_width_px / 2 (centred)
      · y = canvas_height_px × (1 − caption_bottom_pct)

    A block whose text contains any keyword gets ``STYLE_GOLD``; otherwise
    ``STYLE_DEFAULT``.

    Parameters
    ----------
    words              : List of word dicts with start_ms, end_ms, word keys.
    keywords           : Keyword trigger set for gold highlighting.
    canvas_height_px   : Frame height in pixels.
    canvas_width_px    : Frame width in pixels.
    caption_bottom_pct : Fraction of frame height reserved for the caption
                         safe zone (default 0.20 → bottom 20 %).
    font_size_pt       : Caption font size in points.
    max_words          : Maximum words per block.
    max_duration_s     : Maximum seconds per block.
    gap_split_s        : Inter-word gap threshold to force a new block.

    Returns
    -------
    Ordered list of caption VisualElements.
    """
    if not words:
        return []

    y_px = canvas_height_px * (1.0 - caption_bottom_pct)
    x_px = canvas_width_px / 2.0

    # Sort words by start_ms ascending (defensive: should already be sorted)
    sorted_words = sorted(words, key=lambda w: w.get("start_ms", 0))

    captions: list[VisualElement] = []
    block: list[dict] = []

    def _flush(b: list[dict]) -> None:
        if not b:
            return
        text = " ".join(w.get("word", "").strip() for w in b).strip()
        if not text:
            return
        start_s = b[0]["start_ms"] / 1000.0
        end_s = b[-1]["end_ms"] / 1000.0
        style = detect_keyword_style(text, keywords)
        color = STYLE_COLORS.get(style, CAPTION_COLOR)
        captions.append(VisualElement(
            type="caption",
            start=start_s,
            end=end_s,
            text=text,
            style=style,
            position_x_px=x_px,
            position_y_px=y_px,
            font_size_pt=font_size_pt,
            color=color,
        ))

    for word in sorted_words:
        if not block:
            block.append(word)
            continue

        gap_s = (word.get("start_ms", 0) - block[-1].get("end_ms", 0)) / 1000.0
        block_start_s = block[0].get("start_ms", 0) / 1000.0
        block_dur_s = word.get("end_ms", 0) / 1000.0- block_start_s

        split = (
            gap_s >= gap_split_s                 # natural pause
            or len(block) >= max_words           # word count cap
            or block_dur_s > max_duration_s      # duration cap
        )

        if split:
            _flush(block)
            block = [word]
        else:
            block.append(word)

    _flush(block)

    logger.info(
        "generate_captions: %d blocks from %d words (y_px=%.1f, %d highlighted)",
        len(captions), len(words), y_px,
        sum(1 for c in captions if c.style == STYLE_GOLD),
    )
    return captions


# ── Zoom overlay generation ───────────────────────────────────────────────

def build_zoom_overlays(
    cut_segments: list[dict],
    *,
    zoom_factor: float = JUMP_CUT_ZOOM,
    default_anchor_x: float = 0.5,
    default_anchor_y: float = 0.5,
) -> list[VisualElement]:
    """Convert jump-cut effects in cut segments to zoom VisualElements.

    Iterates over ``cut_segments`` (as produced by the Cutter skill) and
    creates a ``zoom`` element for every ``keep`` segment that carries the
    ``jump_cut_zoom_1.1`` effect flag.

    Parameters
    ----------
    cut_segments    : List of segment dicts from cut_list.json.
    zoom_factor     : Scale factor to apply (spec §3: 1.1).
    default_anchor_x: Normalised X anchor when no face detection is available.
    default_anchor_y: Normalised Y anchor when no face detection is available.

    Returns
    -------
    List of zoom VisualElements, one per jump-cut keep segment.
    """
    zooms: list[VisualElement] = []

    for seg in cut_segments:
        if seg.get("action") != "keep":
            continue
        if JUMP_CUT_EFFECT_KEY not in seg.get("effects", []):
            continue

        zooms.append(VisualElement(
            type="zoom",
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            zoom_factor=zoom_factor,
            anchor_x=default_anchor_x,
            anchor_y=default_anchor_y,
        ))
        logger.debug(
            "zoom overlay: %.3f–%.3f s  factor=%.2f×",
            seg["start"], seg["end"], zoom_factor,
        )

    logger.info("build_zoom_overlays: %d zoom elements produced", len(zooms))
    return zooms


# ── Audio ducking event generation ───────────────────────────────────────

def build_duck_events(
    vad_segments: list[dict],
    *,
    vad_threshold: float = VAD_CONFIDENCE_THRESHOLD,
    duck_db: float = DUCK_DB,
    attack_ms: float = DUCK_ATTACK_MS,
    release_ms: float = DUCK_RELEASE_MS,
) -> list[VisualElement]:
    """Convert VAD voice segments to audio duck VisualElements.

    Only VAD segments where ``is_voice=True`` AND ``confidence ≥ vad_threshold``
    produce duck events (spec §4 trigger: VAD confidence ≥ 0.85).

    Parameters
    ----------
    vad_segments  : VAD segment dicts from transcript.json.
                    Each dict has: start_s, end_s, is_voice, confidence.
    vad_threshold : Minimum confidence to trigger ducking.
    duck_db       : Duck depth in dB (relative, negative).
    attack_ms     : Attack time in ms.
    release_ms    : Release time in ms.

    Returns
    -------
    List of duck VisualElements.
    """
    events: list[VisualElement] = []

    for seg in vad_segments:
        if not seg.get("is_voice", False):
            continue

        # Accept either "confidence" (canonical) or "avg_probability" (raw)
        confidence = float(
            seg.get("confidence", seg.get("avg_probability", 0.0))
        )
        if confidence < vad_threshold:
            continue

        start_s = float(seg.get("start_s", seg.get("start", 0.0)))
        end_s = float(seg.get("end_s", seg.get("end", 0.0)))

        events.append(VisualElement(
            type="duck",
            start=start_s,
            end=end_s,
            duck_db=duck_db,
            attack_ms=attack_ms,
            release_ms=release_ms,
        ))
        logger.debug(
            "duck event: %.3f–%.3f s  confidence=%.2f  duck_db=%.1f",
            start_s, end_s, confidence, duck_db,
        )

    logger.info("build_duck_events: %d duck events produced", len(events))
    return events


# ── B-roll element builder ────────────────────────────────────────────────

def build_broll_elements(
    broll_requests: list[dict],
    *,
    total_duration_s: float,
) -> list[VisualElement]:
    """Convert IntentProcessor b-roll requests to VisualElements.

    Each request dict has:
      keyword    : str   — label used to find the asset
      asset_path : str   — absolute path to the video file
      start_s    : float — output-timeline start (default: 0.25 × total)
      end_s      : float — output-timeline end   (default: start + min(5, 0.5×total))
      opacity    : float — 0.0–1.0 (default 1.0)
      mode       : str   — "overlay" | "replace" (default "overlay")

    Requests with a missing or non-existent asset_path are skipped with a warning.
    """
    from pathlib import Path as _Path

    elements: list[VisualElement] = []
    for req in broll_requests:
        asset = req.get("asset_path", "")
        if not asset or not _Path(asset).exists():
            logger.warning(
                "build_broll_elements: asset not found, skipping — %s", asset
            )
            continue

        default_start = round(total_duration_s * 0.25, 3)
        default_dur = min(5.0, total_duration_s * 0.5)
        start_s = float(req.get("start_s", default_start))
        end_s = float(req.get("end_s", start_s + default_dur))
        # Clamp to valid output range
        end_s = min(end_s, total_duration_s)
        start_s = min(start_s, end_s - 0.5)

        elements.append(VisualElement(
            type="b-roll",
            start=round(start_s, 3),
            end=round(end_s, 3),
            asset_path=asset,
            opacity=float(req.get("opacity", 1.0)),
            broll_mode=req.get("mode", "overlay"),
            keyword=req.get("keyword", ""),
        ))
        logger.info(
            "b-roll element: keyword=%s  %.3f–%.3f s  opacity=%.2f  mode=%s",
            req.get("keyword", ""), start_s, end_s,
            float(req.get("opacity", 1.0)), req.get("mode", "overlay"),
        )

    logger.info("build_broll_elements: %d elements produced", len(elements))
    return elements


# ── Sensor validation ─────────────────────────────────────────────────────

def _run_designer_sensors(
    result: DesignerResult,
    *,
    canvas_height_px: int = CANVAS_HEIGHT_PX,
    caption_bottom_pct: float = CAPTION_BOTTOM_PCT,
) -> DesignerResult:
    """Advisory quality gates for the designer skill.

    Gates checked:
      · SAFE_ZONE  : all captions must have position_y_px ≥ safe-zone threshold
      · JUMP_CUT_ZOOM : spec §3 — zoom_factor on zoom elements should match spec
    """
    flags: list[str] = []
    safe_y = canvas_height_px * (1.0 - caption_bottom_pct)

    for i, elem in enumerate(result.visual_elements):
        if elem.type == "caption" and elem.position_y_px < safe_y:
            flags.append(
                f"SAFE_ZONE: caption[{i}] y={elem.position_y_px:.1f}px "
                f"is above safe-zone threshold {safe_y:.1f}px "
                f"(bottom {caption_bottom_pct*100:.0f}%)"
            )

    for i, elem in enumerate(result.visual_elements):
        if elem.type == "zoom" and abs(elem.zoom_factor - JUMP_CUT_ZOOM) > 0.001:
            flags.append(
                f"JUMP_CUT_ZOOM: zoom[{i}] factor={elem.zoom_factor} "
                f"deviates from spec {JUMP_CUT_ZOOM}"
            )

    if flags:
        logger.warning(
            "Designer sensors raised %d flag(s):\n  %s",
            len(flags), "\n  ".join(flags),
        )
    else:
        logger.info("Designer sensors: all gates clean")

    result.sensor_flags.extend(flags)
    return result


# ── Public entry point ────────────────────────────────────────────────────

def run_designer(
    words: list[dict],
    cut_segments: list[dict],
    vad_segments: list[dict],
    *,
    keywords: frozenset[str] = DEFAULT_KEYWORDS,
    canvas_width_px: int = CANVAS_WIDTH_PX,
    canvas_height_px: int = CANVAS_HEIGHT_PX,
    font_size_pt: int = CAPTION_FONT_SIZE_PT,
    caption_bottom_pct: float = CAPTION_BOTTOM_PCT,
    vad_threshold: float = VAD_CONFIDENCE_THRESHOLD,
    duck_db: float = DUCK_DB,
    attack_ms: float = DUCK_ATTACK_MS,
    release_ms: float = DUCK_RELEASE_MS,
    broll_requests: Optional[list[dict]] = None,
) -> DesignerResult:
    """Full Designer pipeline for one context window.

    Produces:
      · Caption blocks (bottom 20 %, 72 pt, keyword gold-highlighting)
      · Zoom overlays (1.1× for jump-cut keep segments)
      · Audio duck envelopes (VAD-triggered, -20 dB, 150/500 ms)

    Parameters
    ----------
    words            : Word dicts from transcript_window (start_ms, end_ms, word).
    cut_segments     : CutSegment dicts from cut_list.json (action, start, end, effects).
    vad_segments     : VAD segment dicts from transcript.json.
    keywords         : Gold-highlight trigger keywords.
    canvas_width_px  : Output frame width.
    canvas_height_px : Output frame height.
    font_size_pt     : Caption font size in points.
    caption_bottom_pct : Fraction of frame height for caption safe zone.
    vad_threshold    : Minimum VAD confidence for duck trigger.
    duck_db          : Duck depth (dB, relative, negative).
    attack_ms        : Duck attack time (ms).
    release_ms       : Duck release time (ms).

    Returns
    -------
    DesignerResult with all visual_elements populated.
    """
    result = DesignerResult()

    # ── 1. Captions ───────────────────────────────────────────────────────
    result.visual_elements.extend(
        generate_captions(
            words,
            keywords=keywords,
            canvas_height_px=canvas_height_px,
            canvas_width_px=canvas_width_px,
            caption_bottom_pct=caption_bottom_pct,
            font_size_pt=font_size_pt,
        )
    )

    # ── 2. Zoom overlays (jump cuts) ──────────────────────────────────────
    result.visual_elements.extend(
        build_zoom_overlays(cut_segments)
    )

    # ── 3. Audio duck events ──────────────────────────────────────────────
    result.visual_elements.extend(
        build_duck_events(
            vad_segments,
            vad_threshold=vad_threshold,
            duck_db=duck_db,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
    )

    # ── 4. B-roll overlays (from IntentProcessor requests) ────────────────
    if broll_requests:
        total_dur = sum(
            max(0.0, seg.get("end", 0.0) - seg.get("start", 0.0))
            for seg in cut_segments
            if seg.get("action") == "keep"
        ) or 30.0  # fallback if no cut segments
        result.visual_elements.extend(
            build_broll_elements(broll_requests, total_duration_s=total_dur)
        )

    # Sort all elements by start time for deterministic output
    result.visual_elements.sort(key=lambda e: e.start)

    result = _run_designer_sensors(
        result,
        canvas_height_px=canvas_height_px,
        caption_bottom_pct=caption_bottom_pct,
    )

    logger.info(
        "run_designer complete: %d elements "
        "(%d captions, %d zooms, %d ducks, %d highlights), %d sensor flags",
        len(result.visual_elements),
        len(result.captions),
        len(result.zooms),
        len(result.duck_events),
        len(result.highlights),
        len(result.sensor_flags),
    )
    return result
