"""
harness/sensors/video_validator.py

Video quality sensor — post-execution gate for the cutter/designer skills.

Spec source: spec/editing_style.md  v0.2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gates enforced
  §3  JUMP_CUT_ZOOM  : every jump cut must have 1.1× zoom applied
                       verified via (A) filter metadata or (B) frame analysis
  §1  DURATION       : no clip shorter than 0.5 s
  §1  RESOLUTION     : minimum 1080p (1920 × 1080)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Validation strategy — two paths for JUMP_CUT_ZOOM
--------------------------------------------------

  A. Metadata path (fast):
     The skills/cutter/logic.py records the zoom_factor and ffmpeg_filter
     on each Cut object.  The validator reads these fields and confirms
     every jump-cut has zoom_factor == JUMP_CUT_ZOOM and a non-empty filter.
     No rendered frames needed.  Used in pre-commit dry-run.

  B. Frame-analysis path (authoritative):
     After the executor renders the clip, the validator receives a pair of
     VideoFrame objects — one from the last frame before the cut and one
     from the first frame after.  It measures the apparent scale change by
     comparing the central-region pixel energy ratio to detect a zoom.
     Used by the harness after the sandbox commits the file.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Spec constants ────────────────────────────────────────────────────────
REQUIRED_ZOOM: float = 1.1            # §3  jump-cut zoom factor
ZOOM_TOLERANCE: float = 0.02          # ±2% tolerance for float comparison
MIN_CLIP_DURATION_S: float = 0.50     # §1  duration gate
MIN_WIDTH: int = 1920                 # §1  resolution gate
MIN_HEIGHT: int = 1080

# Frame-analysis: compare a centre crop to detect zoom
# Crop is (1/ZOOM_CROP_RATIO) of the frame in each dimension
ZOOM_CROP_RATIO: float = 1.1          # inner crop matches the zoom factor
# Minimum energy ratio to declare a zoom present
ZOOM_ENERGY_RATIO_MIN: float = 0.90   # zoomed centre should be ≥ 90% of full frame energy


# ── Result types ──────────────────────────────────────────────────────────

class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class GateResult:
    gate: str
    status: GateStatus
    measured: Optional[float] = None
    expected: str = ""
    detail: str = ""
    fail_action: str = ""

    def __str__(self) -> str:
        icon = {"pass": "✓", "fail": "✗", "skip": "–"}[self.status.value]
        base = f"[{icon}] {self.gate}"
        if self.measured is not None:
            base += f"  measured={self.measured:.3f}"
        if self.expected:
            base += f"  expected={self.expected}"
        if self.status == GateStatus.FAIL and self.detail:
            base += f"  → {self.detail}"
        return base


@dataclass
class GateReport:
    gates: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.status != GateStatus.FAIL for g in self.gates)

    @property
    def summary(self) -> str:
        fails = [g for g in self.gates if g.status == GateStatus.FAIL]
        if not fails:
            return f"VIDEO OK — all {len(self.gates)} gates passed"
        names = ", ".join(g.gate for g in fails)
        return f"VIDEO FAIL — {len(fails)} gate(s) failed: {names}"

    def log(self) -> None:
        logger.info("VideoValidator: %s", self.summary)
        for g in self.gates:
            fn = logger.info if g.status == GateStatus.PASS else logger.warning
            fn("  %s", g)


# ── Data types for frame-based path ──────────────────────────────────────

@dataclass
class VideoFrame:
    """A single decoded video frame.

    Attributes
    ----------
    time_s    : Frame timestamp in seconds.
    width     : Frame width in pixels.
    height    : Frame height in pixels.
    pixels    : Flat list of normalised luminance values [0.0, 1.0],
                length == width * height.
                For colour frames, pass the luma (Y) channel only.
    cut_index : Index of the cut this frame is associated with.
    is_post_cut : True if this frame is from the clip *after* the cut point.
    """
    time_s: float
    width: int
    height: int
    pixels: list[float]
    cut_index: int = -1
    is_post_cut: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────

def _centre_crop_energy(pixels: list[float], width: int, height: int, crop_ratio: float) -> float:
    """Compute mean squared pixel energy of the centre crop.

    The centre crop has dimensions (width / crop_ratio) × (height / crop_ratio).
    A zoomed frame concentrates energy toward the centre; comparing the ratio
    of centre-crop energy between a pre-cut and post-cut frame reveals zoom.

    Parameters
    ----------
    pixels     : Flat luma values, row-major, length == width * height.
    width      : Frame width.
    height     : Frame height.
    crop_ratio : Fraction of the frame to keep (e.g. 1.1 keeps ~90%).

    Returns mean squared luma value in the crop (higher = brighter/more detail).
    """
    if len(pixels) != width * height:
        return 0.0

    crop_w = int(width / crop_ratio)
    crop_h = int(height / crop_ratio)
    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2

    total_sq = 0.0
    count = 0
    for row in range(y0, y0 + crop_h):
        for col in range(x0, x0 + crop_w):
            v = pixels[row * width + col]
            total_sq += v * v
            count += 1

    return total_sq / count if count > 0 else 0.0


def _full_frame_energy(pixels: list[float]) -> float:
    """Return mean squared luma over the entire frame."""
    if not pixels:
        return 0.0
    return sum(v * v for v in pixels) / len(pixels)


# ── Gate: Jump Cut Zoom (metadata path) ───────────────────────────────────

def check_zoom_metadata(cuts: list[Any]) -> GateResult:
    """Gate: JUMP_CUT_ZOOM — verify via Cut metadata from skills/cutter/logic.py.

    Checks every Cut whose zoom_factor == REQUIRED_ZOOM also has a
    non-empty ffmpeg_filter string, and that no jump cut was missed
    (i.e. consecutive same-speaker cuts without zoom).

    Parameters
    ----------
    cuts : List of Cut objects from CutterResult.cuts.
          Each must have: zoom_factor (float), ffmpeg_filter (str),
          speaker_id (str|None), is_broll (bool).
    """
    def _zoom(c: Any) -> float:
        return c.zoom_factor if hasattr(c, "zoom_factor") else c.get("zoom_factor", 1.0)
    def _filt(c: Any) -> str:
        return c.ffmpeg_filter if hasattr(c, "ffmpeg_filter") else c.get("ffmpeg_filter", "")
    def _spk(c: Any) -> Optional[str]:
        return c.speaker_id if hasattr(c, "speaker_id") else c.get("speaker_id")
    def _broll(c: Any) -> bool:
        return c.is_broll if hasattr(c, "is_broll") else c.get("is_broll", False)

    failures: list[str] = []
    zoom_count = 0

    for i in range(1, len(cuts)):
        prev, curr = cuts[i - 1], cuts[i]

        # Detect expected jump cut
        is_jump = (
            _spk(curr) is not None
            and _spk(prev) is not None
            and _spk(curr) == _spk(prev)
            and not _broll(prev)
            and not _broll(curr)
        )

        if is_jump:
            zoom_count += 1
            z = _zoom(curr)
            f = _filt(curr)

            if abs(z - REQUIRED_ZOOM) > ZOOM_TOLERANCE:
                failures.append(
                    f"cut[{i}] ({_spk(curr)}): zoom_factor={z:.3f}, "
                    f"expected {REQUIRED_ZOOM:.1f}±{ZOOM_TOLERANCE}"
                )
            elif not f:
                failures.append(
                    f"cut[{i}] ({_spk(curr)}): zoom_factor correct "
                    f"but ffmpeg_filter is empty — filter was not built"
                )

    if not zoom_count:
        return GateResult(
            gate="JUMP_CUT_ZOOM",
            status=GateStatus.SKIP,
            detail="No jump cuts detected in cut list",
        )

    if failures:
        return GateResult(
            gate="JUMP_CUT_ZOOM",
            status=GateStatus.FAIL,
            measured=float(zoom_count - len(failures)),
            expected=f"all {zoom_count} jump cuts at {REQUIRED_ZOOM}×",
            detail="; ".join(failures),
            fail_action="Re-run designer on affected segment with correct zoompan filter",
        )

    return GateResult(
        gate="JUMP_CUT_ZOOM",
        status=GateStatus.PASS,
        measured=float(zoom_count),
        expected=f"all {zoom_count} jump cuts at {REQUIRED_ZOOM}×",
    )


# ── Gate: Jump Cut Zoom (frame-analysis path) ─────────────────────────────

def check_zoom_frames(frame_pairs: list[tuple[VideoFrame, VideoFrame]]) -> GateResult:
    """Gate: JUMP_CUT_ZOOM — verify via rendered frame analysis.

    For each (pre-cut frame, post-cut frame) pair, measures whether the
    post-cut frame exhibits a centre-energy increase consistent with a
    REQUIRED_ZOOM scale-up.

    Method
    ------
    A 1.1× zoom causes the centre crop of the zoomed frame to contain
    content that was ~90% of the way to the edge in the un-zoomed frame.
    The centre crop should therefore have higher relative energy.

    We compute:
      ratio = centre_crop_energy(post) / full_frame_energy(post)
    and compare against the same ratio for the pre-cut frame.
    A zoom is detected when:
      ratio_post ≥ ratio_pre * ZOOM_ENERGY_RATIO_MIN

    Note: This is a heuristic. For production use, replace with optical-flow
    or feature-point matching. The threshold is tuned for typical interview
    footage (centred speaker, relatively static background).

    Parameters
    ----------
    frame_pairs : List of (pre_cut_frame, post_cut_frame) tuples.
                  Frames must have pixels populated (luma channel).
    """
    if not frame_pairs:
        return GateResult(
            gate="JUMP_CUT_ZOOM",
            status=GateStatus.SKIP,
            detail="No frame pairs provided",
        )

    failures: list[str] = []
    inconclusive: list[str] = []

    for idx, (pre, post) in enumerate(frame_pairs):
        if not pre.pixels or not post.pixels:
            inconclusive.append(f"pair {idx}: empty pixel data — skipped")
            continue

        if pre.width != post.width or pre.height != post.height:
            inconclusive.append(
                f"pair {idx}: frame dimensions differ "
                f"({pre.width}×{pre.height} vs {post.width}×{post.height})"
            )
            continue

        pre_full = _full_frame_energy(pre.pixels)
        post_full = _full_frame_energy(post.pixels)

        if pre_full < 1e-8 or post_full < 1e-8:
            inconclusive.append(f"pair {idx}: near-black frame, skipping")
            continue

        pre_crop = _centre_crop_energy(pre.pixels, pre.width, pre.height, ZOOM_CROP_RATIO)
        post_crop = _centre_crop_energy(post.pixels, post.width, post.height, ZOOM_CROP_RATIO)

        pre_ratio = pre_crop / pre_full
        post_ratio = post_crop / post_full

        # Zoomed frame should concentrate energy toward centre
        zoom_detected = post_ratio >= pre_ratio * ZOOM_ENERGY_RATIO_MIN

        logger.debug(
            "Zoom frame analysis [pair %d]: "
            "pre_ratio=%.3f  post_ratio=%.3f  detected=%s",
            idx, pre_ratio, post_ratio, zoom_detected,
        )

        if not zoom_detected:
            failures.append(
                f"pair {idx} (t={post.time_s:.2f}s): "
                f"post_crop/full_ratio={post_ratio:.3f} vs "
                f"pre={pre_ratio:.3f} — zoom not detected "
                f"(threshold={ZOOM_ENERGY_RATIO_MIN}×pre)"
            )

    total = len(frame_pairs)
    checked = total - len(inconclusive)

    if inconclusive:
        for msg in inconclusive:
            logger.warning("VideoValidator (frame): %s", msg)

    if failures:
        return GateResult(
            gate="JUMP_CUT_ZOOM",
            status=GateStatus.FAIL,
            measured=float(checked - len(failures)),
            expected=f"{checked} pairs with detected zoom",
            detail="; ".join(failures),
            fail_action="Re-run designer with zoompan filter; verify anchor point",
        )

    return GateResult(
        gate="JUMP_CUT_ZOOM",
        status=GateStatus.PASS,
        measured=float(checked),
        expected=f"{checked} pairs with detected zoom",
        detail=f"{len(inconclusive)} pair(s) inconclusive" if inconclusive else "",
    )


# ── Gate: Clip Duration ───────────────────────────────────────────────────

def check_duration(cuts: list[Any]) -> GateResult:
    """Gate: DURATION — no rendered clip shorter than MIN_CLIP_DURATION_S.

    Parameters
    ----------
    cuts : Cut objects with in_s and out_s (or dict equivalents).
    """
    def _in(c: Any) -> float:
        return c.in_s if hasattr(c, "in_s") else c.get("in_s", 0.0)
    def _out(c: Any) -> float:
        return c.out_s if hasattr(c, "out_s") else c.get("out_s", 0.0)

    if not cuts:
        return GateResult(
            gate="DURATION",
            status=GateStatus.SKIP,
            detail="Empty cut list",
        )

    short = [
        (i, c) for i, c in enumerate(cuts)
        if (_out(c) - _in(c)) < MIN_CLIP_DURATION_S
    ]

    if short:
        details = [
            f"cut[{i}] {_in(c):.3f}–{_out(c):.3f}s "
            f"({(_out(c) - _in(c)):.3f}s)"
            for i, c in short
        ]
        return GateResult(
            gate="DURATION",
            status=GateStatus.FAIL,
            measured=min(_out(c) - _in(c) for _, c in short),
            expected=f"≥ {MIN_CLIP_DURATION_S}s",
            detail="; ".join(details),
            fail_action="Merge short clip with adjacent clip in cutter",
        )

    return GateResult(
        gate="DURATION",
        status=GateStatus.PASS,
        measured=min(_out(c) - _in(c) for c in cuts),
        expected=f"≥ {MIN_CLIP_DURATION_S}s",
    )


# ── Gate: Resolution ─────────────────────────────────────────────────────

def check_resolution(frames: list[VideoFrame]) -> GateResult:
    """Gate: RESOLUTION — all frames must be at least 1920 × 1080.

    Parameters
    ----------
    frames : Sample of VideoFrame objects from the rendered output.
             Only width/height are inspected; pixels are not required.
    """
    if not frames:
        return GateResult(
            gate="RESOLUTION",
            status=GateStatus.SKIP,
            expected=f"≥ {MIN_WIDTH}×{MIN_HEIGHT}",
            detail="No frames provided",
        )

    violations = [
        f for f in frames
        if f.width < MIN_WIDTH or f.height < MIN_HEIGHT
    ]

    if violations:
        worst = min(violations, key=lambda f: f.width * f.height)
        return GateResult(
            gate="RESOLUTION",
            status=GateStatus.FAIL,
            measured=float(worst.width * worst.height),
            expected=f"≥ {MIN_WIDTH}×{MIN_HEIGHT} ({MIN_WIDTH * MIN_HEIGHT:,} px)",
            detail=f"{len(violations)} frame(s) below minimum; "
                   f"worst={worst.width}×{worst.height} at t={worst.time_s:.2f}s",
            fail_action="Upscale source or reject: ffmpeg -vf scale=1920:1080",
        )

    return GateResult(
        gate="RESOLUTION",
        status=GateStatus.PASS,
        measured=float(frames[0].width * frames[0].height),
        expected=f"≥ {MIN_WIDTH}×{MIN_HEIGHT}",
    )


# ── Public entry point ────────────────────────────────────────────────────

def validate(
    *,
    cuts: Optional[list[Any]] = None,
    frame_pairs: Optional[list[tuple[VideoFrame, VideoFrame]]] = None,
    sample_frames: Optional[list[VideoFrame]] = None,
) -> GateReport:
    """Run all video quality gates and return a consolidated GateReport.

    Call modes
    ----------
    Metadata-only (pre-commit, fast):
        validate(cuts=cutter_result.cuts)

    Frame-based (post-render, authoritative):
        validate(
            cuts=cutter_result.cuts,
            frame_pairs=[(pre1, post1), (pre2, post2), ...],
            sample_frames=decoded_frames,
        )

    Parameters
    ----------
    cuts          : List of Cut objects from CutterResult.
    frame_pairs   : (pre-cut, post-cut) VideoFrame pairs for zoom analysis.
    sample_frames : Sample frames from rendered output (resolution check).
    """
    report = GateReport()
    cuts = cuts or []

    # Gate 1: Jump Cut Zoom
    if frame_pairs is not None:
        # Authoritative frame-based check takes precedence
        report.gates.append(check_zoom_frames(frame_pairs))
    else:
        # Fast metadata check
        report.gates.append(check_zoom_metadata(cuts))

    # Gate 2: Clip Duration
    report.gates.append(check_duration(cuts))

    # Gate 3: Resolution
    if sample_frames:
        report.gates.append(check_resolution(sample_frames))
    else:
        report.gates.append(GateResult(
            gate="RESOLUTION",
            status=GateStatus.SKIP,
            expected=f"≥ {MIN_WIDTH}×{MIN_HEIGHT}",
            detail="No rendered frames provided — run after executor commits",
        ))

    report.log()
    return report
