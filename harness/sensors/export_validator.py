"""
harness/sensors/export_validator.py

Final export quality sensor — post-render gates for the Exporter skill.

Spec source: spec/editing_style.md  v0.2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gates enforced
  ASPECT_RATIO   : output must be 9:16 (portrait / YouTube Shorts)
  AUDIO_CLIPPING : no sample may reach or exceed 0.999 FS (clipping)
  OUTPUT_EXISTS  : the rendered output file must be present on disk
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Spec constants ────────────────────────────────────────────────────────
TARGET_ASPECT_RATIO: str = "9:16"        # portrait for YouTube Shorts
CLIPPING_THRESHOLD: float = 0.999        # samples ≥ this level are clipped
MIN_OUTPUT_FILE_SIZE_BYTES: int = 1024   # anything smaller is likely corrupted


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
            base += f"  measured={self.measured:.4f}"
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
            return f"EXPORT OK — all {len(self.gates)} gates passed"
        names = ", ".join(g.gate for g in fails)
        return f"EXPORT FAIL — {len(fails)} gate(s) failed: {names}"

    def log(self) -> None:
        logger.info("ExportValidator: %s", self.summary)
        for g in self.gates:
            fn = logger.info if g.status == GateStatus.PASS else logger.warning
            fn("  %s", g)


# ── Helpers ───────────────────────────────────────────────────────────────

def _parse_ratio(ratio_str: str) -> Fraction:
    """Parse 'W:H' or 'W/H' into a Fraction for exact comparison."""
    sep = ":" if ":" in ratio_str else "/"
    w_s, h_s = ratio_str.split(sep, 1)
    return Fraction(int(w_s), int(h_s))


# ── Gate: Aspect Ratio ────────────────────────────────────────────────────

def check_aspect_ratio(
    width: int,
    height: int,
    *,
    expected_ratio: str = TARGET_ASPECT_RATIO,
    tolerance: float = 0.01,
) -> GateResult:
    """Gate: ASPECT_RATIO — output dimensions must match the required ratio.

    Parameters
    ----------
    width           : Output frame width in pixels.
    height          : Output frame height in pixels.
    expected_ratio  : Target ratio as "W:H" (default "9:16" for Shorts).
    tolerance       : Allowed fractional deviation from the target ratio.

    Notes
    -----
    "9:16" (portrait) means width/height = 9/16 ≈ 0.5625.
    Example valid resolutions: 1080×1920, 720×1280, 540×960.
    """
    if width <= 0 or height <= 0:
        return GateResult(
            gate="ASPECT_RATIO",
            status=GateStatus.FAIL,
            expected=expected_ratio,
            detail=f"Invalid dimensions: {width}×{height}",
            fail_action="Re-encode with correct -s WxH dimensions",
        )

    target = _parse_ratio(expected_ratio)
    measured_ratio = width / height
    target_float = float(target)

    deviation = abs(measured_ratio - target_float) / target_float

    if deviation <= tolerance:
        return GateResult(
            gate="ASPECT_RATIO",
            status=GateStatus.PASS,
            measured=measured_ratio,
            expected=f"{expected_ratio} (≈{target_float:.4f})",
        )

    # Determine what the ratio actually is (for reporting)
    gcd_val = math.gcd(width, height)
    actual_ratio = f"{width // gcd_val}:{height // gcd_val}"

    return GateResult(
        gate="ASPECT_RATIO",
        status=GateStatus.FAIL,
        measured=measured_ratio,
        expected=f"{expected_ratio} (≈{target_float:.4f}, ±{tolerance*100:.0f}%)",
        detail=(
            f"Output is {width}×{height} ({actual_ratio}), "
            f"deviation={deviation*100:.1f}% from {expected_ratio}"
        ),
        fail_action=(
            f"Re-encode with correct aspect ratio: "
            f"ffmpeg -vf 'scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2'"
        ),
    )


# ── Gate: Audio Clipping ──────────────────────────────────────────────────

def check_audio_clipping(
    samples: list[float],
    *,
    threshold: float = CLIPPING_THRESHOLD,
) -> GateResult:
    """Gate: AUDIO_CLIPPING — no sample should reach or exceed the clipping threshold.

    Parameters
    ----------
    samples   : Normalised PCM samples in [-1.0, 1.0].
                Pass a representative slice from the rendered output.
    threshold : Level at or above which clipping is declared (default 0.999).

    Notes
    -----
    True digital clipping occurs at 1.0 FS.  The threshold of 0.999 FS
    (≈ -0.009 dBFS) provides a safety margin for quantisation error.
    """
    if not samples:
        return GateResult(
            gate="AUDIO_CLIPPING",
            status=GateStatus.SKIP,
            expected=f"< {threshold:.3f} FS",
            detail="No audio samples provided — skipped",
        )

    clipped = [s for s in samples if abs(s) >= threshold]
    peak = max(abs(s) for s in samples)

    if clipped:
        clipping_pct = len(clipped) / len(samples) * 100.0
        peak_dbfs = 20.0 * math.log10(peak) if peak > 0 else -math.inf
        return GateResult(
            gate="AUDIO_CLIPPING",
            status=GateStatus.FAIL,
            measured=peak,
            expected=f"< {threshold:.3f} FS",
            detail=(
                f"{len(clipped)} clipped sample(s) "
                f"({clipping_pct:.2f}% of {len(samples)}); "
                f"peak={peak:.4f} FS ({peak_dbfs:.1f} dBFS)"
            ),
            fail_action="Apply limiting: ffmpeg -af 'alimiter=limit=0.9:attack=5:release=50'",
        )

    return GateResult(
        gate="AUDIO_CLIPPING",
        status=GateStatus.PASS,
        measured=peak,
        expected=f"< {threshold:.3f} FS",
    )


# ── Gate: Output File Exists ──────────────────────────────────────────────

def check_output_exists(output_path: str) -> GateResult:
    """Gate: OUTPUT_EXISTS — the rendered output file must be present and non-trivial.

    Parameters
    ----------
    output_path : Absolute path to the rendered output file.
    """
    path = Path(output_path)

    if not path.exists():
        return GateResult(
            gate="OUTPUT_EXISTS",
            status=GateStatus.FAIL,
            expected="file present on disk",
            detail=f"Output file not found: {output_path}",
            fail_action="Re-run exporter; check FFmpeg stderr for render errors",
        )

    size_bytes = path.stat().st_size
    if size_bytes < MIN_OUTPUT_FILE_SIZE_BYTES:
        return GateResult(
            gate="OUTPUT_EXISTS",
            status=GateStatus.FAIL,
            measured=float(size_bytes),
            expected=f"≥ {MIN_OUTPUT_FILE_SIZE_BYTES} bytes",
            detail=f"Output file is suspiciously small: {size_bytes} bytes",
            fail_action="Re-run exporter; the render may have failed silently",
        )

    return GateResult(
        gate="OUTPUT_EXISTS",
        status=GateStatus.PASS,
        measured=float(size_bytes),
        expected=f"≥ {MIN_OUTPUT_FILE_SIZE_BYTES} bytes",
    )


# ── Public entry point ────────────────────────────────────────────────────

def validate(
    *,
    width: int = 0,
    height: int = 0,
    samples: Optional[list[float]] = None,
    output_path: Optional[str] = None,
    expected_ratio: str = TARGET_ASPECT_RATIO,
) -> GateReport:
    """Run all final export quality gates and return a consolidated GateReport.

    Call modes
    ----------
    Metadata-only (pre-render, fast):
        validate(width=1080, height=1920)   # aspect ratio only

    Full post-render:
        validate(
            width=1080,
            height=1920,
            samples=rendered_audio_samples,
            output_path="/path/to/output.mp4",
        )

    Parameters
    ----------
    width           : Output frame width.
    height          : Output frame height.
    samples         : Rendered audio samples (normalised PCM).
    output_path     : Path to rendered output file.
    expected_ratio  : Aspect ratio requirement (default "9:16").
    """
    report = GateReport()

    # Gate 1: Aspect Ratio
    if width > 0 or height > 0:
        report.gates.append(
            check_aspect_ratio(width, height, expected_ratio=expected_ratio)
        )
    else:
        report.gates.append(GateResult(
            gate="ASPECT_RATIO",
            status=GateStatus.SKIP,
            expected=expected_ratio,
            detail="No dimensions provided",
        ))

    # Gate 2: Audio Clipping
    report.gates.append(
        check_audio_clipping(samples or [])
    )

    # Gate 3: Output File Exists
    if output_path is not None:
        report.gates.append(check_output_exists(output_path))
    else:
        report.gates.append(GateResult(
            gate="OUTPUT_EXISTS",
            status=GateStatus.SKIP,
            detail="No output path provided — run after render completes",
        ))

    report.log()
    return report
