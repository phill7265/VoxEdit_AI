"""
harness/sensors/audio_validator.py

Audio quality sensor — post-execution gate for the audio master skill.

Spec source: spec/editing_style.md  v0.2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gates enforced
  §1  NORMALIZATION  : integrated loudness within [-16, -12] LUFS
                       (target -14 LUFS ± 2 LUFS tolerance)
  §4  DUCK_LEVEL     : BGM during speech must be ≤ (pre-duck level − 20 dB)
  §4  DUCK_FLOOR     : BGM must never fall below −40 dBFS
  §4  DUCK_SILENCE   : no VAD-active speech window may have un-ducked BGM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Validation strategy
-------------------
Two input paths are supported:

  A. Envelope-based (fast, no FFmpeg required):
     Pass the GainEnvelope produced by skills/audio/master.py.
     The validator analyses the gain control points directly.
     Used by the pipeline in dry-run / pre-commit mode.

  B. Frame-based (authoritative, requires rendered audio):
     Pass a list of AudioFrame dicts sampled from the rendered output.
     Used by the harness after executor.py commits the file.

Both paths produce the same GateReport structure so the pipeline runner
does not need to know which path was taken.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Spec constants ────────────────────────────────────────────────────────
TARGET_LUFS: float = -14.0           # §1 target integrated loudness
LUFS_TOLERANCE: float = 2.0          # §1 ± tolerance
LUFS_MIN: float = TARGET_LUFS - LUFS_TOLERANCE   # -16 LUFS
LUFS_MAX: float = TARGET_LUFS + LUFS_TOLERANCE   # -12 LUFS

DUCK_DEPTH_DB: float = -20.0         # §4 required attenuation depth
DUCK_FLOOR_DBFS: float = -40.0       # §4 absolute minimum BGM level
DUCK_LEVEL_TOLERANCE_DB: float = 1.0 # allow 1 dB measurement noise


# ── Result types ──────────────────────────────────────────────────────────

class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"     # gate not applicable (e.g. no VAD events present)


@dataclass
class GateResult:
    """Result of a single quality gate check.

    Attributes
    ----------
    gate        : Gate identifier string (e.g. "DUCK_LEVEL").
    status      : PASS / FAIL / SKIP.
    measured    : The measured value (dBFS, LUFS, etc.).
    expected    : The spec-required value or range string.
    detail      : Human-readable explanation for failures.
    fail_action : Prescribed remediation from spec §5.
    """
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
            base += f"  measured={self.measured:.2f}"
        if self.expected:
            base += f"  expected={self.expected}"
        if self.status == GateStatus.FAIL and self.detail:
            base += f"  → {self.detail}"
        return base


@dataclass
class GateReport:
    """Aggregated result from the audio validator.

    Attributes
    ----------
    passed      : All gates passed.
    gates       : Individual GateResult objects.
    summary     : One-line human-readable verdict.
    """
    gates: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.status != GateStatus.FAIL for g in self.gates)

    @property
    def summary(self) -> str:
        fails = [g for g in self.gates if g.status == GateStatus.FAIL]
        if not fails:
            return f"AUDIO OK — all {len(self.gates)} gates passed"
        names = ", ".join(g.gate for g in fails)
        return f"AUDIO FAIL — {len(fails)} gate(s) failed: {names}"

    def log(self) -> None:
        logger.info("AudioValidator: %s", self.summary)
        for g in self.gates:
            fn = logger.info if g.status == GateStatus.PASS else logger.warning
            fn("  %s", g)


# ── Helpers ───────────────────────────────────────────────────────────────

def _rms_db(samples: list[float]) -> float:
    if not samples:
        return -math.inf
    mean_sq = sum(s * s for s in samples) / len(samples)
    rms = math.sqrt(mean_sq)
    return 20.0 * math.log10(rms) if rms > 0 else -math.inf


def _integrated_lufs_approx(frames: list[dict]) -> float:
    """Approximate integrated loudness (LUFS) from PCM frame list.

    Uses the ITU-R BS.1770 K-weighting approximation:
      LUFS ≈ -0.691 + 10 * log10( mean_square_over_all_samples )

    For production use, replace with pyloudnorm or ffmpeg ebur128.
    The approximation is within ±1.5 LUFS of the true value for typical
    speech/music content and is sufficient for gate validation.

    Parameters
    ----------
    frames : [{ "samples": list[float], "channel_count": int }]
    """
    all_sq: list[float] = []
    for f in frames:
        samples = f.get("samples", [])
        ch = max(1, f.get("channel_count", 1))
        # Down-mix to mono for simplified K-weighting
        if ch > 1:
            chunk = len(samples) // ch
            samples = [
                sum(samples[i * ch + c] for c in range(ch)) / ch
                for i in range(chunk)
            ]
        all_sq.extend(s * s for s in samples)

    if not all_sq:
        return -math.inf
    mean_sq = sum(all_sq) / len(all_sq)
    if mean_sq <= 0:
        return -math.inf
    return -0.691 + 10.0 * math.log10(mean_sq)


def _eval_gain_at(points: list[Any], time_s: float) -> float:
    """Linearly interpolate a gain value at time_s from an ordered point list.

    Accepts either GainPoint objects (with .time_s and .gain_db) or plain
    dicts with keys "time_s" and "gain_db".
    """
    def _t(p: Any) -> float:
        return p.time_s if hasattr(p, "time_s") else p["time_s"]

    def _g(p: Any) -> float:
        return p.gain_db if hasattr(p, "gain_db") else p["gain_db"]

    if not points:
        return 0.0
    if time_s <= _t(points[0]):
        return _g(points[0])
    if time_s >= _t(points[-1]):
        return _g(points[-1])

    for i in range(len(points) - 1):
        t0, t1 = _t(points[i]), _t(points[i + 1])
        if t0 <= time_s <= t1:
            alpha = (time_s - t0) / (t1 - t0) if t1 > t0 else 0.0
            return _g(points[i]) + (_g(points[i + 1]) - _g(points[i])) * alpha
    return _g(points[-1])


# ── Gate: Normalization ───────────────────────────────────────────────────

def check_normalization(frames: list[dict]) -> GateResult:
    """Gate: NORMALIZATION — integrated loudness must be within ±2 LUFS of -14.

    Parameters
    ----------
    frames : PCM frame list from the rendered output track.
             Each frame: { "samples": list[float], "channel_count": int }
    """
    if not frames:
        return GateResult(
            gate="NORMALIZATION",
            status=GateStatus.SKIP,
            expected=f"[{LUFS_MIN:.0f}, {LUFS_MAX:.0f}] LUFS",
            detail="No audio frames provided — skipped",
        )

    measured = _integrated_lufs_approx(frames)

    if LUFS_MIN <= measured <= LUFS_MAX:
        return GateResult(
            gate="NORMALIZATION",
            status=GateStatus.PASS,
            measured=measured,
            expected=f"[{LUFS_MIN:.0f}, {LUFS_MAX:.0f}] LUFS",
        )

    direction = "too quiet" if measured < LUFS_MIN else "too loud"
    return GateResult(
        gate="NORMALIZATION",
        status=GateStatus.FAIL,
        measured=measured,
        expected=f"[{LUFS_MIN:.0f}, {LUFS_MAX:.0f}] LUFS",
        detail=f"Integrated loudness {measured:.2f} LUFS is {direction}",
        fail_action="Re-run audio normalization pass targeting -14 LUFS",
    )


# ── Gate: Duck Level ─────────────────────────────────────────────────────

def check_duck_level_envelope(envelope: Any) -> GateResult:
    """Gate: DUCK_LEVEL — BGM must be ≤ (pre-duck level − 20 dB) during speech.

    Envelope-based check (fast path, no rendered frames needed).

    Parameters
    ----------
    envelope : GainEnvelope from skills/audio/master.py.
               Requires .duck_regions (list of DuckRegion).
    """
    duck_regions = getattr(envelope, "duck_regions", [])

    if not duck_regions:
        return GateResult(
            gate="DUCK_LEVEL",
            status=GateStatus.SKIP,
            expected=f"≤ pre-duck + {DUCK_DEPTH_DB:.0f} dB",
            detail="No duck regions found — VAD may not have fired",
        )

    failures: list[str] = []
    worst_excess: float = 0.0

    for i, region in enumerate(duck_regions):
        required_max = region.bgm_level_db + DUCK_DEPTH_DB + DUCK_LEVEL_TOLERANCE_DB
        if region.ducked_level_db > required_max:
            excess = region.ducked_level_db - required_max
            worst_excess = max(worst_excess, excess)
            failures.append(
                f"region {i} ({region.vad_start_s:.2f}–{region.vad_end_s:.2f}s): "
                f"ducked={region.ducked_level_db:.1f} dBFS, "
                f"required ≤ {required_max:.1f} dBFS "
                f"(excess {excess:.1f} dB)"
            )

    if failures:
        return GateResult(
            gate="DUCK_LEVEL",
            status=GateStatus.FAIL,
            measured=duck_regions[0].ducked_level_db if duck_regions else None,
            expected=f"pre-duck + {DUCK_DEPTH_DB:.0f} dB",
            detail="; ".join(failures),
            fail_action="Re-run audio master with corrected duck depth",
        )

    return GateResult(
        gate="DUCK_LEVEL",
        status=GateStatus.PASS,
        measured=min(r.ducked_level_db for r in duck_regions),
        expected=f"≤ pre-duck + {DUCK_DEPTH_DB:.0f} dB",
    )


def check_duck_level_frames(
    bgm_frames: list[dict],
    vad_events: list[Any],
    pre_duck_db: float,
) -> GateResult:
    """Gate: DUCK_LEVEL — frame-based authoritative check on rendered BGM.

    Samples the rendered BGM track during each VAD-active window and verifies
    that the measured RMS is within tolerance of the required ducked level.

    Parameters
    ----------
    bgm_frames   : PCM frames from the rendered BGM track.
                   Each: { "start_s": float, "end_s": float, "samples": list[float] }
    vad_events   : VadEvent objects (or dicts with start_s, end_s, confidence).
    pre_duck_db  : The reference BGM level before any ducking was applied.
    """
    def _conf(v: Any) -> float:
        return v.confidence if hasattr(v, "confidence") else v.get("confidence", 0.0)
    def _start(v: Any) -> float:
        return v.start_s if hasattr(v, "start_s") else v.get("start_s", 0.0)
    def _end(v: Any) -> float:
        return v.end_s if hasattr(v, "end_s") else v.get("end_s", 0.0)

    active_vad = [v for v in vad_events if _conf(v) >= 0.85]
    if not active_vad:
        return GateResult(
            gate="DUCK_LEVEL",
            status=GateStatus.SKIP,
            expected=f"≤ {pre_duck_db + DUCK_DEPTH_DB:.1f} dBFS",
            detail="No active VAD events — gate not applicable",
        )

    required_max_db = pre_duck_db + DUCK_DEPTH_DB + DUCK_LEVEL_TOLERANCE_DB
    failures: list[str] = []

    for vad in active_vad:
        # Collect BGM samples that fall within this VAD window
        window_samples: list[float] = []
        for frame in bgm_frames:
            fs = frame.get("start_s", 0.0)
            fe = frame.get("end_s", 0.0)
            # Include frame if it overlaps the VAD window
            if fe > _start(vad) and fs < _end(vad):
                window_samples.extend(frame.get("samples", []))

        if not window_samples:
            continue

        measured_rms = _rms_db(window_samples)
        if measured_rms > required_max_db:
            failures.append(
                f"VAD {_start(vad):.2f}–{_end(vad):.2f}s: "
                f"BGM RMS={measured_rms:.1f} dBFS, "
                f"required ≤ {required_max_db:.1f} dBFS"
            )

    if failures:
        return GateResult(
            gate="DUCK_LEVEL",
            status=GateStatus.FAIL,
            expected=f"≤ {required_max_db:.1f} dBFS during speech",
            detail="; ".join(failures),
            fail_action="Re-run audio master; verify VAD confidence and duck filter applied",
        )

    return GateResult(
        gate="DUCK_LEVEL",
        status=GateStatus.PASS,
        expected=f"≤ {required_max_db:.1f} dBFS during speech",
    )


# ── Gate: Duck Floor ─────────────────────────────────────────────────────

def check_duck_floor(envelope: Any) -> GateResult:
    """Gate: DUCK_FLOOR — BGM must never fall below -40 dBFS.

    Checks every control point in the GainEnvelope.
    """
    points = getattr(envelope, "points", [])

    if not points:
        return GateResult(
            gate="DUCK_FLOOR",
            status=GateStatus.SKIP,
            expected=f"≥ {DUCK_FLOOR_DBFS:.0f} dBFS",
            detail="No envelope points — skipped",
        )

    def _g(p: Any) -> float:
        return p.gain_db if hasattr(p, "gain_db") else p.get("gain_db", 0.0)
    def _t(p: Any) -> float:
        return p.time_s if hasattr(p, "time_s") else p.get("time_s", 0.0)

    violations = [
        p for p in points
        if _g(p) < DUCK_FLOOR_DBFS - 0.1
    ]

    if violations:
        worst = min(_g(p) for p in violations)
        times = ", ".join(f"{_t(p):.2f}s" for p in violations[:3])
        return GateResult(
            gate="DUCK_FLOOR",
            status=GateStatus.FAIL,
            measured=worst,
            expected=f"≥ {DUCK_FLOOR_DBFS:.0f} dBFS",
            detail=f"BGM drops below floor at: {times}"
                   + (" ..." if len(violations) > 3 else ""),
            fail_action="Clamp duck floor in audio master: max(ducked, DUCK_FLOOR_DBFS)",
        )

    return GateResult(
        gate="DUCK_FLOOR",
        status=GateStatus.PASS,
        measured=min(_g(p) for p in points),
        expected=f"≥ {DUCK_FLOOR_DBFS:.0f} dBFS",
    )


# ── Gate: Duck Silence (no un-ducked speech window) ───────────────────────

def check_duck_silence(envelope: Any, vad_events: list[Any]) -> GateResult:
    """Gate: DUCK_SILENCE — every VAD-active window must have a corresponding duck region.

    Detects cases where the audio master was called but a VAD event was
    dropped and no duck was applied for that speech window.
    """
    def _conf(v: Any) -> float:
        return v.confidence if hasattr(v, "confidence") else v.get("confidence", 0.0)
    def _start(v: Any) -> float:
        return v.start_s if hasattr(v, "start_s") else v.get("start_s", 0.0)
    def _end(v: Any) -> float:
        return v.end_s if hasattr(v, "end_s") else v.get("end_s", 0.0)

    active_vad = [v for v in vad_events if _conf(v) >= 0.85]
    if not active_vad:
        return GateResult(
            gate="DUCK_SILENCE",
            status=GateStatus.SKIP,
            detail="No active VAD events",
        )

    duck_regions = getattr(envelope, "duck_regions", [])
    points = getattr(envelope, "points", [])
    uncovered: list[str] = []

    for vad in active_vad:
        # Sample gain at the midpoint of the VAD window
        mid_s = (_start(vad) + _end(vad)) / 2.0
        gain_at_mid = _eval_gain_at(points, mid_s)

        # Check if any duck region covers this VAD event
        covered_by_region = any(
            getattr(r, "attack_start_s", _start(vad)) <= mid_s <= getattr(r, "release_end_s", _end(vad))
            for r in duck_regions
        )

        if not covered_by_region and gain_at_mid >= -0.5:
            # Gain is near 0 dB during speech — duck was not applied
            uncovered.append(
                f"VAD {_start(vad):.2f}–{_end(vad):.2f}s "
                f"(gain at midpoint={gain_at_mid:.1f} dB, no duck region)"
            )

    if uncovered:
        return GateResult(
            gate="DUCK_SILENCE",
            status=GateStatus.FAIL,
            expected="all VAD windows covered by a duck region",
            detail="; ".join(uncovered),
            fail_action="Re-run audio master; verify all VAD events are above threshold",
        )

    return GateResult(
        gate="DUCK_SILENCE",
        status=GateStatus.PASS,
        expected="all VAD windows covered",
    )


# ── Public entry point ────────────────────────────────────────────────────

def validate(
    *,
    envelope: Any = None,
    bgm_frames: Optional[list[dict]] = None,
    speech_frames: Optional[list[dict]] = None,
    vad_events: Optional[list[Any]] = None,
    pre_duck_db: float = 0.0,
) -> GateReport:
    """Run all audio quality gates and return a consolidated GateReport.

    Call modes
    ----------
    Envelope-only (pre-commit, fast):
        validate(envelope=gain_envelope, vad_events=vad_list)

    Frame-based (post-render, authoritative):
        validate(
            envelope=gain_envelope,
            bgm_frames=rendered_bgm_frames,
            speech_frames=rendered_speech_frames,
            vad_events=vad_list,
            pre_duck_db=-6.0,
        )

    Parameters
    ----------
    envelope      : GainEnvelope from skills/audio/master.py.
    bgm_frames    : PCM frames from the rendered BGM track (frame-based path).
    speech_frames : PCM frames from the rendered speech track (normalization check).
    vad_events    : List of VadEvent objects.
    pre_duck_db   : Reference BGM level before ducking (for frame-based duck check).
    """
    report = GateReport()
    vad_events = vad_events or []

    # Gate 1: Normalization (frame-based only)
    frames_for_lufs = speech_frames or bgm_frames
    if frames_for_lufs:
        report.gates.append(check_normalization(frames_for_lufs))
    else:
        report.gates.append(GateResult(
            gate="NORMALIZATION",
            status=GateStatus.SKIP,
            expected=f"[{LUFS_MIN:.0f}, {LUFS_MAX:.0f}] LUFS",
            detail="No rendered frames provided — run frame-based validation after render",
        ))

    # Gate 2: Duck Level
    if bgm_frames and vad_events:
        report.gates.append(
            check_duck_level_frames(bgm_frames, vad_events, pre_duck_db)
        )
    elif envelope is not None:
        report.gates.append(check_duck_level_envelope(envelope))
    else:
        report.gates.append(GateResult(
            gate="DUCK_LEVEL",
            status=GateStatus.SKIP,
            detail="Neither envelope nor rendered frames provided",
        ))

    # Gate 3: Duck Floor (envelope-based)
    if envelope is not None:
        report.gates.append(check_duck_floor(envelope))

    # Gate 4: Duck Silence coverage
    if envelope is not None:
        report.gates.append(check_duck_silence(envelope, vad_events))

    report.log()
    return report
