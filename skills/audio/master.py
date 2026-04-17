"""
skills/audio/master.py

Audio Ducking master — VAD-triggered background music attenuation.

Spec source: spec/editing_style.md  v0.2.0  §4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rule §4  Audio Ducking
  · Trigger  : VAD confidence ≥ 0.85 on primary speaker track
  · Duck by  : −20 dB relative to current music level
  · Attack   : 150 ms (ramp DOWN to ducked level)
  · Release  : 500 ms (ramp UP back to original level)
  · Floor    : −40 dBFS minimum (never drive music to silence)
  · Scope    : BGM track only; SFX tracks are unaffected
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture note
-----------------
This module is STATELESS.  It:
  1. Accepts a sequence of VAD events and a BGM level timeline.
  2. Produces a GainEnvelope — a time-ordered list of (time_s, gain_db) points
     that describe the exact BGM volume curve.
  3. Converts the envelope to an FFmpeg `volume` filter expression.
  4. Runs harness sensors to verify the duck depth and ADSR shape.

The harness/sandbox/executor.py applies the FFmpeg filter; this module
never touches audio files directly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Spec constants ────────────────────────────────────────────────────────
VAD_CONFIDENCE_THRESHOLD: float = 0.85   # §4 trigger
DUCK_DB: float = -20.0                   # §4 attenuation depth
ATTACK_MS: float = 150.0                 # §4 attack  (ramp to ducked level)
RELEASE_MS: float = 500.0               # §4 release (ramp back to original)
DUCK_FLOOR_DBFS: float = -40.0          # §4 absolute minimum ducked level
BGM_DEFAULT_DB: float = 0.0             # reference level if not specified


# ── Helpers ───────────────────────────────────────────────────────────────

def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)

def _linear_to_db(linear: float) -> float:
    if linear <= 0.0:
        return -math.inf
    return 20.0 * math.log10(linear)

def _lerp_db(start_db: float, end_db: float, t: float) -> float:
    """Linear interpolation in dB domain (perceptually linear).

    t ∈ [0, 1]  where 0 = start_db, 1 = end_db.
    Clamped to [0, 1].
    """
    t = max(0.0, min(1.0, t))
    return start_db + (end_db - start_db) * t


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class VadEvent:
    """A single VAD detection on the primary speaker track.

    Attributes
    ----------
    start_s    : Speech onset in seconds.
    end_s      : Speech offset in seconds.
    confidence : Model confidence [0.0, 1.0].
    """
    start_s: float
    end_s: float
    confidence: float

    @property
    def is_active(self) -> bool:
        return self.confidence >= VAD_CONFIDENCE_THRESHOLD


@dataclass
class GainPoint:
    """A single point on the BGM gain envelope.

    Attributes
    ----------
    time_s  : Position in seconds on the timeline.
    gain_db : BGM level at this point in dBFS (relative to 0 dBFS = no change).
    phase   : Human-readable phase label for debugging.
    """
    time_s: float
    gain_db: float
    phase: str = ""


@dataclass
class DuckRegion:
    """One fully-processed duck event with all envelope control points.

    Attributes
    ----------
    vad_start_s    : VAD onset (speech begins).
    vad_end_s      : VAD offset (speech ends).
    bgm_level_db   : Pre-duck BGM level at this position.
    ducked_level_db: Clamped ducked level (≥ DUCK_FLOOR_DBFS).
    attack_start_s : When the gain ramp-down begins.
    attack_end_s   : When the gain reaches the ducked level.
    release_start_s: When the gain ramp-up begins.
    release_end_s  : When the gain returns to bgm_level_db.
    envelope       : Ordered GainPoints for this region.
    """
    vad_start_s: float
    vad_end_s: float
    bgm_level_db: float
    ducked_level_db: float
    attack_start_s: float
    attack_end_s: float
    release_start_s: float
    release_end_s: float
    envelope: list[GainPoint] = field(default_factory=list)


@dataclass
class GainEnvelope:
    """Complete BGM gain envelope for a timeline segment.

    Attributes
    ----------
    points      : Ordered (time_s, gain_db) control points.
    duck_regions: The individual duck events that shaped the envelope.
    sensor_flags: Advisory flags from run_sensors().
    ffmpeg_expr : The `volume` filter expression (set by to_ffmpeg_filter()).
    """
    points: list[GainPoint] = field(default_factory=list)
    duck_regions: list[DuckRegion] = field(default_factory=list)
    sensor_flags: list[str] = field(default_factory=list)
    ffmpeg_expr: str = ""


# ── Core ducking engine ───────────────────────────────────────────────────

def _build_duck_region(
    vad: VadEvent,
    bgm_level_db: float,
    attack_s: float,
    release_s: float,
) -> DuckRegion:
    """Compute all timing and level values for one VAD-triggered duck event.

    Attack phase  : ramp from bgm_level_db → ducked_level_db over attack_s
    Hold phase    : ducked_level_db sustained for vad duration
    Release phase : ramp from ducked_level_db → bgm_level_db over release_s

    The attack ramp *precedes* speech onset so the duck is already applied
    when the first word is spoken.  If there is not enough pre-roll, the
    attack starts at vad_start_s (clipped to 0).
    """
    raw_ducked = bgm_level_db + DUCK_DB
    ducked_level_db = max(raw_ducked, DUCK_FLOOR_DBFS)

    if raw_ducked < DUCK_FLOOR_DBFS:
        logger.debug(
            "Duck floor applied: %.1f dBFS → %.1f dBFS (floor=%.1f)",
            raw_ducked, ducked_level_db, DUCK_FLOOR_DBFS,
        )

    # Attack: start attack_s before speech so it finishes at vad_start_s
    attack_start_s = max(0.0, vad.start_s - attack_s)
    attack_end_s = vad.start_s

    # Release: begins at speech end
    release_start_s = vad.end_s
    release_end_s = vad.end_s + release_s

    # Build envelope control points for this region
    envelope: list[GainPoint] = [
        GainPoint(attack_start_s, bgm_level_db, phase="pre-attack"),
        GainPoint(attack_end_s,   ducked_level_db, phase="ducked"),
        GainPoint(release_start_s, ducked_level_db, phase="hold-end"),
        GainPoint(release_end_s,  bgm_level_db, phase="recovered"),
    ]

    return DuckRegion(
        vad_start_s=vad.start_s,
        vad_end_s=vad.end_s,
        bgm_level_db=bgm_level_db,
        ducked_level_db=ducked_level_db,
        attack_start_s=attack_start_s,
        attack_end_s=attack_end_s,
        release_start_s=release_start_s,
        release_end_s=release_end_s,
        envelope=envelope,
    )


def _merge_envelope_points(
    regions: list[DuckRegion],
    bgm_level_db: float,
    segment_start_s: float,
    segment_end_s: float,
) -> list[GainPoint]:
    """Merge per-region envelopes into a single sorted, de-duplicated gain curve.

    Overlapping duck regions (rapid speech) are resolved by taking the
    lower (more ducked) level at any contested time point.
    """
    # Collect all points
    all_points: list[GainPoint] = [
        GainPoint(segment_start_s, bgm_level_db, "segment-start"),
        GainPoint(segment_end_s, bgm_level_db, "segment-end"),
    ]
    for region in regions:
        all_points.extend(region.envelope)

    # Sort by time
    all_points.sort(key=lambda p: p.time_s)

    # De-duplicate: at the same timestamp keep the lower gain (more ducked)
    merged: list[GainPoint] = []
    for pt in all_points:
        if merged and abs(merged[-1].time_s - pt.time_s) < 1e-6:
            if pt.gain_db < merged[-1].gain_db:
                merged[-1] = pt
        else:
            merged.append(pt)

    return merged


# ── FFmpeg volume filter builder ──────────────────────────────────────────

def _points_to_ffmpeg_volume_expr(points: list[GainPoint]) -> str:
    """Convert a GainEnvelope to an FFmpeg `volume` filter expression.

    Uses the `enable` + piecewise linear `volume` approach via the
    `aeval` / `volume` filter with `eval=frame` for per-sample precision.

    For simplicity, produces a `volume` filter with a `timeline` string
    compatible with FFmpeg's `-filter_complex` syntax:

        volume=enable='between(t,0,10)':volume='...'

    In practice, a proper implementation would use the `sidechaincompress`
    or `acompressor` filter; this expression form serves as an auditable
    audit trail of the intended gain curve that the executor can validate.

    Format emitted:
        volume=eval=frame:volume='
          if(lt(t,t0), db(v0),
          if(lt(t,t1), db(v0)+(db(v1)-db(v0))*(t-t0)/(t1-t0),
          ...
          db(vN)))'
    """
    if not points:
        return "volume=1.0"

    # Build nested if() expression
    # Each segment is a linear ramp from points[i] to points[i+1]
    def db_val(db: float) -> str:
        # FFmpeg volume= expects a linear amplitude multiplier
        lin = _db_to_linear(db)
        return f"{lin:.6f}"

    n = len(points)
    if n == 1:
        return f"volume={db_val(points[0].gain_db)}"

    # Build innermost expression first (last segment) then wrap outward
    expr = db_val(points[-1].gain_db)
    for i in range(n - 2, -1, -1):
        p0 = points[i]
        p1 = points[i + 1]
        t0, t1 = p0.time_s, p1.time_s
        v0, v1 = db_val(p0.gain_db), db_val(p1.gain_db)

        if abs(t1 - t0) < 1e-6:
            # Instantaneous step
            expr = f"if(lt(t,{t1:.6f}),{v0},{expr})"
        else:
            # Linear ramp
            ramp = f"{v0}+({v1}-{v0})*(t-{t0:.6f})/({t1:.6f}-{t0:.6f})"
            expr = f"if(lt(t,{t0:.6f}),{v0},if(lt(t,{t1:.6f}),{ramp},{expr}))"

    return f"volume=eval=frame:volume='{expr}'"


# ── Sensor validation ─────────────────────────────────────────────────────

def run_sensors(envelope: GainEnvelope) -> GainEnvelope:
    """Validate the produced envelope against spec §4 and §5 quality gates.

    Gates checked:
      · DUCK_LEVEL    : each duck region must reach ≤ (bgm_level − 20 dB)
      · DUCK_FLOOR    : no region must duck below DUCK_FLOOR_DBFS
      · ADSR_ATTACK   : attack duration must be ≤ ATTACK_MS + 5 ms tolerance
      · ADSR_RELEASE  : release duration must be ≤ RELEASE_MS + 5 ms tolerance

    Appends advisory flag strings to envelope.sensor_flags.
    """
    flags: list[str] = []
    tolerance_s = 0.005  # 5 ms tolerance for floating-point timing

    for i, region in enumerate(envelope.duck_regions):
        # DUCK_LEVEL gate
        expected_max_duck = region.bgm_level_db + DUCK_DB
        if region.ducked_level_db > expected_max_duck + 0.1:
            flags.append(
                f"DUCK_LEVEL [region {i}]: ducked={region.ducked_level_db:.1f} dBFS "
                f"but expected ≤ {expected_max_duck:.1f} dBFS"
            )

        # DUCK_FLOOR gate
        if region.ducked_level_db < DUCK_FLOOR_DBFS - 0.1:
            flags.append(
                f"DUCK_FLOOR [region {i}]: {region.ducked_level_db:.1f} dBFS "
                f"is below floor {DUCK_FLOOR_DBFS:.1f} dBFS"
            )

        # ADSR_ATTACK gate
        actual_attack_s = region.attack_end_s - region.attack_start_s
        if actual_attack_s > (ATTACK_MS / 1000.0) + tolerance_s:
            flags.append(
                f"ADSR_ATTACK [region {i}]: attack={actual_attack_s*1000:.1f} ms "
                f"> spec {ATTACK_MS:.0f} ms"
            )

        # ADSR_RELEASE gate
        actual_release_s = region.release_end_s - region.release_start_s
        if actual_release_s > (RELEASE_MS / 1000.0) + tolerance_s:
            flags.append(
                f"ADSR_RELEASE [region {i}]: release={actual_release_s*1000:.1f} ms "
                f"> spec {RELEASE_MS:.0f} ms"
            )

    if flags:
        logger.warning("Audio sensors raised %d flag(s):\n  %s", len(flags), "\n  ".join(flags))
    else:
        logger.info("Audio sensors: all gates clean (%d duck regions)", len(envelope.duck_regions))

    envelope.sensor_flags = flags
    return envelope


# ── Public entry point ────────────────────────────────────────────────────

def run(
    vad_events: list[VadEvent],
    *,
    bgm_level_db: float = BGM_DEFAULT_DB,
    segment_start_s: float = 0.0,
    segment_end_s: float = 0.0,
    attack_ms: float = ATTACK_MS,
    release_ms: float = RELEASE_MS,
) -> GainEnvelope:
    """Build the BGM gain envelope for a timeline segment.

    Parameters
    ----------
    vad_events      : Ordered list of VadEvent objects from the speaker track.
                      Events with confidence < VAD_CONFIDENCE_THRESHOLD are ignored.
    bgm_level_db    : Current BGM track level in dBFS before ducking.
    segment_start_s : Start of the timeline segment being processed.
    segment_end_s   : End of the timeline segment.
    attack_ms       : Override for attack time (default = spec value 150 ms).
    release_ms      : Override for release time (default = spec value 500 ms).

    Returns
    -------
    GainEnvelope with duck_regions, merged envelope points, FFmpeg filter
    expression, and sensor flags.
    """
    attack_s = attack_ms / 1000.0
    release_s = release_ms / 1000.0

    active_vad = [v for v in vad_events if v.is_active]
    logger.info(
        "Audio ducking: %d/%d VAD events above threshold (≥%.2f)",
        len(active_vad), len(vad_events), VAD_CONFIDENCE_THRESHOLD,
    )

    envelope = GainEnvelope()

    for vad in active_vad:
        region = _build_duck_region(vad, bgm_level_db, attack_s, release_s)
        envelope.duck_regions.append(region)
        logger.debug(
            "duck region: vad %.3f–%.3f s | duck %.1f→%.1f dBFS | "
            "attack %.0f ms | release %.0f ms",
            vad.start_s, vad.end_s,
            bgm_level_db, region.ducked_level_db,
            attack_ms, release_ms,
        )

    envelope.points = _merge_envelope_points(
        envelope.duck_regions, bgm_level_db, segment_start_s, segment_end_s
    )

    envelope.ffmpeg_expr = _points_to_ffmpeg_volume_expr(envelope.points)

    envelope = run_sensors(envelope)

    logger.info(
        "Audio master run complete: %d duck regions, %d envelope points, "
        "%d sensor flags",
        len(envelope.duck_regions),
        len(envelope.points),
        len(envelope.sensor_flags),
    )
    return envelope
