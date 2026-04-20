"""
harness/sensors/rhythm_gate.py

Stability gate for DynamicZoom and RHYTHM_INTENSITY visual effects.

Gates
-----
ZOOM_STABILITY      : All zoom factors ≤ SAFE_ZOOM_MAX (1.10) — no warping artefacts.
ZOOM_NO_OVERLAP     : No two zoom events overlap in time (prevents stuttering).
RHYTHM_RANGE        : RHYTHM_INTENSITY is within [0.0, 1.0].
DYNAMIC_ZOOM_COUNT  : Number of dynamic zoom events is within sensible range
                      relative to total duration (≤ 10 events/minute).

Usage
-----
    from harness.sensors.rhythm_gate import validate_rhythm
    reports = validate_rhythm(visual_elements, rhythm_intensity=0.5, total_duration_s=60.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SPEC_FILE = Path(__file__).resolve().parents[2] / "spec" / "editing_style.md"

SAFE_ZOOM_MAX: float = 1.10        # §3 spec maximum; dynamic zoom must stay below this
SAFE_ZOOM_MIN: float = 1.0
MAX_ZOOM_EVENTS_PER_MIN: float = 10.0   # density cap


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class RhythmGateResult:
    """Result of a single rhythm/visual stability check.

    Attributes
    ----------
    name    : Gate identifier (e.g. "ZOOM_STABILITY").
    passed  : True when the check passed.
    detail  : Human-readable result or failure reason.
    value   : Key numeric measurement (zoom factor, count, etc.).
    """
    name: str
    passed: bool
    detail: str
    value: float = 0.0


# ── Gate implementations ──────────────────────────────────────────────────────

def check_zoom_stability(visual_elements: list[dict]) -> RhythmGateResult:
    """Verify all zoom_factor values are within safe bounds (1.0 ≤ zf ≤ SAFE_ZOOM_MAX).

    Both jump-cut (1.1×) and dynamic zoom events are checked.  Zoom factors
    above SAFE_ZOOM_MAX risk visible warping/stretching artefacts on narrow
    aspect-ratio content.
    """
    name = "ZOOM_STABILITY"
    zoom_elems = [e for e in visual_elements if e.get("type") == "zoom"]
    if not zoom_elems:
        return RhythmGateResult(name=name, passed=True, value=0,
                                detail="No zoom elements — gate skipped (OK)")

    violations: list[str] = []
    max_seen = 0.0
    for elem in zoom_elems:
        zf = float(elem.get("zoom_factor", 1.0))
        max_seen = max(max_seen, zf)
        if zf > SAFE_ZOOM_MAX:
            violations.append(
                f"zoom_factor={zf:.4f} at t={elem.get('start', 0):.2f}s "
                f"exceeds max {SAFE_ZOOM_MAX}"
            )
        elif zf < SAFE_ZOOM_MIN:
            violations.append(
                f"zoom_factor={zf:.4f} at t={elem.get('start', 0):.2f}s "
                f"below min {SAFE_ZOOM_MIN}"
            )

    if violations:
        return RhythmGateResult(
            name=name, passed=False, value=max_seen,
            detail=f"{len(violations)} violation(s): {'; '.join(violations[:3])}",
        )
    return RhythmGateResult(
        name=name, passed=True, value=max_seen,
        detail=f"All {len(zoom_elems)} zoom elements within [{SAFE_ZOOM_MIN}, {SAFE_ZOOM_MAX}]",
    )


def check_zoom_no_overlap(visual_elements: list[dict]) -> RhythmGateResult:
    """Verify that no two zoom events overlap in time.

    Overlapping zoom events in the filtergraph cause the zoompan filter to
    receive conflicting scale parameters, producing a frame-stutter artefact.
    """
    name = "ZOOM_NO_OVERLAP"
    zoom_elems = sorted(
        [e for e in visual_elements if e.get("type") == "zoom"],
        key=lambda e: float(e.get("start", 0.0)),
    )
    if len(zoom_elems) < 2:
        return RhythmGateResult(name=name, passed=True, value=0,
                                detail=f"{len(zoom_elems)} zoom event(s) — no overlap possible")

    overlaps: list[str] = []
    for i in range(len(zoom_elems) - 1):
        a_end = float(zoom_elems[i].get("end", 0.0))
        b_start = float(zoom_elems[i + 1].get("start", 0.0))
        if a_end > b_start:
            overlaps.append(
                f"[{i}] ends {a_end:.2f}s overlaps [{i+1}] starts {b_start:.2f}s"
            )

    if overlaps:
        return RhythmGateResult(
            name=name, passed=False, value=float(len(overlaps)),
            detail=f"{len(overlaps)} overlap(s): {'; '.join(overlaps[:3])}",
        )
    return RhythmGateResult(
        name=name, passed=True, value=0.0,
        detail=f"All {len(zoom_elems)} zoom events are non-overlapping",
    )


def check_rhythm_intensity_range(rhythm_intensity: float) -> RhythmGateResult:
    """Verify RHYTHM_INTENSITY is within the valid range [0.0, 1.0]."""
    name = "RHYTHM_RANGE"
    if 0.0 <= rhythm_intensity <= 1.0:
        return RhythmGateResult(
            name=name, passed=True, value=rhythm_intensity,
            detail=f"RHYTHM_INTENSITY={rhythm_intensity:.2f} ∈ [0.0, 1.0]",
        )
    return RhythmGateResult(
        name=name, passed=False, value=rhythm_intensity,
        detail=(
            f"RHYTHM_INTENSITY={rhythm_intensity:.2f} "
            f"out of valid range [0.0, 1.0]"
        ),
    )


def check_dynamic_zoom_density(
    visual_elements: list[dict],
    total_duration_s: float,
) -> RhythmGateResult:
    """Verify dynamic zoom event density is not excessive (≤ 10 events/minute).

    Too many zoom pulses within a short window creates visual noise.
    Only ``name="dynamic_zoom"`` elements are counted.
    """
    name = "DYNAMIC_ZOOM_COUNT"
    dz_elems = [
        e for e in visual_elements
        if e.get("type") == "zoom" and e.get("name") == "dynamic_zoom"
    ]
    count = len(dz_elems)

    if total_duration_s <= 0:
        return RhythmGateResult(name=name, passed=True, value=count,
                                detail="Zero duration — density check skipped")

    density = (count / total_duration_s) * 60.0   # events per minute
    if density > MAX_ZOOM_EVENTS_PER_MIN:
        return RhythmGateResult(
            name=name, passed=False, value=density,
            detail=(
                f"DynamicZoom density {density:.1f} evt/min "
                f"exceeds max {MAX_ZOOM_EVENTS_PER_MIN:.0f} evt/min "
                f"({count} events in {total_duration_s:.1f}s)"
            ),
        )
    return RhythmGateResult(
        name=name, passed=True, value=density,
        detail=(
            f"DynamicZoom density OK — {count} events in {total_duration_s:.1f}s "
            f"= {density:.1f} evt/min (max {MAX_ZOOM_EVENTS_PER_MIN:.0f})"
        ),
    )


# ── Spec reader ───────────────────────────────────────────────────────────────

def read_rhythm_intensity_from_spec() -> float:
    """Read RHYTHM_INTENSITY from spec/editing_style.md."""
    try:
        text = _SPEC_FILE.read_text(encoding="utf-8")
        m = re.search(r"RHYTHM_INTENSITY\s*[:=]\s*([\d.]+)", text)
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.5
    except Exception:
        return 0.5


# ── Composite runner ──────────────────────────────────────────────────────────

def validate_rhythm(
    visual_elements: list[dict],
    *,
    rhythm_intensity: Optional[float] = None,
    total_duration_s: float = 0.0,
) -> list[RhythmGateResult]:
    """Run all rhythm stability gates.

    Parameters
    ----------
    visual_elements  : List of visual element dicts from annotated_timeline.json.
    rhythm_intensity : Override (default: read from spec/editing_style.md).
    total_duration_s : Total timeline duration for density check.

    Returns
    -------
    List of RhythmGateResult, one per gate.
    """
    if rhythm_intensity is None:
        rhythm_intensity = read_rhythm_intensity_from_spec()

    return [
        check_zoom_stability(visual_elements),
        check_zoom_no_overlap(visual_elements),
        check_rhythm_intensity_range(rhythm_intensity),
        check_dynamic_zoom_density(visual_elements, total_duration_s),
    ]
