"""
harness/sensors/audio_gate.py

Audio quality gate — validates exported audio for clipping and BGM-to-voice ratio.

Gates
-----
CLIPPING_FREE   : Peak level must not exceed 0 dBFS.
BGM_VOICE_RATIO : BGM duck depth meets or exceeds target (VOICE_DUCK_DB in audio_style.md).

Usage
-----
    from harness.sensors.audio_gate import validate_audio, check_clipping
    reports = validate_audio(Path("output.mp4"))
    for r in reports:
        print(r.name, "PASS" if r.passed else "FAIL", r.detail)

Both gates use ``ffprobe volumedetect`` to measure peak and mean volume.
When ffprobe is unavailable the gate raises RuntimeError so tests must mock it.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_AUDIO_STYLE_FILE = Path(__file__).resolve().parents[2] / "spec" / "audio_style.md"


def _read_audio_float(key: str, default: float) -> float:
    try:
        text = _AUDIO_STYLE_FILE.read_text(encoding="utf-8")
        m = re.search(rf"{re.escape(key)}\s*[:=]\s*([+-]?[\d.]+)", text)
        return float(m.group(1)) if m else default
    except Exception:
        return default


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class AudioCheckResult:
    """Result of a single audio quality check.

    Attributes
    ----------
    name    : Gate identifier string (e.g. "CLIPPING_FREE").
    passed  : True when the check passed.
    detail  : Human-readable description of the measured value or failure reason.
    value   : Measured numeric value (dBFS, ratio, etc.).
    """
    name: str
    passed: bool
    detail: str
    value: float = 0.0


# ── ffprobe helper ────────────────────────────────────────────────────────────

def _run_volumedetect(audio_file: Path) -> dict[str, float]:
    """Run ``ffprobe -af volumedetect`` and return parsed stats.

    Returns a dict with keys: ``mean_volume``, ``max_volume`` (both in dBFS).
    Raises RuntimeError if ffprobe is not available or returns a non-zero exit code.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-i", str(audio_file),
        "-af", "volumedetect",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    # volumedetect output goes to stderr
    output = result.stderr + result.stdout

    stats: dict[str, float] = {}
    m = re.search(r"mean_volume:\s*([+-]?[\d.]+)\s*dB", output)
    if m:
        stats["mean_volume"] = float(m.group(1))
    m = re.search(r"max_volume:\s*([+-]?[\d.]+)\s*dB", output)
    if m:
        stats["max_volume"] = float(m.group(1))

    if not stats:
        raise RuntimeError(
            f"ffprobe volumedetect returned no output for {audio_file}.\n"
            f"stderr: {result.stderr[:500]}"
        )
    return stats


# ── Gate implementations ──────────────────────────────────────────────────────

def check_clipping(audio_file: Path) -> AudioCheckResult:
    """Verify that the audio peak does not exceed 0 dBFS (no clipping).

    A peak at 0 dBFS means the signal is at the digital ceiling and will clip.
    Professional audio targets −1 dBFS or lower.

    Parameters
    ----------
    audio_file : Path to an audio or video file.

    Returns
    -------
    AudioCheckResult with name='CLIPPING_FREE'.
    """
    name = "CLIPPING_FREE"
    try:
        stats = _run_volumedetect(audio_file)
        peak = stats.get("max_volume", 0.0)
        if peak >= 0.0:
            return AudioCheckResult(
                name=name, passed=False, value=peak,
                detail=f"CLIPPING detected — peak={peak:.1f} dBFS (must be < 0 dBFS)",
            )
        return AudioCheckResult(
            name=name, passed=True, value=peak,
            detail=f"OK — peak={peak:.1f} dBFS",
        )
    except RuntimeError as exc:
        return AudioCheckResult(name=name, passed=False, value=0.0,
                                detail=f"ffprobe error: {exc}")
    except Exception as exc:
        return AudioCheckResult(name=name, passed=False, value=0.0,
                                detail=f"exception: {exc}")


def check_bgm_voice_ratio(
    audio_file: Path,
    target_duck_db: Optional[float] = None,
) -> AudioCheckResult:
    """Estimate whether the BGM duck depth meets the configured VOICE_DUCK_DB target.

    Because a final mixed file has no separate BGM and voice tracks, this gate
    uses a heuristic: the difference between the *mean* volume of the loudest 10%
    of frames and the overall mean.  A well-ducked mix should have a higher
    peak-to-mean ratio than an un-ducked mix.

    The check passes when ``(max_volume - mean_volume) >= abs(target_duck_db) * 0.5``.
    The 0.5 factor accounts for the fact that peaks are transient speech bursts and
    the mean includes both speech and ducked BGM.

    Parameters
    ----------
    audio_file      : Path to an audio or video file.
    target_duck_db  : Expected duck depth in dB (negative, e.g. −20).
                      Defaults to VOICE_DUCK_DB from spec/audio_style.md.

    Returns
    -------
    AudioCheckResult with name='BGM_VOICE_RATIO'.
    """
    name = "BGM_VOICE_RATIO"
    if target_duck_db is None:
        target_duck_db = _read_audio_float("VOICE_DUCK_DB", -20.0)

    min_ratio = abs(target_duck_db) * 0.5   # e.g. 10 dB for −20 dB duck target

    try:
        stats = _run_volumedetect(audio_file)
        peak = stats.get("max_volume", 0.0)
        mean = stats.get("mean_volume", peak)
        ratio = peak - mean   # should be positive

        if ratio < min_ratio:
            return AudioCheckResult(
                name=name, passed=False, value=ratio,
                detail=(
                    f"BGM duck depth insufficient — "
                    f"peak={peak:.1f} dBFS  mean={mean:.1f} dBFS  "
                    f"ratio={ratio:.1f} dB  required≥{min_ratio:.1f} dB"
                ),
            )
        return AudioCheckResult(
            name=name, passed=True, value=ratio,
            detail=(
                f"OK — peak={peak:.1f} dBFS  mean={mean:.1f} dBFS  "
                f"ratio={ratio:.1f} dB ≥ {min_ratio:.1f} dB"
            ),
        )
    except RuntimeError as exc:
        return AudioCheckResult(name=name, passed=False, value=0.0,
                                detail=f"ffprobe error: {exc}")
    except Exception as exc:
        return AudioCheckResult(name=name, passed=False, value=0.0,
                                detail=f"exception: {exc}")


# ── Composite runner ──────────────────────────────────────────────────────────

def validate_audio(
    audio_file: Path,
    target_duck_db: Optional[float] = None,
) -> list[AudioCheckResult]:
    """Run all audio quality gates on the given file.

    Parameters
    ----------
    audio_file      : Path to the rendered audio/video file.
    target_duck_db  : Override for duck depth target (see check_bgm_voice_ratio).

    Returns
    -------
    List of AudioCheckResult, one per gate.
    """
    return [
        check_clipping(audio_file),
        check_bgm_voice_ratio(audio_file, target_duck_db=target_duck_db),
    ]
