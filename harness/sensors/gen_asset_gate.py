"""
harness/sensors/gen_asset_gate.py

GenAssetGate — validates generated b-roll images meet pipeline requirements.

Checks
------
1. FILE_EXISTS    : The file at `path` exists and is non-empty.
2. RESOLUTION     : Image dimensions are exactly 1080×1920 (9:16 portrait).
3. FILE_INTEGRITY : The file is a valid, non-corrupt image (Pillow verify).
4. MIN_SIZE       : File size exceeds a minimum byte threshold (default 1 KB).

Usage
-----
    from harness.sensors.gen_asset_gate import check_gen_asset, validate_all

    report = check_gen_asset("assets/generated/cat.png")
    print(report.passed, report.detail)

    reports = validate_all(["cat.png", "ocean.png"])
    all_ok = all(r.passed for r in reports)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
REQUIRED_WIDTH: int  = 1080
REQUIRED_HEIGHT: int = 1920
MIN_FILE_SIZE_BYTES: int = 1024   # 1 KB — a valid PNG is always larger


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class GenAssetReport:
    """Result of a single generated-asset validation.

    Attributes
    ----------
    path            : Absolute (or as-given) path that was checked.
    passed          : True when all checks pass.
    detail          : Human-readable explanation of what passed or failed.
    width           : Image width in pixels (0 if file could not be opened).
    height          : Image height in pixels (0 if file could not be opened).
    file_size_bytes : Size of the file in bytes (0 if file not found).
    checks          : Individual check results {name: bool}.
    """

    path: str
    passed: bool
    detail: str
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    checks: dict[str, bool] = field(default_factory=dict)


# ── Individual checkers ───────────────────────────────────────────────────────

def _check_file_exists(p: Path) -> tuple[bool, str]:
    if not p.exists():
        return False, f"File not found: {p}"
    if p.stat().st_size == 0:
        return False, f"File is empty: {p}"
    return True, "exists"


def _check_min_size(p: Path, min_bytes: int) -> tuple[bool, str]:
    size = p.stat().st_size
    if size < min_bytes:
        return False, f"File too small: {size} bytes (min {min_bytes})"
    return True, f"{size} bytes"


def _check_integrity(p: Path) -> tuple[bool, str, int, int]:
    """Return (ok, detail, width, height)."""
    try:
        from PIL import Image, UnidentifiedImageError
        with Image.open(str(p)) as img:
            img.verify()            # raises on corruption
        # Re-open after verify (verify() closes/invalidates the file handle)
        with Image.open(str(p)) as img:
            w, h = img.size
        return True, "valid image", w, h
    except Exception as exc:
        return False, f"Corrupt or unreadable image: {exc}", 0, 0


def _check_resolution(
    w: int, h: int,
    required_w: int, required_h: int,
) -> tuple[bool, str]:
    if w == required_w and h == required_h:
        return True, f"{w}×{h} ✓"
    return (
        False,
        f"Wrong resolution: {w}×{h} (expected {required_w}×{required_h})",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def check_gen_asset(
    path: str,
    required_width: int = REQUIRED_WIDTH,
    required_height: int = REQUIRED_HEIGHT,
    min_file_size_bytes: int = MIN_FILE_SIZE_BYTES,
) -> GenAssetReport:
    """Validate a single generated image file.

    Parameters
    ----------
    path               : Path to the PNG/image file.
    required_width     : Expected pixel width (default 1080).
    required_height    : Expected pixel height (default 1920).
    min_file_size_bytes: Minimum acceptable file size in bytes.

    Returns
    -------
    GenAssetReport with `passed=True` only when ALL checks succeed.
    """
    p = Path(path)
    checks: dict[str, bool] = {}
    file_size = 0
    w, h = 0, 0

    # ── 1. File exists ────────────────────────────────────────────────────
    exists_ok, exists_detail = _check_file_exists(p)
    checks["file_exists"] = exists_ok
    if not exists_ok:
        return GenAssetReport(
            path=path, passed=False, detail=exists_detail,
            checks=checks,
        )

    file_size = p.stat().st_size

    # ── 2. Minimum size ───────────────────────────────────────────────────
    size_ok, size_detail = _check_min_size(p, min_file_size_bytes)
    checks["min_size"] = size_ok
    if not size_ok:
        return GenAssetReport(
            path=path, passed=False, detail=size_detail,
            file_size_bytes=file_size, checks=checks,
        )

    # ── 3. File integrity ─────────────────────────────────────────────────
    integrity_ok, integrity_detail, w, h = _check_integrity(p)
    checks["file_integrity"] = integrity_ok
    if not integrity_ok:
        return GenAssetReport(
            path=path, passed=False, detail=integrity_detail,
            file_size_bytes=file_size, width=w, height=h, checks=checks,
        )

    # ── 4. Resolution ─────────────────────────────────────────────────────
    res_ok, res_detail = _check_resolution(w, h, required_width, required_height)
    checks["resolution"] = res_ok
    if not res_ok:
        return GenAssetReport(
            path=path, passed=False, detail=res_detail,
            file_size_bytes=file_size, width=w, height=h, checks=checks,
        )

    return GenAssetReport(
        path=path, passed=True, detail="All checks passed",
        file_size_bytes=file_size, width=w, height=h, checks=checks,
    )


def validate_all(
    paths: list[str],
    required_width: int = REQUIRED_WIDTH,
    required_height: int = REQUIRED_HEIGHT,
    min_file_size_bytes: int = MIN_FILE_SIZE_BYTES,
) -> list[GenAssetReport]:
    """Validate a list of generated image paths.

    Returns one GenAssetReport per path, in order.
    """
    return [
        check_gen_asset(
            p,
            required_width=required_width,
            required_height=required_height,
            min_file_size_bytes=min_file_size_bytes,
        )
        for p in paths
    ]
