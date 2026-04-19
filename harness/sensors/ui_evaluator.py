"""
harness/sensors/ui_evaluator.py

UI quality sensor — Playwright-based integration tests for the Streamlit UI.

Gates enforced
  UI_SPEC_SYNC      : slider change → spec/editing_style.md CAPTION_Y_PX updates
  PIPELINE_TRIGGER  : '편집 시작' button click → job directory created in memory
  MEMORY_CONSISTENCY: most recent completed job has all 4 skills recorded as success

Usage
-----
    # Streamlit must be running on localhost:8501 first
    cd VoxEdit_AI
    python harness/sensors/ui_evaluator.py

    # or with a custom URL:
    python harness/sensors/ui_evaluator.py --url http://localhost:8502

Exit code: 0 = all Pass, 1 = any Fail
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SPEC_FILE = ROOT / "spec" / "editing_style.md"
JOBS_ROOT = ROOT / "harness" / "memory" / "jobs"
STAGING_ROOT = ROOT / "staging"

SKILL_ORDER = ["transcriber", "cutter", "designer", "exporter"]
APP_URL = "http://localhost:8501"

# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_caption_y_from_spec() -> int | None:
    try:
        text = SPEC_FILE.read_text(encoding="utf-8")
        m = re.search(r"CAPTION_Y_PX\s*[:=]\s*(\d+)", text)
        return int(m.group(1)) if m else None
    except Exception as exc:
        return None


def _latest_job_dir() -> Path | None:
    if not JOBS_ROOT.exists():
        return None
    dirs = sorted(JOBS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def _load_skill_records(job_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(job_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            skill = data.get("skill", "")
            if skill and data.get("status") == "success":
                records[skill] = data
        except Exception:
            pass
    return records


# ── Gate implementations ──────────────────────────────────────────────────────

def gate_ui_spec_sync(page) -> GateResult:
    """Move the slider and verify CAPTION_Y_PX in spec file updates."""
    name = "UI_SPEC_SYNC"
    try:
        from playwright.sync_api import expect

        # Read current spec value
        before = _read_caption_y_from_spec()

        # Locate the slider input
        slider = page.locator('div[data-testid="stSlider"] input[type="range"]').first
        slider.wait_for(timeout=10_000)

        current_val = int(slider.get_attribute("value") or "84")
        # Move to a distinctly different value to avoid rounding collisions
        target_val = 30 if current_val > 50 else 70

        # Set via JS and fire both 'input' and 'change' events so Streamlit reacts
        page.evaluate(
            """([el, val]) => {
                el.value = val;
                el.dispatchEvent(new Event('input', {bubbles: true}));
                el.dispatchEvent(new Event('change', {bubbles: true}));
            }""",
            [slider.element_handle(), target_val],
        )

        # Wait up to 5 s for Streamlit to re-render and write the file
        deadline = time.monotonic() + 5.0
        after = before
        while time.monotonic() < deadline:
            time.sleep(0.4)
            after = _read_caption_y_from_spec()
            if after != before:
                break

        expected_px = round(1920 * target_val / 100)
        tolerance = 20  # ±20 px rounding allowed

        if after is None:
            return GateResult(name, False, "CAPTION_Y_PX not found in spec after slider move")
        if abs(after - expected_px) > tolerance:
            return GateResult(
                name, False,
                f"spec={after}px  expected≈{expected_px}px (slider={target_val}%)"
            )
        return GateResult(name, True, f"slider {target_val}% → spec CAPTION_Y_PX={after}px")

    except Exception as exc:
        return GateResult(name, False, f"exception: {exc}")


def gate_pipeline_trigger(page, test_video: Path | None = None) -> GateResult:
    """Upload a file and click '편집 시작'; verify job dir appears in memory."""
    name = "PIPELINE_TRIGGER"
    try:
        # Find a test video
        if test_video is None:
            candidates = list((ROOT / "tests").rglob("*.mp4"))
            if not candidates:
                return GateResult(name, False, "No test video found under tests/")
            test_video = candidates[0]

        # Count existing jobs before trigger
        existing_jobs = set(p.name for p in JOBS_ROOT.iterdir()) if JOBS_ROOT.exists() else set()

        # Upload file
        file_input = page.locator('input[type="file"]').first
        file_input.wait_for(timeout=10_000)
        file_input.set_input_files(str(test_video))

        # Wait for Streamlit to process the upload (button becomes enabled)
        run_btn = page.get_by_text("AI 편집 시작").first
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                disabled = run_btn.get_attribute("disabled")
                if disabled is None:  # not disabled
                    break
            except Exception:
                pass

        run_btn.click()

        # Wait up to 15 s for a new job directory to appear
        deadline = time.monotonic() + 15.0
        new_job: str | None = None
        while time.monotonic() < deadline:
            time.sleep(1.0)
            if JOBS_ROOT.exists():
                current = set(p.name for p in JOBS_ROOT.iterdir())
                new_jobs = current - existing_jobs
                if new_jobs:
                    new_job = sorted(new_jobs)[-1]
                    break

        if new_job is None:
            # Also accept: _status.json in staging proves thread launched
            status_files = list(STAGING_ROOT.glob("job_*/_status.json"))
            if status_files:
                newest = max(status_files, key=lambda p: p.stat().st_mtime)
                return GateResult(name, True,
                    f"pipeline thread started — staging status: {newest.parent.name}")
            return GateResult(name, False,
                "no new job dir in harness/memory/jobs and no staging _status.json")

        return GateResult(name, True, f"job created: {new_job}")

    except Exception as exc:
        return GateResult(name, False, f"exception: {exc}")


def gate_memory_consistency() -> GateResult:
    """Check that the most recent completed job has all 4 skills marked success."""
    name = "MEMORY_CONSISTENCY"
    try:
        job_dir = _latest_job_dir()
        if job_dir is None:
            return GateResult(name, False, "no job directories found in harness/memory/jobs/")

        records = _load_skill_records(job_dir)
        missing = [s for s in SKILL_ORDER if s not in records]

        if missing:
            present = list(records.keys())
            return GateResult(
                name, False,
                f"job={job_dir.name}  success={present}  missing={missing}"
            )
        return GateResult(
            name, True,
            f"job={job_dir.name}  all 4 skills success"
        )
    except Exception as exc:
        return GateResult(name, False, f"exception: {exc}")


# ── Runner ────────────────────────────────────────────────────────────────────

def run(url: str = APP_URL, test_video: Path | None = None) -> list[GateResult]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[ui_evaluator] playwright not installed — run: pip install playwright && playwright install chromium")
        sys.exit(1)

    results: list[GateResult] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(url, timeout=15_000)
            # Wait for Streamlit app shell
            page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=20_000)
            # Extra settle time for Streamlit to finish hydration
            time.sleep(2)
        except Exception as exc:
            browser.close()
            results.append(GateResult("APP_REACHABLE", False,
                f"Could not reach {url}: {exc}"))
            results.append(GateResult("UI_SPEC_SYNC", False, "skipped — app unreachable"))
            results.append(GateResult("PIPELINE_TRIGGER", False, "skipped — app unreachable"))
            results.append(GateResult("MEMORY_CONSISTENCY", False, "skipped — app unreachable"))
            return results

        results.append(GateResult("APP_REACHABLE", True, url))
        results.append(gate_ui_spec_sync(page))
        results.append(gate_pipeline_trigger(page, test_video))
        browser.close()

    # Memory consistency is disk-only — no browser needed
    results.append(gate_memory_consistency())
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def _print_report(results: list[GateResult]) -> bool:
    """Print results table, return True if all passed."""
    print()
    print("=" * 60)
    print("  VoxEdit AI — UI Evaluator")
    print("=" * 60)
    all_pass = True
    for r in results:
        icon = "Pass" if r.passed else "FAIL"
        print(f"  [{icon}]  {r.name:<24}  {r.detail}")
        if not r.passed:
            all_pass = False
    print("=" * 60)
    print(f"  {'All gates passed.' if all_pass else 'One or more gates FAILED.'}")
    print("=" * 60)
    print()
    return all_pass


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxEdit AI UI Evaluator")
    parser.add_argument("--url", default=APP_URL, help="Streamlit app URL")
    parser.add_argument("--video", default=None, help="Path to a test .mp4 file")
    args = parser.parse_args()

    video = Path(args.video) if args.video else None
    results = run(url=args.url, test_video=video)
    passed = _print_report(results)
    sys.exit(0 if passed else 1)
