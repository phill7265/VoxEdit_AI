"""
src/ui/windows_app.py

VoxEdit AI — Flet Native Windows UI (Win11 dark mode)

Run:
    cd C:/Users/rearl/Documents/work/sales/VoxEdit_AI
    flet run src/ui/windows_app.py
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

import flet as ft

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC_FILE            = ROOT / "spec" / "editing_style.md"
BROLL_REQUESTS_FILE  = ROOT / "spec" / "broll_requests.json"
JOBS_ROOT            = ROOT / "harness" / "memory" / "jobs"
STAGING_ROOT         = ROOT / "staging"

SKILL_ORDER  = ["transcriber", "cutter", "designer", "exporter"]
SKILL_LABELS = {
    "transcriber": "전사 (Transcriber)",
    "cutter":      "편집 (Cutter)",
    "designer":    "디자인 (Designer)",
    "exporter":    "렌더링 (Exporter)",
}

# ── Win11 colour palette ──────────────────────────────────────────────────────
BG_BASE      = "#1c1c1e"   # window background
BG_SURFACE   = "#2c2c2e"   # card surface
BG_ELEVATED  = "#3a3a3c"   # elevated / hover
ACCENT       = "#0078d4"   # Win11 blue
ACCENT_HOVER = "#106ebe"
TEXT_PRIMARY = "#ffffff"
TEXT_MUTED   = "#8e8e93"
SUCCESS      = "#32d74b"
ERROR        = "#ff453a"
WARNING      = "#ffd60a"
DIVIDER      = "#48484a"


# ═══════════════════════════════════════════════════════════════════════════════
#  Disk helpers (thread-safe, same as Streamlit version)
# ═══════════════════════════════════════════════════════════════════════════════

def _status_path(job_id: str) -> Path:
    return STAGING_ROOT / job_id / "_status.json"


def _read_status(job_id: str) -> dict:
    p = _status_path(job_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": False, "error": None, "output_path": None}


def _write_status(job_id: str, *, done: bool, error: str | None, output_path: str | None) -> None:
    p = _status_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"done": done, "error": error, "output_path": output_path}),
        encoding="utf-8",
    )


# ── Spec helpers ──────────────────────────────────────────────────────────────

def _read_caption_y() -> int:
    if not SPEC_FILE.exists():
        return 1620
    text = SPEC_FILE.read_text(encoding="utf-8")
    m = re.search(r"CAPTION_Y_PX\s*[:=]\s*(\d+)", text)
    return int(m.group(1)) if m else 1620


def _write_caption_y(px: int) -> None:
    text = SPEC_FILE.read_text(encoding="utf-8") if SPEC_FILE.exists() else ""
    new_line = f"CAPTION_Y_PX: {px}"
    if "CAPTION_Y_PX" in text:
        text = re.sub(r"CAPTION_Y_PX\s*[:=]\s*\d+", new_line, text)
    else:
        text += f"\n\n## 6. Caption Position\n\n{new_line}\n"
    SPEC_FILE.write_text(text, encoding="utf-8")


# ── Job memory helpers ────────────────────────────────────────────────────────

def _read_job_records(job_id: str) -> dict[str, dict]:
    job_dir = JOBS_ROOT / job_id
    if not job_dir.exists():
        return {}
    records: dict[str, dict] = {}
    for path in sorted(job_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            skill = data.get("skill", "")
            if skill:
                records[skill] = data
        except Exception:
            pass
    return records


def _active_skill(job_id: str) -> str | None:
    done = set(_read_job_records(job_id).keys())
    for skill in SKILL_ORDER:
        if skill not in done:
            return skill
    return None


def _find_latest_complete_job() -> tuple[str, dict] | tuple[None, None]:
    if not JOBS_ROOT.exists():
        return None, None
    for job_dir in sorted(JOBS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        records: dict[str, dict] = {}
        for path in sorted(job_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") == "success" and data.get("skill"):
                    records[data["skill"]] = data
            except Exception:
                pass
        if all(s in records for s in SKILL_ORDER):
            return job_dir.name, records
    return None, None


# ── B-roll helpers ────────────────────────────────────────────────────────────

def _read_broll_requests() -> list[dict]:
    if not BROLL_REQUESTS_FILE.exists():
        return []
    try:
        return json.loads(BROLL_REQUESTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_broll_requests(requests: list[dict]) -> None:
    BROLL_REQUESTS_FILE.write_text(
        json.dumps(requests, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Pipeline runners ──────────────────────────────────────────────────────────

def _run_pipeline(job_id: str, source_path: str, staging_dir: Path) -> None:
    try:
        from src.pipeline.orchestrator import WorkflowManager
        wm = WorkflowManager(
            job_id=job_id,
            source_file=source_path,
            staging_dir=staging_dir,
            model_name="base",
            dry_run=False,
        )
        result = wm.run()
        if result.succeeded:
            output = staging_dir / "output.mp4"
            _write_status(job_id, done=True, error=None,
                          output_path=str(output) if output.exists() else None)
        else:
            err = result.final_record.error if result.final_record else "Unknown error"
            _write_status(job_id, done=True,
                          error=f"{result.failed_skill}: {err}", output_path=None)
    except Exception as exc:
        _write_status(job_id, done=True, error=str(exc), output_path=None)


def _run_smart_resume(
    job_id: str,
    source_path: str,
    staging_dir: Path,
    force_resume_from: str,
    borrow_records: dict,
) -> None:
    try:
        from src.pipeline.orchestrator import WorkflowManager
        from harness.memory.manager import SkillRecord

        sr_map = {
            skill: SkillRecord(**{
                k: v for k, v in rec.items()
                if k in SkillRecord.__dataclass_fields__
            })
            for skill, rec in borrow_records.items()
        }
        wm = WorkflowManager(
            job_id=job_id,
            source_file=source_path,
            staging_dir=staging_dir,
            model_name="base",
            dry_run=False,
            force_resume_from=force_resume_from,
            borrow_records=sr_map,
        )
        result = wm.run()
        if result.succeeded:
            output = staging_dir / "output.mp4"
            _write_status(job_id, done=True, error=None,
                          output_path=str(output) if output.exists() else None)
        else:
            err = result.final_record.error if result.final_record else "Unknown error"
            _write_status(job_id, done=True,
                          error=f"{result.failed_skill}: {err}", output_path=None)
    except Exception as exc:
        _write_status(job_id, done=True, error=str(exc), output_path=None)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _card(content: ft.Control, padding: int = 16) -> ft.Container:
    return ft.Container(
        content=content,
        bgcolor=BG_SURFACE,
        border_radius=8,
        padding=padding,
        margin=ft.margin.only(bottom=12),
    )


def _label(text: str, size: int = 13, color: str = TEXT_MUTED, weight=ft.FontWeight.NORMAL) -> ft.Text:
    return ft.Text(text, size=size, color=color, weight=weight)


def _section_title(text: str) -> ft.Text:
    return ft.Text(text, size=15, color=TEXT_PRIMARY, weight=ft.FontWeight.W_600)


def _divider() -> ft.Divider:
    return ft.Divider(height=1, color=DIVIDER)


def _skill_row(skill: str, rec: dict | None, active: bool) -> ft.Row:
    label = SKILL_LABELS[skill]
    if rec:
        if rec.get("status") == "success":
            icon = ft.Icon(ft.Icons.CHECK_CIRCLE, color=SUCCESS, size=18)
            detail = f"{rec.get('cursor_start', '')} → {rec.get('cursor_end', '')}"
            text_color = TEXT_PRIMARY
        else:
            icon = ft.Icon(ft.Icons.ERROR, color=ERROR, size=18)
            detail = rec.get("error", "failed")
            text_color = ERROR
        return ft.Row([
            icon,
            ft.Column([
                ft.Text(label, size=13, color=text_color, weight=ft.FontWeight.W_500),
                ft.Text(detail, size=11, color=TEXT_MUTED),
            ], spacing=1, tight=True),
        ], spacing=8)
    elif active:
        return ft.Row([
            ft.ProgressRing(width=16, height=16, stroke_width=2, color=ACCENT),
            ft.Text(label, size=13, color=WARNING, weight=ft.FontWeight.W_500),
            ft.Text("처리 중...", size=11, color=TEXT_MUTED),
        ], spacing=8)
    else:
        return ft.Row([
            ft.Icon(ft.Icons.RADIO_BUTTON_UNCHECKED, color=DIVIDER, size=18),
            ft.Text(label, size=13, color=TEXT_MUTED),
        ], spacing=8)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Flet app
# ═══════════════════════════════════════════════════════════════════════════════

def main(page: ft.Page) -> None:
    # ── Window & theme ─────────────────────────────────────────────────────────
    page.title       = "VoxEdit AI"
    page.window.width  = 1200
    page.window.height = 820
    page.window.min_width  = 900
    page.window.min_height = 640
    page.bgcolor     = BG_BASE
    page.padding     = 0
    page.theme_mode  = ft.ThemeMode.DARK
    page.theme = ft.Theme(
        color_scheme_seed=ACCENT,
        use_material3=True,
    )
    page.fonts = {"Pretendard": "https://fonts.gstatic.com/s/notosanskr/v36/PbykFmXiEBPT4ITbgNA5Cgms3VYcOA-vvnIzzuoyeLTq8H4hfeE.woff2"}

    # ── App state ──────────────────────────────────────────────────────────────
    state: dict = {
        "job_id": None,
        "source_path": None,
        "polling": False,
    }
    chat_history: list[dict] = []
    edit_mode_ref: list[str] = ["Standard"]

    # ── Refs (mutated controls) ────────────────────────────────────────────────
    upload_label        = ft.Text("파일을 선택하세요", size=13, color=TEXT_MUTED)
    run_btn             = ft.Ref[ft.ElevatedButton]()
    caption_slider      = ft.Ref[ft.Slider]()
    caption_value_label = ft.Ref[ft.Text]()
    progress_col        = ft.Ref[ft.Column]()
    result_col          = ft.Ref[ft.Column]()
    chat_list           = ft.Ref[ft.ListView]()
    chat_input_field    = ft.Ref[ft.TextField]()
    broll_col           = ft.Ref[ft.Column]()
    snackbar            = ft.SnackBar(ft.Text(""), bgcolor=BG_ELEVATED)
    page.overlay.append(snackbar)

    def _toast(msg: str, color: str = SUCCESS) -> None:
        snackbar.content = ft.Text(msg, color=TEXT_PRIMARY)
        snackbar.bgcolor = color
        snackbar.open = True
        page.update()

    # ══════════════════════════════════════════════════════════════════════════
    #  Upload
    # ══════════════════════════════════════════════════════════════════════════

    def on_file_pick(e: ft.FilePickerResultEvent) -> None:
        if not e.files:
            return
        f = e.files[0]
        state["_picked_file"] = f
        upload_label.value = f"📁  {f.name}"
        if run_btn.current:
            run_btn.current.disabled = False
        page.update()

    file_picker = ft.FilePicker(on_result=on_file_pick)
    page.overlay.append(file_picker)

    pick_btn = ft.OutlinedButton(
        "파일 선택",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda _: file_picker.pick_files(
            dialog_title="원본 영상 선택",
            allowed_extensions=["mp4", "mov", "avi", "mkv", "mpeg4"],
        ),
        style=ft.ButtonStyle(
            color=TEXT_PRIMARY,
            side=ft.BorderSide(1, DIVIDER),
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  Edit mode radio
    # ══════════════════════════════════════════════════════════════════════════

    def on_mode_change(e: ft.ControlEvent) -> None:
        edit_mode_ref[0] = e.control.value

    mode_radio = ft.RadioGroup(
        value="Standard",
        on_change=on_mode_change,
        content=ft.Row([
            ft.Radio(value="Standard",   label="Standard",   fill_color=ACCENT),
            ft.Radio(value="Aggressive", label="Aggressive", fill_color=ACCENT),
            ft.Radio(value="Minimal",    label="Minimal",    fill_color=ACCENT),
        ]),
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  Caption slider
    # ══════════════════════════════════════════════════════════════════════════

    _init_y_px  = _read_caption_y()
    _init_pct   = round(_init_y_px / 1920 * 100)

    def on_slider_change(e: ft.ControlEvent) -> None:
        pct = int(e.control.value)
        px  = round(1920 * pct / 100)
        _write_caption_y(px)
        if caption_value_label.current:
            caption_value_label.current.value = f"{pct}%  ({px} px)"
            page.update()

    # ══════════════════════════════════════════════════════════════════════════
    #  Progress polling
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_progress() -> None:
        job_id = state.get("job_id")
        if not job_id or not progress_col.current:
            return

        status   = _read_status(job_id)
        records  = _read_job_records(job_id)
        is_done  = status["done"]
        active   = _active_skill(job_id) if not is_done else None

        rows: list[ft.Control] = []
        for skill in SKILL_ORDER:
            rows.append(_skill_row(skill, records.get(skill), skill == active))

        if is_done:
            if status["error"]:
                rows.append(ft.Container(
                    ft.Text(f"오류: {status['error']}", color=ERROR, size=12),
                    padding=ft.padding.only(top=8),
                ))
            else:
                rows.append(ft.Container(
                    ft.Row([
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=SUCCESS),
                        ft.Text("렌더링 완료!", color=SUCCESS, size=14, weight=ft.FontWeight.W_600),
                    ], spacing=6),
                    padding=ft.padding.only(top=8),
                ))
            _refresh_result(status.get("output_path"))
        else:
            rows.append(ft.Container(height=4))

        progress_col.current.controls = rows
        page.update()

    def _refresh_result(output_path: str | None) -> None:
        if not result_col.current:
            return
        if output_path and Path(output_path).exists():
            result_col.current.controls = [
                ft.Text("출력 경로", size=12, color=TEXT_MUTED),
                ft.Text(output_path, size=11, color=TEXT_MUTED, selectable=True),
                ft.ElevatedButton(
                    "📂  파일 탐색기에서 열기",
                    on_click=lambda _: page.launch_url(f"file:///{Path(output_path).parent}"),
                    style=ft.ButtonStyle(bgcolor=BG_ELEVATED, color=TEXT_PRIMARY),
                ),
            ]
        else:
            result_col.current.controls = [
                ft.Text("편집이 완료되면 여기에 출력 경로가 표시됩니다.", size=12, color=TEXT_MUTED),
            ]
        page.update()

    def _poll_loop() -> None:
        while state.get("polling"):
            job_id = state.get("job_id")
            if job_id:
                _refresh_progress()
                status = _read_status(job_id)
                if status["done"]:
                    state["polling"] = False
                    if run_btn.current:
                        run_btn.current.disabled = False
                        page.update()
                    break
            time.sleep(2)

    # ══════════════════════════════════════════════════════════════════════════
    #  Run pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def on_run_click(_: ft.ControlEvent) -> None:
        picked = state.get("_picked_file")
        if not picked:
            _toast("영상 파일을 선택하세요.", color=ERROR)
            return

        job_id      = f"job_{time.strftime('%Y%m%d_%H%M%S')}"
        staging_dir = STAGING_ROOT / job_id
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Copy uploaded file into staging dir
        src_path = staging_dir / picked.name
        if picked.path:
            import shutil
            shutil.copy2(picked.path, src_path)
        else:
            _toast("파일 경로를 읽을 수 없습니다.", color=ERROR)
            return

        _write_status(job_id, done=False, error=None, output_path=None)
        state["job_id"]      = job_id
        state["source_path"] = str(src_path)
        state["polling"]     = True

        run_btn.current.disabled = True
        page.update()

        threading.Thread(
            target=_run_pipeline,
            args=(job_id, str(src_path), staging_dir),
            daemon=True,
        ).start()
        threading.Thread(target=_poll_loop, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  Chat (Smart Resume)
    # ══════════════════════════════════════════════════════════════════════════

    def _add_chat_bubble(role: str, text: str) -> None:
        is_user = role == "user"
        bubble  = ft.Container(
            content=ft.Text(text, size=12, color=TEXT_PRIMARY, selectable=True),
            bgcolor=ACCENT if is_user else BG_ELEVATED,
            border_radius=ft.border_radius.only(
                top_left=12, top_right=12,
                bottom_left=0 if is_user else 12,
                bottom_right=12 if is_user else 0,
            ),
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            margin=ft.margin.only(
                left=80 if is_user else 0,
                right=0 if is_user else 80,
                bottom=4,
            ),
        )
        chat_history.append({"role": role, "content": text})
        if chat_list.current:
            chat_list.current.controls.append(bubble)
            page.update()

    def _fire_smart_resume(source_path: str, result_obj: object) -> str:
        prior_job_id, prior_records = _find_latest_complete_job()
        if not prior_job_id:
            return (
                f"명세 업데이트: **{result_obj.summary}**\n\n"
                "완료된 렌더 작업이 없어 스마트 리줌을 실행할 수 없습니다. "
                "먼저 'AI 편집 시작'으로 전체 파이프라인을 실행해주세요."
            )

        new_job_id  = f"job_{time.strftime('%Y%m%d_%H%M%S')}"
        staging_dir = STAGING_ROOT / new_job_id
        staging_dir.mkdir(parents=True, exist_ok=True)
        _write_status(new_job_id, done=False, error=None, output_path=None)

        state["job_id"]  = new_job_id
        state["polling"] = True

        threading.Thread(
            target=_run_smart_resume,
            args=(new_job_id, source_path, staging_dir,
                  result_obj.restart_from, prior_records),
            daemon=True,
        ).start()
        threading.Thread(target=_poll_loop, daemon=True).start()

        restart_label = {
            "transcriber":    "전사부터",
            "cutter":         "편집(Cutter)부터",
            "designer":       "디자인(Designer)부터",
            "exporter":       "렌더링(Exporter)만",
            "designer_fast":  "B-roll 패치 후 렌더링(Micro-Resume)",
        }.get(result_obj.restart_from, result_obj.restart_from)

        return (
            f"명세 업데이트: **{result_obj.summary}**\n\n"
            f"스마트 리줌: **{restart_label}** 재실행 중... (Job: `{new_job_id}`)"
        )

    def on_chat_submit(_: ft.ControlEvent) -> None:
        if not chat_input_field.current:
            return
        text = chat_input_field.current.value.strip()
        if not text:
            return
        chat_input_field.current.value = ""
        page.update()

        _add_chat_bubble("user", text)

        from src.pipeline.intent_processor import IntentProcessor
        processor = IntentProcessor()
        result    = processor.process(text)

        if not result.applied:
            reply = (
                f"인식되지 않은 명령입니다: {text}\n\n"
                "지원 명령어: 자막 크게/작게, 골드 테마, 파란/빨간 자막, "
                "자막 아래/위, 침묵 짧게/길게"
            )
        else:
            source_path = state.get("source_path")
            if not source_path:
                reply = (
                    f"명세 업데이트: {result.summary}\n\n"
                    "완료된 렌더 작업이 없습니다. 먼저 영상을 업로드하고 'AI 편집 시작'을 눌러주세요."
                )
            else:
                reply = _fire_smart_resume(source_path, result)

        _add_chat_bubble("assistant", reply)

    # ══════════════════════════════════════════════════════════════════════════
    #  B-roll panel
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_broll() -> None:
        if not broll_col.current:
            return
        items = _read_broll_requests()
        rows: list[ft.Control] = []
        if not items:
            rows.append(ft.Text(
                "주입된 B-roll 에셋이 없습니다. 채팅창에 '자료화면 자동으로 채워줘'를 입력해 보세요.",
                size=12, color=TEXT_MUTED,
            ))
        else:
            for i, req in enumerate(items):
                keyword    = req.get("keyword", "?")
                asset_path = req.get("asset_path", "")
                asset_name = Path(asset_path).name if asset_path else "(경로 없음)"
                opacity    = req.get("opacity", 1.0)
                badge      = "🤖" if "generated" in asset_path.replace("\\", "/") else "📁"

                def make_reroll(idx: int, kw: str):
                    def handler(_: ft.ControlEvent) -> None:
                        broll_items = _read_broll_requests()
                        try:
                            from src.utils.asset_generator import AssetGenerator
                            gen    = AssetGenerator()
                            cached = gen.cache_path(kw)
                            if cached.exists():
                                cached.unlink()
                            new_path = gen.generate(kw)
                            if new_path:
                                broll_items[idx]["asset_path"] = new_path
                                _write_broll_requests(broll_items)
                                _trigger_micro_resume_broll()
                                _toast(f"'{kw}' 재생성 완료 → {Path(new_path).name}")
                            else:
                                _toast(f"'{kw}' 재생성 실패", color=ERROR)
                        except Exception as exc:
                            _toast(str(exc), color=ERROR)
                        _refresh_broll()
                    return handler

                def make_delete(idx: int):
                    def handler(_: ft.ControlEvent) -> None:
                        broll_items = _read_broll_requests()
                        broll_items.pop(idx)
                        _write_broll_requests(broll_items)
                        _trigger_micro_resume_broll()
                        _refresh_broll()
                    return handler

                rows.append(ft.Row([
                    ft.Text(f"{i+1}. {badge} {keyword} — {asset_name} (투명도: {opacity:.0%})",
                            size=12, color=TEXT_PRIMARY, expand=True),
                    ft.IconButton(ft.Icons.REFRESH, icon_color=ACCENT,
                                  tooltip="재생성", on_click=make_reroll(i, keyword)),
                    ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_color=ERROR,
                                  tooltip="삭제", on_click=make_delete(i)),
                ], spacing=4))

        broll_col.current.controls = rows
        page.update()

    def _trigger_micro_resume_broll() -> None:
        source_path = state.get("source_path")
        prior_job_id, prior_records = _find_latest_complete_job()
        if not source_path or not prior_job_id:
            return
        new_job_id  = f"job_{time.strftime('%Y%m%d_%H%M%S')}"
        staging_dir = STAGING_ROOT / new_job_id
        staging_dir.mkdir(parents=True, exist_ok=True)
        _write_status(new_job_id, done=False, error=None, output_path=None)
        state["job_id"]  = new_job_id
        state["polling"] = True
        threading.Thread(
            target=_run_smart_resume,
            args=(new_job_id, source_path, staging_dir, "designer_fast", prior_records),
            daemon=True,
        ).start()
        threading.Thread(target=_poll_loop, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  Build layout
    # ══════════════════════════════════════════════════════════════════════════

    # ── Left column ────────────────────────────────────────────────────────────
    left_panel = ft.Column([
        _card(ft.Column([
            _section_title("입력 설정"),
            ft.Container(height=8),
            ft.Row([pick_btn, upload_label], spacing=10, wrap=True),
            ft.Container(height=12),
            _label("편집 모드"),
            mode_radio,
        ])),

        _card(ft.Column([
            _section_title("자막 위치 (Dynamic Spec)"),
            ft.Container(height=4),
            ft.Row([
                _label("자막 Y 위치"),
                ft.Text(ref=caption_value_label,
                        value=f"{_init_pct}%  ({_init_y_px} px)",
                        size=12, color=ACCENT),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Slider(
                ref=caption_slider,
                min=0, max=100, value=_init_pct,
                divisions=100,
                label="{value}%",
                active_color=ACCENT,
                thumb_color=ACCENT,
                on_change_end=on_slider_change,
            ),
            _label("0% = 상단  /  100% = 하단", size=11),
        ])),

        _card(ft.Column([
            ft.ElevatedButton(
                ref=run_btn,
                text="▶  AI 편집 시작",
                icon=ft.Icons.PLAY_ARROW,
                disabled=True,
                on_click=on_run_click,
                expand=True,
                style=ft.ButtonStyle(
                    bgcolor={
                        ft.ControlState.DEFAULT:  ACCENT,
                        ft.ControlState.HOVERED:  ACCENT_HOVER,
                        ft.ControlState.DISABLED: BG_ELEVATED,
                    },
                    color={
                        ft.ControlState.DEFAULT:  TEXT_PRIMARY,
                        ft.ControlState.DISABLED: TEXT_MUTED,
                    },
                    shape=ft.RoundedRectangleBorder(radius=6),
                    padding=ft.padding.symmetric(vertical=14),
                ),
            ),
        ])),
    ], spacing=0, expand=True)

    # ── Right column (progress + result) ──────────────────────────────────────
    right_panel = ft.Column([
        _card(ft.Column([
            _section_title("진행 상황"),
            ft.Container(height=8),
            ft.Column(
                ref=progress_col,
                controls=[ft.Text("영상을 선택하고 'AI 편집 시작'을 누르세요.",
                                  size=12, color=TEXT_MUTED)],
                spacing=8,
            ),
        ])),

        _card(ft.Column([
            _section_title("결과 영상"),
            ft.Container(height=8),
            ft.Column(
                ref=result_col,
                controls=[ft.Text("편집이 완료되면 여기에 출력 경로가 표시됩니다.",
                                  size=12, color=TEXT_MUTED)],
                spacing=6,
            ),
        ])),
    ], spacing=0, expand=True)

    # ── Chat panel ─────────────────────────────────────────────────────────────
    chat_panel = ft.Column([
        _card(ft.Column([
            _section_title("AI 편집 명령"),
            _label("예: '자막 크게' · '골드 테마' · '침묵 짧게' · '자막 아래로' · '파란 자막'", size=11),
            ft.Container(height=6),
            ft.Container(
                content=ft.ListView(
                    ref=chat_list,
                    controls=[],
                    spacing=2,
                    auto_scroll=True,
                ),
                height=160,
                bgcolor=BG_BASE,
                border_radius=6,
                padding=8,
                border=ft.border.all(1, DIVIDER),
            ),
            ft.Container(height=6),
            ft.Row([
                ft.TextField(
                    ref=chat_input_field,
                    hint_text="편집 명령을 입력하세요...",
                    hint_style=ft.TextStyle(color=TEXT_MUTED),
                    bgcolor=BG_BASE,
                    border_color=DIVIDER,
                    focused_border_color=ACCENT,
                    color=TEXT_PRIMARY,
                    text_size=13,
                    expand=True,
                    on_submit=on_chat_submit,
                    border_radius=6,
                    content_padding=ft.padding.symmetric(horizontal=12, vertical=10),
                ),
                ft.IconButton(
                    ft.Icons.SEND,
                    icon_color=ACCENT,
                    on_click=on_chat_submit,
                    tooltip="전송",
                ),
            ], spacing=6),
        ]), padding=16),
    ], spacing=0)

    # ── B-roll panel ──────────────────────────────────────────────────────────
    broll_panel = _card(ft.Column([
        _section_title("B-roll 자산 관리"),
        _label("현재 주입된 자료화면 목록. 각 항목을 재생성하거나 삭제할 수 있습니다.", size=11),
        ft.Container(height=6),
        ft.Column(ref=broll_col, controls=[], spacing=4),
    ]))

    # ── Title bar ─────────────────────────────────────────────────────────────
    title_bar = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.MOVIE_CREATION, color=ACCENT, size=22),
            ft.Text("VoxEdit AI", size=18, color=TEXT_PRIMARY, weight=ft.FontWeight.W_700),
            ft.Text("Transcriber → Cutter → Designer → Exporter",
                    size=11, color=TEXT_MUTED),
        ], spacing=10),
        bgcolor=BG_SURFACE,
        padding=ft.padding.symmetric(horizontal=20, vertical=14),
        border=ft.border.only(bottom=ft.BorderSide(1, DIVIDER)),
    )

    # ── Main body (two-column) ─────────────────────────────────────────────────
    body = ft.Row([
        ft.Container(content=left_panel,  expand=1, padding=ft.padding.only(left=16, right=8, top=16, bottom=16)),
        ft.Container(content=right_panel, expand=1, padding=ft.padding.only(left=8, right=16, top=16, bottom=16)),
    ], expand=True, alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START)

    # ── Bottom section (chat + broll) ─────────────────────────────────────────
    bottom = ft.Container(
        content=ft.Column([
            _divider(),
            ft.Container(
                content=ft.Row([
                    ft.Container(content=chat_panel, expand=3, padding=ft.padding.only(left=16, right=8, bottom=16)),
                    ft.Container(content=broll_panel, expand=2, padding=ft.padding.only(left=8, right=16, bottom=16)),
                ]),
            ),
        ], spacing=0),
    )

    page.add(
        ft.Column([
            title_bar,
            ft.Container(
                content=ft.Column([body, bottom], spacing=0, scroll=ft.ScrollMode.AUTO),
                expand=True,
            ),
        ], spacing=0, expand=True)
    )

    # Initial B-roll render
    _refresh_broll()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point — native window (not browser)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ft.app(target=main)
