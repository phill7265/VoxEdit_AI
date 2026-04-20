"""
src/ui/app.py

VoxEdit AI — Streamlit UI

Run:
    cd C:/Users/rearl/Documents/work/sales/VoxEdit_AI
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC_FILE = ROOT / "spec" / "editing_style.md"
BROLL_REQUESTS_FILE = ROOT / "spec" / "broll_requests.json"
JOBS_ROOT = ROOT / "harness" / "memory" / "jobs"
STAGING_ROOT = ROOT / "staging"

SKILL_ORDER = ["transcriber", "cutter", "designer", "exporter"]
SKILL_LABELS = {
    "transcriber": "전사 (Transcriber)",
    "cutter":      "편집 (Cutter)",
    "designer":    "디자인 (Designer)",
    "exporter":    "렌더링 (Exporter)",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VoxEdit AI", page_icon="🎬", layout="wide")
st.title("🎬 VoxEdit AI")
st.caption("Transcriber → Cutter → Designer → Exporter 자동 편집 파이프라인")


# ── Session state defaults ────────────────────────────────────────────────────
for _k, _v in {"job_id": None, "source_path": None}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Disk-based status (thread-safe) ──────────────────────────────────────────

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
    p.write_text(
        json.dumps({"done": done, "error": error, "output_path": output_path}),
        encoding="utf-8",
    )


# ── Spec helpers ──────────────────────────────────────────────────────────────

def _read_caption_y() -> int:
    text = SPEC_FILE.read_text(encoding="utf-8")
    m = re.search(r"CAPTION_Y_PX\s*[:=]\s*(\d+)", text)
    return int(m.group(1)) if m else 1620


def _write_caption_y(px: int) -> None:
    text = SPEC_FILE.read_text(encoding="utf-8")
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


# ── Smart Resume helpers ──────────────────────────────────────────────────────

def _find_latest_complete_job() -> tuple[str, dict] | tuple[None, None]:
    """Return (job_id, records_dict) for the most recent 4-skill-success job."""
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


def _run_smart_resume(
    job_id: str,
    source_path: str,
    staging_dir: Path,
    force_resume_from: str,
    borrow_records: dict,
) -> None:
    """Smart Resume pipeline runner — re-runs from force_resume_from only."""
    try:
        from src.pipeline.orchestrator import WorkflowManager
        from harness.memory.manager import SkillRecord

        # Convert plain dicts (from JSON) to SkillRecord objects
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


# ── Pipeline runner (background thread) ───────────────────────────────────────

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


# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ══ Left column ══════════════════════════════════════════════════════════════
with col_left:
    st.subheader("입력 설정")

    uploaded = st.file_uploader(
        "원본 영상 (raw_video.mp4)",
        type=["mp4", "mov", "avi", "mkv", "mpeg4"],
    )

    st.radio(
        "편집 모드",
        options=["Standard", "Aggressive", "Minimal"],
        horizontal=True,
        help="Standard: 기본 편집 / Aggressive: 침묵 최대 제거 / Minimal: 침묵만 제거",
    )

    st.divider()
    st.subheader("자막 위치 (Dynamic Spec)")

    current_y_px = _read_caption_y()
    current_pct = round(current_y_px / 1920 * 100)

    caption_pct = st.slider(
        "자막 Y 위치 (%)", min_value=0, max_value=100,
        value=current_pct, step=1, format="%d%%",
        help="0% = 화면 상단 / 100% = 화면 하단",
    )
    if caption_pct != current_pct:
        new_px = round(1920 * caption_pct / 100)
        _write_caption_y(new_px)
        st.success(f"자막 위치 명세가 업데이트되었습니다 (CAPTION_Y_PX = {new_px}px, {caption_pct}%)")

    st.divider()

    # Determine running state from disk
    job_id = st.session_state.get("job_id")
    status = _read_status(job_id) if job_id else {"done": False}
    pipeline_running = job_id is not None and not status["done"]

    run_btn = st.button(
        "▶ AI 편집 시작",
        type="primary",
        disabled=pipeline_running or uploaded is None,
        use_container_width=True,
    )
    if uploaded is None:
        st.caption("영상 파일을 선택하면 편집 버튼이 활성화됩니다.")

    if run_btn and uploaded is not None:
        job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}"
        staging_dir = STAGING_ROOT / job_id
        staging_dir.mkdir(parents=True, exist_ok=True)

        src_path = staging_dir / uploaded.name
        src_path.write_bytes(uploaded.getbuffer())

        # Write initial status so polling sees "not done" immediately
        _write_status(job_id, done=False, error=None, output_path=None)

        st.session_state["job_id"] = job_id
        st.session_state["source_path"] = str(src_path)

        threading.Thread(
            target=_run_pipeline,
            args=(job_id, str(src_path), staging_dir),
            daemon=True,
        ).start()
        st.rerun()


# ══ Right column ══════════════════════════════════════════════════════════════
with col_right:
    st.subheader("진행 상황")

    job_id = st.session_state.get("job_id")

    if job_id is None:
        st.info("영상을 선택하고 'AI 편집 시작'을 누르세요.")
    else:
        status = _read_status(job_id)
        pipeline_done = status["done"]
        pipeline_error = status["error"]
        output_path = status["output_path"]

        records = _read_job_records(job_id)
        active = _active_skill(job_id) if not pipeline_done else None

        st.caption(f"Job ID: `{job_id}`")

        for skill in SKILL_ORDER:
            label = SKILL_LABELS[skill]
            rec = records.get(skill)
            if rec:
                if rec.get("status") == "success":
                    detail = f"완료  |  cursor: {rec.get('cursor_start','')} → {rec.get('cursor_end','')}"
                    st.markdown(f"✅ **{label}** — {detail}")
                else:
                    st.markdown(f"❌ **{label}** — {rec.get('error', 'failed')}")
            elif skill == active:
                st.markdown(f"⏳ **{label}** — 처리 중...")
            else:
                st.markdown(f"⬜ {label}")

        if pipeline_done and not pipeline_error:
            st.success("렌더링 완료!")
        elif pipeline_error:
            st.error(f"파이프라인 오류: {pipeline_error}")

        if not pipeline_done:
            time.sleep(2)
            st.rerun()

    st.divider()
    st.subheader("결과 영상")

    output_path = _read_status(job_id)["output_path"] if job_id else None

    if output_path and Path(output_path).exists():
        st.video(Path(output_path).read_bytes())
        st.caption(f"출력 경로: `{output_path}`")
        with open(output_path, "rb") as f:
            st.download_button(
                label="output.mp4 다운로드",
                data=f,
                file_name="output.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
    else:
        st.info("편집이 완료되면 여기에 영상이 표시됩니다.")

# ══ Chat input: AI 명령 (Smart Resume) ═══════════════════════════════════════
st.divider()
st.subheader("AI 편집 명령")
st.caption(
    "예: '자막 크게' · '골드 테마' · '침묵 짧게' · '자막 아래로' · '파란 자막'"
)

# Chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Render past messages
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

chat_input = st.chat_input("편집 명령을 입력하세요...")

if chat_input:
    st.session_state["chat_history"].append({"role": "user", "content": chat_input})

    from src.pipeline.intent_processor import IntentProcessor
    processor = IntentProcessor()
    result = processor.process(chat_input)

    if not result.applied:
        reply = f"인식되지 않은 명령입니다: **{chat_input}**\n\n지원 명령어: 자막 크게/작게, 골드 테마, 파란/빨간 자막, 자막 아래/위, 침묵 짧게/길게"
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()
    else:
        source_path = st.session_state.get("source_path")
        prior_job_id, prior_records = _find_latest_complete_job()

        if not source_path or not prior_job_id:
            reply = (
                f"명세 업데이트: **{result.summary}**\n\n"
                "완료된 렌더 작업이 없어 스마트 리줌을 실행할 수 없습니다. "
                "먼저 'AI 편집 시작'으로 전체 파이프라인을 실행해주세요."
            )
        else:
            new_job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}"
            staging_dir = STAGING_ROOT / new_job_id
            staging_dir.mkdir(parents=True, exist_ok=True)
            _write_status(new_job_id, done=False, error=None, output_path=None)

            st.session_state["job_id"] = new_job_id

            threading.Thread(
                target=_run_smart_resume,
                args=(new_job_id, source_path, staging_dir,
                      result.restart_from, prior_records),
                daemon=True,
            ).start()

            restart_label = {
                "transcriber": "전사부터",
                "cutter": "편집(Cutter)부터",
                "designer": "디자인(Designer)부터",
                "exporter": "렌더링(Exporter)만",
                "designer_fast": "B-roll 패치 후 렌더링(Micro-Resume)",
            }.get(result.restart_from, result.restart_from)

            reply = (
                f"명세 업데이트: **{result.summary}**\n\n"
                f"스마트 리줌: **{restart_label}** 재실행 중... (Job: `{new_job_id}`)"
            )

        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()


# ══ B-roll 자산 관리자 ══════════════════════════════════════════════════════════
st.divider()
st.subheader("B-roll 자산 관리")
st.caption("현재 주입된 자료화면 목록. 각 항목을 재생성하거나 삭제할 수 있습니다.")


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


def _trigger_micro_resume() -> None:
    """Fire a designer_fast Smart Resume in a background thread."""
    source_path = st.session_state.get("source_path")
    prior_job_id, prior_records = _find_latest_complete_job()
    if not source_path or not prior_job_id:
        return
    new_job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}"
    staging_dir = STAGING_ROOT / new_job_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    _write_status(new_job_id, done=False, error=None, output_path=None)
    st.session_state["job_id"] = new_job_id
    threading.Thread(
        target=_run_smart_resume,
        args=(new_job_id, source_path, staging_dir, "designer_fast", prior_records),
        daemon=True,
    ).start()


broll_items = _read_broll_requests()

if not broll_items:
    st.info("주입된 B-roll 에셋이 없습니다. 채팅창에 '자료화면 자동으로 채워줘'를 입력해 보세요.")
else:
    for i, req in enumerate(broll_items):
        keyword = req.get("keyword", "?")
        asset_path = req.get("asset_path", "")
        asset_name = Path(asset_path).name if asset_path else "(경로 없음)"
        opacity = req.get("opacity", 1.0)
        is_generated = "generated" in asset_path.replace("\\", "/")
        badge = "🤖" if is_generated else "📁"

        col_info, col_reroll, col_del = st.columns([4, 1, 1])
        with col_info:
            st.markdown(
                f"**{i+1}.** {badge} `{keyword}` — {asset_name} "
                f"(투명도: {opacity:.0%})"
            )

        with col_reroll:
            if st.button("🔄 재생성", key=f"reroll_{i}", use_container_width=True):
                from src.utils.asset_generator import AssetGenerator
                gen = AssetGenerator()
                # Delete cache so generator forces a fresh run
                cached = gen.cache_path(keyword)
                if cached.exists():
                    cached.unlink()
                new_path = gen.generate(keyword)
                if new_path:
                    broll_items[i]["asset_path"] = new_path
                    _write_broll_requests(broll_items)
                    _trigger_micro_resume()
                    st.success(f"'{keyword}' 재생성 완료 → {Path(new_path).name}")
                else:
                    st.error(f"'{keyword}' 재생성 실패")
                st.rerun()

        with col_del:
            if st.button("🗑 삭제", key=f"delete_{i}", use_container_width=True):
                broll_items.pop(i)
                _write_broll_requests(broll_items)
                _trigger_micro_resume()
                st.rerun()
