"""
src/pipeline/intent_processor.py

Chat-to-Spec Intent Processor for VoxEdit AI.

Responsibilities
----------------
1. Parse Korean natural-language chat input into a structured intent.
2. Apply the intent to spec/editing_style.md (key-value overrides).
3. Return a ResumeAdvice: which skill the pipeline should restart from.

Smart Resume mapping
--------------------
Spec field              → Restart skill
────────────────────────────────────────
CAPTION_FONT_SIZE_PT    → exporter   (drawtext only)
CAPTION_COLOR           → exporter
HIGHLIGHT_COLOR         → exporter
CAPTION_Y_PX            → exporter
SILENCE_MIN_DURATION_S  → cutter     (changes which segments exist)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_SPEC_FILE = _ROOT / "spec" / "editing_style.md"
_BROLL_REQUESTS_FILE = _ROOT / "spec" / "broll_requests.json"
_ASSETS_BROLL_DIR = _ROOT / "assets" / "broll"
_JOBS_ROOT = _ROOT / "harness" / "memory" / "jobs"
_AUDIO_STYLE_FILE = _ROOT / "spec" / "audio_style.md"

# ── Korean/Arabic ordinal → 1-based integer ──────────────────────────────────
_ORDINAL_MAP: dict[str, int] = {
    "첫": 1, "하나": 1, "한": 1, "일": 1, "1": 1,
    "둘": 2, "두": 2, "이": 2, "2": 2,
    "셋": 3, "세": 3, "삼": 3, "3": 3,
    "넷": 4, "네": 4, "사": 4, "4": 4,
    "다섯": 5, "오": 5, "5": 5,
    "여섯": 6, "육": 6, "6": 6,
    "일곱": 7, "칠": 7, "7": 7,
    "여덟": 8, "팔": 8, "8": 8,
    "아홉": 9, "구": 9, "9": 9,
    "열": 10, "십": 10, "10": 10,
}

# ── Korean stop words for noun extraction ─────────────────────────────────────
_STOP_WORDS: frozenset[str] = frozenset({
    # Korean particles / conjunctions / auxiliaries
    "그리고", "하지만", "그래서", "때문에", "그런데", "있는", "없는", "되는",
    "하는", "이런", "저런", "어떤", "위해", "통해", "대한", "이후", "이전",
    "같은", "이렇게", "저렇게", "정말", "너무", "매우", "아주", "좀", "조금",
    "있어", "없어", "합니다", "이다", "이며", "에서", "으로", "까지",
    # English function words
    "the", "and", "or", "but", "for", "with", "from", "this", "that",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "it", "its", "we", "our", "you", "your", "they", "their",
})

# ── Restart skill precedence (lower index = earlier in pipeline) ──────────────
_SKILL_ORDER = ["transcriber", "cutter", "designer", "exporter"]


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    """Outcome of processing one chat message.

    Attributes
    ----------
    summary      : Human-readable description of what changed.
    changes      : {spec_field: new_value} for every field that was written.
    restart_from : The earliest skill that must be re-run (Smart Resume).
    applied      : False when no intent was recognised.
    """
    summary: str
    changes: dict[str, Any] = field(default_factory=dict)
    restart_from: str = "exporter"
    applied: bool = True


# ── Spec I/O ──────────────────────────────────────────────────────────────────

def _read_spec_value(field_name: str) -> str | None:
    """Return raw string value of a key-value line in editing_style.md."""
    try:
        text = _SPEC_FILE.read_text(encoding="utf-8")
        m = re.search(rf"{re.escape(field_name)}\s*[:=]\s*([^\|\n]+)", text)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def _write_spec_value(field_name: str, value: str) -> None:
    """Upsert a `FIELD: value` line in editing_style.md.

    If a line with this field name exists (including inside a markdown table),
    it is replaced in-place.  Otherwise a new line is appended under
    ## 6. Caption Position (or a new section).
    """
    text = _SPEC_FILE.read_text(encoding="utf-8")
    new_fragment = f"{field_name}: {value}"

    if field_name in text:
        text = re.sub(
            rf"{re.escape(field_name)}\s*[:=]\s*[^\|\n]+",
            new_fragment,
            text,
        )
    else:
        # Append to the Caption Position section or create one
        if "## 6. Caption Position" in text:
            text = text.rstrip() + f"\n{new_fragment}\n"
        else:
            text += f"\n\n## 6. Caption Position\n\n{new_fragment}\n"

    _SPEC_FILE.write_text(text, encoding="utf-8")


def _read_int(field_name: str, default: int) -> int:
    raw = _read_spec_value(field_name)
    if raw is None:
        return default
    try:
        return int(float(raw.strip()))
    except ValueError:
        return default


def _read_float(field_name: str, default: float) -> float:
    raw = _read_spec_value(field_name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _read_str(field_name: str, default: str) -> str:
    raw = _read_spec_value(field_name)
    return raw.strip() if raw else default


# ── Audio style I/O ───────────────────────────────────────────────────────────

def _read_audio_float(field_name: str, default: float) -> float:
    try:
        text = _AUDIO_STYLE_FILE.read_text(encoding="utf-8")
        m = re.search(rf"{re.escape(field_name)}\s*[:=]\s*([+-]?[\d.]+)", text)
        return float(m.group(1)) if m else default
    except Exception:
        return default


def _write_audio_value(field_name: str, value: str) -> None:
    """Upsert a `FIELD: value` line in audio_style.md."""
    try:
        text = _AUDIO_STYLE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = "# VoxEdit AI — Audio Style\n\n"
    new_fragment = f"{field_name}: {value}"
    if field_name in text:
        text = re.sub(
            rf"{re.escape(field_name)}\s*[:=]\s*[^\|\n]+",
            new_fragment,
            text,
        )
    else:
        text = text.rstrip() + f"\n{new_fragment}\n"
    _AUDIO_STYLE_FILE.write_text(text, encoding="utf-8")


# ── Intent handlers ───────────────────────────────────────────────────────────

def _handle_font_larger(_text: str) -> IntentResult:
    current = _read_int("CAPTION_FONT_SIZE_PT", 72)
    new_val = round(current * 1.2)
    _write_spec_value("CAPTION_FONT_SIZE_PT", str(new_val))
    return IntentResult(
        summary=f"자막 폰트 크기 {current}pt → {new_val}pt (+20%)",
        changes={"CAPTION_FONT_SIZE_PT": new_val},
        restart_from="exporter",
    )


def _handle_font_smaller(_text: str) -> IntentResult:
    current = _read_int("CAPTION_FONT_SIZE_PT", 72)
    new_val = max(24, round(current * 0.8))
    _write_spec_value("CAPTION_FONT_SIZE_PT", str(new_val))
    return IntentResult(
        summary=f"자막 폰트 크기 {current}pt → {new_val}pt (-20%)",
        changes={"CAPTION_FONT_SIZE_PT": new_val},
        restart_from="exporter",
    )


def _handle_gold_theme(_text: str) -> IntentResult:
    _write_spec_value("HIGHLIGHT_COLOR", "#FFD700")
    _write_spec_value("CAPTION_COLOR", "#FFFFFF")
    return IntentResult(
        summary="골드 테마 적용 (강조=#FFD700, 자막=#FFFFFF)",
        changes={"HIGHLIGHT_COLOR": "#FFD700", "CAPTION_COLOR": "#FFFFFF"},
        restart_from="exporter",
    )


def _handle_blue_caption(_text: str) -> IntentResult:
    _write_spec_value("CAPTION_COLOR", "#4488FF")
    return IntentResult(
        summary="자막 색상 → 파란색 (#4488FF)",
        changes={"CAPTION_COLOR": "#4488FF"},
        restart_from="exporter",
    )


def _handle_red_caption(_text: str) -> IntentResult:
    _write_spec_value("CAPTION_COLOR", "#FF4444")
    return IntentResult(
        summary="자막 색상 → 빨간색 (#FF4444)",
        changes={"CAPTION_COLOR": "#FF4444"},
        restart_from="exporter",
    )


def _handle_white_caption(_text: str) -> IntentResult:
    _write_spec_value("CAPTION_COLOR", "#FFFFFF")
    return IntentResult(
        summary="자막 색상 → 흰색 (#FFFFFF, 기본)",
        changes={"CAPTION_COLOR": "#FFFFFF"},
        restart_from="exporter",
    )


def _handle_caption_down(_text: str) -> IntentResult:
    current = _read_int("CAPTION_Y_PX", 1344)
    new_val = min(1820, current + 192)  # +10% of 1920, cap near bottom
    _write_spec_value("CAPTION_Y_PX", str(new_val))
    return IntentResult(
        summary=f"자막 위치 아래로: {current}px → {new_val}px",
        changes={"CAPTION_Y_PX": new_val},
        restart_from="exporter",
    )


def _handle_caption_up(_text: str) -> IntentResult:
    current = _read_int("CAPTION_Y_PX", 1344)
    new_val = max(100, current - 192)  # -10% of 1920, cap near top
    _write_spec_value("CAPTION_Y_PX", str(new_val))
    return IntentResult(
        summary=f"자막 위치 위로: {current}px → {new_val}px",
        changes={"CAPTION_Y_PX": new_val},
        restart_from="exporter",
    )


def _handle_silence_longer(_text: str) -> IntentResult:
    current = _read_float("SILENCE_MIN_DURATION_S", 0.5)
    new_val = round(min(3.0, current + 0.2), 2)
    _write_spec_value("SILENCE_MIN_DURATION_S", f"{new_val:.2f}")
    return IntentResult(
        summary=f"침묵 감지 기준 {current:.2f}s → {new_val:.2f}s (더 많이 제거)",
        changes={"SILENCE_MIN_DURATION_S": new_val},
        restart_from="cutter",
    )


def _handle_silence_shorter(_text: str) -> IntentResult:
    current = _read_float("SILENCE_MIN_DURATION_S", 0.5)
    new_val = round(max(0.1, current - 0.1), 2)
    _write_spec_value("SILENCE_MIN_DURATION_S", f"{new_val:.2f}")
    return IntentResult(
        summary=f"침묵 감지 기준 {current:.2f}s → {new_val:.2f}s (덜 제거)",
        changes={"SILENCE_MIN_DURATION_S": new_val},
        restart_from="cutter",
    )


# ── B-roll helpers ────────────────────────────────────────────────────────────

def _find_broll_asset(keyword: str) -> str | None:
    """Search assets/broll/ for a file whose name contains the keyword.

    Matching is case-insensitive.  Returns the first match's absolute path,
    or None when no asset is found.
    """
    import json as _json
    if not _ASSETS_BROLL_DIR.exists():
        return None
    kw_lower = keyword.lower()
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    for f in sorted(_ASSETS_BROLL_DIR.iterdir()):
        if f.suffix.lower() in video_exts and kw_lower in f.stem.lower():
            return str(f)
    return None


def _read_broll_requests() -> list[dict]:
    """Return current broll_requests from spec/broll_requests.json."""
    if not _BROLL_REQUESTS_FILE.exists():
        return []
    try:
        import json as _json
        return _json.loads(_BROLL_REQUESTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_broll_requests(requests: list[dict]) -> None:
    """Persist broll_requests list to spec/broll_requests.json."""
    import json as _json
    _BROLL_REQUESTS_FILE.write_text(
        _json.dumps(requests, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _handle_broll_insert(text: str) -> IntentResult:
    """Parse "자료화면 [키워드] 삽입" and inject a b-roll request."""
    # Extract keyword — everything between 자료화면/B-roll and 삽입/추가 (if present)
    patterns = [
        r"(?:자료화면|b.roll|broll)\s+(.+?)(?:\s*(?:삽입|추가|넣어|넣기))?$",
        r"(.+?)\s+(?:자료화면|b.roll|broll)(?:\s*(?:삽입|추가))?$",
    ]
    keyword = ""
    for pat in patterns:
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            keyword = m.group(1).strip()
            break

    if not keyword:
        return IntentResult(
            summary="B-roll 키워드를 찾지 못했습니다. 예: '자료화면 고양이 삽입'",
            changes={},
            restart_from="designer",
            applied=False,
        )

    asset_path = _find_broll_asset(keyword)
    if not asset_path:
        return IntentResult(
            summary=(
                f"'{keyword}'에 맞는 에셋이 assets/broll/에 없습니다. "
                f"{keyword}.mp4 파일을 추가한 후 다시 시도하세요."
            ),
            changes={},
            restart_from="designer",
            applied=False,
        )

    # Append to existing requests
    requests = _read_broll_requests()
    new_req = {
        "keyword": keyword,
        "asset_path": asset_path,
        "opacity": 1.0,
        "mode": "overlay",
    }
    # Deduplicate by keyword
    requests = [r for r in requests if r.get("keyword") != keyword]
    requests.append(new_req)
    _write_broll_requests(requests)

    return IntentResult(
        summary=f"자료화면 추가: '{keyword}' → {Path(asset_path).name}",
        changes={"BROLL_REQUESTS": new_req},
        restart_from="designer",
    )


def _handle_broll_remove(text: str) -> IntentResult:
    """Remove a b-roll request by keyword."""
    m = re.search(r"(?:자료화면|b.roll)\s+(.+?)\s*(?:제거|삭제|빼)", text, re.IGNORECASE)
    if not m:
        return IntentResult(
            summary="제거할 자료화면 키워드를 찾지 못했습니다.",
            changes={}, restart_from="designer", applied=False,
        )
    keyword = m.group(1).strip()
    requests = _read_broll_requests()
    before = len(requests)
    requests = [r for r in requests if r.get("keyword") != keyword]
    _write_broll_requests(requests)
    removed = before - len(requests)
    return IntentResult(
        summary=f"자료화면 '{keyword}' 제거 ({removed}개)",
        changes={"BROLL_REQUESTS": None},
        restart_from="designer",
    )


def _parse_ordinal(text: str) -> int | None:
    """Extract a 1-based ordinal from text.  Returns None for 'last'/'마지막'.

    Accepts Arabic digits ("2번"), Korean numerals ("두번째"), or "마지막".
    Returns -1 as the sentinel for "last" / "마지막".
    """
    if re.search(r"마지막|last|최후|끝", text):
        return -1  # sentinel for "last"
    # Arabic digits first ("2번째" → 2)
    m = re.search(r"(\d+)번", text)
    if m:
        return int(m.group(1))
    # Korean ordinal words
    for word, idx in sorted(_ORDINAL_MAP.items(), key=lambda kv: -len(kv[0])):
        if word in text:
            return idx
    return None


def _handle_broll_delete_by_index(text: str) -> IntentResult:
    """Delete the N-th b-roll entry: '2번 자료화면 빼줘'."""
    idx = _parse_ordinal(text)
    if idx is None:
        return IntentResult(
            summary="삭제할 번호를 인식하지 못했습니다. 예: '2번 자료화면 빼줘'",
            changes={}, restart_from="designer_fast", applied=False,
        )
    requests = _read_broll_requests()
    if not requests:
        return IntentResult(
            summary="삭제할 자료화면이 없습니다.",
            changes={}, restart_from="designer_fast", applied=False,
        )
    target = len(requests) - 1 if idx == -1 else idx - 1   # 0-based
    if target < 0 or target >= len(requests):
        return IntentResult(
            summary=f"인덱스 {idx}이(가) 범위를 벗어났습니다 (총 {len(requests)}개).",
            changes={}, restart_from="designer_fast", applied=False,
        )
    removed = requests.pop(target)
    _write_broll_requests(requests)
    label = "마지막" if idx == -1 else f"{idx}번"
    return IntentResult(
        summary=f"자료화면 {label} 삭제: '{removed.get('keyword', '?')}'",
        changes={"BROLL_REQUESTS": None},
        restart_from="designer_fast",
    )


def _handle_broll_reroll_by_index(text: str) -> IntentResult:
    """Re-generate the N-th b-roll asset: '마지막 그림 다시 그려줘'."""
    from src.utils.asset_generator import AssetGenerator

    idx = _parse_ordinal(text)
    if idx is None:
        return IntentResult(
            summary="재생성할 번호를 인식하지 못했습니다. 예: '마지막 그림 다시 그려줘'",
            changes={}, restart_from="designer_fast", applied=False,
        )
    requests = _read_broll_requests()
    if not requests:
        return IntentResult(
            summary="재생성할 자료화면이 없습니다.",
            changes={}, restart_from="designer_fast", applied=False,
        )
    target = len(requests) - 1 if idx == -1 else idx - 1
    if target < 0 or target >= len(requests):
        return IntentResult(
            summary=f"인덱스 {idx}이(가) 범위를 벗어났습니다 (총 {len(requests)}개).",
            changes={}, restart_from="designer_fast", applied=False,
        )

    req = requests[target]
    keyword = req.get("keyword", "")

    # Delete cached file so AssetGenerator regenerates it
    generator = AssetGenerator()
    cached = generator.cache_path(keyword)
    if cached.exists():
        cached.unlink()

    # Regenerate
    new_path = generator.generate(keyword)
    if not new_path:
        return IntentResult(
            summary=f"'{keyword}' 에셋 재생성에 실패했습니다.",
            changes={}, restart_from="designer_fast", applied=False,
        )

    requests[target]["asset_path"] = new_path
    _write_broll_requests(requests)

    label = "마지막" if idx == -1 else f"{idx}번"
    return IntentResult(
        summary=f"자료화면 {label} 재생성 완료: '{keyword}' → {Path(new_path).name}",
        changes={"BROLL_REQUESTS": requests[target]},
        restart_from="designer_fast",
    )


def _load_latest_transcript_words() -> list[dict]:
    """Scan harness/memory/jobs/ and return words from the most recent successful transcriber."""
    import json as _json
    if not _JOBS_ROOT.exists():
        return []
    for job_dir in sorted(_JOBS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        for rec_file in sorted(job_dir.glob("*.json")):
            try:
                rec = _json.loads(rec_file.read_text(encoding="utf-8"))
                if rec.get("skill") == "transcriber" and rec.get("status") == "success":
                    t_path = rec.get("output_path", "")
                    if t_path and Path(t_path).exists():
                        data = _json.loads(Path(t_path).read_text(encoding="utf-8"))
                        return data.get("words", [])
            except Exception:
                pass
    return []


def _extract_noun_candidates(words: list[dict], max_candidates: int = 20) -> list[str]:
    """Extract unique candidate nouns from a transcript word list.

    Heuristic: words ≥ 2 chars, strip punctuation, exclude stop words.
    Returns up to max_candidates unique tokens in order of first occurrence.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    for w in words:
        raw = w.get("word", "").strip().lower()
        token = re.sub(r"[^a-zA-Z0-9가-힣]", "", raw)
        if len(token) >= 2 and token not in _STOP_WORDS and token not in seen:
            seen.add(token)
            candidates.append(token)
            if len(candidates) >= max_candidates:
                break
    return candidates


def _handle_broll_auto_fill(_text: str) -> IntentResult:
    """Auto-match b-roll assets from transcript nouns via AssetIndexer.

    Falls back to AssetGenerator for keywords with no local match.
    Generated images are saved to assets/generated/ and registered in
    broll_requests.json.  Smart Resume restarts from Designer.
    """
    from src.utils.asset_indexer import AssetIndexer
    from src.utils.asset_generator import AssetGenerator

    # Load the most recent successful transcript
    words = _load_latest_transcript_words()
    if not words:
        return IntentResult(
            summary=(
                "트랜스크립트를 찾을 수 없습니다. "
                "먼저 'AI 편집 시작'으로 전체 파이프라인을 실행해주세요."
            ),
            changes={},
            restart_from="designer",
            applied=False,
        )

    candidates = _extract_noun_candidates(words)

    # Attach generator as fallback — "auto" uses Replicate if API token is set,
    # otherwise falls back to placeholder (no API call).
    generator = AssetGenerator()
    indexer = AssetIndexer(generator=generator)
    indexer.build()

    matched: list[dict] = []
    seen_paths: set[str] = set()
    generated_count = 0

    for word in candidates:
        path = indexer.find(word)
        if path and path not in seen_paths:
            seen_paths.add(path)
            # Detect whether this path came from the generator
            is_generated = "generated" in path.replace("\\", "/")
            if is_generated:
                generated_count += 1
            matched.append({
                "keyword": word,
                "asset_path": path,
                "opacity": 1.0,
                "mode": "overlay",
            })

    if not matched:
        return IntentResult(
            summary=(
                "트랜스크립트 명사와 일치하는 에셋이 없고 생성에도 실패했습니다. "
                "assets/broll/에 파일을 추가하거나 REPLICATE_API_TOKEN을 설정하세요."
            ),
            changes={},
            restart_from="designer",
            applied=False,
        )

    # Merge with existing (user-added) requests — don't clobber manual entries
    existing = _read_broll_requests()
    existing_keywords: set[str] = {r.get("keyword", "") for r in existing}
    merged = existing + [m for m in matched if m["keyword"] not in existing_keywords]
    _write_broll_requests(merged)

    summary_items = [f"'{m['keyword']}'" for m in matched]
    gen_note = f" (생성됨: {generated_count}개)" if generated_count else ""
    return IntentResult(
        summary=f"자동 매칭 완료: {', '.join(summary_items)} ({len(matched)}개 에셋{gen_note})",
        changes={"BROLL_REQUESTS": matched},
        restart_from="designer",
    )


def _handle_more_dynamic(_text: str) -> IntentResult:
    """'좀 더 역동적으로' / '더 활발하게' — increase RHYTHM_INTENSITY."""
    current = _read_float("RHYTHM_INTENSITY", 0.5)
    new_val = round(min(1.0, current + 0.2), 2)
    _write_spec_value("RHYTHM_INTENSITY", f"{new_val:.2f}")
    return IntentResult(
        summary=f"리듬 강도 {current:.2f} → {new_val:.2f} (+0.2) — DynamicZoom 강화",
        changes={"RHYTHM_INTENSITY": new_val},
        restart_from="visual_fast",
    )


def _handle_less_dynamic(_text: str) -> IntentResult:
    """'좀 차분하게' / '덜 역동적으로' — decrease RHYTHM_INTENSITY."""
    current = _read_float("RHYTHM_INTENSITY", 0.5)
    new_val = round(max(0.0, current - 0.2), 2)
    _write_spec_value("RHYTHM_INTENSITY", f"{new_val:.2f}")
    return IntentResult(
        summary=f"리듬 강도 {current:.2f} → {new_val:.2f} (-0.2) — DynamicZoom 약화",
        changes={"RHYTHM_INTENSITY": new_val},
        restart_from="visual_fast",
    )


def _handle_shake_screen(_text: str) -> IntentResult:
    """'화면 흔들어줘' / '강하게 줌' — set RHYTHM_INTENSITY to high."""
    _write_spec_value("RHYTHM_INTENSITY", "0.80")
    return IntentResult(
        summary="화면 역동성 강화 (RHYTHM_INTENSITY=0.80)",
        changes={"RHYTHM_INTENSITY": 0.80},
        restart_from="visual_fast",
    )


def _handle_focus_here(_text: str) -> IntentResult:
    """'여기에 집중해줘' — enable focus zoom and increase intensity slightly."""
    current = _read_float("RHYTHM_INTENSITY", 0.5)
    new_val = round(min(1.0, max(0.6, current + 0.1)), 2)
    _write_spec_value("RHYTHM_INTENSITY", f"{new_val:.2f}")
    _write_spec_value("ZOOM_FOCUS_ENABLED", "true")
    return IntentResult(
        summary=f"집중 연출 활성화 (RHYTHM_INTENSITY={new_val:.2f}, ZOOM_FOCUS_ENABLED=true)",
        changes={"RHYTHM_INTENSITY": new_val, "ZOOM_FOCUS_ENABLED": "true"},
        restart_from="visual_fast",
    )


def _handle_bgm_louder(_text: str) -> IntentResult:
    """'음악 크게' / 'BGM 키워줘' — increase BGM_BASE_VOLUME."""
    current = _read_audio_float("BGM_BASE_VOLUME", 0.30)
    new_val = round(min(1.0, current + 0.1), 2)
    _write_audio_value("BGM_BASE_VOLUME", f"{new_val:.2f}")
    return IntentResult(
        summary=f"BGM 볼륨 {current:.2f} → {new_val:.2f} (+10%)",
        changes={"BGM_BASE_VOLUME": new_val},
        restart_from="audio_only",
    )


def _handle_bgm_quieter(_text: str) -> IntentResult:
    """'음악 작게' / 'BGM 낮춰줘' — decrease BGM_BASE_VOLUME."""
    current = _read_audio_float("BGM_BASE_VOLUME", 0.30)
    new_val = round(max(0.0, current - 0.1), 2)
    _write_audio_value("BGM_BASE_VOLUME", f"{new_val:.2f}")
    return IntentResult(
        summary=f"BGM 볼륨 {current:.2f} → {new_val:.2f} (-10%)",
        changes={"BGM_BASE_VOLUME": new_val},
        restart_from="audio_only",
    )


def _handle_bgm_off(_text: str) -> IntentResult:
    """'BGM 꺼줘' / '음악 없애줘' — mute BGM."""
    _write_audio_value("BGM_BASE_VOLUME", "0.00")
    _write_audio_value("BGM_STYLE", "off")
    return IntentResult(
        summary="BGM 꺼짐 (BGM_BASE_VOLUME=0.00)",
        changes={"BGM_BASE_VOLUME": 0.0, "BGM_STYLE": "off"},
        restart_from="audio_only",
    )


def _handle_bgm_calm(_text: str) -> IntentResult:
    """'잔잔한 걸로' / '조용한 음악' — set BGM to low soft level."""
    _write_audio_value("BGM_BASE_VOLUME", "0.15")
    _write_audio_value("BGM_STYLE", "calm")
    return IntentResult(
        summary="BGM 잔잔하게 설정 (BGM_BASE_VOLUME=0.15, BGM_STYLE=calm)",
        changes={"BGM_BASE_VOLUME": 0.15, "BGM_STYLE": "calm"},
        restart_from="audio_only",
    )


def _handle_bgm_duck_stronger(_text: str) -> IntentResult:
    """'목소리 더 잘 들리게' / '더킹 강하게' — increase VOICE_DUCK_DB magnitude."""
    current = _read_audio_float("VOICE_DUCK_DB", -20.0)
    new_val = round(max(-40.0, current - 6.0), 1)
    _write_audio_value("VOICE_DUCK_DB", f"{new_val:.1f}")
    return IntentResult(
        summary=f"더킹 강도 {current:.1f}dB → {new_val:.1f}dB (목소리 우선도 ↑)",
        changes={"VOICE_DUCK_DB": new_val},
        restart_from="audio_only",
    )


def _handle_bgm_duck_weaker(_text: str) -> IntentResult:
    """'BGM 더 들리게' / '더킹 약하게' — reduce VOICE_DUCK_DB magnitude."""
    current = _read_audio_float("VOICE_DUCK_DB", -20.0)
    new_val = round(min(-6.0, current + 6.0), 1)
    _write_audio_value("VOICE_DUCK_DB", f"{new_val:.1f}")
    return IntentResult(
        summary=f"더킹 강도 {current:.1f}dB → {new_val:.1f}dB (BGM 우선도 ↑)",
        changes={"VOICE_DUCK_DB": new_val},
        restart_from="audio_only",
    )


def _handle_broll_opacity(text: str) -> IntentResult:
    """Set opacity for a b-roll keyword: "자료화면 고양이 투명도 0.5"."""
    m = re.search(
        r"(?:자료화면|b.roll)\s+(.+?)\s+(?:투명도|opacity)\s+([\d.]+)",
        text, re.IGNORECASE,
    )
    if not m:
        return IntentResult(
            summary="투명도 설정 형식: '자료화면 [키워드] 투명도 0.7'",
            changes={}, restart_from="designer", applied=False,
        )
    keyword = m.group(1).strip()
    opacity = max(0.0, min(1.0, float(m.group(2))))
    requests = _read_broll_requests()
    updated = False
    for req in requests:
        if req.get("keyword") == keyword:
            req["opacity"] = opacity
            updated = True
    if not updated:
        return IntentResult(
            summary=f"'{keyword}' 자료화면이 없습니다. 먼저 삽입하세요.",
            changes={}, restart_from="designer", applied=False,
        )
    _write_broll_requests(requests)
    return IntentResult(
        summary=f"자료화면 '{keyword}' 투명도 → {opacity:.1f}",
        changes={"BROLL_REQUESTS": {"keyword": keyword, "opacity": opacity}},
        restart_from="designer",
    )


# ── Intent routing table ──────────────────────────────────────────────────────
# Each entry: (list_of_regex_patterns, handler_fn)
_INTENT_TABLE: list[tuple[list[str], Any]] = [
    (
        [r"자막.{0,4}크[게거]", r"글씨.{0,4}크[게거]", r"폰트.{0,4}크[게거]", r"크게.{0,4}자막", r"글씨\s*크게"],
        _handle_font_larger,
    ),
    (
        [r"자막.{0,4}작[게거]", r"글씨.{0,4}작[게거]", r"폰트.{0,4}작[게거]", r"작게.{0,4}자막"],
        _handle_font_smaller,
    ),
    (
        [r"골드.{0,4}테마", r"금색.{0,4}자막", r"자막.{0,4}금색", r"황금.{0,4}자막", r"골드"],
        _handle_gold_theme,
    ),
    (
        [r"파란.{0,4}자막", r"자막.{0,4}파란", r"파란색\s*자막", r"파랗게"],
        _handle_blue_caption,
    ),
    (
        [r"빨간.{0,4}자막", r"자막.{0,4}빨간", r"빨간색\s*자막", r"빨갛게"],
        _handle_red_caption,
    ),
    (
        [r"흰.{0,4}자막", r"자막.{0,4}흰", r"기본.{0,4}테마", r"기본.{0,4}자막", r"흰색"],
        _handle_white_caption,
    ),
    (
        [r"자막.{0,6}아래", r"자막.{0,6}하단", r"하단.{0,4}자막", r"아래[로로]+\s*자막", r"자막\s*내려"],
        _handle_caption_down,
    ),
    (
        [r"자막.{0,6}위", r"자막.{0,6}상단", r"상단.{0,4}자막", r"위[로로]+\s*자막", r"자막\s*올려"],
        _handle_caption_up,
    ),
    (
        [r"침묵.{0,6}길[게거]", r"침묵.{0,6}높[여이]", r"기준.{0,4}높", r"더.{0,4}길게\s*침묵", r"침묵\s*더\s*길"],
        _handle_silence_longer,
    ),
    (
        [r"침묵.{0,6}짧[게거]", r"침묵.{0,6}낮[춰워]", r"기준.{0,4}낮", r"더.{0,4}짧게\s*침묵", r"침묵\s*더\s*짧"],
        _handle_silence_shorter,
    ),
    # ── Director's Chair — visual rhythm intents ──────────────────────────
    (
        [r"화면\s*(?:좀\s*)?흔들어", r"강하게\s*줌", r"쭉\s*줌", r"더\s*강하게"],
        _handle_shake_screen,
    ),
    (
        [r"여기.*집중", r"집중.*해줘", r"이\s*부분.*강조", r"포커스"],
        _handle_focus_here,
    ),
    (
        [r"(?:좀\s*)?더\s*역동적", r"역동적으로", r"더\s*활발하게", r"생동감", r"더\s*역동"],
        _handle_more_dynamic,
    ),
    (
        [r"좀\s*차분", r"덜\s*역동", r"조용하게", r"안정적으로", r"부드럽게"],
        _handle_less_dynamic,
    ),
    # ── Audio / BGM interference intents ─────────────────────────────────
    (
        [r"BGM\s*꺼", r"음악\s*꺼", r"배경음\s*꺼", r"음악\s*없애", r"BGM\s*없애"],
        _handle_bgm_off,
    ),
    (
        [r"잔잔한", r"조용한\s*(?:음악|BGM)", r"부드러운\s*(?:음악|BGM)", r"잔잔하게"],
        _handle_bgm_calm,
    ),
    (
        [r"음악\s*크[게거]", r"BGM\s*크[게거]", r"BGM\s*키워", r"음악\s*키워", r"배경음\s*높[여이]"],
        _handle_bgm_louder,
    ),
    (
        [r"음악\s*작[게거]", r"BGM\s*작[게거]", r"BGM\s*낮[춰워]", r"음악\s*낮[춰워]", r"배경음\s*낮[춰워]"],
        _handle_bgm_quieter,
    ),
    (
        [r"더킹\s*강[하하]", r"목소리\s*더\s*잘\s*들", r"더킹\s*강[화하게]", r"ducking\s*strong"],
        _handle_bgm_duck_stronger,
    ),
    (
        [r"더킹\s*약[하하]", r"BGM\s*더\s*들[리려]", r"더킹\s*약[화하게]", r"ducking\s*weak"],
        _handle_bgm_duck_weaker,
    ),
    # Auto-fill must come before specific insert to avoid partial match
    (
        [
            r"자료화면\s*자동",
            r"자동\s*(?:으로\s*)?(?:자료화면|b.roll|broll)",
            r"(?:자료화면|b.roll)\s*자동\s*(?:채워|매칭|삽입|추가)",
            r"자동\s*매칭",
            r"자동으로\s*채워",
        ],
        _handle_broll_auto_fill,
    ),
    # Index-based re-roll: "마지막 그림 다시 그려줘" / "3번 자료화면 재생성"
    (
        [
            r"(?:마지막|\d+번?째?|첫|두|세|네|다섯)\s*(?:자료화면|그림|b.roll|에셋)\s*(?:다시\s*)?(?:그려|재생성|만들|re.roll|재롤)",
            r"(?:다시\s*그려|재생성|re.roll)\s*(?:마지막|\d+번?째?)",
        ],
        _handle_broll_reroll_by_index,
    ),
    # Index-based delete: "2번 자료화면 빼줘" / "마지막 그림 삭제"
    (
        [
            r"(?:마지막|\d+번?째?|첫|두|세|네|다섯)\s*(?:자료화면|그림|b.roll|에셋)\s*(?:빼|삭제|제거|없애)",
            r"(?:빼|삭제|제거)\s*(?:마지막|\d+번?째?)\s*(?:자료화면|그림)",
        ],
        _handle_broll_delete_by_index,
    ),
    # B-roll opacity must come before insert to avoid partial match
    (
        [r"(?:자료화면|b.roll|broll).+(?:투명도|opacity)\s+[\d.]+"],
        _handle_broll_opacity,
    ),
    (
        [r"(?:자료화면|b.roll|broll).+(?:제거|삭제|빼)", r"(?:제거|삭제).+(?:자료화면|b.roll)"],
        _handle_broll_remove,
    ),
    (
        [r"(?:자료화면|b.roll|broll)\s+\S", r"\S+\s+(?:자료화면|b.roll)\s*(?:삽입|추가)?"],
        _handle_broll_insert,
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────

class IntentProcessor:
    """Parse Korean chat input and apply changes to spec/editing_style.md."""

    def process(self, text: str) -> IntentResult:
        """Match `text` against the intent table and apply the first match.

        Returns an IntentResult with `applied=False` when nothing matches.
        """
        text = text.strip()
        for patterns, handler in _INTENT_TABLE:
            for pat in patterns:
                if re.search(pat, text):
                    result = handler(text)
                    # Invalidate context_manager spec cache so next pipeline
                    # run reads the updated values (especially silence_threshold_s)
                    _clear_context_cache()
                    return result

        return IntentResult(
            summary=f"'{text}' — 인식된 명령어가 없습니다.",
            changes={},
            restart_from="exporter",
            applied=False,
        )

    @staticmethod
    def restart_skill_for_fields(changed_fields: list[str]) -> str:
        """Return the earliest pipeline skill that must re-run given changed fields.

        Rules
        -----
        SILENCE_MIN_DURATION_S  → cutter          (structural cut-point change)
        BROLL_INDEX_DELETE/REROLL → designer_fast (broll patch only, skip designer)
        everything else         → exporter        (visual-only, drawtext override)
        """
        if "SILENCE_MIN_DURATION_S" in changed_fields:
            return "cutter"
        if any(f.startswith("BROLL_INDEX") for f in changed_fields):
            return "designer_fast"
        if any(f in changed_fields for f in ("RHYTHM_INTENSITY", "ZOOM_FOCUS_ENABLED")):
            return "visual_fast"
        if any(f in changed_fields for f in ("BGM_BASE_VOLUME", "BGM_STYLE", "VOICE_DUCK_DB",
                                              "DUCK_ATTACK_MS", "DUCK_RELEASE_MS")):
            return "audio_only"
        return "exporter"


def _clear_context_cache() -> None:
    """Invalidate context_manager's module-level spec cache."""
    try:
        from src.pipeline import context_manager
        context_manager.clear_spec_cache()
    except Exception:
        pass
