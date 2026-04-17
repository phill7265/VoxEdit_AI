"""
src/utils/test_designer.py

Unit tests for the Designer skill.

Covers
------
  · detect_keyword_style        — keyword → gold style mapping
  · generate_captions           — safe-zone position, font size, grouping
  · build_zoom_overlays         — jump-cut effect → zoom element
  · build_duck_events           — VAD → audio duck element
  · run_designer                — end-to-end pipeline
  · VisualElement.to_dict       — serialisation contract
  · skills/designer/main.run    — harness integration (SkillRecord)

All tests are mock-based (no audio files, no model downloads needed).

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_designer.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.designer.logic import (
    CANVAS_HEIGHT_PX,
    CANVAS_WIDTH_PX,
    CAPTION_BOTTOM_PCT,
    CAPTION_COLOR,
    CAPTION_FONT_SIZE_PT,
    CAPTION_Y_PX,
    DEFAULT_KEYWORDS,
    DUCK_ATTACK_MS,
    DUCK_DB,
    DUCK_RELEASE_MS,
    HIGHLIGHT_COLOR,
    JUMP_CUT_EFFECT_KEY,
    JUMP_CUT_ZOOM,
    STYLE_DEFAULT,
    STYLE_GOLD,
    VAD_CONFIDENCE_THRESHOLD,
    DesignerResult,
    VisualElement,
    build_duck_events,
    build_zoom_overlays,
    detect_keyword_style,
    generate_captions,
    run_designer,
)
from skills.designer.main import run as designer_run
import harness.memory.manager as mgr_mod


# ── Fixtures ──────────────────────────────────────────────────────────────

def _w(word: str, start_ms: int, end_ms: int) -> dict:
    return {"word": word, "start_ms": start_ms, "end_ms": end_ms, "confidence": 0.95}


def _vad(start_s: float, end_s: float, is_voice: bool, confidence: float) -> dict:
    return {
        "start_s": start_s,
        "end_s": end_s,
        "is_voice": is_voice,
        "confidence": confidence,
        "avg_probability": confidence,
    }


def _cut_seg(start: float, end: float, action: str, effects=None) -> dict:
    return {"start": start, "end": end, "action": action, "effects": effects or []}


def _keep_jump(start: float, end: float) -> dict:
    return _cut_seg(start, end, "keep", [JUMP_CUT_EFFECT_KEY])


def _keep(start: float, end: float) -> dict:
    return _cut_seg(start, end, "keep")


def _delete(start: float, end: float) -> dict:
    return _cut_seg(start, end, "delete")


# ── Caption safe-zone constant ────────────────────────────────────────────
SAFE_Y = CANVAS_HEIGHT_PX * (1.0 - CAPTION_BOTTOM_PCT)   # 864.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestDetectKeywordStyle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDetectKeywordStyle(unittest.TestCase):
    """detect_keyword_style — Gold-highlight keyword detection."""

    def test_keyword_returns_gold_style(self):
        style = detect_keyword_style("이것이 핵심입니다", DEFAULT_KEYWORDS)
        self.assertEqual(style, STYLE_GOLD)

    def test_non_keyword_returns_default_style(self):
        style = detect_keyword_style("안녕하세요 오늘도 좋은 하루", DEFAULT_KEYWORDS)
        self.assertEqual(style, STYLE_DEFAULT)

    def test_keyword_중요(self):
        self.assertEqual(detect_keyword_style("중요한 내용입니다", DEFAULT_KEYWORDS), STYLE_GOLD)

    def test_keyword_주의(self):
        self.assertEqual(detect_keyword_style("주의하세요", DEFAULT_KEYWORDS), STYLE_GOLD)

    def test_keyword_결론(self):
        self.assertEqual(detect_keyword_style("결론은 이렇습니다", DEFAULT_KEYWORDS), STYLE_GOLD)

    def test_custom_keywords_override(self):
        custom = frozenset(["important", "critical"])
        self.assertEqual(detect_keyword_style("this is important", custom), STYLE_GOLD)
        self.assertEqual(detect_keyword_style("nothing here", custom), STYLE_DEFAULT)

    def test_empty_text_returns_default(self):
        self.assertEqual(detect_keyword_style("", DEFAULT_KEYWORDS), STYLE_DEFAULT)

    def test_multiple_keywords_in_text_returns_gold(self):
        style = detect_keyword_style("핵심 중요 포인트", DEFAULT_KEYWORDS)
        self.assertEqual(style, STYLE_GOLD)

    def test_keyword_as_substring_returns_gold(self):
        """'핵심적' contains '핵심' — should still match."""
        style = detect_keyword_style("핵심적인 내용", DEFAULT_KEYWORDS)
        self.assertEqual(style, STYLE_GOLD)

    def test_no_keywords_set_always_default(self):
        style = detect_keyword_style("핵심 중요 주의", frozenset())
        self.assertEqual(style, STYLE_DEFAULT)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestCaptionSafeZone
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCaptionSafeZone(unittest.TestCase):
    """generate_captions — safe-zone positioning (bottom 20 %)."""

    def _captions(self, words=None):
        if words is None:
            words = [_w("hello", 0, 1000), _w("world", 1100, 2000)]
        return generate_captions(words)

    def test_caption_y_position_at_or_below_safe_zone(self):
        """All captions must have position_y_px ≥ SAFE_Y (864.0 px for 1080p)."""
        for cap in self._captions():
            self.assertGreaterEqual(cap.position_y_px, SAFE_Y,
                f"Caption y={cap.position_y_px:.1f} is above safe zone {SAFE_Y:.1f}")

    def test_caption_y_default_value_is_864(self):
        """Default 1080p canvas, 20 % bottom zone → y = 864.0."""
        for cap in self._captions():
            self.assertAlmostEqual(cap.position_y_px, 864.0, places=1)

    def test_caption_y_constant_matches_computed(self):
        """CAPTION_Y_PX constant must equal CANVAS_HEIGHT_PX * (1 - CAPTION_BOTTOM_PCT)."""
        expected = CANVAS_HEIGHT_PX * (1.0 - CAPTION_BOTTOM_PCT)
        self.assertAlmostEqual(CAPTION_Y_PX, expected, places=3)

    def test_caption_font_size_is_72pt(self):
        for cap in self._captions():
            self.assertEqual(cap.font_size_pt, 72)

    def test_caption_x_position_centred(self):
        """Caption should be horizontally centred (x = 960 for 1920 px wide canvas)."""
        for cap in self._captions():
            self.assertAlmostEqual(cap.position_x_px, CANVAS_WIDTH_PX / 2.0, places=1)

    def test_custom_canvas_height_respects_20pct(self):
        """Different canvas height → y = height × 0.80."""
        caps = generate_captions(
            [_w("test", 0, 1000)],
            canvas_height_px=720,
        )
        for cap in caps:
            self.assertAlmostEqual(cap.position_y_px, 720 * 0.80, places=1)

    def test_custom_caption_zone_pct(self):
        """caption_bottom_pct=0.30 → y = height × 0.70."""
        caps = generate_captions(
            [_w("test", 0, 1000)],
            caption_bottom_pct=0.30,
        )
        expected_y = CANVAS_HEIGHT_PX * 0.70
        for cap in caps:
            self.assertAlmostEqual(cap.position_y_px, expected_y, places=1)

    def test_safe_zone_sensor_flag_raised_when_y_too_high(self):
        """Sensor must flag captions placed above the safe zone."""
        from skills.designer.logic import _run_designer_sensors
        result = DesignerResult()
        result.visual_elements.append(
            VisualElement(type="caption", start=0, end=1, text="bad",
                          position_y_px=200.0)   # above safe zone
        )
        result = _run_designer_sensors(result)
        safe_zone_flags = [f for f in result.sensor_flags if "SAFE_ZONE" in f]
        self.assertGreater(len(safe_zone_flags), 1 - 1,
                           "Expected a SAFE_ZONE sensor flag")

    def test_no_sensor_flag_when_caption_in_safe_zone(self):
        from skills.designer.logic import _run_designer_sensors
        result = DesignerResult()
        result.visual_elements.append(
            VisualElement(type="caption", start=0, end=1, text="ok",
                          position_y_px=SAFE_Y + 1)   # within safe zone
        )
        result = _run_designer_sensors(result)
        safe_zone_flags = [f for f in result.sensor_flags if "SAFE_ZONE" in f]
        self.assertEqual(len(safe_zone_flags), 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestGenerateCaptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerateCaptions(unittest.TestCase):
    """generate_captions — text grouping, timing, keyword styling."""

    def test_empty_words_returns_empty_list(self):
        self.assertEqual(generate_captions([]), [])

    def test_returns_caption_visual_elements(self):
        caps = generate_captions([_w("안녕", 0, 500)])
        self.assertTrue(all(c.type == "caption" for c in caps))

    def test_caption_text_from_words(self):
        caps = generate_captions([_w("hello", 0, 500), _w("world", 600, 1100)])
        self.assertEqual(len(caps), 1)
        self.assertIn("hello", caps[0].text)
        self.assertIn("world", caps[0].text)

    def test_caption_start_from_first_word(self):
        caps = generate_captions([_w("hi", 1500, 2000)])
        self.assertAlmostEqual(caps[0].start, 1.5, places=2)

    def test_caption_end_from_last_word(self):
        caps = generate_captions([_w("hi", 1500, 2000)])
        self.assertAlmostEqual(caps[0].end, 2.0, places=2)

    def test_gap_above_threshold_splits_block(self):
        """Gap of 1.0 s (> 0.5 s threshold) → two separate caption blocks."""
        words = [_w("first", 0, 500), _w("second", 1500, 2000)]
        caps = generate_captions(words, gap_split_s=0.5)
        self.assertEqual(len(caps), 2)

    def test_gap_below_threshold_keeps_same_block(self):
        """Gap of 0.2 s (< 0.5 s) → same caption block."""
        words = [_w("first", 0, 500), _w("second", 700, 1200)]
        caps = generate_captions(words, gap_split_s=0.5)
        self.assertEqual(len(caps), 1)

    def test_max_words_triggers_new_block(self):
        """6 words with max_words=5 → at least 2 blocks."""
        words = [_w(f"w{i}", i * 100, i * 100 + 80) for i in range(6)]
        caps = generate_captions(words, max_words=5, gap_split_s=10.0)
        self.assertGreaterEqual(len(caps), 2)

    def test_keyword_in_block_gives_gold_style(self):
        caps = generate_captions([_w("핵심", 0, 500)], keywords=DEFAULT_KEYWORDS)
        self.assertTrue(any(c.style == STYLE_GOLD for c in caps))

    def test_keyword_in_block_gives_gold_color(self):
        caps = generate_captions([_w("핵심", 0, 500)], keywords=DEFAULT_KEYWORDS)
        gold_caps = [c for c in caps if c.style == STYLE_GOLD]
        self.assertTrue(all(c.color == HIGHLIGHT_COLOR for c in gold_caps))

    def test_non_keyword_block_has_default_color(self):
        caps = generate_captions([_w("hello", 0, 500)], keywords=frozenset())
        self.assertTrue(all(c.color == CAPTION_COLOR for c in caps))

    def test_non_keyword_block_has_default_style(self):
        caps = generate_captions([_w("hello", 0, 500)], keywords=frozenset())
        self.assertTrue(all(c.style == STYLE_DEFAULT for c in caps))

    def test_mixed_words_keyword_wins_for_block(self):
        """One keyword in a block → entire block gets gold style."""
        words = [_w("이것이", 0, 400), _w("핵심입니다", 500, 1000)]
        caps = generate_captions(words, keywords=DEFAULT_KEYWORDS, gap_split_s=5.0)
        self.assertTrue(any(c.style == STYLE_GOLD for c in caps))

    def test_unsorted_words_handled_correctly(self):
        """Input words in reversed order should still produce correct captions."""
        words = [_w("second", 1000, 1500), _w("first", 0, 500)]
        caps = generate_captions(words, gap_split_s=5.0)
        # Both words merged into one block; first word comes first
        self.assertAlmostEqual(caps[0].start, 0.0, places=2)

    def test_gold_color_is_ffd700(self):
        self.assertEqual(HIGHLIGHT_COLOR, "#FFD700")

    def test_default_color_is_white(self):
        self.assertEqual(CAPTION_COLOR, "#FFFFFF")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestBuildZoomOverlays
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildZoomOverlays(unittest.TestCase):
    """build_zoom_overlays — jump-cut effect → zoom VisualElement."""

    def test_jump_cut_keep_creates_zoom_element(self):
        segs = [_keep_jump(5.0, 10.0)]
        zooms = build_zoom_overlays(segs)
        self.assertEqual(len(zooms), 1)
        self.assertEqual(zooms[0].type, "zoom")

    def test_keep_without_effect_no_zoom(self):
        segs = [_keep(0.0, 5.0)]
        self.assertEqual(len(build_zoom_overlays(segs)), 0)

    def test_delete_segment_no_zoom(self):
        segs = [_delete(0.0, 1.0)]
        self.assertEqual(len(build_zoom_overlays(segs)), 0)

    def test_zoom_factor_is_1_1(self):
        segs = [_keep_jump(0.0, 5.0)]
        zooms = build_zoom_overlays(segs)
        self.assertAlmostEqual(zooms[0].zoom_factor, JUMP_CUT_ZOOM, places=3)

    def test_zoom_factor_constant_is_1_1(self):
        self.assertAlmostEqual(JUMP_CUT_ZOOM, 1.1, places=3)

    def test_zoom_start_end_match_segment(self):
        segs = [_keep_jump(3.5, 7.2)]
        zooms = build_zoom_overlays(segs)
        self.assertAlmostEqual(zooms[0].start, 3.5, places=2)
        self.assertAlmostEqual(zooms[0].end, 7.2, places=2)

    def test_zoom_anchor_defaults_to_centre(self):
        segs = [_keep_jump(0.0, 5.0)]
        zoom = build_zoom_overlays(segs)[0]
        self.assertAlmostEqual(zoom.anchor_x, 0.5)
        self.assertAlmostEqual(zoom.anchor_y, 0.5)

    def test_multiple_jump_cuts_create_multiple_zooms(self):
        segs = [_keep_jump(0.0, 5.0), _keep(5.0, 8.0), _keep_jump(10.0, 15.0)]
        zooms = build_zoom_overlays(segs)
        self.assertEqual(len(zooms), 2)

    def test_empty_segments_returns_empty(self):
        self.assertEqual(build_zoom_overlays([]), [])

    def test_jump_cut_effect_key_string(self):
        self.assertEqual(JUMP_CUT_EFFECT_KEY, "jump_cut_zoom_1.1")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestBuildDuckEvents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildDuckEvents(unittest.TestCase):
    """build_duck_events — VAD confidence ≥ 0.85 → audio duck event."""

    def test_voice_above_threshold_creates_duck(self):
        segs = [_vad(1.0, 3.0, True, 0.90)]
        events = build_duck_events(segs)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "duck")

    def test_non_voice_vad_no_duck(self):
        segs = [_vad(1.0, 3.0, False, 0.95)]
        self.assertEqual(len(build_duck_events(segs)), 0)

    def test_voice_below_threshold_no_duck(self):
        segs = [_vad(1.0, 3.0, True, 0.70)]
        self.assertEqual(len(build_duck_events(segs)), 0)

    def test_voice_exactly_at_threshold_creates_duck(self):
        segs = [_vad(1.0, 3.0, True, VAD_CONFIDENCE_THRESHOLD)]
        self.assertEqual(len(build_duck_events(segs)), 1)

    def test_duck_values_match_spec(self):
        """duck_db=-20, attack=150ms, release=500ms per spec §4."""
        events = build_duck_events([_vad(0.0, 2.0, True, 0.90)])
        e = events[0]
        self.assertAlmostEqual(e.duck_db, DUCK_DB)
        self.assertAlmostEqual(e.attack_ms, DUCK_ATTACK_MS)
        self.assertAlmostEqual(e.release_ms, DUCK_RELEASE_MS)

    def test_duck_spec_values_match_constants(self):
        self.assertAlmostEqual(DUCK_DB, -20.0)
        self.assertAlmostEqual(DUCK_ATTACK_MS, 150.0)
        self.assertAlmostEqual(DUCK_RELEASE_MS, 500.0)

    def test_duck_start_end_from_vad_segment(self):
        events = build_duck_events([_vad(2.5, 4.8, True, 0.90)])
        self.assertAlmostEqual(events[0].start, 2.5, places=2)
        self.assertAlmostEqual(events[0].end, 4.8, places=2)

    def test_vad_confidence_threshold_constant(self):
        self.assertAlmostEqual(VAD_CONFIDENCE_THRESHOLD, 0.85)

    def test_empty_vad_returns_empty(self):
        self.assertEqual(build_duck_events([]), [])

    def test_mixed_vad_only_qualifying_creates_duck(self):
        segs = [
            _vad(0.0, 1.0, True, 0.90),
            _vad(1.0, 2.0, False, 0.99),
            _vad(2.0, 3.0, True, 0.60),
            _vad(3.0, 4.0, True, 0.88),
        ]
        events = build_duck_events(segs)
        self.assertEqual(len(events), 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestVisualElementToDict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVisualElementToDict(unittest.TestCase):
    """VisualElement.to_dict — serialisation contract."""

    def test_caption_dict_has_required_keys(self):
        elem = VisualElement(type="caption", start=1.0, end=3.0, text="hello")
        d = elem.to_dict()
        for key in ("type", "start", "end", "text", "style", "color",
                    "position_x_px", "position_y_px", "font_size_pt"):
            self.assertIn(key, d)

    def test_overlay_dict_has_name_key(self):
        elem = VisualElement(type="overlay", start=5.0, end=8.0, name="subscribe_cta")
        d = elem.to_dict()
        self.assertIn("name", d)
        self.assertEqual(d["name"], "subscribe_cta")

    def test_zoom_dict_has_zoom_factor(self):
        elem = VisualElement(type="zoom", start=0.0, end=5.0, zoom_factor=1.1)
        d = elem.to_dict()
        self.assertIn("zoom_factor", d)
        self.assertAlmostEqual(d["zoom_factor"], 1.1, places=3)

    def test_duck_dict_has_duck_db(self):
        elem = VisualElement(type="duck", start=0.0, end=3.0, duck_db=-20.0)
        d = elem.to_dict()
        self.assertIn("duck_db", d)
        self.assertAlmostEqual(d["duck_db"], -20.0)

    def test_start_end_are_rounded_to_3dp(self):
        elem = VisualElement(type="caption", start=1.0001, end=3.9999, text="x")
        d = elem.to_dict()
        self.assertEqual(d["start"], round(1.0001, 3))
        self.assertEqual(d["end"], round(3.9999, 3))

    def test_gold_caption_dict_has_gold_color(self):
        elem = VisualElement(
            type="caption", start=0.0, end=1.0,
            text="핵심", style=STYLE_GOLD, color=HIGHLIGHT_COLOR,
        )
        d = elem.to_dict()
        self.assertEqual(d["color"], "#FFD700")
        self.assertEqual(d["style"], STYLE_GOLD)

    def test_type_value_preserved(self):
        for t in ("caption", "overlay", "zoom", "duck"):
            elem = VisualElement(type=t, start=0.0, end=1.0)
            self.assertEqual(elem.to_dict()["type"], t)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestRunDesigner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunDesigner(unittest.TestCase):
    """run_designer — full pipeline integration."""

    def _words(self):
        return [
            _w("이것이", 500, 900),
            _w("핵심입니다", 950, 1600),
            _w("꼭", 2500, 2800),
            _w("기억하세요", 2850, 3500),
        ]

    def _cut_segments(self):
        return [
            _keep(0.0, 2.0),
            _delete(2.0, 3.0),
            _keep_jump(3.0, 5.0),
        ]

    def _vad_segments(self):
        return [
            _vad(0.5, 1.6, True, 0.92),
            _vad(2.5, 3.5, True, 0.88),
        ]

    def test_returns_designer_result(self):
        result = run_designer(self._words(), self._cut_segments(), self._vad_segments())
        self.assertIsInstance(result, DesignerResult)

    def test_result_has_visual_elements(self):
        result = run_designer(self._words(), self._cut_segments(), self._vad_segments())
        self.assertIsInstance(result.visual_elements, list)
        self.assertGreater(len(result.visual_elements), 0)

    def test_captions_produced(self):
        result = run_designer(self._words(), self._cut_segments(), [])
        self.assertGreater(len(result.captions), 0)

    def test_zoom_produced_for_jump_cut(self):
        result = run_designer([], self._cut_segments(), [])
        self.assertGreater(len(result.zooms), 0)

    def test_duck_produced_for_qualifying_vad(self):
        result = run_designer([], [], self._vad_segments())
        self.assertGreater(len(result.duck_events), 0)

    def test_keyword_in_words_produces_highlight(self):
        result = run_designer(self._words(), [], [], keywords=DEFAULT_KEYWORDS)
        self.assertGreater(len(result.highlights), 0)

    def test_all_captions_in_safe_zone(self):
        result = run_designer(self._words(), [], [])
        for cap in result.captions:
            self.assertGreaterEqual(
                cap.position_y_px, SAFE_Y,
                f"Caption '{cap.text}' y={cap.position_y_px} above safe zone {SAFE_Y}",
            )

    def test_elements_sorted_by_start_time(self):
        result = run_designer(self._words(), self._cut_segments(), self._vad_segments())
        starts = [e.start for e in result.visual_elements]
        self.assertEqual(starts, sorted(starts))

    def test_empty_all_inputs_returns_empty_result(self):
        result = run_designer([], [], [])
        self.assertEqual(result.visual_elements, [])

    def test_sensor_flags_is_list(self):
        result = run_designer(self._words(), self._cut_segments(), self._vad_segments())
        self.assertIsInstance(result.sensor_flags, list)

    def test_captions_accessor_returns_only_captions(self):
        result = run_designer(self._words(), self._cut_segments(), self._vad_segments())
        self.assertTrue(all(e.type == "caption" for e in result.captions))

    def test_zooms_accessor_returns_only_zooms(self):
        result = run_designer([], self._cut_segments(), [])
        self.assertTrue(all(e.type == "zoom" for e in result.zooms))

    def test_duck_events_accessor_returns_only_ducks(self):
        result = run_designer([], [], self._vad_segments())
        self.assertTrue(all(e.type == "duck" for e in result.duck_events))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestDesignerMain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDesignerMain(unittest.TestCase):
    """skills/designer/main.run — harness integration."""

    def _make_cut_list(self, tmp_path: Path) -> Path:
        data = {
            "cut_segments": [
                {"start": 0.0, "end": 2.0, "action": "keep", "effects": []},
                {"start": 2.0, "end": 2.5, "action": "delete", "effects": [], "reason": "silence"},
                {"start": 2.5, "end": 5.0, "action": "keep", "effects": ["jump_cut_zoom_1.1"]},
            ],
            "metadata": {"window_start_s": 0.0, "window_end_s": 5.0},
        }
        p = tmp_path / "cut_list.json"
        p.write_text(json.dumps(data))
        return p

    def _make_transcript(self, tmp_path: Path) -> Path:
        data = {
            "full_text": "이것이 핵심입니다 꼭 기억하세요",
            "words": [
                {"word": "이것이", "start_ms": 100, "end_ms": 500, "confidence": 0.9},
                {"word": "핵심입니다", "start_ms": 600, "end_ms": 1200, "confidence": 0.9},
                {"word": "꼭", "start_ms": 2000, "end_ms": 2300, "confidence": 0.9},
                {"word": "기억하세요", "start_ms": 2400, "end_ms": 3000, "confidence": 0.9},
            ],
            "vad_segments": [
                {"start_s": 0.1, "end_s": 1.2, "is_voice": True,
                 "confidence": 0.91, "avg_probability": 0.91},
                {"start_s": 2.0, "end_s": 3.0, "is_voice": True,
                 "confidence": 0.87, "avg_probability": 0.87},
            ],
            "metadata": {"model": "base", "language": "ko",
                         "duration_s": 5.0, "word_count": 4, "avg_confidence": 0.9},
        }
        p = tmp_path / "transcript.json"
        p.write_text(json.dumps(data, ensure_ascii=False))
        return p

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.staging_dir = self.tmp_path / "staging" / "job_designer_001"
        self.jobs_root = self.tmp_path / "jobs"
        self._patcher = patch.object(mgr_mod, "JOBS_ROOT", self.jobs_root)
        self._patcher.start()
        self.cut_list_path = self._make_cut_list(self.tmp_path)
        self.transcript_path = self._make_transcript(self.tmp_path)

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def _run(self):
        return designer_run(
            job_id="job_designer_001",
            staging_dir=self.staging_dir,
            cut_list_path=str(self.cut_list_path),
            transcript_path=str(self.transcript_path),
        )

    # ── Status ────────────────────────────────────────────────────────────

    def test_record_status_is_success(self):
        record = self._run()
        self.assertEqual(record.status, "success",
                         f"Expected success, got: {record.error}")

    def test_record_skill_name_is_designer(self):
        record = self._run()
        self.assertEqual(record.skill, "designer")

    # ── Output file ───────────────────────────────────────────────────────

    def test_annotated_timeline_json_exists(self):
        self._run()
        self.assertTrue((self.staging_dir / "annotated_timeline.json").exists())

    def test_annotated_timeline_has_visual_elements_key(self):
        self._run()
        raw = json.loads((self.staging_dir / "annotated_timeline.json").read_text())
        self.assertIn("visual_elements", raw)

    def test_annotated_timeline_contains_caption(self):
        self._run()
        raw = json.loads((self.staging_dir / "annotated_timeline.json").read_text())
        types = {e["type"] for e in raw["visual_elements"]}
        self.assertIn("caption", types)

    def test_annotated_timeline_contains_zoom(self):
        self._run()
        raw = json.loads((self.staging_dir / "annotated_timeline.json").read_text())
        types = {e["type"] for e in raw["visual_elements"]}
        self.assertIn("zoom", types)

    def test_annotated_captions_in_safe_zone(self):
        self._run()
        raw = json.loads((self.staging_dir / "annotated_timeline.json").read_text())
        for elem in raw["visual_elements"]:
            if elem["type"] == "caption":
                self.assertGreaterEqual(
                    elem["position_y_px"], SAFE_Y,
                    f"Caption y={elem['position_y_px']} above safe zone {SAFE_Y}",
                )

    # ── Handover record ───────────────────────────────────────────────────

    def test_handover_record_written_to_memory(self):
        self._run()
        records = list((self.jobs_root / "job_designer_001").glob("*.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].name, "003_designer.json")

    def test_handover_payload_has_output_key(self):
        record = self._run()
        self.assertIn("output", record.payload)
        self.assertIn("visual_elements", record.payload["output"])

    def test_handover_payload_metadata_has_counts(self):
        record = self._run()
        meta = record.payload.get("metadata", {})
        for key in ("caption_count", "zoom_count", "duck_event_count"):
            self.assertIn(key, meta)

    def test_handover_cursor_end_advances(self):
        record = self._run()
        self.assertNotEqual(record.cursor_end, "00:00:00.000")

    def test_handover_output_path_ends_with_annotated_timeline(self):
        record = self._run()
        self.assertTrue(record.output_path.endswith("annotated_timeline.json"))

    # ── Failure path ──────────────────────────────────────────────────────

    def test_missing_cut_list_returns_failed_record(self):
        record = designer_run(
            job_id="job_no_cutter",
            staging_dir=self.staging_dir,
        )
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.skill, "designer")

    def test_failed_record_has_error_field(self):
        record = designer_run(
            job_id="job_no_cutter_2",
            staging_dir=self.staging_dir,
        )
        self.assertIsNotNone(record.error)
        self.assertGreater(len(record.error), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
