"""
src/utils/test_cutter.py

Unit tests for the Cutter skill.

Covers
------
  · detect_silence_from_transcript  — transcript-gap based silence detection
  · apply_jump_cuts                 — same-speaker jump cut flagging
  · run_cutter                      — end-to-end transcript path
  · CutSegment.to_dict              — serialisation contract
  · skills/cutter/main.run          — harness integration (SkillRecord)

All tests are mock-based (no audio files, no Whisper model needed).

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_cutter.py -v
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

from skills.cutter.logic import (
    CutSegment,
    CutterResult,
    JUMP_CUT_EFFECT,
    JUMP_CUT_ZOOM,
    MIN_CLIP_DURATION_S,
    SILENCE_MIN_DURATION_S,
    HEAD_TAIL_SILENCE_S,
    apply_jump_cuts,
    detect_silence_from_transcript,
    run_cutter,
)
from skills.cutter.main import run as cutter_run
import harness.memory.manager as mgr_mod


# ── Helpers ───────────────────────────────────────────────────────────────

def _w(start_ms: int, end_ms: int, speaker_id: str | None = None) -> dict:
    """Build a minimal word dict (as produced by TranscribeResult.words_as_dicts)."""
    d = {"word": "test", "start_ms": start_ms, "end_ms": end_ms, "confidence": 0.95}
    if speaker_id is not None:
        d["speaker_id"] = speaker_id
    return d


def _keep_segs(segments):
    return [s for s in segments if s.action == "keep"]


def _delete_segs(segments):
    return [s for s in segments if s.action == "delete"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestDetectSilenceFromTranscript
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDetectSilenceFromTranscript(unittest.TestCase):
    """detect_silence_from_transcript — gap detection and window coverage."""

    # ── Basic gap detection ───────────────────────────────────────────────

    def test_gap_above_threshold_creates_delete(self):
        """Gap of 1.0 s (> 0.5 s threshold) between words → delete segment."""
        words = [_w(500, 1000), _w(2000, 3000)]   # gap: 1.0 s
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=5.0)
        deletes = _delete_segs(segs)
        self.assertTrue(
            any(d.reason == "silence" for d in deletes),
            "Expected a 'silence' delete segment for 1.0 s gap",
        )

    def test_gap_exactly_at_threshold_creates_delete(self):
        """Gap exactly equal to threshold (0.5 s) → delete segment."""
        words = [_w(0, 1000), _w(1500, 2500)]   # gap: 0.5 s exactly
        segs = detect_silence_from_transcript(
            words, window_start_s=0.0, window_end_s=5.0, threshold_s=0.5
        )
        deletes = [d for d in _delete_segs(segs) if d.reason == "silence"]
        self.assertEqual(len(deletes), 1)

    def test_gap_below_threshold_no_delete(self):
        """Gap of 0.3 s (< 0.5 s threshold) → no silence delete segment."""
        words = [_w(0, 1000), _w(1300, 2000)]   # gap: 0.3 s
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=5.0)
        silence_deletes = [d for d in _delete_segs(segs) if d.reason == "silence"]
        self.assertEqual(len(silence_deletes), 0)

    def test_multiple_gaps_creates_multiple_deletes(self):
        """Three words with two qualifying gaps → two delete segments."""
        words = [_w(0, 500), _w(1500, 2000), _w(3500, 4000)]  # gaps: 1s, 1.5s
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=5.0)
        silence_deletes = [d for d in _delete_segs(segs) if d.reason == "silence"]
        self.assertEqual(len(silence_deletes), 2)

    def test_words_sorted_by_start_regardless_of_input_order(self):
        """Unsorted input words are handled correctly."""
        words = [_w(2000, 3000), _w(0, 500)]  # reversed order
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=5.0)
        keeps = _keep_segs(segs)
        self.assertGreater(len(keeps), 0)
        # All keep segments must contain at least one word
        self.assertTrue(keeps[0].start_s <= 0.5)  # first word at 0–0.5 s

    # ── Empty/edge cases ──────────────────────────────────────────────────

    def test_empty_words_returns_single_delete_for_full_window(self):
        """No words → entire window is a single silence delete."""
        segs = detect_silence_from_transcript([], window_start_s=0.0, window_end_s=10.0)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].action, "delete")
        self.assertAlmostEqual(segs[0].start_s, 0.0)
        self.assertAlmostEqual(segs[0].end_s, 10.0)

    def test_empty_words_empty_window_returns_empty(self):
        segs = detect_silence_from_transcript([], window_start_s=5.0, window_end_s=5.0)
        self.assertEqual(segs, [])

    def test_single_word_no_internal_gap(self):
        """One word in the middle → at most head + word + tail, no middle silence."""
        words = [_w(2000, 3000)]  # 1 s word in 0–10 s window
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=10.0)
        silence_mid = [d for d in _delete_segs(segs) if d.reason == "silence"]
        self.assertEqual(len(silence_mid), 0)

    # ── Head silence ──────────────────────────────────────────────────────

    def test_head_silence_above_threshold_is_deleted(self):
        """Head gap of 0.5 s (≥ 0.2 s head_tail_s) → delete("head_silence")."""
        words = [_w(500, 1500)]   # head gap: 0.5 s
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=5.0)
        head_deletes = [d for d in _delete_segs(segs) if d.reason == "head_silence"]
        self.assertEqual(len(head_deletes), 1)
        self.assertAlmostEqual(head_deletes[0].start_s, 0.0)
        self.assertAlmostEqual(head_deletes[0].end_s, 0.5)

    def test_head_silence_below_threshold_absorbed_into_keep(self):
        """Head gap of 0.1 s (< 0.2 s head_tail_s) → absorbed, first keep starts at 0."""
        words = [_w(100, 1000)]   # head gap: 0.1 s
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=5.0)
        head_deletes = [d for d in _delete_segs(segs) if d.reason == "head_silence"]
        self.assertEqual(len(head_deletes), 0)
        # First keep segment should start at window_start (0.0) since gap was absorbed
        keeps = _keep_segs(segs)
        self.assertGreater(len(keeps), 0)
        self.assertAlmostEqual(keeps[0].start_s, 0.0)

    # ── Tail silence ──────────────────────────────────────────────────────

    def test_tail_silence_above_threshold_is_deleted(self):
        """Tail gap of 2.0 s (≥ 0.2 s head_tail_s) → delete("tail_silence")."""
        words = [_w(0, 3000)]   # word ends at 3 s, window ends at 5 s
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=5.0)
        tail_deletes = [d for d in _delete_segs(segs) if d.reason == "tail_silence"]
        self.assertEqual(len(tail_deletes), 1)
        self.assertAlmostEqual(tail_deletes[0].start_s, 3.0)
        self.assertAlmostEqual(tail_deletes[0].end_s, 5.0)

    def test_tail_silence_below_threshold_absorbed_into_keep(self):
        """Tail gap of 0.1 s (< 0.2 s) → absorbed, last keep ends at window_end."""
        words = [_w(0, 4900)]  # ends at 4.9 s, window ends at 5.0 s → gap 0.1 s
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=5.0)
        tail_deletes = [d for d in _delete_segs(segs) if d.reason == "tail_silence"]
        self.assertEqual(len(tail_deletes), 0)
        keeps = _keep_segs(segs)
        self.assertGreater(len(keeps), 0)
        self.assertAlmostEqual(keeps[-1].end_s, 5.0)

    # ── Segment coverage ──────────────────────────────────────────────────

    def test_segments_cover_full_window_when_head_qualifies(self):
        """All segments together must span [window_start, window_end] when head silence qualifies."""
        words = [_w(1000, 2000), _w(4000, 5000)]
        window_start, window_end = 0.0, 7.0
        segs = detect_silence_from_transcript(
            words, window_start_s=window_start, window_end_s=window_end
        )
        self.assertAlmostEqual(segs[0].start_s, window_start, places=3)
        self.assertAlmostEqual(segs[-1].end_s, window_end, places=3)

    def test_segments_are_contiguous(self):
        """Adjacent segments must share their boundary (end_s[i] == start_s[i+1])."""
        words = [_w(1000, 2000), _w(4000, 5000)]
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=7.0)
        for i in range(len(segs) - 1):
            self.assertAlmostEqual(
                segs[i].end_s, segs[i + 1].start_s, places=3,
                msg=f"Gap between segment[{i}] and segment[{i+1}]",
            )

    def test_keep_segments_contain_all_word_midpoints(self):
        """Every word's midpoint must fall inside a keep segment."""
        words = [_w(500, 1500), _w(3000, 4000)]
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=6.0)
        keeps = _keep_segs(segs)
        for w in words:
            mid = (w["start_ms"] + w["end_ms"]) / 2000.0
            covered = any(k.start_s <= mid <= k.end_s for k in keeps)
            self.assertTrue(covered, f"Word midpoint {mid:.3f}s not in any keep segment")

    def test_segments_are_ordered_by_start(self):
        """Segments must be in ascending start_s order."""
        words = [_w(1000, 2000), _w(4000, 5000), _w(7000, 8000)]
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=10.0)
        for i in range(len(segs) - 1):
            self.assertLessEqual(segs[i].start_s, segs[i + 1].start_s)

    # ── Reason labels ─────────────────────────────────────────────────────

    def test_keep_segments_have_no_reason(self):
        words = [_w(0, 2000)]
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=3.0)
        for s in _keep_segs(segs):
            self.assertIsNone(s.reason)

    def test_delete_reason_labels_are_correct(self):
        """head_silence / silence / tail_silence reasons are correct."""
        words = [_w(1000, 2000), _w(4000, 5000)]
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=8.0)
        deletes = _delete_segs(segs)
        reasons = {d.reason for d in deletes}
        self.assertIn("head_silence", reasons)
        self.assertIn("silence", reasons)
        self.assertIn("tail_silence", reasons)

    # ── Speaker id ────────────────────────────────────────────────────────

    def test_speaker_id_propagated_to_keep_segment(self):
        """speaker_id from the first word of a keep block is stored on the segment."""
        words = [_w(0, 1000, speaker_id="spkA"), _w(500, 2000, speaker_id="spkA")]
        segs = detect_silence_from_transcript(words, window_start_s=0.0, window_end_s=3.0)
        keeps = _keep_segs(segs)
        self.assertTrue(any(k.speaker_id == "spkA" for k in keeps))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestApplyJumpCuts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestApplyJumpCuts(unittest.TestCase):
    """apply_jump_cuts — same-speaker consecutive keep detection."""

    def _pair(self, spk1, spk2) -> list[CutSegment]:
        """Two consecutive keeps separated by a delete."""
        return [
            CutSegment(start_s=0.0, end_s=2.0, action="keep", speaker_id=spk1),
            CutSegment(start_s=2.0, end_s=2.5, action="delete", reason="silence"),
            CutSegment(start_s=2.5, end_s=5.0, action="keep", speaker_id=spk2),
        ]

    def test_same_speaker_gets_zoom_on_second_keep(self):
        segs = apply_jump_cuts(self._pair("A", "A"))
        second_keep = _keep_segs(segs)[1]
        self.assertIn(JUMP_CUT_EFFECT, second_keep.effects)

    def test_first_keep_never_gets_zoom_effect(self):
        segs = apply_jump_cuts(self._pair("A", "A"))
        first_keep = _keep_segs(segs)[0]
        self.assertNotIn(JUMP_CUT_EFFECT, first_keep.effects)

    def test_different_speakers_no_zoom(self):
        segs = apply_jump_cuts(self._pair("A", "B"))
        for s in segs:
            self.assertNotIn(JUMP_CUT_EFFECT, s.effects)

    def test_none_speaker_id_no_zoom(self):
        segs = apply_jump_cuts(self._pair(None, None))
        for s in segs:
            self.assertNotIn(JUMP_CUT_EFFECT, s.effects)

    def test_one_none_speaker_id_no_zoom(self):
        segs = apply_jump_cuts(self._pair("A", None))
        for s in segs:
            self.assertNotIn(JUMP_CUT_EFFECT, s.effects)

    def test_three_consecutive_same_speaker_second_and_third_get_zoom(self):
        segs = [
            CutSegment(start_s=0.0, end_s=2.0, action="keep", speaker_id="A"),
            CutSegment(start_s=2.0, end_s=2.5, action="delete", reason="silence"),
            CutSegment(start_s=2.5, end_s=5.0, action="keep", speaker_id="A"),
            CutSegment(start_s=5.0, end_s=5.5, action="delete", reason="silence"),
            CutSegment(start_s=5.5, end_s=8.0, action="keep", speaker_id="A"),
        ]
        apply_jump_cuts(segs)
        keeps = _keep_segs(segs)
        self.assertNotIn(JUMP_CUT_EFFECT, keeps[0].effects)
        self.assertIn(JUMP_CUT_EFFECT, keeps[1].effects)
        self.assertIn(JUMP_CUT_EFFECT, keeps[2].effects)

    def test_speaker_change_breaks_chain(self):
        """A→B→A: B breaks the chain — third keep (A again) does NOT get zoom."""
        segs = [
            CutSegment(start_s=0.0, end_s=2.0, action="keep", speaker_id="A"),
            CutSegment(start_s=2.0, end_s=2.5, action="delete", reason="silence"),
            CutSegment(start_s=2.5, end_s=5.0, action="keep", speaker_id="B"),
            CutSegment(start_s=5.0, end_s=5.5, action="delete", reason="silence"),
            CutSegment(start_s=5.5, end_s=8.0, action="keep", speaker_id="A"),
        ]
        apply_jump_cuts(segs)
        keeps = _keep_segs(segs)
        # keeps[0]=A, keeps[1]=B, keeps[2]=A
        # A→B: different speaker, no zoom on keeps[1]
        # B→A: different speaker, no zoom on keeps[2]
        self.assertNotIn(JUMP_CUT_EFFECT, keeps[1].effects)
        self.assertNotIn(JUMP_CUT_EFFECT, keeps[2].effects)

    def test_broll_segment_breaks_jump_cut_chain(self):
        """A keep with is_broll=True is excluded from jump-cut pairing."""
        segs = [
            CutSegment(start_s=0.0, end_s=2.0, action="keep", speaker_id="A"),
            CutSegment(start_s=2.0, end_s=4.0, action="keep", speaker_id="A", is_broll=True),
            CutSegment(start_s=4.0, end_s=6.0, action="keep", speaker_id="A"),
        ]
        apply_jump_cuts(segs)
        # Only non-broll keeps are considered: segs[0] and segs[2]
        # segs[2] is the "second" in the pair → gets zoom
        self.assertNotIn(JUMP_CUT_EFFECT, segs[0].effects)
        self.assertIn(JUMP_CUT_EFFECT, segs[2].effects)
        # B-roll should not be modified
        self.assertNotIn(JUMP_CUT_EFFECT, segs[1].effects)

    def test_effect_not_duplicated_on_repeated_call(self):
        segs = self._pair("A", "A")
        apply_jump_cuts(segs)
        apply_jump_cuts(segs)
        second_keep = _keep_segs(segs)[1]
        self.assertEqual(second_keep.effects.count(JUMP_CUT_EFFECT), 1)

    def test_zoom_effect_string_contains_zoom_factor(self):
        self.assertIn(str(JUMP_CUT_ZOOM), JUMP_CUT_EFFECT)

    def test_zoom_effect_string_format(self):
        self.assertEqual(JUMP_CUT_EFFECT, "jump_cut_zoom_1.1")

    def test_non_keep_segments_never_get_effect(self):
        segs = [
            CutSegment(start_s=0.0, end_s=2.0, action="keep", speaker_id="A"),
            CutSegment(start_s=2.0, end_s=2.5, action="delete", reason="silence"),
            CutSegment(start_s=2.5, end_s=5.0, action="keep", speaker_id="A"),
        ]
        apply_jump_cuts(segs)
        for s in segs:
            if s.action == "delete":
                self.assertNotIn(JUMP_CUT_EFFECT, s.effects)

    def test_single_keep_no_zoom(self):
        segs = [CutSegment(start_s=0.0, end_s=5.0, action="keep", speaker_id="A")]
        apply_jump_cuts(segs)
        self.assertNotIn(JUMP_CUT_EFFECT, segs[0].effects)

    def test_empty_segments_no_error(self):
        result = apply_jump_cuts([])
        self.assertEqual(result, [])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestRunCutter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunCutter(unittest.TestCase):
    """run_cutter — end-to-end transcript-based pipeline."""

    def _words(self):
        """Two speech bursts separated by 1.5 s silence, same speaker."""
        return [
            _w(500, 2000, speaker_id="spkA"),
            _w(2100, 3500, speaker_id="spkA"),  # gap 0.1 s → merged
            _w(5000, 7000, speaker_id="spkA"),  # gap 1.5 s → silence
        ]

    def test_returns_cutter_result(self):
        result = run_cutter(self._words(), window_end_s=10.0)
        self.assertIsInstance(result, CutterResult)

    def test_result_has_segments(self):
        result = run_cutter(self._words(), window_end_s=10.0)
        self.assertIsInstance(result.segments, list)
        self.assertGreater(len(result.segments), 0)

    def test_silence_gap_produces_delete_segment(self):
        result = run_cutter(self._words(), window_end_s=10.0)
        deletes = _delete_segs(result.segments)
        silence = [d for d in deletes if d.reason == "silence"]
        self.assertGreater(len(silence), 0)

    def test_jump_cut_applied_when_same_speaker(self):
        result = run_cutter(self._words(), window_end_s=10.0)
        keeps = _keep_segs(result.segments)
        jump_cut_keeps = [k for k in keeps if JUMP_CUT_EFFECT in k.effects]
        self.assertGreater(len(jump_cut_keeps), 0)

    def test_no_jump_cut_when_different_speakers(self):
        words = [
            _w(0, 2000, speaker_id="spkA"),
            _w(3000, 5000, speaker_id="spkB"),
        ]
        result = run_cutter(words, window_end_s=6.0)
        for s in result.segments:
            self.assertNotIn(JUMP_CUT_EFFECT, s.effects)

    def test_sensor_flags_for_short_clip(self):
        """A keep segment shorter than MIN_CLIP_DURATION_S triggers a sensor flag."""
        words = [
            _w(0, 200),    # 0.2 s keep — shorter than 0.5 s minimum
            _w(1000, 2000),
        ]
        result = run_cutter(words, window_end_s=3.0)
        short_flags = [f for f in result.sensor_flags if "DURATION" in f]
        self.assertGreater(len(short_flags), 0)

    def test_empty_words_returns_single_delete(self):
        result = run_cutter([], window_end_s=5.0)
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].action, "delete")

    def test_sensor_flags_is_list(self):
        result = run_cutter(self._words(), window_end_s=10.0)
        self.assertIsInstance(result.sensor_flags, list)

    def test_window_start_honoured(self):
        """Segments must start at window_start_s, not at 0."""
        words = [_w(15000, 17000)]  # 15–17 s
        result = run_cutter(words, window_start_s=10.0, window_end_s=20.0)
        self.assertAlmostEqual(result.segments[0].start_s, 10.0, places=2)

    def test_window_end_honoured(self):
        """Segments must end at window_end_s."""
        words = [_w(0, 3000)]
        result = run_cutter(words, window_start_s=0.0, window_end_s=5.0)
        self.assertAlmostEqual(result.segments[-1].end_s, 5.0, places=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestCutSegmentToDict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCutSegmentToDict(unittest.TestCase):
    """CutSegment.to_dict — serialisation contract."""

    def test_keep_segment_dict_has_required_keys(self):
        seg = CutSegment(start_s=1.0, end_s=3.0, action="keep")
        d = seg.to_dict()
        for key in ("start", "end", "action", "effects"):
            self.assertIn(key, d)

    def test_keep_segment_no_reason_key(self):
        seg = CutSegment(start_s=0.0, end_s=2.0, action="keep")
        d = seg.to_dict()
        self.assertNotIn("reason", d)

    def test_delete_segment_has_reason_key(self):
        seg = CutSegment(start_s=2.0, end_s=3.0, action="delete", reason="silence")
        d = seg.to_dict()
        self.assertIn("reason", d)
        self.assertEqual(d["reason"], "silence")

    def test_effects_list_in_dict(self):
        seg = CutSegment(
            start_s=3.0, end_s=5.0, action="keep",
            effects=["jump_cut_zoom_1.1"],
        )
        d = seg.to_dict()
        self.assertIn("jump_cut_zoom_1.1", d["effects"])

    def test_start_end_are_rounded(self):
        seg = CutSegment(start_s=1.0001, end_s=3.9999, action="keep")
        d = seg.to_dict()
        self.assertIsInstance(d["start"], float)
        self.assertIsInstance(d["end"], float)

    def test_action_values(self):
        for action in ("keep", "delete"):
            seg = CutSegment(start_s=0.0, end_s=1.0, action=action)
            self.assertEqual(seg.to_dict()["action"], action)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TestCutterMain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCutterMain(unittest.TestCase):
    """skills/cutter/main.run — harness integration."""

    def _make_transcript(self, tmp_path: Path) -> Path:
        """Write a minimal transcript.json to tmp_path and return its path."""
        transcript = {
            "full_text": "Hello world this is a test",
            "words": [
                {"word": "Hello", "start_ms": 500, "end_ms": 1000, "confidence": 0.9, "speaker_id": "A"},
                {"word": "world", "start_ms": 1100, "end_ms": 1600, "confidence": 0.9, "speaker_id": "A"},
                # 2 s silence gap
                {"word": "this", "start_ms": 3700, "end_ms": 4100, "confidence": 0.9, "speaker_id": "A"},
                {"word": "is", "start_ms": 4200, "end_ms": 4500, "confidence": 0.9, "speaker_id": "A"},
                {"word": "a", "start_ms": 4600, "end_ms": 4800, "confidence": 0.9, "speaker_id": "A"},
                {"word": "test", "start_ms": 4900, "end_ms": 5400, "confidence": 0.9, "speaker_id": "A"},
            ],
            "vad_segments": [],
            "metadata": {
                "model": "base",
                "language": "en",
                "duration_s": 6.0,
                "word_count": 6,
                "avg_confidence": 0.9,
            },
        }
        t_path = tmp_path / "transcript.json"
        t_path.write_text(json.dumps(transcript))
        return t_path

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.staging_dir = self.tmp_path / "staging" / "job_cutter_001"
        self.jobs_root = self.tmp_path / "jobs"
        self._patcher = patch.object(mgr_mod, "JOBS_ROOT", self.jobs_root)
        self._patcher.start()
        self.transcript_path = self._make_transcript(self.tmp_path)

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def _run(self):
        return cutter_run(
            job_id="job_cutter_001",
            staging_dir=self.staging_dir,
            transcript_path=str(self.transcript_path),
            total_duration="00:00:06.000",
        )

    # ── Status ────────────────────────────────────────────────────────────

    def test_record_status_is_success(self):
        record = self._run()
        self.assertEqual(record.status, "success",
                         f"Expected success, got: {record.error}")

    def test_record_skill_name_is_cutter(self):
        record = self._run()
        self.assertEqual(record.skill, "cutter")

    # ── Output file ───────────────────────────────────────────────────────

    def test_cut_list_json_exists(self):
        self._run()
        self.assertTrue((self.staging_dir / "cut_list.json").exists())

    def test_cut_list_json_has_cut_segments_key(self):
        self._run()
        raw = json.loads((self.staging_dir / "cut_list.json").read_text())
        self.assertIn("cut_segments", raw)

    def test_cut_list_segments_have_required_fields(self):
        self._run()
        raw = json.loads((self.staging_dir / "cut_list.json").read_text())
        for seg in raw["cut_segments"]:
            for key in ("start", "end", "action", "effects"):
                self.assertIn(key, seg, f"Missing '{key}' in segment {seg}")

    def test_cut_list_has_delete_segment_for_long_gap(self):
        """The 2 s silence gap between 'world' and 'this' must produce a delete."""
        self._run()
        raw = json.loads((self.staging_dir / "cut_list.json").read_text())
        deletes = [s for s in raw["cut_segments"] if s["action"] == "delete"]
        self.assertGreater(len(deletes), 0)

    def test_cut_list_has_jump_cut_effect_for_same_speaker(self):
        """Same-speaker consecutive keeps → 'jump_cut_zoom_1.1' in effects."""
        self._run()
        raw = json.loads((self.staging_dir / "cut_list.json").read_text())
        effects_present = any(
            "jump_cut_zoom_1.1" in seg.get("effects", [])
            for seg in raw["cut_segments"]
        )
        self.assertTrue(effects_present)

    # ── Handover record ───────────────────────────────────────────────────

    def test_handover_record_written_to_memory(self):
        self._run()
        records = list((self.jobs_root / "job_cutter_001").glob("*.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].name, "002_cutter.json")

    def test_handover_output_path_ends_with_cut_list_json(self):
        record = self._run()
        self.assertTrue(record.output_path.endswith("cut_list.json"))

    def test_handover_payload_has_output_key(self):
        record = self._run()
        self.assertIn("output", record.payload)
        self.assertIn("cut_segments", record.payload["output"])

    def test_handover_payload_has_metadata_key(self):
        record = self._run()
        meta = record.payload.get("metadata", {})
        for key in ("keep_count", "delete_count", "jump_cut_count"):
            self.assertIn(key, meta)

    def test_handover_cursor_end_advances(self):
        record = self._run()
        self.assertNotEqual(record.cursor_end, "00:00:00.000")

    # ── Failure path ──────────────────────────────────────────────────────

    def test_missing_transcript_path_returns_failed_record(self):
        """No transcript_path and no transcriber record in memory → failed."""
        record = cutter_run(
            job_id="job_no_transcript",
            staging_dir=self.staging_dir,
            # transcript_path not provided, no transcriber record in memory
        )
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.skill, "cutter")

    def test_failed_record_has_error_field(self):
        record = cutter_run(
            job_id="job_no_transcript_2",
            staging_dir=self.staging_dir,
        )
        self.assertIsNotNone(record.error)
        self.assertGreater(len(record.error), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
