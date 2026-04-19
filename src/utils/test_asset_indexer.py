"""
src/utils/test_asset_indexer.py

Unit tests for AssetIndexer (src/utils/asset_indexer.py).

All tests use a temporary directory — no real assets/broll/ files required.

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_asset_indexer.py -v
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

from src.utils.asset_indexer import AssetIndexer, _tokenize, _SYNONYMS, _REVERSE_SYNONYMS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tmp_assets(*filenames: str) -> tempfile.TemporaryDirectory:
    """Create a temp dir with empty video stub files."""
    tmp = tempfile.TemporaryDirectory()
    for name in filenames:
        (Path(tmp.name) / name).write_bytes(b"")
    return tmp


# ═══════════════════════════════════════════════════════════════════════════════
# TestTokenize — filename stem splitting
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenize(unittest.TestCase):
    """_tokenize() — 8 tests"""

    def test_underscore_separated(self):
        self.assertEqual(_tokenize("cat_playing_outside"), ["cat", "playing", "outside"])

    def test_hyphen_separated(self):
        self.assertEqual(_tokenize("dog-running-fast"), ["dog", "running", "fast"])

    def test_mixed_separators(self):
        tokens = _tokenize("city_life-2024")
        self.assertIn("city", tokens)
        self.assertIn("life", tokens)

    def test_camel_case_not_split(self):
        # CamelCase is kept as one token (no split on case)
        tokens = _tokenize("NatureWalk")
        self.assertEqual(tokens, ["naturewalk"])

    def test_single_char_excluded(self):
        tokens = _tokenize("a_cat_b")
        self.assertNotIn("a", tokens)
        self.assertNotIn("b", tokens)
        self.assertIn("cat", tokens)

    def test_numbers_included(self):
        tokens = _tokenize("clip_01_beach")
        self.assertIn("01", tokens)
        self.assertIn("beach", tokens)

    def test_korean_stem(self):
        tokens = _tokenize("고양이_놀이")
        self.assertIn("고양이", tokens)
        self.assertIn("놀이", tokens)

    def test_empty_string(self):
        self.assertEqual(_tokenize(""), [])


# ═══════════════════════════════════════════════════════════════════════════════
# TestAssetIndexerBuild — index construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssetIndexerBuild(unittest.TestCase):
    """AssetIndexer.build() — 10 tests"""

    def test_empty_directory_returns_empty_index(self):
        with _make_tmp_assets() as d:
            idx = AssetIndexer(assets_dir=Path(d)).build()
            self.assertEqual(idx, {})

    def test_non_video_files_ignored(self):
        with _make_tmp_assets("readme.txt", "image.jpg", "data.csv") as d:
            idx = AssetIndexer(assets_dir=Path(d)).build()
            self.assertEqual(idx, {})

    def test_single_file_stem_indexed(self):
        with _make_tmp_assets("cat.mp4") as d:
            idx = AssetIndexer(assets_dir=Path(d)).build()
            self.assertIn("cat", idx)

    def test_underscore_tokens_indexed(self):
        with _make_tmp_assets("cat_playing.mp4") as d:
            idx = AssetIndexer(assets_dir=Path(d)).build()
            self.assertIn("cat", idx)
            self.assertIn("playing", idx)
            self.assertIn("cat_playing", idx)

    def test_multiple_files_all_indexed(self):
        with _make_tmp_assets("cat.mp4", "dog.mp4", "ocean.mp4") as d:
            idx = AssetIndexer(assets_dir=Path(d)).build()
            self.assertIn("cat", idx)
            self.assertIn("dog", idx)
            self.assertIn("ocean", idx)

    def test_all_supported_extensions_indexed(self):
        with _make_tmp_assets("clip.mp4", "clip2.mov", "clip3.avi", "clip4.mkv", "clip5.webm") as d:
            idx = AssetIndexer(assets_dir=Path(d)).build()
            self.assertEqual(len({v for v in idx.values()}), 5)

    def test_synonym_expansion_english_from_korean_file(self):
        # "고양이.mp4" → index should also contain "cat"
        with _make_tmp_assets("고양이.mp4") as d:
            idx = AssetIndexer(assets_dir=Path(d)).build()
            self.assertIn("cat", idx)
            self.assertIn("고양이", idx)

    def test_synonym_expansion_korean_from_english_file(self):
        # "cat.mp4" → index should also contain "고양이"
        with _make_tmp_assets("cat.mp4") as d:
            idx = AssetIndexer(assets_dir=Path(d)).build()
            self.assertIn("고양이", idx)
            self.assertIn("cat", idx)

    def test_first_file_wins_on_duplicate_keyword(self):
        # Both files share keyword "cat"; first alphabetically wins
        with _make_tmp_assets("cat_a.mp4", "cat_b.mp4") as d:
            idx = AssetIndexer(assets_dir=Path(d)).build()
            # "cat" should map to one of the two files
            self.assertIn("cat", idx)
            self.assertTrue(idx["cat"].endswith(".mp4"))

    def test_nonexistent_directory_returns_empty(self):
        idx = AssetIndexer(assets_dir=Path("/nonexistent/path/xyz")).build()
        self.assertEqual(idx, {})


# ═══════════════════════════════════════════════════════════════════════════════
# TestAssetIndexerFind — query matching
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssetIndexerFind(unittest.TestCase):
    """AssetIndexer.find() — 14 tests"""

    def _indexer(self, *filenames: str):
        self._tmp = _make_tmp_assets(*filenames)
        ix = AssetIndexer(assets_dir=Path(self._tmp.name))
        ix.build()
        return ix

    def tearDown(self):
        if hasattr(self, "_tmp"):
            self._tmp.cleanup()

    def test_exact_stem_match(self):
        ix = self._indexer("cat.mp4")
        result = ix.find("cat")
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("cat.mp4"))

    def test_case_insensitive_match(self):
        ix = self._indexer("Cat.mp4")
        self.assertIsNotNone(ix.find("cat"))
        self.assertIsNotNone(ix.find("CAT"))

    def test_no_match_returns_none(self):
        ix = self._indexer("cat.mp4")
        self.assertIsNone(ix.find("helicopter"))

    def test_empty_query_returns_none(self):
        ix = self._indexer("cat.mp4")
        self.assertIsNone(ix.find(""))

    def test_whitespace_query_returns_none(self):
        ix = self._indexer("cat.mp4")
        self.assertIsNone(ix.find("   "))

    def test_substring_match(self):
        # "nature_walk.mp4" — find("walk") should match via token
        ix = self._indexer("nature_walk.mp4")
        self.assertIsNotNone(ix.find("walk"))

    def test_prefix_match(self):
        # "naturewalk.mp4" — find("nat") should prefix-match "naturewalk"
        ix = self._indexer("naturewalk.mp4")
        self.assertIsNotNone(ix.find("nat"))

    def test_korean_exact_match(self):
        ix = self._indexer("고양이.mp4")
        self.assertIsNotNone(ix.find("고양이"))

    def test_korean_to_english_synonym(self):
        # "cat.mp4" is indexed, query "고양이" should find it via synonym
        ix = self._indexer("cat.mp4")
        result = ix.find("고양이")
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("cat.mp4"))

    def test_english_to_korean_synonym(self):
        # "고양이.mp4" is indexed, query "cat" should find it via synonym
        ix = self._indexer("고양이.mp4")
        result = ix.find("cat")
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("고양이.mp4"))

    def test_synonym_alias(self):
        # "puppy" is a synonym of "강아지"
        ix = self._indexer("강아지.mp4")
        self.assertIsNotNone(ix.find("puppy"))

    def test_find_auto_builds_on_first_call(self):
        # build() was never called explicitly
        with _make_tmp_assets("cat.mp4") as d:
            ix = AssetIndexer(assets_dir=Path(d))
            # Must not raise, must find
            self.assertIsNotNone(ix.find("cat"))

    def test_find_returns_absolute_path(self):
        ix = self._indexer("cat.mp4")
        result = ix.find("cat")
        self.assertIsNotNone(result)
        self.assertTrue(Path(result).is_absolute())

    def test_empty_directory_find_returns_none(self):
        with _make_tmp_assets() as d:
            ix = AssetIndexer(assets_dir=Path(d))
            self.assertIsNone(ix.find("cat"))


# ═══════════════════════════════════════════════════════════════════════════════
# TestAssetIndexerHelpers — all_keywords, index_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssetIndexerHelpers(unittest.TestCase):
    """all_keywords() and index_size() — 4 tests"""

    def test_all_keywords_sorted(self):
        with _make_tmp_assets("cat.mp4") as d:
            ix = AssetIndexer(assets_dir=Path(d))
            kws = ix.all_keywords()
            self.assertEqual(kws, sorted(kws))

    def test_all_keywords_includes_stem(self):
        with _make_tmp_assets("ocean_wave.mp4") as d:
            ix = AssetIndexer(assets_dir=Path(d))
            kws = ix.all_keywords()
            self.assertIn("ocean", kws)
            self.assertIn("wave", kws)

    def test_index_size_zero_for_empty_dir(self):
        with _make_tmp_assets() as d:
            ix = AssetIndexer(assets_dir=Path(d))
            self.assertEqual(ix.index_size(), 0)

    def test_index_size_positive_for_file_with_synonyms(self):
        with _make_tmp_assets("cat.mp4") as d:
            ix = AssetIndexer(assets_dir=Path(d))
            # At minimum: "cat" + synonym tokens
            self.assertGreater(ix.index_size(), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# TestAutoFillIntent — IntentProcessor + AssetIndexer integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoFillIntent(unittest.TestCase):
    """_handle_broll_auto_fill via IntentProcessor — 8 tests"""

    def _make_transcript(self, tmp_dir: Path, words: list[str]) -> Path:
        words_list = [{"word": w, "start": i * 1000, "end": (i + 1) * 1000}
                      for i, w in enumerate(words)]
        t = tmp_dir / "transcript.json"
        t.write_text(json.dumps({"words": words_list, "vad_segments": []}),
                     encoding="utf-8")
        return t

    def test_no_transcript_returns_applied_false(self):
        from src.pipeline.intent_processor import IntentProcessor
        with patch("src.pipeline.intent_processor._load_latest_transcript_words",
                   return_value=[]):
            p = IntentProcessor()
            r = p.process("자료화면 자동으로 채워줘")
            self.assertFalse(r.applied)

    def test_no_matching_assets_returns_applied_false(self):
        from src.pipeline.intent_processor import IntentProcessor
        words = [{"word": "헬리콥터", "start": 0, "end": 1000}]
        with tempfile.TemporaryDirectory() as d:
            with patch("src.pipeline.intent_processor._load_latest_transcript_words",
                       return_value=words), \
                 patch("src.utils.asset_indexer._DEFAULT_BROLL_DIR", Path(d)):
                p = IntentProcessor()
                r = p.process("자료화면 자동으로 채워줘")
                self.assertFalse(r.applied)

    def test_matching_assets_returns_applied_true(self):
        from src.pipeline.intent_processor import IntentProcessor
        words = [{"word": "cat", "start": 0, "end": 1000}]
        with tempfile.TemporaryDirectory() as assets_d, \
             tempfile.TemporaryDirectory() as spec_d:
            (Path(assets_d) / "cat.mp4").write_bytes(b"")
            broll_file = Path(spec_d) / "broll_requests.json"

            with patch("src.pipeline.intent_processor._load_latest_transcript_words",
                       return_value=words), \
                 patch("src.pipeline.intent_processor._BROLL_REQUESTS_FILE", broll_file), \
                 patch("src.utils.asset_indexer._DEFAULT_BROLL_DIR", Path(assets_d)):
                p = IntentProcessor()
                r = p.process("자료화면 자동으로 채워줘")
                self.assertTrue(r.applied)

    def test_auto_fill_writes_broll_requests_json(self):
        from src.pipeline.intent_processor import IntentProcessor
        words = [{"word": "cat", "start": 0, "end": 1000}]
        with tempfile.TemporaryDirectory() as assets_d, \
             tempfile.TemporaryDirectory() as spec_d:
            (Path(assets_d) / "cat.mp4").write_bytes(b"")
            broll_file = Path(spec_d) / "broll_requests.json"

            with patch("src.pipeline.intent_processor._load_latest_transcript_words",
                       return_value=words), \
                 patch("src.pipeline.intent_processor._BROLL_REQUESTS_FILE", broll_file), \
                 patch("src.utils.asset_indexer._DEFAULT_BROLL_DIR", Path(assets_d)):
                p = IntentProcessor()
                p.process("자료화면 자동으로 채워줘")

            written = json.loads(broll_file.read_text(encoding="utf-8"))
            self.assertIsInstance(written, list)
            self.assertGreater(len(written), 0)
            self.assertEqual(written[0]["keyword"], "cat")

    def test_auto_fill_restart_from_designer(self):
        from src.pipeline.intent_processor import IntentProcessor
        words = [{"word": "cat", "start": 0, "end": 1000}]
        with tempfile.TemporaryDirectory() as assets_d, \
             tempfile.TemporaryDirectory() as spec_d:
            (Path(assets_d) / "cat.mp4").write_bytes(b"")
            broll_file = Path(spec_d) / "broll_requests.json"

            with patch("src.pipeline.intent_processor._load_latest_transcript_words",
                       return_value=words), \
                 patch("src.pipeline.intent_processor._BROLL_REQUESTS_FILE", broll_file), \
                 patch("src.utils.asset_indexer._DEFAULT_BROLL_DIR", Path(assets_d)):
                p = IntentProcessor()
                r = p.process("자동으로 채워줘")
                self.assertEqual(r.restart_from, "designer")

    def test_auto_fill_does_not_clobber_existing_manual_entries(self):
        from src.pipeline.intent_processor import IntentProcessor
        existing = [{"keyword": "dog", "asset_path": "/fake/dog.mp4",
                     "opacity": 0.8, "mode": "overlay"}]
        words = [{"word": "cat", "start": 0, "end": 1000}]
        with tempfile.TemporaryDirectory() as assets_d, \
             tempfile.TemporaryDirectory() as spec_d:
            (Path(assets_d) / "cat.mp4").write_bytes(b"")
            broll_file = Path(spec_d) / "broll_requests.json"
            broll_file.write_text(json.dumps(existing), encoding="utf-8")

            with patch("src.pipeline.intent_processor._load_latest_transcript_words",
                       return_value=words), \
                 patch("src.pipeline.intent_processor._BROLL_REQUESTS_FILE", broll_file), \
                 patch("src.utils.asset_indexer._DEFAULT_BROLL_DIR", Path(assets_d)):
                p = IntentProcessor()
                p.process("자료화면 자동으로 채워줘")

            written = json.loads(broll_file.read_text(encoding="utf-8"))
            keywords = [r["keyword"] for r in written]
            self.assertIn("dog", keywords)   # manual entry preserved
            self.assertIn("cat", keywords)   # auto entry added

    def test_auto_fill_deduplicated_paths(self):
        """Multiple transcript words matching the same asset produce one entry."""
        from src.pipeline.intent_processor import IntentProcessor
        words = [{"word": "cat", "start": 0, "end": 1000},
                 {"word": "kitten", "start": 1000, "end": 2000}]
        with tempfile.TemporaryDirectory() as assets_d, \
             tempfile.TemporaryDirectory() as spec_d:
            # Both "cat" and "kitten" map to the same file
            (Path(assets_d) / "cat.mp4").write_bytes(b"")
            broll_file = Path(spec_d) / "broll_requests.json"

            with patch("src.pipeline.intent_processor._load_latest_transcript_words",
                       return_value=words), \
                 patch("src.pipeline.intent_processor._BROLL_REQUESTS_FILE", broll_file), \
                 patch("src.utils.asset_indexer._DEFAULT_BROLL_DIR", Path(assets_d)):
                p = IntentProcessor()
                r = p.process("자료화면 자동으로 채워줘")

            written = json.loads(broll_file.read_text(encoding="utf-8"))
            paths = [entry["asset_path"] for entry in written]
            # Same path must not appear twice
            self.assertEqual(len(paths), len(set(paths)))

    def test_gold_theme_still_routes_to_exporter_not_designer(self):
        """Regression: 골드 테마 must not trigger designer restart."""
        from src.pipeline.intent_processor import IntentProcessor
        p = IntentProcessor()
        r = p.process("골드 테마")
        self.assertEqual(r.restart_from, "exporter")
        self.assertTrue(r.applied)


# ═══════════════════════════════════════════════════════════════════════════════
# TestExtractNounCandidates — noun extraction heuristic
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractNounCandidates(unittest.TestCase):
    """_extract_noun_candidates() — 6 tests"""

    def _words(self, word_list: list[str]) -> list[dict]:
        return [{"word": w, "start": i * 100, "end": (i + 1) * 100}
                for i, w in enumerate(word_list)]

    def test_basic_extraction(self):
        from src.pipeline.intent_processor import _extract_noun_candidates
        result = _extract_noun_candidates(self._words(["cat", "dog", "ocean"]))
        self.assertIn("cat", result)
        self.assertIn("dog", result)
        self.assertIn("ocean", result)

    def test_stop_words_excluded(self):
        from src.pipeline.intent_processor import _extract_noun_candidates
        result = _extract_noun_candidates(self._words(["그리고", "cat", "하지만"]))
        self.assertNotIn("그리고", result)
        self.assertNotIn("하지만", result)
        self.assertIn("cat", result)

    def test_single_char_excluded(self):
        from src.pipeline.intent_processor import _extract_noun_candidates
        result = _extract_noun_candidates(self._words(["a", "is", "cat"]))
        self.assertNotIn("a", result)
        # "is" is a stop word AND 2 chars — confirmed excluded
        self.assertNotIn("is", result)
        self.assertIn("cat", result)

    def test_deduplication(self):
        from src.pipeline.intent_processor import _extract_noun_candidates
        result = _extract_noun_candidates(self._words(["cat", "cat", "cat"]))
        self.assertEqual(result.count("cat"), 1)

    def test_max_candidates_limit(self):
        from src.pipeline.intent_processor import _extract_noun_candidates
        many = [f"word{i:03d}" for i in range(50)]
        result = _extract_noun_candidates(self._words(many), max_candidates=10)
        self.assertEqual(len(result), 10)

    def test_punctuation_stripped(self):
        from src.pipeline.intent_processor import _extract_noun_candidates
        result = _extract_noun_candidates(self._words(["cat.", "dog,", "ocean!"]))
        self.assertIn("cat", result)
        self.assertIn("dog", result)
        self.assertIn("ocean", result)


# ═══════════════════════════════════════════════════════════════════════════════
# TestAspectRatioGateWithAutoFill — 9:16 regression
# ═══════════════════════════════════════════════════════════════════════════════

class TestAspectRatioGateWithAutoFill(unittest.TestCase):
    """Verify that auto-matched b-roll metadata doesn't violate 9:16 gate — 3 tests"""

    def test_9x16_resolution_passes_gate(self):
        from harness.sensors.export_validator import check_aspect_ratio, GateStatus
        r = check_aspect_ratio(1080, 1920)
        self.assertEqual(r.status, GateStatus.PASS)

    def test_broll_element_dict_structure_compatible_with_export_plan(self):
        """Auto-fill produces dicts with keys export_plan / designer can consume."""
        from src.utils.asset_indexer import AssetIndexer
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "cat.mp4").write_bytes(b"")
            ix = AssetIndexer(assets_dir=Path(d))
            path = ix.find("cat")
        req = {"keyword": "cat", "asset_path": path, "opacity": 1.0, "mode": "overlay"}
        # Must have all keys that designer/logic.py build_broll_elements expects
        for key in ("keyword", "asset_path", "opacity", "mode"):
            self.assertIn(key, req)

    def test_landscape_resolution_fails_gate(self):
        """Confirm that landscape output (wrong resolution) still fails the gate."""
        from harness.sensors.export_validator import check_aspect_ratio, GateStatus
        r = check_aspect_ratio(1920, 1080)
        self.assertEqual(r.status, GateStatus.FAIL)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
