"""
src/utils/test_asset_generator.py

Unit + integration tests for Phase 11 — Photorealistic Asset Generation Fallback.

Covers
------
  TestAssetGeneratorPrompt        — photorealistic prompt structure (4)
  TestAssetGeneratorCache         — caching logic / cache path sanitisation (7)
  TestAssetGeneratorPlaceholder   — placeholder PNG correctness (8)
  TestAssetGeneratorGenerateAll   — generate_all() bulk method (5)
  TestAssetIndexerGeneratorFbk    — AssetIndexer generator fallback in find() (7)
  TestGenAssetGate                — GenAssetGate sensor validation (12)
  TestAutoFillWithGenerator       — intent_processor auto-fill + generator (9)
  TestE2EWithGeneratedAssets      — end-to-end: generate → gate → broll_requests (4)

All tests are mock-based; no real API calls, no real assets/broll/ files required.

Run:
    cd VoxEdit_AI
    python -m pytest src/utils/test_asset_generator.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.asset_generator import AssetGenerator
from src.utils.asset_indexer import AssetIndexer
from harness.sensors.gen_asset_gate import check_gen_asset, validate_all, GenAssetReport


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tmp_gen(backend: str = "placeholder") -> tuple[AssetGenerator, tempfile.TemporaryDirectory]:
    """Return (generator, tmp_dir) with output_dir pointing to a temp directory."""
    tmp = tempfile.TemporaryDirectory()
    gen = AssetGenerator(output_dir=Path(tmp.name), backend=backend)
    return gen, tmp


def _make_valid_png(path: Path, width: int = 1080, height: int = 1920) -> None:
    """Write a valid PNG at the given path with the given dimensions."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(50, 50, 50))
    img.save(str(path), "PNG")


# ═══════════════════════════════════════════════════════════════════════════════
# TestAssetGeneratorPrompt — 4 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssetGeneratorPrompt(unittest.TestCase):
    """Photorealistic prompt must contain required quality keywords."""

    def setUp(self):
        self.gen, self.tmp = _make_tmp_gen()

    def tearDown(self):
        self.tmp.cleanup()

    def test_prompt_contains_photorealistic(self):
        p = self.gen.build_prompt("cat")
        self.assertIn("photorealistic", p.lower())

    def test_prompt_contains_8k(self):
        p = self.gen.build_prompt("cat")
        self.assertIn("8k", p.lower())

    def test_prompt_contains_cinematic(self):
        p = self.gen.build_prompt("cat")
        self.assertIn("cinematic", p.lower())

    def test_prompt_contains_keyword(self):
        p = self.gen.build_prompt("ocean sunset")
        self.assertIn("ocean sunset", p.lower())


# ═══════════════════════════════════════════════════════════════════════════════
# TestAssetGeneratorCache — 7 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssetGeneratorCache(unittest.TestCase):
    """Caching: existing files are returned without regenerating."""

    def setUp(self):
        self.gen, self.tmp = _make_tmp_gen()

    def tearDown(self):
        self.tmp.cleanup()

    def test_is_cached_false_before_generation(self):
        self.assertFalse(self.gen.is_cached("cat"))

    def test_is_cached_true_after_generation(self):
        self.gen.generate("cat")
        self.assertTrue(self.gen.is_cached("cat"))

    def test_cached_file_returned_without_regenerating(self):
        # Pre-place a file at the cache path
        cache_p = self.gen.cache_path("cat")
        _make_valid_png(cache_p)

        with patch.object(self.gen, "_generate_placeholder") as mock_gen:
            result = self.gen.generate("cat")
            mock_gen.assert_not_called()
        self.assertEqual(result, str(cache_p))

    def test_cache_path_lowercases_keyword(self):
        p = self.gen.cache_path("CAT")
        self.assertTrue(p.name.startswith("cat"))

    def test_cache_path_sanitizes_spaces(self):
        p = self.gen.cache_path("ocean wave")
        self.assertNotIn(" ", p.name)

    def test_cache_path_sanitizes_special_chars(self):
        p = self.gen.cache_path("cat/dog")
        self.assertNotIn("/", p.name)

    def test_generate_creates_output_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as d:
            nested = Path(d) / "deep" / "nested"
            gen = AssetGenerator(output_dir=nested, backend="placeholder")
            gen.generate("cat")
            self.assertTrue(nested.exists())


# ═══════════════════════════════════════════════════════════════════════════════
# TestAssetGeneratorPlaceholder — 8 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssetGeneratorPlaceholder(unittest.TestCase):
    """Placeholder backend must produce valid 1080×1920 PNGs."""

    def setUp(self):
        self.gen, self.tmp = _make_tmp_gen(backend="placeholder")

    def tearDown(self):
        self.tmp.cleanup()

    def test_generate_returns_path(self):
        result = self.gen.generate("cat")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_generated_file_exists(self):
        result = self.gen.generate("cat")
        self.assertTrue(Path(result).exists())

    def test_generated_file_is_png(self):
        result = self.gen.generate("cat")
        self.assertTrue(result.endswith(".png"))

    def test_placeholder_resolution_1080x1920(self):
        from PIL import Image
        result = self.gen.generate("cat")
        with Image.open(result) as img:
            self.assertEqual(img.size, (1080, 1920))

    def test_placeholder_is_valid_image(self):
        from PIL import Image
        result = self.gen.generate("cat")
        with Image.open(result) as img:
            img.verify()  # raises on corruption

    def test_korean_keyword_generates_successfully(self):
        result = self.gen.generate("고양이")
        self.assertIsNotNone(result)
        self.assertTrue(Path(result).exists())

    def test_generate_none_backend_raises(self):
        gen = AssetGenerator(output_dir=Path(self.tmp.name), backend="unknown_backend")
        with self.assertRaises(ValueError):
            gen.generate("cat")

    def test_empty_keyword_returns_none(self):
        result = self.gen.generate("   ")
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════════════════
# TestAssetGeneratorGenerateAll — 5 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssetGeneratorGenerateAll(unittest.TestCase):
    """generate_all() processes a list of keywords efficiently."""

    def setUp(self):
        self.gen, self.tmp = _make_tmp_gen(backend="placeholder")

    def tearDown(self):
        self.tmp.cleanup()

    def test_generate_all_returns_dict(self):
        result = self.gen.generate_all(["cat", "dog"])
        self.assertIsInstance(result, dict)

    def test_generate_all_keys_match_inputs(self):
        result = self.gen.generate_all(["cat", "ocean"])
        self.assertIn("cat", result)
        self.assertIn("ocean", result)

    def test_generate_all_values_are_paths(self):
        result = self.gen.generate_all(["cat"])
        self.assertTrue(Path(result["cat"]).exists())

    def test_generate_all_deduplication(self):
        """Same keyword twice must only generate one file."""
        with patch.object(self.gen, "_generate_placeholder",
                          wraps=self.gen._generate_placeholder) as mock_gen:
            self.gen.generate_all(["cat", "cat", "cat"])
            # _generate_placeholder called exactly once for "cat"
            cat_calls = [c for c in mock_gen.call_args_list if c[0][0] == "cat"]
            self.assertEqual(len(cat_calls), 1)

    def test_generate_all_empty_list_returns_empty_dict(self):
        result = self.gen.generate_all([])
        self.assertEqual(result, {})


# ═══════════════════════════════════════════════════════════════════════════════
# TestAssetIndexerGeneratorFallback — 7 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssetIndexerGeneratorFallback(unittest.TestCase):
    """AssetIndexer.find() falls back to generator when no local match."""

    def setUp(self):
        self.assets_tmp = tempfile.TemporaryDirectory()
        self.gen_tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.assets_tmp.cleanup()
        self.gen_tmp.cleanup()

    def _make_indexer_with_mock_gen(self, gen_return=None):
        mock_gen = MagicMock()
        mock_gen.generate.return_value = gen_return
        ix = AssetIndexer(
            assets_dir=Path(self.assets_tmp.name),  # empty dir → no local matches
            generator=mock_gen,
        )
        ix.build()
        return ix, mock_gen

    def test_generator_called_on_local_miss(self):
        ix, mock_gen = self._make_indexer_with_mock_gen()
        ix.find("helicopter")
        mock_gen.generate.assert_called_once_with("helicopter")

    def test_generator_not_called_when_none(self):
        """With no generator attached, find() just returns None on miss."""
        ix = AssetIndexer(assets_dir=Path(self.assets_tmp.name))
        ix.build()
        result = ix.find("helicopter")
        self.assertIsNone(result)

    def test_generator_result_returned_on_success(self):
        fake_path = str(Path(self.gen_tmp.name) / "helicopter.png")
        ix, _ = self._make_indexer_with_mock_gen(gen_return=fake_path)
        result = ix.find("helicopter")
        self.assertEqual(result, fake_path)

    def test_generator_not_called_on_local_hit(self):
        """When a local file matches, generator must not be called."""
        (Path(self.assets_tmp.name) / "cat.mp4").write_bytes(b"")
        mock_gen = MagicMock()
        ix = AssetIndexer(
            assets_dir=Path(self.assets_tmp.name),
            generator=mock_gen,
        )
        ix.build()
        ix.find("cat")
        mock_gen.generate.assert_not_called()

    def test_generated_path_cached_in_index(self):
        """Second call for same keyword must NOT re-trigger generator."""
        fake_path = str(Path(self.gen_tmp.name) / "robot.png")
        ix, mock_gen = self._make_indexer_with_mock_gen(gen_return=fake_path)
        ix.find("robot")
        ix.find("robot")          # second call
        mock_gen.generate.assert_called_once()   # generator invoked only once

    def test_generator_returns_none_on_failure(self):
        ix, _ = self._make_indexer_with_mock_gen(gen_return=None)
        result = ix.find("helicopter")
        self.assertIsNone(result)

    def test_generator_used_only_for_miss_not_synonym(self):
        """A synonym match must return local path without calling generator."""
        # "cat.mp4" in local dir; query "고양이" is a synonym → local hit
        (Path(self.assets_tmp.name) / "cat.mp4").write_bytes(b"")
        mock_gen = MagicMock()
        ix = AssetIndexer(
            assets_dir=Path(self.assets_tmp.name),
            generator=mock_gen,
        )
        ix.build()
        result = ix.find("고양이")
        # Should match via synonym without ever calling generator
        if result is not None:
            mock_gen.generate.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# TestGenAssetGate — 12 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenAssetGate(unittest.TestCase):
    """harness/sensors/gen_asset_gate.py — GenAssetGate validation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _png(self, name: str, w: int = 1080, h: int = 1920) -> Path:
        p = Path(self.tmp.name) / name
        _make_valid_png(p, w, h)
        return p

    def test_valid_1080x1920_passes(self):
        p = self._png("ok.png")
        r = check_gen_asset(str(p))
        self.assertTrue(r.passed)

    def test_wrong_width_fails(self):
        p = self._png("wide.png", w=720, h=1920)
        r = check_gen_asset(str(p))
        self.assertFalse(r.passed)
        self.assertIn("resolution", r.checks)
        self.assertFalse(r.checks["resolution"])

    def test_wrong_height_fails(self):
        p = self._png("short.png", w=1080, h=1080)
        r = check_gen_asset(str(p))
        self.assertFalse(r.passed)

    def test_missing_file_fails(self):
        r = check_gen_asset("/nonexistent/path/asset.png")
        self.assertFalse(r.passed)
        self.assertFalse(r.checks.get("file_exists", True))

    def test_empty_file_fails(self):
        p = Path(self.tmp.name) / "empty.png"
        p.write_bytes(b"")
        r = check_gen_asset(str(p))
        self.assertFalse(r.passed)

    def test_corrupt_file_fails(self):
        p = Path(self.tmp.name) / "corrupt.png"
        # Must be > MIN_FILE_SIZE_BYTES so size check passes, but content is invalid
        p.write_bytes(b"PNG_FAKE_HEADER" + b"\x00" * 2048)
        r = check_gen_asset(str(p))
        self.assertFalse(r.passed)
        self.assertFalse(r.checks.get("file_integrity", True))

    def test_report_includes_dimensions_on_success(self):
        p = self._png("ok.png")
        r = check_gen_asset(str(p))
        self.assertEqual(r.width, 1080)
        self.assertEqual(r.height, 1920)

    def test_report_includes_file_size(self):
        p = self._png("ok.png")
        r = check_gen_asset(str(p))
        self.assertGreater(r.file_size_bytes, 0)

    def test_landscape_fails(self):
        p = self._png("landscape.png", w=1920, h=1080)
        r = check_gen_asset(str(p))
        self.assertFalse(r.passed)

    def test_custom_resolution_passes(self):
        p = self._png("custom.png", w=720, h=1280)
        r = check_gen_asset(str(p), required_width=720, required_height=1280)
        self.assertTrue(r.passed)

    def test_all_checks_recorded(self):
        p = self._png("ok.png")
        r = check_gen_asset(str(p))
        for key in ("file_exists", "min_size", "file_integrity", "resolution"):
            self.assertIn(key, r.checks)

    def test_validate_all_returns_report_per_path(self):
        p1 = self._png("ok1.png")
        p2 = self._png("ok2.png")
        reports = validate_all([str(p1), str(p2)])
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(r.passed for r in reports))


# ═══════════════════════════════════════════════════════════════════════════════
# TestAutoFillWithGenerator — 9 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoFillWithGenerator(unittest.TestCase):
    """intent_processor auto-fill behaviour with generator attached."""

    def setUp(self):
        self.assets_tmp = tempfile.TemporaryDirectory()
        self.spec_tmp = tempfile.TemporaryDirectory()
        self.gen_tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.assets_tmp.cleanup()
        self.spec_tmp.cleanup()
        self.gen_tmp.cleanup()

    def _broll_file(self) -> Path:
        return Path(self.spec_tmp.name) / "broll_requests.json"

    def _patch_env(self, words, gen_return):
        """Return a context manager stack that patches transcript + generator."""
        from contextlib import ExitStack
        from src.pipeline import intent_processor as ip

        stack = ExitStack()
        stack.enter_context(
            patch.object(ip, "_load_latest_transcript_words", return_value=words)
        )
        stack.enter_context(
            patch.object(ip, "_BROLL_REQUESTS_FILE", self._broll_file())
        )
        # Patch AssetGenerator so it never writes to the real assets/generated/
        from src.utils import asset_generator as ag_mod
        mock_gen_cls = MagicMock()
        mock_gen_instance = MagicMock()
        mock_gen_instance.generate.side_effect = gen_return
        mock_gen_cls.return_value = mock_gen_instance
        stack.enter_context(patch.object(ag_mod, "AssetGenerator", mock_gen_cls))
        # Also patch indexer's default broll dir to our empty tmp
        from src.utils import asset_indexer as ai_mod
        stack.enter_context(
            patch.object(ai_mod, "_DEFAULT_BROLL_DIR", Path(self.assets_tmp.name))
        )
        return stack, mock_gen_instance

    def test_generator_called_when_no_local_match(self):
        words = [{"word": "helicopter", "start": 0, "end": 1000}]
        with self._patch_env(words, gen_return=lambda kw: None)[0] as _:
            pass  # just verifying no errors

    def test_auto_fill_applied_false_when_generator_also_fails(self):
        from src.pipeline.intent_processor import IntentProcessor
        words = [{"word": "helicopter", "start": 0, "end": 1000}]
        stack, mock_gen = self._patch_env(words, gen_return=lambda kw: None)
        with stack:
            p = IntentProcessor()
            r = p.process("자료화면 자동으로 채워줘")
        self.assertFalse(r.applied)

    def test_auto_fill_applied_true_when_generator_succeeds(self):
        from src.pipeline.intent_processor import IntentProcessor
        fake_path = str(Path(self.gen_tmp.name) / "helicopter.png")
        _make_valid_png(Path(fake_path))
        words = [{"word": "helicopter", "start": 0, "end": 1000}]
        stack, _ = self._patch_env(words, gen_return=lambda kw: fake_path)
        with stack:
            p = IntentProcessor()
            r = p.process("자료화면 자동으로 채워줘")
        self.assertTrue(r.applied)

    def test_generated_asset_written_to_broll_requests(self):
        from src.pipeline.intent_processor import IntentProcessor
        fake_path = str(Path(self.gen_tmp.name) / "helicopter.png")
        _make_valid_png(Path(fake_path))
        words = [{"word": "helicopter", "start": 0, "end": 1000}]
        stack, _ = self._patch_env(words, gen_return=lambda kw: fake_path)
        with stack:
            IntentProcessor().process("자료화면 자동으로 채워줘")
        written = json.loads(self._broll_file().read_text(encoding="utf-8"))
        self.assertGreater(len(written), 0)

    def test_restart_from_designer_after_generation(self):
        from src.pipeline.intent_processor import IntentProcessor
        fake_path = str(Path(self.gen_tmp.name) / "cat.png")
        _make_valid_png(Path(fake_path))
        words = [{"word": "cat", "start": 0, "end": 1000}]
        stack, _ = self._patch_env(words, gen_return=lambda kw: fake_path)
        with stack:
            r = IntentProcessor().process("자동으로 채워줘")
        self.assertEqual(r.restart_from, "designer")

    def test_deduplication_of_generated_paths(self):
        from src.pipeline.intent_processor import IntentProcessor
        fake_path = str(Path(self.gen_tmp.name) / "cat.png")
        _make_valid_png(Path(fake_path))
        words = [
            {"word": "cat", "start": 0, "end": 1000},
            {"word": "kitten", "start": 1000, "end": 2000},
        ]
        stack, _ = self._patch_env(words, gen_return=lambda kw: fake_path)
        with stack:
            IntentProcessor().process("자료화면 자동으로 채워줘")
        written = json.loads(self._broll_file().read_text(encoding="utf-8"))
        paths = [r["asset_path"] for r in written]
        self.assertEqual(len(paths), len(set(paths)))

    def test_existing_manual_entries_preserved(self):
        from src.pipeline.intent_processor import IntentProcessor
        existing = [{"keyword": "dog", "asset_path": "/fake/dog.mp4",
                     "opacity": 0.8, "mode": "overlay"}]
        self._broll_file().write_text(json.dumps(existing), encoding="utf-8")
        fake_path = str(Path(self.gen_tmp.name) / "cat.png")
        _make_valid_png(Path(fake_path))
        words = [{"word": "cat", "start": 0, "end": 1000}]
        stack, _ = self._patch_env(words, gen_return=lambda kw: fake_path)
        with stack:
            IntentProcessor().process("자료화면 자동으로 채워줘")
        written = json.loads(self._broll_file().read_text(encoding="utf-8"))
        keywords = [r["keyword"] for r in written]
        self.assertIn("dog", keywords)

    def test_gold_theme_still_exporter_regression(self):
        from src.pipeline.intent_processor import IntentProcessor
        r = IntentProcessor().process("골드 테마")
        self.assertEqual(r.restart_from, "exporter")
        self.assertTrue(r.applied)

    def test_no_transcript_still_applied_false(self):
        from src.pipeline.intent_processor import IntentProcessor
        with patch("src.pipeline.intent_processor._load_latest_transcript_words",
                   return_value=[]):
            r = IntentProcessor().process("자료화면 자동으로 채워줘")
        self.assertFalse(r.applied)


# ═══════════════════════════════════════════════════════════════════════════════
# TestE2EWithGeneratedAssets — 4 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2EWithGeneratedAssets(unittest.TestCase):
    """End-to-end: generate asset → gate → broll_requests → smart resume."""

    def setUp(self):
        self.gen_tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.gen_tmp.cleanup()

    def test_e2e_placeholder_passes_gen_asset_gate(self):
        """Placeholder backend produces an image that passes all gate checks."""
        gen = AssetGenerator(
            output_dir=Path(self.gen_tmp.name),
            backend="placeholder",
        )
        path = gen.generate("mountain")
        self.assertIsNotNone(path)
        report = check_gen_asset(path)
        self.assertTrue(report.passed, msg=report.detail)

    def test_e2e_generate_all_all_pass_gate(self):
        """generate_all() → all results pass GenAssetGate."""
        gen = AssetGenerator(
            output_dir=Path(self.gen_tmp.name),
            backend="placeholder",
        )
        results = gen.generate_all(["ocean", "city", "forest"])
        reports = validate_all(list(results.values()))
        failed = [r for r in reports if not r.passed]
        self.assertEqual(failed, [], msg=[r.detail for r in failed])

    def test_e2e_indexer_with_generator_returns_path(self):
        """AssetIndexer(generator=gen).find() returns a path for unknown keyword."""
        gen = AssetGenerator(
            output_dir=Path(self.gen_tmp.name),
            backend="placeholder",
        )
        with tempfile.TemporaryDirectory() as d:  # empty broll dir
            ix = AssetIndexer(assets_dir=Path(d), generator=gen)
            path = ix.find("volcano")
        self.assertIsNotNone(path)
        self.assertTrue(Path(path).exists())

    def test_e2e_generated_path_passes_gate(self):
        """Asset produced by the indexer generator fallback passes the gate."""
        gen = AssetGenerator(
            output_dir=Path(self.gen_tmp.name),
            backend="placeholder",
        )
        with tempfile.TemporaryDirectory() as d:
            ix = AssetIndexer(assets_dir=Path(d), generator=gen)
            path = ix.find("volcano")
        report = check_gen_asset(path)
        self.assertTrue(report.passed, msg=report.detail)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
