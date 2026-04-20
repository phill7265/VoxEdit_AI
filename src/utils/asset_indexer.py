"""
src/utils/asset_indexer.py

AssetIndexer — builds a keyword → asset-path index from assets/broll/ filenames.

Index construction
------------------
For each video file in the assets directory:
  1. The full lowercased stem is stored as a keyword.
  2. The stem is split on non-alphanumeric separators into tokens (≥2 chars each).
  3. Each token is stored as a keyword.
  4. Synonym expansions (Korean ↔ English) are added for every token.

Match priority (find)
---------------------
  1. Exact keyword match
  2. Keyword that starts with (or is a prefix of) the query
  3. Keyword that contains the query as a substring
  4. Synonym expansion then re-query

Usage
-----
    indexer = AssetIndexer()
    index   = indexer.build()          # {keyword: abs_path}
    path    = indexer.find("고양이")   # → str | None
    kws     = indexer.all_keywords()   # sorted list
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BROLL_DIR = _ROOT / "assets" / "broll"
_VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})

# ── Synonym table ─────────────────────────────────────────────────────────────
# Maps a Korean keyword to a list of English equivalents (and vice versa).
_SYNONYMS: dict[str, list[str]] = {
    "고양이": ["cat", "kitten", "kitty"],
    "강아지": ["dog", "puppy", "hound"],
    "자연":   ["nature", "outdoor", "forest", "park", "green"],
    "도시":   ["city", "urban", "street", "building", "downtown"],
    "음식":   ["food", "meal", "eating", "restaurant", "dish"],
    "사람":   ["person", "people", "human", "crowd", "man", "woman"],
    "하늘":   ["sky", "cloud", "clouds", "blue", "aerial"],
    "바다":   ["sea", "ocean", "beach", "wave", "coastal"],
    "산":     ["mountain", "hill", "hiking", "peak"],
    "학교":   ["school", "education", "student", "classroom"],
    "회사":   ["office", "business", "work", "meeting", "corporate"],
    "운동":   ["exercise", "workout", "sport", "gym", "fitness"],
    "컴퓨터": ["computer", "laptop", "technology", "screen", "tech"],
    "전화":   ["phone", "call", "mobile", "smartphone"],
    "자동차": ["car", "vehicle", "drive", "road", "automobile"],
    "꽃":     ["flower", "garden", "bloom", "floral"],
    "책":     ["book", "reading", "library", "study"],
    "음악":   ["music", "concert", "instrument", "song", "audio"],
    "요리":   ["cooking", "kitchen", "chef", "recipe"],
    "여행":   ["travel", "trip", "tourism", "vacation", "journey"],
    "아기":   ["baby", "infant", "toddler", "child"],
    "동물":   ["animal", "wildlife", "zoo", "pet"],
    "비":     ["rain", "rainy", "wet", "storm", "umbrella"],
    "눈":     ["snow", "winter", "snowy", "blizzard"],
    "불":     ["fire", "flame", "burning", "campfire"],
    "물":     ["water", "river", "lake", "stream", "liquid"],
}

# Reverse map: English token → Korean keyword
_REVERSE_SYNONYMS: dict[str, str] = {}
for _ko, _en_list in _SYNONYMS.items():
    for _en in _en_list:
        _REVERSE_SYNONYMS[_en.lower()] = _ko


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(stem: str) -> list[str]:
    """Split a filename stem into lowercase tokens (≥2 chars each)."""
    tokens = re.split(r"[^a-zA-Z0-9가-힣]+", stem.lower())
    return [t for t in tokens if len(t) >= 2]


# ── AssetIndexer ──────────────────────────────────────────────────────────────

class AssetIndexer:
    """Build and query a keyword → asset-path index from a broll directory.

    Parameters
    ----------
    assets_dir : Override the default assets/broll/ directory (used in tests).
    generator  : Optional AssetGenerator used as a fallback when find() has no
                 local match.  Pass None (default) to disable generation.
    """

    def __init__(
        self,
        assets_dir: Optional[Path] = None,
        generator: Optional[object] = None,   # AssetGenerator | None
    ) -> None:
        self._dir: Path = assets_dir or _DEFAULT_BROLL_DIR
        self._index: dict[str, str] = {}   # keyword (lowercase) → abs_path
        self._built: bool = False
        self._generator = generator         # AssetGenerator instance or None

    # ── Public API ─────────────────────────────────────────────────────────

    def build(self) -> dict[str, str]:
        """Scan the assets directory and return the keyword → path map.

        Calling build() again clears and rebuilds the index.
        """
        self._index = {}
        self._built = True

        if not self._dir.exists():
            return {}

        for f in sorted(self._dir.iterdir()):
            if f.suffix.lower() not in _VIDEO_EXTS:
                continue
            abs_path = str(f.resolve())
            tokens = _tokenize(f.stem)

            # Full stem as a keyword (first file for this key wins)
            self._index.setdefault(f.stem.lower(), abs_path)

            for tok in tokens:
                self._index.setdefault(tok, abs_path)

                # Synonym expansion: Korean → English
                for syn in _SYNONYMS.get(tok, []):
                    self._index.setdefault(syn.lower(), abs_path)

                # Synonym expansion: English → Korean
                rev = _REVERSE_SYNONYMS.get(tok)
                if rev:
                    self._index.setdefault(rev, abs_path)

        return dict(self._index)

    def find(self, query: str) -> str | None:
        """Return the best-matching asset path for `query`, or None.

        Triggers build() on first call if not already built.

        Match priority
        --------------
        1. Exact keyword match (local index)
        2. Prefix match  (local index)
        3. Substring containment  (local index)
        4. Synonym expansion then exact/prefix/substring retry  (local index)
        5. AssetGenerator fallback — calls self._generator.generate(query)
           only if a generator was provided and all local matches failed.
           On success, the generated path is injected into the index so
           subsequent calls for the same keyword are instant.
        """
        if not self._built:
            self.build()

        q = query.strip().lower()
        if not q:
            return None

        # 1. Exact match
        if q in self._index:
            return self._index[q]

        # 2. Prefix match
        for kw, path in self._index.items():
            if kw.startswith(q) or q.startswith(kw):
                return path

        # 3. Substring containment
        for kw, path in self._index.items():
            if q in kw or kw in q:
                return path

        # 4. Synonym expansion then retry
        for syn in _SYNONYMS.get(q, []):
            result = self._index.get(syn.lower())
            if result:
                return result
        rev = _REVERSE_SYNONYMS.get(q)
        if rev:
            result = self._index.get(rev)
            if result:
                return result

        # 5. Generator fallback (only if a generator is attached)
        if self._generator is not None:
            generated = self._generator.generate(query.strip())
            if generated:
                # Cache in index so repeat calls skip generation
                self._index[q] = generated
                return generated

        return None

    def all_keywords(self) -> list[str]:
        """Return all indexed keywords (sorted)."""
        if not self._built:
            self.build()
        return sorted(self._index.keys())

    def index_size(self) -> int:
        """Return the number of entries in the index."""
        if not self._built:
            self.build()
        return len(self._index)
