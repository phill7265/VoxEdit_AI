"""
src/utils/asset_generator.py

AssetGenerator — generates photorealistic images for missing b-roll keywords.

Backends
--------
"placeholder"  : Creates a valid 1080×1920 PNG via Pillow — no API needed.
                 Always used in tests; useful in development without an API key.
"replicate"    : Calls Replicate API with SDXL (requires REPLICATE_API_TOKEN env var).
                 Generates genuine photorealistic images at 1080×1920.
"auto"         : Tries replicate if REPLICATE_API_TOKEN is set, otherwise placeholder.

Caching
-------
Generated images are stored in assets/generated/{safe_keyword}.png.
If the file already exists, generate() returns the cached path immediately
without calling the API or regenerating the image.

Usage
-----
    gen = AssetGenerator()
    path = gen.generate("고양이")      # str path or None on failure
    paths = gen.generate_all(["고양이", "ocean"])  # {keyword: path}
    ok = gen.is_cached("고양이")       # True / False
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GENERATED_DIR = _ROOT / "assets" / "generated"

# ── Photorealistic prompt template ─────────────────────────────────────────────
_PROMPT_TEMPLATE = (
    "photorealistic {keyword}, cinematic lighting, 8K resolution, "
    "sharp details, professional photography, hyperrealistic, "
    "portrait orientation 9:16, high dynamic range, studio quality"
)

# Replicate SDXL model version (stable-diffusion-xl-base-1.0)
_REPLICATE_MODEL = "stability-ai/sdxl"
_REPLICATE_VERSION = "39ed52f2319f9f1ff27c9f2c9ee7d2a4e26f5e96a8b8e1d3d7c6d5f4e3b2a1c0"


# ── AssetGenerator ────────────────────────────────────────────────────────────

class AssetGenerator:
    """Generate photorealistic 1080×1920 images for b-roll keywords.

    Parameters
    ----------
    output_dir  : Where to save generated images (default: assets/generated/).
    backend     : "placeholder", "replicate", or "auto".
    timeout_s   : Maximum seconds to wait for Replicate API results.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        backend: str = "auto",
        timeout_s: int = 120,
    ) -> None:
        self._output_dir: Path = output_dir or _DEFAULT_GENERATED_DIR
        self._backend: str = backend
        self._timeout_s: int = timeout_s

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(self, keyword: str) -> str | None:
        """Return a path to a 1080×1920 PNG for `keyword`.

        Returns the cached path if already generated.
        Returns None only if the backend fails AND placeholder also fails.
        """
        keyword = keyword.strip()
        if not keyword:
            return None

        cached = self._cache_path(keyword)
        if cached.exists():
            logger.debug("AssetGenerator: cache hit for '%s' → %s", keyword, cached)
            return str(cached)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("AssetGenerator: generating asset for keyword '%s' (backend=%s)",
                    keyword, self._backend)

        if self._backend == "placeholder":
            return self._generate_placeholder(keyword)

        elif self._backend == "replicate":
            return self._generate_replicate(keyword)

        elif self._backend == "auto":
            api_token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
            if api_token:
                result = self._generate_replicate(keyword)
                if result:
                    return result
                logger.warning(
                    "AssetGenerator: Replicate failed for '%s', falling back to placeholder",
                    keyword,
                )
            return self._generate_placeholder(keyword)

        else:
            raise ValueError(f"AssetGenerator: unknown backend {self._backend!r}")

    def generate_all(self, keywords: list[str]) -> dict[str, str]:
        """Generate assets for all keywords.

        Returns {keyword: abs_path} for every keyword that succeeded.
        Already-cached keywords are included without re-generating.
        Duplicate keywords are processed only once.
        """
        seen: set[str] = set()
        results: dict[str, str] = {}
        for kw in keywords:
            kw = kw.strip()
            if not kw or kw in seen:
                continue
            seen.add(kw)
            path = self.generate(kw)
            if path:
                results[kw] = path
        return results

    def is_cached(self, keyword: str) -> bool:
        """Return True if a generated asset already exists for `keyword`."""
        return self._cache_path(keyword.strip()).exists()

    def cache_path(self, keyword: str) -> Path:
        """Return the expected output path for `keyword` (may not exist yet)."""
        return self._cache_path(keyword.strip())

    # ── Prompt builder ──────────────────────────────────────────────────────

    def build_prompt(self, keyword: str) -> str:
        """Build a photorealistic generation prompt for `keyword`."""
        return _PROMPT_TEMPLATE.format(keyword=keyword)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _cache_path(self, keyword: str) -> Path:
        """Sanitize `keyword` and return its cache file path."""
        safe = re.sub(r"[^a-zA-Z0-9가-힣]", "_", keyword.lower())
        safe = re.sub(r"_+", "_", safe).strip("_") or "asset"
        return self._output_dir / f"{safe}.png"

    def _generate_placeholder(self, keyword: str) -> str | None:
        """Create a 1080×1920 placeholder PNG using Pillow.

        The image is a dark background with the keyword and resolution label
        drawn in the centre.  Always succeeds if Pillow is installed.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new("RGB", (1080, 1920), color=(28, 28, 35))
            draw = ImageDraw.Draw(img)

            # ── Gradient-style band ──────────────────────────────────────
            for y in range(1920):
                lum = int(28 + 20 * (y / 1920))
                for x in range(1080):
                    img.putpixel((x, y), (lum, lum, lum + 10))

            draw = ImageDraw.Draw(img)

            # ── Try system font (Korean support) ────────────────────────
            font_large: object
            font_small: object
            try:
                font_large = ImageFont.truetype(
                    "C:/Windows/Fonts/malgun.ttf", size=96
                )
                font_small = ImageFont.truetype(
                    "C:/Windows/Fonts/malgun.ttf", size=44
                )
            except Exception:
                font_large = ImageFont.load_default()
                font_small = font_large

            # ── Keyword label (centred) ──────────────────────────────────
            bbox = draw.textbbox((0, 0), keyword, font=font_large)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                ((1080 - tw) // 2, (1920 - th) // 2 - 60),
                keyword,
                fill=(220, 220, 255),
                font=font_large,
            )

            # ── Metadata labels ──────────────────────────────────────────
            meta = "1080 × 1920  |  photorealistic placeholder"
            draw.text((40, 40), meta, fill=(120, 120, 140), font=font_small)
            draw.text(
                (40, 1920 - 80),
                "VoxEdit AI — generated asset",
                fill=(80, 80, 100),
                font=font_small,
            )

            out_path = self._cache_path(keyword)
            img.save(str(out_path), "PNG")
            logger.info(
                "AssetGenerator: placeholder saved → %s (%dx%d)",
                out_path.name, *img.size
            )
            return str(out_path)

        except Exception as exc:
            logger.error("AssetGenerator: placeholder generation failed — %s", exc)
            return None

    def _generate_replicate(self, keyword: str) -> str | None:
        """Call Replicate REST API to generate a photorealistic image.

        Polls until the prediction succeeds, times out, or fails.
        Returns the local path of the downloaded image, or None on any error.
        """
        import time
        import httpx

        api_token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
        if not api_token:
            return None

        prompt = self.build_prompt(keyword)
        headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": {
                "prompt": prompt,
                "negative_prompt": (
                    "blurry, low quality, distorted, deformed, cartoon, anime"
                ),
                "width": 1080,
                "height": 1920,
                "num_outputs": 1,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            },
        }

        try:
            resp = httpx.post(
                f"https://api.replicate.com/v1/models/{_REPLICATE_MODEL}/predictions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            prediction = resp.json()
            prediction_id = prediction.get("id")
            if not prediction_id:
                logger.error("AssetGenerator: Replicate returned no prediction id")
                return None

        except Exception as exc:
            logger.error("AssetGenerator: Replicate create-prediction failed — %s", exc)
            return None

        # ── Poll for completion ──────────────────────────────────────────
        deadline = time.time() + self._timeout_s
        poll_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        try:
            while time.time() < deadline:
                time.sleep(2)
                poll = httpx.get(poll_url, headers=headers, timeout=30)
                poll.raise_for_status()
                data = poll.json()
                status = data.get("status", "")

                if status == "succeeded":
                    outputs = data.get("output", [])
                    if not outputs:
                        return None
                    return self._download_image(outputs[0], keyword)

                if status in ("failed", "canceled"):
                    err = data.get("error", "unknown")
                    logger.error(
                        "AssetGenerator: Replicate prediction %s — %s", status, err
                    )
                    return None

        except Exception as exc:
            logger.error("AssetGenerator: Replicate poll failed — %s", exc)

        logger.warning(
            "AssetGenerator: Replicate timeout after %ds for '%s'",
            self._timeout_s, keyword,
        )
        return None

    def _download_image(self, url: str, keyword: str) -> str | None:
        """Download an image from `url` and save to the cache path."""
        import httpx
        try:
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            out_path = self._cache_path(keyword)
            out_path.write_bytes(resp.content)
            logger.info(
                "AssetGenerator: downloaded %d bytes → %s",
                len(resp.content), out_path.name,
            )
            return str(out_path)
        except Exception as exc:
            logger.error("AssetGenerator: download failed — %s", exc)
            return None
