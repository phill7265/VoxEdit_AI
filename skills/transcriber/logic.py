"""
skills/transcriber/logic.py

Core transcription logic for VoxEdit AI.

Design
------
The Whisper dependency is hidden behind a WhisperBackend protocol so that:
  · Tests run without whisper/torch installed (inject MockWhisperBackend).
  · The real model is swapped in at runtime without touching this module.
  · A faster backend (e.g. faster-whisper, whisper.cpp) can be plugged in
    later by implementing the same protocol.

Output contract (consumed by skills/cutter and skills/audio/master):
  words[]      → {word, start, end, probability}   (per-word timestamps)
  vad_segments → {start, end, is_voice, avg_probability}  (for audio ducking)

Stateless guarantee
-------------------
This module never reads or writes harness state.
All I/O goes through the caller (skills/transcriber/main.py).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ── Spec constants ────────────────────────────────────────────────────────
VAD_VOICE_PROBABILITY_THRESHOLD: float = 0.85  # mirrors audio/master.py §4
VAD_MERGE_GAP_S: float = 0.30   # merge adjacent voice segments closer than this
DEFAULT_MODEL: str = "base"


# ── Output data models ────────────────────────────────────────────────────

@dataclass
class WordEntry:
    """Single word with per-word timestamps and recognition confidence.

    Fields
    ------
    word        : Recognised word text (may include leading/trailing space).
    start       : Word start time in seconds.
    end         : Word end time in seconds.
    probability : Whisper token log-probability converted to [0, 1].
    """
    word: str
    start: float
    end: float
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word.strip(),
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "probability": round(self.probability, 4),
            # Legacy fields for backward-compat with context_manager._slice_transcript
            "start_ms": int(self.start * 1000),
            "end_ms": int(self.end * 1000),
            "confidence": round(self.probability, 4),
        }


@dataclass
class VadSegment:
    """Voice Activity Detection result for one continuous speech region.

    Used by skills/audio/master.py to trigger background music ducking.

    Fields
    ------
    start           : Segment start in seconds.
    end             : Segment end in seconds.
    is_voice        : True when avg_probability >= VAD_VOICE_PROBABILITY_THRESHOLD.
    avg_probability : Mean word probability within this segment.
    word_count      : Number of words in the segment.
    """
    start: float
    end: float
    is_voice: bool
    avg_probability: float
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "is_voice": self.is_voice,
            "avg_probability": round(self.avg_probability, 4),
            "word_count": self.word_count,
            # Alias used by audio master VAD interface
            "confidence": round(self.avg_probability, 4),
            # Seconds aliases
            "start_s": round(self.start, 3),
            "end_s": round(self.end, 3),
        }


@dataclass
class TranscribeResult:
    """Complete output of one transcription run.

    Fields
    ------
    full_text    : Concatenated transcript string.
    words        : Word-level entries with timestamps and probabilities.
    vad_segments : Merged voice/silence regions for audio ducking.
    model_name   : Whisper model identifier used for this run.
    language     : Detected or forced language code.
    duration_s   : Audio duration in seconds.
    """
    full_text: str
    words: list[WordEntry]
    vad_segments: list[VadSegment]
    model_name: str
    language: str
    duration_s: float

    # ── Derived helpers ───────────────────────────────────────────────────

    def words_as_dicts(self) -> list[dict]:
        return [w.to_dict() for w in self.words]

    def vad_as_dicts(self) -> list[dict]:
        return [s.to_dict() for s in self.vad_segments]

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def avg_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.probability for w in self.words) / len(self.words)


# ── Backend protocol ──────────────────────────────────────────────────────

@runtime_checkable
class WhisperBackend(Protocol):
    """Minimal interface a Whisper backend must implement.

    Implementors
    ------------
    RealWhisperBackend  : wraps openai-whisper (requires torch)
    MockWhisperBackend  : deterministic stub used in tests
    """

    def load(self, model_name: str) -> None:
        """Load / warm up the model."""
        ...

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = True,
    ) -> dict[str, Any]:
        """Run transcription and return a raw Whisper-format result dict.

        Expected output format (subset used by this module):
        {
          "text": str,
          "language": str,
          "segments": [
            {
              "start": float,
              "end": float,
              "text": str,
              "words": [
                { "word": str, "start": float, "end": float,
                  "probability": float }
              ]
            }
          ]
        }
        """
        ...


# ── Real Whisper backend ──────────────────────────────────────────────────

class RealWhisperBackend:
    """Wraps openai-whisper.  Import is deferred so the module loads without torch.

    Audio loading strategy
    ----------------------
    Whisper's default loader calls ``ffmpeg`` (CLI) to decode audio.
    On systems where ffmpeg is not on PATH, we fall back to one of:
      1. imageio-ffmpeg  — registers the bundled binary on PATH automatically.
      2. stdlib wave     — direct WAV PCM read (no ffmpeg needed for .wav files).
    Non-WAV formats without ffmpeg will raise RuntimeError with install hints.
    """

    _WHISPER_SAMPLE_RATE: int = 16_000   # Whisper always expects 16 kHz mono float32

    def __init__(self) -> None:
        self._model: Any = None
        self._loaded_name: str = ""
        self._ffmpeg_ready: Optional[bool] = None   # lazily checked

    def load(self, model_name: str) -> None:
        try:
            import whisper  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            ) from exc

        if self._loaded_name != model_name:
            logger.info("Loading Whisper model '%s' …", model_name)
            self._model = whisper.load_model(model_name)
            self._loaded_name = model_name
            logger.info("Whisper model '%s' ready", model_name)

    # ── ffmpeg availability ───────────────────────────────────────────────

    def _ensure_ffmpeg(self) -> bool:
        """Try to make ffmpeg available; return True if it is reachable."""
        if self._ffmpeg_ready is not None:
            return self._ffmpeg_ready

        import shutil, subprocess
        if shutil.which("ffmpeg"):
            self._ffmpeg_ready = True
            return True

        # Try imageio-ffmpeg: it bundles a static binary but names it
        # differently (e.g. ffmpeg-win-x86_64-v7.1.exe).  Register its
        # directory on PATH so whisper's subprocess call finds "ffmpeg".
        try:
            import imageio_ffmpeg  # type: ignore[import]
            import os
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = str(Path(ffmpeg_exe).parent)
            # Create a symlink / copy named "ffmpeg.exe" if needed
            ffmpeg_target = Path(ffmpeg_dir) / "ffmpeg.exe"
            if not ffmpeg_target.exists():
                import shutil as _sh
                _sh.copy2(ffmpeg_exe, str(ffmpeg_target))
                logger.info("Copied imageio-ffmpeg binary → %s", ffmpeg_target)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("ffmpeg"):
                self._ffmpeg_ready = True
                logger.info("ffmpeg registered via imageio-ffmpeg: %s", ffmpeg_target)
                return True
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("imageio-ffmpeg setup failed: %s", exc)

        self._ffmpeg_ready = False
        return False

    # ── WAV fallback loader ───────────────────────────────────────────────

    @staticmethod
    def _load_wav_as_numpy(audio_path: str) -> Any:
        """Load a PCM WAV file into a float32 numpy array at 16 kHz mono.

        Uses only the stdlib ``wave`` module — no ffmpeg required.
        Resampling is performed with linear interpolation if the file sample
        rate differs from Whisper's expected 16 kHz.
        """
        import wave as _wave
        import numpy as np

        with _wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Decode PCM bytes → int array
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        if sampwidth not in dtype_map:
            raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")
        samples = np.frombuffer(raw, dtype=dtype_map[sampwidth]).astype(np.float32)

        # Down-mix to mono
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # Normalise to [-1, 1]
        samples /= float(2 ** (sampwidth * 8 - 1))

        # Resample to 16 kHz if necessary (linear interpolation)
        target_sr = RealWhisperBackend._WHISPER_SAMPLE_RATE
        if framerate != target_sr:
            import math
            target_len = int(math.ceil(len(samples) * target_sr / framerate))
            old_indices = np.linspace(0, len(samples) - 1, len(samples))
            new_indices = np.linspace(0, len(samples) - 1, target_len)
            samples = np.interp(new_indices, old_indices, samples).astype(np.float32)
            logger.debug("Resampled %d Hz → %d Hz (%d → %d samples)",
                         framerate, target_sr, len(old_indices), len(new_indices))

        return samples.astype(np.float32)

    # ── transcribe ────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = True,
    ) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Call load() before transcribe()")

        options: dict[str, Any] = {"word_timestamps": word_timestamps}
        if language:
            options["language"] = language

        logger.info("Whisper transcribing: %s", audio_path)

        # Resolve audio input: prefer ffmpeg path (supports all formats);
        # fall back to direct numpy load for WAV files.
        audio_input: Any = audio_path
        duration_hint_s: float = 0.0

        ext = Path(audio_path).suffix.lower()

        if not self._ensure_ffmpeg():
            if ext == ".wav":
                logger.info("ffmpeg not found — loading WAV via stdlib wave module")
                audio_input = self._load_wav_as_numpy(audio_path)
                duration_hint_s = len(audio_input) / self._WHISPER_SAMPLE_RATE
            else:
                raise RuntimeError(
                    f"ffmpeg is required to decode '{ext}' files but was not found. "
                    "Install ffmpeg or convert the file to WAV first.\n"
                    "  Windows: winget install ffmpeg\n"
                    "  Or:      pip install imageio-ffmpeg"
                )
        elif ext == ".wav":
            # Even when ffmpeg is available, pre-read WAV duration via stdlib
            # so _parse_whisper_output has a reliable fallback for non-speech
            # audio (pure tones, music) where Whisper returns no segments.
            try:
                import wave as _w
                with _w.open(audio_path, "rb") as wf:
                    duration_hint_s = wf.getnframes() / wf.getframerate()
            except Exception:
                pass

        result: dict[str, Any] = self._model.transcribe(audio_input, **options)

        # Inject actual audio duration so _parse_whisper_output can use it
        # as a fallback when Whisper returns no speech segments (e.g. music/tones).
        if duration_hint_s > 0.0:
            result.setdefault("_duration_hint_s", duration_hint_s)

        return result


# ── VAD derivation ────────────────────────────────────────────────────────

def _derive_vad_segments(
    words: list[WordEntry],
    *,
    threshold: float = VAD_VOICE_PROBABILITY_THRESHOLD,
    merge_gap_s: float = VAD_MERGE_GAP_S,
) -> list[VadSegment]:
    """Build VAD segments from word-level probabilities.

    Algorithm
    ---------
    1. Group consecutive words into preliminary segments by probability gate.
    2. Merge adjacent voice segments whose gap is < merge_gap_s.
    3. Fill gaps between voice segments with silence segments.

    This gives audio/master.py the confidence values it needs to decide when
    to trigger background music ducking (§4: VAD confidence ≥ 0.85).
    """
    if not words:
        return []

    # ── Step 1: tag each word ─────────────────────────────────────────────
    tagged: list[tuple[WordEntry, bool]] = [
        (w, w.probability >= threshold) for w in words
    ]

    # ── Step 2: group into raw segments ──────────────────────────────────
    # Break on tag change OR on a time gap > merge_gap_s so that two
    # voice words separated by a long silence produce two distinct segments
    # (Step 3 will re-merge them only if the gap is small enough).
    raw: list[VadSegment] = []
    i = 0
    while i < len(tagged):
        w, is_voice = tagged[i]
        seg_words = [w]
        j = i + 1
        while j < len(tagged) and tagged[j][1] == is_voice:
            gap = tagged[j][0].start - seg_words[-1].end
            if gap > merge_gap_s:
                break          # large time gap → start a new segment
            seg_words.append(tagged[j][0])
            j += 1

        avg_p = sum(sw.probability for sw in seg_words) / len(seg_words)
        raw.append(VadSegment(
            start=seg_words[0].start,
            end=seg_words[-1].end,
            is_voice=is_voice,
            avg_probability=avg_p,
            word_count=len(seg_words),
        ))
        i = j

    # ── Step 3: merge adjacent voice segments with small gap ──────────────
    merged: list[VadSegment] = []
    for seg in raw:
        if (
            merged
            and seg.is_voice
            and merged[-1].is_voice
            and seg.start - merged[-1].end < merge_gap_s
        ):
            prev = merged[-1]
            total_words = prev.word_count + seg.word_count
            merged_avg = (
                prev.avg_probability * prev.word_count
                + seg.avg_probability * seg.word_count
            ) / total_words
            merged[-1] = VadSegment(
                start=prev.start,
                end=seg.end,
                is_voice=True,
                avg_probability=merged_avg,
                word_count=total_words,
            )
        else:
            merged.append(seg)

    logger.debug(
        "_derive_vad_segments: %d words → %d raw → %d merged segments",
        len(words), len(raw), len(merged),
    )
    return merged


# ── Raw Whisper output parser ─────────────────────────────────────────────

def _parse_whisper_output(raw: dict[str, Any]) -> tuple[list[WordEntry], float]:
    """Extract WordEntry list and audio duration from a raw Whisper result.

    Returns (words, duration_s).
    duration_s is derived from the end timestamp of the last segment.
    Falls back to raw["_duration_hint_s"] when segments carry no end timestamps
    (common when Whisper processes non-speech audio such as music or tones).
    """
    words: list[WordEntry] = []
    duration_s: float = 0.0

    for seg in raw.get("segments", []):
        seg_end: float = seg.get("end", 0.0)
        duration_s = max(duration_s, seg_end)

        raw_words = seg.get("words", [])
        if not raw_words:
            # Segment has no word-level data — synthesise from segment timing
            seg_text: str = seg.get("text", "").strip()
            if seg_text:
                seg_start: float = seg.get("start", 0.0)
                # Distribute time evenly across tokens as a fallback
                tokens = seg_text.split()
                dur_per = (seg_end - seg_start) / max(len(tokens), 1)
                for k, tok in enumerate(tokens):
                    words.append(WordEntry(
                        word=tok,
                        start=seg_start + k * dur_per,
                        end=seg_start + (k + 1) * dur_per,
                        probability=0.5,   # unknown; use neutral value
                    ))
        else:
            for rw in raw_words:
                prob = float(rw.get("probability", 0.0))
                # Whisper stores log-prob as negative float; convert if needed
                if prob < 0.0:
                    prob = math.exp(prob)
                words.append(WordEntry(
                    word=str(rw.get("word", "")),
                    start=float(rw.get("start", 0.0)),
                    end=float(rw.get("end", 0.0)),
                    probability=min(1.0, max(0.0, prob)),
                ))

    # Fallback: use injected hint when segments carry no usable end timestamps
    if duration_s == 0.0:
        hint = raw.get("_duration_hint_s", 0.0)
        if hint > 0.0:
            duration_s = float(hint)
            logger.debug("_parse_whisper_output: duration from hint = %.3f s", duration_s)

    return words, duration_s


# ── Public entry point ────────────────────────────────────────────────────

def transcribe(
    audio_path: str,
    *,
    language: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    backend: Optional[WhisperBackend] = None,
) -> TranscribeResult:
    """Transcribe an audio/video file and return a structured result.

    Parameters
    ----------
    audio_path  : Absolute path to the audio or video file.
    language    : ISO-639-1 language code (e.g. "ko", "en").
                  None lets Whisper auto-detect.
    model_name  : Whisper model size: "tiny", "base", "small", "medium", "large".
    backend     : WhisperBackend implementation.
                  Defaults to RealWhisperBackend() when None.

    Returns
    -------
    TranscribeResult with full_text, words, vad_segments, and metadata.

    Raises
    ------
    FileNotFoundError  if audio_path does not exist.
    RuntimeError       if the backend fails to load or transcribe.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if backend is None:
        backend = RealWhisperBackend()

    backend.load(model_name)
    raw = backend.transcribe(audio_path, language=language, word_timestamps=True)

    words, duration_s = _parse_whisper_output(raw)
    full_text: str = raw.get("text", " ".join(w.word for w in words)).strip()
    detected_lang: str = raw.get("language", language or "unknown")
    vad_segments = _derive_vad_segments(words)

    result = TranscribeResult(
        full_text=full_text,
        words=words,
        vad_segments=vad_segments,
        model_name=model_name,
        language=detected_lang,
        duration_s=duration_s,
    )

    logger.info(
        "transcribe: done — %d words, %.1f s, lang=%s, avg_conf=%.3f",
        result.word_count, result.duration_s,
        result.language, result.avg_confidence,
    )
    return result
