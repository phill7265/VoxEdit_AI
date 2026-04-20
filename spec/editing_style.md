---
title: Editing Style Guide
version: 0.2.0
---

# VoxEdit AI — Editing Style Constitution

---

## 1. Quality Standards

| Parameter | Rule |
|-----------|------|
| Output resolution | Minimum 1080p (1920×1080) |
| Target loudness | -14 LUFS (streaming standard) |
| Audio sync drift | ≤ 2 frames tolerance |
| Minimum clip length | 0.5s in final export |
| Cut rhythm | Match spoken word cadence; no cuts under 1.5s |
| B-roll ratio | Minimum 30% of total runtime |

---

## 2. Silence Removal

**Skill owner:** `skills/cutter`

| Parameter | Value |
|-----------|-------|
| Detection threshold | Silence ≥ 0.5s |
| Action | Cutter adds segment to `silence_removal_candidates[]` |
| Scope | Speech segments only — music/SFX regions are excluded |

**Precise rule:**
> Any gap between spoken words where RMS audio level falls below -40 dBFS
> for a continuous duration of **≥ 0.5 seconds** must be flagged.
> Cutter does **not** remove automatically — it produces a candidate list.
> The pipeline runner confirms removal after sensor validation.

**Edge case:**
- Intentional pauses (e.g., dramatic beat) can be marked `keep: true` in the candidate list via user override.
- Silence at the very start or end of a clip (head/tail silence ≥ 0.2s) is always removed without candidate review.

---

## 3. Jump Cut

**Skill owner:** `skills/cutter`, `skills/designer`

| Parameter | Value |
|-----------|-------|
| Trigger condition | Two consecutive cuts within the same speaker segment |
| Visual treatment | 1.1× zoom-in on the second clip |
| Zoom anchor | Center of detected face bounding box; fallback to frame center |
| Transition duration | 0 frames (hard cut — zoom is baked into the clip, not a transition effect) |

**Precise rule:**
> When the cutter produces two consecutive in/out points on the same speaker track
> with no B-roll between them, the designer must apply a **1.1× scale** to the second clip.
> Scale is applied around the face anchor point detected by the designer's vision sensor.
> If no face is detected, scale around the geometric center of the frame.

---

## 4. Audio Ducking

**Skill owner:** `skills/designer`

| Parameter | Value |
|-----------|-------|
| Trigger | Voice activity detected (VAD confidence ≥ 0.85) |
| Background music adjustment | −20 dB relative to current level |
| Attack time | 150 ms (fade-in to ducked level) |
| Release time | 500 ms (fade-out back to original level) |
| Scope | Background music track only; SFX tracks are unaffected |

**Precise rule:**
> When the voice activity detector fires on the primary speaker track,
> the background music bus is reduced by **20 dB** with a 150 ms attack.
> When VAD clears, music recovers over 500 ms.
> Ducking must not drop music below absolute silence (−∞ dBFS) —
> minimum ducked floor is **−40 dBFS**.

---

## 5. Acceptance Criteria (Quality Gate Summary)

These are the pass/fail conditions checked by `harness/sensors/` after each skill run.

| Gate | Condition | Fail action |
|------|-----------|-------------|
| AUDIO_SYNC | Drift ≤ 2 frames | Reject clip |
| SILENCE | No unflagged gap > 0.5s in speech | Re-run cutter on segment |
| RESOLUTION | ≥ 1080p | Upscale or reject source |
| DURATION | All clips ≥ 0.5s | Merge with adjacent clip |
| JUMP_CUT_ZOOM | 1.1× scale applied to all jump cuts | Re-run designer on segment |
| DUCK_LEVEL | Music ducked to ≤ original − 20 dB during speech | Re-run audio mix |


## 6. Caption Position

| Parameter | Value |
|-----------|-------|
| CAPTION_Y_PX: 768 | Default subtitle Y coordinate (84% of 1920px height) |
HIGHLIGHT_COLOR: #FFD700
CAPTION_COLOR: #FFFFFF

## 7. Dynamic Visual Rhythm

| Parameter | Value |
|-----------|-------|
| RHYTHM_INTENSITY | 0.50 (range 0.0–1.0; controls DynamicZoom strength) |
RHYTHM_INTENSITY: 0.50
ZOOM_FOCUS_ENABLED: false
CAPTION_FONT_SIZE_PT: 86
