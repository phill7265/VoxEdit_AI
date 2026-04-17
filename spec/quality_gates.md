---
title: Quality Gates
version: 0.1.0
---

# Quality Gates

Gates that must pass before any skill writes to the output timeline.

| Gate | Check | Fail Action |
|------|-------|-------------|
| AUDIO_SYNC | drift ≤ 2 frames | reject clip |
| SILENCE | no gap > 0.8s in speech | re-trim |
| RESOLUTION | ≥ 1080p | upscale or reject |
| DURATION | clip ≥ 0.5s | merge with adjacent |
