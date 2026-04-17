# Architecture Overview

## Layers

| Layer | Folder | Purpose |
|-------|--------|---------|
| Constitution | `/spec` | Immutable style rules and quality gates |
| Harness | `/harness` | Safety wrapper: sandbox, memory, sensors |
| Skills | `/skills` | Stateless processing units |
| Application | `/src` | Orchestration, API, pipeline runner |

---

## Execution Model

Skills are stateless — all state lives in the harness memory.
The pipeline runner in `/src/pipeline` sequences skills and wires sensor output to gates.
The sandbox ensures a failed skill cannot corrupt the committed timeline.

---

## Harness Operating Principles

### 1. Handover Notes

Every skill execution must close with a structured handover record written to `harness/memory/`.
This record is the **only** channel through which one skill's output becomes the next skill's input.

```
harness/memory/
  └── jobs/
        └── {job_id}/
              ├── 001_transcriber.json   # { status, output_path, word_count, duration_ms }
              ├── 002_cutter.json        # { status, output_path, cuts_applied, silence_removed }
              ├── 003_designer.json      # { status, output_path, overlays_added }
              └── 004_exporter.json      # { status, output_path, file_size_mb, export_profile }
```

**Rules:**
- A skill may NOT read a prior skill's raw output directly. It reads only the handover record.
- The pipeline runner resolves the actual file paths from the record and passes them in.
- On session resume, the runner replays from the last completed handover record — no skill re-runs from scratch.

---

### 2. Context Firewall

`src/pipeline` acts as a **context firewall** between skills and the full job state.
A skill receives only the minimum context required for its current work segment:

| Skill | Context provided | Context withheld |
|-------|-----------------|-----------------|
| Transcriber | Raw audio path, language hint | Full job config, prior handover records |
| Cutter | Transcript for current segment (±30s window), style rules | Full transcript, designer config |
| Designer | Timeline for current segment, brand kit | Cutter internals, export profile |
| Exporter | Final annotated timeline, export profile | All upstream processing history |

**Why this matters — Context Corruption:**
Passing the full accumulated context into every skill causes two failure modes:
1. Skills make decisions based on stale or irrelevant information from earlier stages.
2. LLM-backed skills drift toward earlier instructions as context grows, degrading output quality.

The firewall prevents both by injecting only the slice of context that is semantically relevant to the current task.

**Implementation contract for `src/pipeline`:**
```
build_skill_context(skill_name, job_id) -> SkillContext
  reads:  harness/memory/jobs/{job_id}/
  emits:  SkillContext { current_segment, required_inputs_only }
  never:  passes raw accumulated history or full job state
```

---

## Sensor → Gate Flow

```
Skill executes (in sandbox)
        │
        ▼
Sensors fire (harness/sensors/)
  ├── audio_sync_sensor   → drift_frames
  ├── silence_sensor      → max_gap_ms
  └── quality_sensor      → resolution, bitrate
        │
        ▼
Quality Gates evaluate (spec/quality_gates.md)
  ├── PASS → commit sandbox to timeline, write handover record
  └── FAIL → rollback sandbox, log failure in handover record, halt pipeline
```

---

## Session Resume

On restart, the pipeline runner:
1. Reads all handover records under `harness/memory/jobs/{job_id}/`
2. Identifies the last record with `status: "success"`
3. Resumes from the next skill in sequence
4. Re-injects only the context window for the resumed segment

No completed work is discarded. No skill that already passed gates is re-run.
