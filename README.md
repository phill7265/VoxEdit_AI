# VoxEdit AI

AI-powered video editing pipeline using harness engineering and agent skill methodology.

## Structure

```
VoxEdit_AI/
├── spec/                    # Project constitution — editing style & quality gates
│   ├── editing_style.md
│   └── quality_gates.md
│
├── harness/                 # Safety wrapper around all skill execution
│   ├── sandbox/             # Staging area isolation + rollback
│   ├── memory/              # Persistent state across skill runs
│   └── sensors/             # Observability — A/V sync, silence, resolution
│
├── skills/                  # Stateless processing units
│   ├── transcriber/         # Audio → word-level timestamped transcript
│   ├── cutter/              # Transcript → cut list (EDL)
│   ├── designer/            # Timeline → visual overlays, color, B-roll
│   └── exporter/            # Timeline → rendered deliverable
│
├── docs/                    # Design documents
│   ├── PRD.md
│   └── architecture.md
│
└── src/                     # Application layer
    ├── api/                 # Job submission & status endpoints
    ├── pipeline/            # Skill orchestration runner
    └── utils/               # Shared helpers
```

## Core Principle
Skills are stateless. The harness owns all state, safety, and observability.
The spec is the source of truth for quality decisions.
