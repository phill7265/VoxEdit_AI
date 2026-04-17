# VoxEdit AI — Product Requirements Document

## Vision
AI-powered video editing pipeline that applies harness engineering and agent skill methodology to automate the post-production workflow for spoken-word content.

## Core Flow
```
[Raw Video] → Transcriber → Cutter → Designer → Exporter → [Deliverable]
                                ↕           ↕          ↕
                            [Harness: Sandbox + Sensors + Memory]
                                        ↕
                                  [Spec / Quality Gates]
```

## Skill Contracts
Each skill must:
1. Accept a typed input schema
2. Write outputs only to the sandbox staging area
3. Pass all relevant quality gates before committing

## Out of Scope (v0.1)
- Multi-camera editing
- Real-time preview streaming
- Cloud render farm integration
