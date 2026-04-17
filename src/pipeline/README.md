# Pipeline Runner

Orchestrates skill execution in order, passing outputs between stages.

Sequence:
1. Transcriber  → transcript.json
2. Cutter       → cut_list.edl
3. Designer     → annotated_timeline.json
4. Exporter     → final_video + manifest

Each step runs inside the harness sandbox and sensor checks fire after each skill completes.
