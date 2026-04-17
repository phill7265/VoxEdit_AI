# Skill: Cutter

Responsible for all timeline cutting decisions.

Inputs:
- Raw video file path
- Transcript JSON (word-level timestamps)
- Style spec (from /spec)

Outputs:
- Cut list (EDL format): list of [in, out] timecodes

Triggers quality gates: SILENCE, DURATION
