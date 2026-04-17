# Skill: Transcriber

Converts audio to word-level timestamped transcript.

Inputs:
- Audio file (extracted from source video)

Outputs:
- transcript.json  — { word, start_ms, end_ms, confidence }[]
- captions.srt     — optional subtitle file

Upstream of: Cutter, Designer (for subtitle overlays)
