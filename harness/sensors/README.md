# Sensors

Observability layer — monitors skill outputs before they leave the harness.

Sensors:
- `audio_sync_sensor`: measures A/V drift
- `silence_sensor`: detects long pauses
- `quality_sensor`: checks resolution and bitrate

Each sensor emits a structured event consumed by the quality gate pipeline.
