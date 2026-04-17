# Memory Manager

Persists project state across skill invocations.

Stores:
- Timeline cursor (current edit position)
- Skill execution history (inputs, outputs, durations)
- User preference cache (style overrides)

Format: append-only JSONL log + periodic snapshot
