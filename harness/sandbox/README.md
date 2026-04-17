# Sandbox

Isolates each skill execution. Skills write only to a staging area;
changes are committed to the main timeline only after all quality gates pass.

Responsibilities:
- Provide a temporary working directory per skill run
- Roll back staging area on gate failure
- Log every file mutation with before/after checksums
