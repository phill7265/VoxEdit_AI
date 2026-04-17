"""
harness/sandbox/executor.py

Validates and runs FFmpeg commands issued by skills.
A skill must NEVER call FFmpeg directly — all commands route through this executor.

Safety guarantees:
  1. Overwrite guard  — rejects any command that would clobber the source file
  2. Resource monitor — kills the subprocess if CPU or memory exceed safe thresholds
  3. Staging enforcement — output path must be inside the designated staging directory
  4. Dry-run mode — validate without execution (used by the pipeline before committing)
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STAGING_ROOT = Path(__file__).resolve().parents[2] / "staging"

# Resource limits
CPU_LIMIT_PCT: float = 85.0        # kill if process exceeds this % for sustained period
MEMORY_LIMIT_MB: float = 2048.0    # kill if RSS exceeds this
RESOURCE_POLL_INTERVAL_S: float = 2.0
CPU_SUSTAINED_SECONDS: float = 10.0  # must exceed CPU limit for this long before kill


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    success: bool
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    killed_reason: Optional[str] = None
    validation_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _parse_output_path(args: list[str]) -> Optional[Path]:
    """Return the output file path from a parsed FFmpeg argument list.

    FFmpeg convention: the last positional argument that is not a flag
    or flag-value is the output file.
    """
    # Walk backwards; skip flag values (preceded by a flag starting with -)
    skip_next = False
    for token in reversed(args):
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            skip_next = False
            continue
        # First non-flag token from the right is the output file
        return Path(token)
    return None


def _parse_input_paths(args: list[str]) -> list[Path]:
    """Return all input file paths (-i <path>) from a parsed FFmpeg argument list."""
    inputs: list[Path] = []
    for i, token in enumerate(args):
        if token == "-i" and i + 1 < len(args):
            inputs.append(Path(args[i + 1]))
    return inputs


def validate_command(command: str, source_file: str) -> list[str]:
    """Validate an FFmpeg command string before execution.

    Returns a list of error strings. Empty list means the command is safe to run.
    """
    errors: list[str] = []

    try:
        args = shlex.split(command)
    except ValueError as exc:
        errors.append(f"Command parse error: {exc}")
        return errors

    # Must start with ffmpeg
    if not args or Path(args[0]).name.lower() not in ("ffmpeg", "ffmpeg.exe"):
        errors.append("Command must start with 'ffmpeg'")
        return errors

    source_path = Path(source_file).resolve()
    output_path = _parse_output_path(args[1:])  # skip 'ffmpeg' itself
    input_paths = _parse_input_paths(args[1:])

    # --- Guard 1: Overwrite check -------------------------------------------
    if output_path is not None:
        resolved_output = output_path.resolve()
        if resolved_output == source_path:
            errors.append(
                f"Overwrite guard: output '{resolved_output}' "
                f"would clobber the source file '{source_path}'"
            )
        for inp in input_paths:
            if inp.resolve() == resolved_output:
                errors.append(
                    f"Overwrite guard: output '{resolved_output}' "
                    f"matches input '{inp.resolve()}'"
                )

    # --- Guard 2: Staging enforcement ----------------------------------------
    if output_path is not None:
        resolved_output = output_path.resolve()
        try:
            resolved_output.relative_to(STAGING_ROOT.resolve())
        except ValueError:
            errors.append(
                f"Staging enforcement: output '{resolved_output}' "
                f"is outside staging directory '{STAGING_ROOT}'. "
                "All skill outputs must be written to the staging area."
            )

    # --- Guard 3: Dangerous flags ---------------------------------------------
    dangerous_flags = {"-y": "disables overwrite prompt (use -n instead)"}
    for flag, reason in dangerous_flags.items():
        if flag in args:
            errors.append(f"Dangerous flag '{flag}': {reason}")

    return errors


# ---------------------------------------------------------------------------
# Resource monitor (runs in a daemon thread)
# ---------------------------------------------------------------------------

class _ResourceMonitor(threading.Thread):
    """Watches a subprocess and terminates it if resource limits are breached."""

    def __init__(self, proc: subprocess.Popen) -> None:
        super().__init__(daemon=True)
        self.proc = proc
        self.killed_reason: Optional[str] = None
        self._stop_event = threading.Event()

        self._cpu_over_limit_since: Optional[float] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            ps_proc = psutil.Process(self.proc.pid)
        except psutil.NoSuchProcess:
            return

        while not self._stop_event.is_set():
            try:
                # CPU: measured over a short interval for accuracy
                cpu_pct = ps_proc.cpu_percent(interval=RESOURCE_POLL_INTERVAL_S)
                mem_mb = ps_proc.memory_info().rss / (1024 * 1024)
            except psutil.NoSuchProcess:
                break

            # Memory hard limit
            if mem_mb > MEMORY_LIMIT_MB:
                self.killed_reason = (
                    f"Memory limit exceeded: {mem_mb:.1f} MB > {MEMORY_LIMIT_MB} MB"
                )
                logger.warning("Executor: %s — killing process %d", self.killed_reason, self.proc.pid)
                self._kill()
                return

            # CPU sustained limit
            if cpu_pct > CPU_LIMIT_PCT:
                if self._cpu_over_limit_since is None:
                    self._cpu_over_limit_since = time.monotonic()
                elif time.monotonic() - self._cpu_over_limit_since >= CPU_SUSTAINED_SECONDS:
                    self.killed_reason = (
                        f"CPU limit exceeded: {cpu_pct:.1f}% > {CPU_LIMIT_PCT}% "
                        f"sustained for {CPU_SUSTAINED_SECONDS}s"
                    )
                    logger.warning("Executor: %s — killing process %d", self.killed_reason, self.proc.pid)
                    self._kill()
                    return
            else:
                self._cpu_over_limit_since = None

    def _kill(self) -> None:
        try:
            self.proc.kill()
        except ProcessLookupError:
            pass
        finally:
            self._stop_event.set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    command: str,
    source_file: str,
    *,
    dry_run: bool = False,
    timeout_s: Optional[float] = None,
) -> ExecutionResult:
    """Validate and execute an FFmpeg command inside the sandbox.

    Args:
        command:     Full FFmpeg command string as the skill produced it.
        source_file: Absolute path to the original source file (used for overwrite guard).
        dry_run:     If True, validate only — do not execute.
        timeout_s:   Hard wall-clock timeout. None = no timeout.

    Returns:
        ExecutionResult with success flag and diagnostic fields.
    """
    errors = validate_command(command, source_file)
    if errors:
        logger.error("Executor validation failed:\n  %s", "\n  ".join(errors))
        return ExecutionResult(success=False, validation_errors=errors)

    if dry_run:
        logger.info("Executor dry-run OK: %s", command)
        return ExecutionResult(success=True)

    # Ensure staging directory exists
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("Executor launching: %s", command)
    args = shlex.split(command)

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    monitor = _ResourceMonitor(proc)
    monitor.start()

    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        monitor.stop()
        return ExecutionResult(
            success=False,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            killed_reason=f"Hard timeout exceeded ({timeout_s}s)",
        )
    finally:
        monitor.stop()
        monitor.join(timeout=2.0)

    if monitor.killed_reason:
        return ExecutionResult(
            success=False,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            killed_reason=monitor.killed_reason,
        )

    success = proc.returncode == 0
    if not success:
        logger.error("Executor: FFmpeg exited %d\n%s", proc.returncode, stderr)
    else:
        logger.info("Executor: completed OK (rc=0)")

    return ExecutionResult(
        success=success,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )
