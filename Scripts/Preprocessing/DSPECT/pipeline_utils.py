#!/usr/bin/env python3
"""Shared utilities for DSPECT preprocessing command orchestration."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence


def setup_logging(log_level: str = "INFO") -> None:
    """Configure process-wide logging for command orchestration scripts."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def run_command(cmd: Sequence[str], description: str) -> bool:
    """Run a subprocess command and emit structured logs for success and failure."""
    logging.info("%s", description)
    logging.info("Command: %s", " ".join(map(str, cmd)))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logging.error("%s failed with exit code %s", description, exc.returncode)
        if exc.stdout:
            logging.error("stdout:\n%s", exc.stdout)
        if exc.stderr:
            logging.error("stderr:\n%s", exc.stderr)
        return False

    logging.info("%s completed successfully", description)
    if result.stdout:
        logging.info("stdout:\n%s", result.stdout)
    if result.stderr:
        logging.info("stderr:\n%s", result.stderr)
    return True


def ensure_script_exists(base_dir: Path, script_name: str) -> Path:
    """Validate that a step script exists and return its full path."""
    script_path = base_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing pipeline step script: {script_path}")
    return script_path
