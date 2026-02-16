#!/usr/bin/env python3
"""Run the DSPECT preprocessing pipeline as a structured sequence of steps."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from pipeline_utils import ensure_script_exists, run_command, setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI parser for DSPECT orchestration."""
    parser = argparse.ArgumentParser(description="Run complete DSPECT preprocessing pipeline")
    parser.add_argument("--diagnosis", choices=["CN", "PD"], required=True, help="Diagnosis group to process")
    parser.add_argument(
        "--is_hazel",
        "--isHasel",
        action="store_true",
        dest="is_hazel",
        help="Enable Hasel server mode for compatible downstream calls",
    )
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if output exists")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=[91, 109, 91],
        metavar=("X", "Y", "Z"),
        help="Target shape for finalisation",
    )
    parser.add_argument("--intensity_norm", action="store_true", help="Apply intensity normalisation in step 5")
    parser.add_argument(
        "--mask_type",
        choices=["occipital", "whole_brain"],
        default="whole_brain",
        help="Masking strategy for step 4",
    )
    parser.add_argument("--isotropic", action="store_true", help="Resample to isotropic 1mm voxels in step 1")
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config.yaml",
        help="YAML step configuration file relative to this script",
    )
    parser.add_argument("--skip_validation", action="store_true", help="Skip post-run validation script")
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser


def load_step_config(config_path: Path) -> List[Dict[str, object]]:
    """Load configured pipeline steps from YAML."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    steps = config.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"No valid steps found in config: {config_path}")
    return steps


def build_command(step: Dict[str, object], args: argparse.Namespace, script_dir: Path) -> List[str]:
    """Build command args for a configured step."""
    script_name = str(step["script"])
    ensure_script_exists(script_dir, script_name)
    cmd: List[str] = ["python3", script_name, "--diagnosis", args.diagnosis]

    if script_name == "1_reorient.py":
        if args.force:
            cmd.append("--force")
        if args.isotropic:
            cmd.append("--isotropic")
    elif script_name == "4_masking.py":
        cmd.extend(["--mask_type", args.mask_type])
    elif script_name == "5_padding.py":
        cmd.extend(["--shape", str(args.shape[0]), str(args.shape[1]), str(args.shape[2])])
        if args.intensity_norm:
            cmd.append("--intensity_norm")
    elif script_name == "6_postprocess.py" and args.is_hazel:
        cmd.append("--isHasel")

    for extra in step.get("extra_args", []):
        cmd.append(str(extra))
    return cmd


def run_pipeline(args: argparse.Namespace) -> int:
    """Execute configured DSPECT preprocessing steps."""
    script_dir = Path(__file__).parent.resolve()
    config_path = (script_dir / args.config).resolve()
    os.chdir(script_dir)
    setup_logging(args.log_level)

    logging.info("Starting DSPECT preprocessing pipeline for diagnosis=%s", args.diagnosis)
    logging.info("Using config: %s", config_path)
    steps = load_step_config(config_path)

    for step in steps:
        description = str(step.get("description", step.get("script", "Unnamed step")))
        command = build_command(step, args, script_dir)
        if not run_command(command, description):
            logging.error("Pipeline failed at %s", description)
            return 1

    logging.info("DSPECT preprocessing pipeline completed successfully")
    if args.skip_validation:
        return 0

    validation_cmd = ["python3", "testing/validate_pipeline.py", "--diagnosis", args.diagnosis]
    if args.is_hazel:
        validation_cmd.append("--isHasel")
    if not run_command(validation_cmd, "ML readiness validation"):
        logging.warning("Validation did not complete successfully")
    return 0


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        sys.exit(run_pipeline(args))
    except Exception as exc:
        logging.error("Pipeline execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
