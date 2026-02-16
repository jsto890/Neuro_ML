#!/usr/bin/env python3
"""Validate DSPECT folder structure for CN and PD cohorts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def count_subject_niftis(subject_dir: Path) -> int:
    """Count NIfTI files in a single subject directory."""
    return sum(1 for p in subject_dir.glob("*.nii.gz") if p.is_file())


def validate_group(group_dir: Path) -> bool:
    """Validate that a group directory exists and contains subject subfolders."""
    if not group_dir.exists():
        logging.error("Missing directory: %s", group_dir)
        return False

    subject_dirs = sorted([p for p in group_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    logging.info("Found %s subject folders in %s", len(subject_dirs), group_dir.name)
    if not subject_dirs:
        return False

    first_subject = subject_dirs[0]
    nifti_count = count_subject_niftis(first_subject)
    logging.info("First subject '%s' has %s NIfTI files", first_subject.name, nifti_count)
    return nifti_count > 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DSPECT directory structure")
    parser.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help="Root folder containing CN_SPECT_PPMI_NIfTI and PD_SPECT_PPMI_NIfTI",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()

    cn_dir = args.base_dir / "CN_SPECT_PPMI_NIfTI"
    pd_dir = args.base_dir / "PD_SPECT_PPMI_NIfTI"

    logging.info("Validating base directory: %s", args.base_dir)
    cn_ok = validate_group(cn_dir)
    pd_ok = validate_group(pd_dir)

    if cn_ok and pd_ok:
        logging.info("Path structure validation passed")
        return 0

    logging.error("Path structure validation failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
