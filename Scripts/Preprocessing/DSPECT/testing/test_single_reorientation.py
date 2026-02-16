#!/usr/bin/env python3
"""Run and validate reorientation for a single NIfTI image."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np


TARGET_ORIENTATION = ("R", "A", "S")


def reorient_to_ras(nifti_path: Path, output_path: Path) -> None:
    """Reorient a NIfTI image to canonical RAS and save it."""
    img = nib.load(str(nifti_path))
    reoriented_img = nib.as_closest_canonical(img)
    nib.save(reoriented_img, str(output_path))


def validate_reorientation(raw_path: Path, reoriented_path: Path) -> bool:
    """Check orientation and approximate voxel integrity after reorientation."""
    raw_img = nib.load(str(raw_path))
    test_img = nib.load(str(reoriented_path))

    raw_data = raw_img.get_fdata()
    test_data = test_img.get_fdata()

    raw_ornt = nib.orientations.aff2axcodes(raw_img.affine)
    test_ornt = nib.orientations.aff2axcodes(test_img.affine)

    logging.info("Raw orientation: %s", raw_ornt)
    logging.info("Reoriented orientation: %s", test_ornt)

    if test_ornt != TARGET_ORIENTATION:
        logging.error("Expected orientation %s but got %s", TARGET_ORIENTATION, test_ornt)
        return False

    raw_nonzero = np.count_nonzero(raw_data)
    test_nonzero = np.count_nonzero(test_data)
    delta = abs(raw_nonzero - test_nonzero)
    logging.info("Non-zero voxel delta: %s", delta)

    if delta >= 100:
        logging.error("Voxel integrity check failed")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate single-image reorientation")
    parser.add_argument("--input", type=Path, required=True, help="Input NIfTI path")
    parser.add_argument("--output", type=Path, required=True, help="Output NIfTI path")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()

    reorient_to_ras(args.input, args.output)
    if validate_reorientation(args.input, args.output):
        logging.info("Reorientation validation passed")
        return 0

    logging.error("Reorientation validation failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
