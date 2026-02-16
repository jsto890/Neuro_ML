#!/usr/bin/env python3
"""Validate DICOM readability and directory content."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pydicom


def test_dicom_access(dicom_path: Path) -> bool:
    """Return True if DICOM file exists and can be read."""
    logging.info("Testing DICOM file: %s", dicom_path)
    if not dicom_path.exists():
        logging.error("DICOM file not found")
        return False

    try:
        ds = pydicom.dcmread(dicom_path)
        logging.info("DICOM read successful")
        logging.info("Patient ID: %s", getattr(ds, "PatientID", "Unknown"))
        logging.info("Modality: %s", getattr(ds, "Modality", "Unknown"))
        return True
    except Exception as exc:  # pragma: no cover - depends on local file validity
        logging.error("Error reading DICOM file: %s", exc)
        return False


def test_folder_structure(base_path: Path) -> bool:
    """Return True if folder exists and contains at least one DICOM file."""
    logging.info("Testing folder: %s", base_path)
    if not base_path.exists():
        logging.error("Base folder not found")
        return False

    dicom_count = sum(1 for _ in base_path.rglob("*.dcm"))
    logging.info("Found %s DICOM files", dicom_count)
    return dicom_count > 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DICOM source inputs")
    parser.add_argument("--dicom_path", type=Path, required=True, help="Path to a sample DICOM file")
    parser.add_argument("--base_path", type=Path, required=True, help="Path to source DICOM folder")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()

    file_ok = test_dicom_access(args.dicom_path)
    folder_ok = test_folder_structure(args.base_path)

    if file_ok and folder_ok:
        logging.info("All checks passed")
        return 0

    logging.error("One or more checks failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
