#!/usr/bin/env python3
"""
batch_convert_pet_dcm.py

Recursively converts all PET DICOM folders into NIfTI+JSON
using dcm2niix, renaming each output folder as subject_{n}_PPMI_converted.

Requirements:
  - Python 3.6+
  - dcm2niix installed and on your PATH

Usage:
  python batch_convert_pet_dcm.py
"""

import subprocess
from pathlib import Path
import logging

# Configuration
RAW_ROOT       = Path("/Users/josephstorey/Desktop/Part_4_Project/data/test_data/pet/AD/raw")
CONVERTED_ROOT = Path("/Users/josephstorey/Desktop/Part_4_Project/data/test_data/pet/AD/converted")
PREFIX         = "subject"
SUFFIX         = "PPMI_converted"


def convert_folder(dicom_folder: Path, out_folder: Path):
    """
    Run dcm2niix on dicom_folder, writing into out_folder.
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Converting {dicom_folder} → {out_folder}")
    try:
        subprocess.run([
            "dcm2niix",
            "-b", "y",          # write sidecar JSON
            "-z", "y",          # gzip output
            "-o", str(out_folder),
            str(dicom_folder)
        ], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting {dicom_folder}: {e}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    if not RAW_ROOT.exists():
        logging.error(f"Raw directory not found: {RAW_ROOT}")
        return
    CONVERTED_ROOT.mkdir(parents=True, exist_ok=True)

    for idx, sub in enumerate(sorted(RAW_ROOT.iterdir()), start=1):
        if not sub.is_dir():
            continue
        new_name = f"{PREFIX}_{idx}_{SUFFIX}"
        out_sub = CONVERTED_ROOT / new_name
        convert_folder(sub, out_sub)

    logging.info("All conversions complete.")
    logging.info(f"Converted folders under: {CONVERTED_ROOT}")


if __name__ == "__main__":
    main()
