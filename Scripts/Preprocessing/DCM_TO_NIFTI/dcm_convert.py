#!/usr/bin/env python3

import os
import re

import subprocess
from pathlib import Path
import pydicom
from datetime import datetime
import shutil


# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASETS = {
    "ADNI": Path("~/reseng202500013-ndd-ml/data/raw/PET/ADNI/AD/ADNI_PET_AD"),
}
DEST_ROOT = Path("~/reseng202500013-ndd-ml/data/raw")
DICOM_EXT  = ".dcm"
# Regex to catch “site_modality_diagnosis” (each part must start with a letter) or with “_1” suffix
SMD_RE     = re.compile(r"^([A-Za-z][^_]*)_([A-Za-z][^_]*)_([A-Za-z][^_]*)(?:_\d+)?$")
# Optional one-off overrides via environment variables
#   P4P_FORCED_SMD: e.g. "ADNI_PET_AD" (forces label instead of inferring)
#   P4P_FLATTEN: set to 1/true/y to place outputs directly under DEST_ROOT/<subject>_<SMD>
#   P4P_DEST_ROOT: override destination root path
FORCED_SMD = os.getenv("P4P_FORCED_SMD")
FLATTEN_OUTPUT = os.getenv("P4P_FLATTEN", "0").lower() in ("1", "y", "yes", "true")
# Provide a way to pass custom dcm2niix flags (space-separated) via env var
_extra_flags = os.getenv("P4P_DCM2NIIX_FLAGS", "").strip()
DCM2NIIX_EXTRA_FLAGS = _extra_flags.split() if _extra_flags else []
OVERWRITE = os.getenv("P4P_OVERWRITE", "0").lower() in ("1", "y", "yes", "true")
NAME_SUFFIX = os.getenv("P4P_NAME_SUFFIX", "")  # e.g., "_%s" to append series number
# ────────────────────────────────────────────────────────────────────────────────

# Expand user (~) in configured paths
DATASETS = {k: Path(str(v)).expanduser() for k, v in DATASETS.items()}
DEST_ROOT = Path(os.getenv("P4P_DEST_ROOT", str(DEST_ROOT))).expanduser()

def find_subject_dirs(root: Path):
    """Yield any folder that contains DICOM files directly under it."""
    for d in root.rglob("*"):
        if d.is_dir() and any(p.suffix.lower() == DICOM_EXT for p in d.iterdir()):
            yield d

def extract_smd(folder: Path):
    """
    Walk up from `folder` until we find a parent whose name matches SMD_RE.
    Returns (site, modality, diagnosis).
    """
    # Prefer higher-level semantic labels (e.g., ADNI_PET_AD) over subject-like labels (e.g., 022_S_0543)
    for anc in reversed(folder.parents):
        m = SMD_RE.match(anc.name)
        if m:
            return m.group(1), m.group(2), m.group(3)
    raise RuntimeError(f"Could not find site_modality_diagnosis for {folder}")

def derive_subject_id(dicom_leaf_dir: Path) -> str:
    """Return the immediate folder name that contains the DICOM files (e.g., I334249)."""
    return dicom_leaf_dir.name

def convert_dicom(dicom_dir: Path, out_dir: Path, prefix: str) -> bool:
    """Run dcm2niix to convert and name the outputs with our prefix.
    Returns True if conversion succeeded, False otherwise.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # PET-friendly defaults: merge slices, do not crop, allow reorient
    # Effective filename template
    filename_template = prefix + NAME_SUFFIX

    cmd = [
        "dcm2niix",
        "-b", "y",   # write JSON sidecar
        "-z", "y",   # gzip
        "-m", "y",   # merge 2D slices/frames into 3D
        "-x", "n",   # do NOT crop
        "-w", "1" if OVERWRITE else "0",  # overwrite vs skip duplicates
        "-f", filename_template,  # filename template
        "-o", str(out_dir),
    ] + DCM2NIIX_EXTRA_FLAGS + [str(dicom_dir)]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] dcm2niix failed (rc={e.returncode}) for {dicom_dir}. Trying fallback with -i y (ignore derived/localizer).")
        fallback_cmd = [
            "dcm2niix",
            "-b", "y",
            "-z", "y",
            "-m", "y",
            "-x", "n",
            "-w", "1" if OVERWRITE else "0",
            "-i", "y",  # ignore derived/localizer/2D images
            "-f", filename_template,
            "-o", str(out_dir),
        ] + DCM2NIIX_EXTRA_FLAGS + [str(dicom_dir)]
        try:
            subprocess.run(fallback_cmd, check=True)
            return True
        except subprocess.CalledProcessError as e2:
            # Log and skip this subject
            try:
                (out_dir / "conversion_failed.txt").write_text(
                    f"Primary rc: {e.returncode}\nFallback rc: {e2.returncode}\nDirectory: {dicom_dir}\n"
                )
            except Exception:
                pass
            print(f"[!] FATAL: dcm2niix failed for {dicom_dir}. Skipping.")
            return False

def create_output_structure(base_output_path, subject_id, scan_date, modality):
    """Create organised output folder structure"""
    # Create main subject folder
    subject_folder = base_output_path / f"Subject_{subject_id}"
    subject_folder.mkdir(parents=True, exist_ok=True)
    
    # Create scan-specific folder
    scan_folder = subject_folder / f"Scan_{scan_date}_{modality}"
    scan_folder.mkdir(parents=True, exist_ok=True)
    
    return scan_folder


        for subj_dir in find_subject_dirs(ds_root):
            if FORCED_SMD:
                parts = FORCED_SMD.split("_")
                if len(parts) == 3:
                    site, modality, diagnosis = parts
                else:
                    site, modality, diagnosis = extract_smd(subj_dir)
            else:
                site, modality, diagnosis = extract_smd(subj_dir)

            subject_id = derive_subject_id(subj_dir)

            if FLATTEN_OUTPUT:
                out_prefix = f"sub-{subject_id}_{site}_{modality}_{diagnosis}"
                dest_folder = DEST_ROOT / out_prefix
                if dest_folder.exists() and not OVERWRITE:
                    print(f"[skip] Exists: {dest_folder}")
                    continue
            else:
                out_prefix = f"sub-{subject_id}_{site}_{modality}_{diagnosis}"
                dest_folder = (
                    DEST_ROOT
                    / modality
                    / site
                    / diagnosis
                    / out_prefix
                )
                if dest_folder.exists() and not OVERWRITE:
                    print(f"[skip] Exists: {dest_folder}")
                    continue

            print(f"Converting {ds_name}/{subj_dir.relative_to(ds_root)} → {dest_folder}")
            convert_dicom(subj_dir, dest_folder, out_prefix)

    print("All done.")


if __name__ == "__main__":
    main()
