#!/usr/bin/env python3
"""
Full DaT-SPECT preprocessing pipeline
–––––––––––––––––––––––––––––––––––––
• dcm2niix -i 2  → splits mixed-orientation stacks
• Picks the true reconstructed series (≈60-120 slices, name hints TOMO/RECON/SPECT)
• RAS+ orientation, 2 mm isotropic resample
• Automatic brain crop, pads to 128 × 128 × 96 voxels
• 2nd–98th-percentile scaling → [0, 1]
• Saves    <subject>/<series>_preproc.nii.gz   ready for CNN input
"""

from __future__ import annotations
import subprocess, sys, json, shutil, tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn.image import resample_img
import scipy.ndimage as ndi
import pydicom

# ───────── USER SETTINGS ────────────────────────────────────────────────────────
INPUT_ROOT   = Path("/Users/josephstorey/Desktop/Part_4_Project/data/test_data/dspect/HC/raw")      # one folder per subject
OUTPUT_ROOT  = Path("/Users/josephstorey/Desktop/Part_4_Project/data/test_data/dspect/HC/converted")
TARGET_SHAPE = (128, 128, 96)                  # final (x, y, z)
VOX_MM       = 2.0                             # isotropic voxel size
DCM2NIIX     = "dcm2niix"                      # full path if not in $PATH
KEEP_JSON    = True                            # copy dcm2niix JSON side-car
# ────────────────────────────────────────────────────────────────────────────────


# ---------- helpers ------------------------------------------------------------
def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 1  DICOM → NIfTI (splitting mixed orientations) -------------------------------
def convert_subject(subj_raw: Path, tmp_dir: Path) -> list[Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        DCM2NIIX, "-z", "y", "-f", "%p_%s", "-i", "2",
        "-o", str(tmp_dir), str(subj_raw)
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError as e:
        print(f"dcm2niix failed for {subj_raw.name}: {e}")
        return []
    return list(tmp_dir.glob("*.nii.gz"))


# 2  pick the reconstructed series ---------------------------------------------
HINTS = ("tomo", "recon", "spect", "ac")

def choose_recon(nii_files: list[Path]) -> Path | None:
    def score(p: Path):
        img = nib.load(p)
        z   = img.shape[2]
        hint = any(h in p.name.lower() for h in HINTS)
        return (hint, 60 <= z <= 120, -abs(z - 80))  # tuple sorts by priority
    return max(nii_files, key=score, default=None)


# 3  preprocess single volume ---------------------------------------------------
def preprocess(nii_in: Path, out_dir: Path) -> Path:
    img = nib.as_closest_canonical(nib.load(nii_in))

    # 3.1 isotropic resample
    iso_aff = np.diag([VOX_MM, VOX_MM, VOX_MM, 1])
    img = resample_img(img, target_affine=iso_aff, interpolation="linear")
    data = img.get_fdata().astype(np.float32)

    # 3.2 crude brain mask + bbox crop
    thresh = np.percentile(data, 5)
    mask   = data > thresh
    if not mask.any():
        raise RuntimeError(f"{nii_in.name}: empty mask after thresholding")
    coords = np.array(np.where(mask))
    mins, maxs = coords.min(1), coords.max(1) + 1
    data = data[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    # 3.3 pad / crop to TARGET_SHAPE
    pad = [(0, max(0, TARGET_SHAPE[i] - data.shape[i])) for i in range(3)]
    data = np.pad(data, pad, mode="constant")
    data = data[:TARGET_SHAPE[0], :TARGET_SHAPE[1], :TARGET_SHAPE[2]]

    # 3.4 intensity normalisation [0, 1]
    lo, hi = np.percentile(data, (2, 98))
    data   = np.clip((data - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    out_img  = nib.Nifti1Image(data, np.diag([VOX_MM]*3 + [1]))
    out_path = out_dir / f"{nii_in.stem}_preproc.nii.gz"
    nib.save(out_img, out_path)
    return out_path


# subject-level driver
def process_subject(subj_raw: Path) -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix=subj_raw.name + "_"))
    nii_files = convert_subject(subj_raw, tmp_dir)
    recon     = choose_recon(nii_files)
    if recon is None:
        print(f"No reconstructed series found for {subj_raw.name}")
        shutil.rmtree(tmp_dir)
        return

    subj_out = OUTPUT_ROOT / subj_raw.name
    subj_out.mkdir(parents=True, exist_ok=True)
    out_nif  = preprocess(recon, subj_out)

    if KEEP_JSON:
        json_src = recon.with_suffix("").with_suffix(".json")
        if json_src.exists():
            shutil.copy2(json_src, subj_out / json_src.name)

    shutil.rmtree(tmp_dir)
    print(f"✓ {subj_raw.name} → {out_nif.relative_to(OUTPUT_ROOT)}")


# MAIN
if __name__ == "__main__":
    if not INPUT_ROOT.is_dir():
        sys.exit(f"INPUT_ROOT not found: {INPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    subjects = [p for p in INPUT_ROOT.iterdir() if p.is_dir()]
    if not subjects:
        sys.exit("No subject subdirectories in INPUT_ROOT")

    for subj in subjects:
        process_subject(subj)
