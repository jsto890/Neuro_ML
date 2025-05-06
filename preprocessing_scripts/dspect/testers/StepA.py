#!/usr/bin/env python3
"""
batch_convert_dspect.py

Batch-convert DaT-SPECT DICOM folders → NIfTI and rescale, producing:
  • <SubjectID>_dspect.nii.gz
  • <SubjectID>_dspect_rescaled.nii.gz   (if APPLY_RESCALE=True and JSON side-car present)

Usage:
  python batch_convert_dspect.py

Edit the USER SETTINGS below to point at your data.
"""

import subprocess
import sys
import json
from pathlib import Path

import nibabel as nib
from nibabel import Nifti1Image
import numpy as np

# ───────── USER SETTINGS ────────────────
INPUT_DIR      = Path('/Users/josephstorey/Desktop/Part_4_Project/data/test_data/dspect/HC/raw')
OUTPUT_DIR     = Path('/Users/josephstorey/Desktop/Part_4_Project/data/test_data/dspect/HC/converted')
DCM2NIIX_CMD   = 'dcm2niix'
COMPRESS       = 'y'      # 'y' for gzip, 'n' for no compression
IGNORE_DERIVED = 1        # 0=no, 1=skip localisers, 2=verbose
REORIENT       = True     # reorient to RAS canonical
APPLY_RESCALE  = True     # apply DICOM RescaleSlope/Intercept from JSON
KEEP_JSON      = True     # copy JSON side-car alongside NIfTI
# ────────────────────────────────────────


def run_dcm2niix(dicom_dir: Path, out_dir: Path, subj_id: str) -> Path:
    """
    Run dcm2niix on the DICOM directory to produce a single NIfTI.
    Returns the path to the generated .nii.gz.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{subj_id}_dspect"
    cmd = [
        DCM2NIIX_CMD,
        '-z', COMPRESS,
        '-i', str(IGNORE_DERIVED),
        '-f', fname,
        '-o', str(out_dir),
        str(dicom_dir)
    ]
    subprocess.run(cmd, check=True)
    return out_dir / f"{fname}.nii.gz"


def pure_python_convert(dicom_dir: Path, out_dir: Path, subj_id: str) -> Path:
    """
    Fallback: read all .dcm files, stack by InstanceNumber, and write NIfTI.
    """
    import pydicom
    files = sorted(
        dicom_dir.glob('*.dcm'),
        key=lambda f: int(getattr(pydicom.dcmread(f, stop_before_pixels=True),
                                  'InstanceNumber', 0))
    )
    if not files:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")
    frames = [pydicom.dcmread(f) for f in files]
    data = np.stack([f.pixel_array for f in frames]).astype(np.float32)
    px, py = frames[0].PixelSpacing
    pz     = frames[0].SliceThickness
    affine = np.diag([px, py, pz, 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subj_id}_dspect.nii.gz"
    img = Nifti1Image(data, affine)
    nib.save(img, str(out_path))
    return out_path


def reorient_to_ras(nifti_path: Path) -> None:
    """
    Reorient the NIfTI in-place to RAS using nibabel.
    """
    img = nib.load(str(nifti_path))
    ras = nib.as_closest_canonical(img)
    nib.save(ras, str(nifti_path))


def apply_rescale(nifti_path: Path) -> Path:
    """
    If a JSON side-car exists, read RescaleSlope/Intercept and apply to data,
    writing a new file with suffix '_rescaled.nii.gz'.
    Returns the path to the rescaled image (or original if no JSON).
    """
    json_path = nifti_path.with_suffix('').with_suffix('.json')
    if not json_path.exists():
        return nifti_path

    meta = json.loads(json_path.read_text())
    slope = float(meta.get('RescaleSlope', 1.0))
    icpt  = float(meta.get('RescaleIntercept', 0.0))

    orig = nib.load(str(nifti_path))
    data = orig.get_fdata(dtype=np.float32) * slope + icpt
    rescaled_img = Nifti1Image(data, orig.affine)

    out_path = nifti_path.with_name(f"{nifti_path.stem}_rescaled.nii.gz")
    nib.save(rescaled_img, str(out_path))
    return out_path


def process_subject(subj_dir: Path) -> None:
    subj_id = subj_dir.name
    dest_dir = OUTPUT_DIR / subj_id
    print(f"\n▶ Processing subject {subj_id}")

    # Step 1: DICOM → NIfTI
    try:
        niipath = run_dcm2niix(subj_dir, dest_dir, subj_id)
    except subprocess.CalledProcessError:
        print("  dcm2niix failed, using pure-Python fallback...")
        niipath = pure_python_convert(subj_dir, dest_dir, subj_id)

    # Step 2: Reorient
    if REORIENT:
        reorient_to_ras(niipath)

    # Step 3: Copy JSON side-car if desired
    if KEEP_JSON:
        json_src = niipath.with_suffix('').with_suffix('.json')
        if json_src.exists():
            (dest_dir / json_src.name).write_bytes(json_src.read_bytes())

    # Step 4: Apply rescale
    if APPLY_RESCALE:
        rescaled = apply_rescale(niipath)
        print(f"  ✔ Wrote rescaled image: {rescaled.name}")
    else:
        print(f"  ✔ Wrote image: {niipath.name}")

    print(f"✓ Finished {subj_id}")


def main():
    if not INPUT_DIR.is_dir():
        sys.exit(f"INPUT_DIR not found: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    if not subjects:
        sys.exit("No subject directories in INPUT_DIR")

    for subj in sorted(subjects):
        process_subject(subj)


if __name__ == '__main__':
    main()
