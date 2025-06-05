#!/usr/bin/env python3
"""
standardise_dspect_step2.py

Stage 2 of the DaT-SPECT pipeline: take preprocessed NIfTIs
(*_dspect.nii.gz and optional *_dspect_rescaled.nii.gz) and produce
fully standardized, ML-ready volumes:

 1. Spatial normalization to MNI DaT-SPECT template (FLIRT or Nilearn)
 2. Crop to brain mask bounding box & pad to template shape
 3. Intensity normalization (reference-region or percentile clipping)
 4. Optional Gaussian smoothing

Outputs per subject:
  • <SubjectID>_reg.nii.gz
  • <SubjectID>_crop.nii.gz
  • <SubjectID>_norm.nii.gz
  • <SubjectID>_std.nii.gz
"""

import sys
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from nilearn.image import resample_to_img, resample_img
from nilearn.image.resampling import BoundingBoxError

# ───────── USER SETTINGS ────────────────
INPUT_DIR       = Path('/Users/josephstorey/Desktop/Part_4_Project/data/test_data/dspect/HC/converted')
OUTPUT_DIR      = Path('/Users/josephstorey/Desktop/Part_4_Project/data/processed_data/DSPECT')
TEMPLATE_NIFTI  = Path('/Users/josephstorey/Desktop/Part_4_Project/data/test_data/Templates/DSPECT_refs/symFPCITtemplate_MNI_norm.nii')
BRAIN_MASK      = Path('/Users/josephstorey/Desktop/Part_4_Project/data/test_data/Templates/DSPECT_refs/occipital_mask.nii.gz')
OCCIPITAL_MASK  = Path('/Users/josephstorey/Desktop/Part_4_Project/data/test_data/Templates/DSPECT_refs/occipital_mask.nii.gz')

METHOD           = 'pct'     # 'ref' (reference-region) or 'pct' (percentile clipping)
LO_PERCENTILE    = 0.1       # lower percentile for clipping
HI_PERCENTILE    = 98.0      # upper percentile for clipping
FWHM             = 6.0       # Gaussian smoothing FWHM in mm (0 → skip)

USE_FLIRT        = False     # True to use FSL FLIRT; False to use Nilearn resample
FLIRT_CMD        = '/Users/josephstorey/fsl/bin/flirt'
FLIRT_DOF        = 12        # degrees of freedom for FLIRT
# ────────────────────────────────────────

# Pre-load template and mask
_template       = nib.load(str(TEMPLATE_NIFTI))
_template_aff   = _template.affine
_template_shape = _template.shape
if not BRAIN_MASK.exists():
    sys.exit(f"Brain mask not found: {BRAIN_MASK}")
_mask_data = nib.load(str(BRAIN_MASK)).get_fdata().astype(bool)


def flirt_register(moving: Path, fixed: Path, out: Path):
    """Affine register moving→fixed using FSL FLIRT."""
    mat = out.with_suffix('.mat')
    cmd = [
        FLIRT_CMD,
        '-in', str(moving),
        '-ref', str(fixed),
        '-out', str(out),
        '-omat', str(mat),
        '-dof', str(FLIRT_DOF)
    ]
    subprocess.run(cmd, check=True)


def nib_resample(moving: Path, fixed: Path, out: Path):
    """Resample moving image onto fixed grid with linear interpolation."""
    img_mov = nib.as_closest_canonical(nib.load(str(moving)))
    img_fix = nib.load(str(fixed))
    try:
        res = resample_to_img(img_mov, img_fix, interpolation='linear')
    except BoundingBoxError:
        res = resample_img(
            img_mov,
            target_affine=img_fix.affine,
            target_shape=img_fix.shape,
            interpolation='linear',
            copy_header=True
        )
    nib.save(res, str(out))


def crop_and_pad(in_path: Path, out_path: Path):
    """Crop to the brain‐mask bounding box, then pad symmetrically to template shape."""
    img  = nib.load(str(in_path))
    data = img.get_fdata().astype(np.float32)

    # Crop to mask bounding box
    coords = np.array(np.where(_mask_data))
    mins, maxs = coords.min(1), coords.max(1) + 1
    cropped = data[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    # Compute padding
    pads = []
    for i in range(3):
        total = _template_shape[i] - cropped.shape[i]
        before = total // 2
        after  = total - before
        pads.append((before, after))

    padded = np.pad(cropped, pads, mode='constant')
    out_img = nib.Nifti1Image(padded.astype(np.float32), _template_aff)
    nib.save(out_img, str(out_path))


def intensity_ref_scaling(img_path: Path, mask_path: Path, out_path: Path):
    """Scale by mean within reference mask (e.g., occipital)."""
    img  = nib.load(str(img_path)); data = img.get_fdata().astype(np.float32)
    mask = nib.load(str(mask_path)).get_fdata().astype(bool)
    mean_ref = data[mask].mean() if mask.any() else 1.0
    normed   = data / (mean_ref + 1e-6)
    nib.save(nib.Nifti1Image(normed, img.affine), str(out_path))


def intensity_percentile(img_path: Path, lo: float, hi: float, out_path: Path):
    """Clip intensities to [lo, hi] percentiles and rescale to [0,1]."""
    img  = nib.load(str(img_path)); data = img.get_fdata().astype(np.float32)
    p_lo, p_hi = np.percentile(data, [lo, hi])
    normed     = np.clip((data - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0)
    nib.save(nib.Nifti1Image(normed, img.affine), str(out_path))


def smooth(img_path: Path, fwhm: float, out_path: Path):
    """Apply Gaussian smoothing with specified FWHM (in mm)."""
    img   = nib.load(str(img_path)); data = img.get_fdata().astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    sigma = [fwhm / (2.355 * z) for z in zooms]
    sm    = ndi.gaussian_filter(data, sigma)
    nib.save(nib.Nifti1Image(sm, img.affine), str(out_path))


def standardise_scan(preproc_nii: Path):
    """
    Run stage-2 preprocessing on a single subject NIfTI:
     1) Spatial normalization
     2) Crop & pad
     3) Intensity normalization
     4) Smoothing
    """
    subject = preproc_nii.stem.split('_')[0]
    out_reg  = OUTPUT_DIR / f"{subject}_reg.nii.gz"
    out_crop = OUTPUT_DIR / f"{subject}_crop.nii.gz"
    out_norm = OUTPUT_DIR / f"{subject}_norm.nii.gz"
    out_std  = OUTPUT_DIR / f"{subject}_std.nii.gz"

    print(f"Processing {preproc_nii.name}…")

    # 1) Spatial normalization
    try:
        if USE_FLIRT:
            flirt_register(preproc_nii, TEMPLATE_NIFTI, out_reg)
        else:
            nib_resample(preproc_nii, TEMPLATE_NIFTI, out_reg)
    except FileNotFoundError:
        print("  FLIRT not found; falling back to Nilearn resample", file=sys.stderr)
        nib_resample(preproc_nii, TEMPLATE_NIFTI, out_reg)

    # 2) Crop & pad
    crop_and_pad(out_reg, out_crop)

    # 3) Intensity normalization
    if METHOD == 'ref':
        if not OCCIPITAL_MASK.exists():
            sys.exit(f"Occipital mask not found: {OCCIPITAL_MASK}")
        intensity_ref_scaling(out_crop, OCCIPITAL_MASK, out_norm)
    else:
        intensity_percentile(out_crop, LO_PERCENTILE, HI_PERCENTILE, out_norm)

    # 4) Smoothing
    if FWHM > 0:
        smooth(out_norm, FWHM, out_std)
    else:
        Path(out_norm).rename(out_std)

    print(f"✓ Done → {out_std.name}\n")


def main():
    if not INPUT_DIR.is_dir():
        sys.exit(f"INPUT_DIR not found: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subj_folder in sorted(INPUT_DIR.iterdir()):
        if not subj_folder.is_dir():
            continue
        # find any *_dspect.nii.gz (or *_dspect_rescaled.nii.gz) files
        for nii in sorted(subj_folder.glob('*_dspect*.nii.gz')):
            standardise_scan(nii)


if __name__ == '__main__':
    main()
