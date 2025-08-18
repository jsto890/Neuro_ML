#!/usr/bin/env python3

"""
Unified skull-strip + brain-median renormalization for PET SUVR.

Features:
- Accept SUVR image and a brain mask (subject MNI brain)
- Resample mask to SUVR grid (nearest-neighbor)
- Morphological cleanup (closing+opening, fill holes, optional erosion)
- Optional soft mask via Gaussian blur (FWHM in mm) for smoother edges
- Apply hard and soft masking
- Optional re-normalize within-brain so the brain-masked median equals 1.0

Example:
python3 Scripts/Preprocessing/PET/03_skullstrip.py \
  --suvr /path/to/sub-XXX_SUVR_s2.nii.gz \
  --mask /path/to/FDG_PET_brainmask.nii.gz \
  --morph_iters 1 --erode_iters 1 --fwhm 2 --renorm_median

Outputs written next to the SUVR file (or --out_dir if provided):
_maskRefined.nii.gz: binary brain mask (uint8, 0/1). It's your input mask 
    resampled to the SUVR grid, cleaned by morphology (closing→opening), optional 
        1‑voxel erosion, and hole filling. Use for hard skull‑strip or QC.

_maskSoft2.nii.gz: soft brain mask (float32, 0–1). Gaussian‑blurred version 
    of the refined mask (2 mm FWHM; number matches your --fwhm). Good for visualisation or 
        gentle edge tapering.

_SUVR_s2_brain.nii.gz: hard skull‑stripped SUVR image. Equals SUVR_s2 multiplied 
    by the refined binary mask. Best for analysis/workflows expecting sharp brain boundaries. 

_SUVR_s2_brain_soft2.nii.gz: soft skull‑stripped SUVR. SUVR_s2 multiplied by the 
    soft mask. Looks cleaner at edges; use mainly for visualisation or if you want 
        slight boundary tapering.

_SUVR_s2_brain_med1.nii.gz (only with --renorm_median): brain‑masked SUVR re‑normalized 
    so the within‑brain median equals 1.0. This removes residual global scaling bias 
        inside brain after masking.

"""

import argparse
import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import (
	binary_closing,
	binary_opening,
	binary_fill_holes,
	binary_erosion,
	gaussian_filter,
	generate_binary_structure,
)


def fwhm_to_sigma_vox(fwhm_mm: float, vx_mm: np.ndarray) -> np.ndarray:
	return (fwhm_mm / 2.355) / vx_mm


def out_path(suvr_path: str, suffix: str, out_dir: str | None) -> str:
	base = os.path.basename(suvr_path)
	if base.endswith(".nii.gz"):
		stub = base[:-7]
	elif base.endswith(".nii"):
		stub = base[:-4]
	else:
		stub = base
	d = out_dir if out_dir else os.path.dirname(suvr_path)
	return os.path.join(d, f"{stub}{suffix}.nii.gz")


def main():
	parser = argparse.ArgumentParser(description="Skull-strip SUVR using refined brain mask and optional median renorm")
	parser.add_argument("--suvr", required=True, help="Path to SUVR image (e.g., *_SUVR_s2.nii.gz)")
	parser.add_argument("--mask", required=True, help="Path to brain mask image (subject MNI brain)")
	parser.add_argument("--out_dir", default=None, help="Optional output directory (defaults to SUVR directory)")
	parser.add_argument("--morph_iters", type=int, default=1, help="Morphology iterations (closing+opening)")
	parser.add_argument("--erode_iters", type=int, default=1, help="Additional erosion iterations after cleanup")
	parser.add_argument("--fwhm", type=float, default=2.0, help="Soft-mask blur FWHM mm (0 disables)")
	parser.add_argument("--renorm_median", action="store_true", help="Re-normalize brain voxels so median==1.0")
	parser.add_argument("--out_suffix", default="_brain", help="Suffix for masked outputs")
	args = parser.parse_args()

	img = nib.load(args.suvr)
	suvr = img.get_fdata().astype(np.float32)

	mask_img = nib.load(args.mask)
	if img.shape != mask_img.shape or not np.allclose(img.affine, mask_img.affine):
		mask_img = resample_from_to(mask_img, img, order=0)

	# Binary mask + cleanup
	mask = (mask_img.get_fdata() > 0)
	st = generate_binary_structure(3, 1)
	for _ in range(max(args.morph_iters, 0)):
		mask = binary_closing(mask, structure=st)
		mask = binary_opening(mask, structure=st)
	if args.erode_iters and args.erode_iters > 0:
		mask = binary_erosion(mask, iterations=args.erode_iters)
	mask = binary_fill_holes(mask)

	# Soft mask for visuals
	soft = mask.astype(np.float32)
	if args.fwhm and args.fwhm > 0:
		vx = np.abs(np.diag(img.affine))[:3]
		sig = fwhm_to_sigma_vox(args.fwhm, vx)
		soft = gaussian_filter(soft, sigma=sig[::-1])
		soft = np.clip(soft, 0.0, 1.0)

	# Apply masks
	hard = suvr * mask.astype(np.float32)
	soft_applied = suvr * soft

	# Optional re-normalization inside brain
	if args.renorm_median:
		vals = hard[mask]
		med = float(np.median(vals)) if vals.size else 1.0
		hard = hard / (med + 1e-8)

	# Write outputs
	refined_mask_path = out_path(args.suvr, "_maskRefined", args.out_dir)
	soft_mask_path = out_path(args.suvr, f"_maskSoft{int(round(args.fwhm)) if args.fwhm>0 else 0}", args.out_dir)
	hard_out = out_path(args.suvr, f"{args.out_suffix}", args.out_dir)
	soft_out = out_path(args.suvr, f"{args.out_suffix}_soft{int(round(args.fwhm)) if args.fwhm>0 else 0}", args.out_dir)
	nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header), refined_mask_path)
	nib.save(nib.Nifti1Image(soft.astype(np.float32), img.affine, img.header), soft_mask_path)
	nib.save(nib.Nifti1Image(hard.astype(np.float32), img.affine, img.header), hard_out)
	nib.save(nib.Nifti1Image(soft_applied.astype(np.float32), img.affine, img.header), soft_out)
	print("Wrote:", refined_mask_path, soft_mask_path, hard_out, soft_out)


if __name__ == "__main__":
	main()