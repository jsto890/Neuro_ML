#!/usr/bin/env python3

"""
python3 refine_mask_apply.py \
  --suvr /home/jsto890/reseng202500013-ndd-ml/data/preprocessed/NEWPET/ADNI/sub-I10938763_ADNI_PET_AD/sub-I10938763_ADNI_PET_AD_SUVR_s2.nii.gz \
  --mask /home/jsto890/reseng202500013-ndd-ml/P4P/Templates/PET/FDG_PET_brainmask.nii.gz \
  --fwhm 2 --morph_iters 1
  
  Notes:
Hard output *_brain.nii.gz is best for analysis; soft output *_brain_soft2.nii.gz looks cleaner for visualisation (slightly tapered boundary).
If edges still protrude, increase --morph_iters to 2 or raise --fwhm to 3 mm.

"""
import argparse, numpy as np, nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes, gaussian_filter, generate_binary_structure

def fwhm_to_sigma_vox(fwhm_mm, vx_mm):
    return (fwhm_mm / 2.355) / vx_mm

p = argparse.ArgumentParser()
p.add_argument("--suvr", required=True)        # e.g. ..._SUVR_s2.nii.gz
p.add_argument("--mask", required=True)        # e.g. Templates/PET/FDG_PET_brainmask.nii.gz
p.add_argument("--fwhm", type=float, default=2.0, help="Mask blur FWHM mm (0 to disable)")
p.add_argument("--morph_iters", type=int, default=1, help="Morphology iters (closing+opening)")
p.add_argument("--out_suffix", default="_brain", help="Suffix for outputs")
args = p.parse_args()

# Load
img = nib.load(args.suvr)
suvr = img.get_fdata().astype(np.float32)
mask_img = nib.load(args.mask)
if img.shape != mask_img.shape or not np.allclose(img.affine, mask_img.affine):
    mask_img = resample_from_to(mask_img, img, order=0)

# Binary mask cleanup
mask = (mask_img.get_fdata() > 0)
st = generate_binary_structure(3, 1)
for _ in range(max(args.morph_iters, 0)):
    mask = binary_closing(mask, structure=st)
    mask = binary_opening(mask, structure=st)
mask = binary_fill_holes(mask)

# Optional soft mask via Gaussian on mask
soft = mask.astype(np.float32)
if args.fwhm > 0:
    vx = np.abs(np.diag(img.affine))[:3]
    sig = fwhm_to_sigma_vox(args.fwhm, vx)
    soft = gaussian_filter(soft, sigma=sig[::-1])
    soft = np.clip(soft, 0.0, 1.0)

# Save refined masks
nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header), args.suvr.replace(".nii.gz", f"_maskRefined.nii.gz"))
nib.save(nib.Nifti1Image(soft.astype(np.float32), img.affine, img.header), args.suvr.replace(".nii.gz", f"_maskSoft{int(round(args.fwhm)) if args.fwhm>0 else 0}.nii.gz"))

# Apply masks
hard_out = args.suvr.replace(".nii.gz", f"{args.out_suffix}.nii.gz")
soft_out = args.suvr.replace(".nii.gz", f"{args.out_suffix}_soft{int(round(args.fwhm)) if args.fwhm>0 else 0}.nii.gz")
nib.save(nib.Nifti1Image(suvr * mask.astype(np.float32), img.affine, img.header), hard_out)
nib.save(nib.Nifti1Image(suvr * soft, img.affine, img.header), soft_out)
print("Wrote:", hard_out, soft_out)