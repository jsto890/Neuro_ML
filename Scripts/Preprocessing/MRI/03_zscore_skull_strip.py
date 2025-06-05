#!/usr/bin/env python3
"""
Skull-strip & z-score normalize sMRIPrep anatomical outputs.

For each subject under the given derivatives folder, this script looks for:
  *_desc-preproc_T1w.nii.gz   (bias-corrected, reoriented “whole-head”)
  *_desc-brain_mask.nii.gz    (binary brain mask)
It then:
  • multiplies them to get a skull-stripped brain volume
  • computes a z-score over all brain voxels
  • writes out *_desc-preproc_T1w_brain_zscore.nii.gz
"""

import os
import glob
import argparse

import nibabel as nib
import numpy as np

def process_folder(anat_dir):
    # find the two files
    preproc = glob.glob(os.path.join(anat_dir, "*_desc-preproc_T1w.nii.gz"))
    mask   = glob.glob(os.path.join(anat_dir, "*_desc-brain_mask.nii.gz"))
    if not preproc or not mask:
        return

    preproc_file = preproc[0]
    mask_file    = mask[0]

    # load images
    img    = nib.load(preproc_file)
    data   = img.get_fdata(dtype=np.float32)
    mask_data = nib.load(mask_file).get_fdata(dtype=np.float32) > 0

    # skull-strip
    brain = data * mask_data

    # z-score within mask
    brain_vals = brain[mask_data]
    mean = brain_vals.mean()
    std  = brain_vals.std()
    z     = (brain - mean) / std

    # save
    out_fname = os.path.basename(preproc_file).replace(
        ".nii.gz",
        "_brain_zscore.nii.gz"
    )
    out_path = os.path.join(anat_dir, out_fname)

    # preserve header + affine, but store float32
    out_img = nib.Nifti1Image(z.astype(np.float32), img.affine, img.header)
    nib.save(out_img, out_path)
    print(f"  → Saved   {out_path}")


def main(deriv_root):
    print(f"Scanning derivatives root: {deriv_root}")
    # look for any “anat” directories
    for root, dirs, files in os.walk(deriv_root):
        if os.path.basename(root) != "anat":
            continue
        print(f"\nProcessing: {root}")
        process_folder(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Skull-strip & z-score sMRIPrep T1w anatomical outputs"
    )
    p.add_argument(
        "derivatives_dir",
        help="Top-level sMRIPrep derivatives folder (e.g. derivatives/smriprep)"
    )
    args = p.parse_args()
    main(os.path.abspath(args.derivatives_dir))