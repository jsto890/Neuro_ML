#!/usr/bin/env python3
"""
Skull-strip & z-score normalize sMRIPrep anatomical outputs (MNI-space only).

For each subject under the given derivatives folder, this script looks for MNI-space files containing:
  • "MNI152NLin2009cAsym" and "_desc-preproc_T1w.nii.gz"   (bias-corrected, reoriented brain in MNI space)
  • "MNI152NLin2009cAsym" and "_desc-brain_mask.nii.gz"    (binary brain mask in MNI space)
It then:
  • deletes any existing *_desc-preproc_T1w_brain_zscore.nii.gz files
  • multiplies the preproc + mask to get a skull-stripped brain volume
  • computes a z-score over all brain voxels
  • writes out *_desc-preproc_T1w_brain_zscore.nii.gz
"""

import os
import argparse
import nibabel as nib
import numpy as np


def find_mni_files(anat_dir, keyword):
    # find files that contain both the MNI tag and the given keyword
    return [os.path.join(anat_dir, f)
            for f in os.listdir(anat_dir)
            if 'MNI152NLin2009cAsym' in f and keyword in f]


def process_folder(anat_dir):
    # check if z-score output already exists
    existing_zscore = None
    for f in os.listdir(anat_dir):
        if f.endswith('_desc-preproc_T1w_brain_zscore.nii.gz'):
            existing_zscore = os.path.join(anat_dir, f)
            break
    
    if existing_zscore:
        print(f"  → Skipped   {existing_zscore} (already exists)")
        return

    # find MNI-space preproc and mask files flexibly
    preproc_list = find_mni_files(anat_dir, '_desc-preproc_T1w.nii.gz')
    mask_list    = find_mni_files(anat_dir, '_desc-brain_mask.nii.gz')
    if not preproc_list or not mask_list:
        print(f"  ! Missing MNI-space files in {anat_dir}")
        return

    preproc_file = preproc_list[0]
    mask_file    = mask_list[0]

    # load images
    img = nib.load(preproc_file)
    data = img.get_fdata(dtype=np.float32)
    mask_data = nib.load(mask_file).get_fdata(dtype=np.float32) > 0

    # skull-strip
    brain = data * mask_data

    # z-score within mask
    brain_vals = brain[mask_data]
    mean = brain_vals.mean()
    std  = brain_vals.std()
    z     = (brain - mean) / std

    # save new z-score file
    base = os.path.basename(preproc_file)
    out_fname = base.replace('_MNI152NLin2009cAsym', '').replace(
        '_desc-preproc_T1w.nii.gz', '_desc-preproc_T1w_brain_zscore.nii.gz'
    )
    out_path = os.path.join(anat_dir, out_fname)

    out_img = nib.Nifti1Image(z.astype(np.float32), img.affine, img.header)
    nib.save(out_img, out_path)
    print(f"  → Saved     {out_path}")


def main(deriv_root):
    print(f"Scanning derivatives root: {deriv_root}")
    for root, dirs, files in os.walk(deriv_root):
        if os.path.basename(root) != 'anat':
            continue
        print(f"\nProcessing: {root}")
        process_folder(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Skull-strip & z-score sMRIPrep T1w anatomical outputs (MNI-space only)'
    )
    parser.add_argument(
        'derivatives_dir',
        help='Top-level sMRIPrep derivatives folder (e.g. derivatives/smriprep)'
    )
    args = parser.parse_args()
    main(os.path.abspath(args.derivatives_dir))
