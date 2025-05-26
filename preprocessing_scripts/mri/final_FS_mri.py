#!/usr/bin/env python3
# =============================================================================
# Final_FS.py — sMRI Preprocessing Pipeline
# =============================================================================
# Author       : Joseph Storey
# Affiliation  : University of Auckland, Biomedical Engineering
# Date         : April 28, 2025
# Version      : 1.0
#
# Description  :
#   This script implements a full preprocessing pipeline for T1-weighted MRI volumes:
#     1) Orientation standardisation (to RAS)
#     2) FreeSurfer recon-all skull-stripping
#     3) Brain mask application
#     4) N4 bias-field correction (SimpleITK)
#     5) Isotropic resampling (1×1×1 mm; B-spline)
#     6) Optional ANTs SyN registration to MNI152 template
#     7) Intensity z-scoring
#     8) Central crop & pad to template dimensions
#
# Usage        :
#   python Final_FS.py \
#     --raw_dir      /path/to/raw_niftis \
#     --subjects_dir /path/to/FS_subjects \
#     --final_dir    /path/to/processed_output \
#     --template     /path/to/MNI152_T1_1mm_brain.nii.gz
#
# Dependencies :
#   • Python 3.8+  
#   • FreeSurfer 8.0.0  
#   • ANTsPy  
#   • nibabel  
#   • SimpleITK  
#   • numpy  
#   • tqdm  

# Import standard libraries for OS operations, subprocess calls, logging, system parameters, and temp files
import os  # filesystem operations
import subprocess  # run external commands
import logging  # log pipeline progress and errors
import sys  # access system-specific parameters and functions
import tempfile  # create temporary files
from fs_nipype import fs_recon_all_nipype

# Import imaging libraries
import ants as ants  # ANTsPy for advanced image registration
import nibabel as nib  # nibabel for reading/writing neuroimaging formats
import nibabel.orientations as nio  # handle image orientation transformations
import numpy as np  # numerical array operations
import SimpleITK as sitk  # SimpleITK for image processing (e.g., N4 bias correction)
from tqdm import tqdm  # progress bar for iterables

# ============================
# CONFIGURATION
# ============================
# Define paths for tools and data directories
FREESURFER_HOME = "/home/jsto890/freesurfer-8.0.0/8.0.0"  # FreeSurfer installation directory
SUBJECTS_DIR   = "/home/jsto890/reseng202500013-ndd-ml/data/raw/MRI/FS_Staged"  # where FreeSurfer subjects live
RAW_DIR        = "/home/jsto890/reseng202500013-ndd-ml/data/raw/MRI"  # raw NIfTI inputs
FINAL_DIR      = "/home/jsto890/reseng202500013-ndd-ml/data/preprocessed/MRI"  # output directory for processed images
LOG_FILE       = os.path.join(FINAL_DIR, "pipeline.log")  # pipeline log file
mni_template_path = "/home/jsto890/reseng202500013-ndd-ml/P4P/Templates/MRI_refs/MNI152_T1_1mm_brain.nii.gz"  # MNI template

# Ensure required directories exist (will create if missing)
for d in (SUBJECTS_DIR, RAW_DIR, FINAL_DIR):
    os.makedirs(d, exist_ok=True)

# ============================
# SET UP LOGGING
# ============================
logging.basicConfig(
    level=logging.INFO,  # log info-level and above
    format="%(asctime)s %(levelname)s: %(message)s",  # include timestamp and level
    handlers=[
        logging.FileHandler(LOG_FILE),  # write logs to file
        logging.StreamHandler(sys.stdout)  # also output logs to console
    ]
)

# ============================
# POST-PROCESSING FUNCTIONS
# ============================
def apply_mask(nib_img, mask_img):
    """
    Apply a binary brain mask to the image data:
    - Zero out voxels where mask == 0
    """
    data = nib_img.get_fdata()  # volume data
    mask = mask_img.get_fdata() > 0  # boolean mask
    return nib.Nifti1Image(data * mask, nib_img.affine)


def standardise_orientation(nib_img, axcodes=('R', 'A', 'S')):
    """
    Reorient image to a standard axis coding (e.g., RAS):
    - Compute transform from original to target orientation
    - Apply transform to data and affine
    """
    orig_ornt = nio.io_orientation(nib_img.affine)
    targ_ornt = nio.axcodes2ornt(axcodes)
    transform = nio.ornt_transform(orig_ornt, targ_ornt)
    data = nio.apply_orientation(nib_img.get_fdata(), transform)
    affine = nib_img.affine @ nio.inv_ornt_aff(transform, nib_img.shape)
    return nib.Nifti1Image(data, affine)


def bias_field_correction(nib_img):
    """
    Perform N4 bias field correction using SimpleITK:
    - Transpose data to z,y,x ordering
    - Run N4 filter
    - Transpose corrected data back
    """
    arr = np.transpose(nib_img.get_fdata(), (2, 1, 0))  # to z,y,x
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_img = corrector.Execute(sitk_img)
    corrected_arr = sitk.GetArrayFromImage(corrected_img)
    corrected_arr = np.transpose(corrected_arr, (2, 1, 0))  # back to x,y,z
    return nib.Nifti1Image(corrected_arr, nib_img.affine)


def spatial_normalise(nib_img, template_img):
    """
    Register the image to a template using ANTs' Symmetric Normalization (SyN):
    - Convert nibabel image to ANTs format
    - Perform SyN registration
    - Return warped image in template space
    """
    img_ants = ants.from_numpy(nib_img.get_fdata(), affine=nib_img.affine)
    template_ants = ants.image_read(template_img)
    reg = ants.registration(fixed=template_ants, moving=img_ants, type_of_transform='syN')
    norm_img = reg['warpedmovout']
    return nib.Nifti1Image(norm_img.numpy(), template_ants.affine)


def resample_to_iso(nib_img, spacing=(1,1,1)):
    """
    Resample image to isotropic voxel spacing:
    - Save nibabel image to temp file
    - Read with SimpleITK
    - Compute new size for desired spacing
    - Resample using B-spline interpolation
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(nib_img, tmp)
    sitk_img = sitk.ReadImage(tmp)
    os.remove(tmp)

    orig_sp, orig_sz = sitk_img.GetSpacing(), sitk_img.GetSize()
    new_sz = [int(round(o*s/p)) for o, s, p in zip(orig_sz, orig_sp, spacing)]
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing(spacing)
    rf.SetSize(new_sz)
    rf.SetOutputOrigin(sitk_img.GetOrigin())
    rf.SetOutputDirection(sitk_img.GetDirection())
    rf.SetInterpolator(sitk.sitkBSpline)
    out_img = rf.Execute(sitk_img)

    arr = sitk.GetArrayFromImage(out_img)  # z,y,x
    arr = np.transpose(arr, (2, 1, 0))  # x,y,z
    return nib.Nifti1Image(arr, nib_img.affine)


def zscore(nib_img):
    """
    Z-score intensity normalization:
    - Compute mean and std of non-zero voxels
    - Subtract mean and divide by std
    """
    data = nib_img.get_fdata()
    mask = data != 0
    m, s = data[mask].mean(), data[mask].std()
    return nib.Nifti1Image((data - m) / s, nib_img.affine)


def center_crop_pad(nib_img, shape=(160,192,192)):
    """
    Center-crop or pad image to the given shape:
    - Crop if larger, pad with zeros if smaller
    """
    data = nib_img.get_fdata()
    for i in range(3):
        delta = data.shape[i] - shape[i]
        if delta > 0:
            # Crop centrally
            start = delta // 2
            sl = [slice(None)]*3
            sl[i] = slice(start, start + shape[i])
            data = data[tuple(sl)]
        elif delta < 0:
            # Pad symmetrically
            pad = -delta
            b = pad // 2
            a = pad - b
            pw = [(0,0)]*3
            pw[i] = (b, a)
            data = np.pad(data, pw, mode="constant")
    out = np.zeros(shape, dtype=data.dtype)
    out[:data.shape[0], :data.shape[1], :data.shape[2]] = data
    return nib.Nifti1Image(out, nib_img.affine)

# ============================
# SUBJECT PROCESSING PIPELINE
# ============================
def process_subject(subj, raw_path, mni_template_path,
                    fs_subj_dir, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    """
    Run full preprocessing for a single subject:
      1) Orientation standardization
      2) FreeSurfer recon-all (skull stripping)
      3) Load FreeSurfer outputs (brain + mask)
      4) Apply brain mask
      5) Bias field correction
      6) Resample to isotropic voxels
      7) Spatial normalization to MNI (optional)
      8) Intensity normalization (z-score)
      9) Center crop/pad to template shape
      10) Save final image
    """
    logging.info(f"--- STARTING {subj} ---")

    # Step 1: standardise orientation
    logging.info("[1/9] orientation standardisation")
    orig_img = nib.load(raw_path)
    img = standardise_orientation(orig_img)

    # Save for FreeSurfer input
    temp_std_path = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(img, temp_std_path)

    # STEP 2 PREP: ensure a clean FS subject directory
    subj_dir = os.path.join(fs_subj_dir, subj)
    if os.path.isdir(subj_dir):
        logging.info(f"→ Removing existing FreeSurfer outputs for {subj}")
        import shutil
        shutil.rmtree(subj_dir)

    # Step 2: run nipye
    logging.info("[2/9] recon-all via NiPype")
    fs_recon_all_nipype(
        input_nii=temp_std_path,
        subj_id=subj,
        subjects_dir=fs_subj_dir,
        num_threads=4
    )

    # Step 3: load FreeSurfer outputs from the fs_subj_dir you passed in
    logging.info("[3/9] loading brain and mask")
    brain_img = nib.load(os.path.join(fs_subj_dir, "mri", "brain.mgz"))
    mask_img  = nib.load(os.path.join(fs_subj_dir, "mri", "brainmask.mgz"))

    # Step 4: apply mask
    logging.info("[4/9] apply mask")
    img = apply_mask(brain_img, mask_img)

    # Step 5: bias field correction
    logging.info("[5/9] bias field correction")
    img = bias_field_correction(img)

    # Step 6: isotropic resampling
    logging.info("[6/9] resample isotropically")
    img = resample_to_iso(img)

    # Step 7: spatial normalisation (if template provided)
    if mni_template_path:
        logging.info("[7/9] spatial normalisation to MNI")
        img = spatial_normalise(img, mni_template_path)

    # Step 8: intensity normalisation (z-score)
    logging.info("[8/9] intensity normalisation")
    img = zscore(img)

    # Step 9: crop and pad to match template dimensions
    logging.info("[9/9] crop and pad")
    mni_template = nib.load(mni_template_path)
    img = center_crop_pad(img, shape=mni_template.shape)

    # Save final processed image
    out_path = os.path.join(FINAL_DIR, f"{subj}_final.nii.gz")
    nib.save(img, os.path.join(out_subdir, f"{subj}_final.nii.gz"))
    logging.info(f"Saved final image ➔ {out_path}")
    logging.info(f"--- FINISHED {subj} ---\n")

# ============================
# MAIN EXECUTION
# ============================
if __name__ == "__main__":
    logging.info(f"Walking RAW_DIR = {RAW_DIR}")
    # Walk raw data tree: RAW_DIR/site/disease/subj_folder/*.nii.gz
    for root, _, files in os.walk(RAW_DIR):
        # Only process .nii.gz files
        for fname in files:
            if not fname.endswith((".nii", ".nii.gz")):
                continue

            # 1) Compute the relative path from RAW_DIR
            rel_dir = os.path.relpath(root, RAW_DIR)  
            #    e.g. "SiteA/AD/sub-001_SiteA_T1_AD"
            site, disease, subj_folder = rel_dir.split(os.sep)

            # 2) Derive subject ID and modality from the filename
            #    e.g. 'sub-001_SiteA_T1_AD.nii.gz' → subj_id = 'sub-001_SiteA_T1_AD'
            subj_id = fname[:-7]

            # 3) Make a matching FS subject directory
            #    e.g. SUBJECTS_DIR/SiteA/AD/sub-001_SiteA_T1_AD
            fs_subj_dir = os.path.join(SUBJECTS_DIR, site, disease, subj_id)
            os.makedirs(fs_subj_dir, exist_ok=True)

            # 4) Run the full pipeline
            raw_path = os.path.join(root, fname)
            try:
                process_subject(subj_id, raw_path, mni_template_path,
                                fs_subj_dir=fs_subj_dir,
                                out_subdir=os.path.join(FINAL_DIR, site, disease, subj_id))
            except Exception:
                logging.exception(f"FAILED {subj_id}")