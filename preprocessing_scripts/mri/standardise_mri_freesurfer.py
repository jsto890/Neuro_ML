#!/usr/bin/env python3
import os
import subprocess
import logging
import sys
import tempfile

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# CONFIGURATION
FREESURFER_HOME = "/Applications/freesurfer/8.0.0"
SUBJECTS_DIR   = "/Users/josephstorey/Desktop/Part_4_Project/data/FS_subjects"
RAW_DIR        = "/Users/josephstorey/Desktop/Part_4_Project/data/test_data/mri/BRAINLAT/AD"
FINAL_DIR      = "/Users/josephstorey/Desktop/Part_4_Project/data/processed_data/MRI"
LOG_FILE       = os.path.join(FINAL_DIR, "pipeline.log")

for d in (SUBJECTS_DIR, RAW_DIR, FINAL_DIR):
    os.makedirs(d, exist_ok=True)

# SET UP LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# FREE SURFER RECON-ALL
def fs_recon_all(input_nii, subj):
    # 
    cmd = (
        f"export FREESURFER_HOME={FREESURFER_HOME} && "
        f"source $FREESURFER_HOME/SetUpFreeSurfer.sh && "
        f"recon-all -i \"{input_nii}\" -s \"{subj}\" -sd \"{SUBJECTS_DIR}\" "
        f"-all"
    )
    subprocess.run(["bash", "-lc", cmd], check=True)

# POST-PROCESSING FUNCTIONS
def apply_mask(nib_img, mask_img):
    data = nib_img.get_fdata()
    return nib.Nifti1Image(data * (mask_img.get_fdata()>0), nib_img.affine)

def resample_to_iso(nib_img, spacing=(1,1,1)):
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(nib_img, tmp)
    sitk_img = sitk.ReadImage(tmp)
    os.remove(tmp)

    orig_sp, orig_sz = sitk_img.GetSpacing(), sitk_img.GetSize()
    new_sz = [int(round(o*s/p)) for o,s,p in zip(orig_sz, orig_sp, spacing)]
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing(spacing); rf.SetSize(new_sz)
    rf.SetOutputOrigin(sitk_img.GetOrigin()); rf.SetOutputDirection(sitk_img.GetDirection())
    rf.SetInterpolator(sitk.sitkBSpline)
    out_img = rf.Execute(sitk_img)

    arr = sitk.GetArrayFromImage(out_img)        # Z–Y–X
    arr = np.transpose(arr, (2,1,0))             # → X–Y–Z
    return nib.Nifti1Image(arr, nib_img.affine)

def zscore(nib_img):
    data = nib_img.get_fdata()
    mask = data!=0
    m, s = data[mask].mean(), data[mask].std()
    return nib.Nifti1Image((data-m)/s, nib_img.affine)

def center_crop_pad(nib_img, shape=(160,192,192)):
    data = nib_img.get_fdata()
    for i in range(3):
        delta = data.shape[i] - shape[i]
        if delta>0:
            start = delta//2
            sl = [slice(None)]*3; sl[i]=slice(start, start+shape[i])
            data = data[tuple(sl)]
        elif delta<0:
            pad = -delta; b=pad//2; a=pad-b
            pw = [(0,0)]*3; pw[i]=(b,a)
            data = np.pad(data, pw, mode="constant")
    out = np.zeros(shape, dtype=data.dtype)
    out[:data.shape[0],:data.shape[1],:data.shape[2]] = data
    return nib.Nifti1Image(out, nib_img.affine)

# SUBJECT PROCESSING
def process_subject(subj, raw_path):
    logging.info(f"--- STARTING {subj} ---")
    
    # 1) FreeSurfer
    logging.info("  [1/6] recon-all")
    fs_recon_all(raw_path, subj)

    # 2) load brain + mask
    brain_img = nib.load(os.path.join(SUBJECTS_DIR, subj, "mri", "brain.mgz"))
    mask_img  = nib.load(os.path.join(SUBJECTS_DIR, subj, "mri", "brainmask.mgz"))

    # 3–6) mask, resample, normalize, crop
    logging.info("  [2/6] apply mask");     img = apply_mask(brain_img, mask_img)
    logging.info("  [3/6] resample");       img = resample_to_iso(img)
    logging.info("  [4/6] z-score normalize"); img = zscore(img)
    logging.info("  [5/6] center crop/pad"); img = center_crop_pad(img)

    # 7) save
    out_path = os.path.join(FINAL_DIR, f"{subj}_final.nii.gz")
    nib.save(img, out_path)
    logging.info(f"  [6/6] saved final ➔ {out_path}")
    logging.info(f"--- FINISHED {subj} ---\n")

# MAIN
if __name__ == "__main__":
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".nii.gz")]
    for f in tqdm(files, desc="Processing subjects"):
        subj     = f[:-7]  # strip .nii.gz
        raw_path = os.path.join(RAW_DIR, f)
        try:
            process_subject(subj, raw_path)
        except Exception:
            logging.exception(f"FAILED {subj}")