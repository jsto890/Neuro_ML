#!/usr/bin/env python3
"""
pure_pet_standardize.py  (PET-only standardisation via antsRegistration CLI)

Pipeline:
 1. (Optional) frame averaging (no motion correction)
 2. Static frame averaging (with dynamic window adjustment)
 3. Low-res Rigid+SyN via antsRegistration CLI
 4. Apply full-res transform via antsApplyTransforms
 5. SUV → SUVR (cerebellum) in template space
 6. Center-of-image fixed crop (160×192×192)
 7. Save outputs into site/disease folders

Author: Joseph Storey, 2025-05-24
"""
import os
# force single-threaded for ANTs binaries
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"]               = "8"
os.environ["MKL_NUM_THREADS"]               = "8"

import logging
import re
import subprocess
from pathlib import Path
from typing import Tuple

import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
from scipy import ndimage
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PET_ROOT      = Path("/home/jsto890/reseng202500013-ndd-ml/data/raw/PET")
OUT_ROOT      = Path("/home/jsto890/reseng202500013-ndd-ml/data/preprocessed/PET")
REF_DIR       = Path("/home/jsto890/reseng202500013-ndd-ml/P4P/Templates/PET_refs")

TEMPLATE      = REF_DIR/"FDG-PET-template.nii.gz"
PAD_TEMPLATE  = REF_DIR/"FDG-PET-template_padded.nii.gz"
CEREB_MASK    = REF_DIR/"cereb_in_petspace.nii.gz"

STATIC_FRAMES = (50,70)          # inclusive start, exclusive end
CROP_SHAPE    = (160,192,192)     # final box dims (in voxels at template resolution)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# Prepare padded template once...
if not PAD_TEMPLATE.exists():
    tpl = nib.load(str(TEMPLATE))
    data = tpl.get_fdata(dtype=np.float32)
    orig_shape = np.array(data.shape)
    pad_before = ((np.array(CROP_SHAPE) - orig_shape) // 2).astype(int)
    pad_after  = (np.array(CROP_SHAPE) - orig_shape - pad_before).astype(int)
    padded = np.pad(data, tuple(zip(pad_before,pad_after)), mode='constant')
    affine = tpl.affine.copy()
    zooms = tpl.header.get_zooms()[:3]
    affine[:3,3] -= affine[:3,:3].dot(pad_before * zooms)
    nib.save(nib.Nifti1Image(padded, affine), str(PAD_TEMPLATE))
    logging.info(f"Saved padded template: {PAD_TEMPLATE} (shape {padded.shape})")

FNAME_RE = re.compile(r"^sub-(?P<id>[^_]+)_(?P<site>ADNI|PPMI)_PET_(?P<dx>AD|PD|CN)$")

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def log_nifti_info(path: Path, label: str):
    img = nib.load(str(path), mmap=False)
    hdr = img.header
    logging.info(
        "%s • %s\n    shape = %s\n    voxels = %s\n    dtype  = %s\n    affine =\n%s", 
        label, path.name, img.shape, hdr.get_zooms(), hdr.get_data_dtype(), img.affine.round(3)
    )
    return img


def average_window(data4d: np.ndarray, window: Tuple[int,int]) -> np.ndarray:
    n = data4d.shape[3]
    s,e = window
    if n <= s:
        raise RuntimeError(f"{n} frames < start {s}")
    if n < e:
        logging.warning(f"{n} frames < end {e} → using [{s}:{n}]")
        e = n
    return data4d[..., s:e].mean(axis=3)


def crop_to_box(img: nib.Nifti1Image, box: Tuple[int,int,int]) -> nib.Nifti1Image:
    arr = img.get_fdata(dtype=np.float32)
    shape = np.array(arr.shape)
    start = ((shape - box) // 2).astype(int)
    end   = start + np.array(box)
    img_start = np.maximum(start, 0)
    img_end   = np.minimum(end, shape)
    pad_before = img_start - start
    pad_after  = end - img_end
    cropped = np.zeros(box, dtype=np.float32)
    cropped[
        pad_before[0]:box[0]-pad_after[0],
        pad_before[1]:box[1]-pad_after[1],
        pad_before[2]:box[2]-pad_after[2]
    ] = arr[
        img_start[0]:img_end[0],
        img_start[1]:img_end[1],
        img_start[2]:img_end[2]
    ]
    new_affine = img.affine.copy()
    new_affine[:3,3] += img.affine[:3,:3].dot(start)
    return nib.Nifti1Image(cropped, new_affine)

# ─── MAIN PROCESS ─────────────────────────────────────────────────────────────

def process_subject(d: Path):
    name = d.name; m = FNAME_RE.match(name)
    if not m:
        logging.warning("Skipping unexpected %s", name); return
    site,dx = m.group('site'), m.group('dx')
    final_dir = OUT_ROOT/site/dx/name
    if (final_dir/"pet_mni_crop.nii.gz").exists():
        logging.info("→ Already processed %s", name); return

    logging.info("=== Processing %s (%s, %s) ===", name, site, dx)
    raws = list(d.glob(f"{name}.nii*"))
    if not raws: logging.error("No raw PET for %s", name); return
    pet = nib.load(str(raws[0])); data4d,aff = pet.get_fdata(dtype=np.float32), pet.affine

    # Static averaging or collapse
    if data4d.ndim==4 and data4d.shape[3]>=STATIC_FRAMES[0]:
        arr = average_window(data4d, STATIC_FRAMES)
    else:
        arr = data4d.mean(axis=3) if data4d.ndim==4 else data4d
    static_path=d/"pet_static.nii.gz"
    nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), str(static_path)); log_nifti_info(static_path,"STATIC")

    # Low-res static
    logging.info(" 5) Downsampling static")
    try:
        low_stat = resample_to_output(nib.load(str(static_path)), (2,2,2))
    except Exception as e:
        logging.warning("Resample failed %s", e)
        sliced=arr[::2,::2,::2]; aff2=aff.copy(); aff2[:3,:3]*=2.0
        low_stat=nib.Nifti1Image(sliced.astype(np.float32),aff2)
    nib.save(low_stat,str(d/"low_static.nii.gz")); log_nifti_info(d/"low_static.nii.gz","LOW-RES STATIC")

    # Low-res padded template
    tpl_hi = nib.load(str(PAD_TEMPLATE))
    low_tpl = resample_to_output(tpl_hi,(2,2,2))
    nib.save(low_tpl,str(d/"low_templ_padded.nii.gz")); log_nifti_info(d/"low_templ_padded.nii.gz","LOW-RES TPL PADDED")

        # 7) ANTs registration
    logging.info("  7) Running ANTs registration:")
    reg_pref = str(d/"reg_")

    # define these first so the f-string stays simple
    low_tpl_padded_path = d/"low_templ_padded.nii.gz"
    low_static_path     = d/"low_static.nii.gz"

    cmd_reg = [
        "antsRegistration", "--dimensionality", "3", "--float", "1",
        "--output", f"[{reg_pref},{reg_pref}Warped.nii.gz]",
        "--interpolation", "Linear",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{low_tpl_padded_path},{low_static_path},1,32]",
        "--convergence", "500x250x125x50",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{low_tpl_padded_path},{low_static_path},1,4]",
        "--convergence", "50x35x25x10",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]
    subprocess.run(cmd_reg, check=True)

    # Apply transforms to padded template
    logging.info(" 8) Apply transforms (padded grid)")
    out_raw=d/"pet_mni_raw.nii.gz"
    apply_cmd=["antsApplyTransforms","--dimensionality","3","--input",str(static_path),\
               "--reference-image",str(PAD_TEMPLATE),"--output",str(out_raw),\
               "--interpolation","Linear","--transform",f"{pref}1Warp.nii.gz","--transform",f"{pref}0GenericAffine.mat"]
    subprocess.run(apply_cmd,check=True); log_nifti_info(out_raw,"PET_MNI_RAW_PADDED")

    # SUVR
    logging.info(" 9) SUVR calculation")
    raw_img=nib.load(str(out_raw)); raw_np=raw_img.get_fdata(dtype=np.float32)
    cereb=nib.load(str(CEREB_MASK)).get_fdata().astype(bool)
    mean_ref=raw_np[cereb].mean(); logging.info(" mean_ref=%.3f",mean_ref)
    suvr=raw_np/mean_ref; nib.save(nib.Nifti1Image(suvr.astype(np.float32),raw_img.affine),str(d/"pet_mni_suvr.nii.gz")); log_nifti_info(d/"pet_mni_suvr.nii.gz","PET_MNI_SUVR")

    # Center crop
    logging.info(" 10) Center crop to %s",CROP_SHAPE)
    crop= crop_to_box(nib.load(str(d/"pet_mni_suvr.nii.gz")),CROP_SHAPE)
    nib.save(crop,str(d/"pet_mni_crop.nii.gz")); log_nifti_info(d/"pet_mni_crop.nii.gz","PET_MNI_CROP")

    # Move outputs
    final=OUT_ROOT/site/dx/name; final.mkdir(parents=True,exist_ok=True)
    for f in d.glob("pet_*nii.gz"): f.replace(final/f.name)
    logging.info("→ Done %s", name)

if __name__=="__main__":
    all_jsons=sorted(PET_ROOT.rglob("sub-*_PET_*.json"))
    logging.info("Found %d subjects",len(all_jsons))
    for js in tqdm(all_jsons,desc="Subjects"): process_subject(js.parent)