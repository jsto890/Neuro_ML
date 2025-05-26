#!/usr/bin/env python3
"""
pure_pet_standardize.py  (PET-only standardisation with external CLI registration)

Pipeline:
 1. (Optional) 4-D motion correction (only if ≥ STATIC_FRAMES[0])
 2. Static frame averaging (with dynamic window adjustment)
 3. Rigid+SyN registration of the static image via antsRegistration CLI
    • first on a 2× downsampled grid for speed
    • then apply the resulting transform via antsApplyTransforms to
      the full-resolution static image
 4. SUV → SUVR (cerebellum) in template space
 5. Brain mask + fixed crop (160×192×192) in template space
 6. Save outputs into site/disease folders

Author: Joseph Storey, 2025-04-23 (updated 2025-05-24)
"""

import os
# limit threads for ANTs binaries too
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging, re, subprocess
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
PET_CONVERTED_ROOT = Path("/home/jsto890/reseng202500013-ndd-ml/data/raw/PET")
STD_OUTPUT_ROOT    = Path("/home/jsto890/reseng202500013-ndd-ml/data/preprocessed/PET")
REF_DIR            = Path("/home/jsto890/reseng202500013-ndd-ml/P4P/Templates/PET_refs")

MNI_PET_TEMPLATE = REF_DIR / "FDG-PET-template.nii.gz"
MNI_BRAIN_MASK   = REF_DIR / "MNI152_T1_1mm_brain_mask.nii.gz"
REF_CEREB_MASK   = REF_DIR / "cereb_mask_thr25_bin.nii.gz"

STATIC_FRAMES = (50, 70)          # inclusive start, exclusive end
CROP_SHAPE    = (160, 192, 192)   # final crop dims

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

FNAME_RE = re.compile(
    r"^sub-(?P<sub>[^_]+)_(?P<site>ADNI|PPMI)_PET_(?P<dx>AD|PD|CN)$"
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def average_frames(data4d: np.ndarray, fr: Tuple[int,int]) -> np.ndarray:
    n_time = data4d.shape[3]
    s, e   = fr
    if n_time <= s:
        raise RuntimeError(f"Only {n_time} frames (<{s}) → cannot average")
    if n_time < e:
        logging.warning(f"Only {n_time} frames → averaging {s}:{n_time}")
        e = n_time
    return data4d[..., s:e].mean(axis=3)

def motion_correct(data4d: np.ndarray) -> np.ndarray:
    import ants
    frames = [ants.from_numpy(data4d[..., i]) for i in range(data4d.shape[3])]
    ref    = frames[len(frames)//2]
    aligned = []
    for frm in frames:
        tx     = ants.registration(fixed=ref, moving=frm, type_of_transform="Rigid")
        warped = ants.apply_transforms(
            fixed=ref, moving=frm, transformlist=tx["fwdtransforms"], interpolator="linear"
        )
        aligned.append(warped.numpy())
    return np.stack(aligned, axis=3)

def crop_nifti(img_nii: nib.Nifti1Image, mask_nii: nib.Nifti1Image, box: Tuple[int,int,int]):
    data = img_nii.get_fdata(dtype=np.float32)
    m    = mask_nii.get_fdata().astype(bool)
    com  = np.array(ndimage.center_of_mass(m))[[2,1,0]]
    half = np.array(box)//2
    start = np.clip((com-half).round().astype(int),
                    0, np.array(data.shape)-box)
    sl = tuple(slice(start[d], start[d]+box[d]) for d in range(3))
    return nib.Nifti1Image(data[sl].astype(np.float32), img_nii.affine)

# ─── Per-subject pipeline ─────────────────────────────────────────────────────

def process_subject(subject_dir: Path):
    name = subject_dir.name
    m = FNAME_RE.match(name)
    if not m:
        logging.warning(f"Skipping unexpected folder: {name}")
        return
    site, dx = m.group("site"), m.group("dx")

    # find the PET NIfTI
    pet_files = list(subject_dir.glob("*.nii*"))
    if not pet_files:
        logging.error(f"No NIfTI in {name}")
        return
    pet_path = pet_files[0]

    logging.info(f"=== {name} ===")
    # --- 1) Load once, memory-mapped ---
    pet_img  = nib.load(str(pet_path), mmap=True)
    data     = pet_img.get_fdata(dtype=np.float32)
    affine   = pet_img.affine

    # --- 2) Dynamic vs static? ---
    if data.ndim == 4 and data.shape[3] >= STATIC_FRAMES[0]:
        logging.info(f"{name}: dynamic ({data.shape[3]} frames)")
        data4d      = motion_correct(data)
        static_data = average_frames(data4d, STATIC_FRAMES)
    else:
        logging.info(f"{name}: static or short scan")
        static_data = data.mean(axis=3) if data.ndim == 4 else data

    # --- 3) Save pet_static.nii.gz ---
    static_path = subject_dir / "pet_static.nii.gz"
    nib.save(nib.Nifti1Image(static_data.astype(np.float32), affine),
             str(static_path))

    # --- 3b) Prepare low-res images ---
    # We'll downsample with ANTsPy to avoid Python crashes
    import ants
    static_low = ants.image_read(str(static_path))
    templ_low  = ants.image_read(str(MNI_PET_TEMPLATE))
    orig_sp    = np.array(static_low.spacing)
    new_sp     = list(orig_sp * 2.0)
    low_static = ants.resample_image(static_low, new_sp,
                                     use_voxels=False, interp_type=1)
    low_templ  = ants.resample_image(templ_low, new_sp,
                                     use_voxels=False, interp_type=1)
    low_static.to_filename(str(subject_dir/"lowres_static.nii.gz"))
    low_templ.to_filename(str(subject_dir/"lowres_template.nii.gz"))

    # --- 4) antsRegistration CLI (low-res) ---
    pre = str(subject_dir/"reg_lowres_")
    cmd_reg = [
      "antsRegistration",
      "--dimensionality", "3",
      "--float", "1",
      "--output", f"[{pre},{pre}Warped.nii.gz]",
      "--interpolation", "Linear",
      "--transform", "Rigid[0.1]",
      f"--metric MI[{pre}Template.nii.gz,{pre}Static.nii.gz,1,32]",
      "--convergence", "1000x500x250x100",
      "--shrink-factors", "8x4x2x1",
      "--smoothing-sigmas", "3x2x1x0vox",
      "--transform", "SyN[0.1,3,0]",
      f"--metric CC[{pre}Template.nii.gz,{pre}Static.nii.gz,1,4]",
      "--convergence", "100x70x50x20",
      "--shrink-factors", "8x4x2x1",
      "--smoothing-sigmas", "3x2x1x0vox",
    ]
    subprocess.run(cmd_reg, check=True)

    # --- 5) antsApplyTransforms CLI (full-res) ---
    out_raw = subject_dir/"pet_mni_raw.nii.gz"
    cmd_apply = [
      "antsApplyTransforms",
      "--dimensionality", "3",
      "--input", str(static_path),
      "--reference-image", str(MNI_PET_TEMPLATE),
      "--output", str(out_raw),
      "--interpolation", "Linear",
      "--transform", f"{pre}0GenericAffine.mat",
      "--transform", f"{pre}1Warp.nii.gz"
    ]
    subprocess.run(cmd_apply, check=True)

    # --- 6) Compute SUVR in template space ---
    warped_img = nib.load(str(out_raw))
    warped_np  = warped_img.get_fdata(dtype=np.float32)
    cereb_mask = nib.load(str(REF_CEREB_MASK)).get_fdata().astype(bool)
    mean_ref   = warped_np[cereb_mask].mean()
    suvr_np    = warped_np / mean_ref
    suvr_path  = subject_dir/"pet_mni_suvr.nii.gz"
    nib.save(nib.Nifti1Image(suvr_np.astype(np.float32),
                              warped_img.affine),
             str(suvr_path))

    # --- 7) Crop in template space ---
    brain_mask = nib.load(str(MNI_BRAIN_MASK))
    crop_nii   = crop_nifti(nib.load(str(suvr_path)),
                             brain_mask, CROP_SHAPE)
    nib.save(crop_nii, str(subject_dir/"pet_mni_crop.nii.gz"))

    # --- 8) Move outputs to final dir ---
    final = STD_OUTPUT_ROOT / site / dx / name
    final.mkdir(parents=True, exist_ok=True)
    for f in subject_dir.glob("pet_*nii.gz"):
        f.replace(final / f.name)

    logging.info(f"→ outputs in {final}\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    subjects = sorted(PET_CONVERTED_ROOT.rglob("sub-*_PET_*.json"))
    for js in tqdm(subjects, desc="Subjects"):
        process_subject(js.parent)
