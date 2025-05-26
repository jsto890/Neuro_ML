#!/usr/bin/env python3
"""
pure_pet_standardize.py  (PET-only standardisation via antsRegistration CLI)

Pipeline:
 1. (Optional) frame averaging (no motion correction)
 2. Static frame averaging (with dynamic window adjustment)
 3. Low-res Rigid+SyN via antsRegistration CLI
 4. Apply full-res transform via antsApplyTransforms
 5. SUV → SUVR (cerebellum) in template space
 6. Brain mask + fixed crop (160×192×192)
 7. Save outputs into site/disease folders

Author: Joseph Storey, 2025-05-24
"""
import os
# force single-threaded for ANTs binaries
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]               = "1"
os.environ["MKL_NUM_THREADS"]               = "1"

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
BRAIN_MASK    = REF_DIR/"brain_in_petspace.nii.gz"
CEREB_MASK    = REF_DIR/"cereb_in_petspace.nii.gz"

STATIC_FRAMES = (50,70)          # inclusive start, exclusive end
CROP_SHAPE    = (160,192,192)     # final box dims

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

FNAME_RE = re.compile(
    r"^sub-(?P<id>[^_]+)_(?P<site>ADNI|PPMI)_PET_(?P<dx>AD|PD|CN)$"
)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def average_window(data4d: np.ndarray, window: Tuple[int,int]) -> np.ndarray:
    """Mean-average frames in the given [start,end) window, with guard."""
    n = data4d.shape[3]
    s,e = window
    if n <= s:
        raise RuntimeError(f"{n} frames < start {s}")
    if n < e:
        logging.warning(f"{n} frames < end {e} → using [{s}:{n}]")
        e = n
    return data4d[..., s:e].mean(axis=3)

def crop_to_box(img: nib.Nifti1Image,
                mask: nib.Nifti1Image,
                box: Tuple[int,int,int]) -> nib.Nifti1Image:
    """Center-of-mass crop to fixed box in template space."""
    arr = img.get_fdata(dtype=np.float32)
    m   = mask.get_fdata().astype(bool)
    com = np.array(ndimage.center_of_mass(m))[[2,1,0]]
    half= np.array(box)//2
    start = np.clip((com-half).round().astype(int),
                    0, np.array(arr.shape)-box)
    sl   = tuple(slice(start[i], start[i]+box[i]) for i in range(3))
    return nib.Nifti1Image(arr[sl].astype(np.float32), img.affine)

# ─── PER-SUBJECT ──────────────────────────────────────────────────────────────
def process_subject(d: Path):
    name = d.name
    m = FNAME_RE.match(name)
    if not m:
        logging.warning("Skipping unexpected folder %s", name)
        return
    site, dx = m.group("site"), m.group("dx")

    logging.info("=== Processing %s (%s, %s) ===", name, site, dx)

    # locate **only** the original PET scan
    raw_pattern = f"{name}.nii*"
    raws = list(d.glob(raw_pattern))
    if not raws:
        logging.error("  No raw PET (%s) found in %s", raw_pattern, name)
        return
    pet_path = raws[0]
    logging.info("  1) Loading raw PET volume: %s", pet_path)
    pet_img = nib.load(str(pet_path), mmap=True)

    logging.info("  2) Extracting data array (dtype=float32)")
    data4d  = pet_img.get_fdata(dtype=np.float32)
    aff     = pet_img.affine

    # 3) static vs dynamic → average
    if data4d.ndim == 4 and data4d.shape[3] >= STATIC_FRAMES[0]:
        logging.info("  3) Averaging frames %d–%d", *STATIC_FRAMES)
        arr = average_window(data4d, STATIC_FRAMES)
    else:
        logging.info("  3) Collapsing entire volume (static/short)")
        arr = data4d.mean(axis=3) if data4d.ndim == 4 else data4d

    # 4) save static
    static_path = d/"pet_static.nii.gz"
    logging.info("  4) Saving static PET to %s", static_path)
    static_nii  = nib.Nifti1Image(arr.astype(np.float32), aff)
    nib.save(static_nii, str(static_path))

    # 5) build low-res for ANTs
    logging.info("  5) Downsampling static to 2 mm for registration")
    try:
        low_static_nii = resample_to_output(static_nii, voxel_sizes=(2,2,2))
    except Exception as e:
        logging.warning("    nibabel resample failed (%s), slicing fallback", e)
        sliced = arr[::2,::2,::2]
        aff2   = aff.copy()
        aff2[:3,:3] *= 2.0
        low_static_nii = nib.Nifti1Image(sliced.astype(np.float32), aff2)
    low_static_path = d/"low_static.nii.gz"
    nib.save(low_static_nii, str(low_static_path))

    logging.info("  6) Downsampling template to 2 mm")
    tpl_nii = nib.load(str(TEMPLATE), mmap=False)
    try:
        low_tpl_nii = resample_to_output(tpl_nii, voxel_sizes=(2,2,2))
    except Exception as e:
        logging.warning("    template resample failed (%s), using full-res", e)
        low_tpl_nii = tpl_nii
    low_tpl_path = d/"low_templ.nii.gz"
    nib.save(low_tpl_nii, str(low_tpl_path))

    # 7) ANTs registration
    reg_prefix = str(d/"reg_")
    cmd_reg = [
        "antsRegistration", "--dimensionality", "3", "--float", "1",
        "--output", f"[{reg_prefix},{reg_prefix}Warped.nii.gz]",
        "--interpolation", "Linear",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{low_tpl_path},{low_static_path},1,32]",
        "--convergence", "1000x500x250x100",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{low_tpl_path},{low_static_path},1,4]",
        "--convergence", "100x70x50x20",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]
    logging.info("  7) Running ANTs registration:")
    logging.info("     %s", " ".join(cmd_reg))
    subprocess.run(cmd_reg, check=True)

    # 8) ANTs apply transforms
    out_raw = d/"pet_mni_raw.nii.gz"
    cmd_apply = [
        "antsApplyTransforms", "--dimensionality", "3",
        "--input", str(static_path),
        "--reference-image", str(TEMPLATE),
        "--output", str(out_raw),
        "--interpolation", "Linear",
        "--transform", f"{reg_prefix}1Warp.nii.gz",
        "--transform", f"{reg_prefix}0GenericAffine.mat",
    ]
    logging.info("  8) Applying transforms:")
    logging.info("     %s", " ".join(cmd_apply))
    try:
        subprocess.run(cmd_apply, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("    antsApplyTransforms error:\n%s", e.stderr)
        return

    # 9) SUVR calculation
    logging.info("  9) Computing SUVR (cerebellum normalisation)")
    raw_img = nib.load(str(out_raw))
    raw_np  = raw_img.get_fdata(dtype=np.float32)
    cereb   = nib.load(str(CEREB_MASK)).get_fdata().astype(bool)
    mean_ref= raw_np[cereb].mean()
    suvr_np = raw_np / mean_ref
    suvr_path = d/"pet_mni_suvr.nii.gz"
    nib.save(nib.Nifti1Image(suvr_np.astype(np.float32), raw_img.affine),
             str(suvr_path))

    # 10) Crop
    logging.info(" 10) Cropping to box %s around brain", CROP_SHAPE)
    brain_m  = nib.load(str(BRAIN_MASK))
    crop_nii = crop_to_box(nib.load(str(suvr_path)), brain_m, CROP_SHAPE)
    crop_path= d/"pet_mni_crop.nii.gz"
    nib.save(crop_nii, str(crop_path))

    # 11) Move outputs
    logging.info(" 11) Moving outputs to %s", OUT_ROOT/site/dx/name)
    final = OUT_ROOT/site/dx/name
    final.mkdir(parents=True, exist_ok=True)
    for f in d.glob("pet_*nii.gz"):
        f.replace(final/f.name)

    logging.info("→ Done %s\n", name)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    all_jsons = sorted(PET_ROOT.rglob("sub-*_PET_*.json"))
    logging.info("Found %d subjects", len(all_jsons))
    for js in tqdm(all_jsons, desc="Subjects"):
        process_subject(js.parent)
