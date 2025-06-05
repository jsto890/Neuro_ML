#!/usr/bin/env python3
"""
pure_pet_standardize.py  (PET-only standardisation via antsRegistration CLI)

Pipeline:
 1. (Optional) frame averaging (no motion correction)
 2. Static frame averaging (with dynamic window adjustment)
 3. Full-res Rigid+SyN via antsRegistration CLI
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
import hashlib
import time
import re
import platform
import multiprocessing
import subprocess
from pathlib import Path
from typing import Tuple

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import psutil
from scipy import ndimage as ndi
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PET_ROOT      = Path("/home/jsto890/reseng202500013-ndd-ml/data/raw/PET")
OUT_ROOT      = Path("/home/jsto890/reseng202500013-ndd-ml/data/preprocessed/PET")
REF_DIR       = Path("/home/jsto890/reseng202500013-ndd-ml/P4P/Templates/PET_refs")

TEMPLATE      = REF_DIR/"FDG-PET-template_padded.nii.gz"
CEREB_MASK    = REF_DIR/"cereb_in_petspace.nii.gz"

STATIC_FRAMES = (50,70)          # inclusive start, exclusive end
CROP_SHAPE    = (160,192,192)     # final box dims

# Configure logging to DEBUG level
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s: %(message)s")

FNAME_RE = re.compile(
    r"^sub-(?P<id>[^_]+)_(?P<site>ADNI|PPMI)_PET_(?P<dx>AD|PD|CN)$"
)

# ─── ENVIRONMENT INFO ─────────────────────────────────────────────────────────
logging.info(f"[ENV] Python {platform.python_version()}, nibabel {nib.__version__}, numpy {np.__version__}")
logging.info(f"[ENV] Host={platform.node()}, OS={platform.platform()}, CPUs={multiprocessing.cpu_count()}")

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def log_nifti_info(path: Path, label: str):
    """Load a NIfTI and log shape, zooms, dtype and affine."""
    if not path.exists():
        logging.error(f"[I/O] Missing file: {path}")
        return None
    size_mb = path.stat().st_size / 1e6
    logging.debug(f"[I/O] {label}: {path}, size={size_mb:.2f} MB")
    img = nib.load(str(path), mmap=False)
    hdr = img.header
    shape = img.shape
    zooms = hdr.get_zooms()[:3]
    dtype = img.get_data_dtype()
    affine = img.affine
    logging.debug(f"[{label}] shape={shape}")
    logging.debug(f"[{label}] voxel size (mm) = {zooms}")
    logging.debug(f"[{label}] dtype={dtype}")
    logging.debug(f"[{label}] affine=\n{affine.round(3)}")
    # Compute world corners
    corners = np.array([
        [0,0,0,1], [0,0,shape[2]-1,1], [0,shape[1]-1,0,1],
        [0,shape[1]-1,shape[2]-1,1], [shape[0]-1,0,0,1], [shape[0]-1,0,shape[2]-1,1],
        [shape[0]-1,shape[1]-1,0,1], [shape[0]-1,shape[1]-1,shape[2]-1,1]
    ]).T
    world_corners = affine @ corners
    logging.debug(f"[{label}] World corners = {world_corners.round(1)}")
    return img


def average_window(data4d: np.ndarray, window: Tuple[int,int]) -> np.ndarray:
    """Mean-average frames in the given [start,end) window, with guard."""
    n = data4d.shape[3]
    s,e = window
    logging.debug(f"[average_window] data4d.shape={data4d.shape}, window=({s},{e})")
    if n <= s:
        logging.error(f"[average_window] {n} frames < start {s}")
        raise RuntimeError(f"{n} frames < start {s}")
    if n < e:
        logging.warning(f"[average_window] {n} frames < end {e} → using [{s}:{n}]")
        e = n
    averaged = data4d[..., s:e].mean(axis=3)
    logging.debug(f"[average_window] averaged.shape={averaged.shape}")
    return averaged


def crop_to_box(img: nib.Nifti1Image, box: Tuple[int,int,int]) -> nib.Nifti1Image:
    """
    Crop img to a fixed-size box (box = (X, Y, Z)) centered on the image volume.
    Pads with zeros when the box extends outside img.
    Adjusts the affine so that the resulting NIfTI remains in world-space.
    """
    arr = img.get_fdata(dtype=np.float32)
    shape = arr.shape
    logging.debug(f"[crop_to_box] input img.shape={shape}, box={box}")

    # 1) Compute center-of-image start/end in voxel space
    img_shape = np.array(shape)
    half      = np.array(box) / 2.0
    start     = ((img_shape - box) / 2.0).astype(int)
    end       = start + np.array(box)
    logging.debug(f"[crop_to_box] img_shape={img_shape}")
    logging.debug(f"[crop_to_box] half={half}")
    logging.debug(f"[crop_to_box] start={start}, end={end}")

    # 2) Determine overlap of [start:end] with actual image bounds
    img_start = np.maximum(start, 0)
    img_end   = np.minimum(end, img_shape)
    logging.debug(f"[crop_to_box] img_start={img_start}, img_end={img_end}")

    # 3) Compute padding before/after along each axis
    pad_before = img_start - start          # zeros needed at front
    pad_after  = end - img_end              # zeros needed at back
    logging.debug(f"[crop_to_box] pad_before={pad_before}, pad_after={pad_after}")
    if (pad_before < 0).any() or (pad_after < 0).any():
        logging.error(f"[crop_to_box] Negative padding detected: before={pad_before}, after={pad_after}")

    # 4) Allocate output array and copy overlapping region
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
    logging.debug(f"[crop_to_box] cropped.shape={cropped.shape}")

    # 5) Fix affine: shift origin so voxel (0,0,0) maps to original (start) in world-space
    new_affine = img.affine.copy()
    shift = img.affine[:3, :3].dot(start)
    new_affine[:3, 3] += shift
    logging.debug(f"[crop_to_box] original affine=\n{img.affine}")
    logging.debug(f"[crop_to_box] shift={shift}")
    logging.debug(f"[crop_to_box] new affine=\n{new_affine}")

    # Compute world corners of cropped
    new_shape = box
    new_corners = np.vstack([
        [0,0,0,1],
        [new_shape[0]-1, 0, 0,1],
        [0, new_shape[1]-1, 0,1],
        [0, 0, new_shape[2]-1,1],
        [new_shape[0]-1, new_shape[1]-1, 0,1],
        [new_shape[0]-1, 0, new_shape[2]-1,1],
        [0, new_shape[1]-1, new_shape[2]-1,1],
        [new_shape[0]-1, new_shape[1]-1, new_shape[2]-1,1],
    ]).T
    world_new = new_affine @ new_corners
    logging.debug(f"[CROP] New world corners = {world_new[:3,:].round(1)}")

    return nib.Nifti1Image(cropped, new_affine)

# ─── PER-SUBJECT ──────────────────────────────────────────────────────────────

def process_subject(d: Path):
    logging.info(f"[process_subject] Starting on directory: {d}")
    name = d.name
    logging.debug(f"[process_subject] subject directory name={name}")
    m = FNAME_RE.match(name)
    if not m:
        logging.warning(f"[process_subject] Skipping unexpected folder {name}")
        return
    site, dx = m.group("site"), m.group("dx")
    logging.info(f"[process_subject] Parsed site={site}, dx={dx}")

    # skip if we've already done this subject
    final_dir = OUT_ROOT/site/dx/name
    done_file = final_dir/"pet_mni_crop.nii.gz"
    logging.debug(f"[process_subject] final_dir={final_dir}")
    logging.debug(f"[process_subject] done_file={done_file}")
    if done_file.exists():
        logging.info(f"[process_subject] → Already processed {name}, skipping")
        return

    logging.info(f"=== Processing {name} ({site}, {dx}) ===")

    # 1) Load raw PET
    raws = list(d.glob(f"{name}.nii*"))
    logging.debug(f"[process_subject] raw file list={raws}")
    if not raws:
        logging.error(f"[process_subject]   No raw PET ({name}) found")
        return
    path_raw = raws[0]
    size_mb = path_raw.stat().st_size / 1e6
    logging.info(f"[process_subject] Raw PET: {path_raw}, size={size_mb:.2f} MB")
    pet_img = nib.load(str(path_raw), mmap=True)
    data4d, aff = pet_img.get_fdata(dtype=np.float32), pet_img.affine
    logging.debug(f"[process_subject] data4d.shape={data4d.shape}")
    logging.debug(f"[process_subject] raw affine=\n{aff}")
    # Voxel-wise stats for raw PET (if 3D, treat as arr)
    if data4d.ndim == 4:
        flat = data4d.reshape(-1, data4d.shape[3]).mean(axis=1).reshape(data4d.shape[:3])
        data_stat = flat
    else:
        data_stat = data4d
    nz = int((data_stat != 0).sum())
    total = data_stat.size
    logging.debug(f"[STATS] Raw PET intensity range = ({data_stat.min():.3f}, {data_stat.max():.3f}), "
                  f"nonzero {nz}/{total} ({nz/total:.3%})")

    # 2-3) Static averaging
    t0_static = time.time()
    if data4d.ndim == 4 and data4d.shape[3] >= STATIC_FRAMES[0]:
        logging.info(f"[process_subject]   3) Averaging frames {STATIC_FRAMES[0]}–{STATIC_FRAMES[1]}")
        arr = average_window(data4d, STATIC_FRAMES)
    else:
        logging.info("[process_subject]   3) Collapsing entire volume (static/short)")
        if data4d.ndim == 4:
            arr = data4d.mean(axis=3)
            logging.debug(f"[process_subject] collapsed 4D to 3D, arr.shape={arr.shape}")
        else:
            arr = data4d
            logging.debug(f"[process_subject] single volume 3D, arr.shape={arr.shape}")
    elapsed_static = time.time() - t0_static
    logging.info(f"[TIMING] Static averaging took {elapsed_static:.1f}s")

    # Voxel-wise stats for static arr
    nz_arr = int((arr != 0).sum())
    total_arr = arr.size
    p25, p75 = np.percentile(arr, [25, 75])
    logging.debug(f"[STATS] Static PET global mean={arr.mean():.3f}, std={arr.std():.3f}, med={np.median(arr):.3f}")
    logging.debug(f"[STATS] Static PET intensity range = ({arr.min():.3f}, {arr.max():.3f}), "
                  f"nonzero {nz_arr}/{total_arr} ({nz_arr/total_arr:.3%})")
    logging.debug(f"[STATS] Static PET percentiles: 25%={p25:.3f}, 75%={p75:.3f}")

    # 4) Save static
    static_path = d/"pet_static.nii.gz"
    nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), str(static_path))
    log_nifti_info(static_path, "STATIC")

    # ─── FULL-RES REGISTRATION (static → template) ─────────────────────────────
    logging.info("[process_subject]   5) Running full-res ANTs registration:")
    reg_pref = str(d/"reg_full_")
    cmd_reg_full = [
        "antsRegistration", "--dimensionality", "3", "--float", "1",
        "--output", f"[{reg_pref},{reg_pref}Warped.nii.gz]",
        "--interpolation", "Linear",
        # Rigid-body alignment (full-res static → full-res template)
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{TEMPLATE},{static_path},1,32]",
        "--convergence", "500x250x125x50",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        # SyN nonlinear refinement
        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{TEMPLATE},{static_path},1,4]",
        "--convergence", "50x35x25x10",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]
    logging.debug(f"[process_subject] cmd_reg_full={cmd_reg_full}")
    t0_reg = time.time()
    result = subprocess.run(cmd_reg_full, capture_output=True, text=True)
    elapsed_reg = time.time() - t0_reg
    logging.info(f"[TIMING] antsRegistration took {elapsed_reg:.1f}s")
    logging.debug(f"[ANTS] STDOUT:\n{result.stdout}")
    logging.debug(f"[ANTS] STDERR:\n{result.stderr}")
    if result.returncode != 0:
        logging.error(f"[ANTS] Registration failed, returncode={result.returncode}")

    # Log transform matrix and warp info
    aff_mat_path = f"{reg_pref}0GenericAffine.mat"
    warp_field_path = f"{reg_pref}1Warp.nii.gz"

    # Try to load the affine matrix; skip if it fails
    try:
        aff_mat = np.loadtxt(aff_mat_path)
        det = np.linalg.det(aff_mat[:3, :3])
        logging.debug(f"[TRANSFORM] Affine determinant={det:.6f}")
    except Exception as e:
        logging.warning(f"[TRANSFORM] Could not load affine matrix ({aff_mat_path}): {e}")

    if Path(warp_field_path).exists():
        warp_img = nib.load(warp_field_path)
        warp_arr = warp_img.get_fdata()
        sample_pt = np.array(warp_arr.shape[:3]) // 2
        disp = warp_arr[sample_pt[0], sample_pt[1], sample_pt[2], :]
        logging.debug(f"[TRANSFORM] Warp displacement at center voxel={disp}")

    # 6) Apply those full-res transforms directly
    logging.info("[process_subject]   6) Applying transforms (full-res):")
    cmd_apply_full = [
        "antsApplyTransforms", "--dimensionality", "3",
        "--input",    str(static_path),
        "--reference-image", str(TEMPLATE),
        "--output",   str(d/"pet_mni_raw.nii.gz"),
        "--interpolation", "Linear",
        "--transform", f"{reg_pref}1Warp.nii.gz",
        "--transform", f"{reg_pref}0GenericAffine.mat",
    ]
    logging.debug(f"[process_subject] cmd_apply_full={cmd_apply_full}")
    t0_apply = time.time()
    result2 = subprocess.run(cmd_apply_full, capture_output=True, text=True)
    elapsed_apply = time.time() - t0_apply
    logging.info(f"[TIMING] antsApplyTransforms took {elapsed_apply:.1f}s")
    logging.debug(f"[ANTS] APPLY STDOUT:\n{result2.stdout}")
    logging.debug(f"[ANTS] APPLY STDERR:\n{result2.stderr}")
    raw_img = log_nifti_info(d/"pet_mni_raw.nii.gz", "PET_MNI_RAW")
    if raw_img is None:
        return
    raw_np  = raw_img.get_fdata(dtype=np.float32)
    logging.debug(f"[process_subject] raw_np.shape={raw_np.shape}")
    # Shape check
    expected_shape = CROP_SHAPE
    if raw_np.shape != expected_shape:
        logging.warning(f"[CHECK] pet_mni_raw shape {raw_np.shape} != expected {expected_shape}")

    # 7) SUVR calculation
    logging.info("[process_subject]   7) Computing SUVR (cerebellum normalisation)")
    mask_img = nib.load(str(CEREB_MASK))
    logging.debug(f"[process_subject] Loaded cerebellum mask: {CEREB_MASK}")
    mask_data0 = mask_img.get_fdata()
    orig_nz = int((mask_data0 > 0).sum())
    logging.debug(f"[MASK] Original mask nonzero={orig_nz} of {mask_data0.size} ({orig_nz/mask_data0.size:.3%})")
    mask_resampled_img = resample_from_to(
        mask_img,
        (raw_img.shape, raw_img.affine),
        order=0
    )
    mask_data = mask_resampled_img.get_fdata().astype(bool)
    new_nz  = int(mask_data.sum())
    logging.debug(f"[MASK] Resampled mask nonzero={new_nz} of {mask_data.size} ({new_nz/mask_data.size:.3%})")
    overlap = int((mask_data & (raw_np > 0)).sum())
    logging.debug(f"[MASK] Overlap mask & PET signal = {overlap} voxels")

    num_cereb_voxels = new_nz
    vox_with_signal  = int(np.count_nonzero(mask_data & (raw_np != 0)))
    logging.info(f"[process_subject]   #cereb_voxels = {num_cereb_voxels}")
    logging.info(f"[process_subject]   #cereb_voxels_with_PET_signal = {vox_with_signal}")

    if vox_with_signal == 0:
        logging.warning("[process_subject]   Mask misses PET signal. Filling SUVR with zeros.")
        suvr_np = np.zeros_like(raw_np, dtype=np.float32)
    else:
        mean_ref = float(raw_np[mask_data].mean())
        logging.debug(f"[process_subject] mean_ref={mean_ref}")
        if mean_ref == 0:
            logging.warning("[process_subject]   mean_ref=0. Filling SUVR with zeros.")
            suvr_np = np.zeros_like(raw_np, dtype=np.float32)
        else:
            logging.info(f"[process_subject]   SUVR reference mean = {mean_ref:.3f}")
            suvr_np = raw_np / mean_ref

    suvr_path = d/"pet_mni_suvr.nii.gz"
    nib.save(nib.Nifti1Image(suvr_np.astype(np.float32), raw_img.affine),
             str(suvr_path))
    log_nifti_info(suvr_path, "PET_MNI_SUVR")
    logging.debug(f"[process_subject] suvr_np.shape={suvr_np.shape}")
    # NaN / Inf checks
    if np.isnan(suvr_np).any():
        logging.warning("[CHECK] SUVR contains NaNs")
    if np.isinf(suvr_np).any():
        logging.warning("[CHECK] SUVR contains Infs")

    # ─── STEP 8) PAD → SHIFT → CROP (with equal padding on both sides) ─────

    # 8a) Reload the SUVR volume (so we know its data + original affine).
    suvr_nii  = nib.load(str(suvr_path))                   # saved with raw_img.affine
    suvr_data = suvr_nii.get_fdata(dtype=np.float32)       # shape (160, 192, 192)
    suvr_aff  = suvr_nii.affine.copy()

    # 8b) Resample the cerebellum mask exactly as before, and compute COM in voxel-space:
    mask_img0  = nib.load(str(CEREB_MASK))
    mask_img1  = resample_from_to(
        mask_img0,
        (suvr_data.shape, suvr_aff),
        order=0
    )
    mask_data1 = mask_img1.get_fdata().astype(bool)
    com_orig   = ndi.center_of_mass(mask_data1)  # e.g. (~103.7, 127.9, 140.8)
    logging.debug(f"[COMPUTE] Original COM = {com_orig}")

    # 8c) Compute how many voxels we wish to move that COM so it lands at [80,96,96]:
    desired_center = np.array(suvr_data.shape) / 2.0     # [80.0, 96.0, 96.0]
    raw_shift      = desired_center - np.array(com_orig) # e.g. [80 – 103.7, 96 – 127.9, 96 – 140.8]
    shift_vox_int  = np.round(raw_shift).astype(int)      # e.g. [-24, -32, -45]
    sx, sy, sz     = shift_vox_int
    logging.debug(f"[COMPUTE] Want to shift by {tuple(shift_vox_int)} voxels")

    # 8d) Pad *equally* on both sides of each axis by abs(shift_vox_int):
    px = abs(sx)
    py = abs(sy)
    pz = abs(sz)
    logging.debug(f"[PAD] pad_x=( {px},{px} ), pad_y=( {py},{py} ), pad_z=( {pz},{pz} )")

    padded_data = np.pad(
        suvr_data,
        ((px, px),
         (py, py),
         (pz, pz)),
        mode='constant',
        constant_values=0
    )
    # Now padded_data.shape = (160 + 2*px,  192 + 2*py,  192 + 2*pz).
    # In our example px=24,py=32,pz=45 → padded_shape=(208,256,282).

    # 8e) Shift that padded array by the same shift_vox_int:
    shifted_data = ndi.shift(
        padded_data,
        shift=(sx, sy, sz),
        order=1,        # trilinear interpolation for SUVR values
        mode='constant',
        cval=0.0
    )
    # After this shift, the COM is now located at padded_center = (208//2, 256//2, 282//2) = (104, 128, 141).

    # 8f) Build the “padded+shifted” affine:
    #     - First, because we padded px voxels at the “front” of each axis, the padded array’s index [0,0,0]
    #       used to be “original” index [ -px, -py, -pz ]. To keep world‐coords consistent, subtract
    #       [px,py,pz]·(voxel‐size‐matrix) from suvr_aff’s translation.
    padded_aff = suvr_aff.copy()
    front_offset = np.array([px, py, pz])
    padded_aff[:3, 3] -= padded_aff[:3, :3].dot(front_offset)

    #     - Next, we shifted *that padded_data* by (+sx, +sy, +sz) voxels.  In world coordinates, shifting
    #       an image by +sx voxels means we add (sx × voxel_size_X) to the translation component. Thus:
    shift_offset_mm = padded_aff[:3, :3].dot(np.array([sx, sy, sz]))
    padded_aff[:3, 3] += shift_offset_mm

    # 8g) Crop out a 160×192×192 box *around the exact centre* of that padded+shifted volume:
    px_new, py_new, pz_new = shifted_data.shape      # (208, 256, 282)
    cx, cy, cz          = px_new // 2, py_new // 2, pz_new // 2
    hx, hy, hz          = CROP_SHAPE[0] // 2, CROP_SHAPE[1] // 2, CROP_SHAPE[2] // 2

    sx_crop = cx - hx    # 104 – 80 = 24
    sy_crop = cy - hy    # 128 – 96 = 32
    sz_crop = cz - hz    # 141 – 96 = 45

    ex_crop = sx_crop + CROP_SHAPE[0]   # 24 + 160 = 184
    ey_crop = sy_crop + CROP_SHAPE[1]   # 32 + 192 = 224
    ez_crop = sz_crop + CROP_SHAPE[2]   # 45 + 192 = 237

    # Sanity check: 
    assert 0 <= sx_crop < ex_crop <= px_new
    assert 0 <= sy_crop < ey_crop <= py_new
    assert 0 <= sz_crop < ez_crop <= pz_new

    cropped_data = shifted_data[sx_crop:ex_crop, sy_crop:ey_crop, sz_crop:ez_crop]
    # cropped_data.shape == (160, 192, 192)

    # 8h) Compute the final 160×192×192 affine:
    #     - At this point, “cropped_data[i,j,k]” in the final array corresponds to
    #       “shifted_data[i + sx_crop, j + sy_crop, k + sz_crop]” in the padded volume.
    #       If padded_aff maps padded_data voxel [u,v,w] → world, then:
    #         world = padded_aff[:3,:3] @ [u,v,w]  +  padded_aff[:3,3].
    #       We want an affine final_aff so that final_aff maps [i,j,k] → that same world.  Hence:
    final_aff = padded_aff.copy()
    crop_offset_mm = padded_aff[:3, :3].dot(np.array([sx_crop, sy_crop, sz_crop]))
    final_aff[:3, 3] += crop_offset_mm

    # 8i) Save the final cropped volume (this will be your “pet_mni_crop.nii.gz”):
    crop_path = d/"pet_mni_crop.nii.gz"
    nib.save(
        nib.Nifti1Image(cropped_data.astype(np.float32), final_aff),
        str(crop_path)
    )
    log_nifti_info(crop_path, "PET_MNI_CROP_FINAL")
    # ────────────────────────────────────────────────────────────────────────────────

    # 10) Move outputs (unchanged)…
    final = OUT_ROOT/site/dx/name
    logging.debug(f"[process_subject] final output directory={final}")
    final.mkdir(parents=True, exist_ok=True)
    for f in d.glob("pet_*nii.gz"):
        dest = final/f.name
        size_out = f.stat().st_size / 1e6
        logging.debug(f"[process_subject] Moving {f} ({size_out:.2f} MB) -> {dest}")
        f.replace(dest)

    logging.info(f"[process_subject] → Done {name}\n")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    all_jsons = sorted(PET_ROOT.rglob("sub-*_PET_*.json"))
    logging.info(f"Found {len(all_jsons)} subjects")
    for idx, js in enumerate(tqdm(all_jsons, desc="Subjects"), 1):
        logging.info(f"[MAIN] Processing subject {idx}/{len(all_jsons)}: {js.parent}")
        process_subject(js.parent)
