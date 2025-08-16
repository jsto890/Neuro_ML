#!/usr/bin/env python3
"""
pure_pet_standardise.py

Automatically preprocess raw 4D/3D amyloid PET scans from ADNI:
 1. Locate the single <sub-ID>.nii (4D or 3D) in each subject folder.
 2. Collapse 4D → 3D by averaging all timeframes (static average), save as <sub-ID>_static.nii.gz.
 3. Resample that static 3D to 2 mm isotropic (low-res).
 4. Rigid+SyN registration of low-res → template.
 5. Apply transforms to the static 3D to get full-res warped.
 6. Compute SUVR using a cerebellum mask.
 7. Center-crop or pad SUVR to 160×192×192 around the whole-brain COM.
 8. Record QC statistics at each stage and write per-subject & master CSVs.

Usage:
  python3 02_norm_stand.py \
    --input_root /home/jsto890/reseng202500013-ndd-ml/data/raw/PET/ADNI \
    --output_root /home/jsto890/reseng202500013-ndd-ml/data/preprocessed/PET/ADNI \
    --lowres_template ~/reseng202500013-ndd-ml/P4P/Templates/PET/FDG_PET.nii.gz \
    --cerebellum_mask ~/reseng202500013-ndd-ml/P4P/Templates/PET/cereb_mask25_bin.nii.gz \
    --brain_mask_template ~/reseng202500013-ndd-ml/P4P/Templates/PET/MNI152_T1_1mm_brain_mask.nii.gz \
    --crop_dims 160 192 192 \
    --threads 8
"""

import os
import sys
import argparse
import logging
import subprocess
import shutil
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, label

try:
    # TemplateFlow is optional; we use it if requested/available
    from templateflow.api import get as tf_get
except Exception:  # pragma: no cover
    tf_get = None


# -----------------------------
# Helper Functions & Constants
# -----------------------------

def compute_voxel_stats(data: np.ndarray) -> dict:
    """
    Compute voxel-wise statistics on a 3D NumPy array.
    Returns a dict with keys: min, max, nonzero_frac, mean, median, std.
    """
    flat = data.flatten()
    nonzero_count = int(np.count_nonzero(flat))
    total_voxels = flat.size
    nonzero_frac = nonzero_count / total_voxels if total_voxels > 0 else 0.0

    voxel_min = float(np.min(flat))
    voxel_max = float(np.max(flat))
    voxel_mean = float(np.mean(flat))
    voxel_median = float(np.median(flat))
    voxel_std = float(np.std(flat))

    return {
        "min": voxel_min,
        "max": voxel_max,
        "nonzero_frac": nonzero_frac,
        "mean": voxel_mean,
        "median": voxel_median,
        "std": voxel_std
    }


def find_executable(names: list) -> str:
    """Return the first executable found in PATH from names, else empty string."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return ""


def run_cmd(cmd: list, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command with logging."""
    logging.info("Running: %s", " ".join(map(str, cmd)))
    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
    )
    if result.returncode != 0:
        logging.error("Command failed [%s]", result.returncode)
        if capture_output:
            if result.stdout:
                logging.debug("stdout:\n%s", result.stdout)
            if result.stderr:
                logging.error("stderr:\n%s", result.stderr)
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def fetch_templateflow_paths() -> tuple:
    """Fetch MNI152NLin2009cAsym 2mm T1 and its brain mask via TemplateFlow."""
    if tf_get is None:
        raise RuntimeError("TemplateFlow is not installed. Install with 'pip install templateflow'.")
    t1 = tf_get("MNI152NLin2009cAsym", resolution=2, suffix="T1w", desc=None, extension=".nii.gz")
    brain_mask = tf_get("MNI152NLin2009cAsym", resolution=2, suffix="mask", desc="brain", extension=".nii.gz")
    return str(t1), str(brain_mask)


def run_synthstrip(input_img: Path, out_brain: Path, out_mask: Path) -> bool:
    """Run SynthStrip (tries FreeSurfer mri_synthstrip or synthstrip). Returns True if succeeded."""
    exe = find_executable(["mri_synthstrip", "synthstrip"])
    if not exe:
        logging.warning("SynthStrip not found in PATH; skipping PET brain extraction.")
        return False
    # Prefer FreeSurfer mri_synthstrip CLI flags
    try:
        if os.path.basename(exe) == "mri_synthstrip":
            cmd = [exe, "-i", str(input_img), "-o", str(out_brain), "-m", str(out_mask)]
        else:
            # synthstrip python package CLI may use long flags
            cmd = [exe, "--i", str(input_img), "--o", str(out_brain), "--m", str(out_mask)]
        run_cmd(cmd, check=True)
        if out_brain.is_file() and out_mask.is_file():
            return True
    except Exception as e:  # pragma: no cover
        logging.warning("SynthStrip failed: %s", e)
    return False


def compute_sigma_voxels(fwhm_mm: float, voxel_sizes_mm: np.ndarray) -> np.ndarray:
    """Convert FWHM in mm to per-axis sigma in voxels using sigma = FWHM/2.355 / voxel_size."""
    if fwhm_mm <= 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    sigma_mm = fwhm_mm / 2.355
    # Guard against zeros/invalid voxel sizes
    safe_vx = np.array(voxel_sizes_mm, dtype=float)
    safe_vx[~np.isfinite(safe_vx)] = 1.0
    safe_vx = np.maximum(safe_vx, 1e-6)
    return sigma_mm / safe_vx


def get_voxel_sizes_mm(img: nib.Nifti1Image) -> np.ndarray:
    """Robust per-axis voxel sizes in mm from a NIfTI image (fallback to affine if needed)."""
    try:
        zooms = img.header.get_zooms()[:3]
        vx = np.array(zooms, dtype=float)
    except Exception:
        # Compute as column norms of affine (handles rotations)
        vx = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    vx[~np.isfinite(vx)] = 1.0
    vx = np.maximum(vx, 1e-6)
    return vx


def apply_brain_mask(img: nib.Nifti1Image, mask_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Multiply image by binary brain mask (nearest-neighbor resampled if needed)."""
    if img.shape != mask_img.shape or not np.allclose(img.affine, mask_img.affine, atol=1e-6):
        mask_img = resample_from_to(mask_img, img, order=0)
    data = img.get_fdata(dtype=np.float32)
    mask = (mask_img.get_fdata(dtype=np.float32) > 0).astype(np.float32)
    return nib.Nifti1Image(data * mask, img.affine, img.header)


def registration_syNQuick(moving_brain: Path, fixed: Path, out_prefix: Path, threads: int) -> bool:
    """Try antsRegistrationSyNQuick.sh; return True if success, else False."""
    exe = find_executable(["antsRegistrationSyNQuick.sh", "antsRegistrationSyNQuick".replace(".sh", "")])
    if not exe:
        return False
    cmd = [
        exe,
        "-d", "3",
        "-f", str(fixed),
        "-m", str(moving_brain),
        "-o", str(out_prefix),
        "-t", "s",
        "-n", str(threads),
    ]
    try:
        run_cmd(cmd, check=True)
        # SyNQuick outputs: <prefix>0GenericAffine.mat and <prefix>1Warp.nii.gz
        aff = Path(f"{out_prefix}0GenericAffine.mat")
        warp = Path(f"{out_prefix}1Warp.nii.gz")
        return aff.is_file() and warp.is_file()
    except Exception:
        return False


def registration_MI_fallback(moving_brain: Path, fixed: Path, out_prefix: Path) -> None:
    """Fallback antsRegistration with MI + SyN, similar to previous implementation."""
    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        "--output", f"[{out_prefix},{out_prefix}Warped.nii.gz]",
        "--interpolation", "Linear",
        "--initial-moving-transform", f"[{fixed},{moving_brain},1]",
        # Rigid stage
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed},{moving_brain},1,32]",
        "--convergence", "1000x500x250x100",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        # SyN stage
        "--transform", "BSplineSyN[0.1,26,0]",
        "--metric", f"CC[{fixed},{moving_brain},1,4]",
        "--convergence", "50x30x20x10",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "4x3x2x1vox",
    ]
    run_cmd(cmd, check=True)


def ants_apply_transforms(input_img: Path, reference_img: Path, output_img: Path, affine_mat: Path, warp_field: Path) -> None:
    cmd = [
        "antsApplyTransforms",
        "--dimensionality", "3",
        "--float", "1",
        "--input", str(input_img),
        "--reference-image", str(reference_img),
        "--output", str(output_img),
        "--interpolation", "Linear",
        "--transform", f"[{affine_mat},0]",
        "--transform", str(warp_field),
    ]
    run_cmd(cmd, check=True, capture_output=False)


def write_qc_csv(header: list, row: dict, out_path: Path, append: bool = False):
    """
    Write a single-row CSV with given header and values (row dict).
    If append=False, create/overwrite file; if True, append without header.
    """
    df = pd.DataFrame([row], columns=header)
    if append and out_path.exists():
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, mode="w", header=True, index=False)


def ensure_matched_affine_and_shape(img1: nib.Nifti1Image, img2: nib.Nifti1Image) -> bool:
    """
    Check that two NIfTI images have identical shape and affine (within tolerance).
    """
    shape_match = img1.shape == img2.shape
    affine_match = np.allclose(img1.affine, img2.affine, atol=1e-6)
    return shape_match and affine_match


def crop_around_com(
    data: np.ndarray,
    mask: np.ndarray,
    target_dims: tuple,
    pad_value: float = 0.0
    ) -> np.ndarray:
    """
    Crop or pad a 3D volume so that the resulting volume (target_dims) is centered 
    on the center-of-mass of the provided binary mask.
    - data: 3D NumPy array to be cropped/padded.
    - mask: 3D binary NumPy array (same shape as data) whose COM defines the center.
    - target_dims: (Z_t, Y_t, X_t) target shape.
    - pad_value: fill value for padding.
    
    Returns a new NumPy array of shape target_dims.
    """
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise RuntimeError("Empty whole-brain mask: cannot compute COM for cropping.")
    com = coords.mean(axis=0).astype(int)  # [z_com, y_com, x_com]

    z_t, y_t, x_t = target_dims
    Z, Y, X = data.shape

    # Compute start indices based on COM
    z_start = int(com[0] - z_t // 2)
    y_start = int(com[1] - y_t // 2)
    x_start = int(com[2] - x_t // 2)

    # Initialize output with pad_value
    result = np.full(target_dims, pad_value, dtype=data.dtype)

    # Helper for each axis: compute source/target slices
    def compute_slices(start, src_dim, tgt_dim):
        """
        Given start index in source, compute slicing for source and target.
        Returns: src_slice, tgt_slice
        """
        if start < 0:
            pad_before = -start
            src_s = 0
            tgt_s = pad_before
            length = min(src_dim, tgt_dim - pad_before)
        else:
            pad_before = 0
            src_s = start
            tgt_s = 0
            length = min(src_dim - start, tgt_dim)
        if length < 0:
            length = 0
        src_slice = slice(src_s, src_s + length)
        tgt_slice = slice(tgt_s, tgt_s + length)
        return src_slice, tgt_slice

    src_z_slice, tgt_z_slice = compute_slices(z_start, Z, z_t)
    src_y_slice, tgt_y_slice = compute_slices(y_start, Y, y_t)
    src_x_slice, tgt_x_slice = compute_slices(x_start, X, x_t)

    result[
        tgt_z_slice,
        tgt_y_slice,
        tgt_x_slice
    ] = data[
        src_z_slice,
        src_y_slice,
        src_x_slice
    ]

    return result


def collapse_4d_to_static(raw_path: Path, out_static_path: Path) -> nib.Nifti1Image:
    """
    Load a raw PET NIfTI (3D or 4D). If 4D, average across the 4th dimension to produce
    a static 3D image. Save that 3D image to out_static_path and return the Nifti1Image.
    Raises an error if raw cannot be loaded or is <3D/ >4D.
    """
    img = nib.load(str(raw_path))
    data = img.get_fdata(dtype=np.float32)
    aff = img.affine
    hdr = img.header.copy()
    # If 4D, collapse mean over time
    if data.ndim == 4:
        logging.info(f"[{raw_path.stem}] Input is 4D; collapsing to 3D via mean over time.")
        static_arr = data.mean(axis=3)
    elif data.ndim == 3:
        logging.info(f"[{raw_path.stem}] Input is already 3D; using as static.")
        static_arr = data
    else:
        raise RuntimeError(f"Unsupported NIfTI dimensions ({data.ndim}D); expected 3D or 4D.")

    static_img = nib.Nifti1Image(static_arr.astype(np.float32), aff, hdr)
    nib.save(static_img, str(out_static_path))
    if not out_static_path.is_file():
        raise RuntimeError(f"Failed to save static image to {out_static_path}")
    return static_img


# ---------------------------------
# Main Pipeline Implementation
# ---------------------------------

def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess PET scans: 4D→3D static, SynthStrip brain extraction, register to MNI2009c 2mm, brain mask, SUVR (+optional smoothing), z-score, QC & manifest."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root folder of raw PET data (e.g., /data/raw/PET/ADNI)"
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root folder for processed PET output (e.g., /data/processed/PET/ADNI)"
    )
    parser.add_argument(
        "--lowres_template",
        type=Path,
        required=False,
        help="Path to 2 mm isotropic MNI template. If omitted and --use_templateflow is set, will be fetched automatically."
    )
    parser.add_argument(
        "--cerebellum_mask",
        type=Path,
        required=False,
        default=None,
        help="Path to binary cerebellum mask in MNI2009c 2mm. If absent, fallback to global mean."
    )
    parser.add_argument(
        "--brain_mask_template",
        type=Path,
        required=False,
        help="Path to MNI2009c 2mm brain mask. If omitted and --use_templateflow is set, will be fetched automatically."
    )
    parser.add_argument(
        "--crop_dims",
        type=int,
        nargs=3,
        default=[160, 192, 192],
        metavar=("Z", "Y", "X"),
        help="Target dimensions for final crop (Z Y X), default = 160 192 192"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of CPU threads for ANTs calls"
    )
    parser.add_argument(
        "--use_templateflow",
        action="store_true",
        help="Use TemplateFlow to fetch MNI152NLin2009cAsym 2mm T1 and brain mask. Overrides provided template/mask paths if set."
    )
    parser.add_argument(
        "--smooth_fwhm",
        type=float,
        default=0.0,
        help="Optional Gaussian smoothing FWHM in mm applied to SUVR (0 disables)."
    )
    parser.add_argument(
        "--presmooth_fwhm",
        type=float,
        default=0.0,
        help="Optional light pre-smoothing FWHM in mm before registration to stabilise MI (0 disables)."
    )
    parser.add_argument(
        "--write_manifest",
        action="store_true",
        help="Write a per-subject JSON provenance manifest next to outputs."
    )
    parser.add_argument(
        "--suvr_ref_stat",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="Statistic to use for SUVR reference (mean or median) for cerebellum/global. Default: mean."
    )
    parser.add_argument(
        "--skip_cropping",
        action="store_true",
        help="Skip COM-based cropping; keep outputs at template resolution."
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip a subject if its expected final output already exists (e.g., SUVR_s{int(smooth_fwhm)}.nii.gz when smoothing>0, otherwise SUVR.nii.gz)."
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional list of subject directory names to process (e.g., sub-XXX ...). If omitted, process all."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete any existing output subject folder before processing to avoid using stale intermediates."
    )
    parser.add_argument(
        "--reg_mode",
        type=str,
        choices=["syn", "syn_light", "affine", "rigid"],
        default="syn",
        help="Registration mode: SyN (syn), lighter/less-deformable SyN (syn_light), affine only (affine), or rigid only (rigid). Default: syn."
    )
    args = parser.parse_args()

    # 2. Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("Starting PET preprocessing pipeline")
    logging.info(f"Arguments: {args}")

    # 3. Set environment variables to constrain threading for ANTs and NumPy
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    # Helper: progress tracking utilities
    def render_bar(completed: int, total: int, width: int = 24) -> str:
        total = max(total, 1)
        k = int(width * completed / total)
        return "[" + ("#" * k) + ("." * (width - k)) + "]"

    progress = {}
    def write_progress_file():
        try:
            out_path = args.output_root / "progress.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception:
            pass

    def set_subject_status(cohort: str, group: str, subject: str, status: str):
        grp = progress.setdefault(cohort, {}).setdefault(group, {
            "total": 0, "completed": 0, "success": 0, "error": 0, "subjects": {}
        })
        prev = grp["subjects"].get(subject, {}).get("status")
        if prev in ("SUCCESS", "ERROR"):
            return
        grp["subjects"][subject] = {"status": status}
        if status in ("SUCCESS", "ERROR"):
            grp["completed"] += 1
            if status == "SUCCESS":
                grp["success"] += 1
            else:
                grp["error"] += 1
        write_progress_file()
        # log concise progress line
        logging.info(f"Progress [{cohort}/{group}] {render_bar(grp['completed'], grp['total'])} {grp['completed']}/{grp['total']} ok:{grp['success']} err:{grp['error']}")

    # 4. Validate / resolve templates
    if not args.input_root.is_dir():
        logging.error(f"Input root not found: {args.input_root}")
        sys.exit(1)
    template_path: Path
    brain_mask_path: Path
    if args.use_templateflow:
        try:
            t1_path, bm_path = fetch_templateflow_paths()
            template_path = Path(t1_path)
            brain_mask_path = Path(bm_path)
            logging.info("Using TemplateFlow MNI2009c 2mm template: %s", template_path)
            logging.info("Using TemplateFlow brain mask: %s", brain_mask_path)
        except Exception as e:
            logging.error("TemplateFlow fetch failed: %s", e)
            sys.exit(1)
    else:
        if not args.lowres_template or not args.lowres_template.is_file():
            logging.error("Low-res template not found or not provided. Provide --lowres_template or use --use_templateflow.")
            sys.exit(1)
        if not args.brain_mask_template or not args.brain_mask_template.is_file():
            logging.error("Brain mask template not found or not provided. Provide --brain_mask_template or use --use_templateflow.")
            sys.exit(1)
        template_path = args.lowres_template
        brain_mask_path = args.brain_mask_template

    # 5. Prepare master QC CSV
    master_qc_path = args.output_root / "qc_stats_master.csv"
    qc_header = [
        "subject_id",
        # low-res stats
        "lowres_min", "lowres_max", "lowres_nonzero_frac", "lowres_mean", "lowres_median", "lowres_std",
        # low-res warped stats
        "lowres_warped_min", "lowres_warped_max", "lowres_warped_nonzero_frac", "lowres_warped_mean",
        "lowres_warped_median", "lowres_warped_std",
        # full-res warped stats
        "fullres_warped_min", "fullres_warped_max", "fullres_warped_nonzero_frac", "fullres_warped_mean",
        "fullres_warped_median", "fullres_warped_std",
        # SUVR stats
        "suvr_min", "suvr_max", "suvr_nonzero_frac", "suvr_mean", "suvr_median", "suvr_std",
        # crop status
        "crop_status"
    ]
    if not args.output_root.is_dir():
        args.output_root.mkdir(parents=True, exist_ok=True)
    if not master_qc_path.exists():
        logging.info(f"Creating master QC CSV: {master_qc_path}")
        write_qc_csv(qc_header, {k: "" for k in qc_header}, master_qc_path, append=False)
        master_qc_path.unlink()
        write_qc_csv(qc_header, {}, master_qc_path, append=False)

    # 6. Load template, cerebellum mask (optional), and whole-brain mask
    try:
        tmpl_img = nib.load(str(template_path))
    except Exception as e:
        logging.error(f"Failed to load template: {e}")
        sys.exit(1)
    try:
        if args.cerebellum_mask and Path(args.cerebellum_mask).is_file():
            cereb_img = nib.load(str(args.cerebellum_mask))
            if cereb_img.shape != tmpl_img.shape or not np.allclose(cereb_img.affine, tmpl_img.affine, atol=1e-6):
                logging.info(f"Resampling cerebellum mask {cereb_img.shape} → {tmpl_img.shape}")
            cereb_img = resample_from_to(cereb_img, tmpl_img, order=0)  # nearest-neighbor
        else:
            cereb_img = None
            logging.warning("No cerebellum mask provided; will fallback to global brain mean for SUVR.")
    except Exception as e:
        logging.error(f"Failed to load cerebellum mask: {e}")
        cereb_img = None
    try:
        brain_img = nib.load(str(brain_mask_path))
        if brain_img.shape != tmpl_img.shape or not np.allclose(brain_img.affine, tmpl_img.affine, atol=1e-6):
            logging.info(f"Resampling brain mask {brain_img.shape} → {tmpl_img.shape}")
        brain_img = resample_from_to(brain_img, tmpl_img, order=0)  # nearest‐neighbor
    except Exception as e:
        logging.error(f"Failed to load brain mask template: {e}")
        sys.exit(1)

    # 7. Iterate over cohorts (directories under input_root)
    for cohort_dir in sorted(args.input_root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        cohort_name = cohort_dir.name
        logging.info(f"Processing cohort: {cohort_name}")

        # Handle nested directory structure (ADNI/AD/sub-xxx vs direct ADNI/sub-xxx)
        all_subject_dirs = []
        for item in sorted(cohort_dir.iterdir()):
            if not item.is_dir():
                continue
            # Check if this directory contains .nii files (subject dir) or subdirectories (group dir)
            nii_files = list(item.glob("*.nii*"))
            if nii_files:
                # This is a subject directory
                all_subject_dirs.append(item)
                grp = progress.setdefault(cohort_name, {}).setdefault("subjects", {"total": 0, "completed": 0, "success": 0, "error": 0, "subjects": {}})
                grp["total"] += 1
            else:
                # This might be a group directory, check for subdirectories
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        all_subject_dirs.append(subitem)
                        grp = progress.setdefault(cohort_name, {}).setdefault(item.name, {"total": 0, "completed": 0, "success": 0, "error": 0, "subjects": {}})
                        grp["total"] += 1
        write_progress_file()
        
        for subject_dir in all_subject_dirs:
            if not subject_dir.is_dir():
                continue
            sub_id = subject_dir.name
            if args.subjects and sub_id not in set(args.subjects):
                continue
            logging.info(f"--- Subject: {sub_id} ---")
            # infer group for progress accounting
            group_name = subject_dir.parent.name if subject_dir.parent != cohort_dir else "subjects"
            set_subject_status(cohort_name, group_name, sub_id, "RUNNING")
            # 7.1. Locate .nii file in subject_dir (flexible pattern matching)
            raw_candidates = list(subject_dir.glob(f"{sub_id}.nii*"))
            if len(raw_candidates) == 0:
                # Try any .nii file, but filter out processed files
                all_nii = list(subject_dir.glob("*.nii*"))
                raw_candidates = [f for f in all_nii if not any(x in f.name.lower() for x in ['static', 'warped', 'reg_', 'low_'])]
            
            if len(raw_candidates) == 0:
                logging.error(f"[{sub_id}] No suitable .nii file found; skipping.")
                continue
            elif len(raw_candidates) > 1:
                logging.warning(f"[{sub_id}] Multiple .nii files found, using: {raw_candidates[0].name}")
            
            raw_nifti = raw_candidates[0]
            logging.info(f"[{sub_id}] Processing file: {raw_nifti.name}")

            # 7.2. Create output subject directory
            out_sub_dir = args.output_root / cohort_name / sub_id
            if args.overwrite and out_sub_dir.exists():
                logging.info(f"[{sub_id}] --overwrite set: removing existing {out_sub_dir}")
                shutil.rmtree(out_sub_dir, ignore_errors=True)
            out_sub_dir.mkdir(parents=True, exist_ok=True)

            # Optionally skip if final output already exists
            if args.skip_if_exists:
                expected = out_sub_dir / (f"{sub_id}_SUVR_s{int(round(args.smooth_fwhm))}.nii.gz" if (args.smooth_fwhm and args.smooth_fwhm>0) else f"{sub_id}_SUVR.nii.gz")
                if expected.exists():
                    logging.info(f"[{sub_id}] Skipping (final output exists): {expected}")
                    # mark as success to advance progress
                    set_subject_status(cohort_name, group_name, sub_id, "SUCCESS")
                    continue

            # Initialize QC dict
            qc_stats = {key: "" for key in qc_header}
            qc_stats["subject_id"] = sub_id

            # Define paths for intermediate/final files
            static_path = out_sub_dir / f"{sub_id}_static.nii.gz"
            lowres_path = out_sub_dir / f"{sub_id}_lowres.nii.gz"
            lowres_warped_path = out_sub_dir / f"{sub_id}_lowres_Warped.nii.gz"
            fullres_warped_path = out_sub_dir / f"{sub_id}_fullres_warped.nii.gz"
            suvr_path = out_sub_dir / f"{sub_id}_SUVR.nii.gz"
            suvr_cropped_path = out_sub_dir / f"{sub_id}_SUVR_cropped.nii.gz"
            qc_csv_path = out_sub_dir / "qc_stats.csv"

            # ---------------------
            # STEP A: Collapse 4D → 3D static
            # ---------------------
            try:
                logging.info(f"[{sub_id}] Collapsing 4D→3D (static) if needed")
                static_img = collapse_4d_to_static(raw_nifti, static_path)
                static_data = static_img.get_fdata(dtype=np.float32)
                # (Optional) Could QC static here, but pipeline continues
            except Exception as e:
                logging.error(f"[{sub_id}] Error during 4D→3D collapse: {e}")
                qc_stats["crop_status"] = "STATIC_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                set_subject_status(cohort_name, group_name, sub_id, "ERROR")
                continue

            # ---------------------
            # STEP 1: Optional pre-smoothing before registration
            # ---------------------
            static_for_reg_path = out_sub_dir / f"{sub_id}_static_presmooth.nii.gz"
            if args.presmooth_fwhm and args.presmooth_fwhm > 0:
                logging.info(f"[{sub_id}] Pre-smoothing static with FWHM={args.presmooth_fwhm:.2f} mm before registration")
                static_data = static_img.get_fdata(dtype=np.float32)
                # Robust voxel sizes
                vx = get_voxel_sizes_mm(static_img)
                sigmas = compute_sigma_voxels(args.presmooth_fwhm, vx)
                smoothed = gaussian_filter(static_data, sigma=sigmas[::-1])  # scipy uses z,y,x order; our vx is x,y,z
                sm_img = nib.Nifti1Image(smoothed, static_img.affine, static_img.header)
                nib.save(sm_img, str(static_for_reg_path))
                moving_static_path = static_for_reg_path
            else:
                moving_static_path = static_path

            # ---------------------
            # STEP 2: Brain extraction (SynthStrip)
            # ---------------------
            pet_brain_path = out_sub_dir / f"{sub_id}_static_brain.nii.gz"
            pet_brainmask_path = out_sub_dir / f"{sub_id}_static_brainmask.nii.gz"
            brain_extracted = run_synthstrip(moving_static_path, pet_brain_path, pet_brainmask_path)
            if not brain_extracted:
                # Fallback: use the input as brain, and a dummy all-ones mask at native space
                logging.warning(f"[{sub_id}] Proceeding without SynthStrip. Using original as brain for registration.")
                shutil.copyfile(moving_static_path, pet_brain_path)
                # create mask of ones with same shape
                tmp_img = nib.load(str(moving_static_path))
                msk = np.ones(tmp_img.shape, dtype=np.uint8)
                nib.save(nib.Nifti1Image(msk, tmp_img.affine, tmp_img.header), str(pet_brainmask_path))
            else:
                # Quick sanity on brainmask; adjust if wildly off
                msk_img = nib.load(str(pet_brainmask_path))
                frac = float(np.count_nonzero(msk_img.get_fdata() > 0) / np.prod(msk_img.shape))
                if frac < 0.05 or frac > 0.95:
                    logging.warning(f"[{sub_id}] SynthStrip mask fraction {frac:.1%} suspicious; applying 1-voxel morphology fix")
                    arr = (msk_img.get_fdata() > 0).astype(np.uint8)
                    arr = binary_dilation(arr) if frac < 0.05 else binary_erosion(arr)
                    nib.save(nib.Nifti1Image(arr.astype(np.uint8), msk_img.affine, msk_img.header), str(pet_brainmask_path))

            # ---------------------
            # STEP 3: Registration (SyNQuick preferred, fallback to MI+SyN)
            # ---------------------
            out_prefix = str(out_sub_dir / f"{sub_id}_reg_")
            # Ensure environment threads
            os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
            os.environ["OMP_NUM_THREADS"] = str(args.threads)
            os.environ["MKL_NUM_THREADS"] = str(args.threads)

            moved_brain = Path(pet_brain_path)
            fixed_tmpl = template_path
            used_method = ""
            if args.reg_mode in ("syn","syn_light"):
                used_method = "SyNQuick"
                if not registration_syNQuick(moved_brain, fixed_tmpl, out_prefix, args.threads):
                    logging.info(f"[{sub_id}] SyNQuick unavailable/failed; using MI+SyN fallback")
                    used_method = "MI+SyN"
                    registration_MI_fallback(moved_brain, fixed_tmpl, out_prefix)
                # If syn_light, down-weight/suppress the non-linear warp by converting to affine-only application
                if args.reg_mode == "syn_light":
                    # Remove warp if present to effectively apply only affine
                    try:
                        warp_path = Path(f"{out_prefix}1Warp.nii.gz")
                        if warp_path.exists():
                            warp_path.unlink()
                    except Exception:
                        pass
            else:
                # Affine or rigid only via antsRegistration
                used_method = "AffineOnly" if args.reg_mode == "affine" else "RigidOnly"
                reg_cmd = [
                    "antsRegistration","--dimensionality","3","--float","1",
                    "--output", f"[{out_prefix},{out_prefix}Warped.nii.gz]",
                    "--interpolation","Linear",
                    "--initial-moving-transform", f"[{fixed_tmpl},{moved_brain},1]",
                ]
                if args.reg_mode == "rigid":
                    reg_cmd += [
                        "--transform","Rigid[0.1]",
                        "--metric",f"MI[{fixed_tmpl},{moved_brain},1,32]",
                        "--convergence","1000x500x250x100",
                        "--shrink-factors","8x4x2x1",
                        "--smoothing-sigmas","3x2x1x0vox",
                    ]
                else:
                    reg_cmd += [
                        "--transform","Affine[0.1]",
                        "--metric",f"MI[{fixed_tmpl},{moved_brain},1,32]",
                        "--convergence","1000x500x250x100",
                        "--shrink-factors","8x4x2x1",
                        "--smoothing-sigmas","3x2x1x0vox",
                    ]
                run_cmd(reg_cmd, check=True)

            # Collect transform paths
            transform_affine = Path(f"{out_prefix}0GenericAffine.mat")
            transform_warp = Path(f"{out_prefix}1Warp.nii.gz")
            # In affine/rigid modes there is no non-linear warp; fake an identity by reusing affine-only application
            if args.reg_mode in ("affine","rigid","syn_light"):
                if not transform_affine.is_file():
                    logging.error(f"[{sub_id}] Missing affine transform from registration")
                    qc_stats["crop_status"] = "REGISTRATION_ERROR"
                    write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                    write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                    continue
                transform_warp = None
            if not transform_affine.is_file() or (args.reg_mode=="syn" and not transform_warp.is_file()):
                logging.error(f"[{sub_id}] Missing transform files from registration")
                qc_stats["crop_status"] = "REGISTRATION_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                set_subject_status(cohort_name, group_name, sub_id, "ERROR")
                continue

            # ---------------------
            # STEP 4: Apply Transforms to Static (Full-Res → MNI), then apply MNI brain mask
            # ---------------------
            try:
                logging.info(f"[{sub_id}] Applying transforms to static and masking with MNI brain mask")
                if args.reg_mode in ("affine","rigid","syn_light"):
                    cmd = [
                        "antsApplyTransforms","--dimensionality","3","--float","1",
                        "--input", str(static_path),
                        "--reference-image", str(template_path),
                        "--output", str(fullres_warped_path),
                        "--interpolation","Linear",
                        "--transform", f"[{transform_affine},0]",
                    ]
                    run_cmd(cmd, check=True, capture_output=False)
                else:
                    ants_apply_transforms(static_path, template_path, fullres_warped_path, transform_affine, transform_warp)
                if not fullres_warped_path.is_file():
                    raise RuntimeError("antsApplyTransforms did not produce full-res warped image")

                fullres_warped_img = nib.load(str(fullres_warped_path))
                frw_data = fullres_warped_img.get_fdata(dtype=np.float32)
                stats = compute_voxel_stats(frw_data)
                qc_stats.update({
                    "fullres_warped_min": stats["min"],
                    "fullres_warped_max": stats["max"],
                    "fullres_warped_nonzero_frac": stats["nonzero_frac"],
                    "fullres_warped_mean": stats["mean"],
                    "fullres_warped_median": stats["median"],
                    "fullres_warped_std": stats["std"]
                })
                logging.debug(f"[{sub_id}] Full-res warped stats: {stats}")

                # Apply standard brain mask
                masked_img = apply_brain_mask(fullres_warped_img, brain_img)
                nib.save(masked_img, str(out_sub_dir / f"{sub_id}_MNI_brain.nii.gz"))
            except subprocess.CalledProcessError as e:
                logging.error(f"[{sub_id}] antsApplyTransforms failed: {e}")
                qc_stats["crop_status"] = "APPLY_TRANSFORMS_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                set_subject_status(cohort_name, group_name, sub_id, "ERROR")
                continue
            except Exception as e:
                logging.error(f"[{sub_id}] Error during full-res apply: {e}")
                qc_stats["crop_status"] = "APPLY_TRANSFORMS_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                set_subject_status(cohort_name, group_name, sub_id, "ERROR")
                continue

            # ---------------------
            # STEP 5: Compute SUVR (Cerebellum Normalization; fallback to global mean)
            # ---------------------
            try:
                logging.info(f"[{sub_id}] Computing SUVR")
                # Ensure brain mask aligned
                brain_mask_img = brain_img
                if not ensure_matched_affine_and_shape(fullres_warped_img, brain_mask_img):
                    brain_mask_img = resample_from_to(brain_mask_img, fullres_warped_img, order=0)
                brain_mask = (brain_mask_img.get_fdata() > 0)

                suvr_reference = "cerebellum" if cereb_img is not None else "global"
                ref_values = None
                # Prefer cerebellum reference if available and aligned
                if cereb_img is not None and ensure_matched_affine_and_shape(fullres_warped_img, cereb_img):
                    cereb_array = (cereb_img.get_fdata(dtype=np.float32) > 0)
                    if np.any(cereb_array):
                        ref_values = frw_data[cereb_array]
                    else:
                        suvr_reference = "global"
                # Fallback to global brain reference
                if ref_values is None:
                    ref_values = frw_data[brain_mask]
                    suvr_reference = "global"

                ref_mean_val = float(np.mean(ref_values)) if ref_values.size > 0 else float("nan")
                ref_median_val = float(np.median(ref_values)) if ref_values.size > 0 else float("nan")
                chosen_stat = args.suvr_ref_stat if hasattr(args, "suvr_ref_stat") else "mean"
                ref_value = ref_mean_val if chosen_stat == "mean" else ref_median_val
                # Guard against non-finite/zero reference
                if not np.isfinite(ref_value) or ref_value <= 0:
                    ref_values = frw_data[brain_mask]
                    ref_mean_val = float(np.mean(ref_values)) if ref_values.size > 0 else float("nan")
                    ref_median_val = float(np.median(ref_values)) if ref_values.size > 0 else float("nan")
                    ref_value = ref_mean_val if chosen_stat == "mean" else ref_median_val
                    suvr_reference = "global"

                suvr_array = frw_data / (ref_value + 1e-8)
                suvr_img = nib.Nifti1Image(suvr_array, fullres_warped_img.affine, fullres_warped_img.header)
                nib.save(suvr_img, str(suvr_path))
                if not suvr_path.is_file():
                    raise RuntimeError("Failed to save SUVR image")

                stats = compute_voxel_stats(suvr_array)
                qc_stats.update({
                    "suvr_min": stats["min"],
                    "suvr_max": stats["max"],
                    "suvr_nonzero_frac": stats["nonzero_frac"],
                    "suvr_mean": stats["mean"],
                    "suvr_median": stats["median"],
                    "suvr_std": stats["std"]
                })
                logging.debug(f"[{sub_id}] SUVR stats: {stats}")
                qc_stats["registration_method"] = used_method
                qc_stats["suvr_reference"] = suvr_reference
                qc_stats["suvr_ref_stat"] = chosen_stat
                qc_stats["reference_mean"] = ref_mean_val
                qc_stats["reference_median"] = ref_median_val
            except Exception as e:
                logging.error(f"[{sub_id}] Error during SUVR computation: {e}")
                qc_stats["crop_status"] = "SUVR_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                set_subject_status(cohort_name, group_name, sub_id, "ERROR")
                continue
            
            # ---------------------
            # STEP 6: Optional smoothing and Z-score
            # ---------------------
            z_img_path = out_sub_dir / f"{sub_id}_SUVR_Z.nii.gz"
            try:
                # Optional smoothing
                if args.smooth_fwhm and args.smooth_fwhm > 0:
                    logging.info(f"[{sub_id}] Applying Gaussian smoothing to SUVR with FWHM={args.smooth_fwhm:.2f} mm")
                    suvr_img = nib.load(str(suvr_path))
                    suvr_data = suvr_img.get_fdata(dtype=np.float32)
                    vx = get_voxel_sizes_mm(suvr_img)
                    sig = compute_sigma_voxels(args.smooth_fwhm, vx)
                    smoothed = gaussian_filter(suvr_data, sigma=sig[::-1])
                    suvr_smooth_img = nib.Nifti1Image(smoothed, suvr_img.affine, suvr_img.header)
                    suvr_s_path = out_sub_dir / f"{sub_id}_SUVR_s{int(round(args.smooth_fwhm))}.nii.gz"
                    nib.save(suvr_smooth_img, str(suvr_s_path))
                    qc_stats["smoothing_fwhm"] = float(args.smooth_fwhm)
                else:
                    qc_stats["smoothing_fwhm"] = 0.0

                # Z-score inside brain mask
                suvr_img = nib.load(str(suvr_path))
                suvr_data = suvr_img.get_fdata(dtype=np.float32)
                mask_img_res = brain_img
                if not ensure_matched_affine_and_shape(suvr_img, mask_img_res):
                    mask_img_res = resample_from_to(mask_img_res, suvr_img, order=0)
                brain_mask = (mask_img_res.get_fdata() > 0)
                brain_vals = suvr_data[brain_mask]
                mu = float(np.mean(brain_vals)) if brain_vals.size > 0 else 0.0
                sd = float(np.std(brain_vals)) if brain_vals.size > 0 else 1.0
                z = np.zeros_like(suvr_data, dtype=np.float32)
                if sd > 0:
                    z[brain_mask] = (suvr_data[brain_mask] - mu) / sd
                nib.save(nib.Nifti1Image(z, suvr_img.affine, suvr_img.header), str(z_img_path))
                qc_stats["qc_z_mean"] = mu
                qc_stats["qc_z_std"] = sd
            except Exception as e:
                logging.warning(f"[{sub_id}] Smoothing/Z-score step failed: {e}")

            # ---------------------
            # STEP 7: Robust subject-based crop & affine update (optional)
            # ---------------------
            try:
                if args.skip_cropping:
                    logging.info(f"[{sub_id}] Skipping cropping (keeping template resolution)")
                    qc_stats["crop_status"] = "SKIPPED"
                    raise RuntimeError("CROP_SKIPPED")
                logging.info(f"[{sub_id}] Robust cropping/padding around subject COM to {tuple(args.crop_dims)}")

                # 5.1 load SUVR
                suvr_img   = nib.load(str(suvr_path))
                suvr_array = suvr_img.get_fdata(dtype=np.float32)

                # 5.2 compute a data-driven threshold: 25th percentile of non-zero SUVR
                nonzero = suvr_array[suvr_array>0]
                if nonzero.size < 100:             # too few non-zero voxels → bad SUVR?
                    raise RuntimeError("Too few nonzero SUVR voxels for reliable mask.")
                thresh = np.percentile(nonzero, 25)

                # 5.3 build initial mask & keep only largest CC
                init_mask = suvr_array > thresh
                labels_arr, num = label(init_mask)
                if num == 0:
                    raise RuntimeError("Empty mask after threshold.")
                # pick largest component (ignore background label 0)
                counts = np.bincount(labels_arr.flat)
                counts[0] = 0
                main_lbl = counts.argmax()
                subj_mask = (labels_arr == main_lbl)

                # 5.4 if the mask is too small, fallback to template COM
                vol_frac = subj_mask.sum() / subj_mask.size
                if vol_frac < 0.10:
                    logging.warning(f"[{sub_id}] Mask too small ({vol_frac:.1%}); using template COM.")
                    # compute COM of template brain mask
                    coords = np.argwhere(brain_img.get_fdata()>0)
                else:
                    coords = np.argwhere(subj_mask)

                com = coords.mean(axis=0).astype(int)  # [z, y, x]

                # 5.5 crop/pad
                z_t, y_t, x_t = tuple(args.crop_dims)
                cropped_array = crop_around_com(suvr_array, subj_mask, (z_t, y_t, x_t), pad_value=0.0)

                # 5.6 fix the affine translation
                vs = np.array([suvr_img.affine[0,0], suvr_img.affine[1,1], suvr_img.affine[2,2]])
                # recalc start indices for affine shift
                z_start = int(com[0] - z_t // 2)
                y_start = int(com[1] - y_t // 2)
                x_start = int(com[2] - x_t // 2)

                new_affine = suvr_img.affine.copy()
                new_affine[:3,3] += np.array([x_start, y_start, z_start]) * vs

                # 5.7 save
                cropped_img = nib.Nifti1Image(cropped_array, new_affine, suvr_img.header)
                nib.save(cropped_img, str(suvr_cropped_path))
                if not suvr_cropped_path.is_file():
                    raise RuntimeError("Failed to save SUVR cropped image")

                qc_stats["crop_status"] = "SUCCESS"
                logging.info(f"[{sub_id}] Cropping/padding successful")

            except Exception as e:
                if str(e) == "CROP_SKIPPED":
                    pass
                else:
                    logging.error(f"[{sub_id}] Error during cropping/padding: {e}")
                    qc_stats["crop_status"] = "CROP_ERROR"

            # ---------------------
            # STEP 8: Compute QC gates and write QC Stats & Manifest
            # ---------------------
            try:
                # Minimal QC gates
                try:
                    # Registration sanity: fraction of warped brain voxels outside MNI brain mask
                    frw_img = nib.load(str(fullres_warped_path))
                    frw_masked = apply_brain_mask(frw_img, brain_img)
                    brain_mask_res = resample_from_to(brain_img, frw_img, order=0)
                    m = (brain_mask_res.get_fdata() > 0)
                    nz = frw_img.get_fdata() > 0
                    outside = np.logical_and(nz, ~m)
                    overlap_frac = 1.0 - float(np.count_nonzero(outside) / max(1, np.count_nonzero(nz)))
                    qc_stats["qc_registration_overlap"] = overlap_frac
                    qc_stats["qc_registration_pass"] = overlap_frac > 0.9
                except Exception:
                    qc_stats["qc_registration_overlap"] = ""
                    qc_stats["qc_registration_pass"] = ""

                try:
                    chosen_stat = str(qc_stats.get("suvr_ref_stat", "mean"))
                    # value actually used for SUVR scaling
                    ref_val = float(qc_stats.get("reference_mean", float("nan")))
                    if chosen_stat == "median":
                        ref_val = float(qc_stats.get("reference_median", float("nan")))
                    qc_stats["qc_reference_value"] = ref_val
                    # flag if <= 5th percentile of brain intensities
                    frw_img = nib.load(str(fullres_warped_path))
                    m = (resample_from_to(brain_img, frw_img, order=0).get_fdata() > 0)
                    vals = frw_img.get_fdata()[m]
                    p5 = float(np.percentile(vals, 5)) if vals.size > 0 else 0.0
                    qc_stats["qc_reference_pass"] = np.isfinite(ref_val) and (ref_val > 0) and (ref_val > p5)
                except Exception:
                    qc_stats["qc_reference_pass"] = ""

                try:
                    qc_stats["qc_suvr_median"] = qc_stats.get("suvr_median", "")
                    med = qc_stats.get("suvr_median", None)
                    qc_stats["qc_suvr_median_pass"] = (med is not None) and (0.7 <= float(med) <= 1.3)
                except Exception:
                    qc_stats["qc_suvr_median_pass"] = ""

                try:
                    # z-score check (inside-brain mean≈0, SD≈1)
                    z_ok = abs(float(qc_stats.get("qc_z_mean", 0.0))) < 0.1 and abs(float(qc_stats.get("qc_z_std", 1.0)) - 1.0) < 0.1
                    qc_stats["qc_z_pass"] = z_ok
                except Exception:
                    qc_stats["qc_z_pass"] = ""

                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                logging.info(f"[{sub_id}] QC stats written (status: {qc_stats['crop_status']})")
            except Exception as e:
                logging.error(f"[{sub_id}] Failed to write QC CSV: {e}")

            # Write provenance manifest
            try:
                if args.write_manifest:
                    manifest = {
                        "subject_id": sub_id,
                        "registration_method": used_method,
                        "template_path": str(template_path),
                        "brain_mask_path": str(brain_mask_path),
                        "cerebellum_mask_path": str(args.cerebellum_mask) if args.cerebellum_mask else None,
                        "suvr_reference": qc_stats.get("suvr_reference", None),
                        "suvr_ref_stat": qc_stats.get("suvr_ref_stat", None),
                        "reference_mean": qc_stats.get("reference_mean", None),
                        "reference_median": qc_stats.get("reference_median", None),
                        "reference_value": qc_stats.get("qc_reference_value", None),
                        "smoothing_fwhm": qc_stats.get("smoothing_fwhm", 0.0),
                        "presmooth_fwhm": float(args.presmooth_fwhm),
                        "transforms": {
                            "affine": str(transform_affine),
                            "warp": str(transform_warp),
                        },
                        "outputs": {
                            "warped": str(fullres_warped_path),
                            "suvr": str(suvr_path),
                            "zmap": str(z_img_path),
                        },
                        "qc": {
                            "registration_overlap": qc_stats.get("qc_registration_overlap", None),
                            "registration_pass": qc_stats.get("qc_registration_pass", None),
                            "reference_mean": qc_stats.get("qc_reference_mean", None),
                            "reference_pass": qc_stats.get("qc_reference_pass", None),
                            "suvr_median": qc_stats.get("suvr_median", None),
                            "suvr_median_pass": qc_stats.get("qc_suvr_median_pass", None),
                            "z_mean": qc_stats.get("qc_z_mean", None),
                            "z_std": qc_stats.get("qc_z_std", None),
                            "z_pass": qc_stats.get("qc_z_pass", None),
                        },
                    }
                    with open(out_sub_dir / f"{sub_id}_provenance.json", "w") as f:
                        json.dump(manifest, f, indent=2)
                    qc_stats["provenance_manifest"] = str(out_sub_dir / f"{sub_id}_provenance.json")
            except Exception as e:
                logging.warning(f"[{sub_id}] Failed to write manifest: {e}")

    logging.info("PET preprocessing pipeline completed")


if __name__ == "__main__":
    main()
