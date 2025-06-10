#!/usr/bin/env python3
"""
pure_pet_standardise.py (saved as 04JUNE.py)

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
    --lowres_template ~/reseng202500013-ndd-ml/P4P/Templates/PET_refs/FDG-PET-template_padded.nii.gz \
    --cerebellum_mask ~/reseng202500013-ndd-ml/P4P/Templates/PET_refs/cereb_in_petspace.nii.gz \
    --brain_mask_template ~/reseng202500013-ndd-ml/P4P/Templates/PET_refs/brain_in_petspace.nii.gz \
    --crop_dims 160 192 192 \
    --threads 8
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_to_output


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
        description="Preprocess ADNI PET scans: 4D→3D static, resample, register, SUVR, crop (COM-based), QC."
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
        required=True,
        help="Path to 2 mm isotropic PET template (160×192×192)"
    )
    parser.add_argument(
        "--cerebellum_mask",
        type=Path,
        required=True,
        help="Path to binary cerebellum mask (160×192×192)"
    )
    parser.add_argument(
        "--brain_mask_template",
        type=Path,
        required=True,
        help="Path to whole-brain mask in PET template space (160×192×192)"
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

    # 4. Validate paths
    if not args.input_root.is_dir():
        logging.error(f"Input root not found: {args.input_root}")
        sys.exit(1)
    if not args.lowres_template.is_file():
        logging.error(f"Low-res template not found: {args.lowres_template}")
        sys.exit(1)
    if not args.cerebellum_mask.is_file():
        logging.error(f"Cerebellum mask not found: {args.cerebellum_mask}")
        sys.exit(1)
    if not args.brain_mask_template.is_file():
        logging.error(f"Brain mask template not found: {args.brain_mask_template}")
        sys.exit(1)

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

    # 6. Load static template, cerebellum mask, and whole-brain mask
    try:
        tmpl_img = nib.load(str(args.lowres_template))
    except Exception as e:
        logging.error(f"Failed to load template: {e}")
        sys.exit(1)
    try:
        cereb_img = nib.load(str(args.cerebellum_mask))
    except Exception as e:
        logging.error(f"Failed to load cerebellum mask: {e}")
        sys.exit(1)
    try:
        brain_img = nib.load(str(args.brain_mask_template))
    except Exception as e:
        logging.error(f"Failed to load brain mask template: {e}")
        sys.exit(1)

    # 7. Iterate over cohorts (directories under input_root)
    for cohort_dir in sorted(args.input_root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        cohort_name = cohort_dir.name
        logging.info(f"Processing cohort: {cohort_name}")

        for subject_dir in sorted(cohort_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            sub_id = subject_dir.name
            logging.info(f"--- Subject: {sub_id} ---")

            # 7.1. Locate the single <sub-ID>.nii file in subject_dir
            raw_candidates = list(subject_dir.glob(f"{sub_id}.nii*"))
            if len(raw_candidates) == 0:
                logging.error(f"[{sub_id}] No {sub_id}.nii file found; skipping.")
                continue
            raw_nifti = raw_candidates[0]

            # 7.2. Create output subject directory
            out_sub_dir = args.output_root / cohort_name / sub_id
            out_sub_dir.mkdir(parents=True, exist_ok=True)

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
                continue

            # ---------------------
            # STEP 1: Resample static→ Low-Res (2 mm)
            # ---------------------
            try:
                logging.info(f"[{sub_id}] Resampling static to 2 mm isotropic (low-res)")
                lowres_img = resample_to_output(static_img, (2.0, 2.0, 2.0))
                nib.save(lowres_img, str(lowres_path))
                if not lowres_path.is_file():
                    raise RuntimeError("Failed to save low-res image")

                lowres_data = lowres_img.get_fdata(dtype=np.float32)
                stats = compute_voxel_stats(lowres_data)
                qc_stats.update({
                    "lowres_min": stats["min"],
                    "lowres_max": stats["max"],
                    "lowres_nonzero_frac": stats["nonzero_frac"],
                    "lowres_mean": stats["mean"],
                    "lowres_median": stats["median"],
                    "lowres_std": stats["std"]
                })
                logging.debug(f"[{sub_id}] Low-res stats: {stats}")
            except Exception as e:
                logging.error(f"[{sub_id}] Error during low-res resampling: {e}")
                qc_stats["crop_status"] = "LOWRES_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                continue

            # ---------------------
            # STEP 2: ANTs Rigid + SyN Registration (Low-Res → Template)
            # ---------------------
            out_prefix = str(out_sub_dir / f"{sub_id}_lowres_")
            ants_reg_cmd = [
                "antsRegistration",
                "--dimensionality", "3",
                "--float", "1",
                "--output", f"[{out_prefix},{out_prefix}Warped.nii.gz]",
                "--interpolation", "Linear",
                # Rigid stage
                "--transform", "Rigid[0.1]",
                "--metric", f"MI[{args.lowres_template},{lowres_path},1,32]",
                "--convergence", "1000x500x250x100",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox",
                # SyN stage
                "--transform", "SyN[0.1,3,0]",
                "--metric", f"CC[{args.lowres_template},{lowres_path},1,4]",
                "--convergence", "100x70x50x20",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox"
            ]
            try:
                logging.info(f"[{sub_id}] About to run antsRegistration with the following command:")
                logging.info("  " + " \\\n  ".join(ants_reg_cmd))

                result = subprocess.run(
                    ants_reg_cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    logging.error(f"[{sub_id}] antsRegistration stderr:\n{result.stderr.strip()}")
                    raise RuntimeError("antsRegistration failed")
                    # At this point you can copy-paste the logged command exactly into your shell
                    # to see the full ANTs help/error message.
                if not lowres_warped_path.is_file():
                    raise RuntimeError("antsRegistration did not produce low-res warped image")

                lowres_warped_img = nib.load(str(lowres_warped_path))
                lrw_data = lowres_warped_img.get_fdata(dtype=np.float32)
                stats = compute_voxel_stats(lrw_data)
                qc_stats.update({
                    "lowres_warped_min": stats["min"],
                    "lowres_warped_max": stats["max"],
                    "lowres_warped_nonzero_frac": stats["nonzero_frac"],
                    "lowres_warped_mean": stats["mean"],
                    "lowres_warped_median": stats["median"],
                    "lowres_warped_std": stats["std"]
                })
                logging.debug(f"[{sub_id}] Low-res warped stats: {stats}")
            except Exception as e:
                logging.error(f"[{sub_id}] Error after antsRegistration: {e}")
                qc_stats["crop_status"] = "REGISTRATION_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                continue

            # ---------------------
            # STEP 3: Apply Transforms to Static (Full-Res → MNI)
            # ---------------------
            try:
                logging.info(f"[{sub_id}] Applying ANTs transforms to static")
                # ​Make sure we use the matrix first (with “,0” to disable inversion),
                # and then the warp field, so ANTs applies them in the correct forward chain.
                transform_affine = f"{out_prefix}0GenericAffine.mat"
                transform_warp   = f"{out_prefix}1Warp.nii.gz"
                if not Path(transform_affine).is_file() or not Path(transform_warp).is_file():
                    raise RuntimeError("Missing transform files from registration")

                ants_apply_cmd = [
                    "antsApplyTransforms",
                    "--dimensionality", "3",
                    "--float", "1",
                    "--input", str(static_path),
                    "--reference-image", str(args.lowres_template),
                    "--output", str(fullres_warped_path),
                    "--interpolation", "Linear",
                    # 1) apply affine (forward), do NOT invert – hence the “,0”
                    "--transform", f"[{transform_affine},0]",
                    # 2) then apply the nonlinear warp
                    "--transform", transform_warp
                ]
                subprocess.run(ants_apply_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
            except subprocess.CalledProcessError as e:
                logging.error(f"[{sub_id}] antsApplyTransforms failed: {e}")
                qc_stats["crop_status"] = "APPLY_TRANSFORMS_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                continue
            except Exception as e:
                logging.error(f"[{sub_id}] Error during full-res apply: {e}")
                qc_stats["crop_status"] = "APPLY_TRANSFORMS_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                continue

            # ---------------------
            # STEP 4: Compute SUVR (Cerebellum Normalization)
            # ---------------------
            try:
                logging.info(f"[{sub_id}] Computing SUVR")
                if not ensure_matched_affine_and_shape(fullres_warped_img, cereb_img):
                    raise RuntimeError("Cerebellum mask/template mismatch")

                cereb_array = cereb_img.get_fdata(dtype=np.float32)
                cereb_bool = cereb_array > 0
                if not np.any(cereb_bool):
                    raise RuntimeError("Empty cerebellum mask")

                ref_values = frw_data[cereb_bool]
                ref_mean = float(np.mean(ref_values))
                if np.isnan(ref_mean) or ref_mean <= 0:
                    raise RuntimeError(f"Invalid cerebellum reference mean = {ref_mean:.4f}")

                suvr_array = frw_data / (ref_mean + 1e-8)
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
            except Exception as e:
                logging.error(f"[{sub_id}] Error during SUVR computation: {e}")
                qc_stats["crop_status"] = "SUVR_ERROR"
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                continue

            # ---------------------
            # STEP 5: Center-Crop/Pad SUVR to target dims (COM-based)
            # ---------------------
            try:
                logging.info(f"[{sub_id}] Cropping/padding around whole-brain COM to {tuple(args.crop_dims)}")
                suvr_img = nib.load(str(suvr_path))
                if not ensure_matched_affine_and_shape(suvr_img, brain_img):
                    raise RuntimeError("Brain mask/SUVR template mismatch")

                suvr_array = suvr_img.get_fdata(dtype=np.float32)
                brain_array = brain_img.get_fdata(dtype=np.float32)
                brain_bool = brain_array > 0

                cropped_array = crop_around_com(suvr_array, brain_bool, tuple(args.crop_dims), pad_value=0.0)
                if cropped_array.shape != tuple(args.crop_dims):
                    raise RuntimeError(f"Cropped shape mismatch: got {cropped_array.shape}")

                cropped_img = nib.Nifti1Image(cropped_array, suvr_img.affine, suvr_img.header)
                nib.save(cropped_img, str(suvr_cropped_path))
                if not suvr_cropped_path.is_file():
                    raise RuntimeError("Failed to save SUVR cropped image")

                qc_stats["crop_status"] = "SUCCESS"
                logging.info(f"[{sub_id}] Cropping/padding successful")
            except Exception as e:
                logging.error(f"[{sub_id}] Error during cropping/padding: {e}")
                qc_stats["crop_status"] = "CROP_ERROR"

            # ---------------------
            # STEP 6: Write QC Stats CSVs
            # ---------------------
            try:
                write_qc_csv(qc_header, qc_stats, qc_csv_path, append=False)
                write_qc_csv(qc_header, qc_stats, master_qc_path, append=True)
                logging.info(f"[{sub_id}] QC stats written (status: {qc_stats['crop_status']})")
            except Exception as e:
                logging.error(f"[{sub_id}] Failed to write QC CSV: {e}")

    logging.info("PET preprocessing pipeline completed")


if __name__ == "__main__":
    main()
