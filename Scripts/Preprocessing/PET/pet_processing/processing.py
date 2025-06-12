"""Core processing functions for PET preprocessing pipeline."""

import subprocess
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import nibabel as nib
from scipy.ndimage import label

from .utils import compute_voxel_stats, ensure_matched_affine_and_shape, crop_around_com

def process_static(sub_id: str, subject_dir: Path, static_path: Path) -> nib.Nifti1Image:
    """Process static image from raw input."""
    raw_candidates = list(subject_dir.glob(f"{sub_id}.nii*"))
    if not raw_candidates:
        raise FileNotFoundError(f"No {sub_id}.nii file found")
        
    raw_nifti = raw_candidates[0]
    img = nib.load(str(raw_nifti))
    data = img.get_fdata(dtype=np.float32)
    
    if data.ndim == 4:
        static_arr = data.mean(axis=3)
    elif data.ndim == 3:
        static_arr = data
    else:
        raise ValueError(f"Unsupported NIfTI dimensions ({data.ndim}D); expected 3D or 4D")
        
    static_img = nib.Nifti1Image(static_arr.astype(np.float32), img.affine, img.header)
    nib.save(static_img, str(static_path))
    
    if not static_path.is_file():
        raise RuntimeError(f"Failed to save static image to {static_path}")
        
    return static_img

def process_lowres(sub_id: str, static_path: Path, lowres_path: Path, template_img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, Dict[str, float]]:
    """Resample static image to low resolution."""
    static_img = nib.load(str(static_path))
    lowres_img = nib.processing.resample_from_to(static_img, template_img, order=1)
    nib.save(lowres_img, str(lowres_path))
    
    if not lowres_path.is_file():
        raise RuntimeError("Failed to save low-res image")
        
    lowres_data = lowres_img.get_fdata(dtype=np.float32)
    stats = compute_voxel_stats(lowres_data)
    
    return lowres_img, stats

def process_registration(
    sub_id: str,
    lowres_path: Path,
    lowres_warped_path: Path,
    template_path: Path,
    config
) -> Tuple[nib.Nifti1Image, Dict[str, float]]:
    """Perform registration using ANTs."""
    out_prefix = str(lowres_path.parent / f"{sub_id}_lowres_")
    ants_reg_cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        "--output", f"[{out_prefix},{out_prefix}Warped.nii.gz]",
        "--interpolation", "Linear",
        "--initial-moving-transform", f"[{template_path},{lowres_path},1]",
        # Rigid stage
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{template_path},{lowres_path},1,32]",
        "--convergence", "1000x500x250x100",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        # SyN stage
        "--transform", "BSplineSyN[0.1,26,0]",
        "--metric", f"CC[{template_path},{lowres_path},1,4]",
        "--convergence", "50x30x20x10",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "4x3x2x1vox"
    ]
    
    result = subprocess.run(
        ants_reg_cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"antsRegistration failed: {result.stderr.strip()}")
        
    if not lowres_warped_path.is_file():
        raise RuntimeError("antsRegistration did not produce low-res warped image")
        
    lowres_warped_img = nib.load(str(lowres_warped_path))
    lrw_data = lowres_warped_img.get_fdata(dtype=np.float32)
    stats = compute_voxel_stats(lrw_data)
    
    return lowres_warped_img, stats

def process_fullres(
    sub_id: str,
    static_path: Path,
    lowres_warped_path: Path,
    fullres_warped_path: Path,
    template_path: Path
) -> Tuple[nib.Nifti1Image, Dict[str, float]]:
    """Apply transforms to full resolution image."""
    out_prefix = str(lowres_warped_path.parent / f"{sub_id}_lowres_")
    transform_affine = f"{out_prefix}0GenericAffine.mat"
    transform_warp = f"{out_prefix}1Warp.nii.gz"
    
    if not Path(transform_affine).is_file() or not Path(transform_warp).is_file():
        raise RuntimeError("Missing transform files from registration")
        
    ants_apply_cmd = [
        "antsApplyTransforms",
        "--dimensionality", "3",
        "--float", "1",
        "--input", str(static_path),
        "--reference-image", str(template_path),
        "--output", str(fullres_warped_path),
        "--interpolation", "Linear",
        "--transform", f"[{transform_affine},0]",
        "--transform", transform_warp
    ]
    
    subprocess.run(ants_apply_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not fullres_warped_path.is_file():
        raise RuntimeError("antsApplyTransforms did not produce full-res warped image")
        
    fullres_warped_img = nib.load(str(fullres_warped_path))
    frw_data = fullres_warped_img.get_fdata(dtype=np.float32)
    stats = compute_voxel_stats(frw_data)
    
    return fullres_warped_img, stats

def process_suvr(
    sub_id: str,
    fullres_warped_path: Path,
    suvr_path: Path,
    cereb_img: nib.Nifti1Image
) -> Tuple[nib.Nifti1Image, Dict[str, float]]:
    """Compute SUVR using cerebellum mask."""
    fullres_warped_img = nib.load(str(fullres_warped_path))
    if not ensure_matched_affine_and_shape(fullres_warped_img, cereb_img):
        raise RuntimeError("Cerebellum mask/template mismatch")
        
    cereb_array = cereb_img.get_fdata(dtype=np.float32)
    cereb_bool = cereb_array > 0
    
    if not np.any(cereb_bool):
        raise RuntimeError("Empty cerebellum mask")
        
    frw_data = fullres_warped_img.get_fdata(dtype=np.float32)
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
    return suvr_img, stats

def process_cropping(
    sub_id: str,
    suvr_path: Path,
    suvr_cropped_path: Path,
    brain_img: nib.Nifti1Image,
    config
) -> Tuple[nib.Nifti1Image, str]:
    """Crop SUVR image around center of mass."""
    suvr_img = nib.load(str(suvr_path))
    suvr_array = suvr_img.get_fdata(dtype=np.float32)
    
    # Compute data-driven threshold
    nonzero = suvr_array[suvr_array > 0]
    if nonzero.size < config.MIN_NONZERO_VOXELS:
        raise RuntimeError("Too few nonzero SUVR voxels for reliable mask")
        
    thresh = np.percentile(nonzero, config.SUVR_THRESHOLD_PERCENTILE)
    
    # Build initial mask & keep only largest CC
    init_mask = suvr_array > thresh
    labels, num = label(init_mask)
    
    if num == 0:
        raise RuntimeError("Empty mask after threshold")
        
    # Pick largest component (ignore background label 0)
    counts = np.bincount(labels.flat)
    counts[0] = 0
    main_lbl = counts.argmax()
    subj_mask = (labels == main_lbl)
    
    # Check mask size
    vol_frac = subj_mask.sum() / subj_mask.size
    if vol_frac < config.MIN_MASK_VOLUME_FRACTION:
        coords = np.argwhere(brain_img.get_fdata() > 0)
    else:
        coords = np.argwhere(subj_mask)
        
    com = coords.mean(axis=0).astype(int)
    
    # Crop/pad
    cropped_array = crop_around_com(suvr_array, subj_mask, config.crop_dims, pad_value=0.0)
    
    # Fix affine translation
    vs = np.array([suvr_img.affine[0,0], suvr_img.affine[1,1], suvr_img.affine[2,2]])
    z_start = int(com[0] - config.crop_dims[0] // 2)
    y_start = int(com[1] - config.crop_dims[1] // 2)
    x_start = int(com[2] - config.crop_dims[2] // 2)
    
    new_affine = suvr_img.affine.copy()
    new_affine[:3,3] += np.array([x_start, y_start, z_start]) * vs
    
    # Save
    cropped_img = nib.Nifti1Image(cropped_array, new_affine, suvr_img.header)
    nib.save(cropped_img, str(suvr_cropped_path))
    
    if not suvr_cropped_path.is_file():
        raise RuntimeError("Failed to save SUVR cropped image")
        
    return cropped_img, "SUCCESS" 