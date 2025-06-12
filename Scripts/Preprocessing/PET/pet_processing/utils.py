"""Utility functions for PET preprocessing pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging with both file and console handlers."""
    logger = logging.getLogger("pet_preprocessing")
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = log_dir / "pet_preprocessing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def compute_voxel_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute voxel-wise statistics on a 3D NumPy array."""
    flat = data.flatten()
    nonzero_count = int(np.count_nonzero(flat))
    total_voxels = flat.size
    nonzero_frac = nonzero_count / total_voxels if total_voxels > 0 else 0.0

    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "nonzero_frac": nonzero_frac,
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat))
    }

def write_qc_csv(header: List[str], row: Dict[str, str], out_path: Path, append: bool = False):
    """Write a single-row CSV with given header and values."""
    df = pd.DataFrame([row], columns=header)
    if append and out_path.exists():
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, mode="w", header=True, index=False)

def ensure_matched_affine_and_shape(img1: nib.Nifti1Image, img2: nib.Nifti1Image) -> bool:
    """Check that two NIfTI images have identical shape and affine."""
    shape_match = img1.shape == img2.shape
    affine_match = np.allclose(img1.affine, img2.affine, atol=1e-6)
    return shape_match and affine_match

def crop_around_com(
    data: np.ndarray,
    mask: np.ndarray,
    target_dims: Tuple[int, int, int],
    pad_value: float = 0.0
) -> np.ndarray:
    """Crop or pad a 3D volume around the center of mass of a mask."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Empty mask: cannot compute COM for cropping.")
    
    com = coords.mean(axis=0).astype(int)
    z_t, y_t, x_t = target_dims
    Z, Y, X = data.shape

    # Compute start indices
    z_start = int(com[0] - z_t // 2)
    y_start = int(com[1] - y_t // 2)
    x_start = int(com[2] - x_t // 2)

    # Initialize output
    result = np.full(target_dims, pad_value, dtype=data.dtype)

    def compute_slices(start: int, src_dim: int, tgt_dim: int) -> Tuple[slice, slice]:
        """Compute source and target slices for cropping/padding."""
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
            
        return slice(src_s, src_s + length), slice(tgt_s, tgt_s + length)

    # Compute slices for each dimension
    src_z_slice, tgt_z_slice = compute_slices(z_start, Z, z_t)
    src_y_slice, tgt_y_slice = compute_slices(y_start, Y, y_t)
    src_x_slice, tgt_x_slice = compute_slices(x_start, X, x_t)

    # Copy data
    result[tgt_z_slice, tgt_y_slice, tgt_x_slice] = data[src_z_slice, src_y_slice, src_x_slice]
    return result 