"""Configuration module for PET preprocessing pipeline."""

from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """Configuration for the PET preprocessing pipeline."""
    input_root: Path
    output_root: Path
    lowres_template: Path
    cerebellum_mask: Path
    brain_mask_template: Path
    crop_dims: Tuple[int, int, int]
    threads: int
    
    # Processing parameters
    SUVR_THRESHOLD_PERCENTILE: float = 25.0
    MIN_NONZERO_VOXELS: int = 100
    MIN_MASK_VOLUME_FRACTION: float = 0.10
    REGISTRATION_METRIC: str = "CC"
    REGISTRATION_SHRINK_FACTORS: List[int] = [8, 4, 2, 1]
    REGISTRATION_SMOOTHING_SIGMAS: List[float] = [4, 3, 2, 1]

# QC CSV header
QC_HEADER = [
    "subject_id",
    "lowres_min", "lowres_max", "lowres_nonzero_frac", "lowres_mean", "lowres_median", "lowres_std",
    "lowres_warped_min", "lowres_warped_max", "lowres_warped_nonzero_frac", "lowres_warped_mean",
    "lowres_warped_median", "lowres_warped_std",
    "fullres_warped_min", "fullres_warped_max", "fullres_warped_nonzero_frac", "fullres_warped_mean",
    "fullres_warped_median", "fullres_warped_std",
    "suvr_min", "suvr_max", "suvr_nonzero_frac", "suvr_mean", "suvr_median", "suvr_std",
    "crop_status"
] 