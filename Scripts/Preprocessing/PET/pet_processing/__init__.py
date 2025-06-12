"""PET preprocessing package."""

from .config import PipelineConfig, QC_HEADER
from .pipeline import PETPreprocessingPipeline
from .processing import (
    process_static,
    process_lowres,
    process_registration,
    process_fullres,
    process_suvr,
    process_cropping
)
from .utils import (
    setup_logging,
    compute_voxel_stats,
    write_qc_csv,
    ensure_matched_affine_and_shape,
    crop_around_com
)
from pathlib import Path

__all__ = [
    'PipelineConfig',
    'QC_HEADER',
    'PETPreprocessingPipeline',
    'process_static',
    'process_lowres',
    'process_registration',
    'process_fullres',
    'process_suvr',
    'process_cropping',
    'setup_logging',
    'compute_voxel_stats',
    'write_qc_csv',
    'ensure_matched_affine_and_shape',
    'crop_around_com'
]

config = PipelineConfig(
    input_root=Path("/home/jsto890/reseng202500013-ndd-ml/data/raw/PET"),
    output_root=Path("/home/jsto890/reseng202500013-ndd-ml/data/preprocessed/PET"),
    lowres_template=Path("~/reseng202500013-ndd-ml/P4P/Templates/PET/FDG_PET.nii.gz"),
    cerebellum_mask=Path("~/reseng202500013-ndd-ml/P4P/Templates/PET/cereb_mask25_bin.nii.gz"),
    brain_mask_template=Path("~/reseng202500013-ndd-ml/P4P/Templates/PET/MNI152_T1_1mm_brain_mask.nii.gz"),
    crop_dims=(160, 192, 192),
    threads=8
)

pipeline = PETPreprocessingPipeline(config)
pipeline.run() 