"""Main entry point for PET preprocessing pipeline."""

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import PETPreprocessingPipeline

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Preprocess PET scans: 4D→3D static, resample, register, SUVR, crop (COM-based), QC."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root folder of raw PET data"
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root folder for processed PET output"
    )
    parser.add_argument(
        "--lowres_template",
        type=Path,
        required=True,
        help="Path to 2 mm isotropic PET template"
    )
    parser.add_argument(
        "--cerebellum_mask",
        type=Path,
        required=True,
        help="Path to binary cerebellum mask"
    )
    parser.add_argument(
        "--brain_mask_template",
        type=Path,
        required=True,
        help="Path to whole-brain mask in PET template space"
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
    config = PipelineConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        lowres_template=args.lowres_template,
        cerebellum_mask=args.cerebellum_mask,
        brain_mask_template=args.brain_mask_template,
        crop_dims=tuple(args.crop_dims),
        threads=args.threads
    )
    
    pipeline = PETPreprocessingPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main() 