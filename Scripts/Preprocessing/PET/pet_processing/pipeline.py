"""Main pipeline class for PET preprocessing."""

import os
import sys
from pathlib import Path
from typing import Dict

import nibabel as nib
from nibabel.processing import resample_from_to

from .config import PipelineConfig, QC_HEADER
from .utils import setup_logging, write_qc_csv, ensure_matched_affine_and_shape
from .processing import (
    process_static,
    process_lowres,
    process_registration,
    process_fullres,
    process_suvr,
    process_cropping
)

class PETPreprocessingPipeline:
    """Main class for PET image preprocessing pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logging(config.output_root)
        self._setup_environment()
        
    def _setup_environment(self):
        """Set up environment variables for threading."""
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.config.threads)
        os.environ["OMP_NUM_THREADS"] = str(self.config.threads)
        os.environ["MKL_NUM_THREADS"] = str(self.config.threads)
    
    def _validate_paths(self):
        """Validate all required paths exist."""
        if not self.config.input_root.is_dir():
            raise FileNotFoundError(f"Input root not found: {self.config.input_root}")
        if not self.config.lowres_template.is_file():
            raise FileNotFoundError(f"Low-res template not found: {self.config.lowres_template}")
        if not self.config.cerebellum_mask.is_file():
            raise FileNotFoundError(f"Cerebellum mask not found: {self.config.cerebellum_mask}")
        if not self.config.brain_mask_template.is_file():
            raise FileNotFoundError(f"Brain mask template not found: {self.config.brain_mask_template}")
    
    def _load_templates(self):
        """Load and validate template images."""
        try:
            self.tmpl_img = nib.load(str(self.config.lowres_template))
            self.cereb_img = nib.load(str(self.config.cerebellum_mask))
            self.brain_img = nib.load(str(self.config.brain_mask_template))
            
            # Resample masks if needed
            if not ensure_matched_affine_and_shape(self.cereb_img, self.tmpl_img):
                self.logger.info(f"Resampling cerebellum mask {self.cereb_img.shape} → {self.tmpl_img.shape}")
                self.cereb_img = resample_from_to(self.cereb_img, self.tmpl_img, order=0)
            
            if not ensure_matched_affine_and_shape(self.brain_img, self.tmpl_img):
                self.logger.info(f"Resampling brain mask {self.brain_img.shape} → {self.tmpl_img.shape}")
                self.brain_img = resample_from_to(self.brain_img, self.tmpl_img, order=0)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load templates: {e}")
    
    def _prepare_qc_files(self):
        """Prepare QC CSV files."""
        self.master_qc_path = self.config.output_root / "qc_stats_master.csv"
        if not self.master_qc_path.exists():
            self.logger.info(f"Creating master QC CSV: {self.master_qc_path}")
            write_qc_csv(QC_HEADER, {k: "" for k in QC_HEADER}, self.master_qc_path, append=False)
            self.master_qc_path.unlink()
            write_qc_csv(QC_HEADER, {}, self.master_qc_path, append=False)
    
    def process_subject(self, subject_dir: Path) -> bool:
        """Process a single subject directory."""
        sub_id = subject_dir.name
        self.logger.info(f"Processing subject: {sub_id}")
        
        # Initialize QC stats
        qc_stats = {key: "" for key in QC_HEADER}
        qc_stats["subject_id"] = sub_id
        
        try:
            # Create output directory
            out_sub_dir = self.config.output_root / subject_dir.parent.name / sub_id
            out_sub_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output paths
            static_path = out_sub_dir / f"{sub_id}_static.nii.gz"
            lowres_path = out_sub_dir / f"{sub_id}_lowres.nii.gz"
            lowres_warped_path = out_sub_dir / f"{sub_id}_lowres_Warped.nii.gz"
            fullres_warped_path = out_sub_dir / f"{sub_id}_fullres_warped.nii.gz"
            suvr_path = out_sub_dir / f"{sub_id}_SUVR.nii.gz"
            suvr_cropped_path = out_sub_dir / f"{sub_id}_SUVR_cropped.nii.gz"
            qc_csv_path = out_sub_dir / "qc_stats.csv"
            
            # Process subject
            process_static(sub_id, subject_dir, static_path)
            
            lowres_img, lowres_stats = process_lowres(sub_id, static_path, lowres_path, self.tmpl_img)
            qc_stats.update({
                "lowres_min": lowres_stats["min"],
                "lowres_max": lowres_stats["max"],
                "lowres_nonzero_frac": lowres_stats["nonzero_frac"],
                "lowres_mean": lowres_stats["mean"],
                "lowres_median": lowres_stats["median"],
                "lowres_std": lowres_stats["std"]
            })
            
            lowres_warped_img, lowres_warped_stats = process_registration(
                sub_id, lowres_path, lowres_warped_path, self.config.lowres_template, self.config
            )
            qc_stats.update({
                "lowres_warped_min": lowres_warped_stats["min"],
                "lowres_warped_max": lowres_warped_stats["max"],
                "lowres_warped_nonzero_frac": lowres_warped_stats["nonzero_frac"],
                "lowres_warped_mean": lowres_warped_stats["mean"],
                "lowres_warped_median": lowres_warped_stats["median"],
                "lowres_warped_std": lowres_warped_stats["std"]
            })
            
            fullres_warped_img, fullres_warped_stats = process_fullres(
                sub_id, static_path, lowres_warped_path, fullres_warped_path, self.config.lowres_template
            )
            qc_stats.update({
                "fullres_warped_min": fullres_warped_stats["min"],
                "fullres_warped_max": fullres_warped_stats["max"],
                "fullres_warped_nonzero_frac": fullres_warped_stats["nonzero_frac"],
                "fullres_warped_mean": fullres_warped_stats["mean"],
                "fullres_warped_median": fullres_warped_stats["median"],
                "fullres_warped_std": fullres_warped_stats["std"]
            })
            
            suvr_img, suvr_stats = process_suvr(sub_id, fullres_warped_path, suvr_path, self.cereb_img)
            qc_stats.update({
                "suvr_min": suvr_stats["min"],
                "suvr_max": suvr_stats["max"],
                "suvr_nonzero_frac": suvr_stats["nonzero_frac"],
                "suvr_mean": suvr_stats["mean"],
                "suvr_median": suvr_stats["median"],
                "suvr_std": suvr_stats["std"]
            })
            
            _, crop_status = process_cropping(
                sub_id, suvr_path, suvr_cropped_path, self.brain_img, self.config
            )
            qc_stats["crop_status"] = crop_status
            
            # Write QC stats
            write_qc_csv(QC_HEADER, qc_stats, qc_csv_path, append=False)
            write_qc_csv(QC_HEADER, qc_stats, self.master_qc_path, append=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing subject {sub_id}: {e}")
            qc_stats["crop_status"] = "PROCESSING_ERROR"
            write_qc_csv(QC_HEADER, qc_stats, qc_csv_path, append=False)
            write_qc_csv(QC_HEADER, qc_stats, self.master_qc_path, append=True)
            return False
    
    def run(self):
        """Run the full preprocessing pipeline."""
        try:
            self._validate_paths()
            self._load_templates()
            self._prepare_qc_files()
            
            # Process each subject
            for cohort_dir in sorted(self.config.input_root.iterdir()):
                if not cohort_dir.is_dir():
                    continue
                    
                self.logger.info(f"Processing cohort: {cohort_dir.name}")
                
                for subject_dir in sorted(cohort_dir.iterdir()):
                    if not subject_dir.is_dir():
                        continue
                        
                    self.process_subject(subject_dir)
            
            self.logger.info("PET preprocessing pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            sys.exit(1) 