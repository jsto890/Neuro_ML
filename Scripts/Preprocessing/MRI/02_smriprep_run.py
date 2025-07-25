#!/usr/bin/env python3
"""
sMRI Preprocessing Pipeline Runner
=================================

This script checks for existing smriprep outputs and only processes subjects
that haven't been completed yet. It looks for both the preprocessed T1w image
and brain mask to determine if a subject is complete.

Usage:
    python 02_smriprep_run.py

Requirements:
    - smriprep installed and accessible
    - FS_LICENSE environment variable set
    - Sufficient disk space and memory
"""

import os
import sys
import subprocess
import glob
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smriprep_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SMRIPrepRunner:
    """Manages sMRI preprocessing with smriprep."""
    
    def __init__(self, 
                 raw_data_dir="/home/jsto890/reseng202500013-ndd-ml/data/raw/MRI/ADNI/CN",
                 output_dir="/home/jsto890/reseng202500013-ndd-ml/data/preprocessed/MRI",
                 work_dir="/home/jsto890/reseng202500013-ndd-ml/data/intermediate/smri",
                 fs_license=None,
                 nprocs=8,
                 omp_nthreads=4,
                 mem_gb=96):
        """
        Initialize the sMRI preprocessing runner.
        
        Args:
            raw_data_dir: Directory containing raw MRI data
            output_dir: Directory for smriprep outputs
            work_dir: Directory for intermediate files
            fs_license: Path to FreeSurfer license file
            nprocs: Number of processes for smriprep
            omp_nthreads: Number of OpenMP threads
            mem_gb: Memory limit in GB
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.work_dir = Path(work_dir)
        self.fs_license = fs_license or os.environ.get('FS_LICENSE')
        self.nprocs = nprocs
        self.omp_nthreads = omp_nthreads
        self.mem_gb = mem_gb
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self):
        """Validate that required paths exist."""
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")
        
        if not self.fs_license or not Path(self.fs_license).exists():
            raise FileNotFoundError(f"FreeSurfer license file not found: {self.fs_license}")
        
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Work directory: {self.work_dir}")
        logger.info(f"FreeSurfer license: {self.fs_license}")
    
    def get_subject_list(self):
        """Get list of all subjects from raw data directory."""
        subjects = []
        
        # Look for subject directories (sub-*)
        subject_dirs = list(self.raw_data_dir.glob("sub-*"))
        
        for subject_dir in subject_dirs:
            if subject_dir.is_dir():
                subject_id = subject_dir.name
                subjects.append(subject_id)
        
        logger.info(f"Found {len(subjects)} subjects in raw data directory")
        return sorted(subjects)
    
    def check_subject_completion(self, subject_id):
        """
        Check if a subject has been fully processed by smriprep.
        
        Args:
            subject_id: Subject ID (e.g., 'sub-I166845')
            
        Returns:
            bool: True if subject is complete, False otherwise
        """
        subject_output_dir = self.output_dir / "smriprep" / subject_id / "anat"
        
        if not subject_output_dir.exists():
            logger.debug(f"Subject output directory does not exist: {subject_output_dir}")
            return False
        
        # Check for required files
        required_files = [
            f"{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz",
            f"{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
        ]
        
        missing_files = []
        for required_file in required_files:
            file_path = subject_output_dir / required_file
            if not file_path.exists():
                missing_files.append(required_file)
        
        if missing_files:
            logger.debug(f"Subject {subject_id} missing files: {missing_files}")
            return False
        
        logger.debug(f"Subject {subject_id} is complete")
        return True
    
    def get_incomplete_subjects(self):
        """Get list of subjects that need processing."""
        all_subjects = self.get_subject_list()
        incomplete_subjects = []
        
        logger.info("Checking completion status of all subjects...")
        
        for subject_id in all_subjects:
            if not self.check_subject_completion(subject_id):
                incomplete_subjects.append(subject_id)
        
        logger.info(f"Found {len(incomplete_subjects)} subjects that need processing")
        return incomplete_subjects
    
    def run_smriprep(self, subject_list=None):
        """
        Run smriprep on specified subjects or all incomplete subjects.
        
        Args:
            subject_list: List of specific subjects to process (if None, process all incomplete)
        """
        if subject_list is None:
            subjects_to_process = self.get_incomplete_subjects()
        else:
            subjects_to_process = subject_list
        
        if not subjects_to_process:
            logger.info("No subjects need processing. All subjects are complete!")
            return
        
        logger.info(f"Processing {len(subjects_to_process)} subjects with smriprep")
        
        # Build smriprep command
        cmd = [
            "smriprep",
            str(self.raw_data_dir),
            str(self.output_dir),
            "participant",
            "--fs-license-file", self.fs_license,
            "--nprocs", str(self.nprocs),
            "--omp-nthreads", str(self.omp_nthreads),
            "--mem-gb", str(self.mem_gb),
            "--output-spaces", "MNI152NLin2009cAsym:res-2",
            "--resource-monitor",
            "--fs-no-reconall",
            "-w", str(self.work_dir)
        ]
        
        # Add participant filter if specific subjects
        if len(subjects_to_process) < len(self.get_subject_list()):
            cmd.extend(["--participant-label"] + [sub.replace("sub-", "") for sub in subjects_to_process])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run smriprep
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=dict(os.environ, OMP_NUM_THREADS=str(self.omp_nthreads))
            )
            
            logger.info("smriprep completed successfully!")
            logger.info(f"stdout: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"smriprep failed with exit code {e.returncode}")
            logger.error(f"stderr: {e.stderr}")
            raise
    
    def verify_processing(self):
        """Verify that all subjects have been processed correctly."""
        all_subjects = self.get_subject_list()
        complete_subjects = []
        incomplete_subjects = []
        
        logger.info("Verifying processing completion...")
        
        for subject_id in all_subjects:
            if self.check_subject_completion(subject_id):
                complete_subjects.append(subject_id)
            else:
                incomplete_subjects.append(subject_id)
        
        logger.info(f"Processing verification complete:")
        logger.info(f"  Complete subjects: {len(complete_subjects)}")
        logger.info(f"  Incomplete subjects: {len(incomplete_subjects)}")
        
        if incomplete_subjects:
            logger.warning(f"Incomplete subjects: {incomplete_subjects}")
        
        return complete_subjects, incomplete_subjects

    def process_all_datasets(self):
        """Process all specified datasets with their subject limits."""
        datasets = [
            {
                'name': 'ADNI_AD',
                'raw_dir': '/home/jsto890/reseng202500013-ndd-ml/data/raw/MRI/ADNI/AD',
                'subject_limit': None  # Process all
            },
            {
                'name': 'ADNI_CN', 
                'raw_dir': '/home/jsto890/reseng202500013-ndd-ml/data/raw/MRI/ADNI/CN',
                'subject_limit': None  # Process all
            },
            {
                'name': 'PPMI_CN',
                'raw_dir': '/home/jsto890/reseng202500013-ndd-ml/data/raw/MRI/PPMI/CN', 
                'subject_limit': None  # Process all
            },
            {
                'name': 'PPMI_PD',
                'raw_dir': '/home/jsto890/reseng202500013-ndd-ml/data/raw/MRI/PPMI/PD',
                'subject_limit': 1152  # Limit to 1152 subjects
            }
        ]
        
        total_processed = 0
        
        for dataset in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset: {dataset['name']}")
            logger.info(f"{'='*60}")
            
            try:
                # Update raw data directory for this dataset
                self.raw_data_dir = Path(dataset['raw_dir'])
                
                if not self.raw_data_dir.exists():
                    logger.error(f"Dataset directory not found: {self.raw_data_dir}")
                    continue
                
                logger.info(f"Updated raw data directory to: {self.raw_data_dir}")
                
                # Get subjects for this dataset
                all_subjects = self.get_subject_list()
                incomplete_subjects = self.get_incomplete_subjects()
                
                logger.info(f"Dataset {dataset['name']}:")
                logger.info(f"  Total subjects: {len(all_subjects)}")
                logger.info(f"  Incomplete subjects: {len(incomplete_subjects)}")
                
                if not incomplete_subjects:
                    logger.info(f"All subjects in {dataset['name']} are complete!")
                    continue
                
                # Apply subject limit if specified
                if dataset['subject_limit'] is not None:
                    if len(incomplete_subjects) > dataset['subject_limit']:
                        incomplete_subjects = incomplete_subjects[:dataset['subject_limit']]
                        logger.info(f"Limited to {dataset['subject_limit']} subjects")
                
                # Process this dataset
                logger.info(f"Processing {len(incomplete_subjects)} subjects from {dataset['name']}")
                
                # Run smriprep for this dataset
                self.run_smriprep(subject_list=incomplete_subjects)
                
                # Verify completion for this dataset
                complete, incomplete = self.verify_processing()
                total_processed += len(complete)
                
                logger.info(f"Dataset {dataset['name']} completed:")
                logger.info(f"  Successfully processed: {len(complete)}")
                logger.info(f"  Still incomplete: {len(incomplete)}")
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset['name']}: {e}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ALL DATASETS COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total subjects processed: {total_processed}")
        
        return total_processed

def main():
    """Main function to run the sMRI preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Run sMRI preprocessing with smriprep")
    parser.add_argument("--raw-dir", 
                       default="/home/jsto890/reseng202500013-ndd-ml/data/raw/MRI/ADNI/CN",
                       help="Raw data directory")
    parser.add_argument("--output-dir",
                       default="/home/jsto890/reseng202500013-ndd-ml/data/preprocessed/MRI",
                       help="Output directory")
    parser.add_argument("--work-dir",
                       default="/home/jsto890/reseng202500013-ndd-ml/data/intermediate/smri",
                       help="Work directory for intermediate files")
    parser.add_argument("--fs-license",
                       help="Path to FreeSurfer license file")
    parser.add_argument("--nprocs", type=int, default=8,
                       help="Number of processes")
    parser.add_argument("--omp-nthreads", type=int, default=4,
                       help="Number of OpenMP threads")
    parser.add_argument("--mem-gb", type=int, default=96,
                       help="Memory limit in GB")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check completion status, don't run smriprep")
    parser.add_argument("--subjects", nargs="+",
                       help="Specific subjects to process")
    parser.add_argument("--all-datasets", action="store_true",
                       help="Process all datasets automatically (ADNI/AD, ADNI/CN, PPMI/CN, PPMI/PD with 1152 limit)")
    
    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = SMRIPrepRunner(
            raw_data_dir=args.raw_dir,
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            fs_license=args.fs_license,
            nprocs=args.nprocs,
            omp_nthreads=args.omp_nthreads,
            mem_gb=args.mem_gb
        )
        
        if args.all_datasets:
            # Process all datasets automatically
            logger.info("Starting automatic processing of all datasets...")
            total_processed = runner.process_all_datasets()
            logger.info(f"🎉 All datasets completed! Total subjects processed: {total_processed}")
            
        elif args.check_only:
            # Only check completion status
            complete, incomplete = runner.verify_processing()
            logger.info("Check-only mode completed")
        else:
            # Run smriprep on specified or incomplete subjects
            runner.run_smriprep(subject_list=args.subjects)
            
            # Verify processing
            complete, incomplete = runner.verify_processing()
            
            if not incomplete:
                logger.info("🎉 All subjects processed successfully!")
            else:
                logger.warning(f"⚠️  {len(incomplete)} subjects still incomplete")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
