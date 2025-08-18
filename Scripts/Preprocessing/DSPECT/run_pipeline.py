#!/usr/bin/env python3
"""
DSPECT Preprocessing Pipeline Runner

This script runs the complete DSPECT preprocessing pipeline with proper
SPECT-specific normalization and masking approaches.

Usage:
    python run_pipeline.py --diagnosis CN --isHasel
    python run_pipeline.py --diagnosis PD
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run complete DSPECT preprocessing pipeline")
    parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True,
                        help="Diagnosis group to process")
    parser.add_argument("--isHasel", action="store_true", 
                        help="Set this flag if running on the Hasel server")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if output exists")
    parser.add_argument("--shape", type=int, nargs=3, default=[91, 109, 91],
                        help="Target shape for finalization (default: 91 109 91)")
    parser.add_argument("--intensity_norm", action="store_true",
                        help="Apply intensity normalization in step 5")
    # Add masking options
    parser.add_argument("--mask_type", type=str, choices=['occipital', 'whole_brain'], default='whole_brain',
                    help="Mask type: occipital (SPECT-specific) or whole_brain (ML-friendly)")
    parser.add_argument("--isotropic", action="store_true",
                    help="Resample to isotropic 1mm voxels in step 1")
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"🚀 Starting DSPECT Preprocessing Pipeline for {args.diagnosis}")
    print("=" * 60)
    print("📁 New Folder Structure:")
    print(f"   Step 1: {args.diagnosis}_SPECT_PPMI_reoriented")
    print(f"   Step 2: {args.diagnosis}_SPECT_PPMI_normalised")
    print(f"   Step 3: {args.diagnosis}_SPECT_PPMI_registered")
    print(f"   Step 4: {args.diagnosis}_SPECT_PPMI_masked")
    print(f"   Step 5: {args.diagnosis}_SPECT_PPMI_finalised")
    print(f"   Step 6: {args.diagnosis}_SPECT_PPMI_postprocessed")
    print("=" * 60)
    
    # Step 1: Reorientation
    cmd1 = ["python3", "1_reorient.py", "--diagnosis", args.diagnosis]
    if args.force:
        cmd1.append("--force")
    if args.isotropic:
        cmd1.append("--isotropic")
    
    if not run_command(cmd1, "Step 1: Reorientation"):
        print("❌ Pipeline failed at Step 1")
        sys.exit(1)
    
    # Step 2: Normalization (SPECT-specific)
    cmd2 = ["python3", "2_normalise.py", "--diagnosis", args.diagnosis, "--method", "reference"]
    
    if not run_command(cmd2, "Step 2: SPECT Normalization (Reference Region)"):
        print("❌ Pipeline failed at Step 2")
        sys.exit(1)
    
    # Step 3: Registration
    cmd3 = ["python3", "3_register.py", "--diagnosis", args.diagnosis]
    
    if not run_command(cmd3, "Step 3: Registration"):
        print("❌ Pipeline failed at Step 3")
        sys.exit(1)
    
    # Step 4: Masking (SPECT-specific)
    cmd4 = ["python3", "4_masking.py", "--diagnosis", args.diagnosis, "--mask_type", args.mask_type]
    
    if not run_command(cmd4, "Step 4: SPECT Masking"):
        print("❌ Pipeline failed at Step 4")
        sys.exit(1)
    
    # Step 5: Finalization
    cmd5 = ["python3", "5_padding.py", "--diagnosis", args.diagnosis, 
            "--shape", str(args.shape[0]), str(args.shape[1]), str(args.shape[2])]
    if args.intensity_norm:
        cmd5.append("--intensity_norm")
    
    if not run_command(cmd5, "Step 5: Finalization"):
        print("❌ Pipeline failed at Step 5")
        sys.exit(1)
    
    # Step 6: Postprocessing
    cmd6 = ["python3", "6_postprocess.py", "--diagnosis", args.diagnosis]
    if args.isHasel:
        cmd6.append("--isHasel")
    
    if not run_command(cmd6, "Step 6: Postprocessing"):
        print("❌ Pipeline failed at Step 6")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ DSPECT Preprocessing Pipeline completed successfully!")
    print(f"🎯 Data for {args.diagnosis} subjects is ready for machine learning!")
    print(f"📁 All steps completed with dedicated folders in /Users/jacksonschofield/Desktop/SPECT/")
    
    # Run validation
    print("\n🔍 Running ML readiness validation...")
    cmd_validate = ["python3", "testing/validate_pipeline.py", "--diagnosis", args.diagnosis]
    if args.isHasel:
        cmd_validate.append("--isHasel")
    
    run_command(cmd_validate, "ML Readiness Validation")

if __name__ == "__main__":
    main() 