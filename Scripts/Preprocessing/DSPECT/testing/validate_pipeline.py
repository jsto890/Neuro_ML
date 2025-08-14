import os
import nibabel as nib
import numpy as np
import argparse
import yaml
from pathlib import Path

def fix_path(path):
    return os.path.expanduser(path)

parser = argparse.ArgumentParser(description="Comprehensive validation of DSPECT preprocessing pipeline.")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True)
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
args = parser.parse_args()

if args.isHasel:
    base_dir = os.path.expanduser('~/reseng202500013-ndd-ml')
else:
    base_dir = '/Volumes/reseng202500013-ndd-ml'

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def validate_step(step_name, input_dir, expected_suffix, min_subjects=1):
    """Validate a preprocessing step"""
    print(f"\n🔍 Validating {step_name}...")
    
    if not os.path.exists(input_dir):
        print(f"   ❌ Directory not found: {input_dir}")
        return False
    
    subjects = [d for d in os.listdir(input_dir) if d.startswith('sub-')]
    if len(subjects) < min_subjects:
        print(f"   ❌ Too few subjects: {len(subjects)} < {min_subjects}")
        return False
    
    valid_subjects = 0
    for subject in subjects[:5]:  # Check first 5 subjects
        subject_dir = os.path.join(input_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
            
        nii_files = [f for f in os.listdir(subject_dir) if f.endswith('.nii.gz')]
        if not nii_files:
            continue
            
        expected_file = f"{subject}{expected_suffix}.nii.gz"
        if expected_file in nii_files:
            file_path = os.path.join(subject_dir, expected_file)
            try:
                img = nib.load(file_path)
                data = img.get_fdata()
                if np.count_nonzero(data) > 1000:  # Reasonable brain coverage
                    valid_subjects += 1
                else:
                    print(f"   ⚠️ {subject}: Low brain coverage")
            except Exception as e:
                print(f"   ❌ {subject}: {e}")
        else:
            print(f"   ❌ {subject}: Expected file {expected_file} not found")
    
    print(f"   ✅ {valid_subjects}/{len(subjects[:5])} subjects validated")
    return valid_subjects > 0

def validate_normalization(input_dir):
    """Special validation for normalization step"""
    print(f"\n🔍 Validating Normalization (SPECT-specific)...")
    
    if not os.path.exists(input_dir):
        print(f"   ❌ Directory not found: {input_dir}")
        return False
    
    subjects = [d for d in os.listdir(input_dir) if d.startswith('sub-')]
    if len(subjects) == 0:
        print(f"   ❌ No subjects found")
        return False
    
    valid_subjects = 0
    for subject in subjects[:3]:  # Check first 3 subjects
        subject_dir = os.path.join(input_dir, subject)
        expected_file = f"{subject}_RAS.nii.gz"
        file_path = os.path.join(subject_dir, expected_file)
        
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            non_zero_data = data[data > 0]
            
            if len(non_zero_data) == 0:
                print(f"   ❌ {subject}: No non-zero voxels")
                continue
                
            mean_val = np.mean(non_zero_data)
            std_val = np.std(non_zero_data)
            
            # Check if normalization looks reasonable
            if 0.1 < mean_val < 10.0 and std_val > 0.1:
                valid_subjects += 1
                print(f"   ✅ {subject}: mean={mean_val:.3f}, std={std_val:.3f}")
            else:
                print(f"   ⚠️ {subject}: mean={mean_val:.3f}, std={std_val:.3f} (suspicious values)")
                
        except Exception as e:
            print(f"   ❌ {subject}: {e}")
    
    print(f"   ✅ {valid_subjects}/{len(subjects[:3])} subjects validated")
    return valid_subjects > 0

def validate_masking(input_dir):
    """Special validation for masking step"""
    print(f"\n🔍 Validating Masking (SPECT-specific)...")
    
    if not os.path.exists(input_dir):
        print(f"   ❌ Directory not found: {input_dir}")
        return False
    
    subjects = [d for d in os.listdir(input_dir) if d.startswith('sub-')]
    if len(subjects) == 0:
        print(f"   ❌ No subjects found")
        return False
    
    valid_subjects = 0
    for subject in subjects[:3]:  # Check first 3 subjects
        subject_dir = os.path.join(input_dir, subject)
        expected_file = f"{subject}_masked.nii.gz"
        file_path = os.path.join(subject_dir, expected_file)
        
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            coverage = np.count_nonzero(data) / data.size * 100
            
            if 5 < coverage < 50:  # Reasonable SPECT coverage
                valid_subjects += 1
                print(f"   ✅ {subject}: {coverage:.1f}% coverage")
            else:
                print(f"   ⚠️ {subject}: {coverage:.1f}% coverage (suspicious)")
                
        except Exception as e:
            print(f"   ❌ {subject}: {e}")
    
    print(f"   ✅ {valid_subjects}/{len(subjects[:3])} subjects validated")
    return valid_subjects > 0

def main():
    spect_base = os.path.join(base_dir, "data/preprocessed/SPECT")
    
    print(f"🚀 DSPECT Pipeline Validation for {args.diagnosis}")
    print("=" * 60)
    
    all_valid = True
    
    # Step 1: Reorientation
    step1_valid = validate_step("Reorientation", 
                               os.path.join(spect_base, "reoriented", args.diagnosis), 
                               "_RAS")
    if not step1_valid:
        all_valid = False
    
    # Step 2: Normalization (SPECT-specific)
    step2_valid = validate_normalization(os.path.join(spect_base, "normalised", args.diagnosis))
    if not step2_valid:
        all_valid = False
    
    # Step 3: Registration
    step3_valid = validate_step("Registration", 
                               os.path.join(spect_base, "registered", args.diagnosis), 
                               "_registered")
    if not step3_valid:
        all_valid = False
    
    # Step 4: Masking (SPECT-specific)
    step4_valid = validate_masking(os.path.join(spect_base, "masked", args.diagnosis))
    if not step4_valid:
        all_valid = False
    
    # Step 5: Finalization
    step5_valid = validate_step("Finalization", 
                               os.path.join(spect_base, "finalised", args.diagnosis), 
                               "_finalised")
    if not step5_valid:
        all_valid = False
    
    # Step 6: Postprocessing
    step6_valid = validate_step("Postprocessing", 
                               os.path.join(spect_base, "postprocessed", args.diagnosis), 
                               "_postprocessed")
    if not step6_valid:
        all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All preprocessing steps validated successfully!")
        print("🎯 Your DSPECT data is ready for machine learning!")
    else:
        print("❌ Some preprocessing steps failed validation.")
        print("🔧 Please check the failed steps and re-run the pipeline.")

if __name__ == "__main__":
    main() 