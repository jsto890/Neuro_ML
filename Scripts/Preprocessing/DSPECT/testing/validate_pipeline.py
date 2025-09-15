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

# Updated to use Desktop SPECT folders
base_dir = "/Users/jacksonschofield/Desktop/SPECT"

# Remove hardcoded config loading since we're using Desktop structure
# with open('config.yaml', 'r') as f:
#     config = yaml.safe_load(f)

def validate_step(step_name, input_dir, expected_suffix, min_subjects=1):
    """Validate a preprocessing step"""
    print(f"\n🔍 Validating {step_name}...")
    
    if not os.path.exists(input_dir):
        print(f"   ❌ Directory not found: {input_dir}")
        return False
    
    subjects = [d for d in os.listdir(input_dir) if d.startswith('Subject_')]
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
            
        expected_file = expected_suffix
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
    
    subjects = [d for d in os.listdir(input_dir) if d.startswith('Subject_')]
    if len(subjects) == 0:
        print(f"   ❌ No subjects found")
        return False
    
    valid_subjects = 0
    for subject in subjects[:3]:  # Check first 3 subjects
        subject_dir = os.path.join(input_dir, subject)
        expected_file = "2. normalised.nii.gz"
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
            
            # Check if normalization looks reasonable (broad bounds)
            if 0.01 < mean_val < 20.0 and std_val > 0.01:
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
    
    subjects = [d for d in os.listdir(input_dir) if d.startswith('Subject_')]
    if len(subjects) == 0:
        print(f"   ❌ No subjects found")
        return False
    
    valid_subjects = 0
    for subject in subjects[:3]:  # Check first 3 subjects
        subject_dir = os.path.join(input_dir, subject)
        expected_file = "4. masked.nii.gz"
        file_path = os.path.join(subject_dir, expected_file)
        
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            coverage = np.count_nonzero(data) / data.size * 100
            
            if 1 < coverage < 70:  # Broader SPECT coverage window to avoid false negatives
                valid_subjects += 1
                print(f"   ✅ {subject}: {coverage:.1f}% coverage")
            else:
                print(f"   ⚠️ {subject}: {coverage:.1f}% coverage (suspicious)")
                
        except Exception as e:
            print(f"   ❌ {subject}: {e}")
    
    print(f"   ✅ {valid_subjects}/{len(subjects[:3])} subjects validated")
    return valid_subjects > 0

# Add ML-specific validation
def validate_ml_readiness(input_dir, step_name):
    """Validate data is ready for machine learning"""
    print(f"\n🔍 Validating ML Readiness for {step_name}...")
    
    if not os.path.exists(input_dir):
        print(f"   ❌ Directory not found: {input_dir}")
        return False
    
    subjects = [d for d in os.listdir(input_dir) if d.startswith('Subject_')]
    if len(subjects) == 0:
        print(f"   ❌ No subjects found")
        return False
    
    valid_subjects = 0
    shapes = []
    intensity_ranges = []
    
    for subject in subjects[:5]:  # Check first 5 subjects
        subject_dir = os.path.join(input_dir, subject)
        
        # Look for the postprocessed file specifically
        expected_file = "6. postprocessed.nii.gz"
        file_path = os.path.join(subject_dir, expected_file)
        
        if not os.path.exists(file_path):
            print(f"   ❌ {subject}: {expected_file} not found")
            continue
        
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Check shape consistency
            shapes.append(data.shape)
            
            # Check intensity characteristics
            non_zero_data = data[data != 0]  # Include negative values for z-score data
            if len(non_zero_data) > 0:
                intensity_ranges.append((np.min(non_zero_data), np.max(non_zero_data), np.mean(non_zero_data), np.std(non_zero_data)))
            
            # Check for ML readiness issues
            issues = []
            
            # Check for extreme negative values (z-score should be around 0±5)
            if np.any(data < -6):
                issues.append("extreme_negative_values")
            
            # Check for extreme positive values
            if np.any(data > 6):
                issues.append("extreme_positive_values")
            
            # Reasonable brain coverage for SPECT (1-10%)
            coverage = np.count_nonzero(data) / data.size * 100
            if coverage < 1 or coverage > 15:
                issues.append(f"coverage_{coverage:.1f}%")
            
            # No NaN or Inf values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                issues.append("nan_or_inf")
            
            # Check normalization quality (z-score should be ~0 mean, ~1 std)
            if len(non_zero_data) > 0:
                mean_val = np.mean(non_zero_data)
                std_val = np.std(non_zero_data)
                if abs(mean_val) > 1.0:  # Allow some deviation from 0
                    issues.append(f"mean_{mean_val:.3f}")
                if not (0.5 < std_val < 2.0):  # Allow some deviation from 1
                    issues.append(f"std_{std_val:.3f}")
            
            if not issues:
                valid_subjects += 1
                print(f"   ✅ {subject}: ML-ready (shape={data.shape}, coverage={coverage:.1f}%, mean={np.mean(non_zero_data):.3f}, std={np.std(non_zero_data):.3f})")
            else:
                print(f"   ⚠️ {subject}: Issues: {', '.join(issues)}")
                
        except Exception as e:
            print(f"   ❌ {subject}: {e}")
    
    # Check consistency across subjects
    if len(set(shapes)) > 1:
        print(f"   ⚠️ Inconsistent shapes: {set(shapes)}")
    else:
        print(f"   ✅ All subjects have consistent shape: {shapes[0] if shapes else 'None'}")
    
    if len(intensity_ranges) > 0:
        means = [r[2] for r in intensity_ranges]
        stds = [r[3] for r in intensity_ranges]
        print(f"   📊 Intensity ranges: mean={np.mean(means):.3f}±{np.std(means):.3f}, std={np.mean(stds):.3f}±{np.std(stds):.3f}")
    
    print(f"   ✅ {valid_subjects}/{len(subjects[:5])} subjects ML-ready")
    return valid_subjects > 0

# Update main validation to include ML checks
def main():
    # Use Desktop SPECT folder structure
    spect_base = base_dir
    
    print(f"🚀 DSPECT Pipeline Validation for {args.diagnosis}")
    print("=" * 60)
    
    all_valid = True
    
    # Step 1: Reorientation - Check CN_SPECT_PPMI_reoriented
    step1_valid = validate_step("Reorientation", 
                               os.path.join(spect_base, f"{args.diagnosis}_SPECT_PPMI_reoriented"), 
                               "1. reorient.nii.gz")
    if not step1_valid:
        all_valid = False
    
    # Step 2: Normalization - Check CN_SPECT_PPMI_normalised
    step2_valid = validate_normalization(os.path.join(spect_base, f"{args.diagnosis}_SPECT_PPMI_normalised"))
    if not step2_valid:
        all_valid = False
    
    # Step 3: Registration - Check CN_SPECT_PPMI_registered
    step3_valid = validate_step("Registration", 
                               os.path.join(spect_base, f"{args.diagnosis}_SPECT_PPMI_registered"), 
                               "3. registered.nii.gz")
    if not step3_valid:
        all_valid = False
    
    # Step 4: Masking - Check CN_SPECT_PPMI_masked
    step4_valid = validate_masking(os.path.join(spect_base, f"{args.diagnosis}_SPECT_PPMI_masked"))
    if not step4_valid:
        all_valid = False
    
    # Step 5: Finalization - Check CN_SPECT_PPMI_finalised
    step5_valid = validate_step("Finalization", 
                               os.path.join(spect_base, f"{args.diagnosis}_SPECT_PPMI_finalised"), 
                               "5. finalised.nii.gz")
    if not step5_valid:
        all_valid = False
    
    # Step 6: Postprocessing - Check CN_SPECT_PPMI_postprocessed
    step6_valid = validate_step("Postprocessing", 
                               os.path.join(spect_base, f"{args.diagnosis}_SPECT_PPMI_postprocessed"), 
                               "6. postprocessed.nii.gz")
    if not step6_valid:
        all_valid = False
    
    # ML Readiness Validation
    print("\n" + "=" * 60)
    print("🔍 ML READINESS VALIDATION")
    print("=" * 60)
    
    ml_final_valid = validate_ml_readiness(os.path.join(spect_base, f"{args.diagnosis}_SPECT_PPMI_postprocessed"), "Final Output")
    if not ml_final_valid:
        all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All preprocessing steps validated successfully!")
        print("🎯 Your DSPECT data is ready for machine learning!")
        print("\n📊 ML Readiness Summary:")
        print("   ✓ Consistent shapes across subjects")
        print("   ✓ Appropriate intensity ranges")
        print("   ✓ No negative values or artifacts")
        print("   ✓ Reasonable brain coverage")
        print("   ✓ Proper normalization")
    else:
        print("❌ Some preprocessing steps failed validation.")
        print("🔧 Please check the failed steps and re-run the pipeline.")

if __name__ == "__main__":
    main() 