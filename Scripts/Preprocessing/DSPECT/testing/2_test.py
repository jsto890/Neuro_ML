import nibabel as nib
import numpy as np
import os
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Test script for SPECT normalization.")
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
args = parser.parse_args()

# --- Path Configuration ---
# Updated to use Desktop SPECT folders
data_root = "/Users/jacksonschofield/Desktop/SPECT"

print(f"INFO: Using data root: {data_root}")

# --- Find a subject to test ---
subject_id = None
try:
    # Use CN_SPECT_PPMI_NIfTI for testing
    normalised_base_dir = os.path.join(data_root, "CN_SPECT_PPMI_NIfTI")
    for f in sorted(os.listdir(normalised_base_dir)):
        if f.startswith('sub-'):
            subject_id = f
            break
except FileNotFoundError:
    print(f"❌ Could not find normalised data directory. Looked in: {normalised_base_dir}")
    exit(1)

if not subject_id:
    print(f"❌ No subject found in {normalised_base_dir}")
    exit(1)

print(f"INFO: Testing with subject: {subject_id}")

# --- Construct Paths ---
# Use CN_SPECT_PPMI_NIfTI for both before and after
before_path = os.path.join(data_root, "CN_SPECT_PPMI_NIfTI", subject_id, f"{subject_id}_RAS.nii.gz")
after_path = os.path.join(normalised_base_dir, subject_id, f"{subject_id}_RAS.nii.gz")

def get_stats(path):
    img = nib.load(path)
    data = img.get_fdata()
    non_zero_data = data[data > 0]
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "shape": data.shape,
        "non_zero_mean": np.mean(non_zero_data) if len(non_zero_data) > 0 else 0,
        "non_zero_std": np.std(non_zero_data) if len(non_zero_data) > 0 else 0,
        "non_zero_count": len(non_zero_data)
    }

try:
    before_stats = get_stats(before_path)
    after_stats = get_stats(after_path)

    print(f"\n📊 SPECT Normalization Quality Check for {subject_id}_RAS.nii.gz\n")
    print(f"{'Metric':<15} {'Before':>18} {'After':>18}")
    print("-" * 55)
    
    metrics = ['mean', 'std', 'min', 'max', 'non_zero_mean', 'non_zero_std']
    for key in metrics:
        print(f"{key:<15} {before_stats[key]:>18.5f} {after_stats[key]:>18.5f}")
    
    print(f"{'shape':<15} {str(before_stats['shape']):>18} {str(after_stats['shape']):>18}")
    print(f"{'non_zero_count':<15} {before_stats['non_zero_count']:>18,} {after_stats['non_zero_count']:>18,}")
    
    print(f"\n✅ Normalization validation:")
    if after_stats['non_zero_mean'] > 0 and after_stats['non_zero_mean'] < 10:
        print("   ✓ Reference region normalization appears successful")
    elif after_stats['max'] <= 1.0 and after_stats['min'] >= 0.0:
        print("   ✓ Percentile normalization appears successful")
    else:
        print("   ⚠️ Normalization results may need review")
    
    if after_stats['std'] > 0:
        print("   ✓ Standard deviation is non-zero")
    else:
        print("   ❌ Standard deviation is zero - normalization may have failed")

except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("Please ensure you have run steps 1 and 2.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")