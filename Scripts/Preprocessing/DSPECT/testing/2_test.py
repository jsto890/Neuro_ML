import nibabel as nib
import numpy as np
import os
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Test script for normalization.")
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
args = parser.parse_args()

# --- Path Configuration ---
if args.isHasel:
    data_root = os.path.expanduser('~/reseng202500013-ndd-ml')
else:
    data_root = '/Volumes/reseng202500013-ndd-ml'

print(f"INFO: Using data root: {data_root}")

# --- Find a subject to test ---
subject_id = None
try:
    normalised_base_dir = os.path.join(data_root, "data/preprocessed/SPECT/normalised/CN")
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
before_path = os.path.join(data_root, "data/preprocessed/SPECT/reoriented/CN", subject_id, f"{subject_id}_RAS.nii.gz")
after_path = os.path.join(normalised_base_dir, subject_id, f"{subject_id}_RAS.nii.gz")

def get_stats(path):
    img = nib.load(path)
    data = img.get_fdata()
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "shape": data.shape
    }

try:
    before_stats = get_stats(before_path)
    after_stats = get_stats(after_path)

    print(f"\n📊 Intensity Comparison for {subject_id}_RAS.nii.gz\n")
    print(f"{'Metric':<10} {'Before':>18} {'After':>18}")
    print("-" * 48)
    for key in before_stats:
        if isinstance(before_stats[key], tuple):  # For shape
            print(f"{key:<10} {str(before_stats[key]):>18} {str(after_stats[key]):>18}")
        else:
            print(f"{key:<10} {before_stats[key]:>18.5f} {after_stats[key]:>18.5f}")

except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("Please ensure you have run steps 1 and 2.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")