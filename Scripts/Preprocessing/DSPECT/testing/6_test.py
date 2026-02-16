import os
import nibabel as nib
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser(description="Test script for step 6 postprocessed images and summary CSV.")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True)
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
args = parser.parse_args()

# Updated to use Desktop SPECT folders
base_dir = "/Users/jacksonschofield/Desktop/SPECT"

def full_path(path):
    # Simplified path handling for Desktop structure
    if path.startswith("~/") or path.startswith("/Volumes/"):
        return os.path.join(base_dir, os.path.relpath(os.path.expanduser(path), start=os.path.expanduser('~/reseng202500013-ndd-ml')))
    return path

# Use CN_SPECT_PPMI_NIfTI for testing
input_dir = os.path.join(base_dir, "CN_SPECT_PPMI_NIfTI")
csv_path = os.path.join(input_dir, f"summary_CN.csv")
subjects = [f for f in os.listdir(input_dir) if f.startswith('sub-') and f.endswith('.nii.gz')]

# Check CSV
if not os.path.exists(csv_path):
    print(f" Summary CSV not found: {csv_path}")
else:
    print(f" Found summary CSV: {csv_path}")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"CSV contains {len(rows)} rows.")

# Check images
failed = []
for nii_file in subjects:
    nii_path = os.path.join(input_dir, nii_file)
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        if np.count_nonzero(data) == 0:
            print(f" {nii_file}: All voxels zero")
            failed.append(nii_file)
        else:
            print(f" {nii_file}: OK")
    except Exception as e:
        print(f" {nii_file}: {e}")
        failed.append(nii_file)

if failed:
    print("\nFAILED IMAGES:")
    for f in failed:
        print(f)
else:
    print("\nAll postprocessed images passed step 6 test!") 