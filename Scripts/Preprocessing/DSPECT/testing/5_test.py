import os
import nibabel as nib
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Test script for step 5 finalised images.")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True)
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
parser.add_argument("--shape", type=int, nargs=3, default=[91, 109, 91], help="Expected shape (default: 91 109 91)")
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
subjects = [d for d in os.listdir(input_dir) if d.startswith('sub-')]

print(f"Testing {len(subjects)} subjects in {input_dir}")
failed = []
for subject_id in subjects:
    nii_path = os.path.join(input_dir, subject_id, f"{subject_id}_finalised.nii.gz")
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        if data.shape != tuple(args.shape):
            print(f"❌ {subject_id}: Wrong shape {data.shape}")
            failed.append(subject_id)
        elif np.count_nonzero(data) == 0:
            print(f"❌ {subject_id}: All voxels zero")
            failed.append(subject_id)
        else:
            print(f"✅ {subject_id}: OK")
    except Exception as e:
        print(f"❌ {subject_id}: {e}")
        failed.append(subject_id)

if failed:
    print("\nFAILED SUBJECTS:")
    for sid in failed:
        print(sid)
else:
    print("\nAll subjects passed step 5 test!") /Users/jacksonschofield/Desktop/SPECT/PD_SPECT_PPMI_NIfTI