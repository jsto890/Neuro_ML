import nibabel as nib
from nibabel.orientations import aff2axcodes
import os
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Test script for reorientation.")
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
    reoriented_base_dir = os.path.join(data_root, "CN_SPECT_PPMI_NIfTI")
    for f in sorted(os.listdir(reoriented_base_dir)):
        if f.startswith('sub-'):
            subject_id = f
            break
except FileNotFoundError:
    print(f" Could not find reoriented data directory. Looked in: {reoriented_base_dir}")
    exit(1)

if not subject_id:
    print(f" No subject found in {reoriented_base_dir}")
    exit(1)

print(f"INFO: Testing with subject: {subject_id}")

# --- Construct Paths ---
# Use CN_SPECT_PPMI_NIfTI for raw data
raw_path = os.path.join(data_root, "CN_SPECT_PPMI_NIfTI", subject_id, f"{subject_id}.nii")
reoriented_path = os.path.join(reoriented_base_dir, subject_id, f"{subject_id}_RAS.nii.gz")

# --- Run Test ---
try:
    if not os.path.exists(raw_path):
        raw_path = os.path.join(data_root, "CN_SPECT_PPMI_NIfTI", subject_id, f"{subject_id}.nii.gz")

    if not os.path.exists(raw_path):
        print(f" Raw file not found: {raw_path}")
        print("Raw data may be in different location. Checking alternative paths...")
        
        # Try alternative paths
        alt_paths = [
            os.path.join(data_root, "CN_SPECT_PPMI_NIfTI", subject_id, f"{subject_id}.nii"),
            os.path.join(data_root, "CN_SPECT_PPMI_NIfTI", subject_id, f"{subject_id}.nii.gz"),
            os.path.join(data_root, "PD_SPECT_PPMI_NIfTI", subject_id, f"{subject_id}.nii"),
            os.path.join(data_root, "PD_SPECT_PPMI_NIfTI", subject_id, f"{subject_id}.nii.gz")
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                raw_path = alt_path
                print(f" Found raw data: {raw_path}")
                break
        else:
            print(f" Raw data not found in any location")
            exit(1)

    orig = nib.load(raw_path)
    reoriented = nib.load(reoriented_path)

    print("Original orientation:", aff2axcodes(orig.affine))
    print("Reoriented orientation:", aff2axcodes(reoriented.affine))
    print(" Reorientation test passed!")

except FileNotFoundError as e:
    print(f"\n File not found: {e}")
    print("Please ensure you have raw data and have run step 1 (reorient).")
except Exception as e:
    print(f"\n An unexpected error occurred: {e}")