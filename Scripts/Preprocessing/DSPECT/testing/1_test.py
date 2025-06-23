import nibabel as nib
from nibabel.orientations import aff2axcodes
import os
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Test script for reorientation.")
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
    reoriented_base_dir = os.path.join(data_root, "data/preprocessed/SPECT/reoriented/CN")
    for f in sorted(os.listdir(reoriented_base_dir)):
        if f.startswith('sub-'):
            subject_id = f
            break
except FileNotFoundError:
    print(f"❌ Could not find reoriented data directory. Looked in: {reoriented_base_dir}")
    exit(1)

if not subject_id:
    print(f"❌ No subject found in {reoriented_base_dir}")
    exit(1)

print(f"INFO: Testing with subject: {subject_id}")

# --- Construct Paths ---
raw_path = os.path.join(data_root, "data/raw/SPECT/PPMI/CN", subject_id, f"{subject_id}.nii")
reoriented_path = os.path.join(reoriented_base_dir, subject_id, f"{subject_id}_RAS.nii.gz")

# --- Run Test ---
try:
    if not os.path.exists(raw_path):
        # Some subjects might be .nii.gz
        raw_path = os.path.join(data_root, "data/raw/SPECT/PPMI/CN", subject_id, f"{subject_id}.nii.gz")

    orig = nib.load(raw_path)
    reoriented = nib.load(reoriented_path)

    print("Original orientation:", aff2axcodes(orig.affine))
    print("Reoriented orientation:", aff2axcodes(reoriented.affine))

except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("Please ensure you have raw data and have run step 1 (reorient).")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")