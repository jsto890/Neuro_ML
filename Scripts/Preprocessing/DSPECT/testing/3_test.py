import nibabel as nib
import numpy as np
import os
import sys
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Test script for registration quality.")
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
args = parser.parse_args()

# --- Path Configuration ---
# Updated to use Desktop SPECT folders
data_root = "/Users/jacksonschofield/Desktop/SPECT"

print(f"INFO: Using data root: {data_root}")

def find_first_valid_subject_file(base_dir):
    """Finds the first valid (non-empty) registered subject file."""
    if not os.path.isdir(base_dir):
        print(f" Base directory not found: {base_dir}")
        return None, None
    
    for subject_dir in sorted(os.listdir(base_dir)):
        if subject_dir.startswith("sub-"):
            subject_path = os.path.join(base_dir, subject_dir)
            if os.path.isdir(subject_path):
                expected_file = f"{subject_dir}_registered.nii.gz"
                file_path = os.path.join(subject_path, expected_file)
                if os.path.exists(file_path):
                    # Check if the file is valid (not empty)
                    img = nib.load(file_path)
                    if np.count_nonzero(img.get_fdata()) > 1000:
                        print(f"INFO: Found valid subject to test: {subject_dir}")
                        return subject_dir, file_path
                    else:
                        print(f"WARN: Skipping empty registered file for subject: {subject_dir}")
                        
    print(f" Could not find any valid (non-empty) registered files in: {base_dir}")
    return None, None

# === CONFIG ===
# Search for the first available subject in the CN_SPECT_PPMI_NIfTI directory
registered_base_dir = os.path.join(data_root, "CN_SPECT_PPMI_NIfTI")
subject_id, registered_path = find_first_valid_subject_file(registered_base_dir)

if not registered_path:
    sys.exit(1)

# Make template path relative to the project root for portability
try:
    # Assumes script is in .../P4P/Scripts/Preprocessing/DSPECT/testing
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, *['..']*5))
    template_path = os.path.join(project_root, "Templates/SPECT/symFPCITtemplate_MNI_norm.nii")
    if not os.path.exists(template_path):
         template_path = "Templates/SPECT/symFPCITtemplate_MNI_norm.nii" # Fallback if CWD is project root
except NameError:
    # __file__ is not defined in some interactive environments, assume CWD is project root
    template_path = "Templates/SPECT/symFPCITtemplate_MNI_norm.nii"

print(f"\n Registration Quality Test for {subject_id}\n")
print(f"Found registered file: {registered_path}")
print(f"Using template file: {template_path}\n")

# === HELPERS ===
def get_image_info(path):
    img = nib.load(path)
    data = img.get_fdata()
    voxel_sizes = np.round(np.sqrt((img.affine[:3, :3] ** 2).sum(0)), 3)
    return {
        "shape": data.shape,
        "voxel_sizes": voxel_sizes,
        "nonzero_voxels": np.count_nonzero(data),
        "mean": np.mean(data),
        "max": np.max(data),
        "min": np.min(data)
    }

# === RUN ===
try:
    reg_info = get_image_info(registered_path)
    tmpl_info = get_image_info(template_path)

    print(f"{'Metric':<18} {'Registered':>18} {'Template':>18}")
    print("-" * 60)
    print(f"{'Shape':<18} {str(reg_info['shape']):>18} {str(tmpl_info['shape']):>18}")
    print(f"{'Voxel Sizes':<18} {str(reg_info['voxel_sizes']):>18} {str(tmpl_info['voxel_sizes']):>18}")
    print(f"{'Non-zero Voxels':<18} {reg_info['nonzero_voxels']:>18,} {tmpl_info['nonzero_voxels']:>18,}")
    print(f"{'Mean Intensity':<18} {reg_info['mean']:>18.4f} {tmpl_info['mean']:>18.4f}")
    print(f"{'Max Intensity':<18} {reg_info['max']:>18.4f} {tmpl_info['max']:>18.4f}")
    print(f"{'Min Intensity':<18} {reg_info['min']:>18.4f} {tmpl_info['min']:>18.4f}")

    if reg_info["nonzero_voxels"] < 1000:
        print("\n️ WARNING: Too few non-zero voxels. This might indicate a failed registration.")

except FileNotFoundError as e:
    print(f"\n File not found: {e}")
    print("Please ensure that step 3 (registration) has been run and that the paths are correct.")
except Exception as e:
    print(f"\n An unexpected error occurred: {e}")