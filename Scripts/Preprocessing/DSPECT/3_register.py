import os
from pathlib import Path
import nibabel as nib
import numpy as np
import shutil
from nilearn.image import resample_to_img, load_img
from nilearn.image.resampling import BoundingBoxError
import yaml
import argparse

print("Hello World! Step 3 registration script starting...")

def fix_path(path):
    """Convert config path to actual mounted path"""
    return os.path.expanduser(path)

# Set up argument parser
parser = argparse.ArgumentParser(description="Register SPECT images to template.")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True, 
                    help="Diagnosis group to process (CN or PD)")
args = parser.parse_args()

# Find config file in project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
config_path = os.path.join(project_root, 'config.yaml')

# Load config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# === CONFIG ===
input_dir = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_normalised"
template_path = "/Users/jacksonschofield/Desktop/P4P/Templates/SPECT/symFPCITtemplate_MNI_norm.nii"
output_base = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_registered"

print(f"\n Processing {args.diagnosis} subjects")
print(f" Input directory: {input_dir}")
print(f" Output directory: {output_base}")
print(f" Template: {template_path}\n")

# Create output directory
os.makedirs(output_base, exist_ok=True)

# === HELPERS ===
def recenter_affine(img): 
    data = img.get_fdata()
    shape = data.shape
    current_affine = img.affine
    voxel_sizes = np.sqrt((current_affine[:3, :3] ** 2).sum(0))
    new_affine = np.diag(list(voxel_sizes) + [1])
    new_affine[:3, 3] = -np.array(shape) * voxel_sizes / 2
    return nib.Nifti1Image(data, new_affine, img.header)

# === RUN ===
template = load_img(template_path)
subject_dirs = [d for d in os.listdir(input_dir) if d.startswith("Subject_")]

failed_subjects = []

for subject_id in subject_dirs:
    if subject_id.startswith("._"):
        continue  # skip macOS junk

    print(f"\n Registering {subject_id}")
    subj_input_dir = os.path.join(input_dir, subject_id)
    output_dir = os.path.join(output_base, subject_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Copy only essential files (NIfTI + JSON, skip DICOM folders)
    print(f"[COPY] {subject_id}: Copying essential files")
    for item in os.listdir(subj_input_dir):
        source_path = os.path.join(subj_input_dir, item)
        dest_path = os.path.join(output_dir, item)
        
        if os.path.isfile(source_path):
            # Copy JSON files and the normalized NIfTI
            if item.endswith('.json') or item == "2. normalised.nii.gz":
                shutil.copy2(source_path, dest_path)
                print(f"   [COPY] {item}")
        elif os.path.isdir(source_path) and not item.startswith('Original_DICOM'):
            # Copy only non-DICOM directories
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
            print(f"   [COPY] {item}/ (directory)")
        else:
            print(f"   [SKIP] {item} (DICOM folder - not needed)")

    # Proceed with registration
    input_nii = os.path.join(subj_input_dir, "2. normalised.nii.gz")
    output_nii = os.path.join(output_dir, "3. registered.nii.gz")

    try:
        img = load_img(input_nii)
        # Remove NaN/Inf and clamp negatives (SPECT should be non-negative)
        data = img.get_fdata()
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, 0, None)
        img = nib.Nifti1Image(data, img.affine, img.header)

        # Robust resampling with header copy and explicit force
        resampled = resample_to_img(
            img,
            template,
            interpolation='continuous',
            copy_header=True,
            force_resample=True,
        )
        resampled.to_filename(output_nii)
        print(f" Saved: {output_nii}")

    except BoundingBoxError as e:
        print(f" BoundingBoxError for {subject_id}: {e}")
        print(" Trying fallback: recenter affine...")
        try:
            recentered_img = recenter_affine(img)
            resampled = resample_to_img(
                recentered_img,
                template,
                interpolation='continuous',
                copy_header=True,
                force_resample=True
            )
            resampled.to_filename(output_nii)
            print(f" Saved after recentering: {output_nii}")
        except Exception as e2:
            print(f" Recentered registration failed for {subject_id}: {e2}")
            failed_subjects.append(subject_id)

    except Exception as e:
        print(f" Unexpected error for {subject_id}: {e}")
        failed_subjects.append(subject_id)

# === SUMMARY ===
if failed_subjects:
    print("\n======================")
    print(" FAILED SUBJECTS")
    print("======================")
    for sid in failed_subjects:
        print(sid)
else:
    print("\n All subjects registered successfully.")

print(f"\n Registration complete! Output saved to: {output_base}")
print(f" Each subject now has a dedicated folder with registered data and all original files.")