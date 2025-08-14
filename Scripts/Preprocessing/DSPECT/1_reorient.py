import os
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
import argparse
import shutil
import yaml
import numpy as np

def fix_path(path):
    """Convert config path to actual mounted path"""
    return os.path.expanduser(path)

# Set up argument parser
parser = argparse.ArgumentParser(description="Reorient NIfTI files to RAS orientation.")
parser.add_argument("--force", action="store_true", help="Force reprocessing even if output exists")
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

# Define input and output directories
root_dir = fix_path(config['raw_data']['spect'])
input_dir = os.path.join(root_dir, "PPMI", args.diagnosis)
output_dir = os.path.join(fix_path(config['preprocessed_data']['spect_p']), "reoriented", args.diagnosis)

print("\n=== Path Configuration ===")
print(f"Config file: {config_path}")
print(f"Config SPECT path: {config['raw_data']['spect']}")
print(f"Root directory: {root_dir}")
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print("========================\n")

# Verify input directory exists
if not os.path.exists(input_dir):
    print(f"❌ Error: Input directory does not exist: {input_dir}")
    print("Please check that the path is correct and the drive is mounted.")
    exit(1)

# Create output dir if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def reorient_to_RAS(nifti_path, output_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine

    print("\n--- Reorientation Debug ---")
    print(f"Processing file: {nifti_path}")
    print("Original affine:")
    print(affine)
    print("Original shape:", data.shape)
    orig_ornt_codes = nib.orientations.aff2axcodes(affine)
    print("Original orientation codes:", orig_ornt_codes)
    target_ornt_codes = ('R', 'A', 'S')
    print("Target orientation codes:", target_ornt_codes)

    if orig_ornt_codes != target_ornt_codes:
        print("Reorienting to RAS...")
        
        # Get current and target orientations
        current_ornt = io_orientation(affine)
        target_ornt = axcodes2ornt(target_ornt_codes)
        
        # Calculate the transformation
        transform = ornt_transform(current_ornt, target_ornt)
        
        # Use nibabel's built-in reorientation which handles affine correctly
        reoriented_img = nib.as_closest_canonical(img)
        data = reoriented_img.get_fdata()
        new_affine = reoriented_img.affine
        
        # Resample to isotropic 1mm voxels to avoid visual distortion
        from nilearn.image import resample_img
        target_affine = np.eye(4)
        target_affine[0,0] = 1  # 1mm voxel size
        target_affine[1,1] = 1
        target_affine[2,2] = 1
        # Preserve center
        center = np.dot(new_affine, np.array(data.shape + (1,)) / 2)[:3]
        target_affine[:3,3] = center - np.dot(target_affine[:3,:3], np.array(reoriented_img.shape) / 2)
        reoriented_img = resample_img(reoriented_img, target_affine=target_affine, interpolation='continuous')
        data = reoriented_img.get_fdata()
        new_affine = reoriented_img.affine
        
        print("New data shape after reorient:", data.shape)
        print("New affine matrix:")
        print(new_affine)
    else:
        print("Image is already RAS. No reorientation needed.")
        new_affine = affine

    # Always print final orientation codes
    final_ornt_codes = nib.orientations.aff2axcodes(new_affine)
    print("Final orientation codes:", final_ornt_codes)
    print("Final data shape:", data.shape)
    print("--- End Debug ---\n")

    reoriented_img = nib.Nifti1Image(data, new_affine)
    nib.save(reoriented_img, output_path)

    # Copy matching JSON file if it exists
    json_basename = os.path.splitext(os.path.splitext(nifti_path)[0])[0] + ".json"
    if os.path.exists(json_basename):
        json_output_path = os.path.join(os.path.dirname(output_path), os.path.basename(json_basename))
        shutil.copy2(json_basename, json_output_path)
        print(f"[OK] Copied JSON -> {json_output_path}")
    else:
        print(f"[WARN] No matching JSON file found")

print(f"\n🔄 Processing {args.diagnosis} subjects from: {input_dir}")
print(f"📁 Output directory: {output_dir}\n")

# Process all subjects
for subject in sorted(os.listdir(input_dir)):
    subject_path = os.path.join(input_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    nii_files = [f for f in os.listdir(subject_path) if f.endswith(".nii") or f.endswith(".nii.gz")]
    if not nii_files:
        print(f"[WARN] {subject}: No NIfTI file found in {subject_path}")
        continue

    input_nii = os.path.join(subject_path, nii_files[0])
    subject_output_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_output_dir, exist_ok=True)
    output_nii = os.path.join(subject_output_dir, f"{subject}_RAS.nii.gz")

    if os.path.exists(output_nii) and not args.force:
        print(f"[SKIP] {subject}: already reoriented at {output_nii}")
        continue
    elif os.path.exists(output_nii) and args.force:
        print(f"[FORCE] {subject}: Reprocessing")

    try:
        reorient_to_RAS(input_nii, output_nii)
        print(f"[OK] {subject}: Successfully reoriented -> {output_nii}")
    except Exception as e:
        print(f"[ERROR] {subject}: failed: {e}")