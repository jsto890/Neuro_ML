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
parser.add_argument("--isotropic", action="store_true", help="Resample to isotropic 1mm voxels for ML")
args = parser.parse_args()

# Find config file in project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
config_path = os.path.join(project_root, 'config.yaml')

# Load config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Define input and output directories - Use Desktop SPECT folders directly
if args.diagnosis == 'CN':
    input_dir = "/Users/jacksonschofield/Desktop/SPECT/CN_SPECT_PPMI_NIfTI"
elif args.diagnosis == 'PD':
    input_dir = "/Users/jacksonschofield/Desktop/SPECT/PD_SPECT_PPMI_NIfTI"
else:
    print(f" Error: Invalid diagnosis {args.diagnosis}")
    exit(1)

# Create dedicated reoriented output directory
output_base = "/Users/jacksonschofield/Desktop/SPECT"
output_dir = os.path.join(output_base, f"{args.diagnosis}_SPECT_PPMI_reoriented")

print("\n=== Path Configuration ===")
print(f"Diagnosis: {args.diagnosis}")
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print("========================\n")

# Verify input directory exists
if not os.path.exists(input_dir):
    print(f" Error: Input directory does not exist: {input_dir}")
    print("Please check that the path is correct and the drive is mounted.")
    exit(1)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

def reorient_to_RAS(nifti_path, output_path, isotropic=False):
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
        
        # Resample to isotropic 1mm if enabled (for ML consistency)
        if isotropic:
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

print(f"\n Processing {args.diagnosis} subjects from: {input_dir}")
print(f" Output directory: {output_dir}\n")

# Process all subjects
for subject in sorted(os.listdir(input_dir)):
    subject_path = os.path.join(input_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    # Create subject output directory
    subject_output_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_output_dir, exist_ok=True)

    # Look for NIfTI files recursively in the subject folder and its subfolders
    nii_files = []
    for root, dirs, files in os.walk(subject_path):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nii_files.append(os.path.join(root, file))
    
    if not nii_files:
        print(f"[WARN] {subject}: No NIfTI file found in {subject_path}")
        continue

    # Use the first NIfTI file found
    input_nii = nii_files[0]
    output_nii = os.path.join(subject_output_dir, "1. reorient.nii.gz")

    if os.path.exists(output_nii) and not args.force:
        print(f"[SKIP] {subject}: already reoriented at {output_nii}")
        continue
    elif os.path.exists(output_nii) and args.force:
        print(f"[FORCE] {subject}: Reprocessing")

    try:
        # Reorient the NIfTI file
        reorient_to_RAS(input_nii, output_nii, args.isotropic)
        print(f"[OK] {subject}: Successfully reoriented -> {output_nii}")
        
        # Copy only essential files (NIfTI + JSON, skip DICOM folders)
        scan_folder = os.path.dirname(input_nii)
        print(f"[COPY] {subject}: Copying essential files from {scan_folder}")
        
        for item in os.listdir(scan_folder):
            source_path = os.path.join(scan_folder, item)
            dest_path = os.path.join(subject_output_dir, item)
            
            if os.path.isfile(source_path):
                # Copy JSON files and the reoriented NIfTI
                if item.endswith('.json') or item == "1. reorient.nii.gz":
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
        
        print(f"[OK] {subject}: All files copied to {subject_output_dir}")
        
    except Exception as e:
        print(f"[ERROR] {subject}: failed: {e}")

print(f"\n Reorientation complete! Output saved to: {output_dir}")
print(f" Each subject now has a dedicated folder with reoriented data and all original files.")