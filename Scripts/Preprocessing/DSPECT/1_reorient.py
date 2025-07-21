import os
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
import argparse
import shutil
import yaml

def fix_path(path):
    """Convert config path to actual mounted path"""
    # Since we're running from inside the research drive, just expand ~
    return os.path.expanduser(path)

# Set up argument parser
parser = argparse.ArgumentParser(description="Reorient NIfTI files to RAS orientation.")
parser.add_argument("--force", action="store_true", help="Force reprocessing even if output exists")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True, 
                    help="Diagnosis group to process (CN or PD)")
args = parser.parse_args()

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define input and output directories
root_dir = fix_path(config['raw_data']['spect'])
input_dir = os.path.join(root_dir, "PPMI", args.diagnosis)
output_dir = os.path.join(fix_path(config['preprocessed_data']['spect_p']), "reoriented", args.diagnosis)

print("\n=== Path Configuration ===")
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
        current_ornt = io_orientation(affine)
        target_ornt = axcodes2ornt(target_ornt_codes)
        transform = ornt_transform(current_ornt, target_ornt)
        data = nib.orientations.apply_orientation(data, transform)
        print("New data shape after reorient:", data.shape)
        # affine is unchanged here!
    else:
        print("Image is already RAS. No reorientation needed.")

    # Always print final orientation codes
    print("Final data shape:", data.shape)
    print("--- End Debug ---\n")

    reoriented_img = nib.Nifti1Image(data, affine)
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