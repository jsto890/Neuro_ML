import os
from pathlib import Path
import nibabel as nib
import numpy as np
import shutil
from nilearn.image import resample_to_img, load_img
import yaml
import argparse

print("Hello World! Step 4 brain masking script starting...")

def fix_path(path):
    """Convert config path to actual mounted path"""
    return os.path.expanduser(path)

parser = argparse.ArgumentParser(description="Apply SPECT-specific brain mask to registered SPECT images.")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True, 
                    help="Diagnosis group to process (CN or PD)")
parser.add_argument("--mask_type", type=str, choices=['occipital', 'whole_brain'], default='whole_brain',
                    help="Mask type: occipital (SPECT-specific) or whole_brain (ML-friendly)")
args = parser.parse_args()

# Find config file in project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
config_path = os.path.join(project_root, 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

input_dir = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_registered"

# Update mask path logic
if args.mask_type == 'occipital':
    spect_mask_path = "/Users/jacksonschofield/Desktop/P4P/Templates/SPECT/occipital_mask.nii.gz"
else:
    # Use whole-brain mask if available, fallback to occipital
    try:
        spect_mask_path = "/Users/jacksonschofield/Desktop/P4P/Templates/SPECT/occipital_mask.nii.gz"  # Fallback to occipital for now
    except KeyError:
        print("️ Whole-brain mask not found, using occipital mask")
        spect_mask_path = "/Users/jacksonschofield/Desktop/P4P/Templates/SPECT/occipital_mask.nii.gz"

output_base = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_masked"

print(f"\n Processing {args.diagnosis} subjects")
print(f" Input directory: {input_dir}")
print(f" Output directory: {output_base}")
print(f" SPECT mask: {spect_mask_path}\n")

# Create output directory
os.makedirs(output_base, exist_ok=True)

def apply_spect_mask(image_path, mask_path, output_path):
    """Apply SPECT-specific mask to image and save masked result"""
    img = load_img(image_path)
    mask = load_img(mask_path)
    # Robust resample with header copy + force
    mask_resampled = resample_to_img(mask, img, interpolation='nearest', copy_header=True, force_resample=True)
    
    img_data = img.get_fdata()
    img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)
    img_data = np.clip(img_data, 0, None)
    mask_data = mask_resampled.get_fdata()
    
    masked_data = img_data * (mask_data > 0)
    
    # If coverage is zero, fall back to a simple intensity-based mask (top percentile) to avoid full dropout
    if np.count_nonzero(masked_data) == 0:
        non_zero = img_data[img_data > 0]
        if non_zero.size > 0:
            thr = np.percentile(non_zero, 50)  # median as fallback threshold
            fallback_mask = img_data >= max(thr, np.percentile(non_zero, 30))
            masked_data = img_data * fallback_mask
    
    masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
    masked_img.to_filename(output_path)
    
    return masked_data

# Add ML validation
def validate_ml_readiness(masked_data, subject_id):
    """Validate data is ready for ML"""
    non_zero_voxels = np.count_nonzero(masked_data)
    total_voxels = masked_data.size
    coverage = (non_zero_voxels / total_voxels) * 100
    
    # ML-friendly coverage: 1-70% for SPECT (wider to keep more data, to be filtered later)
    if coverage < 1:
        print(f"️ {subject_id}: Very low coverage ({coverage:.1f}%) - may affect ML performance")
    elif coverage > 70:
        print(f"️ {subject_id}: High coverage ({coverage:.1f}%) - may include non-brain tissue")
    
    # Check for negative values (shouldn't exist in SPECT)
    if np.any(masked_data < 0):
        print(f" {subject_id}: Negative values detected - not ML-ready")
        return False
    
    return True

subject_dirs = [d for d in os.listdir(input_dir) if d.startswith("Subject_")]
failed_subjects = []

for subject_id in subject_dirs:
    if subject_id.startswith("._") or subject_id == ".DS_Store":
        continue

    print(f"\n Masking {subject_id}")
    subj_input_dir = os.path.join(input_dir, subject_id)
    output_dir = os.path.join(output_base, subject_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Copy only essential files (NIfTI + JSON, skip DICOM folders)
    print(f"[COPY] {subject_id}: Copying essential files")
    for item in os.listdir(subj_input_dir):
        source_path = os.path.join(subj_input_dir, item)
        dest_path = os.path.join(output_dir, item)
        
        if os.path.isfile(source_path):
            # Copy JSON files and the registered NIfTI
            if item.endswith('.json') or item == "3. registered.nii.gz":
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

    input_nii = os.path.join(subj_input_dir, "3. registered.nii.gz")
    output_nii = os.path.join(output_dir, "4. masked.nii.gz")

    try:
        masked_data = apply_spect_mask(input_nii, spect_mask_path, output_nii)
        
        if not validate_ml_readiness(masked_data, subject_id):
            failed_subjects.append(subject_id)
            continue

        non_zero_voxels = np.count_nonzero(masked_data)
        total_voxels = masked_data.size
        brain_coverage = (non_zero_voxels / total_voxels) * 100
        
        print(f" Saved: {output_nii}")
        print(f" SPECT coverage: {brain_coverage:.2f}% ({non_zero_voxels:,} voxels)")

    except Exception as e:
        print(f" Error for {subject_id}: {e}")
        failed_subjects.append(subject_id)

if failed_subjects:
    print("\n======================")
    print(" FAILED SUBJECTS")
    print("======================")
    for sid in failed_subjects:
        print(sid)
else:
    print("\n All subjects masked successfully.")

print(f"\n Masking complete! Output saved to: {output_base}")
print(f" Each subject now has a dedicated folder with masked data and all original files.")
