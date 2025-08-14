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
args = parser.parse_args()

# Find config file in project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
config_path = os.path.join(project_root, 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

input_dir = os.path.join(fix_path(config['preprocessed_data']['spect_p']), 'registered', args.diagnosis)
spect_mask_path = fix_path(config['templates']['SPECT_occipital'])
output_base = os.path.join(fix_path(config['preprocessed_data']['spect_p']), 'masked', args.diagnosis)

print(f"\n🔄 Processing {args.diagnosis} subjects")
print(f"📁 Input directory: {input_dir}")
print(f"📁 Output directory: {output_base}")
print(f"🎭 SPECT mask: {spect_mask_path}\n")

def apply_spect_mask(image_path, mask_path, output_path):
    """Apply SPECT-specific mask to image and save masked result"""
    img = load_img(image_path)
    mask = load_img(mask_path)
    
    mask_resampled = resample_to_img(mask, img, interpolation='nearest')
    
    img_data = img.get_fdata()
    mask_data = mask_resampled.get_fdata()
    
    masked_data = img_data * (mask_data > 0)
    
    masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
    masked_img.to_filename(output_path)
    
    return masked_data

subject_dirs = [d for d in os.listdir(input_dir) if d.startswith("sub-")]
failed_subjects = []

for subject_id in subject_dirs:
    if subject_id.startswith("._") or subject_id == ".DS_Store":
        continue

    print(f"\n🔄 Masking {subject_id}")
    subj_input_dir = os.path.join(input_dir, subject_id)
    output_dir = os.path.join(output_base, subject_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for fname in os.listdir(subj_input_dir):
        if fname.endswith(".json") and not fname.startswith("._"):
            shutil.copy(os.path.join(subj_input_dir, fname), os.path.join(output_dir, fname))

    input_nii = os.path.join(subj_input_dir, f"{subject_id}_registered.nii.gz")
    output_nii = os.path.join(output_dir, f"{subject_id}_masked.nii.gz")

    try:
        masked_data = apply_spect_mask(input_nii, spect_mask_path, output_nii)
        
        non_zero_voxels = np.count_nonzero(masked_data)
        total_voxels = masked_data.size
        brain_coverage = (non_zero_voxels / total_voxels) * 100
        
        print(f"✅ Saved: {output_nii}")
        print(f"📊 SPECT coverage: {brain_coverage:.2f}% ({non_zero_voxels:,} voxels)")

    except Exception as e:
        print(f"❌ Error for {subject_id}: {e}")
        failed_subjects.append(subject_id)

if failed_subjects:
    print("\n======================")
    print("❌ FAILED SUBJECTS")
    print("======================")
    for sid in failed_subjects:
        print(sid)
else:
    print("\n✅ All subjects masked successfully.")
