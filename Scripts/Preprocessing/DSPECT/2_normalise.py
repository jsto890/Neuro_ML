import os
import nibabel as nib
import numpy as np
import shutil
import yaml
import argparse

def fix_path(path):
    """Convert config path to actual mounted path"""
    return os.path.expanduser(path)

parser = argparse.ArgumentParser(description="Normalize SPECT images using reference region method.")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True, 
                    help="Diagnosis group to process (CN or PD)")
parser.add_argument("--method", type=str, choices=['reference', 'percentile'], default='reference',
                    help="Normalization method: reference region or percentile clipping")
args = parser.parse_args()

# Find config file in project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
config_path = os.path.join(project_root, 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

input_root = os.path.join(fix_path(config['preprocessed_data']['spect_p']), 'reoriented', args.diagnosis)
output_root = os.path.join(fix_path(config['preprocessed_data']['spect_p']), 'normalised', args.diagnosis)
occipital_mask_path = fix_path(config['templates']['SPECT_occipital'])

print(f"\n🔄 Processing {args.diagnosis} subjects")
print(f"📁 Input directory: {input_root}")
print(f"📁 Output directory: {output_root}")
print(f"🎯 Normalization method: {args.method}")
if args.method == 'reference':
    print(f"🎭 Reference mask: {occipital_mask_path}\n")

def normalize_with_reference_region(img_data, mask_data):
    """Normalize using occipital reference region"""
    reference_region = img_data[mask_data > 0]
    if len(reference_region) == 0:
        print("Warning: No voxels in reference region, using global stats")
        return (img_data - np.mean(img_data)) / np.std(img_data)
    
    reference_mean = np.mean(reference_region)
    if reference_mean == 0:
        print("Warning: Reference region mean is zero, using global stats")
        return (img_data - np.mean(img_data)) / np.std(img_data)
    
    return img_data / reference_mean

def normalize_with_percentile(img_data, low_percentile=5, high_percentile=95):
    """Normalize using percentile clipping"""
    non_zero_data = img_data[img_data > 0]
    if len(non_zero_data) == 0:
        return img_data
    
    low_val = np.percentile(non_zero_data, low_percentile)
    high_val = np.percentile(non_zero_data, high_percentile)
    clipped = np.clip(img_data, low_val, high_val)
    return (clipped - low_val) / (high_val - low_val)

for subject in os.listdir(input_root):
    if subject.startswith("._") or subject == ".DS_Store":
        continue

    in_dir = os.path.join(input_root, subject)
    if not os.path.isdir(in_dir):
        continue
    
    out_dir = os.path.join(output_root, subject)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if fname.startswith("._") or fname == ".DS_Store":
            continue

        input_path = os.path.join(in_dir, fname)
        output_path = os.path.join(out_dir, fname)

        if fname.endswith(".nii.gz"):
            print(f"Normalising {input_path} → {output_path}")
            try:
                img = nib.load(input_path)
                data = img.get_fdata()
                
                if args.method == 'reference':
                    mask_img = nib.load(occipital_mask_path)
                    mask_data = mask_img.get_fdata()
                    norm_data = normalize_with_reference_region(data, mask_data)
                else:
                    norm_data = normalize_with_percentile(data)
                
                norm_img = nib.Nifti1Image(norm_data, img.affine, img.header)
                nib.save(norm_img, output_path)
                print(f"✅ Successfully normalized {subject}")
            except Exception as e:
                print(f"❌ Failed on {input_path}: {e}")

        elif fname.endswith(".json"):
            shutil.copy(input_path, output_path)