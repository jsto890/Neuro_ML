import os
import nibabel as nib
import numpy as np
import shutil
import yaml
import argparse

def fix_path(path):
    """Convert config path to actual mounted path"""
    # Since we're running from inside the research drive, just expand ~
    return os.path.expanduser(path)

# Set up argument parser
parser = argparse.ArgumentParser(description="Normalize SPECT images.")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True, 
                    help="Diagnosis group to process (CN or PD)")
args = parser.parse_args()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

input_root = os.path.join(fix_path(config['preprocessed_data']['spect_p']), 'reoriented', args.diagnosis)
output_root = os.path.join(fix_path(config['preprocessed_data']['spect_p']), 'normalised', args.diagnosis)

print(f"\n🔄 Processing {args.diagnosis} subjects")
print(f"📁 Input directory: {input_root}")
print(f"📁 Output directory: {output_root}\n")

for subject in os.listdir(input_root):
    if subject.startswith("._") or subject == ".DS_Store":
        continue  # skip macOS metadata

    in_dir = os.path.join(input_root, subject)
    
    # Check if it's actually a directory
    if not os.path.isdir(in_dir):
        continue  # skip non-directory items
    
    out_dir = os.path.join(output_root, subject)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if fname.startswith("._") or fname == ".DS_Store":
            continue  # skip macOS files

        input_path = os.path.join(in_dir, fname)
        output_path = os.path.join(out_dir, fname)

        if fname.endswith(".nii.gz"):
            print(f"Normalising {input_path} → {output_path}")
            try:
                img = nib.load(input_path)
                data = img.get_fdata()
                norm_data = (data - np.mean(data)) / np.std(data)
                norm_img = nib.Nifti1Image(norm_data, img.affine, img.header)
                nib.save(norm_img, output_path)
                print(f"✅ Successfully normalized {subject}")
            except Exception as e:
                print(f"❌ Failed on {input_path}: {e}")

        elif fname.endswith(".json"):
            shutil.copy(input_path, output_path)  # Copy sidecar JSON