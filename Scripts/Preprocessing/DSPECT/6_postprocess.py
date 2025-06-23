import os
from pathlib import Path
import nibabel as nib
import numpy as np
import yaml
import argparse
import csv

print("Hello World! Step 6 postprocess script starting...")

def fix_path(path):
    return os.path.expanduser(path)

parser = argparse.ArgumentParser(description="Step 6: Postprocess SPECT images (global z-score normalisation, summary CSV).")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True, help="Diagnosis group to process (CN or PD)")
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
args = parser.parse_args()

if args.isHasel:
    base_dir = os.path.expanduser('~/reseng202500013-ndd-ml')
else:
    base_dir = '/Volumes/reseng202500013-ndd-ml'

def full_path(path):
    if path.startswith("~/") or path.startswith("/Volumes/"):
        return os.path.join(base_dir, os.path.relpath(os.path.expanduser(path), start=os.path.expanduser('~/reseng202500013-ndd-ml')))
    return path

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

input_dir = os.path.join(full_path(config['preprocessed_data']['spect_p']), 'finalised', args.diagnosis)
output_dir = os.path.join(full_path(config['preprocessed_data']['spect_p']), 'postprocessed', args.diagnosis)
Path(output_dir).mkdir(parents=True, exist_ok=True)

summary_csv = os.path.join(output_dir, f"summary_{args.diagnosis}.csv")

print(f"\n🔄 Processing {args.diagnosis} subjects")
print(f"📁 Input directory: {input_dir}")
print(f"📁 Output directory: {output_dir}")
print(f"📄 Summary CSV: {summary_csv}\n")

subjects = [d for d in os.listdir(input_dir) if d.startswith('sub-')]
all_voxels = []
subject_stats = []

def find_first_nii_gz(subject_folder):
    files = [f for f in os.listdir(subject_folder) if f.endswith('.nii.gz')]
    if not files:
        return None
    return os.path.join(subject_folder, files[0])

# First pass: collect all nonzero voxels for global stats
for subject_id in subjects:
    subject_folder = os.path.join(input_dir, subject_id)
    input_nii = find_first_nii_gz(subject_folder)
    if input_nii is None:
        print(f"❌ {subject_id}: No .nii.gz file found in {subject_folder}")
        continue
    try:
        img = nib.load(input_nii)
        data = img.get_fdata()
        mask = data != 0
        all_voxels.append(data[mask])
    except Exception as e:
        print(f"❌ {subject_id}: {e}")

if not all_voxels:
    print("❌ No valid voxels found. Exiting.")
    exit(1)

all_voxels_flat = np.concatenate(all_voxels)
global_mean = all_voxels_flat.mean()
global_std = all_voxels_flat.std()

print(f"Global mean: {global_mean:.4f}, Global std: {global_std:.4f}")

# Second pass: normalise and save, collect stats
for subject_id in subjects:
    subject_folder = os.path.join(input_dir, subject_id)
    input_nii = find_first_nii_gz(subject_folder)
    output_nii = os.path.join(output_dir, f"{subject_id}_postprocessed.nii.gz")
    if input_nii is None:
        print(f"❌ {subject_id}: No .nii.gz file found in {subject_folder}")
        continue
    try:
        img = nib.load(input_nii)
        data = img.get_fdata()
        mask = data != 0
        norm_data = data.copy()
        norm_data[mask] = (data[mask] - global_mean) / global_std
        norm_img = nib.Nifti1Image(norm_data, img.affine, img.header)
        nib.save(norm_img, output_nii)
        # Collect stats
        subject_stats.append({
            'subject_id': subject_id,
            'mean': float(norm_data[mask].mean()) if np.any(mask) else 0,
            'std': float(norm_data[mask].std()) if np.any(mask) else 0,
            'min': float(norm_data[mask].min()) if np.any(mask) else 0,
            'max': float(norm_data[mask].max()) if np.any(mask) else 0,
            'nonzero_voxels': int(np.count_nonzero(mask))
        })
        print(f"✅ {subject_id}: Postprocessed and saved.")
    except Exception as e:
        print(f"❌ {subject_id}: {e}")

# Write summary CSV
with open(summary_csv, 'w', newline='') as csvfile:
    fieldnames = ['subject_id', 'mean', 'std', 'min', 'max', 'nonzero_voxels']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in subject_stats:
        writer.writerow(row)

print(f"\n✅ Step 6 complete. Summary written to {summary_csv}") 