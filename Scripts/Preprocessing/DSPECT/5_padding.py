import os
from pathlib import Path
import nibabel as nib
import numpy as np
import yaml
import argparse

print("Hello World! Step 5 finalise script starting...")

def fix_path(path):
    # Use os.path.expanduser for portability
    return os.path.expanduser(path)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Finalise SPECT images: pad/crop to fixed shape and optionally normalise intensity.")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True, help="Diagnosis group to process (CN or PD)")
parser.add_argument("--shape", type=int, nargs=3, default=[91, 109, 91], help="Target shape for all images (default: 91 109 91)")
parser.add_argument("--intensity_norm", action="store_true", help="Apply zero-mean, unit-variance normalisation (default: off)")
args = parser.parse_args()

# Find config file in project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
config_path = os.path.join(project_root, 'config.yaml')

# --- Load config ---
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

input_dir = os.path.join(fix_path(config['preprocessed_data']['spect_p']), 'masked', args.diagnosis)
output_dir = os.path.join(fix_path(config['preprocessed_data']['spect_p']), 'finalised', args.diagnosis)
Path(output_dir).mkdir(parents=True, exist_ok=True)

print(f"\n🔄 Processing {args.diagnosis} subjects")
print(f"📁 Input directory: {input_dir}")
print(f"📁 Output directory: {output_dir}")
print(f"📐 Target shape: {args.shape}")
print(f"⚖️ Intensity normalisation: {'ON' if args.intensity_norm else 'OFF'}\n")

def pad_or_crop(data, target_shape):
    """Pad or crop a 3D numpy array to the target shape."""
    current_shape = data.shape
    pad_width = []
    slices = []
    for i in range(3):
        diff = target_shape[i] - current_shape[i]
        if diff > 0:
            # Pad
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
            slices.append(slice(0, current_shape[i]))
        elif diff < 0:
            # Crop
            crop_before = (-diff) // 2
            crop_after = crop_before + target_shape[i]
            pad_width.append((0, 0))
            slices.append(slice(crop_before, crop_after))
        else:
            pad_width.append((0, 0))
            slices.append(slice(0, current_shape[i]))
    # Crop first, then pad
    cropped = data[slices[0], slices[1], slices[2]]
    padded = np.pad(cropped, pad_width, mode='constant', constant_values=0)
    return padded

# --- Process subjects ---
subjects = [d for d in os.listdir(input_dir) if d.startswith('sub-')]
failed_subjects = []

for subject_id in subjects:
    subj_input_dir = os.path.join(input_dir, subject_id)
    subj_output_dir = os.path.join(output_dir, subject_id)
    Path(subj_output_dir).mkdir(parents=True, exist_ok=True)
    input_nii = os.path.join(subj_input_dir, f"{subject_id}_masked.nii.gz")
    output_nii = os.path.join(subj_output_dir, f"{subject_id}_finalised.nii.gz")

    try:
        img = nib.load(input_nii)
        data = img.get_fdata()
        # --- Pad or crop ---
        final_data = pad_or_crop(data, tuple(args.shape))
        # --- Optional intensity normalisation ---
        if args.intensity_norm:
            # Only normalise nonzero voxels (brain region)
            mask = final_data != 0
            mean = final_data[mask].mean() if np.any(mask) else 0
            std = final_data[mask].std() if np.any(mask) else 1
            final_data[mask] = (final_data[mask] - mean) / std
        # --- Save ---
        final_img = nib.Nifti1Image(final_data, img.affine, img.header)
        nib.save(final_img, output_nii)
        print(f"✅ {subject_id}: Saved {output_nii}")
    except Exception as e:
        print(f"❌ {subject_id}: {e}")
        failed_subjects.append(subject_id)

# --- Summary ---
if failed_subjects:
    print("\n======================")
    print("❌ FAILED SUBJECTS")
    print("======================")
    for sid in failed_subjects:
        print(sid)
else:
    print("\n✅ All subjects finalised successfully.")
