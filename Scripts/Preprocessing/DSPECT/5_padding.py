import os
from pathlib import Path
import nibabel as nib
import numpy as np
import yaml
import argparse
import shutil

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

input_dir = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_masked"
output_dir = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_finalised"
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
subjects = [d for d in os.listdir(input_dir) if d.startswith('Subject_')]
failed_subjects = []

for subject_id in subjects:
    subj_input_dir = os.path.join(input_dir, subject_id)
    subj_output_dir = os.path.join(output_dir, subject_id)
    Path(subj_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Copy only essential files (NIfTI + JSON, skip DICOM folders)
    print(f"[COPY] {subject_id}: Copying essential files")
    for item in os.listdir(subj_input_dir):
        source_path = os.path.join(subj_input_dir, item)
        dest_path = os.path.join(subj_output_dir, item)
        
        if os.path.isfile(source_path):
            # Copy JSON files and the masked NIfTI
            if item.endswith('.json') or item == "4. masked.nii.gz":
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
    
    input_nii = os.path.join(subj_input_dir, "4. masked.nii.gz")
    output_nii = os.path.join(subj_output_dir, "5. finalised.nii.gz")

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

print(f"\n✅ Finalization complete! Output saved to: {output_dir}")
print(f"📁 Each subject now has a dedicated folder with finalised data and all original files.")
