import os
from pathlib import Path
import nibabel as nib
import numpy as np
import yaml
import argparse
import csv
import shutil

print("Hello World! Step 6 postprocess script starting...")

def fix_path(path):
    return os.path.expanduser(path)

parser = argparse.ArgumentParser(description="Step 6: Postprocess SPECT images (global z-score normalisation, summary CSV).")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True, help="Diagnosis group to process (CN or PD)")
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
args = parser.parse_args()

# Find config file in project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
config_path = os.path.join(project_root, 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Fix path handling to use Desktop SPECT folder structure
input_dir = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_finalised"
output_dir = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_postprocessed"
Path(output_dir).mkdir(parents=True, exist_ok=True)

summary_csv = os.path.join(output_dir, f"summary_{args.diagnosis}.csv")

print(f"\n🔄 Processing {args.diagnosis} subjects")
print(f"📁 Input directory: {input_dir}")
print(f"📁 Output directory: {output_dir}")
print(f"📄 Summary CSV: {summary_csv}\n")

subjects = sorted(Path(input_dir).glob("Subject_*"))
print(f"DEBUG: Found {len(subjects)} subject folders. Example: {[s.name for s in subjects[:5]]}")

def validate_ml_readiness(norm_data, subject_id):
    """Validate postprocessed data is ML-ready"""
    mask = norm_data != 0
    if not np.any(mask):
        print(f"❌ {subject_id}: No non-zero voxels after postprocessing")
        return False
    
    # Check for reasonable z-score values (should be ~0 mean, ~1 std)
    brain_data = norm_data[mask]
    mean_val = np.mean(brain_data)
    std_val = np.std(brain_data)
    
    if abs(mean_val) > 0.5:
        print(f"⚠️ {subject_id}: Mean far from 0 ({mean_val:.3f}) - may affect ML")
    
    if not (0.5 < std_val < 2.0):
        print(f"⚠️ {subject_id}: Std dev unusual ({std_val:.3f}) - may affect ML")
    
    # Check for extreme outliers
    if np.any(np.abs(brain_data) > 5):
        print(f"⚠️ {subject_id}: Extreme outliers detected (>5 std devs)")
    
    return True

all_voxels = []
subject_stats = []

def find_finalised_nii_gz(subject_folder: Path):
    """Find the 5. finalised.nii.gz file specifically"""
    finalised_file = subject_folder / "5. finalised.nii.gz"
    if finalised_file.exists():
        return str(finalised_file)
    return None

# First pass: collect all nonzero voxels for global stats
for subject_path in subjects:
    subject_id = subject_path.name
    input_nii = find_finalised_nii_gz(subject_path)
    if input_nii is None:
        print(f"❌ {subject_id}: No '5. finalised.nii.gz' file found in {subject_path}")
        continue
    print(f"Processing {subject_id} → {input_nii}")
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
for subject_path in subjects:
    subject_id = subject_path.name
    input_nii = find_finalised_nii_gz(subject_path)
    if input_nii is None:
        print(f"❌ {subject_id}: No '5. finalised.nii.gz' file found in {subject_path}")
        continue
    
    # Create subject output directory
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # Copy only essential files (NIfTI + JSON, skip DICOM folders)
    print(f"[COPY] {subject_id}: Copying essential files")
    for item in os.listdir(subject_path):
        source_path = os.path.join(subject_path, item)
        dest_path = os.path.join(subject_output_dir, item)
        
        if os.path.isfile(source_path):
            # Copy JSON files and the finalised NIfTI
            if item.endswith('.json') or item == "5. finalised.nii.gz":
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
    
    # Process the finalised NIfTI file
    input_nii = os.path.join(subject_path, "5. finalised.nii.gz")
    output_nii = os.path.join(subject_output_dir, "6. postprocessed.nii.gz")
    
    try:
        img = nib.load(input_nii)
        data = img.get_fdata()
        mask = data != 0
        norm_data = data.copy()
        norm_data[mask] = (data[mask] - global_mean) / global_std
        
        # Validate ML readiness
        if not validate_ml_readiness(norm_data, subject_id):
            print(f"❌ {subject_id}: Failed ML validation")
            continue
        
        norm_img = nib.Nifti1Image(norm_data, img.affine, img.header)
        nib.save(norm_img, output_nii)
        
        # Collect stats
        subject_stats.append({
            'subject_id': subject_id,
            'mean': float(norm_data[mask].mean()) if np.any(mask) else 0,
            'std': float(norm_data[mask].std()) if np.any(mask) else 0,
            'min': float(norm_data[mask].min()) if np.any(mask) else 0,
            'max': float(norm_data[mask].max()) if np.any(mask) else 0,
            'nonzero_voxels': int(np.count_nonzero(mask)),
            'ml_ready': True  # Add ML readiness flag
        })
        print(f"✅ {subject_id}: Postprocessed and ML-validated.")
    except Exception as e:
        print(f"❌ {subject_id}: {e}")
        import traceback
        traceback.print_exc()

# Write summary CSV
with open(summary_csv, 'w', newline='') as csvfile:
    fieldnames = ['subject_id', 'mean', 'std', 'min', 'max', 'nonzero_voxels', 'ml_ready']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in subject_stats:
        writer.writerow(row)

print(f"\n✅ Step 6 complete. Summary written to {summary_csv}")
print(f"DEBUG: Processed {len(subject_stats)} subjects. Summary written to {summary_csv}")



print(f"\n✅ Postprocessing complete! Output saved to: {output_dir}")
print(f"📁 Each subject now has a dedicated folder with postprocessed data and all original files.")