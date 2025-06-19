import os
from pathlib import Path
import nibabel as nib
import numpy as np
import shutil
from nilearn.image import resample_to_img, load_img
from nilearn.image.resampling import BoundingBoxError

# === CONFIG ===
input_dir = "/Volumes/FlaireHD/P4P/SPECT/CN/normalised"
template_path = "/Users/jacksonschofield/Desktop/P4P/Templates/DSPECT_refs/symFPCITtemplate_MNI_norm.nii"
output_base = "/Volumes/FlaireHD/P4P/SPECT/CN/registered"

# === HELPERS ===
def recenter_affine(img):
    data = img.get_fdata()
    shape = data.shape
    current_affine = img.affine
    voxel_sizes = np.sqrt((current_affine[:3, :3] ** 2).sum(0))
    new_affine = np.diag(list(voxel_sizes) + [1])
    new_affine[:3, 3] = -np.array(shape) * voxel_sizes / 2
    return nib.Nifti1Image(data, new_affine, img.header)

# === RUN ===
template = load_img(template_path)
subject_dirs = [d for d in os.listdir(input_dir) if d.startswith("sub-")]

failed_subjects = []

for subject_id in subject_dirs:
    if subject_id.startswith("._"):
        continue  # skip macOS junk

    print(f"\n🔄 Registering {subject_id}")
    subj_input_dir = os.path.join(input_dir, subject_id)
    output_dir = os.path.join(output_base, subject_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Copy ALL JSON files
    for fname in os.listdir(subj_input_dir):
        if fname.endswith(".json") and not fname.startswith("._"):
            shutil.copy(os.path.join(subj_input_dir, fname), os.path.join(output_dir, fname))

    # Proceed with registration
    input_nii = os.path.join(subj_input_dir, f"{subject_id}_RAS.nii.gz")
    output_nii = os.path.join(output_dir, f"{subject_id}_registered.nii.gz")

    try:
        img = load_img(input_nii)
        resampled = resample_to_img(img, template, interpolation='continuous')
        resampled.to_filename(output_nii)
        print(f"✅ Saved: {output_nii}")

    except BoundingBoxError as e:
        print(f"❌ BoundingBoxError for {subject_id}: {e}")
        print("🛠 Trying fallback: recenter affine...")
        try:
            recentered_img = recenter_affine(img)
            resampled = resample_to_img(
                recentered_img,
                template,
                interpolation='continuous',
                copy_header=True,
                force_resample=True
            )
            resampled.to_filename(output_nii)
            print(f"✅ Saved after recentering: {output_nii}")
        except Exception as e2:
            print(f"❌ Recentered registration failed for {subject_id}: {e2}")
            failed_subjects.append(subject_id)

    except Exception as e:
        print(f"❌ Unexpected error for {subject_id}: {e}")
        failed_subjects.append(subject_id)

# === SUMMARY ===
if failed_subjects:
    print("\n======================")
    print("❌ FAILED SUBJECTS")
    print("======================")
    for sid in failed_subjects:
        print(sid)
else:
    print("\n✅ All subjects registered successfully.")