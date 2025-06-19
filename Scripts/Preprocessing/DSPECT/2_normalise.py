import os
import nibabel as nib
import numpy as np
import shutil

input_root = "/Volumes/FlaireHD/P4P/SPECT/CN/reoriented"
output_root = "/Volumes/FlaireHD/P4P/SPECT/CN/normalised"

for subject in os.listdir(input_root):
    if subject.startswith("._"):
        continue  # skip macOS metadata

    in_dir = os.path.join(input_root, subject)
    out_dir = os.path.join(output_root, subject)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if fname.startswith("._"):
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
            except Exception as e:
                print(f"❌ Failed on {input_path}: {e}")

        elif fname.endswith(".json"):
            shutil.copy(input_path, output_path)  # Copy sidecar JSON