import nibabel as nib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser(description="Visualise a single DAT SPECT NIfTI image.")
parser.add_argument("--nii_path", type=str, help="Path to the NIfTI file to visualise.")
args = parser.parse_args()

nii_path = args.nii_path
if not nii_path:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        nii_path = filedialog.askopenfilename(
            title="Select NIfTI file",
            filetypes=[
                ("All files", "*.*")
            ],
            initialdir="/Volumes/reseng202500013-ndd-ml/data/preprocessed/SPECT/reoriented/CN",
        )
        if not nii_path:
            sys.exit(0)
    except Exception as e:
        print(f"Could not open file dialog. Error: {e}")
        print("Please provide --nii_path.")
        sys.exit(1)

img = nib.load(nii_path)
data = img.get_fdata()

# === Basic sanity checks ===
flat = data.ravel()
total_voxels = flat.size

# Check for excessive zeros
zero_count = np.sum(flat == 0)
zero_pct = zero_count / total_voxels * 100
if zero_pct > 90:
    print(f"Warning: {zero_pct:.2f}% of voxels are zero.")

# Check for NaNs and Infs
nan_count = np.isnan(flat).sum()
if nan_count > 0:
    print(f"Warning: {nan_count} NaN voxels detected.")

inf_count = np.isinf(flat).sum()
if inf_count > 0:
    print(f"Warning: {inf_count} infinite voxels detected.")

# Check for unexpected negative values
neg_count = np.sum(flat < 0)
if neg_count > 0:
    print(f"Warning: {neg_count} negative voxels detected (min={flat.min():.2f}).")

z_index = data.shape[2] // 2
slice_data = data[:, :, z_index].T

vmin, vmax = np.percentile(slice_data[slice_data > 0], [5, 95]) if np.any(slice_data > 0) else (None, None)

plt.imshow(slice_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
plt.title(f"Middle Slice: {nii_path}")
plt.axis("off")
plt.show() 