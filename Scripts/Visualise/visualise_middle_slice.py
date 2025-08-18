import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

# Load the NIfTI file
nii_file = "/Volumes/reseng202500013-ndd-ml/data/preprocessed/NEWPET/ADNI/sub-I1373209_ADNI_PET_CN/sub-I1373209_ADNI_PET_CN_SUVR_s2.nii.gz"

img = nib.load(nii_file)
data = img.get_fdata()

print("  shape:", img.shape)
print("  affine:\n", img.affine)
print("  header zooms:", img.header.get_zooms())

# Get the shape of the image
shape = data.shape
print(f"Image shape: {shape}")

# Initial slice indices for each orientation
init_axial = shape[2] // 2    # Axial plane (Z-axis)
init_coronal = shape[1] // 3   # Coronal plane (Y-axis)
init_sagittal = shape[0] // 2  # Sagittal plane (X-axis)

def view_slices(axial_idx, coronal_idx, sagittal_idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial view (Top-down)
    axes[0].imshow(data[:, :, axial_idx].T, cmap="gray", origin="lower")
    axes[0].set_title(f"Axial View (slice {axial_idx})")

    # Coronal view (Front-facing)
    axes[1].imshow(data[:, coronal_idx, :].T, cmap="gray", origin="lower")
    axes[1].set_title(f"Coronal View (slice {coronal_idx})")

    # Sagittal view (Side-facing)
    axes[2].imshow(data[sagittal_idx, :, :].T, cmap="gray", origin="lower")
    axes[2].set_title(f"Sagittal View (slice {sagittal_idx})")

    plt.show()

# Create interactive sliders for each plane
interact(view_slices,
         axial_idx=IntSlider(min=0, max=shape[2]-1, step=1, value=init_axial, description="Axial"),
         coronal_idx=IntSlider(min=0, max=shape[1]-1, step=1, value=init_coronal, description="Coronal"),
         sagittal_idx=IntSlider(min=0, max=shape[0]-1, step=1, value=init_sagittal, description="Sagittal"))
