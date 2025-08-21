import os
import re
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interact, IntSlider

# Load the NIfTI file
nii_file = "/Volumes/reseng202500013-ndd-ml/data/preprocessed/smriprep/smriprep/sub-I288115/anat/sub-I288115_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz"

img = nib.load(nii_file)
data = img.get_fdata()

print("  shape:", img.shape)
print("  affine:\n", img.affine)
print("  header zooms:", img.header.get_zooms())

# Load metadata from imaging_records.csv and match this file
records_csv = os.path.expanduser("/Volumes/reseng202500013-ndd-ml/data/imaging_records.csv")
metadata = {"SubjectID": "Unknown", "Dataset": "Unknown", "Modality": "Unknown", "Disease": "Unknown"}
if os.path.exists(records_csv):
    try:
        df = pd.read_csv(records_csv)
        # Try to match by exact FilePath first
        row = df.loc[df.get('FilePath', '').astype(str) == nii_file]
        if row.empty:
            # Extract subject candidate from path (e.g., sub-AR00163 -> AR00163)
            m = re.search(r"sub-([A-Za-z0-9]+)", nii_file)
            subject_candidate = m.group(1) if m else None
            if subject_candidate:
                candidates = df[df['SubjectID'].astype(str) == subject_candidate]
                # Optional: filter by modality inferred from path
                modality_guess = None
                for mod in ["MRI", "SPECT", "PET"]:
                    if mod in nii_file:
                        modality_guess = mod
                        break
                if modality_guess is not None:
                    candidates = candidates[candidates['Modality'].astype(str).str.upper() == modality_guess]
                if not candidates.empty:
                    row = candidates.iloc[[0]]
        if not row.empty:
            r = row.iloc[0]
            metadata = {
                "SubjectID": str(r.get('SubjectID', 'Unknown')),
                "Dataset": str(r.get('Site', 'Unknown')),
                "Modality": str(r.get('Modality', 'Unknown')),
                "Disease": str(r.get('Disease', 'Unknown')),
            }
    except Exception as e:
        print(f"Warning: could not read metadata: {e}")

print(f"SubjectID: {metadata['SubjectID']} | Dataset: {metadata['Dataset']} | Modality: {metadata['Modality']} | Disease: {metadata['Disease']}")

# Get the shape of the image
shape = data.shape
print(f"Image shape: {shape}")

# Initial slice indices for each orientation
init_axial = shape[2] // 2    # Axial plane (Z-axis)
init_coronal = shape[1] // 3   # Coronal plane (Y-axis)
init_sagittal = shape[0] // 2  # Sagittal plane (X-axis)

def view_slices(axial_idx, coronal_idx, sagittal_idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"SubjectID: {metadata['SubjectID']} | Dataset: {metadata['Dataset']} | Modality: {metadata['Modality']} | Disease: {metadata['Disease']}",
        fontsize=12
    )

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
