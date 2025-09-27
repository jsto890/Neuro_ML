import os
import re
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def _resolve_nii_path() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nii", type=str, default=None,
                        help="Path to a NIfTI file (.nii or .nii.gz)")
    args, _ = parser.parse_known_args()
    if args.nii:
        return os.path.expanduser(args.nii)
    # Fallback to previous default path (edit if desired)
    return "/Volumes/reseng202500013-ndd-ml/data/preprocessed/PET/PD/sub-I1518677_PPMI_PET_PD/sub-I1518677_PPMI_PET_PD_SUVR_s2_brain_soft4.nii.gz"

# Load the NIfTI file
nii_file = _resolve_nii_path()

img = nib.load(nii_file)
data = img.get_fdata()

# If 4D (e.g., dynamic PET), reduce to 3D for display
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--reduce", type=str, default="mean", choices=["mean", "first", "last", "frame"],
                    help="How to reduce a 4D volume to 3D for display")
_parser.add_argument("--frame", type=int, default=0, help="Frame index when --reduce=frame")
_args, _ = _parser.parse_known_args()
if data.ndim == 4:
    if _args.reduce == "mean":
        data = data.mean(axis=-1)
    elif _args.reduce == "first":
        data = data[..., 0]
    elif _args.reduce == "last":
        data = data[..., -1]
    elif _args.reduce == "frame":
        fi = max(0, min(_args.frame, data.shape[-1] - 1))
        data = data[..., fi]

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

# Static 3-panel figure (no sliders)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"SubjectID: {metadata['SubjectID']} | Dataset: {metadata['Dataset']} | Modality: {metadata['Modality']} | Disease: {metadata['Disease']}",
    fontsize=12
)

axes[0].imshow(data[:, :, init_axial].T, cmap="gray", origin="lower")
axes[0].set_title(f"Axial View (slice {init_axial})")

axes[1].imshow(data[:, init_coronal, :].T, cmap="gray", origin="lower")
axes[1].set_title(f"Coronal View (slice {init_coronal})")

axes[2].imshow(data[init_sagittal, :, :].T, cmap="gray", origin="lower")
axes[2].set_title(f"Sagittal View (slice {init_sagittal})")

plt.tight_layout()
plt.show()
