import os
import re
import argparse
from typing import Optional
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import glob
from nibabel.orientations import OrientationError

# Detect runthrough early to control initial printing
_rt_parser = argparse.ArgumentParser(add_help=False)
_rt_parser.add_argument("--runthrough", type=str, default=None)
_rt_args, _ = _rt_parser.parse_known_args()
_is_runthrough = _rt_args.runthrough is not None

def _resolve_nii_path() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nii", type=str, default=None,
                        help="Path to a NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--runthrough", type=str, default=None,
                        help="Parent directory containing subject folders with NIfTI files. Enables left/right navigation.")
    args, _ = parser.parse_known_args()
    if args.nii:
        return os.path.expanduser(args.nii)
    # Fallback to previous default path (edit if desired)
    return "/Volumes/reseng202500013-ndd-ml/data/preprocessed/PET/PD/sub-I1518677_PPMI_PET_PD/sub-I1518677_PPMI_PET_PD_SUVR_s2_brain_soft4.nii.gz"

# Affine safety helpers (adapted from PET step 02)
def _is_affine_degenerate(affine: np.ndarray) -> bool:
    try:
        if affine is None or not np.all(np.isfinite(affine)):
            return True
        A = np.asarray(affine)[:3, :3]
        col_norms = np.linalg.norm(A, axis=0)
        if np.any(col_norms < 1e-6) or not np.all(np.isfinite(col_norms)):
            return True
        detA = float(np.linalg.det(A))
        if not np.isfinite(detA) or abs(detA) < 1e-6:
            return True
        return False
    except Exception:
        return True

def _repair_affine_image(img: nib.Nifti1Image) -> nib.Nifti1Image:
    if not _is_affine_degenerate(img.affine):
        return img
    try:
        zooms = img.header.get_zooms()[:3]
    except Exception:
        zooms = (2.0, 2.0, 2.0)
    safe_vx = np.array(zooms, dtype=float)
    safe_vx[~np.isfinite(safe_vx)] = 2.0
    safe_vx = np.maximum(np.abs(safe_vx), 0.5)

    original = np.asarray(img.affine)
    diag = np.diag(original[:3, :3])
    signs = np.sign(diag)
    signs[signs == 0] = 1.0

    new_affine = np.eye(4, dtype=float)
    new_affine[0, 0] = signs[0] * safe_vx[0]
    new_affine[1, 1] = signs[1] * safe_vx[1]
    new_affine[2, 2] = signs[2] * safe_vx[2]
    new_affine[:3, 3] = original[:3, 3]

    repaired = nib.Nifti1Image(img.get_fdata(dtype=np.float32), new_affine, img.header)
    try:
        repaired.set_sform(new_affine, code=1)
        repaired.set_qform(new_affine, code=1)
    except Exception:
        pass
    return repaired

# Load helpers: reduce 4D → 3D BEFORE canonical reorientation (match PET step 02)
def _load_reduced_image(path: str, reduce: str = "mean", frame: int = 0):
    """
    Load a NIfTI image. If 4D, collapse to 3D per `reduce` before calling
    nib.as_closest_canonical. Returns (img_3d_canonical, data_3d).
    """
    raw = nib.load(os.path.expanduser(path))
    arr = raw.get_fdata()
    if arr.ndim == 4:
        if reduce == "mean":
            arr3 = arr.mean(axis=3)
        elif reduce == "first":
            arr3 = arr[..., 0]
        elif reduce == "last":
            arr3 = arr[..., -1]
        elif reduce == "frame":
            fi = max(0, min(int(frame), arr.shape[3] - 1))
            arr3 = arr[..., fi]
        else:
            arr3 = arr.mean(axis=3)
    else:
        arr3 = arr
    # Ensure header shape matches 3D data shape
    hdr = raw.header.copy()
    try:
        hdr.set_data_shape(arr3.shape)
    except Exception:
        pass
    img3 = nib.Nifti1Image(arr3.astype(np.float32), raw.affine, hdr)
    # Keep original orientation/dimensions; only average frames. Repair degenerate affine if needed.
    if _is_affine_degenerate(img3.affine):
        img3 = _repair_affine_image(img3)
    return img3, img3.get_fdata()

# Load the NIfTI file and parse reduction args
nii_file = _resolve_nii_path()

# Reduction args (shared by initial load and runthrough updates)
_r_parser = argparse.ArgumentParser(add_help=False)
_r_parser.add_argument("--reduce", type=str, default="mean", choices=["mean", "first", "last", "frame"],
                    help="How to reduce a 4D volume to 3D for display")
_r_parser.add_argument("--frame", type=int, default=0, help="Frame index when --reduce=frame")
_rargs, _ = _r_parser.parse_known_args()

# Use the robust loader that collapses 4D first, then canonicalizes
img, data = _load_reduced_image(nii_file, reduce=_rargs.reduce, frame=_rargs.frame)

def _robust_vrange(vol: np.ndarray) -> tuple[float, float]:
    arr = vol.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mask = arr != 0
    if np.any(mask):
        lo, hi = np.percentile(arr[mask], [2.0, 98.0])
        if hi - lo < 1e-6:
            lo, hi = float(arr.min()), float(arr.max() if arr.max() > arr.min() else arr.min() + 1.0)
    else:
        lo, hi = 0.0, 1.0
    return float(lo), float(hi)

# (No-op: reduction already applied in _load_reduced_image)

if not _is_runthrough:
    print("  shape:", img.shape)
    print("  affine:\n", img.affine)
    print("  header zooms:", img.header.get_zooms())

# Optional: skip slow metadata lookup unless explicitly enabled
_meta_parser = argparse.ArgumentParser(add_help=False)
_meta_parser.add_argument("--skip-metadata", action="store_true", help="Skip loading imaging_records.csv for faster startup")
_meta_parser.add_argument("--force-metadata", action="store_true", help="Force metadata load even in runthrough mode")
_meta_parser.add_argument("--runthrough", type=str, default=None)
_meta_parser.add_argument("--save-dir", type=str, default=None, help="If set, save a PNG snapshot to this directory (non-interactive environments)")
_margs, _ = _meta_parser.parse_known_args()

metadata = {"SubjectID": "Unknown", "Dataset": "Unknown", "Modality": "Unknown", "Disease": "Unknown"}
do_meta = (not _margs.skip_metadata) and (not (_margs.runthrough and not _margs.force_metadata))
if do_meta:
    records_csv = os.path.expanduser("/Volumes/reseng202500013-ndd-ml/data/imaging_records.csv")
    if os.path.exists(records_csv):
        try:
            df = pd.read_csv(records_csv)
            row = df.loc[df.get('FilePath', '').astype(str) == nii_file]
            if row.empty:
                m = re.search(r"sub-([A-Za-z0-9]+)", nii_file)
                subject_candidate = m.group(1) if m else None
                if subject_candidate:
                    candidates = df[df['SubjectID'].astype(str) == subject_candidate]
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

# Lightweight metadata inference from file path when CSV is skipped
def _infer_metadata_from_path(path: str) -> dict:
    subj = "Unknown"; site = "Unknown"; modality = "Unknown"; disease = "Unknown"
    m = re.search(r"sub-([A-Za-z0-9]+)", path)
    if m:
        subj = m.group(1)
    for s in ["ADNI", "PPMI", "OASIS", "UKB"]:
        if s in path:
            site = s
            break
    for mod in ["MRI", "SPECT", "PET"]:
        if mod in path:
            modality = mod
            break
    for dis in ["CN", "PD", "AD", "MCI"]:
        if re.search(rf"[_/\\-]{dis}[_/\\-]", path):
            disease = dis
            break
    return {"SubjectID": subj, "Dataset": site, "Modality": modality, "Disease": disease}

# If metadata load was skipped or unresolved, infer from path for initial display
if not do_meta or metadata["SubjectID"] == "Unknown":
    try:
        metadata = _infer_metadata_from_path(nii_file)
    except Exception:
        pass

if not _is_runthrough:
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
if _is_runthrough:
    fig.suptitle("Runthrough mode: use ←/→ to navigate subjects", fontsize=12)
else:
    fig.suptitle(
        f"SubjectID: {metadata['SubjectID']} | Dataset: {metadata['Dataset']} | Modality: {metadata['Modality']} | Disease: {metadata['Disease']}",
        fontsize=12
    )

vmin, vmax = _robust_vrange(data)
im_ax = axes[0].imshow(data[:, :, init_axial].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
axes[0].set_title(f"Axial View (slice {init_axial})")

im_cor = axes[1].imshow(data[:, init_coronal, :].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
axes[1].set_title(f"Coronal View (slice {init_coronal})")

im_sag = axes[2].imshow(data[init_sagittal, :, :].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
axes[2].set_title(f"Sagittal View (slice {init_sagittal})")

plt.tight_layout()

# Optional runthrough mode: navigate a directory of subject folders with left/right keys
_nav_parser = argparse.ArgumentParser(add_help=False)
_nav_parser.add_argument("--runthrough", type=str, default=None,
                        help="Parent directory containing subject folders with NIfTI files. Enables left/right navigation.")
_nav_args, _ = _nav_parser.parse_known_args()

def _find_subject_nii(subject_dir: str) -> Optional[str]:
    # Fast non-recursive glob within subject_dir with explicit priority
    patterns = [
        # Highest priority: final SUVRs (explicit PET patterns first)
        "sub-*_*_PET_*_SUVR_s2_brain_soft4.nii.gz",
        "sub-*_*_PET_*_SUVR_s2_brain_round_med1.nii.gz",
        "sub-*_*_PET_*_SUVR_s2_brain_med1.nii.gz",
        "sub-*_*_PET_*_SUVR_s2_brain_round.nii.gz",
        "sub-*_*_PET_*_SUVR_s2_brain.nii.gz",
        "sub-*_*_PET_*_SUVR_s2.nii.gz",
        "sub-*_*_PET_*_SUVR_Z.nii.gz",
        "sub-*_*_PET_*_SUVR.nii.gz",
        # Generic SUVR fallbacks
        "*_SUVR_s2_brain_soft4.nii.gz",
        "*_SUVR_s2_brain_round_med1.nii.gz",
        "*_SUVR_s2_brain_med1.nii.gz",
        "*_SUVR_s2_brain_round.nii.gz",
        "*_SUVR_s2_brain.nii.gz",
        "*_SUVR_s2.nii.gz",
        "*_SUVR_Z.nii.gz",
        "*_SUVR.nii.gz",
        # Useful anatomical fallbacks
        "*_MNI_brain.nii.gz",
        "*_static_brain.nii.gz",
        "*_static_presmooth.nii.gz",
        "*_static.nii.gz",
        # Generic fallbacks
        "*.nii.gz",
        "*.nii",
    ]
    for pat in patterns:
        hits = glob.glob(os.path.join(subject_dir, pat))
        if hits:
            return hits[0]
    # If nothing found, check one level below (common single nested folder)
    for name in os.listdir(subject_dir):
        p = os.path.join(subject_dir, name)
        if os.path.isdir(p):
            for pat in patterns:
                hits = glob.glob(os.path.join(p, pat))
                if hits:
                    return hits[0]
    return None

if _nav_args.runthrough:
    base_dir = os.path.expanduser(_nav_args.runthrough)
    if os.path.isdir(base_dir):
        subjects = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
        # Lazy resolution of NIfTI per subject for fast startup
        nii_cache: dict[int, str] = {}
        if len(subjects) > 0:
            current_idx = 0
            # Start at the first subject that has a preferred NIfTI to avoid mismatched first display
            for i, d in enumerate(subjects):
                p = _find_subject_nii(os.path.join(base_dir, d))
                if p is not None:
                    current_idx = i
                    nii_cache[current_idx] = p
                    break

            def _update_to(path: str):
                global img, data, init_axial, init_coronal, init_sagittal, metadata
                try:
                    img, data = _load_reduced_image(path, reduce=_rargs.reduce, frame=_rargs.frame)
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    return
                shape = data.shape
                init_axial = shape[2] // 2
                init_coronal = shape[1] // 3
                init_sagittal = shape[0] // 2
                vmin, vmax = _robust_vrange(data)
                im_ax.set_data(data[:, :, init_axial].T)
                im_ax.set_clim(vmin=vmin, vmax=vmax)
                axes[0].set_title(f"Axial View (slice {init_axial})")
                im_cor.set_data(data[:, init_coronal, :].T)
                im_cor.set_clim(vmin=vmin, vmax=vmax)
                axes[1].set_title(f"Coronal View (slice {init_coronal})")
                im_sag.set_data(data[init_sagittal, :, :].T)
                im_sag.set_clim(vmin=vmin, vmax=vmax)
                axes[2].set_title(f"Sagittal View (slice {init_sagittal})")
                # Refresh metadata per subject when navigating
                try:
                    metadata = _infer_metadata_from_path(path)
                except Exception:
                    pass
                fig.suptitle(
                    f"SubjectID: {metadata['SubjectID']} | Dataset: {metadata['Dataset']} | Modality: {metadata['Modality']} | Disease: {metadata['Disease']}",
                    fontsize=12
                )
                fig.canvas.draw_idle()

            def _on_key(event):
                global current_idx
                if event.key in ('right', 'n', 'd'):
                    current_idx = (current_idx + 1) % len(subjects)
                elif event.key in ('left', 'p', 'a'):
                    current_idx = (current_idx - 1) % len(subjects)
                elif event.key in ('s', 'S'):
                    # Save snapshot of current view
                    out_dir = os.path.expanduser(_margs.save_dir) if _margs.save_dir else base_dir
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"middle_slice_{subjects[current_idx]}.png")
                    fig.savefig(out_path, dpi=200, bbox_inches='tight')
                    print(f"Saved snapshot: {out_path}")
                else:
                    return
                # Resolve NIfTI for the current subject lazily
                if current_idx not in nii_cache:
                    subj_dir = os.path.join(base_dir, subjects[current_idx])
                    nii_path = _find_subject_nii(subj_dir)
                    if nii_path is None:
                        print(f"No NIfTI found for {subjects[current_idx]}")
                        return
                    nii_cache[current_idx] = nii_path
                _update_to(nii_cache[current_idx])

            fig.canvas.mpl_connect('key_press_event', _on_key)
            # Show the selected starting subject immediately
            if current_idx in nii_cache:
                _update_to(nii_cache[current_idx])
        else:
            print(f"No NIfTI files found under: {base_dir}")
    else:
        print(f"--runthrough directory not found: {base_dir}")

# In headless backends (e.g., Agg) show() does nothing; save a snapshot instead
backend = mpl.get_backend().lower()

# Always save a preview if --save-dir is provided (useful when GUI can't open)
if _margs.save_dir:
    out_dir = os.path.expanduser(_margs.save_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "middle_slice_preview.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved preview to: {out_path}")

if 'agg' in backend and not _margs.save_dir:
    # Headless and no save-dir specified: save next to the input
    out_dir = os.path.dirname(nii_file)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "middle_slice_preview.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Non-interactive backend detected ({backend}). Saved preview to: {out_path}")
else:
    # Try to show the window (may still not appear in some remote sessions)
    try:
        plt.show()
    except Exception as e:
        print(f"GUI display failed: {e}")
