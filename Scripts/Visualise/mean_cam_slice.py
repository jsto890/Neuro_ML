#!/usr/bin/env python3
"""
Mean Grad-CAM slice visualisation (MNI space)
============================================

Compute an average Grad-CAM (or Grad-CAM++) volume across subjects and save:
- mean NIfTI
- a mid-sagittal slice PNG (optionally overlaid on a base anatomical image)

Typical use-case: average ~30 subject CAMs for a disease group and include a single
summary panel in a report/manuscript.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception as e:
    raise SystemExit("Please install nibabel (e.g. pip install nibabel)") from e

try:
    from nibabel.processing import resample_from_to  # type: ignore
except Exception:
    resample_from_to = None  # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _load_3d(path: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data.mean(axis=-1)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI at {path}, got shape {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return img, data


def _robust01(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    x = x.astype(np.float32)
    v0 = float(np.percentile(x, lo))
    v1 = float(np.percentile(x, hi))
    if v1 - v0 < 1e-6:
        return x
    y = (x - v0) / (v1 - v0)
    return np.clip(y, 0.0, 1.0)


def _find_inputs(input_root: str, pattern: str) -> List[str]:
    p = Path(input_root)
    if not p.is_dir():
        raise ValueError(f"--input_root is not a directory: {input_root}")
    hits = sorted([str(x) for x in p.rglob(pattern) if x.is_file()])
    return hits


def _stack_mean(paths: List[str], normalise_each: bool = False) -> Tuple[nib.Nifti1Image, np.ndarray]:
    if not paths:
        raise ValueError("No input CAM files found.")
    ref_img, ref = _load_3d(paths[0])
    stack = np.zeros((len(paths),) + ref.shape, dtype=np.float32)
    stack[0] = _robust01(ref) if normalise_each else ref
    for i, fp in enumerate(paths[1:], start=1):
        _, d = _load_3d(fp)
        if d.shape != ref.shape:
            raise ValueError(f"Shape mismatch: {fp} has {d.shape}, expected {ref.shape}")
        stack[i] = _robust01(d) if normalise_each else d
    return ref_img, stack.mean(axis=0).astype(np.float32)


def _get_mid_sagittal_idx(shape: Tuple[int, int, int]) -> int:
    # X axis is dim0 in nibabel array ordering for many MNI images, but we treat it as "sagittal axis"
    return int(shape[0] // 2)


def _save_mid_sagittal_png(
    base: Optional[np.ndarray],
    heat: np.ndarray,
    out_png: str,
    title: str,
    alpha: float = 0.45,
    white_bg: bool = True,
    base_interp: str = "bicubic",
    heat_interp: str = "nearest",
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.5, 5.5),
) -> None:
    x = _get_mid_sagittal_idx(heat.shape)
    heat2d = heat[x, :, :].T  # display Y-Z plane
    heat2d = _robust01(heat2d, 90.0, 99.5)  # emphasise highlights

    fig = plt.figure(figsize=figsize)
    if white_bg:
        fig.patch.set_facecolor("white")
    if base is not None:
        base2d = base[x, :, :].T
        base2d = _robust01(base2d, 2.0, 98.0)
        plt.imshow(base2d, cmap="gray", origin="lower", interpolation=str(base_interp))
        plt.imshow(heat2d, cmap="hot", origin="lower", alpha=float(alpha), interpolation=str(heat_interp))
    else:
        plt.imshow(heat2d, cmap="hot", origin="lower", interpolation=str(heat_interp))
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=int(dpi), bbox_inches="tight", facecolor=("white" if white_bg else None))
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Average CAM NIfTIs and save a mid-sagittal slice PNG.")
    ap.add_argument("--input_root", required=True, type=str, help="Directory to search recursively for CAM NIfTIs.")
    ap.add_argument("--pattern", required=True, type=str, help="Glob pattern (rglob) to match CAM NIfTIs, e.g. '*_gradcam_class1.nii.gz'.")
    ap.add_argument("--output_dir", required=True, type=str, help="Absolute output directory.")
    ap.add_argument("--name", type=str, default="cam_mean", help="Output name prefix.")
    ap.add_argument("--base_nifti", type=str, default=None, help="Optional base anatomical NIfTI for overlay (will be resampled to CAM grid if needed).")
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha.")
    ap.add_argument("--white_bg", action="store_true", help="Save PNG with a white background (recommended for manuscripts).")
    ap.add_argument("--dpi", type=int, default=300, help="PNG DPI (300 recommended for manuscripts).")
    ap.add_argument("--figsize", type=float, nargs=2, default=[6.5, 5.5], metavar=("W", "H"), help="Figure size in inches.")
    ap.add_argument("--base_interp", type=str, default="bicubic", choices=["nearest", "bilinear", "bicubic", "lanczos"], help="Interpolation used for the anatomical underlay.")
    ap.add_argument("--heat_interp", type=str, default="nearest", choices=["nearest", "bilinear", "bicubic"], help="Interpolation used for the heatmap overlay.")
    ap.add_argument("--normalise_each", action="store_true", help="Robust-normalise each subject CAM before averaging (useful if scales differ).")
    ap.add_argument("--title", type=str, default=None, help="Optional figure title.")
    args = ap.parse_args()

    out_dir = _expand(args.output_dir)
    if not os.path.isabs(out_dir):
        raise SystemExit("--output_dir must be an absolute path")
    os.makedirs(out_dir, exist_ok=True)

    input_root = _expand(args.input_root)
    paths = _find_inputs(input_root, args.pattern)
    if not paths:
        raise SystemExit(f"No inputs found under {input_root} matching pattern {args.pattern!r}")

    ref_img, mean_cam = _stack_mean(paths, normalise_each=bool(args.normalise_each))

    out_mean = os.path.join(out_dir, f"{args.name}_mean.nii.gz")
    nib.save(nib.Nifti1Image(mean_cam.astype(np.float32), affine=ref_img.affine), out_mean)

    base_vol = None
    if args.base_nifti:
        base_path = _expand(args.base_nifti)
        base_img = nib.load(base_path)
        if tuple(base_img.shape[:3]) != tuple(ref_img.shape[:3]) or not np.allclose(base_img.affine, ref_img.affine):
            if resample_from_to is None:
                raise SystemExit("Base NIfTI grid differs from CAM grid, but nibabel.resample_from_to is unavailable. Install a newer nibabel or resample base manually.")
            base_img = resample_from_to(base_img, ref_img, order=1)  # trilinear for anatomy
        base_vol = base_img.get_fdata().astype(np.float32)
        if base_vol.ndim == 4:
            base_vol = base_vol.mean(axis=-1)

    title = args.title if args.title else f"{args.name} (n={len(paths)})"
    out_png = os.path.join(out_dir, f"{args.name}_mid_sagittal.png")
    _save_mid_sagittal_png(
        base_vol,
        mean_cam,
        out_png,
        title=title,
        alpha=float(args.alpha),
        white_bg=bool(args.white_bg),
        base_interp=str(args.base_interp),
        heat_interp=str(args.heat_interp),
        dpi=int(args.dpi),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
    )

    print(f"✓ Found {len(paths)} CAMs")
    print(f"✓ Saved mean NIfTI: {out_mean}")
    print(f"✓ Saved PNG: {out_png}")


if __name__ == "__main__":
    main()

