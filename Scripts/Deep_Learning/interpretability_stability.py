#!/usr/bin/env python3
"""
Inter-fold interpretability stability for 3D Grad-CAM maps.

This script is intentionally simple and journal-friendly:
- For each fold directory, load all Grad-CAM NIfTIs (e.g. *_gradcam.nii.gz)
- Compute a fold-mean Grad-CAM map
- Quantify stability across folds via:
    1) Pearson correlation of fold-mean maps (within an optional brain mask)
    2) Dice overlap of the top X% voxels of fold-mean maps (within an optional brain mask)

Outputs:
- fold mean maps (NIfTI)
- pairwise correlation + Dice matrices (CSV)
- summary JSON

Note:
- This assumes inputs are already registered to the same space (e.g., MNI) and shape.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def _is_under_path(path: str, parent: str) -> bool:
    try:
        path_abs = os.path.abspath(path)
        parent_abs = os.path.abspath(parent)
        common = os.path.commonpath([path_abs, parent_abs])
        return common == parent_abs
    except Exception:
        return False


def _load_nifti_3d(path: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data.mean(axis=-1)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D (or 4D) NIfTI at {path}, got shape {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return img, data


def _find_maps(fold_dir: str, pattern: str) -> List[str]:
    p = Path(fold_dir)
    return sorted([str(x) for x in p.glob(pattern) if x.is_file()])


def _stack_and_mean(map_paths: List[str]) -> Tuple[nib.Nifti1Image, np.ndarray]:
    if not map_paths:
        raise ValueError("No maps provided")
    first_img, first_data = _load_nifti_3d(map_paths[0])
    stack = np.zeros((len(map_paths),) + first_data.shape, dtype=np.float32)
    stack[0] = first_data

    for i, p in enumerate(map_paths[1:], start=1):
        _, d = _load_nifti_3d(p)
        if d.shape != first_data.shape:
            raise ValueError(f"Shape mismatch: {p} has {d.shape} but expected {first_data.shape}")
        stack[i] = d

    mean_map = stack.mean(axis=0)
    return first_img, mean_map


def _flatten_in_mask(arr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return arr.reshape(-1)
    return arr[mask]


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size != b.size:
        raise ValueError("Vector size mismatch for correlation")
    if a.size == 0:
        return float("nan")
    a_std = np.std(a)
    b_std = np.std(b)
    if a_std < 1e-12 or b_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _dice(a_bin: np.ndarray, b_bin: np.ndarray) -> float:
    a_bin = a_bin.astype(bool)
    b_bin = b_bin.astype(bool)
    inter = np.logical_and(a_bin, b_bin).sum()
    denom = a_bin.sum() + b_bin.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def _top_percent_mask(arr: np.ndarray, mask: Optional[np.ndarray], top_percent: float) -> np.ndarray:
    if top_percent <= 0 or top_percent >= 100:
        raise ValueError("--top_percent must be in (0, 100)")
    if mask is None:
        vals = arr.reshape(-1)
        valid = np.isfinite(vals)
        vals = vals[valid]
        thr = np.percentile(vals, 100.0 - top_percent) if vals.size else float("inf")
        return arr >= thr
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    thr = np.percentile(vals, 100.0 - top_percent) if vals.size else float("inf")
    out = np.zeros(arr.shape, dtype=bool)
    out[mask] = arr[mask] >= thr
    return out


def _write_matrix_csv(path: str, labels: List[str], mat: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, row_label in enumerate(labels):
            w.writerow([row_label] + ["" if np.isnan(x) else f"{x:.6f}" for x in mat[i]])


def _plot_heatmap(path: str, labels: List[str], mat: np.ndarray, title: str, vmin: Optional[float], vmax: Optional[float]) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@dataclass
class FoldSummary:
    fold_label: str
    fold_dir: str
    n_maps: int
    mean_map_path: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify inter-fold stability of 3D Grad-CAM maps.")
    parser.add_argument(
        "--fold_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Absolute paths to per-fold directories containing Grad-CAM NIfTIs.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_gradcam.nii.gz",
        help="Glob pattern to match Grad-CAM NIfTIs within each fold directory.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional absolute path to a brain mask NIfTI (same space/shape as maps).",
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=5.0,
        help="Top percent of voxels used for Dice overlap (on fold-mean maps).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Absolute output directory (recommended outside the repo).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="gradcam_stability",
        help="Prefix for output files.",
    )
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        raise SystemExit("--output_dir must be an absolute path")
    for d in args.fold_dirs:
        if not os.path.isabs(d):
            raise SystemExit(f"Fold dir must be an absolute path: {d}")
    if args.mask is not None and not os.path.isabs(args.mask):
        raise SystemExit("--mask must be an absolute path if provided")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warn if output is under the repo root (users often prefer outputs elsewhere).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _is_under_path(str(out_dir), repo_root):
        print(f"Warning: output_dir is under repo root ({repo_root}). Prefer saving outside this folder.")

    mask_bool: Optional[np.ndarray] = None
    if args.mask is not None:
        _, mask_data = _load_nifti_3d(args.mask)
        mask_bool = mask_data > 0

    fold_labels: List[str] = []
    fold_mean_maps: List[np.ndarray] = []
    fold_summaries: List[FoldSummary] = []

    for i, fold_dir in enumerate(args.fold_dirs, start=1):
        fold_label = f"F{i}"
        fold_labels.append(fold_label)
        paths = _find_maps(fold_dir, args.pattern)
        if not paths:
            raise ValueError(f"No maps found in {fold_dir} matching pattern {args.pattern!r}")

        img, mean_map = _stack_and_mean(paths)
        fold_mean_maps.append(mean_map)

        mean_path = out_dir / f"{args.name}_{fold_label}_mean.nii.gz"
        nib.save(nib.Nifti1Image(mean_map.astype(np.float32), affine=img.affine), str(mean_path))
        fold_summaries.append(
            FoldSummary(
                fold_label=fold_label,
                fold_dir=fold_dir,
                n_maps=len(paths),
                mean_map_path=str(mean_path),
            )
        )
        print(f"[{fold_label}] {len(paths)} maps -> mean saved: {mean_path}")

    # Compute correlation and Dice matrices across fold means.
    n = len(fold_mean_maps)
    corr = np.full((n, n), np.nan, dtype=np.float64)
    dice = np.full((n, n), np.nan, dtype=np.float64)

    flat_means = [_flatten_in_mask(m, mask_bool) for m in fold_mean_maps]
    top_masks = [_top_percent_mask(m, mask_bool, args.top_percent) for m in fold_mean_maps]

    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
                dice[i, j] = 1.0
                continue
            corr[i, j] = _pearson_corr(flat_means[i], flat_means[j])
            dice[i, j] = _dice(top_masks[i], top_masks[j])

    # Summaries over upper triangle (excluding diagonal)
    upper = np.triu_indices(n, k=1)
    corr_vals = corr[upper]
    dice_vals = dice[upper]
    corr_mean = float(np.nanmean(corr_vals)) if np.any(~np.isnan(corr_vals)) else float("nan")
    corr_std = float(np.nanstd(corr_vals)) if np.any(~np.isnan(corr_vals)) else float("nan")
    dice_mean = float(np.nanmean(dice_vals)) if np.any(~np.isnan(dice_vals)) else float("nan")
    dice_std = float(np.nanstd(dice_vals)) if np.any(~np.isnan(dice_vals)) else float("nan")

    corr_csv = out_dir / f"{args.name}_pairwise_corr.csv"
    dice_csv = out_dir / f"{args.name}_pairwise_dice_top{args.top_percent:g}pct.csv"
    _write_matrix_csv(str(corr_csv), fold_labels, corr)
    _write_matrix_csv(str(dice_csv), fold_labels, dice)

    _plot_heatmap(
        str(out_dir / f"{args.name}_pairwise_corr.png"),
        fold_labels,
        corr,
        title="Inter-fold mean Grad-CAM correlation",
        vmin=-1.0,
        vmax=1.0,
    )
    _plot_heatmap(
        str(out_dir / f"{args.name}_pairwise_dice.png"),
        fold_labels,
        dice,
        title=f"Inter-fold mean Grad-CAM Dice (top {args.top_percent:g}%)",
        vmin=0.0,
        vmax=1.0,
    )

    summary: Dict[str, object] = {
        "name": args.name,
        "n_folds": n,
        "pattern": args.pattern,
        "mask": args.mask,
        "top_percent": float(args.top_percent),
        "folds": [fs.__dict__ for fs in fold_summaries],
        "pairwise": {
            "corr_mean": corr_mean,
            "corr_std": corr_std,
            "dice_mean": dice_mean,
            "dice_std": dice_std,
        },
        "outputs": {
            "corr_csv": str(corr_csv),
            "dice_csv": str(dice_csv),
        },
    }

    summary_path = out_dir / f"{args.name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")
    print(f"Correlation (mean±sd): {corr_mean:.3f} ± {corr_std:.3f}")
    print(f"Dice top {args.top_percent:g}% (mean±sd): {dice_mean:.3f} ± {dice_std:.3f}")


if __name__ == "__main__":
    main()

