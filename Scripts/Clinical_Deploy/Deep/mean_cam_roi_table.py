#!/usr/bin/env python3
"""
Summarise a mean Grad-CAM volume at the ROI level (MNI space).

Use-case
--------
You have already created a group-mean CAM NIfTI (e.g. from 30 AD subjects) and want a
table of "top ROIs by mean attribution" that directly corresponds to the visual map.

Inputs
------
- --mean_cam: path to a mean CAM NIfTI
- --atlas: integer-labelled atlas NIfTI (FSL HarvardOxford etc.)
- --atlas_xml: optional FSL atlas XML to map ROI IDs -> names

The atlas will be resampled to the CAM grid (nearest-neighbour) if shapes differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Please install nibabel (e.g. pip install nibabel)") from e

try:
    from nibabel.processing import resample_from_to  # type: ignore
except Exception:
    resample_from_to = None  # type: ignore


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


def _read_fsl_atlas_xml(path: str) -> Dict[int, str]:
    import xml.etree.ElementTree as ET

    mapping: Dict[int, str] = {}
    tree = ET.parse(path)
    root = tree.getroot()
    for lab in root.iter("label"):
        idx = lab.attrib.get("index", None)
        if idx is None:
            continue
        try:
            rid = int(idx)
        except Exception:
            continue
        name = (lab.text or "").strip()
        if name:
            mapping[rid] = name
    return mapping


def _align_roi_name_map_to_atlas(roi_name_map: Dict[int, str], atlas_labels: List[int]) -> Dict[int, str]:
    """
    Detect and correct 0-based vs 1-based label index mismatches between FSL XML and label NIfTI.
    """
    if not roi_name_map:
        return roi_name_map
    atlas_set = set(int(x) for x in atlas_labels)
    keys = set(int(k) for k in roi_name_map.keys())
    if not atlas_set:
        return roi_name_map

    cov0 = len(atlas_set.intersection(keys))
    cov_p1 = len(atlas_set.intersection(set(k + 1 for k in keys)))
    cov_m1 = len(atlas_set.intersection(set(k - 1 for k in keys)))

    if cov_p1 > cov0 and cov_p1 >= cov_m1:
        return {int(k) + 1: v for k, v in roi_name_map.items()}
    if cov_m1 > cov0 and cov_m1 > cov_p1:
        return {int(k) - 1: v for k, v in roi_name_map.items() if int(k) - 1 >= 0}
    return roi_name_map


def _summarise_mapping_and_check_bilateral(roi_ids: List[int], roi_name_map: Dict[int, str], atl_i: np.ndarray,
                                          strict: bool = False) -> None:
    roi_ids = [int(x) for x in roi_ids]
    named = 0
    unnamed: List[int] = []
    for rid in roi_ids:
        nm = roi_name_map.get(int(rid), f"roi_{int(rid)}")
        if nm.startswith("roi_"):
            unnamed.append(int(rid))
        else:
            named += 1
    print(f"[MAPPING] ROI IDs present: {len(roi_ids)} | named: {named} | unnamed: {len(unnamed)}")
    if unnamed:
        print(f"[MAPPING] Unnamed ROI IDs (first 10): {unnamed[:10]}")

    vox_counts: Dict[int, int] = {rid: int(np.sum(atl_i == int(rid))) for rid in roi_ids}
    left: Dict[str, int] = {}
    right: Dict[str, int] = {}
    for rid in roi_ids:
        nm = roi_name_map.get(int(rid), f"roi_{int(rid)}")
        if nm.lower().startswith("left "):
            left[nm[5:].strip()] = int(rid)
        elif nm.lower().startswith("right "):
            right[nm[6:].strip()] = int(rid)
    bad: List[Tuple[str, float, int, int]] = []
    for base in sorted(set(left.keys()).intersection(right.keys())):
        l_id = left[base]; r_id = right[base]
        l_n = vox_counts.get(l_id, 0); r_n = vox_counts.get(r_id, 0)
        if l_n <= 0 or r_n <= 0:
            continue
        ratio = float(r_n) / float(l_n)
        if ratio < 0.3 or ratio > 3.0:
            bad.append((base, ratio, l_n, r_n))
    if bad:
        msg = "[MAPPING] Extreme Left/Right voxel-count asymmetry detected (possible atlas↔name mismatch):\n"
        for base, ratio, l_n, r_n in bad[:12]:
            msg += f"  - {base}: R/L={ratio:.3f} (L={l_n}, R={r_n})\n"
        print(msg.rstrip())
        if strict:
            raise SystemExit("Aborting due to strict mapping checks. Fix ROI name mapping / atlas resampling before proceeding.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ROI means from a mean CAM NIfTI.")
    ap.add_argument("--mean_cam", required=True, type=str, help="Absolute path to mean CAM NIfTI.")
    ap.add_argument("--atlas", required=True, type=str, help="Absolute path to atlas label NIfTI (integer labels, 0=background).")
    ap.add_argument("--atlas_xml", type=str, default=None, help="Optional FSL atlas XML to map ROI IDs to names.")
    ap.add_argument("--output_csv", required=True, type=str, help="Absolute output CSV path.")
    ap.add_argument("--save_resampled_atlas", action="store_true", help="If resampling occurs, save the resampled atlas next to the CSV.")
    ap.add_argument("--strict_mapping", action="store_true", help="Abort if mapping sanity checks detect extreme Left/Right voxel-count asymmetry.")
    args = ap.parse_args()

    mean_cam_path = _expand(args.mean_cam)
    atlas_path = _expand(args.atlas)
    out_csv = _expand(args.output_csv)
    if not os.path.isabs(mean_cam_path) or not os.path.isabs(atlas_path) or not os.path.isabs(out_csv):
        raise SystemExit("Please use absolute paths for --mean_cam, --atlas, and --output_csv")

    cam_img, cam = _load_3d(mean_cam_path)
    atl_img, atl = _load_3d(atlas_path)

    resampled_atlas_path: Optional[str] = None
    if atl.shape != cam.shape or not np.allclose(atl_img.affine, cam_img.affine):
        if resample_from_to is None:
            raise SystemExit("Atlas grid differs from CAM grid, but nibabel.resample_from_to is unavailable. Install a newer nibabel or resample with FSL (nearest-neighbour).")
        atl_img = resample_from_to(atl_img, cam_img, order=0)  # nearest for labels
        atl = atl_img.get_fdata().astype(np.float32)
        if bool(args.save_resampled_atlas):
            out_dir = str(Path(out_csv).parent)
            resampled_atlas_path = os.path.join(out_dir, f"atlas_resampled_to_cam_{Path(atlas_path).name}")
            nib.save(atl_img, resampled_atlas_path)

    atl_i = np.round(atl).astype(np.int32)
    roi_ids = sorted(set(int(x) for x in np.unique(atl_i) if int(x) != 0))
    if not roi_ids:
        raise SystemExit("No non-zero ROI IDs found in atlas.")

    name_map: Dict[int, str] = {}
    if args.atlas_xml:
        name_map = _read_fsl_atlas_xml(_expand(args.atlas_xml))
        name_map = _align_roi_name_map_to_atlas(name_map, roi_ids)
    _summarise_mapping_and_check_bilateral(roi_ids=roi_ids, roi_name_map=name_map, atl_i=atl_i, strict=bool(getattr(args, "strict_mapping", False)))

    rows: List[Dict[str, object]] = []
    for rid in roi_ids:
        m = atl_i == int(rid)
        vals = cam[m]
        vals = vals[np.isfinite(vals)]
        mean_val = float(np.mean(vals)) if vals.size else float("nan")
        rows.append(
            {
                "roi_id": int(rid),
                "roi_name": name_map.get(int(rid), f"roi_{int(rid)}"),
                "n_vox": int(m.sum()),
                "mean_cam": mean_val,
            }
        )

    # Sort descending by mean_cam
    rows_sorted = sorted(rows, key=lambda r: (-(r["mean_cam"] if np.isfinite(r["mean_cam"]) else -np.inf)))

    Path(Path(out_csv).parent).mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_id", "roi_name", "n_vox", "mean_cam"])
        for r in rows_sorted:
            w.writerow([r["roi_id"], r["roi_name"], r["n_vox"], r["mean_cam"]])

    meta = {
        "mean_cam": mean_cam_path,
        "atlas": atlas_path,
        "atlas_xml": _expand(args.atlas_xml) if args.atlas_xml else None,
        "cam_shape": list(cam.shape),
        "atlas_shape_after": list(atl_i.shape),
        "resampled_atlas_path": resampled_atlas_path,
        "output_csv": out_csv,
    }
    with open(str(Path(out_csv).with_suffix(".meta.json")), "w") as f:
        json.dump(meta, f, indent=2)

    print(f" Wrote ROI table: {out_csv}")


if __name__ == "__main__":
    main()

