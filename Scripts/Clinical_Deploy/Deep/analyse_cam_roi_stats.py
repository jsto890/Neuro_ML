#!/usr/bin/env python3
"""
ROI-based statistical analysis for 3D Grad-CAM / Grad-CAM++ NIfTIs (MNI2mm).

What this does
--------------
- Scans an explain output root (e.g. ~/.../data/explain/MRI/) for per-subject folders
- Finds CAM NIfTIs inside each subject folder (gradcam + gradcam_plusplus)
- Uses a user-provided atlas NIfTI in the SAME space/shape (MNI 2mm) to compute ROI summaries per subject
- Runs group comparisons across labels (CN/AD/PD) at the ROI level with FDR correction

Outputs
-------
- Per-subject ROI matrices (CSV) for each CAM type
- Per-ROI group summary stats (CSV)
- Per-ROI hypothesis test results (CSV) incl. q-values (Benjamini–Hochberg)

Required inputs
---------------
- labels CSV: subject_id,label (label is an int class index, e.g. 0/1/2)
- atlas NIfTI: integer ROI labels, 0=background

Notes
-----
- This is designed for modest sample sizes (e.g. 30 per class), where ROI-level stats are more robust
  than voxelwise correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Please install nibabel (e.g. pip install nibabel)") from e

try:
    from nibabel.processing import resample_from_to  # type: ignore
except Exception:
    resample_from_to = None  # type: ignore

try:
    from scipy.stats import kruskal, mannwhitneyu  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Please install scipy (e.g. pip install scipy).") from e


def _expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _norm_sid(s: str) -> str:
    s = str(s).strip()
    if s.lower().startswith("sub-"):
        return s[4:]
    return s


def _read_labels_csv(path: str, subject_col: str = "subject_id", label_col: str = "label") -> Dict[str, int]:
    out: Dict[str, int] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"No header detected in labels CSV: {path}")
        if subject_col not in r.fieldnames or label_col not in r.fieldnames:
            raise ValueError(f"CSV must contain columns {subject_col!r} and {label_col!r}. Found: {r.fieldnames}")
        for row in r:
            sid_raw = row.get(subject_col, "")
            lab_raw = row.get(label_col, "")
            if sid_raw is None or lab_raw is None:
                continue
            sid = _norm_sid(sid_raw)
            if sid == "":
                continue
            try:
                lab = int(str(lab_raw).strip())
            except Exception:
                continue
            out[sid] = lab
    return out


def _load_nifti_3d(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data.mean(axis=-1)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI at {path}, got shape {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, img.affine


def _read_roi_name_map(path: str) -> Dict[int, str]:
    """
    Read a simple ROI ID -> name mapping file.

    Supports common FSL HarvardOxford label text files where each non-empty line begins
    with an integer label ID followed by the region name, e.g.:
      1  Frontal Pole
      2  Insular Cortex
    """
    mapping: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Allow comments
            if s.startswith("#") or s.startswith("//"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                rid = int(parts[0])
            except Exception:
                continue
            name = " ".join(parts[1:]).strip()
            if name:
                mapping[int(rid)] = name
    return mapping


def _read_fsl_atlas_xml(path: str) -> Dict[int, str]:
    """
    Parse FSL atlas XML (e.g. HarvardOxford-Cortical.xml) into ROI ID -> name mapping.

    FSL XML format typically contains entries like:
      <label index="1" x=".." y=".." z="..">Frontal Pole</label>
    """
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


def _align_roi_name_map_to_atlas(roi_name_map: Dict[int, str], atlas_labels: Sequence[int]) -> Dict[int, str]:
    """
    FSL atlas XML files sometimes use 0-based label indices while the NIfTI label
    image uses 1-based integers (0 reserved for background). This helper detects
    that mismatch and shifts the mapping keys if it improves coverage.
    """
    if not roi_name_map:
        return roi_name_map
    atlas_set = set(int(x) for x in atlas_labels)
    keys = set(int(k) for k in roi_name_map.keys())
    if not atlas_set:
        return roi_name_map

    def coverage(kset: set[int]) -> int:
        return len(atlas_set.intersection(kset))

    cov0 = coverage(keys)
    keys_p1 = set(k + 1 for k in keys)
    keys_m1 = set(k - 1 for k in keys)
    cov_p1 = coverage(keys_p1)
    cov_m1 = coverage(keys_m1)

    # Prefer +1 shift if XML is 0-based and atlas is 1-based.
    if cov_p1 > cov0 and cov_p1 >= cov_m1:
        return {int(k) + 1: v for k, v in roi_name_map.items()}
    if cov_m1 > cov0 and cov_m1 > cov_p1:
        return {int(k) - 1: v for k, v in roi_name_map.items() if int(k) - 1 >= 0}
    return roi_name_map


def _summarise_mapping_and_check_bilateral(roi_ids: Sequence[int], roi_name_map: Dict[int, str], atlas_i: np.ndarray,
                                          strict: bool = False) -> None:
    """
    Basic sanity checks to catch common ROI name mapping errors (e.g., 0-based vs 1-based shifts).

    - Reports how many ROI IDs have an associated name
    - Checks bilateral symmetry of voxel counts for any ROI names in the form "Left X" / "Right X"
      and warns (or aborts in strict mode) on extreme asymmetry.
    """
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

    # Build voxel counts
    vox_counts: Dict[int, int] = {rid: int(np.sum(atlas_i == int(rid))) for rid in roi_ids}

    # Build left/right pairs from names
    left: Dict[str, int] = {}
    right: Dict[str, int] = {}
    for rid in roi_ids:
        nm = roi_name_map.get(int(rid), f"roi_{int(rid)}")
        if nm.lower().startswith("left "):
            base = nm[5:].strip()
            left[base] = int(rid)
        elif nm.lower().startswith("right "):
            base = nm[6:].strip()
            right[base] = int(rid)

    bad_pairs: List[Tuple[str, float, int, int]] = []
    for base in sorted(set(left.keys()).intersection(right.keys())):
        l_id = left[base]
        r_id = right[base]
        l_n = vox_counts.get(l_id, 0)
        r_n = vox_counts.get(r_id, 0)
        if l_n <= 0 or r_n <= 0:
            continue
        ratio = float(r_n) / float(l_n)
        if ratio < 0.3 or ratio > 3.0:
            bad_pairs.append((base, ratio, l_n, r_n))

    if bad_pairs:
        msg = "[MAPPING] Extreme Left/Right voxel-count asymmetry detected (possible atlas↔name mismatch):\n"
        for base, ratio, l_n, r_n in bad_pairs[:12]:
            msg += f"  - {base}: R/L={ratio:.3f} (L={l_n}, R={r_n})\n"
        print(msg.rstrip())
        if strict:
            raise SystemExit("Aborting due to strict mapping checks. Fix ROI name mapping / atlas resampling before proceeding.")


def _bh_fdr(pvals: Sequence[float]) -> List[float]:
    """
    Benjamini–Hochberg FDR correction.
    Returns q-values in original order (nan preserved).
    """
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    ok = np.isfinite(p)
    if not np.any(ok):
        return [float(x) for x in q]
    p_ok = p[ok]
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    m = float(len(ranked))
    q_ranked = ranked * m / (np.arange(1, len(ranked) + 1, dtype=np.float64))
    # enforce monotonicity
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    # map back
    q_ok = np.empty_like(p_ok)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return [float(x) for x in q]


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    nx, ny = x.size, y.size
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(1.0, (nx + ny - 2))
    if pooled <= 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


@dataclass
class SubjectCam:
    sid: str
    label: int
    cam_path: str


def _find_subject_dirs(explain_root: str) -> List[str]:
    p = Path(explain_root)
    if not p.is_dir():
        raise ValueError(f"Explain root is not a directory: {explain_root}")
    # Expect per-subject folders
    return sorted([str(d) for d in p.iterdir() if d.is_dir()])


def _pick_cam_file(subject_dir: str, sid: str, cam_kind: str) -> Optional[str]:
    """
    cam_kind: 'gradcam' or 'gradcam_plusplus'
    Tries common naming patterns from predict_clinical_deep.py outputs.
    """
    p = Path(subject_dir)
    sid_norm = _norm_sid(sid)

    # Most common: stem includes sub-<ID>_..._zscore_... and file names include _gradcam_classX.nii.gz
    patterns = [
        f"sub-{sid_norm}*_{cam_kind}_class*.nii.gz",
        f"{sid_norm}*_{cam_kind}_class*.nii.gz",
        f"*{sid_norm}*_{cam_kind}_class*.nii.gz",
    ]
    for pat in patterns:
        hits = sorted([str(x) for x in p.glob(pat) if x.is_file()])
        if hits:
            # If multiple classes exist, we keep them all at the caller level by filtering later if desired.
            # Here we return first; caller can use --cam_class to disambiguate.
            return hits[0]
    return None


def _collect_cams(
    explain_root: str,
    labels_map: Dict[str, int],
    cam_kind: str,
    cam_class: Optional[int] = None,
) -> List[SubjectCam]:
    subs: List[SubjectCam] = []
    for d in _find_subject_dirs(explain_root):
        # folder name begins with subject id typically (e.g. I1581414_PD_x)
        base = os.path.basename(d)
        sid_guess = _norm_sid(base.split("_")[0])
        if sid_guess not in labels_map:
            continue
        label = int(labels_map[sid_guess])
        # Find matching CAM file(s)
        if cam_class is None:
            cam_path = _pick_cam_file(d, sid_guess, cam_kind)
            if cam_path:
                subs.append(SubjectCam(sid=sid_guess, label=label, cam_path=cam_path))
        else:
            p = Path(d)
            sid_norm = _norm_sid(sid_guess)
            pats = [
                f"sub-{sid_norm}*_{cam_kind}_class{int(cam_class)}.nii.gz",
                f"{sid_norm}*_{cam_kind}_class{int(cam_class)}.nii.gz",
                f"*{sid_norm}*_{cam_kind}_class{int(cam_class)}.nii.gz",
            ]
            found = None
            for pat in pats:
                hits = sorted([str(x) for x in p.glob(pat) if x.is_file()])
                if hits:
                    found = hits[0]
                    break
            if found:
                subs.append(SubjectCam(sid=sid_guess, label=label, cam_path=found))
    return subs


def _roi_ids_from_atlas(atlas: np.ndarray) -> List[int]:
    ids = sorted(set(int(x) for x in np.unique(atlas) if int(x) != 0))
    return ids


def _roi_summary(cam: np.ndarray, atlas: np.ndarray, roi_id: int) -> float:
    m = atlas == int(roi_id)
    if not np.any(m):
        return float("nan")
    vals = cam[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="ROI statistics for Grad-CAM/Grad-CAM++ NIfTIs in MNI2mm space.")
    ap.add_argument("--explain_root", required=True, type=str, help="Absolute path to explain/MRI root containing per-subject folders.")
    ap.add_argument("--labels_csv", required=True, type=str, help="Absolute path to labels CSV (subject_id,label).")
    ap.add_argument("--atlas", required=True, type=str, help="Absolute path to atlas NIfTI (integer labels; 0=background) in MNI2mm.")
    ap.add_argument("--atlas_labels", type=str, default=None, help="Optional path to ROI label names text file (maps atlas integer IDs -> names).")
    ap.add_argument("--atlas_xml", type=str, default=None, help="Optional FSL atlas XML file for ROI names (e.g. HarvardOxford-Cortical.xml).")
    ap.add_argument("--auto_resample_atlas", action="store_true", help="If set, resample atlas labels to CAM grid when shapes differ (nearest-neighbour).")
    ap.add_argument("--output_dir", required=True, type=str, help="Absolute output directory.")

    ap.add_argument("--subject_col", type=str, default="subject_id")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--label_names", type=str, nargs="+", default=["CN", "AD", "PD"], help="Optional names for labels 0..K-1 (used in outputs).")
    ap.add_argument("--cam_kind", choices=["gradcam", "gradcam_plusplus"], default="gradcam", help="Which attribution type to analyse.")
    ap.add_argument("--cam_class", type=int, default=None, help="If set, use only CAM files for this class index (e.g. 0/1/2).")
    ap.add_argument("--min_per_group", type=int, default=5, help="Minimum subjects per group required to run group tests.")
    ap.add_argument("--strict_mapping", action="store_true", help="Abort if ROI mapping sanity checks detect extreme Left/Right voxel-count asymmetry.")

    args = ap.parse_args()

    explain_root = _expand(args.explain_root)
    labels_csv = _expand(args.labels_csv)
    atlas_path = _expand(args.atlas)
    out_dir = Path(_expand(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_map = _read_labels_csv(labels_csv, args.subject_col, args.label_col)
    cams = _collect_cams(explain_root, labels_map, cam_kind=str(args.cam_kind), cam_class=args.cam_class)
    if not cams:
        raise SystemExit("No CAM files found. Check --explain_root, labels CSV IDs, and filename patterns.")

    # Load first CAM to validate shape
    cam_img = nib.load(cams[0].cam_path)
    cam_data = cam_img.get_fdata().astype(np.float32)
    if cam_data.ndim == 4:
        cam_data = cam_data.mean(axis=-1)
    if cam_data.ndim != 3:
        raise SystemExit(f"Expected 3D CAM NIfTI at {cams[0].cam_path}, got shape {cam_data.shape}")

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(np.float32)
    if atlas_data.ndim == 4:
        atlas_data = atlas_data.mean(axis=-1)
    if atlas_data.ndim != 3:
        raise SystemExit(f"Expected 3D atlas NIfTI at {atlas_path}, got shape {atlas_data.shape}")

    resampled_atlas_path = None
    if tuple(atlas_data.shape) != tuple(cam_data.shape) or (not np.allclose(atlas_img.affine, cam_img.affine)):
        if not bool(getattr(args, "auto_resample_atlas", False)):
            raise SystemExit(
                f"Shape mismatch: CAM {cam_data.shape} vs atlas {atlas_data.shape}. "
                f"Re-run with --auto_resample_atlas to resample atlas labels onto the CAM grid (nearest-neighbour)."
            )
        if resample_from_to is None:
            raise SystemExit("nibabel.processing.resample_from_to not available; install a newer nibabel or resample the atlas with FSL (flirt -interp nearestneighbour).")
        # Nearest-neighbour resample atlas label image onto CAM grid
        atlas_res = resample_from_to(atlas_img, cam_img, order=0)  # type: ignore
        resampled_atlas_path = str(out_dir / f"atlas_resampled_to_cam_{Path(atlas_path).name}")
        nib.save(atlas_res, resampled_atlas_path)
        atlas_img = atlas_res
        atlas_data = atlas_img.get_fdata().astype(np.float32)

    atlas_i = np.round(atlas_data).astype(np.int32)
    roi_ids = _roi_ids_from_atlas(atlas_i)
    if not roi_ids:
        raise SystemExit("No ROI IDs found in atlas (non-zero labels).")

    roi_name_map: Dict[int, str] = {}
    if getattr(args, "atlas_xml", None):
        roi_name_map = _read_fsl_atlas_xml(_expand(str(getattr(args, "atlas_xml"))))
    elif getattr(args, "atlas_labels", None):
        roi_name_map = _read_roi_name_map(_expand(str(getattr(args, "atlas_labels"))))
    # Auto-align potential 0-based XML indices to atlas label values
    roi_name_map = _align_roi_name_map_to_atlas(roi_name_map, roi_ids)
    _summarise_mapping_and_check_bilateral(roi_ids=roi_ids, roi_name_map=roi_name_map, atlas_i=atlas_i, strict=bool(getattr(args, "strict_mapping", False)))

    # Build per-subject ROI matrix
    rows: List[Dict[str, object]] = []
    roi_cols = [f"roi_{rid}" for rid in roi_ids]
    roi_name_cols = [f"roi_name_{rid}" for rid in roi_ids]
    for sc in cams:
        cam, _ = _load_nifti_3d(sc.cam_path)
        if cam.shape != atlas_i.shape:
            raise SystemExit(f"Shape mismatch for {sc.sid}: CAM {cam.shape} vs atlas {atlas_i.shape} (did you resample atlas to CAM grid?)")
        rec: Dict[str, object] = {"subject_id": sc.sid, "label": int(sc.label), "cam_path": sc.cam_path}
        # Repeat ROI names per row for convenience (keeps a single CSV self-contained).
        # Consumers can ignore these columns if undesired.
        for rid, ncol in zip(roi_ids, roi_name_cols):
            rec[ncol] = roi_name_map.get(int(rid), f"roi_{int(rid)}")
        for rid, col in zip(roi_ids, roi_cols):
            rec[col] = _roi_summary(cam, atlas_i, rid)
        rows.append(rec)

    # Write per-subject matrix
    matrix_fname = f"roi_matrix_{args.cam_kind}" + (f"_class{args.cam_class}" if args.cam_class is not None else "") + ".csv"
    matrix_path = out_dir / matrix_fname
    with open(matrix_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "label", "cam_path"] + roi_name_cols + roi_cols)
        for r in rows:
            w.writerow([r["subject_id"], r["label"], r["cam_path"]] + [r[c] for c in roi_name_cols] + [r[c] for c in roi_cols])

    # Group summaries + tests
    labels = sorted(set(int(r["label"]) for r in rows))
    groups: Dict[int, List[Dict[str, object]]] = {lab: [r for r in rows if int(r["label"]) == lab] for lab in labels}

    def lab_name(lab: int) -> str:
        return args.label_names[lab] if 0 <= lab < len(args.label_names) else str(lab)

    # Per-ROI: Kruskal across all groups + pairwise MWU
    test_rows: List[Dict[str, object]] = []
    p_omnibus: List[float] = []

    for rid, col in zip(roi_ids, roi_cols):
        vals_by_group = {lab: np.array([float(r[col]) for r in groups[lab]], dtype=np.float64) for lab in labels}
        # drop nans
        vals_by_group = {lab: v[np.isfinite(v)] for lab, v in vals_by_group.items()}
        ns = {lab: int(v.size) for lab, v in vals_by_group.items()}

        rec: Dict[str, object] = {
            "roi_id": int(rid),
            "roi_name": roi_name_map.get(int(rid), f"roi_{int(rid)}"),
            "n_total": int(sum(ns.values())),
            **{f"n_{lab_name(lab)}": int(ns[lab]) for lab in labels},
            **{f"mean_{lab_name(lab)}": float(np.mean(vals_by_group[lab])) if ns[lab] > 0 else float("nan") for lab in labels},
            **{f"std_{lab_name(lab)}": float(np.std(vals_by_group[lab], ddof=1)) if ns[lab] > 1 else float("nan") for lab in labels},
        }

        # omnibus
        ok_groups = [vals_by_group[lab] for lab in labels if ns[lab] >= int(args.min_per_group)]
        if len(ok_groups) >= 2:
            try:
                stat, p = kruskal(*ok_groups)
                rec["kruskal_H"] = float(stat)
                rec["kruskal_p"] = float(p)
            except Exception:
                rec["kruskal_H"] = float("nan")
                rec["kruskal_p"] = float("nan")
        else:
            rec["kruskal_H"] = float("nan")
            rec["kruskal_p"] = float("nan")

        p_omnibus.append(float(rec["kruskal_p"]) if np.isfinite(rec["kruskal_p"]) else float("nan"))

        # pairwise tests
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a = labels[i]
                b = labels[j]
                va = vals_by_group[a]
                vb = vals_by_group[b]
                key = f"{lab_name(a)}_vs_{lab_name(b)}"
                if va.size >= int(args.min_per_group) and vb.size >= int(args.min_per_group):
                    try:
                        u, pp = mannwhitneyu(va, vb, alternative="two-sided")
                        rec[f"mwu_U_{key}"] = float(u)
                        rec[f"mwu_p_{key}"] = float(pp)
                        rec[f"cohens_d_{key}"] = _cohens_d(va, vb)
                    except Exception:
                        rec[f"mwu_U_{key}"] = float("nan")
                        rec[f"mwu_p_{key}"] = float("nan")
                        rec[f"cohens_d_{key}"] = float("nan")
                else:
                    rec[f"mwu_U_{key}"] = float("nan")
                    rec[f"mwu_p_{key}"] = float("nan")
                    rec[f"cohens_d_{key}"] = float("nan")

        test_rows.append(rec)

    # FDR for omnibus p-values
    q_omnibus = _bh_fdr(p_omnibus)
    for rec, q in zip(test_rows, q_omnibus):
        rec["kruskal_q_fdr"] = float(q)

    # Write test table
    # Determine columns deterministically
    base_cols = [
        "roi_id",
        "roi_name",
        "n_total",
        "kruskal_H",
        "kruskal_p",
        "kruskal_q_fdr",
    ]
    group_cols = []
    for lab in labels:
        group_cols.extend([f"n_{lab_name(lab)}", f"mean_{lab_name(lab)}", f"std_{lab_name(lab)}"])
    pair_cols = sorted([k for k in test_rows[0].keys() if k.startswith("mwu_") or k.startswith("cohens_d_")])
    cols = base_cols + group_cols + pair_cols

    tests_fname = f"roi_tests_{args.cam_kind}" + (f"_class{args.cam_class}" if args.cam_class is not None else "") + ".csv"
    out_tests = out_dir / tests_fname
    with open(out_tests, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in test_rows:
            w.writerow([rec.get(c, "") for c in cols])

    # Save small metadata JSON for reproducibility
    meta = {
        "explain_root": explain_root,
        "labels_csv": labels_csv,
        "atlas": atlas_path,
        "atlas_labels": _expand(str(getattr(args, "atlas_labels"))) if getattr(args, "atlas_labels", None) else None,
        "atlas_xml": _expand(str(getattr(args, "atlas_xml"))) if getattr(args, "atlas_xml", None) else None,
        "auto_resample_atlas": bool(getattr(args, "auto_resample_atlas", False)),
        "resampled_atlas_path": resampled_atlas_path,
        "cam_kind": args.cam_kind,
        "cam_class": args.cam_class,
        "n_subjects": len(rows),
        "labels_present": labels,
        "roi_count": len(roi_ids),
        "outputs": {
            "roi_matrix_csv": str(matrix_path),
            "roi_tests_csv": str(out_tests),
        },
    }
    with open(out_dir / "roi_analysis_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f" Wrote ROI matrix: {matrix_path}")
    print(f" Wrote ROI tests:  {out_tests}")


if __name__ == "__main__":
    main()

