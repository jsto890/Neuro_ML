#!/usr/bin/env python3
"""
Batch Grad-CAM generation for MRI (CN/AD/PD)
===========================================

Purpose
-------
Select N subjects per class from a labels CSV and run:
  Scripts/Clinical_Deploy/Deep/predict_clinical_deep.py
to generate per-subject outputs (Grad-CAM + Grad-CAM++).

Why this script exists
----------------------
The clinical predictor operates on *one* NIfTI at a time. For cohort-level
interpretability analyses you typically want a reproducible, stratified batch
of subjects (e.g. 20 per disease) with consistent settings.

Notes
-----
- This script does not require pandas.
- It supports subject IDs with or without the 'sub-' prefix.
- It is careful about output directories and can skip existing runs.
"""

import argparse
import csv
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _norm_sid(s: str) -> str:
    s = str(s).strip()
    if s.lower().startswith("sub-"):
        return s[4:]
    return s


def _read_labels_csv(path: str, subject_col: str, label_col: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
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
            out.append((sid, lab))
    return out


def _image_path_for_sid(preprocessed_mri_root: str, sid: str) -> str:
    """
    Construct the expected smriprep path:
      <root>/smriprep/sub-<ID>/anat/sub-<ID>_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz
    """
    sid2 = _norm_sid(sid)
    return os.path.join(
        preprocessed_mri_root,
        "smriprep",
        f"sub-{sid2}",
        "anat",
        f"sub-{sid2}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz",
    )


def _safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _already_done(out_dir: str, sid: str) -> bool:
    """
    Consider the run done if both CAM NIfTIs exist for the predicted (or all) classes.
    We check for any class file presence to avoid overfitting to class naming.
    """
    p = Path(out_dir)
    # At minimum: one gradcam and one gradcam++ nifti should exist.
    has_gc = any(p.glob(f"{sid}*_gradcam_class*.nii.gz"))
    has_gcpp = any(p.glob(f"{sid}*_gradcam_plusplus_class*.nii.gz"))
    return bool(has_gc and has_gcpp)


@dataclass
class RunSpec:
    sid: str
    label: int
    label_name: str
    image_path: str
    output_dir: str


def _build_predict_cmd(predict_script: str, spec: RunSpec, passthrough: Sequence[str]) -> List[str]:
    cmd = [
        "python3",
        predict_script,
        "--image",
        spec.image_path,
        "--output-dir",
        spec.output_dir,
        # ensure we actually produce both nifti types for cohort analyses
        "--run",
        "gradcam",
        "gradcam_plusplus",
        # cohort analysis: generate maps for the TRUE label class (stable across subjects)
        "--cam-classes",
        str(int(spec.label)),
    ]
    cmd.extend(list(passthrough))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate Grad-CAM + Grad-CAM++ NIfTIs for MRI cohorts.")
    parser.add_argument("--labels_csv", required=True, type=str, help="Absolute path to labels CSV (must include subject_id,label columns by default).")
    parser.add_argument("--subject_col", type=str, default="subject_id", help="Subject ID column name in labels CSV.")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name in labels CSV.")

    parser.add_argument("--preprocessed_mri_root", required=True, type=str, help="Absolute path to preprocessed/MRI root (contains smriprep/).")
    parser.add_argument("--output_root", required=True, type=str, help="Absolute path to explain/MRI output root.")
    parser.add_argument("--n_per_class", type=int, default=20, help="Subjects per class to run (per label).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling.")
    parser.add_argument("--labels", type=int, nargs="+", default=[0, 1, 2], help="Label integers to run (default: 0 1 2).")
    parser.add_argument("--label_names", type=str, nargs="+", default=["CN", "AD", "PD"], help="Names for labels in same order as --labels.")
    parser.add_argument("--run_tag", type=str, default="mri_batch", help="Tag added to output folder names.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip subjects whose outputs already exist.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands but do not execute.")
    parser.add_argument("--max_workers", type=int, default=1, help="Parallel workers (1 = sequential).")

    # Everything after '--' is passed to predict_clinical_deep.py unchanged.
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Arguments after '--' are passed to predict_clinical_deep.py.")
    args = parser.parse_args()

    labels_csv = _expand(args.labels_csv)
    pre_root = _expand(args.preprocessed_mri_root)
    out_root = _expand(args.output_root)

    if not os.path.isabs(labels_csv) or not os.path.isabs(pre_root) or not os.path.isabs(out_root):
        raise SystemExit("Please provide absolute paths for --labels_csv, --preprocessed_mri_root, and --output_root")

    if len(args.labels) != len(args.label_names):
        raise SystemExit("--labels and --label_names must have the same length")

    label_name_map: Dict[int, str] = {int(l): str(n) for l, n in zip(args.labels, args.label_names)}

    rows = _read_labels_csv(labels_csv, args.subject_col, args.label_col)
    if not rows:
        raise SystemExit("No valid rows parsed from labels CSV.")

    # Group subjects by label
    by_label: Dict[int, List[str]] = {int(l): [] for l in args.labels}
    for sid, lab in rows:
        if int(lab) in by_label:
            by_label[int(lab)].append(_norm_sid(sid))

    rng = random.Random(int(args.seed))

    # Build run list
    runs: List[RunSpec] = []
    for lab in args.labels:
        lab = int(lab)
        sids = sorted(set(by_label.get(lab, [])))
        if len(sids) == 0:
            print(f"Warning: no subjects found for label {lab}", file=sys.stderr)
            continue
        rng.shuffle(sids)
        chosen = sids[: max(0, int(args.n_per_class))]
        name = label_name_map.get(lab, str(lab))
        for sid in chosen:
            img_path = _image_path_for_sid(pre_root, sid)
            sid_full = Path(img_path).stem.replace(".nii", "").replace(".gz", "")
            # Mirror your current layout: explain/MRI/<subject>_<label>_<tag>/
            out_dir = os.path.join(out_root, f"{_norm_sid(sid)}_{name}_{args.run_tag}")
            runs.append(RunSpec(sid=sid_full, label=lab, label_name=name, image_path=img_path, output_dir=out_dir))

    if not runs:
        raise SystemExit("No runs selected (check labels and n_per_class).")

    # Resolve predict script path relative to this file (works on cluster checkouts)
    this_dir = Path(__file__).resolve().parent
    predict_script = str((this_dir / "predict_clinical_deep.py").resolve())
    if not os.path.isfile(predict_script):
        raise SystemExit(f"Could not find predict script at: {predict_script}")

    # Clean passthrough: drop leading '--' if present
    passthrough = list(args.passthrough)
    if len(passthrough) > 0 and passthrough[0] == "--":
        passthrough = passthrough[1:]

    _safe_mkdir(out_root)

    # Optionally parallelise (simple worker pool)
    max_workers = max(1, int(args.max_workers))
    if max_workers != 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def run_one(spec: RunSpec) -> Tuple[str, int]:
            if not os.path.isfile(spec.image_path):
                return (spec.output_dir, 2)
            _safe_mkdir(spec.output_dir)
            if args.skip_existing and _already_done(spec.output_dir, spec.sid):
                return (spec.output_dir, 0)
            cmd = _build_predict_cmd(predict_script, spec, passthrough)
            if args.dry_run:
                print(" ".join(cmd))
                return (spec.output_dir, 0)
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if p.returncode != 0:
                # write log into output directory
                try:
                    with open(os.path.join(spec.output_dir, "batch_explain.log"), "w") as f:
                        f.write(p.stdout)
                except Exception:
                    pass
            return (spec.output_dir, int(p.returncode))

        failures = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_one, spec) for spec in runs]
            for fut in as_completed(futs):
                out_dir, code = fut.result()
                if code != 0:
                    failures += 1
                    print(f"[FAIL] {out_dir} (exit {code})", file=sys.stderr)
        if failures:
            raise SystemExit(f"{failures} runs failed.")
        return

    # Sequential (default; easiest to debug)
    for spec in runs:
        if not os.path.isfile(spec.image_path):
            print(f"[SKIP] Missing image: {spec.image_path}", file=sys.stderr)
            continue
        _safe_mkdir(spec.output_dir)
        if args.skip_existing and _already_done(spec.output_dir, spec.sid):
            print(f"[SKIP] Exists: {spec.output_dir}")
            continue
        cmd = _build_predict_cmd(predict_script, spec, passthrough)
        print(f"[RUN] {spec.label_name} {spec.sid} -> {spec.output_dir}")
        if args.dry_run:
            print(" ".join(cmd))
            continue
        p = subprocess.run(cmd)
        if p.returncode != 0:
            raise SystemExit(p.returncode)


if __name__ == "__main__":
    main()

