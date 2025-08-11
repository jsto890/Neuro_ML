#!/usr/bin/env python3
"""
Find subjects missing MNI-space sMRIPrep outputs, grouped by site and disease.

Reference: Uses the same MNI filename logic as `03_zscore_skull_strip.py`:
  - Must contain both "MNI152NLin2009cAsym" and "_desc-preproc_T1w.nii.gz"
  - Must contain both "MNI152NLin2009cAsym" and "_desc-brain_mask.nii.gz"

The script compares subjects present in raw MRI directories (organized as
<raw_smri>/<site>/<disease>/sub-*) against sMRIPrep derivatives at
<smri_p>/sub-*/anat.

Outputs a readable report to stdout and optionally a CSV of missing subjects.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set

try:
    import yaml  # type: ignore
except Exception as yaml_import_error:  # pragma: no cover
    yaml = None


def load_paths_from_config(config_path: Path) -> Tuple[Path, Path]:
    """Load raw and preprocessed MRI paths from config.yaml.

    Returns:
        (raw_smri_root, smri_p_root)
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if yaml is None:
        raise RuntimeError("pyyaml is required to read the config. Install with 'pip install pyyaml'.")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_smri = cfg.get("raw_data", {}).get("smri", None)
    smri_p = cfg.get("preprocessed_data", {}).get("smri_p", None)
    if not raw_smri or not smri_p:
        raise KeyError("Missing 'raw_data.smri' or 'preprocessed_data.smri_p' in config.yaml")

    raw_smri_root = Path(os.path.expanduser(raw_smri)).resolve()
    smri_p_root = Path(os.path.expanduser(smri_p)).resolve()
    return raw_smri_root, smri_p_root


def enumerate_subjects_by_site_and_disease(raw_smri_root: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Return mapping: site -> disease -> set(subject_ids) discovered in raw data.

    Expects directory structure: raw_smri_root/<site>/<disease>/sub-*.
    Subject IDs are directory basenames (e.g., 'sub-XXXX').
    """
    mapping: Dict[str, Dict[str, Set[str]]] = {}
    if not raw_smri_root.is_dir():
        return mapping

    for site_dir in sorted([p for p in raw_smri_root.iterdir() if p.is_dir()]):
        site_name = site_dir.name
        for disease_dir in sorted([p for p in site_dir.iterdir() if p.is_dir()]):
            disease_name = disease_dir.name
            subs = set()
            for subj_dir in sorted(disease_dir.glob("sub-*")):
                if subj_dir.is_dir():
                    subs.add(subj_dir.name)
            if subs:
                mapping.setdefault(site_name, {}).setdefault(disease_name, set()).update(subs)
    return mapping


def subject_is_complete_in_mni(smriprep_root: Path, subject_id: str) -> bool:
    """Check for required MNI-space files in sMRIPrep derivatives for subject.

    Looks under: <smriprep_root>/<subject_id>/anat
    and searches for files containing both the MNI tag and the keyword
    (preproc and mask), mirroring 03_zscore_skull_strip.py.
    """
    anat_dir = smriprep_root / subject_id / "anat"
    if not anat_dir.is_dir():
        return False

    try:
        files = os.listdir(anat_dir)
    except Exception:
        return False

    def has(keyword: str) -> bool:
        for fname in files:
            if "MNI152NLin2009cAsym" in fname and keyword in fname:
                return True
        return False

    has_preproc = has("_desc-preproc_T1w.nii.gz")
    has_mask = has("_desc-brain_mask.nii.gz")
    return has_preproc and has_mask


def find_missing_by_group(
    subjects_by_site_disease: Dict[str, Dict[str, Set[str]]],
    smriprep_root: Path,
) -> Dict[str, Dict[str, List[str]]]:
    """Return mapping: site -> disease -> [missing_subject_ids]."""
    missing: Dict[str, Dict[str, List[str]]] = {}
    for site_name, diseases in subjects_by_site_disease.items():
        for disease_name, subject_ids in diseases.items():
            missing_list: List[str] = []
            for subject_id in sorted(subject_ids):
                if not subject_is_complete_in_mni(smriprep_root, subject_id):
                    missing_list.append(subject_id)
            if missing_list:
                missing.setdefault(site_name, {})[disease_name] = missing_list
    return missing


def print_report(missing: Dict[str, Dict[str, List[str]]]) -> None:
    """Pretty-print grouped report to stdout."""
    if not missing:
        print("All subjects appear to have required MNI outputs. ✅")
        return

    print("\nMissing MNI subjects grouped by site and disease:\n")
    for site_name in sorted(missing.keys()):
        print(f"Site: {site_name}")
        diseases = missing[site_name]
        for disease_name in sorted(diseases.keys()):
            subs = diseases[disease_name]
            print(f"  Disease: {disease_name}  | Missing: {len(subs)}")
            # Print in columns of reasonable width
            line: List[str] = []
            for idx, subj in enumerate(subs, start=1):
                line.append(subj)
                if idx % 8 == 0:
                    print("    " + ", ".join(line))
                    line = []
            if line:
                print("    " + ", ".join(line))
        print("")


def write_csv(missing: Dict[str, Dict[str, List[str]]], out_csv: Path) -> None:
    """Write missing subjects to CSV with columns: site,disease,subject_id."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("site,disease,subject_id\n")
        for site_name, diseases in sorted(missing.items()):
            for disease_name, subject_ids in sorted(diseases.items()):
                for subject_id in subject_ids:
                    f.write(f"{site_name},{disease_name},{subject_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List subjects missing MNI-space sMRIPrep outputs, grouped by site and disease. "
            "By default reads raw and preprocessed paths from config.yaml."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("~/reseng202500013-ndd-ml/P4P/config.yaml"),
        help="Path to config.yaml containing raw_data.smri and preprocessed_data.smri_p",
    )
    parser.add_argument(
        "--raw-smri",
        type=Path,
        default=None,
        help="Override for raw MRI root (expects <site>/<disease>/sub-* under this).",
    )
    parser.add_argument(
        "--smriprep-root",
        type=Path,
        default=None,
        help="Override for sMRIPrep derivatives root (expects sub-*/anat under this).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV path to write missing subjects (site,disease,subject_id). "
            "No default is set."
        ),
    )

    args = parser.parse_args()

    # Resolve paths
    if args.raw_smri is not None and args.smriprep_root is not None:
        raw_smri_root = args.raw_smri.expanduser().resolve()
        smri_p_root = args.smriprep_root.expanduser().resolve()
    else:
        raw_smri_root, smri_p_root = load_paths_from_config(args.config.expanduser().resolve())

    # Validate
    if not raw_smri_root.is_dir():
        raise FileNotFoundError(f"Raw MRI root not found: {raw_smri_root}")
    if not smri_p_root.is_dir():
        raise FileNotFoundError(f"sMRIPrep root not found: {smri_p_root}")

    subjects_by_group = enumerate_subjects_by_site_and_disease(raw_smri_root)
    missing = find_missing_by_group(subjects_by_group, smri_p_root)

    print(f"Raw MRI root:        {raw_smri_root}")
    print(f"sMRIPrep root:       {smri_p_root}")
    print_report(missing)

    if args.out_csv is not None:
        out_csv = args.out_csv.expanduser().resolve()
        write_csv(missing, out_csv)
        print(f"CSV written: {out_csv}")


if __name__ == "__main__":
    main()


