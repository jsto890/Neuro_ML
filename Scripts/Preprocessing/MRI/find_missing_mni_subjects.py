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
from typing import Dict, List, Tuple, Set, Optional

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


def enumerate_subjects_in_smriprep(smriprep_root: Path) -> List[str]:
    """Return list of subject IDs discovered under sMRIPrep root (sub-*/anat)."""
    if not smriprep_root.is_dir():
        return []
    subjects: List[str] = []
    for entry in sorted(smriprep_root.iterdir()):
        if entry.is_dir() and entry.name.startswith("sub-"):
            # require anat folder exists to be considered a subject folder
            if (entry / "anat").is_dir():
                subjects.append(entry.name)
    return subjects


def normalize_subject_id(value: str) -> str:
    v = str(value).strip()
    if not v:
        return v
    if v.startswith("sub-"):
        return v
    # add sub- prefix if missing
    return f"sub-{v}"


def load_site_disease_mapping(records_csv: Path) -> Dict[str, Tuple[str, str]]:
    """Load mapping from subject_id -> (site, disease) from imaging_records.csv.

    Tries common column names for subject, site, and disease.
    Subject IDs are normalized to 'sub-XXXX' format.
    """
    import csv

    if not records_csv.is_file():
        raise FileNotFoundError(f"Records CSV not found: {records_csv}")

    subject_cols = [
        "subject_id",
        "SubjectID",
        "Subject",
        "subject",
        "SUBJECT",
        "participant_id",
        "ParticipantID",
        "ID",
        "id",
    ]
    site_cols = ["Site", "site", "SITE"]
    disease_cols = ["Disease", "disease", "DISEASE", "Diagnosis", "diagnosis", "Group", "group"]
    modality_cols = ["Modality", "modality", "MODALITY", "mod", "Mod"]

    mapping: Dict[str, Tuple[str, str]] = {}
    with open(records_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Records CSV has no header")
        fields_lower = {name.lower(): name for name in reader.fieldnames}

        def pick(cols: List[str]) -> Optional[str]:
            for c in cols:
                # exact match first
                if c in reader.fieldnames:
                    return c
                # case-insensitive fallback
                cl = c.lower()
                if cl in fields_lower:
                    return fields_lower[cl]
            return None

        subj_col = pick(subject_cols)
        site_col = pick(site_cols)
        dis_col = pick(disease_cols)
        mod_col = pick(modality_cols)
        if subj_col is None:
            raise KeyError(
                f"Could not find subject column in CSV. Tried: {subject_cols}. Found: {reader.fieldnames}"
            )
        if site_col is None or dis_col is None:
            # allow missing site or disease; will fallback to 'UNKNOWN'
            pass

        rows = list(reader)
        # Prefer MRI rows if modality column is present and any MRI exist
        if mod_col is not None:
            mri_rows = [r for r in rows if str(r.get(mod_col, "")).strip().lower() == "mri"]
            if len(mri_rows) > 0:
                rows = mri_rows

        for row in rows:
            sub_raw = row.get(subj_col, "")
            if sub_raw is None:
                continue
            sub_id = normalize_subject_id(str(sub_raw))
            site_val = row.get(site_col, "UNKNOWN") if site_col else "UNKNOWN"
            dis_val = row.get(dis_col, "UNKNOWN") if dis_col else "UNKNOWN"
            mapping[sub_id] = (str(site_val), str(dis_val))
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


def find_missing_by_group_from_smriprep(
    subjects: List[str],
    smriprep_root: Path,
    site_dis_map: Dict[str, Tuple[str, str]],
) -> Dict[str, Dict[str, List[str]]]:
    """Return mapping: site -> disease -> [missing_subject_ids] using sMRIPrep subject list."""
    missing: Dict[str, Dict[str, List[str]]] = {}
    for subject_id in subjects:
        complete = subject_is_complete_in_mni(smriprep_root, subject_id)
        if complete:
            continue
        site, disease = site_dis_map.get(subject_id, ("UNKNOWN", "UNKNOWN"))
        missing.setdefault(site, {}).setdefault(disease, []).append(subject_id)
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
        "--smriprep-root",
        type=Path,
        default=None,
        help="Override for sMRIPrep derivatives root (expects sub-*/anat under this).",
    )
    parser.add_argument(
        "--records-csv",
        type=Path,
        default=Path("~/reseng202500013-ndd-ml/data/imaging_records.csv"),
        help="Path to imaging_records.csv used to map subject IDs to Site and Disease.",
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

    # Resolve sMRIPrep root
    if args.smriprep_root is not None:
        smri_p_root = args.smriprep_root.expanduser().resolve()
    else:
        # Load from config if not explicitly provided
        _, smri_p_root = load_paths_from_config(args.config.expanduser().resolve())

    # Validate
    if not smri_p_root.is_dir():
        raise FileNotFoundError(f"sMRIPrep root not found: {smri_p_root}")
    records_csv = args.records_csv.expanduser().resolve()

    # Enumerate subjects from sMRIPrep and load records mapping
    subjects = enumerate_subjects_in_smriprep(smri_p_root)
    site_dis_map = load_site_disease_mapping(records_csv)
    missing = find_missing_by_group_from_smriprep(subjects, smri_p_root, site_dis_map)

    print(f"sMRIPrep root:       {smri_p_root}")
    print(f"Records CSV:         {records_csv}")
    print_report(missing)

    if args.out_csv is not None:
        out_csv = args.out_csv.expanduser().resolve()
        write_csv(missing, out_csv)
        print(f"CSV written: {out_csv}")


if __name__ == "__main__":
    main()


