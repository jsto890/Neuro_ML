from pathlib import Path
import pandas as pd
import os
import argparse

# Define constants (override with CLI if needed)
DATA_DIR = Path.home() / "reseng202500013-ndd-ml" / "data"
# Default to preprocessed DSPECT structure; can be overridden via --data-root
SPECT_ROOT = DATA_DIR / "preprocessed" / "SPECT"
CN_DIRNAME = "CN_SPECT_PPMI_postprocessed"
PD_DIRNAME = "PD_SPECT_PPMI_postprocessed"
IMAGING_RECORDS_PATH = DATA_DIR / "imaging_records.csv"
DEFAULT_OUTPUT_LABELS_PATH = SPECT_ROOT / "spect_labels.csv"

# Disease label mapping (binary for SPECT): 0=CN, 1=PD
label_map = {"CN": 0, "PD": 1}


def validate_existing_labels(records_path: Path, labels_path: Path, out_dir: Path | None = None, write_reports: bool = False) -> int:
    """
    Validate an existing SPECT labels CSV (subject_id,label) against imaging_records.csv
    using canonical mapping 0=CN, 1=PD (AD is ignored for SPECT).

    Returns number of mismatches found. Optionally writes a mismatches CSV in out_dir.
    """
    if not labels_path.exists():
        print(f"[ERROR] Labels CSV not found: {labels_path}")
        return -1

    print(f"[INFO] Validating labels in: {labels_path}")
    df_labels = pd.read_csv(labels_path)
    if "subject_id" not in df_labels.columns or "label" not in df_labels.columns:
        raise ValueError("Labels CSV must contain columns: subject_id,label")

    print(f"[INFO] Loading imaging records: {records_path}")
    df_records = pd.read_csv(records_path)
    if "SubjectID" not in df_records.columns or "Disease" not in df_records.columns:
        raise ValueError("imaging_records.csv must contain columns: SubjectID,Disease")

    # Normalize disease strings
    def normalize_disease(d):
        s = str(d).strip().upper()
        if s.endswith('A') and s[:-1] in {"CN", "AD", "PD"}:
            s = s[:-1]
        if s.startswith('CN'):
            return 'CN'
        if s.startswith('PD'):
            return 'PD'
        if s.startswith('AD'):
            return 'AD'
        return s

    df_records = df_records.copy()
    df_records["DiseaseNorm"] = df_records["Disease"].apply(normalize_disease)

    # Build SubjectID -> expected_label mapping (only CN/PD)
    rec_map = {}
    for _, row in df_records.iterrows():
        sid = str(row["SubjectID"]).strip()
        dis = row["DiseaseNorm"]
        if dis in label_map:
            rec_map[sid] = label_map[dis]

    mismatches = []
    not_in_records = []
    for _, row in df_labels.iterrows():
        subj_full = str(row["subject_id"]).strip()
        # For SPECT we expect folder names like 'Subject_*', but imaging records use numeric IDs
        subj_num = subj_full
        if subj_full.startswith('sub-'):
            subj_num = subj_full[4:]
        elif subj_full.lower().startswith('subject_'):
            subj_num = subj_full.split('_', 1)[-1]
        lbl = int(row["label"]) if pd.notna(row["label"]) else None
        exp = rec_map.get(subj_num, None)
        if exp is None:
            not_in_records.append({"subject_id": subj_full, "label_csv": lbl, "records": None})
            continue
        if lbl != exp:
            inv_map = {v: k for k, v in label_map.items()}
            mismatches.append({
                "subject_id": subj_full,
                "label_csv": lbl,
                "label_csv_name": inv_map.get(lbl, str(lbl)),
                "expected_label": exp,
                "expected_name": inv_map.get(exp, str(exp)),
            })

    inv_map = {v: k for k, v in label_map.items()}
    print("\n[VALIDATION] SPECT labels vs imaging_records (0=CN,1=PD)")
    print(f"  Total in labels: {len(df_labels)}")
    print(f"  Not found in records: {len(not_in_records)}")
    print(f"  Mismatches: {len(mismatches)}")

    if out_dir is None:
        out_dir = labels_path.parent
    out_dir = Path(out_dir)

    # Binary mismatch pair counts
    pair_counts = {k: 0 for k in [
        "CN->PD", "PD->CN"
    ]}
    for m in mismatches:
        k = f"{m['expected_name']}->{m['label_csv_name']}"
        if k in pair_counts:
            pair_counts[k] += 1
    for key in ["CN->PD", "PD->CN"]:
        print(f"{key} {pair_counts[key]}")

    if write_reports:
        try:
            if mismatches:
                out_mis = out_dir / "spect_label_mismatches.csv"
                pd.DataFrame(mismatches).to_csv(out_mis, index=False)
                print(f"  → Wrote mismatches CSV: {out_mis}")
            if not_in_records:
                out_miss = out_dir / "spect_subjects_not_in_records.csv"
                pd.DataFrame(not_in_records).to_csv(out_miss, index=False)
                print(f"  → Wrote missing-in-records CSV: {out_miss}")
        except Exception as e:
            print(f"[WARN] Could not write validation reports: {e}")

    if mismatches:
        sample = mismatches[:10]
        sample_str = ", ".join([f"{m['subject_id']}: CSV {m['label_csv_name']} vs EXP {m['expected_name']}" for m in sample])
        print(f"  Sample mismatches: {sample_str}")

    return len(mismatches)


def collect_spect_labels(data_root: Path) -> list[dict]:
    """
    Scan DSPECT-processed folders for CN/PD subjects and build label rows.
    Expects:
      data_root/CN_SPECT_PPMI_postprocessed/Subject_*/6. postprocessed.nii.gz
      data_root/PD_SPECT_PPMI_postprocessed/Subject_*/6. postprocessed.nii.gz
    Returns list of dicts: {'subject_id': 'Subject_*', 'label': 0|1}
    """
    rows: list[dict] = []
    cn_dir = data_root / CN_DIRNAME
    pd_dir = data_root / PD_DIRNAME

    if not cn_dir.exists():
        raise FileNotFoundError(f"CN directory not found: {cn_dir}")
    if not pd_dir.exists():
        raise FileNotFoundError(f"PD directory not found: {pd_dir}")

    # Helper to scan a class directory
    def scan_class_dir(root: Path, label: int):
        n_found = 0
        n_missing = 0
        for item in root.iterdir():
            if item.is_dir() and item.name.startswith('Subject_'):
                nii = item / "6. postprocessed.nii.gz"
                if nii.exists():
                    rows.append({"subject_id": item.name, "label": label})
                    n_found += 1
                else:
                    print(f"[WARNING] Missing SPECT file: {nii}")
                    n_missing += 1
        print(f"[INFO] {root.name}: found={n_found}, missing_file={n_missing}")

    print(f"[INFO] Scanning SPECT CN dir: {cn_dir}")
    scan_class_dir(cn_dir, label_map["CN"])  # 0
    print(f"[INFO] Scanning SPECT PD dir: {pd_dir}")
    scan_class_dir(pd_dir, label_map["PD"])  # 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate or validate SPECT labels (0=CN,1=PD)")
    parser.add_argument("--data-root", type=str, default=str(SPECT_ROOT),
                        help="Root directory containing DSPECT postprocessed data (CN_/PD_ folders)")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_LABELS_PATH),
                        help="Path to write spect_labels.csv (default: Final_SPECT/spect_labels.csv)")
    parser.add_argument("--records-path", type=str, default=str(IMAGING_RECORDS_PATH),
                        help="Path to imaging_records.csv for validation")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing spect_labels.csv against imaging_records and exit")
    parser.add_argument("--write-reports", action="store_true",
                        help="Also write CSV reports for mismatches and missing records")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to write validation reports (default: same as labels.csv)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    labels_path = Path(args.output_path)
    records_path = Path(args.records_path)

    if args.validate_only:
        validate_existing_labels(records_path, labels_path, Path(args.out_dir) if args.out_dir else None, write_reports=args.write_reports)
        return

    # Generate labels by scanning DSPECT folders
    print(f"[INFO] Generating SPECT labels from: {data_root}")
    rows = collect_spect_labels(data_root)

    if not rows:
        print("[ERROR] No SPECT subjects found. Check data_root and folder structure.")
        return

    # Save labels
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    df_labels = pd.DataFrame(rows)
    df_labels = df_labels.sort_values(by=["label", "subject_id"]).reset_index(drop=True)
    df_labels.to_csv(labels_path, index=False)
    print(f"[INFO] Written {len(df_labels)} labels to: {labels_path}")

    # Print distribution
    counts = df_labels['label'].value_counts().sort_index()
    inv_map = {v: k for k, v in label_map.items()}
    print("\n[INFO] Label distribution:")
    for label_value, count in counts.items():
        print(f"  {inv_map.get(int(label_value), label_value)} (label {int(label_value)}): {count} subjects")

    # Optional validation pass
    if records_path.exists():
        print("\n[INFO] Running validation pass on newly written SPECT labels...")
        validate_existing_labels(records_path, labels_path, Path(args.out_dir) if args.out_dir else None, write_reports=args.write_reports)
    else:
        print(f"[WARN] imaging_records.csv not found: {records_path} — skipping validation")


if __name__ == "__main__":
    main()


