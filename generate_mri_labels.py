from pathlib import Path
import pandas as pd
import os
import argparse

# Define constants
DATA_DIR = Path.home() / "reseng202500013-ndd-ml" / "data"
PREPROCESSED_MRI_DIR = DATA_DIR / "preprocessed" / "MRI" / "smriprep"
IMAGING_RECORDS_PATH = DATA_DIR / "imaging_records.csv"
OUTPUT_LABELS_PATH = DATA_DIR / "mri_labels.csv"

# Disease label mapping (canonical for validation): 0=CN, 1=AD, 2=PD
label_map = {"CN": 0, "AD": 1, "PD": 2}


def validate_existing_labels(records_path: Path, labels_path: Path, out_dir: Path | None = None) -> int:
    """
    Validate an existing labels CSV (subject_id,label) against imaging_records.csv using
    canonical mapping 0=CN, 1=AD, 2=PD.

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
        # Treat trailing 'A' variant as base class
        if s.endswith('A') and s[:-1] in label_map:
            s = s[:-1]
        # Collapse prefixes
        if s.startswith('CN'):
            return 'CN'
        if s.startswith('AD'):
            return 'AD'
        if s.startswith('PD'):
            return 'PD'
        return s

    df_records = df_records.copy()
    df_records["DiseaseNorm"] = df_records["Disease"].apply(normalize_disease)

    # Build SubjectID -> expected_label mapping
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
        # Accept either 'sub-XXXX' or 'XXXX' in CSV; unify to numeric id
        subj_num = subj_full[4:] if subj_full.startswith('sub-') else subj_full
        lbl = int(row["label"]) if pd.notna(row["label"]) else None
        exp = rec_map.get(subj_num, None)
        if exp is None:
            not_in_records.append({"subject_id": subj_full, "label_csv": lbl, "records": None})
            continue
        if lbl != exp:
            # Expand names for reporting
            inv_map = {v: k for k, v in label_map.items()}
            mismatches.append({
                "subject_id": subj_full,
                "label_csv": lbl,
                "label_csv_name": inv_map.get(lbl, str(lbl)),
                "expected_label": exp,
                "expected_name": inv_map.get(exp, str(exp)),
            })

    inv_map = {v: k for k, v in label_map.items()}
    print("\n[VALIDATION] MRI labels vs imaging_records (0=CN,1=AD,2=PD)")
    print(f"  Total in labels: {len(df_labels)}")
    print(f"  Not found in records: {len(not_in_records)}")
    print(f"  Mismatches: {len(mismatches)}")

    if out_dir is None:
        out_dir = labels_path.parent
    out_dir = Path(out_dir)
    try:
        if mismatches:
            out_mis = out_dir / "mri_label_mismatches.csv"
            pd.DataFrame(mismatches).to_csv(out_mis, index=False)
            print(f"  → Wrote mismatches CSV: {out_mis}")
        if not_in_records:
            out_miss = out_dir / "mri_subjects_not_in_records.csv"
            pd.DataFrame(not_in_records).to_csv(out_miss, index=False)
            print(f"  → Wrote missing-in-records CSV: {out_miss}")
    except Exception as e:
        print(f"[WARN] Could not write validation reports: {e}")

    if mismatches:
        # Show a brief sample
        sample = mismatches[:10]
        sample_str = ", ".join([f"{m['subject_id']}: CSV {m['label_csv_name']} vs EXP {m['expected_name']}" for m in sample])
        print(f"  Sample mismatches: {sample_str}")

    return len(mismatches)

# Required file pattern for MRI subjects
# Only subjects with this file will be included: {subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz

def main():
    parser = argparse.ArgumentParser(description="Generate or validate MRI labels (0=CN,1=AD,2=PD)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing mri_labels.csv against imaging_records and exit")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to write validation reports (default: same as labels.csv)")
    args = parser.parse_args()

    if args.validate_only:
        validate_existing_labels(IMAGING_RECORDS_PATH, OUTPUT_LABELS_PATH, Path(args.out_dir) if args.out_dir else None)
        return

    print(f"[INFO] Scanning directory: {PREPROCESSED_MRI_DIR}")
    
    # Check if preprocessed directory exists
    if not PREPROCESSED_MRI_DIR.exists():
        raise FileNotFoundError(f"Preprocessed MRI directory not found: {PREPROCESSED_MRI_DIR}")
    
    # Load existing labels if file exists
    existing_subjects = set()
    existing_labels_data = []
    if OUTPUT_LABELS_PATH.exists():
        print(f"[INFO] Loading existing labels from: {OUTPUT_LABELS_PATH}")
        df_existing = pd.read_csv(OUTPUT_LABELS_PATH)
        existing_subjects = set(df_existing['subject_id'].tolist())
        existing_labels_data = df_existing.to_dict('records')
        print(f"[INFO] Found {len(existing_subjects)} existing subjects")
    
    # Find all subject folders with the required zscore file
    subject_folders = []
    for item in PREPROCESSED_MRI_DIR.iterdir():
        if item.is_dir() and item.name.startswith('sub-'):
            # Check if the required zscore file exists
            zscore_file = item / "anat" / f"{item.name}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz"
            if zscore_file.exists():
                # Extract the subject ID (remove 'sub-' prefix)
                subject_id = item.name[4:]  # Remove 'sub-' prefix
                subject_folders.append(subject_id)
                print(f"[INFO] Found zscore file for subject: {item.name}")
            else:
                print(f"[WARNING] Missing zscore file for subject: {item.name}")
    
    print(f"[INFO] Found {len(subject_folders)} subjects with zscore files")
    
    if not subject_folders:
        raise ValueError("No subjects with zscore files found in preprocessed directory")
    
    # Load imaging records
    print(f"[INFO] Loading imaging records from: {IMAGING_RECORDS_PATH}")
    df_records = pd.read_csv(IMAGING_RECORDS_PATH)
    
    # Create labels list (start with existing data if available)
    labels_data = []
    if OUTPUT_LABELS_PATH.exists():
        labels_data = existing_labels_data.copy()
        print(f"[INFO] Loaded {len(labels_data)} existing labels")
    
    new_subjects_count = 0
    removed_subjects_count = 0
    
    for subject_id in subject_folders:
        subject_name = f'sub-{subject_id}'
        
        # Skip if subject already exists in labels
        if subject_name in existing_subjects:
            print(f"[INFO] Skipping existing subject: {subject_name}")
            continue
        
        # Find this subject in the imaging records
        subject_record = df_records[df_records['SubjectID'] == subject_id]
        
        if subject_record.empty:
            print(f"[WARNING] Subject {subject_id} not found in imaging records, skipping")
            continue
        
        # Get and normalize the disease label
        disease_raw = subject_record.iloc[0]['Disease']
        disease = str(disease_raw).strip().upper()
        # Treat trailing 'a' variants (e.g., CNa, ADa, PDa) as their base class
        if disease.endswith('A') and disease[:-1] in label_map:
            disease = disease[:-1]
        
        if disease not in label_map:
            print(f"[WARNING] Unknown disease '{disease_raw}' normalized to '{disease}' for subject {subject_id}, skipping")
            continue
        
        label = label_map[disease]
        labels_data.append({
            'subject_id': subject_name,
            'label': label
        })
        
        new_subjects_count += 1
        print(f"[INFO] Added: {subject_name} -> {disease} (label {label})")
    
    # Remove subjects from existing data that don't have zscore files
    if OUTPUT_LABELS_PATH.exists():
        subjects_to_remove = []
        for i, label_entry in enumerate(labels_data):
            subject_name = label_entry['subject_id']
            # Check if this subject has a zscore file
            subject_dir = PREPROCESSED_MRI_DIR / subject_name
            zscore_file = subject_dir / "anat" / f"{subject_name}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz"
            
            if not zscore_file.exists():
                subjects_to_remove.append(i)
                print(f"[INFO] Removing: {subject_name} (no zscore file)")
        
        # Remove subjects in reverse order to maintain indices
        for i in reversed(subjects_to_remove):
            del labels_data[i]
            removed_subjects_count += 1
    
    # Create DataFrame and save
    if labels_data:
        df_labels = pd.DataFrame(labels_data)
        df_labels.to_csv(OUTPUT_LABELS_PATH, index=False)
        print(f"[INFO] Written {len(labels_data)} total labels to: {OUTPUT_LABELS_PATH}")
        print(f"[INFO] Added {new_subjects_count} new subjects")
        print(f"[INFO] Removed {removed_subjects_count} subjects without zscore files")
        
        # Print summary
        label_counts = df_labels['label'].value_counts().sort_index()
        print("\n[INFO] Label distribution:")
        for label, count in label_counts.items():
            disease_name = [k for k, v in label_map.items() if v == label][0]
            print(f"  {disease_name} (label {label}): {count} subjects")

        # Run a validation pass after writing
        print("\n[INFO] Running validation pass on newly written labels...")
        validate_existing_labels(IMAGING_RECORDS_PATH, OUTPUT_LABELS_PATH)
    else:
        print("[ERROR] No valid labels found!")

if __name__ == "__main__":
    main()
