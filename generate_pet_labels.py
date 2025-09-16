from pathlib import Path
import pandas as pd
import os
import argparse

# Define constants
DATA_DIR = Path.home() / "reseng202500013-ndd-ml" / "data"
PREPROCESSED_PET_DIR = DATA_DIR / "preprocessed" / "PET"
IMAGING_RECORDS_PATH = DATA_DIR / "imaging_records.csv"
OUTPUT_LABELS_PATH = DATA_DIR / "pet_labels.csv"

# Disease label mapping (canonical for validation): 0=CN, 1=AD, 2=PD
label_map = {"CN": 0, "AD": 1, "PD": 2}

def normalize_disease(disease_raw):
    s = str(disease_raw).strip().upper()
    if s.endswith('A') and s[:-1] in label_map:
        s = s[:-1]
    if s.startswith("CN"):
        return "CN"
    if s.startswith("AD"):
        return "AD"
    if s.startswith("PD"):
        return "PD"
    return s

# Required file pattern for PET subjects
# Includes subjects with files like:
#   {subject_id}_{site}_PET_{disease}_SUVR_s2_brain_soft4.nii.gz
#   {subject_id}_{site}_PET_{disease}_SUVR_s2_brain_soft4.nii
#   {subject_id}_{site}_PET_{disease}_SUVR.nii.gz
#   {subject_id}_{site}_PET_{disease}_SUVR.nii


def validate_existing_labels(records_path: Path, labels_path: Path, out_dir: Path | None = None) -> int:
    """
    Validate an existing PET labels CSV (subject_id,label) against imaging_records.csv using
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
        subj_num = subj_full[4:] if subj_full.startswith('sub-') else subj_full
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

    print("\n[VALIDATION] PET labels vs imaging_records (0=CN,1=AD,2=PD)")
    print(f"  Total in labels: {len(df_labels)}")
    print(f"  Not found in records: {len(not_in_records)}")
    print(f"  Mismatches: {len(mismatches)}")

    if out_dir is None:
        out_dir = labels_path.parent
    out_dir = Path(out_dir)
    try:
        if mismatches:
            out_mis = out_dir / "pet_label_mismatches.csv"
            pd.DataFrame(mismatches).to_csv(out_mis, index=False)
            print(f"  → Wrote mismatches CSV: {out_mis}")
        if not_in_records:
            out_miss = out_dir / "pet_subjects_not_in_records.csv"
            pd.DataFrame(not_in_records).to_csv(out_miss, index=False)
            print(f"  → Wrote missing-in-records CSV: {out_miss}")
    except Exception as e:
        print(f"[WARN] Could not write validation reports: {e}")

    if mismatches:
        sample = mismatches[:10]
        sample_str = ", ".join([f"{m['subject_id']}: CSV {m['label_csv_name']} vs EXP {m['expected_name']}" for m in sample])
        print(f"  Sample mismatches: {sample_str}")

    return len(mismatches)


def main():
    parser = argparse.ArgumentParser(description="Generate or validate PET labels (0=CN,1=AD,2=PD)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing pet_labels.csv against imaging_records and exit")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to write validation reports (default: same as labels.csv)")
    args = parser.parse_args()

    if args.validate_only:
        validate_existing_labels(IMAGING_RECORDS_PATH, OUTPUT_LABELS_PATH, Path(args.out_dir) if args.out_dir else None)
        return

    print(f"[INFO] Scanning directory: {PREPROCESSED_PET_DIR}")
    
    # Check if preprocessed directory exists
    if not PREPROCESSED_PET_DIR.exists():
        raise FileNotFoundError(f"Preprocessed PET directory not found: {PREPROCESSED_PET_DIR}")
    
    # Check if output file exists and load existing data
    existing_labels_data = []
    if OUTPUT_LABELS_PATH.exists():
        print(f"[INFO] Loading existing labels from: {OUTPUT_LABELS_PATH}")
        df_existing = pd.read_csv(OUTPUT_LABELS_PATH)
        existing_labels_data = df_existing.to_dict('records')
        print(f"[INFO] Found {len(existing_labels_data)} existing subjects")
        print(f"[INFO] Will remove subjects without PET SUVR files")
    
    # Find all PET SUVR files in the preprocessed directory
    # Expected structure: PET/{disease}/{subject_dir}/
    pet_files = []
    subject_ids = set()

    for disease_dir in PREPROCESSED_PET_DIR.iterdir():
        if not disease_dir.is_dir():
            continue

        disease_folder_name = disease_dir.name
        print(f"[INFO] Scanning disease: {disease_folder_name}")

        for subject_dir in disease_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject_dir_name = subject_dir.name
            if not subject_dir_name.startswith('sub-'):
                continue

            # Derive subject_id and disease token from directory name, e.g., sub-XXX_ADNI_PET_CN
            parts = subject_dir_name.split('_')
            subject_id = parts[0] if parts else subject_dir_name
            disease_token = parts[-1].upper() if len(parts) >= 4 else disease_folder_name.upper()

            # Prefer soft4, then legacy; prefer .nii.gz, then .nii
            patterns = [
                f"{subject_id}_*_PET_{disease_token}_SUVR_s2_brain_soft4.nii.gz",
                f"{subject_id}_*_PET_{disease_token}_SUVR_s2_brain_soft4.nii",
                f"{subject_id}_*_PET_{disease_token}_SUVR.nii.gz",
                f"{subject_id}_*_PET_{disease_token}_SUVR.nii",
            ]

            found_path = None
            for pat in patterns:
                matches = list(subject_dir.glob(pat))
                if matches:
                    found_path = matches[0]
                    break

            if found_path is not None:
                pet_files.append(found_path)
                subject_ids.add(subject_id)
                print(f"[INFO] Found PET SUVR file: {found_path}")
            else:
                print(f"[WARNING] No SUVR file found for {subject_id} in {subject_dir}")
                print(f"[WARNING] Tried patterns: {patterns}")
    
    print(f"[INFO] Found {len(subject_ids)} unique subjects with PET SUVR files")
    print(f"[INFO] Found {len(pet_files)} PET SUVR files")
    
    if not subject_ids:
        raise ValueError("No PET files found in preprocessed directory")
    
    # Load imaging records
    print(f"[INFO] Loading imaging records from: {IMAGING_RECORDS_PATH}")
    df_records = pd.read_csv(IMAGING_RECORDS_PATH)
    
    # Create labels list (only for subjects with PET SUVR files)
    labels_data = []
    new_subjects_count = 0
    removed_subjects_count = 0
    
    for subject_id in subject_ids:
        # Extract the numeric part (e.g., "sub-001" -> "001")
        subject_numeric = subject_id[4:]  # Remove 'sub-' prefix
        
        # Find this subject in the imaging records
        subject_record = df_records[df_records['SubjectID'] == subject_numeric]
        
        if subject_record.empty:
            print(f"[WARNING] Subject {subject_numeric} not found in imaging records, skipping")
            continue
        
        # Get the disease label
        disease = subject_record.iloc[0]['Disease']
        disease_norm = normalize_disease(disease)
        
        if disease_norm not in label_map:
            print(f"[WARNING] Unknown disease '{disease}' for subject {subject_numeric}, skipping")
            continue
        
        label = label_map[disease_norm]
        labels_data.append({
            'subject_id': subject_id,
            'label': label
        })
        
        new_subjects_count += 1
        print(f"[INFO] Added: {subject_id} -> {disease_norm} (label {label})")
    
    # Remove subjects from existing data that don't have SUVR files
    if OUTPUT_LABELS_PATH.exists():
        subjects_to_remove = []
        for i, label_entry in enumerate(existing_labels_data):
            subject_name = label_entry['subject_id']
            # Check if this subject has a SUVR file in any disease folder
            subject_found = False
            for disease_dir in PREPROCESSED_PET_DIR.iterdir():
                if not disease_dir.is_dir():
                    continue
                for subject_dir in disease_dir.iterdir():
                    if not subject_dir.is_dir():
                        continue
                    subject_dir_name = subject_dir.name
                    if not subject_dir_name.startswith('sub-'):
                        continue
                    parts = subject_dir_name.split('_')
                    subject_id_check = parts[0] if parts else subject_dir_name
                    if subject_id_check != subject_name:
                        continue
                    disease_token = parts[-1].upper() if len(parts) >= 4 else disease_dir.name.upper()
                    patterns = [
                        f"{subject_id_check}_*_PET_{disease_token}_SUVR_s2_brain_soft4.nii.gz",
                        f"{subject_id_check}_*_PET_{disease_token}_SUVR_s2_brain_soft4.nii",
                        f"{subject_id_check}_*_PET_{disease_token}_SUVR.nii.gz",
                        f"{subject_id_check}_*_PET_{disease_token}_SUVR.nii",
                    ]
                    for pat in patterns:
                        if list(subject_dir.glob(pat)):
                            subject_found = True
                            break
                    if subject_found:
                        break
                if subject_found:
                    break
            
            if not subject_found:
                subjects_to_remove.append(i)
                print(f"[INFO] Removing: {subject_name} (no SUVR file)")
        
        # Remove subjects in reverse order to maintain indices
        for i in reversed(subjects_to_remove):
            del existing_labels_data[i]
            removed_subjects_count += 1
    
    print(f"[INFO] Added {new_subjects_count} subjects with PET SUVR files")
    print(f"[INFO] Removed {removed_subjects_count} subjects without PET SUVR files")
    
    # Create DataFrame and save
    if labels_data:
        df_labels = pd.DataFrame(labels_data)
        df_labels.to_csv(OUTPUT_LABELS_PATH, index=False)
        print(f"[INFO] Written {len(labels_data)} total labels to: {OUTPUT_LABELS_PATH}")
        print(f"[INFO] Added {new_subjects_count} new subjects")
        print(f"[INFO] Removed {removed_subjects_count} subjects without SUVR files")
        
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