from pathlib import Path
import pandas as pd
import os

# Define constants
DATA_DIR = Path.home() / "reseng202500013-ndd-ml" / "data"
PREPROCESSED_PET_DIR = DATA_DIR / "preprocessed" / "PET"
IMAGING_RECORDS_PATH = DATA_DIR / "imaging_records.csv"
OUTPUT_LABELS_PATH = DATA_DIR / "pet_labels.csv"

# Disease label mapping
label_map = {"AD": 0, "CN": 1, "PD": 2}

# Required file pattern for PET subjects
# Includes subjects with files like:
#   {subject_id}_{site}_PET_{disease}_SUVR_s2_brain_soft4.nii.gz
#   {subject_id}_{site}_PET_{disease}_SUVR_s2_brain_soft4.nii
#   {subject_id}_{site}_PET_{disease}_SUVR.nii.gz
#   {subject_id}_{site}_PET_{disease}_SUVR.nii


def main():
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
        
        if disease not in label_map:
            print(f"[WARNING] Unknown disease '{disease}' for subject {subject_numeric}, skipping")
            continue
        
        label = label_map[disease]
        labels_data.append({
            'subject_id': subject_id,
            'label': label
        })
        
        new_subjects_count += 1
        print(f"[INFO] Added: {subject_id} -> {disease} (label {label})")
    
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
    else:
        print("[ERROR] No valid labels found!")

if __name__ == "__main__":
    main() 