from pathlib import Path
import pandas as pd
import os

# Define constants
DATA_DIR = Path.home() / "reseng202500013-ndd-ml" / "data"
PREPROCESSED_MRI_DIR = DATA_DIR / "preprocessed" / "MRI" / "smriprep"
IMAGING_RECORDS_PATH = DATA_DIR / "imaging_records.csv"
OUTPUT_LABELS_PATH = DATA_DIR / "mri_labels.csv"

# Disease label mapping
label_map = {"AD": 0, "CN": 1, "PD": 2}

# Required file pattern for MRI subjects
# Only subjects with this file will be included: {subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz

def main():
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
    else:
        print("[ERROR] No valid labels found!")

if __name__ == "__main__":
    main()
