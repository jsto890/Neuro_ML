from pathlib import Path
import pandas as pd
import os

# Define constants
DATA_DIR = Path.home() / "reseng202500013-ndd-ml" / "data"
PREPROCESSED_PET_DIR = DATA_DIR / "preprocessed" / "PET"
IMAGING_RECORDS_PATH = DATA_DIR / "imaging_records.csv"
OUTPUT_LABELS_PATH = DATA_DIR / "pet_labels.csv"

# Disease label mapping
label_map = {"CN": 0, "AD": 1, "PD": 2}

def main():
    print(f"[INFO] Scanning directory: {PREPROCESSED_PET_DIR}")
    
    # Check if preprocessed directory exists
    if not PREPROCESSED_PET_DIR.exists():
        raise FileNotFoundError(f"Preprocessed PET directory not found: {PREPROCESSED_PET_DIR}")
    
    # Check if output file exists and warn about overwriting
    if OUTPUT_LABELS_PATH.exists():
        print(f"[WARNING] Output file already exists: {OUTPUT_LABELS_PATH}")
        print(f"[WARNING] This will create a new file with only subjects that have PET SUVR files")
        print(f"[WARNING] Existing subjects without PET SUVR files will be removed")
    
    # Find all PET SUVR files in the preprocessed directory
    # Structure: PET/(site)/(disease)/((sub-id)_SITE_PET_DISEASE)/(sub-id)_SITE_PET_DISEASE_SUVR.nii.gz
    pet_files = []
    subject_ids = set()
    
    for site_dir in PREPROCESSED_PET_DIR.iterdir():
        if not site_dir.is_dir():
            continue
            
        site_name = site_dir.name
        print(f"[INFO] Scanning site: {site_name}")
        
        for disease_dir in site_dir.iterdir():
            if not disease_dir.is_dir():
                continue
                
            disease_name = disease_dir.name
            print(f"[INFO] Scanning disease: {disease_name}")
            
            for subject_dir in disease_dir.iterdir():
                if not subject_dir.is_dir():
                    continue
                    
                # Extract subject ID from directory name (e.g., "sub-001_ADNI_PET_CN" -> "sub-001")
                subject_dir_name = subject_dir.name
                if subject_dir_name.startswith('sub-'):
                    # Extract the subject ID part before the first underscore after sub-
                    parts = subject_dir_name.split('_')
                    if len(parts) >= 2:
                        subject_id = parts[0]  # This will be "sub-001"
                        
                        # Look for SUVR files with dynamic naming
                        sites = ['ADNI', 'PPMI']
                        diseases = ['CN', 'PD', 'AD']
                        
                        suvr_found = False
                        for site in sites:
                            for disease in diseases:
                                suvr_file = subject_dir / f"{subject_id}_{site}_PET_{disease}_SUVR.nii.gz"
                                if suvr_file.exists():
                                    pet_files.append(suvr_file)
                                    subject_ids.add(subject_id)
                                    suvr_found = True
                                    print(f"[INFO] Found PET SUVR file: {suvr_file}")
                                    break
                            if suvr_found:
                                break
                        
                        if not suvr_found:
                            print(f"[WARNING] No SUVR file found for {subject_id} in {subject_dir}")
    
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
    skipped_subjects = 0
    
    for subject_id in subject_ids:
        # Extract the numeric part (e.g., "sub-001" -> "001")
        subject_numeric = subject_id[4:]  # Remove 'sub-' prefix
        
        # Find this subject in the imaging records
        subject_record = df_records[df_records['SubjectID'] == subject_numeric]
        
        if subject_record.empty:
            print(f"[WARNING] Subject {subject_numeric} not found in imaging records, skipping")
            skipped_subjects += 1
            continue
        
        # Get the disease label
        disease = subject_record.iloc[0]['Disease']
        
        if disease not in label_map:
            print(f"[WARNING] Unknown disease '{disease}' for subject {subject_numeric}, skipping")
            skipped_subjects += 1
            continue
        
        label = label_map[disease]
        labels_data.append({
            'subject_id': subject_id,
            'label': label
        })
        
        new_subjects_count += 1
        print(f"[INFO] Added: {subject_id} -> {disease} (label {label})")
    
    print(f"[INFO] Added {new_subjects_count} subjects with PET SUVR files")
    print(f"[INFO] Skipped {skipped_subjects} subjects without PET SUVR files or invalid records")
    
    # Create DataFrame and save
    if labels_data:
        df_labels = pd.DataFrame(labels_data)
        df_labels.to_csv(OUTPUT_LABELS_PATH, index=False)
        print(f"[INFO] Written {len(labels_data)} total labels to: {OUTPUT_LABELS_PATH}")
        print(f"[INFO] Added {new_subjects_count} new subjects")
        
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