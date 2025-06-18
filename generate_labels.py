from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Define constants
DATA_DIR      = Path.home() / "reseng202500013-ndd-ml" / "data"
CSV_PATH      = DATA_DIR / "imaging_records.csv"
OUTPUT_TRAIN  = DATA_DIR / "train_labels.csv"
OUTPUT_VAL    = DATA_DIR / "val_labels.csv"
N_SUBJECTS    = 30

# Disease label mapping
label_map = {"CN": 0, "AD": 1, "PD": 2}

# Load all data
df = pd.read_csv(
    CSV_PATH,
    header=None,
    names=["subject_name", "site", "modality", "disease", "path"]
)

# Keep only MRI entries
df_mri = df[df["modality"] == "MRI"]

train_rows = []
val_rows   = []

for disease, label in label_map.items():
    # Filter to this disease AND MRI modality
    disease_df = df_mri[df_mri["disease"] == disease]
    
    subjects = disease_df["subject_name"].unique()
    if len(subjects) < N_SUBJECTS:
        raise ValueError(f"Not enough MRI subjects for {disease} (found {len(subjects)})")
    
    selected = pd.Series(subjects).sample(n=N_SUBJECTS, random_state=42)
    train, val = train_test_split(selected, test_size=0.2, random_state=42)
    
    train_rows += [(f"sub-{subj}", label) for subj in train]
    val_rows   += [(f"sub-{subj}", label) for subj in val]

# Build and save DataFrames
pd.DataFrame(train_rows, columns=["subject_name", "label"])\
  .to_csv(OUTPUT_TRAIN, index=False)
pd.DataFrame(val_rows, columns=["subject_name", "label"])\
  .to_csv(OUTPUT_VAL,   index=False)

print(f"[INFO] Written: {OUTPUT_TRAIN}")
print(f"[INFO] Written: {OUTPUT_VAL}")
