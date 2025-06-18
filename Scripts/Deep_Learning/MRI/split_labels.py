import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_labels(input_csv, train_csv, val_csv, test_size=0.2, random_state=42):
    """
    Split the labels CSV file into train and validation sets.
    
    Args:
        input_csv: Path to the input labels CSV file
        train_csv: Path to save the training labels CSV
        val_csv: Path to save the validation labels CSV
        test_size: Proportion of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility
    """
    # Load the labels
    df = pd.read_csv(input_csv)
    
    # Split the data
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']  # Ensure balanced split across classes
    )
    
    # Save the splits
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"Total subjects: {len(df)}")
    print(f"Training subjects: {len(train_df)}")
    print(f"Validation subjects: {len(val_df)}")
    
    # Print class distribution
    print("\nTraining set class distribution:")
    train_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_counts.items():
        print(f"  Label {label}: {count} subjects")
    
    print("\nValidation set class distribution:")
    val_counts = val_df['label'].value_counts().sort_index()
    for label, count in val_counts.items():
        print(f"  Label {label}: {count} subjects")

if __name__ == "__main__":
    # Define paths using absolute paths
    data_dir = Path.home() / "reseng202500013-ndd-ml" / "data"
    input_csv = data_dir / "mri_labels.csv"
    
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    
    # Create the split
    split_labels(input_csv, train_csv, val_csv) 