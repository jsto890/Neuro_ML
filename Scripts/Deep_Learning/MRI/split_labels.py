import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_labels(input_csv, train_csv, val_csv, test_csv, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split the labels CSV file into train, validation, and test sets.
    Args:
        input_csv: Path to the input labels CSV file
        train_csv: Path to save the training labels CSV
        val_csv: Path to save the validation labels CSV
        test_csv: Path to save the test labels CSV
        test_size: Proportion of data to use for test set (default: 0.15)
        val_size: Proportion of data to use for validation set (default: 0.15)
        random_state: Random seed for reproducibility
    """
    # Load the labels
    df = pd.read_csv(input_csv)

    # First split off the test set
    trainval_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    # Now split trainval into train and val
    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=trainval_df['label']
    )

    # Save the splits
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Total subjects: {len(df)}")
    print(f"Training subjects: {len(train_df)}")
    print(f"Validation subjects: {len(val_df)}")
    print(f"Test subjects: {len(test_df)}")

    # Print class distribution
    print("\nTraining set class distribution:")
    train_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_counts.items():
        print(f"  Label {label}: {count} subjects")

    print("\nValidation set class distribution:")
    val_counts = val_df['label'].value_counts().sort_index()
    for label, count in val_counts.items():
        print(f"  Label {label}: {count} subjects")

    print("\nTest set class distribution:")
    test_counts = test_df['label'].value_counts().sort_index()
    for label, count in test_counts.items():
        print(f"  Label {label}: {count} subjects")

if __name__ == "__main__":
    # Define paths using absolute paths
    data_dir = Path.home() / "reseng202500013-ndd-ml" / "data"
    input_csv = data_dir / "mri_labels.csv"
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"
    # Create the split
    split_labels(input_csv, train_csv, val_csv, test_csv) 