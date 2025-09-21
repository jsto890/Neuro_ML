#!/usr/bin/env python3
"""
Script to regenerate the temporary CSV files for a specific fold using the same random seed.
This is useful when you need to evaluate a specific fold but the temporary CSV files are missing.
"""

import os
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split

def regenerate_fold_csv(master_csv, run_timestamp, model_name, fold_num, 
                       data_dir, random_seed=42, val_ratio=0.2, labels=None):
    """
    Regenerate the temporary CSV files for a specific fold.
    
    Args:
        master_csv: Path to the master CSV file
        run_timestamp: Timestamp of the run (e.g., "20250919_150821")
        model_name: Name of the model (e.g., "EfficientNetB0_3D")
        fold_num: Fold number to regenerate (1-5)
        data_dir: Directory where temporary CSV files are stored
        random_seed: Random seed used in original training
        val_ratio: Validation ratio used in original training
        labels: List of labels to filter (e.g., [0, 1, 2])
    """
    
    print(f"Regenerating CSV files for {model_name}, fold {fold_num}")
    print(f"Master CSV: {master_csv}")
    print(f"Run timestamp: {run_timestamp}")
    print(f"Random seed: {random_seed}")
    
    # Load master dataset
    print("Loading master dataset...")
    master_df = pd.read_csv(master_csv)
    print(f"Master dataset: {len(master_df)} total subjects")
    
    # Filter for specific labels if provided
    if labels is not None:
        filtered_df = master_df[master_df['label'].isin(labels)].copy()
        print(f"After filtering for labels {labels}: {len(filtered_df)} subjects")
        for label in labels:
            count = len(filtered_df[filtered_df['label'] == label])
            print(f"  Label {label}: {count} subjects ({count/len(filtered_df)*100:.1f}%)")
    else:
        filtered_df = master_df.copy()
    
    # Check if dataset balancing was used (you may need to adjust this)
    # For now, we'll assume no balancing was used
    dataset_for_cv = filtered_df
    
    # Generate the same k-fold splits as in the original training
    print(f"Generating {5}-fold cross-validation splits...")
    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    outer_splits = list(outer_skf.split(range(len(dataset_for_cv)), dataset_for_cv['label']))
    
    # Get the specific fold
    if fold_num < 1 or fold_num > 5:
        raise ValueError(f"Fold number must be between 1 and 5, got {fold_num}")
    
    train_pool_idx, test_idx = outer_splits[fold_num - 1]  # Convert to 0-based index
    
    # Get fold-specific TrainPool and Test
    test_df = dataset_for_cv.iloc[test_idx].copy()
    train_pool_df = dataset_for_cv.iloc[train_pool_idx].copy()
    
    print(f"Fold {fold_num}: TrainPool = {len(train_pool_df)}, Test = {len(test_df)}")
    
    # Apply dataset balancing if it was used in original training
    # For EfficientNet, check if balance_dataset was used
    # You may need to adjust this based on your original command
    balanced_train_pool_df = train_pool_df  # Assume no balancing for now
    
    # Train/Val split on balanced TrainPool (stratified)
    train_df, val_df = train_test_split(
        balanced_train_pool_df,
        test_size=val_ratio,
        stratify=balanced_train_pool_df['label'],
        random_state=random_seed,
    )
    
    print(f"Train/Val split: Train = {len(train_df)}, Val = {len(val_df)}")
    
    # Generate the same filenames as in the original training
    fold_tag = f"run_{run_timestamp}_{model_name}_fold_{fold_num}"
    temp_train_csv = os.path.join(data_dir, f"temp_train_{fold_tag}.csv")
    temp_val_csv = os.path.join(data_dir, f"temp_val_{fold_tag}.csv")
    temp_test_csv = os.path.join(data_dir, f"temp_test_{fold_tag}.csv")
    
    # Save the CSV files
    print(f"Saving CSV files:")
    print(f"  Train: {temp_train_csv}")
    print(f"  Val: {temp_val_csv}")
    print(f"  Test: {temp_test_csv}")
    
    train_df.to_csv(temp_train_csv, index=False)
    val_df.to_csv(temp_val_csv, index=False)
    test_df.to_csv(temp_test_csv, index=False)
    
    print("✅ CSV files regenerated successfully!")
    
    return temp_train_csv, temp_val_csv, temp_test_csv

def main():
    parser = argparse.ArgumentParser(description="Regenerate temporary CSV files for a specific fold")
    parser.add_argument("--master_csv", type=str, required=True,
                        help="Path to master CSV file")
    parser.add_argument("--run_timestamp", type=str, required=True,
                        help="Run timestamp (e.g., 20250919_150821)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g., EfficientNetB0_3D)")
    parser.add_argument("--fold_num", type=int, required=True,
                        help="Fold number to regenerate (1-5)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory where temporary CSV files are stored")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed used in original training")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation ratio used in original training")
    parser.add_argument("--labels", nargs='+', type=int,
                        help="Labels to filter (e.g., 0 1 2)")
    
    args = parser.parse_args()
    
    regenerate_fold_csv(
        master_csv=args.master_csv,
        run_timestamp=args.run_timestamp,
        model_name=args.model_name,
        fold_num=args.fold_num,
        data_dir=args.data_dir,
        random_seed=args.random_seed,
        val_ratio=args.val_ratio,
        labels=args.labels
    )

if __name__ == "__main__":
    main()
