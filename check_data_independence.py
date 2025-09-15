#!/usr/bin/env python3
"""
Script to check for data leakage between train and validation sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def check_data_independence():
    # Load the PET labels
    csv_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/pet_labels.csv")
    df = pd.read_csv(csv_path)
    
    # Filter for labels [1, 2]
    filtered_df = df[df['label'].isin([1, 2])].copy()
    
    print(f"Total subjects: {len(filtered_df)}")
    print(f"Label distribution:")
    print(filtered_df['label'].value_counts().sort_index())
    print()
    
    # Simulate the balancing process
    print("=== BALANCING PROCESS ===")
    label_counts = filtered_df['label'].value_counts()
    minority_class = label_counts.idxmin()
    minority_count = label_counts.min()
    
    print(f"Minority class: {minority_class} with {minority_count} subjects")
    
    # Create balanced dataset
    balanced_data = []
    for label in [1, 2]:
        label_data = filtered_df[filtered_df['label'] == label]
        if label == minority_class:
            balanced_data.append(label_data)
        else:
            # Undersample majority class
            balanced_data.append(label_data.sample(n=minority_count, random_state=42))
    
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    print(f"Balanced dataset size: {len(balanced_df)}")
    print(f"Balanced distribution:")
    print(balanced_df['label'].value_counts().sort_index())
    print()
    
    # Simulate k-fold split
    print("=== K-FOLD SPLIT ANALYSIS ===")
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skfold.split(balanced_df, balanced_df['label'])):
        train_subjects = balanced_df.iloc[train_idx]['subject_id'].tolist()
        val_subjects = balanced_df.iloc[val_idx]['subject_id'].tolist()
        
        print(f"Fold {fold + 1}:")
        print(f"  Train subjects: {len(train_subjects)}")
        print(f"  Val subjects: {len(val_subjects)}")
        print(f"  Overlap: {len(set(train_subjects) & set(val_subjects))}")
        
        # Check if any subjects appear in both train and val
        overlap = set(train_subjects) & set(val_subjects)
        if overlap:
            print(f"  WARNING: Subjects in both train and val: {overlap}")
        else:
            print(f"  ✓ No overlap between train and val")
        print()

if __name__ == "__main__":
    check_data_independence() 