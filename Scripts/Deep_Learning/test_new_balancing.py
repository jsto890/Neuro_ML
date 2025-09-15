#!/usr/bin/env python3
"""
Test script to demonstrate the new data balancing strategy:
1. Undersample to balance classes (keeping track of removed subjects)
2. Split balanced dataset into 70/20/10 (train/val/test)
3. Add removed subjects to test set
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def balance_and_split_dataset(df, val_ratio=0.2, test_ratio=0.1, random_state=None):
    """
    New data balancing strategy:
    1. Undersample to balance classes (keeping track of removed subjects)
    2. Split balanced dataset into 70/20/10 (train/val/test)
    3. Add removed subjects to test set
    
    Args:
        df: DataFrame with 'subject_id' and 'label' columns
        val_ratio: Proportion of balanced data for validation (default: 0.2)
        test_ratio: Proportion of balanced data for test (default: 0.1)
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (train_df, val_df, test_df, removed_subjects_df)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get class counts
    class_counts = df['label'].value_counts()
    min_count = class_counts.min()
    
    print(f"Original class distribution:")
    for label, count in class_counts.items():
        print(f"  Label {label}: {count} subjects")
    
    balanced_dfs = []
    removed_subjects_dfs = []
    
    for label in df['label'].unique():
        class_df = df[df['label'] == label].copy()
        class_count = len(class_df)
        
        # Reduce to minimum count (undersample majority classes)
        if class_count > min_count:
            # Sample the subjects to keep
            kept_subjects = class_df.sample(n=min_count, random_state=random_state)
            # Get the removed subjects
            removed_subjects = class_df.drop(kept_subjects.index)
            
            balanced_dfs.append(kept_subjects)
            removed_subjects_dfs.append(removed_subjects)
        else:
            # No undersampling needed for this class
            balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    removed_subjects_df = pd.concat(removed_subjects_dfs, ignore_index=True) if removed_subjects_dfs else pd.DataFrame(columns=df.columns)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nBalanced class distribution (undersampled):")
    balanced_counts = balanced_df['label'].value_counts()
    for label, count in balanced_counts.items():
        print(f"  Label {label}: {count} subjects")
    
    print(f"Removed subjects: {len(removed_subjects_df)} subjects")
    if len(removed_subjects_df) > 0:
        removed_counts = removed_subjects_df['label'].value_counts()
        for label, count in removed_counts.items():
            print(f"  Label {label}: {count} subjects")
    
    # Split balanced dataset into train/val/test (70/20/10)
    print(f"\nSplitting balanced dataset (train: {1-val_ratio-test_ratio:.1%}, val: {val_ratio:.1%}, test: {test_ratio:.1%})")
    
    # First split: train+val vs test
    train_val, test_balanced = train_test_split(
        balanced_df, 
        test_size=test_ratio, 
        stratify=balanced_df['label'], 
        random_state=random_state
    )
    
    # Second split: train vs val
    val_relative_size = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train_val, 
        test_size=val_relative_size, 
        stratify=train_val['label'], 
        random_state=random_state
    )
    
    # Add removed subjects to test set
    test_final = pd.concat([test_balanced, removed_subjects_df], ignore_index=True)
    
    print(f"\nFinal dataset splits:")
    print(f"Training set: {len(train)} subjects")
    print(f"Validation set: {len(val)} subjects")
    print(f"Test set: {len(test_final)} subjects (balanced: {len(test_balanced)}, added: {len(removed_subjects_df)})")
    
    return train, val, test_final, removed_subjects_df

def main():
    """Test the new balancing strategy with synthetic data."""
    
    # Create synthetic dataset with imbalanced classes
    np.random.seed(42)
    
    # Simulate imbalanced dataset: 100 CN, 200 AD, 150 PD
    cn_subjects = [f"CN_{i:03d}" for i in range(100)]
    ad_subjects = [f"AD_{i:03d}" for i in range(200)]
    pd_subjects = [f"PD_{i:03d}" for i in range(150)]
    
    # Create DataFrame
    data = []
    for subject in cn_subjects:
        data.append({'subject_id': subject, 'label': 0})
    for subject in ad_subjects:
        data.append({'subject_id': subject, 'label': 1})
    for subject in pd_subjects:
        data.append({'subject_id': subject, 'label': 2})
    
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("TESTING NEW DATA BALANCING STRATEGY")
    print("=" * 60)
    print(f"Total subjects: {len(df)}")
    
    # Test the new balancing strategy
    train, val, test, removed = balance_and_split_dataset(
        df, val_ratio=0.2, test_ratio=0.1, random_state=42
    )
    
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    
    # Show detailed breakdown
    print(f"\nTraining set ({len(train)} subjects):")
    train_counts = train['label'].value_counts().sort_index()
    for label, count in train_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(train)*100:.1f}%)")
    
    print(f"\nValidation set ({len(val)} subjects):")
    val_counts = val['label'].value_counts().sort_index()
    for label, count in val_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(val)*100:.1f}%)")
    
    print(f"\nTest set ({len(test)} subjects):")
    test_counts = test['label'].value_counts().sort_index()
    for label, count in test_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(test)*100:.1f}%)")
    
    print(f"\nRemoved subjects ({len(removed)} subjects):")
    if len(removed) > 0:
        removed_counts = removed['label'].value_counts().sort_index()
        for label, count in removed_counts.items():
            print(f"  Label {label}: {count} subjects ({count/len(removed)*100:.1f}%)")
    else:
        print("  No subjects were removed (dataset was already balanced)")
    
    # Verify that all subjects are accounted for
    total_accounted = len(train) + len(val) + len(test)
    print(f"\nVerification:")
    print(f"  Original total: {len(df)}")
    print(f"  Accounted for: {total_accounted}")
    print(f"  Match: {len(df) == total_accounted}")
    
    # Show the split ratios
    print(f"\nSplit ratios:")
    print(f"  Train: {len(train)/len(df)*100:.1f}%")
    print(f"  Val:   {len(val)/len(df)*100:.1f}%")
    print(f"  Test:  {len(test)/len(df)*100:.1f}%")

if __name__ == "__main__":
    main() 