# scripts/train_smri.py

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import Sampler
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support, matthews_corrcoef
import csv
from datetime import datetime
import uuid
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import contextlib
import math
import random

from dataset import SMRIDataset
from models_smri import Simple3DCNN, get_3d_model
from evaluate_model import evaluate_model, calculate_metrics, create_evaluation_plots

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def compute_summary_stats(values):
    """Compute mean, std, 95% CI, min, max, range for a list of numeric values."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {'mean': 0.0, 'std': 0.0, 'ci95': 0.0, 'ci95_lower': 0.0, 'ci95_upper': 0.0, 'min': 0.0, 'max': 0.0, 'range': 0.0, 'n': 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    ci = 1.96 * se if n > 1 else 0.0
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    rng = max_v - min_v
    return {
        'mean': mean,
        'std': std,
        'ci95': ci,
        'ci95_lower': mean - ci,
        'ci95_upper': mean + ci,
        'min': min_v,
        'max': max_v,
        'range': rng,
        'n': int(n)
    }

def get_train_transform(args):
    """Return a callable that applies light 3D augmentations on tensors [1, D, H, W].
    Uses MONAI when available for spatial ops; otherwise falls back to intensity-only jitters.
    """
    transforms = []

    # Try MONAI for spatial transforms
    try:
        from monai.transforms import Compose, RandAffine, RandBiasField, RandGaussianNoise, RandAdjustContrast

        def monai_transform(x: torch.Tensor) -> torch.Tensor:
            # x: [1, D, H, W] -> MONAI expects [C, H, W, D]
            x_m = x.permute(0, 2, 3, 1)
            t = Compose([
                RandAffine(prob=0.5, rotate_range=(
                    math.radians(5.0), math.radians(5.0), math.radians(5.0)
                ), scale_range=(0.05, 0.05, 0.05), mode="bilinear"),
                RandBiasField(prob=0.3, coeff_range=(0.0, 0.3)),
                RandAdjustContrast(prob=0.5, gamma=(0.9, 1.1)),
                RandGaussianNoise(prob=0.3, mean=0.0, std=0.02),
            ])
            x_m = t(x_m)
            x = x_m.permute(0, 3, 1, 2)
            return x

        transforms.append(monai_transform)
    except Exception:
        # Fall back to simple intensity-only transforms
        def intensity_jitter(x: torch.Tensor) -> torch.Tensor:
            # Brightness and contrast jitter (small)
            b = float(torch.empty(1).uniform_(-0.05, 0.05))
            c = float(torch.empty(1).uniform_(0.95, 1.05))
            x = x * c + b
            return x

        def gaussian_noise(x: torch.Tensor) -> torch.Tensor:
            std = 0.02
            noise = torch.randn_like(x) * std
            return x + noise

        transforms.extend([intensity_jitter, gaussian_noise])

    def apply_all(x: torch.Tensor) -> torch.Tensor:
        for t in transforms:
            x = t(x)
        return x

    return apply_all

class BalancedBatchSampler(Sampler):
    """Batch sampler that yields balanced batches across classes.
    - Ensures each batch contains batch_size // num_classes samples per class (with replacement).
    - Supports per-index hard mining weights to sample some indices more often.
    """
    def __init__(self, labels_list, batch_size: int, hard_index_weights=None):
        self.labels_list = list(labels_list)
        self.classes = sorted(list(set(self.labels_list)))
        self.num_classes = len(self.classes)
        self.k_per_class = max(1, batch_size // self.num_classes)
        self.effective_batch_size = self.k_per_class * self.num_classes
        self.batch_size = self.effective_batch_size
        self.hard_index_weights = hard_index_weights or {}

        # Build per-class indices
        self.indices_per_class = {c: [i for i, y in enumerate(self.labels_list) if y == c] for c in self.classes}

        # Define nominal epoch length in number of batches
        self.num_batches = max(1, len(self.labels_list) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            for c in self.classes:
                idxs = self.indices_per_class[c]
                if not idxs:
                    continue
                # Build sampling probabilities with hard-mining multipliers
                weights = np.array([float(self.hard_index_weights.get(i, 1.0)) for i in idxs], dtype=np.float64)
                if np.any(weights <= 0):
                    weights = np.clip(weights, 1e-8, None)
                probs = weights / weights.sum()
                chosen = np.random.choice(idxs, size=self.k_per_class, replace=True, p=probs)
                batch_indices.extend(chosen.tolist())
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches

def filter_labels(csv_path, labels):
    """Filter the CSV file to only include specified labels."""
    df = pd.read_csv(csv_path)
    filtered_df = df[df.iloc[:, 1].isin(labels)]
    return filtered_df

def balance_dataset(df, random_state=None):
    """
    Balance the dataset by reducing majority classes to match the minority class count.
    
    Args:
        df: DataFrame with 'subject_id' and 'label' columns
        random_state: Random seed for reproducibility
    
    Returns:
        Balanced DataFrame
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
    
    for label in df['label'].unique():
        class_df = df[df['label'] == label].copy()
        class_count = len(class_df)
        
        # Reduce to minimum count (undersample majority classes)
        if class_count > min_count:
            class_df = class_df.sample(n=min_count, random_state=random_state)
        
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the final dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nBalanced class distribution (undersampled):")
    balanced_counts = balanced_df['label'].value_counts()
    for label, count in balanced_counts.items():
        print(f"  Label {label}: {count} subjects")
    
    return balanced_df

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


def balance_and_split_dataset_90_10(df, test_ratio=0.1, random_state=None):
    """
    Balance dataset using undersampling and split into 90/10 train/test.
    No separate validation set - k-fold will be used on training set.
    
    Args:
        df: DataFrame with 'subject_id' and 'label' columns
        test_ratio: Proportion for test set (default: 0.1 for 90/10 split)
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, test_df, removed_subjects
    """
    print("Original class distribution:")
    for label in sorted(df['label'].unique()):
        count = len(df[df['label'] == label])
        print(f"  Label {label}: {count} subjects")
    
    # Find the minority class
    label_counts = df['label'].value_counts()
    minority_class = label_counts.idxmin()
    minority_count = label_counts.min()
    
    print(f"\nMinority class: {minority_class} with {minority_count} subjects")
    
    # Undersample majority classes to match minority class
    balanced_dfs = []
    removed_subjects = []
    
    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        if label == minority_class:
            # Keep all minority class samples
            balanced_dfs.append(class_df)
        else:
            # Undersample majority classes
            if len(class_df) > minority_count:
                # Randomly sample minority_count samples
                sampled_df = class_df.sample(n=minority_count, random_state=random_state)
                balanced_dfs.append(sampled_df)
                
                # Store removed subjects
                removed_df = class_df.drop(sampled_df.index)
                removed_subjects.append(removed_df)
            else:
                # If already balanced, keep as is
                balanced_dfs.append(class_df)
    
    # Combine balanced data
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"\nBalanced class distribution (undersampled):")
    for label in sorted(balanced_df['label'].unique()):
        count = len(balanced_df[balanced_df['label'] == label])
        print(f"  Label {label}: {count} subjects")
    
    # Combine all removed subjects
    if removed_subjects:
        all_removed = pd.concat(removed_subjects, ignore_index=True)
        print(f"\nRemoved subjects: {len(all_removed)} subjects")
        for label in sorted(all_removed['label'].unique()):
            count = len(all_removed[all_removed['label'] == label])
            print(f"  Label {label}: {count} subjects")
    else:
        all_removed = pd.DataFrame(columns=df.columns)
        print(f"\nRemoved subjects: 0 subjects")
    
    # Split balanced data into train/test (90/10)
    print(f"\nSplitting balanced dataset (train: 90.0%, test: 10.0%)")
    
    train_df, test_df = train_test_split(
        balanced_df,
        test_size=test_ratio,
        stratify=balanced_df['label'],
        random_state=random_state
    )
    
    # Add removed subjects to test set
    if len(all_removed) > 0:
        test_df = pd.concat([test_df, all_removed], ignore_index=True)
        print(f"Added {len(all_removed)} removed subjects to test set")
    
    print(f"\nFinal dataset splits:")
    print(f"Training set: {len(train_df)} subjects")
    print(f"Test set: {len(test_df)} subjects (balanced: {len(test_df) - len(all_removed)}, added: {len(all_removed)})")
    
    return train_df, test_df, all_removed


def get_label_description(labels):
    """Convert numeric labels to descriptive names."""
    label_map = {0: 'CN', 1: 'AD', 2: 'PD'}
    return ' vs '.join([label_map[label] for label in sorted(labels)])

def log_metrics(run_id, model_name, args, best_val_auc, best_val_acc, final_train_loss, final_train_acc, notes=""):
    """Log training metrics to CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get descriptive label names
    label_description = get_label_description(args.labels)
    
    # Prepare the row data
    row = {
        'run_id': run_id,
        'timestamp': timestamp,
        'model_name': model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'data_root': args.data_root,
        'checkpoint_dir': args.checkpoint_dir,
        'labels': label_description,
        'best_val_auc': best_val_auc,
        'best_val_acc': best_val_acc,
        'final_train_loss': final_train_loss,
        'final_train_acc': final_train_acc,
        'notes': notes
    }
    
    # Define the logging file path
    log_file_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/logging.csv")
    
    # Ensure the directory exists
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(log_file_path)
    
    with open(log_file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"Metrics logged to: {log_file_path}")

def create_training_plots(folds_data, output_dir="./deep_learning_plots", model_name="Model"):
    """Create comprehensive training plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive plot with more subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f'Deep Learning Training Results - {model_name} - AD vs CN Classification', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    for fold_data in folds_data:
        epochs = [d['epoch'] for d in fold_data['data']]
        losses = [d['train_loss'] for d in fold_data['data']]
        ax1.plot(epochs, losses, alpha=0.7, label=f"Fold {fold_data['fold']}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{model_name} - Training Loss by Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Accuracy
    ax2 = axes[0, 1]
    for fold_data in folds_data:
        epochs = [d['epoch'] for d in fold_data['data']]
        accs = [d['train_acc'] for d in fold_data['data']]
        ax2.plot(epochs, accs, alpha=0.7, label=f"Fold {fold_data['fold']}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy')
    ax2.set_title(f'{model_name} - Training Accuracy by Fold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation AUC
    ax3 = axes[0, 2]
    for fold_data in folds_data:
        epochs = [d['epoch'] for d in fold_data['data']]
        aucs = [d['val_auc'] for d in fold_data['data']]
        ax3.plot(epochs, aucs, alpha=0.7, label=f"Fold {fold_data['fold']}")
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation AUC')
    ax3.set_title(f'{model_name} - Validation AUC by Fold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Best AUC per fold
    ax4 = axes[1, 0]
    best_aucs = []
    fold_numbers = []
    for fold_data in folds_data:
        best_auc = max([d['val_auc'] for d in fold_data['data']])
        best_aucs.append(best_auc)
        fold_numbers.append(fold_data['fold'])
    
    bars = ax4.bar(fold_numbers, best_aucs, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Best Validation AUC')
    ax4.set_title(f'{model_name} - Best AUC per Fold')
    ax4.set_ylim(0.5, 1.0)
    
    # Add value labels on bars
    for bar, auc in zip(bars, best_aucs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom')
    
    ax4.grid(True, alpha=0.3)
    
    # 5. Final metrics comparison
    ax5 = axes[1, 1]
    final_metrics = []
    metric_names = []
    
    for fold_data in folds_data:
        final_epoch = fold_data['data'][-1]
        final_metrics.extend([
            final_epoch['val_auc'],
            final_epoch['val_acc'],
            final_epoch['train_acc']
        ])
        metric_names.extend(['Val AUC', 'Val Acc', 'Train Acc'])
    
    # Reshape for plotting
    metrics_array = np.array(final_metrics).reshape(len(folds_data), 3)
    
    x = np.arange(len(folds_data))
    width = 0.25
    
    ax5.bar(x - width, metrics_array[:, 0], width, label='Val AUC', alpha=0.8)
    ax5.bar(x, metrics_array[:, 1], width, label='Val Acc', alpha=0.8)
    ax5.bar(x + width, metrics_array[:, 2], width, label='Train Acc', alpha=0.8)
    
    ax5.set_xlabel('Fold')
    ax5.set_ylabel('Score')
    ax5.set_title(f'{model_name} - Final Metrics by Fold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(fold_numbers)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Training vs Validation Performance
    ax6 = axes[1, 2]
    final_train_accs = [fold_data['data'][-1]['train_acc'] for fold_data in folds_data]
    final_val_accs = [fold_data['data'][-1]['val_acc'] for fold_data in folds_data]
    
    ax6.scatter(final_train_accs, final_val_accs, s=100, alpha=0.7, c='green')
    ax6.plot([0.5, 1.0], [0.5, 1.0], 'k--', alpha=0.5)
    
    for i, fold_num in enumerate(fold_numbers):
        ax6.annotate(f'F{fold_num}', (final_train_accs[i], final_val_accs[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax6.set_xlabel('Final Training Accuracy')
    ax6.set_ylabel('Final Validation Accuracy')
    ax6.set_title(f'{model_name} - Training vs Validation Performance')
    ax6.grid(True, alpha=0.3)
    
    # 7. Precision, Recall, F1 by Fold
    ax7 = axes[2, 0]
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for fold_data in folds_data:
        final_epoch = fold_data['data'][-1]
        precision_scores.append(final_epoch['precision_macro'])
        recall_scores.append(final_epoch['recall_macro'])
        f1_scores.append(final_epoch['f1_macro'])
    
    x = np.arange(len(folds_data))
    width = 0.25
    
    ax7.bar(x - width, precision_scores, width, label='Precision', alpha=0.8, color='lightcoral')
    ax7.bar(x, recall_scores, width, label='Recall', alpha=0.8, color='lightblue')
    ax7.bar(x + width, f1_scores, width, label='F1', alpha=0.8, color='lightgreen')
    
    ax7.set_xlabel('Fold')
    ax7.set_ylabel('Score')
    ax7.set_title(f'{model_name} - Precision, Recall, F1 by Fold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(fold_numbers)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Confusion Matrix (average across folds)
    ax8 = axes[2, 1]
    avg_cm = np.zeros((2, 2))  # Assuming binary classification
    cm_count = 0
    
    for fold_data in folds_data:
        final_epoch = fold_data['data'][-1]
        if 'confusion_matrix' in final_epoch and final_epoch['confusion_matrix'] is not None:
            cm = np.array(final_epoch['confusion_matrix'])
            if cm.shape == (2, 2):  # Ensure it's 2x2
                avg_cm += cm
                cm_count += 1
    
    if cm_count > 0:
        avg_cm /= cm_count
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'],
                   ax=ax8)
        ax8.set_title(f'{model_name} - Average Confusion Matrix')
        ax8.set_xlabel('Predicted')
        ax8.set_ylabel('Actual')
    else:
        ax8.text(0.5, 0.5, 'No confusion matrix data', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title(f'{model_name} - Confusion Matrix (No Data)')
    
    # 9. Learning Rate over time
    ax9 = axes[2, 2]
    for fold_data in folds_data:
        epochs = [d['epoch'] for d in fold_data['data']]
        lrs = [d['lr'] for d in fold_data['data']]
        ax9.plot(epochs, lrs, alpha=0.7, label=f"Fold {fold_data['fold']}")
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Learning Rate')
    ax9.set_title(f'{model_name} - Learning Rate by Fold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path / f'{model_name}_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate MCC scores for each fold
    mcc_scores = []
    for fold_data in folds_data:
        final_epoch = fold_data['data'][-1]
        if 'val_mcc' in final_epoch:
            mcc_scores.append(final_epoch['val_mcc'])
        else:
            mcc_scores.append(0.0)  # Fallback if MCC not available
    
    # Create summary statistics
    summary = {
        'model_name': model_name,
        'total_folds': len(folds_data),
        'average_best_auc': np.mean(best_aucs),
        'std_best_auc': np.std(best_aucs),
        'min_best_auc': np.min(best_aucs),
        'max_best_auc': np.max(best_aucs),
        'average_precision_macro': np.mean(precision_scores),
        'average_recall_macro': np.mean(recall_scores),
        'average_f1_macro': np.mean(f1_scores),
        'average_mcc': np.mean(mcc_scores),
        'fold_results': []
    }
    
    for fold_data in folds_data:
        final_epoch = fold_data['data'][-1]
        fold_result = {
            'fold': fold_data['fold'],
            'epochs_trained': len(fold_data['data']),
            'best_val_auc': max([d['val_auc'] for d in fold_data['data']]),
            'final_val_auc': final_epoch['val_auc'],
            'final_val_acc': final_epoch['val_acc'],
            'final_train_acc': final_epoch['train_acc'],
            'final_precision_macro': final_epoch['precision_macro'],
            'final_recall_macro': final_epoch['recall_macro'],
            'final_f1_macro': final_epoch['f1_macro'],
            'final_mcc': final_epoch.get('val_mcc', 0.0)
        }
        summary['fold_results'].append(fold_result)
    
    # Save summary
    with open(output_path / f'{model_name}_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print(f"DEEP LEARNING TRAINING SUMMARY - {model_name}")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Total folds: {summary['total_folds']}")
    print(f"Average best AUC: {summary['average_best_auc']:.4f} ± {summary['std_best_auc']:.4f}")
    print(f"AUC range: {summary['min_best_auc']:.4f} - {summary['max_best_auc']:.4f}")
    print(f"Average Precision: {summary['average_precision_macro']:.4f}")
    print(f"Average Recall: {summary['average_recall_macro']:.4f}")
    print(f"Average F1: {summary['average_f1_macro']:.4f}")
    print(f"Average MCC: {summary['average_mcc']:.4f}")
    print("\nFOLD DETAILS:")
    for fold_result in summary['fold_results']:
        print(f"Fold {fold_result['fold']}: {fold_result['epochs_trained']} epochs, "
              f"Best AUC: {fold_result['best_val_auc']:.4f}, "
              f"Final Val Acc: {fold_result['final_val_acc']:.4f}, "
              f"F1: {fold_result['final_f1_macro']:.4f}, "
              f"MCC: {fold_result['final_mcc']:.4f}")
    print("="*60)
    
    return summary

def create_test_summary_plots(fold_test_metrics, output_dir="./deep_learning_plots", model_name="Model", classification_description=None):
    """Create comprehensive test summary plots aggregated across folds and save JSON stats.

    fold_test_metrics: list of dicts with keys {'fold': int, 'metrics': dict}
    The metrics dict should contain: 'accuracy', 'precision', 'recall', 'f1_score', 'auc', 'mcc', 'confusion_matrix'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not fold_test_metrics:
        print(f"No test metrics provided for {model_name}; skipping test summary plots.")
        return None

    folds = [d['fold'] for d in fold_test_metrics]
    accs = [d['metrics']['accuracy'] for d in fold_test_metrics]
    precs = [d['metrics']['precision'] for d in fold_test_metrics]
    recalls = [d['metrics']['recall'] for d in fold_test_metrics]
    f1s = [d['metrics']['f1_score'] for d in fold_test_metrics]
    aucs = [d['metrics']['auc'] for d in fold_test_metrics]
    mccs = [d['metrics']['mcc'] for d in fold_test_metrics]
    thresholds = [d['threshold_used'] for d in fold_test_metrics if d.get('threshold_used') is not None]

    # Average confusion matrix
    cms = [np.array(d['metrics']['confusion_matrix']) for d in fold_test_metrics if 'confusion_matrix' in d['metrics'] and d['metrics']['confusion_matrix'] is not None]
    avg_cm = None
    if cms:
        try:
            avg_cm = np.mean(np.stack(cms, axis=0), axis=0)
        except Exception:
            # Fallback: sum then divide
            avg_cm = sum(cms) / len(cms)

    # Create plots (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    title_suffix = f" - {classification_description} Classification" if classification_description else ""
    fig.suptitle(f'Test Performance Summary - {model_name}{title_suffix}', fontsize=16, fontweight='bold')

    # 1. Test AUC by Fold
    ax1 = axes[0, 0]
    bars = ax1.bar(folds, aucs, alpha=0.8, color='skyblue', edgecolor='black')
    for bar, v in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('AUC')
    ax1.set_ylim(0, 1)
    ax1.set_title('Test AUC by Fold')
    ax1.grid(True, alpha=0.3)

    # 2. Test Accuracy by Fold
    ax2 = axes[0, 1]
    bars = ax2.bar(folds, accs, alpha=0.8, color='lightcoral', edgecolor='black')
    for bar, v in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.set_title('Test Accuracy by Fold')
    ax2.grid(True, alpha=0.3)

    # 3. Test F1-Score by Fold
    ax3 = axes[0, 2]
    bars = ax3.bar(folds, f1s, alpha=0.8, color='orange', edgecolor='black')
    for bar, v in zip(bars, f1s):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('F1-Score')
    ax3.set_ylim(0, 1)
    ax3.set_title('Test F1-Score by Fold')
    ax3.grid(True, alpha=0.3)

    # 4. Test Precision and Recall by Fold (grouped)
    ax4 = axes[1, 0]
    x = np.arange(len(folds))
    width = 0.35
    ax4.bar(x - width/2, precs, width, label='Precision', alpha=0.8, color='lightgreen', edgecolor='black')
    ax4.bar(x + width/2, recalls, width, label='Recall', alpha=0.8, color='lightblue', edgecolor='black')
    ax4.set_xticks(x)
    ax4.set_xticklabels(folds)
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1)
    ax4.set_title('Test Precision and Recall by Fold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Average Confusion Matrix
    ax5 = axes[1, 1]
    if avg_cm is not None:
        # Determine labels based on confusion matrix shape
        n_classes = avg_cm.shape[0]
        if n_classes == 2:
            class_labels = ['CN', 'AD']
        elif n_classes == 3:
            class_labels = ['CN', 'AD', 'PD']
        else:
            class_labels = [f'Class {i}' for i in range(n_classes)]
        
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax5)
        ax5.set_title('Average Confusion Matrix (Test)')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
    else:
        ax5.text(0.5, 0.5, 'No confusion matrix data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Confusion Matrix (No Data)')

    # 6. Mean ± 95% CI for all metrics
    ax6 = axes[1, 2]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']
    metric_values = [accs, precs, recalls, f1s, aucs, mccs]
    stats = [compute_summary_stats(vals) for vals in metric_values]
    means = [s['mean'] for s in stats]
    cis = [s['ci95'] for s in stats]
    bars = ax6.bar(metric_names, means, yerr=cis, capsize=4, alpha=0.8, color='mediumpurple', edgecolor='black')
    ax6.set_ylim(0, 1)
    ax6.set_ylabel('Score')
    ax6.set_title('Test Metrics: Mean ± 95% CI')
    for bar, mean in zip(bars, means):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{mean:.3f}', ha='center', va='bottom')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f'{model_name}_test_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Build summary JSON
    summary = {
        'model_name': model_name,
        'total_folds': len(folds),
        'metrics': {
            'accuracy': compute_summary_stats(accs),
            'precision': compute_summary_stats(precs),
            'recall': compute_summary_stats(recalls),
            'f1_score': compute_summary_stats(f1s),
            'auc': compute_summary_stats(aucs),
            'mcc': compute_summary_stats(mccs)
        },
        'average_confusion_matrix': (avg_cm.tolist() if avg_cm is not None else None),
        'parameters': {
            'threshold': (compute_summary_stats(thresholds) if thresholds else None)
        }
    }

    with open(output_path / f'{model_name}_test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print concise summary
    print("\n" + "="*60)
    print(f"DEEP LEARNING TEST SUMMARY - {model_name}")
    print("="*60)
    for name, stat in summary['metrics'].items():
        print(f"{name.capitalize():<10}: {stat['mean']:.4f}  (95% CI ± {stat['ci95']:.4f}; range {stat['min']:.4f}-{stat['max']:.4f})")
    print("="*60)

    return summary

def create_threshold_optimization_plot(threshold_results, output_dir, model_name="Model", fold_num=1):
    """
    Create plots showing threshold optimization results.
    
    Args:
        threshold_results: list of threshold results from optimize_threshold
        output_dir: directory to save plots
        model_name: name of the model
        fold_num: fold number
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    thresholds = [r['threshold'] for r in threshold_results]
    accuracies = [r['accuracy'] for r in threshold_results]
    
    # Find best threshold
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy vs Threshold
    ax1.plot(thresholds, accuracies, 'b-', linewidth=2, alpha=0.7)
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Default (0.5)')
    ax1.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7, label=f'Optimal ({best_threshold:.3f})')
    ax1.scatter(best_threshold, best_accuracy, color='green', s=100, zorder=5, label=f'Best: {best_accuracy:.4f}')
    ax1.scatter(0.5, accuracies[thresholds.index(0.5)], color='red', s=100, zorder=5, label=f'Default: {accuracies[thresholds.index(0.5)]:.4f}')
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{model_name} - Threshold Optimization (Fold {fold_num})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.1, 0.9)
    
    # Plot 2: Accuracy improvement distribution
    default_acc = accuracies[thresholds.index(0.5)]
    improvements = [acc - default_acc for acc in accuracies]
    
    ax2.plot(thresholds, improvements, 'orange', linewidth=2, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Default (0.5)')
    ax2.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7, label=f'Optimal ({best_threshold:.3f})')
    ax2.scatter(best_threshold, max(improvements), color='green', s=100, zorder=5, label=f'Max improvement: {max(improvements):.4f}')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Accuracy Improvement')
    ax2.set_title(f'{model_name} - Accuracy Improvement vs Threshold (Fold {fold_num})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.1, 0.9)
    
    plt.tight_layout()
    plot_path = output_path / f'{model_name}_threshold_optimization_fold_{fold_num}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Threshold optimization plot saved to: {plot_path}")
    
    return {
        'best_threshold': best_threshold,
        'best_accuracy': best_accuracy,
        'default_accuracy': default_acc,
        'improvement': max(improvements),
        'plot_path': str(plot_path)
    }

def train_sMRI_model(model, train_loader, val_loader, epochs, device, checkpoint_dir, args, fold_num=None, label_mapping=None):
    """
    Trains model; saves best checkpoint by validation AUC into checkpoint_dir.
    Returns the model loaded with best weights and training history.
    """
    # Calculate class counts from dataset to avoid sampling duplication bias
    class_counts_map = {}
    try:
        # Prefer dataset-level labels
        ds_labels = train_loader.dataset.df['label'].to_numpy().tolist()
    except Exception:
        # Fallback: iterate dataset
        ds_labels = []
        ds_obj = train_loader.dataset
        for i in range(len(ds_obj)):
            item = ds_obj[i]
            if isinstance(item, (list, tuple)) and len(item) == 3:
                _, y, _ = item
            else:
                _, y = item
            ds_labels.append(int(y if isinstance(y, int) else (y.item() if hasattr(y, 'item') else int(y))))

    for y in ds_labels:
        class_counts_map[y] = class_counts_map.get(y, 0) + 1

    # If a label mapping is provided, reorder counts to mapped class indices [0..K-1]
    if label_mapping is not None:
        # Build inverse mapping: mapped_id -> original_label
        inv_map = {v: k for k, v in label_mapping.items()}
        num_classes = len(inv_map)
        class_counts = np.array([class_counts_map.get(inv_map[i], 0) for i in range(num_classes)], dtype=np.int64)
    else:
        # Use natural order over sorted unique labels
        unique_sorted = sorted(class_counts_map.keys())
        class_counts = np.array([class_counts_map[l] for l in unique_sorted], dtype=np.int64)
        num_classes = len(class_counts)
    
    # Avoid division by zero if any class absent (should be rare)
    safe_counts = np.where(class_counts == 0, 1, class_counts)
    inv_freq = 1.0 / safe_counts.astype(np.float64)
    class_weights = inv_freq / inv_freq.sum()
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Check if this is a Vision Transformer model
    is_vit_model = any(name.lower() in str(type(model)).lower() for name in 
                       ['visiontransformer3d', 'swinunetrclassifier', 'fullswinunetrclassifier'])
    
    # Use appropriate loss function based on model type
    if is_vit_model:
        # CrossEntropy for ViT (with label smoothing)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"[INFO] Using CrossEntropyLoss with label smoothing {args.label_smoothing} for ViT model")
    else:
        if num_classes > 2:
            # Multiclass CNN: Class-Balanced Focal Loss using effective number of samples
            class ClassBalancedFocalLoss(nn.Module):
                def __init__(self, samples_per_class, beta: float = 0.999, gamma: float = 2.0):
                    super().__init__()
                    self.gamma = float(gamma)
                    self.beta = float(beta)
                    spc = torch.tensor(samples_per_class, dtype=torch.float32)
                    effective_num = 1.0 - torch.pow(self.beta, spc)
                    weights = (1.0 - self.beta) / (effective_num + 1e-8)
                    weights = weights / weights.sum() * len(spc)
                    self.register_buffer('class_weights', weights)
                
                def to(self, device):
                    super().to(device)
                    if hasattr(self, 'class_weights'):
                        self.class_weights = self.class_weights.to(device)
                    return self
                def forward(self, logits, targets):
                    ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')(logits, targets)
                    pt = torch.exp(-ce)
                    focal = ((1 - pt) ** self.gamma) * ce
                    return focal.mean()
            samples_per_class = class_counts
            criterion = ClassBalancedFocalLoss(samples_per_class=samples_per_class, beta=args.cb_beta, gamma=args.focal_gamma)
            criterion = criterion.to(device)  # Move criterion to device
            print(f"[INFO] Using Class-Balanced Focal Loss (beta={args.cb_beta}, gamma={args.focal_gamma}) for multiclass CNN")
        else:
            # Binary CNN: Focal loss
            class FocalLoss(nn.Module):
                def __init__(self, alpha=0.25, gamma=2.0):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma

                def forward(self, inputs, targets):
                    ce = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
                    pt = torch.exp(-ce)
                    loss = (self.alpha * (1 - pt) ** self.gamma * ce).mean()
                    return loss
            criterion = FocalLoss(alpha=0.25, gamma=args.focal_gamma)
            print(f"[INFO] Using FocalLoss for binary CNN")

    model.to(device)
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_state = None
    final_train_loss = 0.0
    final_train_acc = 0.0
    no_improvement_count = 0
    
    # Early stopping variables
    early_stopping_patience = getattr(args, 'early_stopping_patience', 30)
    early_stopping_min_delta = getattr(args, 'early_stopping_min_delta', 0.001)
    early_stopping_monitor = getattr(args, 'early_stopping_monitor', 'val_auc')
    best_monitored_metric = 0.0
    early_stopping_count = 0
    
    print(f"[INFO] Early stopping enabled: patience={early_stopping_patience}, monitor={early_stopping_monitor}, min_delta={early_stopping_min_delta}")
    
    # Store best metrics
    best_precision_macro = 0.0
    best_recall_macro = 0.0
    best_f1_macro = 0.0
    best_class_metrics = {}
    best_confusion_matrix = None
    
    # Store training history
    training_history = []
    
    # Initialize mixed precision training with compatibility across torch versions
    autocast_context = None
    if device.startswith('cuda'):
        try:
            scaler = torch.amp.GradScaler()
            autocast_context = lambda: torch.amp.autocast(device_type='cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
            autocast_context = lambda: torch.cuda.amp.autocast()
    else:
        scaler = None
        autocast_context = contextlib.nullcontext

    # Warm-up forward pass to initialize any dynamic layers (e.g., Simple3DCNN classifier)
    # Run in eval mode to avoid affecting BatchNorm/Dropout statistics
    # This ensures optimizer/EMA capture the correct parameter shapes
    model.eval()
    with torch.no_grad():
        try:
            sample_smri, _ = next(iter(train_loader))
            sample_smri = sample_smri.to(device)
            _ = model(sample_smri)
        except StopIteration:
            pass
        except Exception:
            # If warm-up fails for any reason, continue; model may not require it
            pass
    # Ensure we start epoch loop in train mode
    model.train()

    # Configure optimizer and scheduler based on model type
    if is_vit_model:
        # Vision Transformer optimizations
        if args.vit_optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=args.learning_rate, 
                weight_decay=args.vit_weight_decay
            )
            print(f"[INFO] Using AdamW optimizer with weight decay {args.vit_weight_decay} for ViT model")
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=args.learning_rate, 
                weight_decay=args.vit_weight_decay
            )
            print(f"[INFO] Using Adam optimizer with weight decay {args.vit_weight_decay} for ViT model")
        
        # Cosine schedule with warmup for ViT models (epoch-based)
        if args.vit_use_cosine_schedule:
            total_epochs = epochs
            warmup_epochs = args.vit_warmup_epochs

            def lr_lambda(current_epoch: int):
                # Linear warmup for the first warmup_epochs
                if current_epoch < warmup_epochs:
                    return float(current_epoch + 1) / float(max(1, warmup_epochs))
                # Cosine decay for the remaining epochs
                progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            print(f"[INFO] Using cosine schedule (epoch-based) with {args.vit_warmup_epochs} epoch warmup for ViT model")
        else:
            # Fallback to plateau scheduler for ViT
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=args.lr_scheduler_factor,
                patience=args.lr_scheduler_patience, threshold=1e-4, min_lr=1e-5
            )
            print(f"[INFO] Using plateau scheduler for ViT model")
    else:
        # CNN optimizer/scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        print(f"[INFO] Using AdamW optimizer with weight decay {args.weight_decay} for CNN model")
        # Plateau LR scheduler on validation AUC for CNN models
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=args.lr_scheduler_factor,
                patience=args.lr_scheduler_patience,
                threshold=1e-4,
                min_lr=1e-5,
                verbose=False,
            )
        except TypeError:
            try:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=args.lr_scheduler_factor,
                    patience=args.lr_scheduler_patience,
                    threshold=1e-4,
                    min_lr=1e-5,
                )
            except TypeError:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=args.lr_scheduler_factor,
                    patience=args.lr_scheduler_patience,
                )
    
    current_eval_weights = 'raw'  # track which weights are used for eval

    # --- EMA setup (disabled to match old behavior) ---
    use_ema = False
    ema_decay = 0.999
    ema_params = [p.detach().clone() for p in model.parameters()] if use_ema else []

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        # Hard-example tracking (multiclass only)
        hard_indices_epoch = []

        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                smri, labels, batch_indices = batch
            else:
                smri, labels = batch
                batch_indices = None
            smri, labels = smri.to(device), labels.to(device)
            
            # Apply label mapping if provided
            if label_mapping is not None:
                labels = torch.tensor([label_mapping[label.item()] for label in labels], device=device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training if available
            if scaler is not None:
                with autocast_context():
                    logits = model(smri)              # [B, 2]
                    loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                
                # Apply gradient clipping for ViT models
                if is_vit_model and args.grad_clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_max_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU
                logits = model(smri)              # [B, 2]
                loss = criterion(logits, labels)
                
                loss.backward()
                
                # Apply gradient clipping for ViT models
                if is_vit_model and args.grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_max_norm)
                
                optimizer.step()

            # Update EMA after optimizer step (disabled)
            if use_ema:
                with torch.no_grad():
                    params_list = list(model.parameters())
                    need_reset_ema = False
                    if len(ema_params) != len(params_list):
                        need_reset_ema = True
                    else:
                        for p, e in zip(params_list, ema_params):
                            if e.shape != p.data.shape:
                                need_reset_ema = True
                                break
                    if need_reset_ema:
                        ema_params = [p.detach().clone() for p in params_list]
                    else:
                        for p, e in zip(params_list, ema_params):
                            e.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

            running_loss += loss.item() * smri.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += (preds == labels).sum().item()
            total_samples += smri.size(0)

            # Collect hard examples for PD<->CN confusion emphasis
            if batch_indices is not None and logits.size(1) > 2:
                # Map labels back to original class ids if mapping provided
                if label_mapping is not None:
                    inv_map = {v: k for k, v in label_mapping.items()}
                    true_cls = torch.tensor([inv_map[int(x)] for x in labels.cpu().tolist()])
                else:
                    true_cls = labels.detach().cpu()
                pred_cls = preds.detach().cpu()
                batch_indices_cpu = batch_indices if isinstance(batch_indices, torch.Tensor) else torch.tensor(batch_indices)
                # Identify CN<->PD confusions (assuming 1=CN, 2=PD per labeling doc)
                mask_pdcn = (
                    ((true_cls == 1) & (pred_cls == 2)) | ((true_cls == 2) & (pred_cls == 1))
                )
                if mask_pdcn.any():
                    hard_indices_epoch.extend(batch_indices_cpu[mask_pdcn].tolist())
            
            # Memory optimization: clear cache periodically
            if args.memory_efficient and total_samples % (args.batch_size * 10) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Clear memory after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_loss = running_loss / total_samples
        epoch_acc  = running_corrects / total_samples
        
        if epoch == epochs:
            final_train_loss = epoch_loss
            final_train_acc = epoch_acc

        # --- Validation phase with threshold optimization ---
        # Evaluate with EMA weights for smoother performance
        if use_ema:
            raw_params = [p.detach().clone() for p in model.parameters()]
            with torch.no_grad():
                for p, e in zip(model.parameters(), ema_params):
                    p.copy_(e)
            current_eval_weights = 'ema'
        val_results = evaluate_model_with_threshold_optimization(model, val_loader, device, optimize_threshold_flag=args.optimize_threshold, label_mapping=label_mapping)
        # Restore raw weights after eval
        if use_ema:
            with torch.no_grad():
                for p, r in zip(model.parameters(), raw_params):
                    p.copy_(r)
            current_eval_weights = 'raw'
        
        val_auc = val_results['default_auc']
        val_acc = val_results['optimal_accuracy']  # Use optimized accuracy
        probs = val_results['probabilities']
        val_labels = val_results['labels']
        optimal_threshold = val_results['optimal_threshold']
        
        # Debug: print shapes and types
        print(f"[DEBUG] probs shape: {probs.shape}, type: {type(probs)}")
        print(f"[DEBUG] val_labels shape: {val_labels.shape}, type: {type(val_labels)}")
        print(f"[DEBUG] optimal_threshold: {optimal_threshold}")
        
        # Calculate additional metrics using optimal threshold (binary) or argmax (multiclass)
        if optimal_threshold is not None and probs.shape[1] == 2:
            # Binary classification with threshold optimization
            optimal_preds = (probs >= optimal_threshold).astype(int)
        else:
            # Multiclass or no threshold: use argmax
            optimal_preds = np.argmax(probs, axis=1)
        
        precision, recall, f1, support = precision_recall_fscore_support(val_labels, optimal_preds, average=None, zero_division=0)
        cm = confusion_matrix(val_labels, optimal_preds)
        
        # Calculate macro averages for multi-class
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(val_labels, optimal_preds, average='macro', zero_division=0)
        
        # Store per-class metrics
        class_metrics = {}
        for i, class_label in enumerate(sorted(set(val_labels))):
            class_metrics[f'class_{class_label}'] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

        # Step scheduler (handle both LambdaLR and ReduceLROnPlateau robustly)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_auc)
            new_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
        
        # Store epoch data
        val_mcc = matthews_corrcoef(val_labels, optimal_preds)
        epoch_data = {
            'epoch': int(epoch),
            'train_loss': float(epoch_loss),
            'train_acc': float(epoch_acc),
            'val_auc': float(val_auc),
            'val_acc': float(val_acc),
            'optimal_threshold': float(optimal_threshold) if optimal_threshold is not None else None,
            'accuracy_improvement': float(val_results.get('accuracy_improvement', 0.0)),
            'lr': float(new_lr),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'val_mcc': float(val_mcc)
        }
        training_history.append(epoch_data)

        # --- Update hard-mining weights for BalancedBatchSampler (next epoch) ---
        sampler_obj = None
        if hasattr(train_loader, 'batch_sampler') and isinstance(train_loader.batch_sampler, BalancedBatchSampler):
            sampler_obj = train_loader.batch_sampler
        elif hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, BalancedBatchSampler):
            sampler_obj = train_loader.sampler
        if sampler_obj is not None and num_classes > 2:
            # Increase sampling weight for CN<->PD confused indices
            hard_weight = getattr(args, 'hard_mining_weight', 3.0)
            # Reset weights but keep previous if any
            new_weights = {}
            for i in getattr(sampler_obj, 'hard_index_weights', {}).keys():
                new_weights[i] = 1.0
            for i in hard_indices_epoch:
                new_weights[i] = hard_weight
            sampler_obj.hard_index_weights = new_weights
        
        # Print learning rate (constant if no scheduler)
        lr_change = ""

        print(f"Epoch {epoch}/{epochs}  "
              f"Train loss={epoch_loss:.4f}, Train acc={epoch_acc:.4f}  "
              f"Val AUC={val_auc:.4f}, Val acc={val_acc:.4f}  "
              f"LR={new_lr:.6f}{lr_change}")

        # Checkpoint if this is the best AUC so far
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            # Deep copy state dict to avoid mutation by future training steps
            import copy
            best_state = copy.deepcopy(model.state_dict())
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model with fold-specific filename
            if fold_num is not None:
                model_filename = f"best_smri_model_fold_{fold_num}.pth"
            else:
                model_filename = "best_smri_model.pth"
            model_path = os.path.join(checkpoint_dir, model_filename)
            
            torch.save(best_state, model_path)
            print(f"  [Checkpoint] Saved new best model (AUC={val_auc:.4f}) -> {model_filename}")
            
            # Also save with general filename for backward compatibility
            general_model_path = os.path.join(checkpoint_dir, "best_smri_model.pth")
            torch.save(best_state, general_model_path)
            no_improvement_count = 0
            
            # Update best metrics
            best_precision_macro = precision_macro
            best_recall_macro = recall_macro
            best_f1_macro = f1_macro
            best_class_metrics = class_metrics
            best_confusion_matrix = cm
            best_threshold = optimal_threshold
            best_threshold_results = val_results.get('threshold_results', [])
        else:
            no_improvement_count += 1
            
        # Early stopping logic based on monitored metric
        current_monitored_metric = 0.0
        if early_stopping_monitor == 'val_auc':
            current_monitored_metric = val_auc
        elif early_stopping_monitor == 'val_acc':
            current_monitored_metric = val_acc
        elif early_stopping_monitor == 'val_loss':
            current_monitored_metric = -epoch_loss  # Negative because we want to maximize
        
        # Check if we have improvement
        if current_monitored_metric > best_monitored_metric + early_stopping_min_delta:
            best_monitored_metric = current_monitored_metric
            early_stopping_count = 0
        else:
            early_stopping_count += 1
        
        # Early stopping check
        if early_stopping_count >= early_stopping_patience:
            print(f"\n[EARLY STOPPING] No improvement in {early_stopping_monitor} for {early_stopping_patience} epochs")
            print(f"[EARLY STOPPING] Best {early_stopping_monitor}: {best_monitored_metric:.6f}")
            print(f"[EARLY STOPPING] Stopping training at epoch {epoch}/{epochs}")
            break
            
        # Legacy early stopping (keeping for backward compatibility)
        if no_improvement_count >= 20:
            print(f"\n[LEGACY] Early stopping triggered after {epoch} epochs (no improvement in AUC)")
            break

    # Load best model weights before returning (prefer saved checkpoint to avoid accidental mutation)
    if fold_num is not None:
        best_model_path = os.path.join(checkpoint_dir, f"best_smri_model_fold_{fold_num}.pth")
    else:
        best_model_path = os.path.join(checkpoint_dir, "best_smri_model.pth")
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
    elif best_state is not None:
        model.load_state_dict(best_state)
    
    # Print training summary
    legacy_early_stopped = (no_improvement_count >= 20)
    if early_stopping_count >= early_stopping_patience or legacy_early_stopped:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED WITH EARLY STOPPING")
        print(f"{'='*60}")
        print(f"Final epoch: {epoch}/{epochs}")
        if legacy_early_stopped:
            print(f"[LEGACY] Early stopping triggered (no improvement in AUC for 20 epochs)")
        else:
            print(f"Early stopping triggered: {early_stopping_monitor} did not improve for {early_stopping_patience} epochs")
        print(f"Best {early_stopping_monitor}: {best_monitored_metric:.6f}")
        print(f"Best validation AUC: {best_val_auc:.6f}")
        print(f"Best validation accuracy: {best_val_acc:.6f}")
        print(f"Training completed in {len(training_history)} epochs")
    else:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Completed all {epochs} epochs")
        print(f"Best {early_stopping_monitor}: {best_monitored_metric:.6f}")
        print(f"Best validation AUC: {best_val_auc:.6f}")
        print(f"Best validation accuracy: {best_val_acc:.6f}")
    
    return model, best_val_auc, best_val_acc, final_train_loss, final_train_acc, training_history, best_precision_macro, best_recall_macro, best_f1_macro, best_class_metrics, best_confusion_matrix, best_threshold, best_threshold_results

def optimize_threshold(y_probs, y_true, thresholds=None):
    """
    Find the optimal threshold that maximizes accuracy.
    
    Args:
        y_probs: predicted probabilities (numpy array)
        y_true: ground truth labels (numpy array)
        thresholds: array of thresholds to test (default: 0.1 to 0.9)
    
    Returns:
        best_threshold: optimal threshold for accuracy
        best_accuracy: accuracy at optimal threshold
        threshold_results: dict with all threshold results
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    
    best_acc = 0
    best_thresh = 0.5
    threshold_results = []
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        threshold_results.append({
            'threshold': t,
            'accuracy': acc,
            'predictions': preds
        })
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    
    return best_thresh, best_acc, threshold_results

def evaluate_model_with_threshold_optimization(model, val_loader, device, optimize_threshold_flag=True, label_mapping=None):
    """
    Evaluate model on validation set with optional threshold optimization.
    For multiclass, threshold optimization is disabled as it's not applicable.
    """
    model.eval()
    val_logits = []
    val_labels = []
    
    with torch.no_grad():
        for smri, labels in val_loader:
            smri = smri.to(device)
            
            # Apply label mapping if provided
            if label_mapping is not None:
                labels = torch.tensor([label_mapping[label.item()] for label in labels], device=device)
            
            logits = model(smri)
            val_logits.append(logits.cpu())
            val_labels.append(labels.cpu())
    
    val_logits = torch.cat(val_logits, dim=0).numpy()
    val_labels = torch.cat(val_labels, dim=0).numpy()
    
    # Apply softmax to get probabilities
    probs_softmax = nn.Softmax(dim=1)(torch.from_numpy(val_logits)).numpy()
    
    # Handle both binary and multiclass
    if probs_softmax.shape[1] == 2:
        # Binary classification: use positive class probability
        probs = probs_softmax[:, 1]
        is_binary = True
    else:
        # Multiclass: return full probability matrix for proper evaluation
        probs = probs_softmax
        is_binary = False
    
    # Calculate metrics with default threshold (0.5 for binary, argmax for multiclass)
    if is_binary:
        # Binary classification
        default_preds = (probs >= 0.5).astype(int)
        default_acc = accuracy_score(val_labels, default_preds)
        default_auc = roc_auc_score(val_labels, probs)
    else:
        # Multiclass: use argmax for predictions
        default_preds = np.argmax(probs, axis=1)
        default_acc = accuracy_score(val_labels, default_preds)
        # For multiclass, calculate AUC using one-vs-rest
        default_auc = roc_auc_score(val_labels, probs, multi_class='ovr', average='macro')
    
    results = {
        'probabilities': probs,
        'labels': val_labels,
        'default_threshold': 0.5 if is_binary else None,
        'default_accuracy': default_acc,
        'default_auc': default_auc,
        'optimal_threshold': 0.5 if is_binary else None,
        'optimal_accuracy': default_acc,
        'threshold_optimized': False,
        'is_binary': is_binary
    }
    
    if optimize_threshold_flag and is_binary:
        # Only optimize threshold for binary classification
        best_thresh, best_acc, threshold_results = optimize_threshold(probs, val_labels)
        
        # Calculate metrics with optimal threshold
        optimal_preds = (probs >= best_thresh).astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(val_labels, optimal_preds, average='macro', zero_division=0)
        mcc = matthews_corrcoef(val_labels, optimal_preds)
        
        results.update({
            'optimal_threshold': best_thresh,
            'optimal_accuracy': best_acc,
            'threshold_optimized': True,
            'accuracy_improvement': best_acc - default_acc,
            'optimal_precision': precision,
            'optimal_recall': recall,
            'optimal_f1': f1,
            'optimal_mcc': mcc,
            'threshold_results': threshold_results
        })
        
        print(f"Threshold optimization: {default_acc:.4f} -> {best_acc:.4f} (improvement: {best_acc - default_acc:.4f})")
        print(f"Optimal threshold: {best_thresh:.3f} (default: 0.5)")
    elif not is_binary:
        # For multiclass, calculate per-class metrics without threshold optimization
        precision, recall, f1, support = precision_recall_fscore_support(val_labels, default_preds, average='macro', zero_division=0)
        mcc = matthews_corrcoef(val_labels, default_preds)
        
        results.update({
            'optimal_precision': precision,
            'optimal_recall': recall,
            'optimal_f1': f1,
            'optimal_mcc': mcc,
            'threshold_optimized': False
        })
        
        print(f"Multiclass evaluation (no threshold optimization): Accuracy = {default_acc:.4f}, AUC = {default_auc:.4f}")
    
    return results

def k_fold_training(args, k_folds=5, models_to_run=None):
    """
    Outer stratified k-fold with per-fold test sets:
    - For each fold, use 20% (or args.test_ratio) of subjects as Test (non-overlapping across folds)
    - From remaining 80% as TrainPool, optionally undersample to balance classes (leftovers discarded)
    - Split TrainPool into Train/Val stratified (val_ratio)
    - Train, validate, then immediately evaluate on that fold's Test
    """
    import copy
    from sklearn.model_selection import train_test_split
    
    # Create dated folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{timestamp}"
    run_dir = os.path.join(args.checkpoint_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n" + "="*60)
    print(f"STARTING NEW TRAINING RUN: {run_folder}")
    print(f"Output directory: {run_dir}")
    print("="*60)

    # Clean up any existing temporary CSV files from previous runs
    data_dir = os.path.dirname(args.master_csv)
    temp_files_pattern = os.path.join(data_dir, "temp_*.csv")
    import glob
    existing_temp_files = glob.glob(temp_files_pattern)
    if existing_temp_files:
        print(f"Cleaning up {len(existing_temp_files)} previous temporary CSV files...")
        for temp_file in existing_temp_files:
            try:
                os.remove(temp_file)
                print(f"  Removed: {os.path.basename(temp_file)}")
            except Exception as e:
                print(f"  Warning: Could not remove {os.path.basename(temp_file)}: {e}")
        print("Cleanup completed.")
    else:
        print("No previous temporary CSV files found.")

    # Set random seed for reproducible splits (if specified)
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        print(f"Using random seed: {args.random_seed}")
    else:
        print("Using random seed for different subject mix each run")

    # Load and filter master dataset
    print(f"Loading master dataset from: {args.master_csv}")
    master_df = pd.read_csv(args.master_csv)
    if 'subject_id' not in master_df.columns or 'label' not in master_df.columns:
        master_df = pd.read_csv(args.master_csv, header=None, names=['subject_id', 'label'])
    
    # Drop header rows if present
    master_df = master_df[~master_df['subject_id'].isin(['subject_id', ''])]
    master_df = master_df[~master_df['label'].isin(['label', ''])]
    
    # Convert labels to int and filter
    master_df['label'] = master_df['label'].astype(int)
    filtered_df = master_df[master_df['label'].isin(args.labels)].reset_index(drop=True)
    
    print(f"Master dataset: {len(master_df)} total subjects")
    print(f"After filtering for labels {args.labels}: {len(filtered_df)} subjects")

    # Optional: balance entire dataset BEFORE outer CV; leftovers are discarded
    if args.balance_dataset:
        print("\nBalancing entire dataset before outer CV (undersampling; leftovers discarded)...")
        dataset_for_cv = balance_dataset(filtered_df, random_state=args.random_seed)
    else:
        dataset_for_cv = filtered_df

    # Show label distribution used for CV
    label_counts = dataset_for_cv['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(dataset_for_cv)*100:.1f}%)")

    # Build outer folds (test sets) with StratifiedKFold -> each fold's test set is unique
    # Note: n_splits defines the test proportion as 1/n_splits
    outer_skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.random_seed)
    outer_splits = list(outer_skf.split(range(len(dataset_for_cv)), dataset_for_cv['label']))

    # Model variants to try
    if models_to_run is None:
        models_to_run = [
            "Simple3DCNN", "ResNet18_3D", "DenseNet121_3D", "EfficientNetB0_3D",
            "VisionTransformer3D", "SwinUNETRClassifier", "FullSwinUNETRClassifier"
        ]

    all_model_results = []
    temp_files_this_run = []

    for model_name in models_to_run:
        print(f"\n{'#'*30}\nTraining model: {model_name}\n{'#'*30}")
        model_dir = os.path.join(run_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        fold_results = []
        folds_data = []
        fold_test_metrics = []

        for fold_idx, (train_pool_idx, test_idx) in enumerate(outer_splits, start=1):
            print(f"\nFOLD {fold_idx}/{k_folds} [{model_name}]")

            # Get fold-specific TrainPool and Test
            test_df = dataset_for_cv.iloc[test_idx].copy()
            train_pool_df = dataset_for_cv.iloc[train_pool_idx].copy()
            print(f"Train pool (pre-balance): {len(train_pool_df)} | Test: {len(test_df)}")

            # If balanced globally already, do not re-balance per fold
            balanced_train_pool_df = train_pool_df

            # Train/Val split on balanced TrainPool (stratified)
            train_df, val_df = train_test_split(
                balanced_train_pool_df,
                test_size=args.val_ratio,
                stratify=balanced_train_pool_df['label'],
                random_state=args.random_seed,
            )
            print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

            # Save CSVs per fold
            fold_tag = f"{run_folder}_{model_name}_fold_{fold_idx}"
            temp_train_csv = os.path.join(data_dir, f"temp_train_{fold_tag}.csv")
            temp_val_csv = os.path.join(data_dir, f"temp_val_{fold_tag}.csv")
            temp_test_csv = os.path.join(data_dir, f"temp_test_{fold_tag}.csv")
            train_df.to_csv(temp_train_csv, index=False)
            val_df.to_csv(temp_val_csv, index=False)
            test_df.to_csv(temp_test_csv, index=False)
            temp_files_this_run.extend([temp_train_csv, temp_val_csv, temp_test_csv])

            # Datasets and loaders
            train_transform = get_train_transform(args) if args.enable_augment else None
            train_dataset = SMRIDataset(csv_path=temp_train_csv, data_root=args.data_root, transform=train_transform, return_index=True)
            val_dataset = SMRIDataset(csv_path=temp_val_csv, data_root=args.data_root, transform=None, return_index=False)
            test_dataset = SMRIDataset(csv_path=temp_test_csv, data_root=args.data_root, transform=None, return_index=False)
            
            # Initialize model for this fold
            unique_labels = sorted(train_dataset.df['label'].unique())
            num_classes = len(unique_labels)
            print(f"[INFO] Model initialized with {num_classes} classes for labels: {unique_labels}")
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            print(f"[INFO] Label mapping: {label_mapping}")
            
            if model_name == "VisionTransformer3D":
                model = get_3d_model(
                    model_name,
                    num_classes=num_classes,
                    in_channels=1,
                    base_channels=args.base_channels,
                    use_pretrained=args.use_pretrained,
                    dropout_p=0.0,
                    vit_drop_rate=args.vit_drop_rate,
                    vit_attn_drop_rate=args.vit_attn_drop_rate,
                    vit_drop_path_rate=args.vit_drop_path_rate,
                )
            else:
                model = get_3d_model(
                    model_name,
                    num_classes=num_classes,
                    in_channels=1,
                    base_channels=args.base_channels,
                    use_pretrained=args.use_pretrained,
                    dropout_p=(args.cnn_drop_rate if model_name == "Simple3DCNN" else 0.0),
                )
            
            # Move model to device immediately after creation
            model = model.to(args.device)
            print(f"[INFO] Model moved to device: {args.device}")
            
            # Verify model is on correct device
            model_device = next(model.parameters()).device
            if str(model_device) != args.device:
                print(f"[WARNING] Model device mismatch! Expected: {args.device}, Got: {model_device}")
                print(f"[INFO] Moving model to correct device...")
                model = model.to(args.device)
                print(f"[INFO] Model now on device: {next(model.parameters()).device}")
            
            # Create memory-optimized data loaders
            train_loader, val_loader, test_loader, working_batch_size = create_data_loaders_with_memory_optimization(
                train_dataset, val_dataset, test_dataset, args, model
            )
            
            print(f"[MEMORY] Using batch size: {working_batch_size}")
            
            # Print memory status before training
            print_memory_status(args.device, "[MEMORY] Before training")
            
            # Train
            (
                model,
                best_val_auc,
                best_val_acc,
                final_train_loss,
                final_train_acc,
                training_history,
                best_precision_macro,
                best_recall_macro,
                best_f1_macro,
                best_class_metrics,
                best_confusion_matrix,
                best_threshold,
                best_threshold_results,
            ) = train_sMRI_model(
                model,
                train_loader,
                val_loader,
                args.epochs,
                args.device,
                model_dir,
                args,
                fold_num=fold_idx,
                label_mapping=label_mapping,
            )

            # Threshold optimization plot (only for binary classification)
            if best_threshold_results and num_classes == 2:
                threshold_plot_dir = os.path.join(model_dir, "threshold_optimization_plots")
                threshold_plot_info = create_threshold_optimization_plot(
                    best_threshold_results, threshold_plot_dir, model_name, fold_idx
                )
            else:
                threshold_plot_info = None
            
            # Test immediately on this fold's Test set
            print(f"Evaluating {model_name} on fold {fold_idx} test set...")
            # Use optimal validation threshold for test predictions (binary only)
            test_threshold = None
            if num_classes == 2 and best_threshold is not None:
                test_threshold = float(best_threshold)
            
            # For multiclass, use temperature scaling if available
            if num_classes > 2:
                # Use temperature scaling for multiclass calibration
                eval_result = evaluate_model(
                    model, test_loader, args.device, label_mapping=label_mapping, 
                    threshold=test_threshold, use_temperature_scaling=True, val_loader=val_loader
                )
                
                # Handle different return values
                if len(eval_result) == 4:
                    predictions, probabilities, labels, temperature_info = eval_result
                else:
                    predictions, probabilities, labels = eval_result
                    temperature_info = None
                
                # Log temperature scaling info
                if temperature_info and temperature_info.get('calibrated', False):
                    print(f"  Temperature scaling applied: T = {temperature_info['temperature']:.3f}")
                    # Save temperature info
                    temp_info_path = os.path.join(model_dir, f"temperature_info_fold_{fold_idx}.json")
                    with open(temp_info_path, 'w') as f:
                        json.dump(temperature_info, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
            else:
                # Binary classification: no temperature scaling
                eval_result = evaluate_model(
                model, test_loader, args.device, label_mapping=label_mapping, threshold=test_threshold
            )
                predictions, probabilities, labels = eval_result
                temperature_info = None
            
            metrics = calculate_metrics(predictions, probabilities, labels)
            test_eval_dir = os.path.join(model_dir, f"test_evaluation_plots_fold_{fold_idx}")
            create_evaluation_plots(predictions, probabilities, labels, metrics, test_eval_dir)
            test_metrics_path = os.path.join(model_dir, f"test_metrics_fold_{fold_idx}.json")
            with open(test_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
            print(f"Fold {fold_idx} test evaluation saved to: {test_eval_dir}")

            # Print memory status after evaluation
            print_memory_status(args.device, "[MEMORY] After evaluation")

            # Store test metrics for aggregation
            safe_metrics = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'auc': float(metrics['auc']),
                'mcc': float(metrics['mcc']),
                'confusion_matrix': metrics['confusion_matrix'].tolist() if hasattr(metrics.get('confusion_matrix', None), 'tolist') else metrics.get('confusion_matrix')
            }
            
            # Add temperature scaling info if available
            if temperature_info and temperature_info.get('calibrated', False):
                safe_metrics['temperature_scaling'] = {
                    'temperature': temperature_info['temperature'],
                    'calibrated': True
                }
            
            fold_test_metrics.append({
                'fold': fold_idx, 
                'metrics': safe_metrics, 
                'threshold_used': float(test_threshold) if test_threshold is not None else None,
                'temperature_scaling': temperature_info.get('temperature', 1.0) if temperature_info else None
            })

            # Collect results
            fold_results.append({
                'fold': fold_idx,
                'best_val_auc': float(best_val_auc),
                'best_val_acc': float(best_val_acc),
                'final_train_loss': float(final_train_loss),
                'final_train_acc': float(final_train_acc),
                'best_precision_macro': float(best_precision_macro),
                'best_recall_macro': float(best_recall_macro),
                'best_f1_macro': float(best_f1_macro),
                'best_class_metrics': best_class_metrics,
                'best_confusion_matrix': best_confusion_matrix.tolist() if best_confusion_matrix is not None else None,
                'best_threshold': float(best_threshold) if best_threshold is not None else None,
                'threshold_optimization': threshold_plot_info,
                'test_metrics_path': test_metrics_path,
            })
            folds_data.append({'fold': fold_idx, 'data': training_history})

            run_id = f"fold_{fold_idx}_{uuid.uuid4().hex[:8]}"
            log_metrics(
                run_id=run_id,
                model_name=model_name,
                args=args,
                best_val_auc=best_val_auc,
                best_val_acc=best_val_acc,
                final_train_loss=final_train_loss,
                final_train_acc=final_train_acc,
                notes=f"{model_name} Fold {fold_idx}/{k_folds}"
            )
        
        # Aggregate per-model results and save
        avg_val_auc = float(np.mean([r['best_val_auc'] for r in fold_results]))
        avg_val_acc = float(np.mean([r['best_val_acc'] for r in fold_results]))
        avg_precision_macro = float(np.mean([r['best_precision_macro'] for r in fold_results]))
        avg_recall_macro = float(np.mean([r['best_recall_macro'] for r in fold_results]))
        avg_f1_macro = float(np.mean([r['best_f1_macro'] for r in fold_results]))
        avg_mcc = float(np.mean([r.get('best_mcc', 0.0) for r in fold_results]))
        
        # Handle threshold aggregation (only for binary classification)
        if num_classes == 2:
            avg_threshold = float(np.mean([r.get('best_threshold', 0.5) for r in fold_results if r.get('best_threshold') is not None]))
            avg_accuracy_improvement = float(np.mean([r.get('threshold_optimization', {}).get('improvement', 0.0) for r in fold_results]))
        else:
            avg_threshold = None
            avg_accuracy_improvement = None

        evaluation_dir = os.path.join(model_dir, "evaluation_plots")
        os.makedirs(evaluation_dir, exist_ok=True)
        create_training_plots(folds_data, evaluation_dir, model_name)
        # Create aggregated test plots and summary
        classification_description = get_label_description(args.labels)
        test_summary = create_test_summary_plots(fold_test_metrics, evaluation_dir, model_name, classification_description=classification_description)
        folds_data_filename = f"{model_name}_folds_data.json"
        folds_data_path = os.path.join(model_dir, folds_data_filename)
        with open(folds_data_path, "w") as f:
            json.dump(folds_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        # Compute validation stats across folds for this model
        val_stats = {
            'auc': compute_summary_stats([r['best_val_auc'] for r in fold_results]),
            'accuracy': compute_summary_stats([r['best_val_acc'] for r in fold_results]),
            'precision': compute_summary_stats([r['best_precision_macro'] for r in fold_results]),
            'recall': compute_summary_stats([r['best_recall_macro'] for r in fold_results]),
            'f1': compute_summary_stats([r['best_f1_macro'] for r in fold_results])
        }

        run_summary = {
            'timestamp': timestamp,
            'run_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'run_folder': run_folder,
            'model_name': model_name,
            'model_filename': f"{model_name}_best_smri_model.pth",
            'folds_data_filename': folds_data_filename,
            'average_val_auc': avg_val_auc,
            'average_val_acc': avg_val_acc,
            'average_precision_macro': avg_precision_macro,
            'average_recall_macro': avg_recall_macro,
            'average_f1_macro': avg_f1_macro,
            'average_mcc': avg_mcc,
            'average_threshold': avg_threshold,
            'average_accuracy_improvement': avg_accuracy_improvement,
            'validation_stats': val_stats,
            'test_summary': test_summary,
            'total_folds': len(fold_results),
            'training_params': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'base_channels': args.base_channels,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'k_folds': args.k_folds,
                'labels': args.labels,
                'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio,
                'random_seed': args.random_seed,
                'optimize_threshold': args.optimize_threshold
            },
            'fold_results': fold_results
        }
        summary_filename = f"{model_name}_run_summary.json"
        summary_path = os.path.join(model_dir, summary_filename)
        with open(summary_path, "w") as f:
            json.dump(run_summary, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        
        # Clean up memory before next model
        print(f"[MEMORY] Cleaning up memory after {model_name}...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_status(args.device, "[MEMORY] After cleanup")
        
        all_model_results.append({
            'model_name': model_name,
            'avg_val_auc': float(avg_val_auc),
            'avg_val_acc': float(avg_val_acc),
            'avg_precision_macro': float(avg_precision_macro),
            'avg_recall_macro': float(avg_recall_macro),
            'avg_f1_macro': float(avg_f1_macro),
            'avg_mcc': float(avg_mcc),
            'val_stats': val_stats,
            'test_stats': (test_summary['metrics'] if isinstance(test_summary, dict) and 'metrics' in test_summary else None),
            'fold_results': fold_results
        })
        print(f"\n{model_name} results saved to: {model_dir}")
        if avg_threshold is not None:
            print(f"Average optimal threshold: {avg_threshold:.3f} (default: 0.5)")
        else:
            print("Average optimal threshold: Not applicable (multiclass)")
        if avg_accuracy_improvement is not None:
            print(f"Average accuracy improvement: {avg_accuracy_improvement:.4f}")
        else:
            print("Average accuracy improvement: Not applicable (multiclass)")
            print("Average accuracy improvement: Not applicable (multiclass)")
        
    # Clean up temporary CSVs created for this run
    removed_count = 0
    for fpath in temp_files_this_run:
        try:
            os.remove(fpath)
            removed_count += 1
        except Exception:
            pass
    if removed_count:
        print(f"Cleaned up {removed_count} temporary CSV files from data directory")
    
    # --- Summary comparison plot ---
    print("\nGenerating summary comparison plot for all models...")
    model_names = [r['model_name'] for r in all_model_results]
    avg_aucs = [r['avg_val_auc'] for r in all_model_results]
    avg_accs = [r['avg_val_acc'] for r in all_model_results]
    avg_precisions = [r['avg_precision_macro'] for r in all_model_results]
    avg_recalls = [r['avg_recall_macro'] for r in all_model_results]
    avg_f1s = [r['avg_f1_macro'] for r in all_model_results]
    # Also prepare test metric means if available
    test_auc_means = []
    test_auc_cis = []
    for r in all_model_results:
        ts = r.get('test_stats')
        if ts and 'auc' in ts:
            test_auc_means.append(ts['auc'].get('mean', None))
            test_auc_cis.append(ts['auc'].get('ci95', 0.0))
        else:
            test_auc_means.append(None)
            test_auc_cis.append(0.0)
    
    # Create a larger figure for more metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Pull per-model validation stats for error bars
    val_auc_means = []
    val_auc_cis = []
    val_acc_means = []
    val_acc_cis = []
    prec_means = []
    prec_cis = []
    rec_means = []
    rec_cis = []
    f1_means = []
    f1_cis = []
    for r in all_model_results:
        if 'val_stats' in r and r['val_stats']:
            val_auc_means.append(r['val_stats']['auc']['mean'])
            val_auc_cis.append(r['val_stats']['auc']['ci95'])
            val_acc_means.append(r['val_stats']['accuracy']['mean'])
            val_acc_cis.append(r['val_stats']['accuracy']['ci95'])
            prec_means.append(r['val_stats']['precision']['mean'])
            prec_cis.append(r['val_stats']['precision']['ci95'])
            rec_means.append(r['val_stats']['recall']['mean'])
            rec_cis.append(r['val_stats']['recall']['ci95'])
            f1_means.append(r['val_stats']['f1']['mean'])
            f1_cis.append(r['val_stats']['f1']['ci95'])
        else:
            val_auc_means.append(0.0)
            val_auc_cis.append(0.0)
            val_acc_means.append(0.0)
            val_acc_cis.append(0.0)
            prec_means.append(0.0)
            prec_cis.append(0.0)
            rec_means.append(0.0)
            rec_cis.append(0.0)
            f1_means.append(0.0)
            f1_cis.append(0.0)

    # Plot 1: Validation AUC and Accuracy with 95% CI error bars
    x = np.arange(len(model_names))
    width = 0.35
    ax1.bar(x - width/2, val_auc_means, width, yerr=val_auc_cis, capsize=4, label='Val AUC (mean ± 95% CI)', alpha=0.85, color='skyblue', edgecolor='black')
    ax1.bar(x + width/2, val_acc_means, width, yerr=val_acc_cis, capsize=4, label='Val Acc (mean ± 95% CI)', alpha=0.85, color='lightcoral', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=20)
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.set_title(f'Validation AUC and Accuracy Comparison\n{get_label_description(args.labels)} Classification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Precision, Recall, F1 with error bars
    ax2.bar(x - width, prec_means, width, yerr=prec_cis, capsize=4, label='Precision', alpha=0.85, color='lightgreen', edgecolor='black')
    ax2.bar(x, rec_means, width, yerr=rec_cis, capsize=4, label='Recall', alpha=0.85, color='lightblue', edgecolor='black')
    ax2.bar(x + width, f1_means, width, yerr=f1_cis, capsize=4, label='F1 Score', alpha=0.85, color='orange', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=20)
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'Validation Precision, Recall, F1 Comparison\n{get_label_description(args.labels)} Classification')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_plot_path = os.path.join(run_dir, f'model_comparison_summary_{get_label_description(args.labels).replace(" vs ", "_vs_")}.png')
    plt.savefig(summary_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Summary comparison plot saved to: {summary_plot_path}")
    
    # Save overall comparison summary
    # Extend comparison summary with mean, 95% CI and range per model for validation and test (if available)
    model_detailed_stats = {}
    for r in all_model_results:
        name = r['model_name']
        # Validation stats already computed per model
        val_stats = r.get('val_stats', None)
        # Optional test stats (from aggregated test summary)
        test_stats = r.get('test_stats', None)
        model_detailed_stats[name] = {
            'validation': val_stats,
            'test': test_stats
        }

    comparison_summary = {
        'run_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_folder': run_folder,
        'classification_task': get_label_description(args.labels),
        'models_tested': model_names,
        'comparison_results': all_model_results,
        'per_model_stats': model_detailed_stats,
        'best_model_by_auc': model_names[np.argmax(avg_aucs)],
        'best_model_by_acc': model_names[np.argmax(avg_accs)],
        'best_model_by_f1': model_names[np.argmax(avg_f1s)],
        'best_auc': max(avg_aucs),
        'best_acc': max(avg_accs),
        'best_f1': max(avg_f1s),
        'best_precision': max(avg_precisions),
        'best_recall': max(avg_recalls)
    }
    comparison_summary_path = os.path.join(run_dir, f'model_comparison_summary_{get_label_description(args.labels).replace(" vs ", "_vs_")}.json')
    with open(comparison_summary_path, "w") as f:
        json.dump(comparison_summary, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    print(f"Model comparison summary saved to: {comparison_summary_path}")
    
    print(f"\n{'='*60}\nALL MODEL TRAINING COMPLETED\n{'='*60}")
    print(f"Best model by AUC: {comparison_summary['best_model_by_auc']} ({comparison_summary['best_auc']:.4f})")
    print(f"Best model by Accuracy: {comparison_summary['best_model_by_acc']} ({comparison_summary['best_acc']:.4f})")
    print(f"Best model by F1: {comparison_summary['best_model_by_f1']} ({comparison_summary['best_f1']:.4f})")
    print(f"Best Precision: {comparison_summary['best_precision']:.4f}")
    print(f"Best Recall: {comparison_summary['best_recall']:.4f}")
    print(f"All outputs saved to: {run_dir}")
    return all_model_results

def ensemble_evaluate_models(model_name, model_dir, test_loader, device, args, fold_results):
    """
    Evaluate ensemble of all fold models on test set.
    
    Args:
        model_name: name of the model architecture
        model_dir: directory containing fold models
        test_loader: test data loader
        device: device to run inference on
        args: training arguments
        fold_results: results from all folds
    
    Returns:
        ensemble_predictions, ensemble_probabilities, test_labels, ensemble_metrics
    """
    print(f"Loading ensemble of {len(fold_results)} fold models...")
    
    models = []
    fold_aucs = []
    
    # Load all fold models
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        fold_auc = fold_result['best_val_auc']
        fold_aucs.append(fold_auc)
        
        model_path = os.path.join(model_dir, f"best_smri_model_fold_{fold_num}.pth")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold_num} not found: {model_path}")
            continue
            
        # Load model
        state_dict = torch.load(model_path, map_location=device)
        
        # Initialize model with correct architecture
        if model_name == "Simple3DCNN":
            classifier_weight = state_dict['classifier.0.weight']
            actual_input_size = classifier_weight.shape[1]
            model = get_3d_model(model_name, num_classes=len(args.labels), in_channels=1, base_channels=args.base_channels, use_pretrained=args.use_pretrained, dropout_p=0.25)
            model.classifier[0] = nn.Linear(actual_input_size, 256)
            model._initialized = True
            model.load_state_dict(state_dict)
        elif model_name == "SwinUNETRClassifier":
            classifier_weight = state_dict['classifier.0.weight']
            actual_input_size = classifier_weight.shape[1]
            model = get_3d_model(model_name, num_classes=len(args.labels), in_channels=1, base_channels=args.base_channels, use_pretrained=args.use_pretrained, dropout_p=0.25)
            model.classifier[0] = nn.Linear(actual_input_size, 512)
            model._initialized = True
            model.load_state_dict(state_dict)
        elif model_name == "FullSwinUNETRClassifier":
            classifier_weight = state_dict['classifier.0.weight']
            actual_input_size = classifier_weight.shape[1]
            model = get_3d_model(model_name, num_classes=len(args.labels), in_channels=1, base_channels=args.base_channels, use_pretrained=args.use_pretrained, dropout_p=0.25)
            model.classifier[0] = nn.Linear(actual_input_size, 512)
            model._initialized = True
            model.load_state_dict(state_dict)
        else:
            model = get_3d_model(model_name, num_classes=len(args.labels), in_channels=1, base_channels=args.base_channels, use_pretrained=args.use_pretrained)
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        models.append(model)
        
        print(f"  Fold {fold_num}: AUC = {fold_auc:.4f}")
    
    if not models:
        raise ValueError("No models could be loaded for ensemble evaluation")
    
    print(f"Successfully loaded {len(models)} models for ensemble")
    
    # Get ensemble predictions
    all_probabilities = []
    test_labels = []
    
    with torch.no_grad():
        for batch_idx, (smri, labels) in enumerate(test_loader):
            try:
                smri = smri.to(device)
                batch_probabilities = []
                
                # Get predictions from each model (with error handling)
                for i, model in enumerate(models):
                    try:
                        logits = model(smri)
                        probs = torch.softmax(logits, dim=1)
                        batch_probabilities.append(probs.cpu().numpy())
                    except Exception as e:
                        print(f"Warning: Error in model {i+1}: {e}")
                        # Use zero probabilities as fallback
                        batch_probabilities.append(np.zeros_like(probs.cpu().numpy()))
                
                if batch_probabilities:
                    # Average probabilities across models
                    ensemble_probs = np.mean(batch_probabilities, axis=0)
                    all_probabilities.append(ensemble_probs)
                    test_labels.append(labels.numpy())
                else:
                    print(f"Warning: No valid predictions for batch {batch_idx}")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Concatenate all batches
    if not all_probabilities or not test_labels:
        raise ValueError("No valid predictions generated during ensemble evaluation")
    
    ensemble_probabilities = np.concatenate(all_probabilities, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Get predictions (argmax of averaged probabilities)
    ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
    
    # Calculate metrics
    ensemble_metrics = calculate_metrics(ensemble_predictions, ensemble_probabilities, test_labels)
    
    # Add ensemble-specific information
    ensemble_metrics['ensemble_info'] = {
        'num_models': len(models),
        'fold_aucs': fold_aucs,
        'average_fold_auc': np.mean(fold_aucs),
        'std_fold_auc': np.std(fold_aucs),
        'min_fold_auc': np.min(fold_aucs),
        'max_fold_auc': np.max(fold_aucs)
    }
    
    print(f"Ensemble evaluation completed:")
    print(f"  Models used: {len(models)}")
    print(f"  Average fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  Ensemble accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"  Ensemble AUC: {ensemble_metrics['auc']:.4f}")
    
    return ensemble_predictions, ensemble_probabilities, test_labels, ensemble_metrics

def get_gpu_memory_info(device):
    """
    Get current GPU memory usage information.
    
    Args:
        device: CUDA device string (e.g., 'cuda:3')
    
    Returns:
        dict: Memory usage information
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    try:
        gpu_id = int(device.split(':')[1]) if ':' in device else 0
        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3    # GB
        memory_free = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3 - memory_reserved  # GB
        
        return {
            'gpu_id': gpu_id,
            'allocated_gb': round(memory_allocated, 2),
            'reserved_gb': round(memory_reserved, 2),
            'free_gb': round(memory_free, 2),
            'total_gb': round(torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3, 2)
        }
    except Exception as e:
        return {'error': str(e)}


def print_memory_status(device, prefix="[MEMORY]"):
    """
    Print current GPU memory status.
    
    Args:
        device: CUDA device string
        prefix: Prefix for the log message
    """
    memory_info = get_gpu_memory_info(device)
    if 'error' not in memory_info:
        print(f"{prefix} GPU {memory_info['gpu_id']}: "
              f"Allocated: {memory_info['allocated_gb']}GB, "
              f"Reserved: {memory_info['reserved_gb']}GB, "
              f"Free: {memory_info['free_gb']}GB, "
              f"Total: {memory_info['total_gb']}GB")
    else:
        print(f"{prefix} {memory_info['error']}")


def auto_reduce_batch_size(model, initial_batch_size, min_batch_size, device, args):
    """
    Automatically reduce batch size if CUDA out of memory occurs.
    
    Args:
        model: The model to test
        initial_batch_size: Starting batch size
        min_batch_size: Minimum batch size to try
        device: Device to test on
        args: Training arguments
    
    Returns:
        int: Working batch size
    """
    print(f"[MEMORY] Testing batch size {initial_batch_size}...")
    
    for batch_size in range(initial_batch_size, min_batch_size - 1, -4):
        try:
            # Create a dummy batch to test memory
            dummy_input = torch.randn(batch_size, 1, 96, 112, 96).to(device)
            
            # Ensure model is on the same device as input
            if next(model.parameters()).device != dummy_input.device:
                print(f"[MEMORY] Moving model to {device} for testing...")
                model = model.to(device)
            
            # Test forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Test backward pass with a dummy loss
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=device)
            dummy_loss.backward()
            
            print(f"[MEMORY] ✅ Batch size {batch_size} works! Using this for training.")
            return batch_size
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"[MEMORY] ❌ Batch size {batch_size} failed: {e}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if batch_size <= min_batch_size:
                print(f"[MEMORY] ⚠️  Reached minimum batch size {min_batch_size}. Training may be slow.")
                return min_batch_size
        except RuntimeError as e:
            if "Input type" in str(e) and "weight type" in str(e):
                print(f"[MEMORY] ❌ Device mismatch error: {e}")
                print(f"[MEMORY] Attempting to fix device placement...")
                try:
                    model = model.to(device)
                    # Retry with corrected device placement
                    with torch.no_grad():
                        _ = model(dummy_input)
                    print(f"[MEMORY] ✅ Device issue fixed! Batch size {batch_size} works.")
                    return batch_size
                except Exception as retry_e:
                    print(f"[MEMORY] ❌ Device fix failed: {retry_e}")
                    return min_batch_size
            else:
                print(f"[MEMORY] ❌ Runtime error: {e}")
                return min_batch_size
        except Exception as e:
            print(f"[MEMORY] ❌ Unexpected error: {e}")
            return min_batch_size
    
    print(f"[MEMORY] ⚠️  Could not find working batch size. Using minimum: {min_batch_size}")
    return min_batch_size


def create_data_loaders_with_memory_optimization(train_dataset, val_dataset, test_dataset, args, model):
    """
    Create data loaders with automatic batch size optimization.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        args: Training arguments
        model: Model to test memory usage with
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, working_batch_size)
    """
    working_batch_size = args.batch_size
    
    if args.auto_batch_size:
        print(f"[MEMORY] Auto-batch size optimization enabled.")
        print(f"[MEMORY] Testing memory usage with model: {type(model).__name__}")
        
        # Ensure model is on the correct device before testing
        if 'cuda' in args.device:
            model = model.to(args.device)
            print(f"[MEMORY] Model moved to {args.device}")
        
        # Test with current batch size
        working_batch_size = auto_reduce_batch_size(
            model, args.batch_size, args.min_batch_size, args.device, args
        )
        
        if working_batch_size != args.batch_size:
            print(f"[MEMORY] Reduced batch size from {args.batch_size} to {working_batch_size}")
    
    # Build sampler: if multiclass and flagged, use balanced batch sampler; else optionally weighted sampler
    # Determine if multiclass from train dataset labels
    try:
        train_labels_np = train_dataset.df['label'].to_numpy()
    except Exception:
        # Fallback: extract from dataset items (slower)
        tmp_labels = []
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            if isinstance(item, tuple) and len(item) == 3:
                _, y, _ = item
            else:
                _, y = item
            tmp_labels.append(int(y))
        train_labels_np = np.array(tmp_labels)
    unique_train_labels = np.unique(train_labels_np)
    is_multiclass = len(unique_train_labels) > 2

    sampler = None
    if is_multiclass and getattr(args, 'use_balanced_batches', False):
        # Balanced per-batch sampler. Requires dataset to return indices; ensure dataset has return_index
        if not getattr(train_dataset, 'return_index', False):
            # Re-wrap dataset to enable index return
            train_dataset.return_index = True
        sampler = BalancedBatchSampler(train_labels_np.tolist(), batch_size=working_batch_size)
    elif is_multiclass:
        from collections import Counter
        counts = Counter(train_labels_np.tolist())
        sample_weights = np.array([1.0 / counts[l] for l in train_labels_np], dtype=np.float64)
        weights_tensor = torch.from_numpy(sample_weights)
        sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(train_labels_np), replacement=True)

    # Create data loaders with working batch size
    if sampler is not None and isinstance(sampler, BalancedBatchSampler):
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True if 'cuda' in args.device else False
        )
    elif sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=working_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if 'cuda' in args.device else False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=working_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if 'cuda' in args.device else False
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=working_batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if 'cuda' in args.device else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=working_batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if 'cuda' in args.device else False
    )
    
    return train_loader, val_loader, test_loader, working_batch_size

def main():
    parser = argparse.ArgumentParser(description='Train sMRI models with k-fold cross-validation')
    
    # Data arguments
    parser.add_argument("--master_csv", type=str, required=True,
                        help="Path to master CSV file with subject IDs and labels")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing preprocessed sMRI data")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory to save model checkpoints and results")
    
    # Model arguments
    parser.add_argument("--labels", nargs='+', type=int, required=True,
                        help="List of label values to use for classification")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to train (alternative to --models)")
    parser.add_argument("--models", nargs='+', type=str, default=None,
                        choices=['Simple3DCNN', 'VisionTransformer3D', 'SwinUNETRClassifier', 'FullSwinUNETRClassifier'],
                        help="List of models to train (alternative to --model)")
    parser.add_argument("--run_all", action='store_true', default=False,
                        help="Run all available models")
    parser.add_argument("--use_pretrained", action='store_true', default=False,
                        help="Use pretrained weights for models that support it")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.00001, help="Weight decay for CNN models")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5, help="LR scheduler patience")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5, help="LR scheduler factor (0.5 = halve LR)")
    
    # Early stopping arguments
    parser.add_argument("--early_stopping_patience", type=int, default=30,
                        help="Early stopping patience (stop if no improvement for N epochs)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001,
                        help="Minimum improvement threshold for early stopping")
    parser.add_argument("--early_stopping_monitor", type=str, default="val_auc",
                        choices=["val_auc", "val_acc", "val_loss"],
                        help="Metric to monitor for early stopping")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu, or specific GPU)")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of k-folds for cross-validation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--base_channels", type=int, default=64, help="Base number of channels for CNN models")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Data split arguments
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--balance_dataset", action='store_true', default=False,
                        help="Balance dataset by undersampling majority classes")
    
    # CNN-specific arguments
    parser.add_argument("--cnn_drop_rate", type=float, default=0.0, help="Dropout rate for CNN models")
    
    # Vision Transformer specific arguments
    parser.add_argument("--vit_warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs for Vision Transformer models")
    parser.add_argument("--vit_use_cosine_schedule", action='store_true', default=True,
                        help="Use cosine learning rate schedule for Vision Transformer models")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.0 for sharper CN/PD boundaries; 0.1 for ViT)")
    parser.add_argument("--use_balanced_batches", action='store_true', default=False,
                        help="Use per-batch class balancing for multiclass training")
    parser.add_argument("--enable_augment", action='store_true', default=False,
                        help="Enable light 3D augmentations on training data")
    parser.add_argument("--cb_beta", type=float, default=0.999,
                        help="Beta for Class-Balanced Loss effective number (0.99–0.999)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma for focal loss focusing parameter")
    parser.add_argument("--hard_mining_weight", type=float, default=3.0,
                        help="Sampling weight multiplier for CN/PD hard examples")
    parser.add_argument("--vit_optimizer", type=str, default="adamw", choices=["adamw", "adam"],
                        help="Optimizer for Vision Transformer models")
    parser.add_argument("--vit_weight_decay", type=float, default=0.05,
                        help="Weight decay for Vision Transformer models (0.02-0.1 recommended)")
    parser.add_argument("--vit_drop_rate", type=float, default=0.1,
                        help="Dropout rate for Vision Transformer models (0.1-0.3 recommended)")
    parser.add_argument("--vit_attn_drop_rate", type=float, default=0.0,
                        help="Attention dropout rate for Vision Transformer models (0.0-0.1 recommended)")
    parser.add_argument("--vit_drop_path_rate", type=float, default=0.1,
                        help="Stochastic depth rate for Vision Transformer models (0.1-0.2 recommended)")
    
    # General training arguments
    parser.add_argument("--grad_clip_max_norm", type=float, default=0.0,
                        help="Gradient clipping max norm (1.0 recommended for ViT)")
    parser.add_argument("--optimize_threshold", action='store_true', default=True,
                        help="Optimize classification threshold on validation set")
    parser.add_argument("--use_temperature_scaling", action='store_true', default=True,
                        help="Enable temperature scaling for multiclass calibration (improves accuracy by 2-5%)")
    
    # Memory optimization arguments
    parser.add_argument("--auto_batch_size", action='store_true', default=True,
                        help="Automatically reduce batch size if CUDA out of memory occurs")
    parser.add_argument("--max_batch_size", type=int, default=32,
                        help="Maximum batch size to try during auto-reduction")
    parser.add_argument("--min_batch_size", type=int, default=4,
                        help="Minimum batch size to try during auto-reduction")
    parser.add_argument("--memory_efficient", action='store_true', default=False,
                        help="Enable memory-efficient training (gradient checkpointing, etc.)")
    
    args = parser.parse_args()

    # Define available models
    available_models = ["Simple3DCNN", "ResNet18_3D", "ResNet50_3D", "DenseNet121_3D", "EfficientNetB0_3D",
                       "VisionTransformer3D", "SwinUNETRClassifier", "FullSwinUNETRClassifier"]
    
    # Determine which models to run
    if args.model:
        # Single model specified
        if args.model not in available_models:
            raise ValueError(f"Unknown model: {args.model}. Available models: {available_models}")
        models_to_run = [args.model]
        print(f"Running single model: {args.model}")
    elif args.models:
        # Multiple specific models specified
        for model in args.models:
            if model not in available_models:
                raise ValueError(f"Unknown model: {model}. Available models: {available_models}")
        models_to_run = args.models
        print(f"Running models: {', '.join(models_to_run)}")
    elif args.run_all:
        # Run all models (explicit)
        models_to_run = available_models
        print(f"Running all models: {', '.join(models_to_run)}")
    else:
        # Default behavior: run all models
        models_to_run = available_models
        print(f"No model selection specified. Running all models: {', '.join(models_to_run)}")

    # Perform k-fold cross validation with selected models
    fold_results = k_fold_training(args, k_folds=args.k_folds, models_to_run=models_to_run)

if __name__ == "__main__":
    main()
