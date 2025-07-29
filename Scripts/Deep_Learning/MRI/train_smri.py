# scripts/train_smri.py

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
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

from dataset import SMRIDataset
from models_smri import Simple3DCNN, get_3d_model
from evaluate_model import evaluate_model, calculate_metrics, create_evaluation_plots

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

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

def train_sMRI_model(model, train_loader, val_loader, epochs, device, checkpoint_dir, args):
    """
    Trains model; saves best checkpoint by validation AUC into checkpoint_dir.
    Returns the model loaded with best weights and training history.
    """
    # Calculate class weights for imbalanced data
    labels = []
    for _, label in train_loader:
        labels.extend(label.numpy())
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()

    criterion = FocalLoss(alpha=0.25, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Add learning rate scheduler with more patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=10,  # Increased patience
        min_lr=1e-6   # Minimum learning rate
    )

    model.to(device)
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_state = None
    final_train_loss = 0.0
    final_train_acc = 0.0
    no_improvement_count = 0
    
    # Store best metrics
    best_precision_macro = 0.0
    best_recall_macro = 0.0
    best_f1_macro = 0.0
    best_class_metrics = {}
    best_confusion_matrix = None
    
    # Store training history
    training_history = []
    
    # Initialize mixed precision training for memory efficiency
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for smri, labels in train_loader:
            smri, labels = smri.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Use mixed precision training
            with torch.amp.autocast('cuda'):
                logits = model(smri)              # [B, 2]
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * smri.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += (preds == labels).sum().item()
            total_samples += smri.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc  = running_corrects / total_samples
        
        if epoch == epochs:
            final_train_loss = epoch_loss
            final_train_acc = epoch_acc

        # --- Validation phase ---
        model.eval()
        val_logits = []
        val_labels = []

        with torch.no_grad():
            for smri, labels in val_loader:
                smri = smri.to(device)
                logits = model(smri)          # [B, 2]
                val_logits.append(logits.cpu().numpy())
                val_labels.append(labels.numpy())

        val_logits = np.concatenate(val_logits, axis=0)  # [N_val, 2]
        val_labels = np.concatenate(val_labels, axis=0)  # [N_val]

        # Convert logits → probabilities for binary classification
        probs = nn.Softmax(dim=1)(torch.from_numpy(val_logits)).numpy()[:, 1]  # Take probability of positive class
        val_auc = roc_auc_score(val_labels, probs)  # Binary classification
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Calculate additional metrics
        precision, recall, f1, support = precision_recall_fscore_support(val_labels, val_preds, average=None, zero_division=0)
        cm = confusion_matrix(val_labels, val_preds)
        
        # Calculate macro averages for multi-class
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)
        
        # Store per-class metrics
        class_metrics = {}
        for i, class_label in enumerate(sorted(set(val_labels))):
            class_metrics[f'class_{class_label}'] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

        # Update learning rate based on validation AUC
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Store epoch data
        val_mcc = matthews_corrcoef(val_labels, val_preds)
        epoch_data = {
            'epoch': int(epoch),
            'train_loss': float(epoch_loss),
            'train_acc': float(epoch_acc),
            'val_auc': float(val_auc),
            'val_acc': float(val_acc),
            'lr': float(new_lr),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'val_mcc': float(val_mcc)
        }
        training_history.append(epoch_data)
        
        # Print learning rate change if it occurred
        lr_change = ""
        if new_lr != old_lr:
            lr_change = f"  [LR reduced to {new_lr:.6f}]"

        print(f"Epoch {epoch}/{epochs}  "
              f"Train loss={epoch_loss:.4f}, Train acc={epoch_acc:.4f}  "
              f"Val AUC={val_auc:.4f}, Val acc={val_acc:.4f}  "
              f"LR={new_lr:.6f}{lr_change}")

        # Checkpoint if this is the best AUC so far
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model with simple filename since it's in a dated folder
            model_filename = "best_smri_model.pth"
            model_path = os.path.join(checkpoint_dir, model_filename)
            
            torch.save(best_state, model_path)
            print(f"  [Checkpoint] Saved new best model (AUC={val_auc:.4f}) -> {model_filename}")
            no_improvement_count = 0
            
            # Update best metrics
            best_precision_macro = precision_macro
            best_recall_macro = recall_macro
            best_f1_macro = f1_macro
            best_class_metrics = class_metrics
            best_confusion_matrix = cm
        else:
            no_improvement_count += 1
            
        # Early stopping if no improvement for 20 epochs
        if no_improvement_count >= 20:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Load best model weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_auc, best_val_acc, final_train_loss, final_train_acc, training_history, best_precision_macro, best_recall_macro, best_f1_macro, best_class_metrics, best_confusion_matrix

def k_fold_training(args, k_folds=5, models_to_run=None):
    """
    Perform k-fold cross validation on master dataset with proper train/val/test splits.
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
    filtered_df = master_df[master_df['label'].isin(args.labels)]
    
    print(f"Master dataset: {len(master_df)} total subjects")
    print(f"After filtering for labels {args.labels}: {len(filtered_df)} subjects")

    # Show label distribution
    label_counts = filtered_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(filtered_df)*100:.1f}%)")

    # Balance dataset and create splits using new strategy
    if args.balance_dataset:
        print(f"\nUsing new balancing strategy: undersample -> split 70/20/10 -> add remaining to test")
        train, val, test, removed_subjects = balance_and_split_dataset(
            filtered_df, 
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio, 
            random_state=args.random_seed
        )
    else:
        # Create stratified train/val/test splits (original method)
        print(f"\nCreating stratified splits (train: {1-args.val_ratio-args.test_ratio:.1%}, val: {args.val_ratio:.1%}, test: {args.test_ratio:.1%})")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            filtered_df, 
            test_size=args.test_ratio, 
            stratify=filtered_df['label'], 
            random_state=args.random_seed
        )
        
        # Second split: train vs val
        val_relative_size = args.val_ratio / (1 - args.test_ratio)
        train, val = train_test_split(
            train_val, 
            test_size=val_relative_size, 
            stratify=train_val['label'], 
            random_state=args.random_seed
        )

    # Save splits to data directory
    data_dir = os.path.dirname(args.master_csv)
    temp_train_csv = os.path.join(data_dir, f'temp_train_{run_folder}.csv')
    temp_val_csv = os.path.join(data_dir, f'temp_val_{run_folder}.csv')
    temp_test_csv = os.path.join(data_dir, f'temp_test_{run_folder}.csv')
    
    train.to_csv(temp_train_csv, index=False)
    val.to_csv(temp_val_csv, index=False)
    test.to_csv(temp_test_csv, index=False)

    # Create datasets with split data
    train_dataset = SMRIDataset(csv_path=temp_train_csv, data_root=args.data_root)
    val_dataset = SMRIDataset(csv_path=temp_val_csv, data_root=args.data_root)
    test_dataset = SMRIDataset(csv_path=temp_test_csv, data_root=args.data_root)

    # Print split information
    print(f"\nDataset splits:")
    print(f"Training set: {len(train_dataset)} subjects ({len(train_dataset)/len(filtered_df)*100:.1f}%)")
    train_labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
    train_counts = pd.Series(train_labels).value_counts().sort_index()
    for label, count in train_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(train_labels)*100:.1f}%)")
    
    print(f"Validation set: {len(val_dataset)} subjects ({len(val_dataset)/len(filtered_df)*100:.1f}%)")
    val_labels = [val_dataset.labels[i] for i in range(len(val_dataset))]
    val_counts = pd.Series(val_labels).value_counts().sort_index()
    for label, count in val_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(val_labels)*100:.1f}%)")
    
    print(f"Test set: {len(test_dataset)} subjects ({len(test_dataset)/len(filtered_df)*100:.1f}%)")
    test_labels = [test_dataset.labels[i] for i in range(len(test_dataset))]
    test_counts = pd.Series(test_labels).value_counts().sort_index()
    for label, count in test_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(test_labels)*100:.1f}%)")

    # Create fixed validation and test loaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Get labels for stratification
    train_labels = [train_dataset.labels[i] for i in range(len(train_dataset))]

    # Initialize stratified k-fold to ensure balanced class distribution
    skfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.random_seed)
    splits = list(skfold.split(range(len(train_dataset)), train_labels))

    # Model variants to try
    if models_to_run is None:
        models_to_run = ["Simple3DCNN", "ResNet18_3D", "DenseNet121_3D", "EfficientNetB0_3D", 
                        "VisionTransformer3D", "SwinUNETRClassifier", "FullSwinUNETRClassifier"]

    all_model_results = []

    for model_name in models_to_run:
        print(f"\n{'#'*30}\nTraining model: {model_name}\n{'#'*30}")
        model_dir = os.path.join(run_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        fold_results = []
        folds_data = []
        for fold, (train_ids, val_fold_ids) in enumerate(splits):
            print(f'\nFOLD {fold + 1}/{k_folds} [{model_name}]')
            print(f'Training on {len(train_ids)} subjects, validating on {len(val_fold_ids)} subjects')
            train_labels_fold = [train_labels[i] for i in train_ids]
            val_fold_labels = [train_labels[i] for i in val_fold_ids]
            print(f'Training set class distribution:')
            train_counts = pd.Series(train_labels_fold).value_counts().sort_index()
            for label, count in train_counts.items():
                print(f'  Label {label}: {count} subjects ({count/len(train_labels_fold)*100:.1f}%)')
            print(f'Validation fold class distribution:')
            val_fold_counts = pd.Series(val_fold_labels).value_counts().sort_index()
            for label, count in val_fold_counts.items():
                print(f'  Label {label}: {count} subjects ({count/len(val_fold_labels)*100:.1f}%)')
            
            train_sampler = SubsetRandomSampler(train_ids)
            val_fold_sampler = SubsetRandomSampler(val_fold_ids)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers
            )
            val_fold_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=val_fold_sampler,
                num_workers=args.num_workers
            )
            
            # Initialize model for this fold
            model = get_3d_model(model_name, num_classes=len(args.labels), in_channels=1, base_channels=args.base_channels, use_pretrained=args.use_pretrained)
            
            # Train the model using training fold and validate on validation fold
            model, best_val_auc, best_val_acc, final_train_loss, final_train_acc, training_history, best_precision_macro, best_recall_macro, best_f1_macro, best_class_metrics, best_confusion_matrix = train_sMRI_model(
                model, train_loader, val_fold_loader, args.epochs, args.device, model_dir, args
            )
            
            fold_results.append({
                'fold': fold + 1,
                'best_val_auc': float(best_val_auc),
                'best_val_acc': float(best_val_acc),
                'final_train_loss': float(final_train_loss),
                'final_train_acc': float(final_train_acc),
                'best_precision_macro': float(best_precision_macro),
                'best_recall_macro': float(best_recall_macro),
                'best_f1_macro': float(best_f1_macro),
                'best_class_metrics': best_class_metrics,
                'best_confusion_matrix': best_confusion_matrix.tolist() if best_confusion_matrix is not None else None
            })
            folds_data.append({
                'fold': fold + 1,
                'data': training_history
            })
            run_id = f"fold_{fold + 1}_{uuid.uuid4().hex[:8]}"
            log_metrics(
                run_id=run_id,
                model_name=model_name,
                args=args,
                best_val_auc=best_val_auc,
                best_val_acc=best_val_acc,
                final_train_loss=final_train_loss,
                final_train_acc=final_train_acc,
                notes=f"{model_name} Fold {fold + 1}/{k_folds}"
            )
        
        # Save per-model results
        avg_val_auc = float(np.mean([r['best_val_auc'] for r in fold_results]))
        avg_val_acc = float(np.mean([r['best_val_acc'] for r in fold_results]))
        avg_precision_macro = float(np.mean([r['best_precision_macro'] for r in fold_results]))
        avg_recall_macro = float(np.mean([r['best_recall_macro'] for r in fold_results]))
        avg_f1_macro = float(np.mean([r['best_f1_macro'] for r in fold_results]))
        avg_mcc = float(np.mean([r.get('best_mcc', 0.0) for r in fold_results]))
        
        evaluation_dir = os.path.join(model_dir, "evaluation_plots")
        create_training_plots(folds_data, evaluation_dir, model_name)
        folds_data_filename = f"{model_name}_folds_data.json"
        folds_data_path = os.path.join(model_dir, folds_data_filename)
        with open(folds_data_path, "w") as f:
            json.dump(folds_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
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
                'random_seed': args.random_seed
            },
            'fold_results': fold_results
        }
        summary_filename = f"{model_name}_run_summary.json"
        summary_path = os.path.join(model_dir, summary_filename)
        with open(summary_path, "w") as f:
            json.dump(run_summary, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        all_model_results.append({
            'model_name': model_name,
            'avg_val_auc': float(avg_val_auc),
            'avg_val_acc': float(avg_val_acc),
            'avg_precision_macro': float(avg_precision_macro),
            'avg_recall_macro': float(avg_recall_macro),
            'avg_f1_macro': float(avg_f1_macro),
            'avg_mcc': float(avg_mcc),
            'fold_results': fold_results
        })
        print(f"\n{model_name} results saved to: {model_dir}")
        
        # Test set evaluation (always available now)
        print(f"\nEvaluating {model_name} on the test set...")
        best_model_path = os.path.join(model_dir, "best_smri_model.pth")
        if os.path.exists(best_model_path):
            file_size = os.path.getsize(best_model_path) / (1024*1024)  # MB
            print(f"Model file size: {file_size:.2f} MB")
            state_dict = torch.load(best_model_path, map_location=args.device)
            # For Simple3DCNN, we need to handle the classifier size mismatch
            if model_name == "Simple3DCNN":
                classifier_weight = state_dict['classifier.0.weight']
                actual_input_size = classifier_weight.shape[1]
                model = get_3d_model(model_name, num_classes=len(args.labels), in_channels=1, base_channels=args.base_channels, use_pretrained=args.use_pretrained)
                model.classifier[0] = nn.Linear(actual_input_size, 256)
                model._initialized = True
                model.load_state_dict(state_dict)
            elif model_name == "SwinUNETRClassifier":
                classifier_weight = state_dict['classifier.0.weight']
                actual_input_size = classifier_weight.shape[1]
                model = get_3d_model(model_name, num_classes=len(args.labels), in_channels=1, base_channels=args.base_channels, use_pretrained=args.use_pretrained)
                model.classifier[0] = nn.Linear(actual_input_size, 512)
                model._initialized = True
                model.load_state_dict(state_dict)
            elif model_name == "FullSwinUNETRClassifier":
                classifier_weight = state_dict['classifier.0.weight']
                actual_input_size = classifier_weight.shape[1]
                model = get_3d_model(model_name, num_classes=len(args.labels), in_channels=1, base_channels=args.base_channels, use_pretrained=args.use_pretrained)
                model.classifier[0] = nn.Linear(actual_input_size, 512)
                model._initialized = True
                model.load_state_dict(state_dict)
            else:
                model = get_3d_model(model_name, num_classes=len(args.labels), in_channels=1, base_channels=args.base_channels, use_pretrained=args.use_pretrained)
                model.load_state_dict(state_dict)
            model.to(args.device)
            model.eval()
            # Evaluate
            predictions, probabilities, labels = evaluate_model(model, test_loader, args.device)
            metrics = calculate_metrics(predictions, probabilities, labels)
            # Save metrics
            test_metrics_path = os.path.join(model_dir, "test_metrics.json")
            with open(test_metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
            # Save plots
            test_eval_dir = os.path.join(model_dir, "test_evaluation_plots")
            create_evaluation_plots(predictions, probabilities, labels, metrics, test_eval_dir)
            print(f"Test set evaluation for {model_name} saved to: {test_eval_dir}")
        else:
            print(f"ERROR: Model file not found: {best_model_path}")
            continue

    # Clean up temporary files
    try:
        os.remove(temp_train_csv)
        os.remove(temp_val_csv)
        os.remove(temp_test_csv)
        print(f"Cleaned up temporary files from data directory")
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")
    
    # --- Summary comparison plot ---
    print("\nGenerating summary comparison plot for all models...")
    model_names = [r['model_name'] for r in all_model_results]
    avg_aucs = [r['avg_val_auc'] for r in all_model_results]
    avg_accs = [r['avg_val_acc'] for r in all_model_results]
    avg_precisions = [r['avg_precision_macro'] for r in all_model_results]
    avg_recalls = [r['avg_recall_macro'] for r in all_model_results]
    avg_f1s = [r['avg_f1_macro'] for r in all_model_results]
    
    # Create a larger figure for more metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: AUC and Accuracy
    x = np.arange(len(model_names))
    width = 0.35
    ax1.bar(x - width/2, avg_aucs, width, label='Avg Val AUC', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, avg_accs, width, label='Avg Val Acc', alpha=0.8, color='lightcoral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=20)
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.set_title(f'AUC and Accuracy Comparison\n{get_label_description(args.labels)} Classification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision, Recall, F1
    ax2.bar(x - width, avg_precisions, width, label='Precision', alpha=0.8, color='lightgreen')
    ax2.bar(x, avg_recalls, width, label='Recall', alpha=0.8, color='lightblue')
    ax2.bar(x + width, avg_f1s, width, label='F1 Score', alpha=0.8, color='orange')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=20)
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'Precision, Recall, F1 Comparison\n{get_label_description(args.labels)} Classification')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_plot_path = os.path.join(run_dir, f'model_comparison_summary_{get_label_description(args.labels).replace(" vs ", "_vs_")}.png')
    plt.savefig(summary_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Summary comparison plot saved to: {summary_plot_path}")
    
    # Save overall comparison summary
    comparison_summary = {
        'run_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_folder': run_folder,
        'classification_task': get_label_description(args.labels),
        'models_tested': model_names,
        'comparison_results': all_model_results,
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

def main():
    parser = argparse.ArgumentParser(
        description="Train a 3D‐CNN on sMRI volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  Simple3DCNN        - Simple 3D CNN baseline
  ResNet18_3D        - 3D ResNet-18 (requires MONAI)
  DenseNet121_3D     - 3D DenseNet-121 (requires MONAI)
  EfficientNetB0_3D  - 3D EfficientNet-B0 (requires efficientnet_pytorch_3d)

Examples:
  # Run all models (default) - Memory optimized for 24GB GPU
  python train_smri.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root /path/to/data --labels 0 1 --batch_size 8

  # Run with smaller batch size if out of memory
  python train_smri.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root /path/to/data --labels 0 1 --batch_size 4

  # Run with pretrained models (recommended for better performance)
  python train_smri.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root /path/to/data --labels 0 1 --use_pretrained

  # Run single model with pretrained weights
  python train_smri.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root /path/to/data --labels 0 1 --model EfficientNetB0_3D --use_pretrained

  # Run specific models
  python train_smri.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root /path/to/data --labels 0 1 --models Simple3DCNN EfficientNetB0_3D

  # Run with balanced dataset (reduce majority classes)
  python train_smri.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root /path/to/data --labels 0 1 --balance_dataset

  # Explicitly run all models
  python train_smri.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root /path/to/data --labels 0 1 --run_all
        """
    )
    parser.add_argument("--master_csv", type=str, default="~/reseng202500013-ndd-ml/data/mri_labels.csv",
                        help="Path to master labels CSV file")
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Folder containing sMRI NIfTIs, e.g. data/preprocessed/sMRI")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=8,
                        help="Batch size (reduce to 4-6 if out of memory)")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Number of base channels for CNN models (default: 32)")
    parser.add_argument("--use_pretrained", action='store_true',
                        help="Use pretrained weights for ResNet, DenseNet, and EfficientNet models")
    parser.add_argument("--checkpoint_dir", type=str, default="~/reseng202500013-ndd-ml/data/checkpoints_ad_cn")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--k_folds",     type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--labels",      type=int, nargs='+', required=True,
                        help="Labels to include in training (e.g., 0 1 for CN vs AD)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Proportion of balanced data for validation set (default: 0.2 for 70/20/10 split)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Proportion of balanced data for test set (default: 0.1 for 70/20/10 split)")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducible splits (None for random)")
    parser.add_argument("--balance_dataset", action='store_true',
                        help="Use new balancing strategy: undersample -> split 70/20/10 -> add remaining subjects to test set")
    
    # New arguments for model selection
    parser.add_argument("--model",       type=str, default=None,
                        help="Single model to train (e.g., 'Simple3DCNN', 'ResNet18_3D', 'DenseNet121_3D', 'EfficientNetB0_3D', 'VisionTransformer3D', 'SwinUNETRClassifier', 'FullSwinUNETRClassifier')")
    parser.add_argument("--models",      type=str, nargs='+', default=None,
                        help="Specific models to train (e.g., 'Simple3DCNN' 'EfficientNetB0_3D')")
    parser.add_argument("--run_all",     action='store_true',
                        help="Run all available models (default behavior)")
    
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
