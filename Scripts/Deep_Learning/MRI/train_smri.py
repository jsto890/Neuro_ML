# scripts/train_smri.py

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import csv
from datetime import datetime
import uuid
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from dataset import SMRIDataset
from models_smri import Simple3DCNN

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def filter_labels(csv_path, labels):
    """Filter the CSV file to only include specified labels."""
    df = pd.read_csv(csv_path)
    filtered_df = df[df.iloc[:, 1].isin(labels)]
    return filtered_df

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
    log_file_path = os.path.expanduser("~/reseng20215-ndd-ml/data/logging.csv")
    
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

def create_training_plots(folds_data, output_dir="./deep_learning_plots"):
    """Create comprehensive training plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Deep Learning Training Results - AD vs CN Classification', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    for fold_data in folds_data:
        epochs = [d['epoch'] for d in fold_data['data']]
        losses = [d['train_loss'] for d in fold_data['data']]
        ax1.plot(epochs, losses, alpha=0.7, label=f"Fold {fold_data['fold']}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss by Fold')
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
    ax2.set_title('Training Accuracy by Fold')
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
    ax3.set_title('Validation AUC by Fold')
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
    ax4.set_title('Best AUC per Fold')
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
    ax5.set_title('Final Metrics by Fold')
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
    ax6.set_title('Training vs Validation Performance')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'deep_learning_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics
    summary = {
        'total_folds': len(folds_data),
        'average_best_auc': np.mean(best_aucs),
        'std_best_auc': np.std(best_aucs),
        'min_best_auc': np.min(best_aucs),
        'max_best_auc': np.max(best_aucs),
        'fold_results': []
    }
    
    for fold_data in folds_data:
        fold_result = {
            'fold': fold_data['fold'],
            'epochs_trained': len(fold_data['data']),
            'best_val_auc': max([d['val_auc'] for d in fold_data['data']]),
            'final_val_auc': fold_data['data'][-1]['val_auc'],
            'final_val_acc': fold_data['data'][-1]['val_acc'],
            'final_train_acc': fold_data['data'][-1]['train_acc']
        }
        summary['fold_results'].append(fold_result)
    
    # Save summary
    with open(output_path / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DEEP LEARNING TRAINING SUMMARY")
    print("="*60)
    print(f"Total folds: {summary['total_folds']}")
    print(f"Average best AUC: {summary['average_best_auc']:.4f} ± {summary['std_best_auc']:.4f}")
    print(f"AUC range: {summary['min_best_auc']:.4f} - {summary['max_best_auc']:.4f}")
    print("\nFOLD DETAILS:")
    for fold_result in summary['fold_results']:
        print(f"Fold {fold_result['fold']}: {fold_result['epochs_trained']} epochs, "
              f"Best AUC: {fold_result['best_val_auc']:.4f}, "
              f"Final Val Acc: {fold_result['final_val_acc']:.4f}")
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
    
    # Store training history
    training_history = []

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for smri, labels in train_loader:
            smri, labels = smri.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(smri)              # [B, 2]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

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

        # Update learning rate based on validation AUC
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Store epoch data
        epoch_data = {
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_auc': val_auc,
            'val_acc': val_acc,
            'lr': new_lr
        }
        training_history.append(epoch_data)
        
        # Print learning rate change if it occurred
        lr_change = ""
        if new_lr != old_lr:
            lr_change = f"  [LR reduced to {new_lr:.6f}]"

        print(f"Epoch {epoch}/{epochs}  "
              f"Train loss={epoch_loss:.4f}, acc={epoch_acc:.4f}  "
              f"Val AUC={val_auc:.4f}, acc={val_acc:.4f}  "
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
        else:
            no_improvement_count += 1
            
        # Early stopping if no improvement for 20 epochs
        if no_improvement_count >= 20:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Load best model weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_auc, best_val_acc, final_train_loss, final_train_acc, training_history

def k_fold_training(args, k_folds=5):
    """
    Perform k-fold cross validation on training set, with fixed validation set.
    """
    # Create dated folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{timestamp}"
    run_dir = os.path.join(args.checkpoint_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n" + "="*60)
    print(f"STARTING NEW TRAINING RUN: {run_folder}")
    print(f"Output directory: {run_dir}")
    print("="*60)
    
    # Filter the datasets based on labels
    train_df = filter_labels(args.train_csv, args.labels)
    val_df = filter_labels(args.val_csv, args.labels)
    
    # Create temporary CSV files for filtered data
    temp_train_csv = 'temp_train_filtered.csv'
    temp_val_csv = 'temp_val_filtered.csv'
    train_df.to_csv(temp_train_csv, index=False)
    val_df.to_csv(temp_val_csv, index=False)
    
    # Create datasets with filtered data
    train_dataset = SMRIDataset(csv_path=temp_train_csv, data_root=args.data_root)
    val_dataset = SMRIDataset(csv_path=temp_val_csv, data_root=args.data_root)
    
    # Create fixed validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Get labels for stratification
    train_labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
    
    # Initialize stratified k-fold to ensure balanced class distribution
    skfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    folds_data = []  # For plotting
    
    for fold, (train_ids, test_ids) in enumerate(skfold.split(range(len(train_dataset)), train_labels)):
        print(f'\nFOLD {fold + 1}/{k_folds}')
        print(f'Training on {len(train_ids)} subjects, testing on {len(test_ids)} subjects')
        
        # Print class distribution for this fold
        train_labels_fold = [train_labels[i] for i in train_ids]
        test_labels_fold = [train_labels[i] for i in test_ids]
        
        print(f'Training set class distribution:')
        train_counts = pd.Series(train_labels_fold).value_counts().sort_index()
        for label, count in train_counts.items():
            print(f'  Label {label}: {count} subjects ({count/len(train_labels_fold)*100:.1f}%)')
        
        print(f'Test set class distribution:')
        test_counts = pd.Series(test_labels_fold).value_counts().sort_index()
        for label, count in test_counts.items():
            print(f'  Label {label}: {count} subjects ({count/len(test_labels_fold)*100:.1f}%)')
        
        # Create data samplers for this fold
        train_sampler = SubsetRandomSampler(train_ids)
        test_sampler = SubsetRandomSampler(test_ids)
        
        # Create data loaders for this fold
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers
        )
        
        test_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=args.num_workers
        )
        
        # Initialize model for this fold
        model = Simple3DCNN(num_classes=len(args.labels))
        
        # Train the model using training set and test on validation set
        model, best_val_auc, best_val_acc, final_train_loss, final_train_acc, training_history = train_sMRI_model(
            model, train_loader, val_loader, args.epochs, args.device, run_dir, args
        )
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'best_val_auc': best_val_auc,
            'best_val_acc': best_val_acc,
            'final_train_loss': final_train_loss,
            'final_train_acc': final_train_acc
        })
        
        # Store training history for plotting
        folds_data.append({
            'fold': fold + 1,
            'data': training_history
        })
        
        # Log metrics for this fold
        run_id = f"fold_{fold + 1}_{uuid.uuid4().hex[:8]}"
        log_metrics(
            run_id=run_id,
            model_name="Simple3DCNN",
            args=args,
            best_val_auc=best_val_auc,
            best_val_acc=best_val_acc,
            final_train_loss=final_train_loss,
            final_train_acc=final_train_acc,
            notes=f"Fold {fold + 1}/{k_folds}"
        )
    
    # Clean up temporary files
    os.remove(temp_train_csv)
    os.remove(temp_val_csv)
    
    # Print average results across folds
    avg_val_auc = np.mean([r['best_val_auc'] for r in fold_results])
    avg_val_acc = np.mean([r['best_val_acc'] for r in fold_results])
    print(f"\nAverage across {k_folds} folds:")
    print(f"Validation AUC: {avg_val_auc:.4f}")
    print(f"Validation Accuracy: {avg_val_acc:.4f}")
    
    # Create evaluation plots in the run directory
    print("\nGenerating evaluation plots...")
    evaluation_dir = os.path.join(run_dir, "evaluation_plots")
    create_training_plots(folds_data, evaluation_dir)
    print(f"Evaluation plots saved to: {evaluation_dir}")

    # Save folds_data for later plotting
    folds_data_filename = f"folds_data.json"
    folds_data_path = os.path.join(run_dir, folds_data_filename)
    with open(folds_data_path, "w") as f:
        json.dump(folds_data, f)
    print(f"folds_data saved to: {folds_data_path}")
    
    # Create run summary
    run_summary = {
        'timestamp': timestamp,
        'run_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_folder': run_folder,
        'model_filename': f"best_smri_model.pth",
        'folds_data_filename': folds_data_filename,
        'average_val_auc': avg_val_auc,
        'average_val_acc': avg_val_acc,
        'total_folds': len(fold_results),
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'k_folds': args.k_folds,
            'labels': args.labels
        },
        'fold_results': fold_results
    }
    
    # Save run summary
    summary_filename = f"run_summary.json"
    summary_path = os.path.join(run_dir, summary_filename)
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"Run summary saved to: {summary_path}")
    
    print(f"\n" + "="*60)
    print(f"TRAINING RUN COMPLETED: {run_folder}")
    print(f"All outputs saved to: {run_dir}")
    print("="*60)
    
    return fold_results

def main():
    parser = argparse.ArgumentParser(description="Train a 3D‐CNN on sMRI volumes")
    parser.add_argument("--train_csv",   type=str, required=True,
                        help="Path to train_labels.csv")
    parser.add_argument("--val_csv",     type=str, required=True,
                        help="Path to val_labels.csv")
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Folder containing sMRI NIfTIs, e.g. data/preprocessed/sMRI")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--k_folds",     type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--labels",      type=int, nargs='+', required=True,
                        help="Labels to include in training (e.g., 0 1 for CN vs AD)")
    args = parser.parse_args()

    # Perform k-fold cross validation
    fold_results = k_fold_training(args, k_folds=args.k_folds)

if __name__ == "__main__":
    main()
