#!/usr/bin/env python3
"""
Transformer Model Training Script
================================

Specialized training script for 3D Vision Transformers and Swin UNETR models
with advanced features like warmup scheduling, mixed precision training,
and gradient accumulation.
"""

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
import yaml
from torch.amp import GradScaler, autocast

from dataset import SMRIDataset
from models_smri import get_3d_model
from transformer_models import get_transformer_model
from evaluate_model import evaluate_model, calculate_metrics, create_evaluation_plots

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_transformer_config(config_path):
    """Load transformer-specific configuration with proper type conversion."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure proper type conversion for training parameters
    if 'training' in config:
        training_config = config['training']
        
        # Convert numeric values to proper types
        if 'learning_rate' in training_config:
            training_config['learning_rate'] = float(training_config['learning_rate'])
        if 'weight_decay' in training_config:
            training_config['weight_decay'] = float(training_config['weight_decay'])
        if 'warmup_epochs' in training_config:
            training_config['warmup_epochs'] = float(training_config['warmup_epochs'])
        if 'label_smoothing' in training_config:
            training_config['label_smoothing'] = float(training_config['label_smoothing'])
        if 'gradient_accumulation_steps' in training_config:
            training_config['gradient_accumulation_steps'] = int(training_config['gradient_accumulation_steps'])
        if 'eps' in training_config:
            training_config['eps'] = float(training_config['eps'])
        if 'betas' in training_config:
            training_config['betas'] = [float(b) for b in training_config['betas']]
    
    return config

def create_transformer_optimizer(model, config):
    """Create optimizer with transformer-specific settings."""
    # Ensure proper type conversion
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    if config['training']['optimizer'].lower() == 'adamw':
        betas = tuple(config['training']['betas'])
        eps = float(config['training']['eps'])
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    return optimizer

def create_transformer_scheduler(optimizer, config, total_steps):
    """Create learning rate scheduler with warmup for transformers."""
    warmup_steps = int(float(config['training']['warmup_epochs']) * total_steps)
    
    if config['training']['cosine_schedule']:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return warmup_scheduler.get_last_lr()[0] / optimizer.param_groups[0]['lr']
            else:
                return main_scheduler.get_last_lr()[0] / optimizer.param_groups[0]['lr']
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )
    
    return scheduler, warmup_steps

def create_transformer_loss(config):
    """Create loss function with label smoothing for transformers."""
    label_smoothing = float(config['training']['label_smoothing'])
    if label_smoothing > 0:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        return nn.CrossEntropyLoss()

def train_transformer_model(model, train_loader, val_loader, epochs, device, checkpoint_dir, args, config):
    """
    Train transformer model with advanced features.
    """
    # Create optimizer and scheduler
    optimizer = create_transformer_optimizer(model, config)
    total_steps = len(train_loader) * epochs
    scheduler, warmup_steps = create_transformer_scheduler(optimizer, config, total_steps)
    
    # Create loss function
    criterion = create_transformer_loss(config)
    
    # Mixed precision training
    scaler = GradScaler('cuda') if config['training']['mixed_precision'] else None
    
    # Gradient accumulation
    accumulation_steps = int(config['training']['gradient_accumulation_steps'])
    
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
    global_step = 0

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        optimizer.zero_grad()

        for batch_idx, (smri, labels) in enumerate(train_loader):
            smri, labels = smri.to(device), labels.to(device)
            
            # Mixed precision forward pass
            if scaler is not None:
                with autocast('cuda'):
                    logits = model(smri)
                    loss = criterion(logits, labels) / accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update learning rate
                    if scheduler is not None and global_step < warmup_steps:
                        scheduler.step()
            else:
                logits = model(smri)
                loss = criterion(logits, labels) / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update learning rate
                    if scheduler is not None and global_step < warmup_steps:
                        scheduler.step()

            running_loss += loss.item() * smri.size(0) * accumulation_steps
            preds = torch.argmax(logits, dim=1)
            running_corrects += (preds == labels).sum().item()
            total_samples += smri.size(0)
            
            global_step += 1

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
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
                if scaler is not None:
                    with autocast('cuda'):
                        logits = model(smri)
                else:
                    logits = model(smri)
                val_logits.append(logits.cpu().numpy())
                val_labels.append(labels.numpy())

        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        # Calculate metrics
        probs = nn.Softmax(dim=1)(torch.from_numpy(val_logits)).numpy()[:, 1]
        val_auc = roc_auc_score(val_labels, probs)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Additional metrics
        precision, recall, f1, support = precision_recall_fscore_support(val_labels, val_preds, average=None, zero_division=0)
        cm = confusion_matrix(val_labels, val_preds)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)
        
        # Store per-class metrics
        class_metrics = {}
        for i, class_label in enumerate(sorted(set(val_labels))):
            class_metrics[f'class_{class_label}'] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }

        # Update learning rate for non-warmup schedulers
        if scheduler is not None and global_step >= warmup_steps:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        # Store epoch data
        val_mcc = matthews_corrcoef(val_labels, val_preds)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_data = {
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_auc': val_auc,
            'val_acc': val_acc,
            'lr': current_lr,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'val_mcc': val_mcc
        }
        training_history.append(epoch_data)

        print(f"Epoch {epoch}/{epochs} (Step {global_step})  "
              f"Train loss={epoch_loss:.4f}, Train acc={epoch_acc:.4f}  "
              f"Val AUC={val_auc:.4f}, Val acc={val_acc:.4f}  "
              f"LR={current_lr:.6f}")

        # Checkpoint if this is the best AUC so far
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            os.makedirs(checkpoint_dir, exist_ok=True)
            
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
            
        # Early stopping
        if no_improvement_count >= 20:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Load best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_auc, best_val_acc, final_train_loss, final_train_acc, training_history, best_precision_macro, best_recall_macro, best_f1_macro, best_class_metrics, best_confusion_matrix

def main():
    parser = argparse.ArgumentParser(
        description="Train transformer models on sMRI volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available transformer models:
  VisionTransformer3D        - 3D Vision Transformer
  SwinUNETRClassifier        - Swin UNETR for classification
  SwinUNETRClassifier_GradCAM - Swin UNETR with Grad-CAM support

Examples:
  # Train Vision Transformer with automatic splits
  python train_transformers.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root ~/reseng202500013-ndd-ml/data/preprocessed/MRI --labels 0 1 --model VisionTransformer3D --config config_transformers.yaml

  # Train Swin UNETR with custom split ratios
  python train_transformers.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root ~/reseng202500013-ndd-ml/data/preprocessed/MRI --labels 0 1 --model SwinUNETRClassifier --config config_hardware_optimized.yaml --val_ratio 0.2 --test_ratio 0.2

  # Train with reproducible splits
  python train_transformers.py --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv --data_root ~/reseng202500013-ndd-ml/data/preprocessed/MRI --labels 0 1 --model SwinUNETRClassifier --config config_hardware_optimized.yaml --random_seed 42
        """
    )
    
    # Required arguments
    parser.add_argument("--master_csv", type=str, default="~/reseng202500013-ndd-ml/data/mri_labels.csv",
                        help="Path to master labels CSV file")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing sMRI NIfTIs")
    parser.add_argument("--labels", type=int, nargs='+', required=True,
                        help="Labels to include in training (e.g., 0 1 for CN vs AD)")
    parser.add_argument("--model", type=str, required=True,
                        help="Transformer model to train")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to transformer configuration file")
    
    # Optional arguments
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (optimized for RTX 6000)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers (optimized for 128-thread CPU)")
    parser.add_argument("--checkpoint_dir", type=str, default="~/reseng202500013-ndd-ml/data/checkpoints_ad_cn")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="Proportion of data for test set")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Proportion of data for validation set")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducible splits (None for random)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_transformer_config(args.config)
    
    # Create dated folder for this run (same pattern as train_smri.py)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{timestamp}"
    
    # Expand user path and create directory structure
    checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    run_dir = os.path.join(checkpoint_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"TRANSFORMER TRAINING RUN: {run_folder}")
    print(f"Model: {args.model}")
    print(f"Output directory: {run_dir}")
    print(f"{'='*60}")
    
    # Set random seed for reproducible splits (if specified)
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        print(f"Using random seed: {args.random_seed}")
    else:
        print("Using random seed for different subject mix each run")
    
    # Load and filter master dataset
    def load_and_filter_master_dataset(master_csv_path, labels):
        """Load master dataset and filter by specified labels."""
        print(f"Loading master dataset from: {master_csv_path}")
        
        # Load CSV
        df = pd.read_csv(master_csv_path)
        if 'subject_id' not in df.columns or 'label' not in df.columns:
            df = pd.read_csv(master_csv_path, header=None, names=['subject_id', 'label'])
        
        # Drop header rows if present
        df = df[~df['subject_id'].isin(['subject_id', ''])]
        df = df[~df['label'].isin(['label', ''])]
        
        # Convert labels to int and filter
        df['label'] = df['label'].astype(int)
        filtered_df = df[df['label'].isin(labels)]
        
        print(f"Master dataset: {len(df)} total subjects")
        print(f"After filtering for labels {labels}: {len(filtered_df)} subjects")
        
        # Show label distribution
        label_counts = filtered_df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} subjects ({count/len(filtered_df)*100:.1f}%)")
        
        return filtered_df
    
    # Create stratified train/val/test splits
    def create_stratified_splits(df, val_ratio, test_ratio, labels):
        """Create stratified train/val/test splits."""
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=test_ratio, 
            stratify=df['label'], 
            random_state=args.random_seed
        )
        
        # Second split: train vs val
        val_relative_size = val_ratio / (1 - test_ratio)
        train, val = train_test_split(
            train_val, 
            test_size=val_relative_size, 
            stratify=train_val['label'], 
            random_state=args.random_seed
        )
        
        return train, val, test
    
    # Load and filter master dataset
    master_df = load_and_filter_master_dataset(args.master_csv, args.labels)
    
    # Create splits
    train_df, val_df, test_df = create_stratified_splits(
        master_df, args.val_ratio, args.test_ratio, args.labels
    )
    
    # Save splits to data directory
    data_dir = os.path.dirname(args.master_csv)
    temp_train_csv = os.path.join(data_dir, f'temp_train_{run_folder}.csv')
    temp_val_csv = os.path.join(data_dir, f'temp_val_{run_folder}.csv')
    temp_test_csv = os.path.join(data_dir, f'temp_test_{run_folder}.csv')
    
    train_df.to_csv(temp_train_csv, index=False)
    val_df.to_csv(temp_val_csv, index=False)
    test_df.to_csv(temp_test_csv, index=False)
    
    # Create datasets with split data
    train_dataset = SMRIDataset(csv_path=temp_train_csv, data_root=args.data_root)
    val_dataset = SMRIDataset(csv_path=temp_val_csv, data_root=args.data_root)
    test_dataset = SMRIDataset(csv_path=temp_test_csv, data_root=args.data_root)
    
    # Print split information
    print(f"\nDataset splits:")
    print(f"Training set: {len(train_dataset)} subjects ({len(train_dataset)/len(master_df)*100:.1f}%)")
    train_labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
    train_counts = pd.Series(train_labels).value_counts().sort_index()
    for label, count in train_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(train_labels)*100:.1f}%)")
    
    print(f"Validation set: {len(val_dataset)} subjects ({len(val_dataset)/len(master_df)*100:.1f}%)")
    val_labels = [val_dataset.labels[i] for i in range(len(val_dataset))]
    val_counts = pd.Series(val_labels).value_counts().sort_index()
    for label, count in val_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(val_labels)*100:.1f}%)")
    
    print(f"Test set: {len(test_dataset)} subjects ({len(test_dataset)/len(master_df)*100:.1f}%)")
    test_labels = [test_dataset.labels[i] for i in range(len(test_dataset))]
    test_counts = pd.Series(test_labels).value_counts().sort_index()
    for label, count in test_counts.items():
        print(f"  Label {label}: {count} subjects ({count/len(test_labels)*100:.1f}%)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    if args.model.lower() in ["visiontransformer3d", "swinunetrclassifier", "swinunetrclassifier_gradcam"]:
        model = get_transformer_model(
            args.model.lower(),
            num_classes=len(args.labels),
            in_channels=1,
            **config.get(args.model.lower(), {})
        )
    else:
        raise ValueError(f"Unknown transformer model: {args.model}")
    
    print(f"Model initialized with {len(args.labels)} classes: {args.labels}")
    print(f"Expected label range: 0 to {len(args.labels)-1}")
    
    print(f"Model initialized: {args.model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    model, best_val_auc, best_val_acc, final_train_loss, final_train_acc, training_history, best_precision_macro, best_recall_macro, best_f1_macro, best_class_metrics, best_confusion_matrix = train_transformer_model(
        model, train_loader, val_loader, args.epochs, args.device, run_dir, args, config
    )
    
    # Save training results
    results = {
        'model_name': args.model,
        'best_val_auc': best_val_auc,
        'best_val_acc': best_val_acc,
        'final_train_loss': final_train_loss,
        'final_train_acc': final_train_acc,
        'best_precision_macro': best_precision_macro,
        'best_recall_macro': best_recall_macro,
        'best_f1_macro': best_f1_macro,
        'training_history': training_history,
        'config': config
    }
    
    results_path = os.path.join(run_dir, f"{args.model}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    print(f"\nTraining completed!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {run_dir}")
    
    # Test set evaluation (always available now)
    print(f"\nEvaluating on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load best model
    best_model_path = os.path.join(run_dir, "best_smri_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    # Evaluate
    predictions, probabilities, labels = evaluate_model(model, test_loader, args.device)
    metrics = calculate_metrics(predictions, probabilities, labels)
    
    # Save test metrics
    test_metrics_path = os.path.join(run_dir, "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    # Save test plots
    test_eval_dir = os.path.join(run_dir, "test_evaluation_plots")
    create_evaluation_plots(predictions, probabilities, labels, metrics, test_eval_dir)
    
    print(f"Test evaluation completed!")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    
    # Clean up temporary files
    try:
        os.remove(temp_train_csv)
        os.remove(temp_val_csv)
        os.remove(temp_test_csv)
        print(f"Cleaned up temporary files from data directory")
    except:
        pass

if __name__ == "__main__":
    main() 