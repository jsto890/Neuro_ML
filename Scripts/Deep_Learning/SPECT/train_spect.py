#!/usr/bin/env python3
"""
SPECT Deep Learning Training Script with K-Fold Cross-Validation
Optimized for training CN vs PD classification models on preprocessed SPECT data

Features:
- K-fold cross-validation with stratified splits
- New data balancing strategy (undersampling with removed subjects added to test set)
- Multiple model architectures (Simple3DCNN, ResNet3D, EfficientNet3D)
- Comprehensive training pipeline with validation
- Automatic checkpointing and model saving
- Performance monitoring and logging
- Statistical analysis and summary reporting
- Support for both local training and HPC deployment
"""

import os
import sys
import argparse
import yaml
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from datetime import datetime
import uuid
import shutil
import contextlib
import math

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import SPECTDataset, SPECTDatasetBalanced, split_spect_dataset
from models_spect import get_spect_model, get_model_summary

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
    metrics_file = os.path.join(args.checkpoint_dir, "training_metrics.csv")
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(metrics_file)
    
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if file is new
        if not file_exists:
            writer.writerow([
                'run_id', 'timestamp', 'model_name', 'labels', 'epochs', 'batch_size', 
                'learning_rate', 'weight_decay', 'best_val_auc', 'best_val_acc', 
                'final_train_loss', 'final_train_acc', 'notes'
            ])
        
        # Write metrics
        writer.writerow([
            run_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name,
            get_label_description(args.labels), args.epochs, args.batch_size,
            args.learning_rate, args.weight_decay, best_val_auc, best_val_acc,
            final_train_loss, final_train_acc, notes
        ])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spect_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SPECTTrainer:
    """
    Comprehensive trainer for SPECT deep learning models.
    Handles training, validation, and evaluation with proper logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SPECT trainer.
        
        Args:
            config: Configuration dictionary containing training parameters
        """
        self.config = config
        self.device = self._setup_device()
        
        # Set random seeds
        set_random_seeds(config.get('random_seed', 42))
        
        # Setup paths
        self.data_root = Path(config['data_root'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.training_history = []
        
        logger.info(f"SPECT Trainer initialized with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device (GPU/CPU)."""
        if torch.cuda.is_available() and self.config.get('use_gpu', True):
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for training")
        
        return device
    
    def _setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data loaders for training, validation, and testing."""
        logger.info("Setting up data loaders...")
        
        # Check if labels exist, otherwise create them
        labels_dir = self.output_dir / 'labels'
        if not labels_dir.exists() or not list(labels_dir.glob('*.csv')):
            logger.info("Creating dataset splits...")
            split_spect_dataset(
                data_root=str(self.data_root),
                output_dir=str(labels_dir),
                train_ratio=self.config.get('train_ratio', 0.7),
                val_ratio=self.config.get('val_ratio', 0.2),
                test_ratio=self.config.get('test_ratio', 0.1),
                random_seed=self.config.get('random_seed', 42)
            )
        
        # Load datasets
        train_dataset = SPECTDataset(
            data_root=str(self.data_root),
            labels_csv=str(labels_dir / 'spect_labels_train.csv'),
            validate_data=True
        )
        
        val_dataset = SPECTDataset(
            data_root=str(self.data_root),
            labels_csv=str(labels_dir / 'spect_labels_val.csv'),
            validate_data=True
        )
        
        test_dataset = SPECTDataset(
            data_root=str(self.data_root),
            labels_csv=str(labels_dir / 'spect_labels_test.csv'),
            validate_data=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Training: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Testing: {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def _setup_model(self) -> nn.Module:
        """Setup the SPECT model."""
        logger.info("Setting up model...")
        
        model = get_spect_model(
            model_type=self.config.get('model_type', 'simple'),
            num_classes=self.config.get('num_classes', 2),
            **self.config.get('model_params', {})
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Print model summary
        logger.info(f"Model created: {model.get_model_info()}")
        
        return model
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, and loss function."""
        logger.info("Setting up training components...")
        
        # Loss function
        if self.config.get('use_class_weights', True):
            # Calculate class weights from training data
            train_loader, _, _ = self._setup_data()
            class_weights = train_loader.dataset.get_class_weights().to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-3),
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_name = self.config.get('scheduler', 'step').lower()
        if scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 30),
                gamma=self.config.get('lr_gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.get('lr_factor', 0.5),
                patience=self.config.get('lr_patience', 10),
                verbose=True
            )
        
        logger.info(f"Training components setup complete:")
        logger.info(f"  Optimizer: {type(self.optimizer).__name__}")
        logger.info(f"  Scheduler: {type(self.scheduler).__name__}")
        logger.info(f"  Loss: {type(self.criterion).__name__}")
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Progress logging
            if batch_idx % self.config.get('log_interval', 10) == 0:
                logger.info(f"Train Epoch: {self.current_epoch} "
                          f"[{batch_idx}/{len(train_loader)} "
                          f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                          f"Loss: {loss.item():.6f}")
        
        # Calculate epoch statistics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Store predictions and probabilities
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(torch.softmax(output, dim=1).cpu().numpy())
        
        # Calculate epoch statistics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # ROC AUC (for binary classification)
        if len(np.unique(all_targets)) == 2:
            try:
                auc = roc_auc_score(all_targets, [p[1] for p in all_probabilities])
            except:
                auc = 0.0
        else:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if this is the best so far
        if is_best:
            best_checkpoint_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"New best model saved with validation metric: {self.best_val_metric:.4f}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup components
        train_loader, val_loader, test_loader = self._setup_data()
        self.model = self._setup_model()
        self._setup_training_components()
        
        # Load checkpoint if resuming
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.get('epochs', 100)):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self._train_epoch(train_loader)
            
            # Validate
            val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch}: "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                       f"LR: {current_lr:.6f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Store training history
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': current_lr
            }
            epoch_data.update(val_metrics)
            self.training_history.append(epoch_data)
            
            # Check if this is the best model
            is_best = val_metrics['accuracy'] > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics['accuracy']
            
            # Save checkpoint
            self._save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config.get('early_stopping_patience', 0) > 0:
                if len(self.training_history) >= self.config['early_stopping_patience']:
                    recent_metrics = [h['val_accuracy'] for h in self.training_history[-self.config['early_stopping_patience']:]]
                    if all(m <= self.best_val_metric for m in recent_metrics):
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        # Final evaluation on test set
        logger.info("Training complete. Evaluating on test set...")
        test_metrics = self._validate_epoch(test_loader)
        
        logger.info(f"Final Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save final results
        results = {
            'test_metrics': test_metrics,
            'training_history': self.training_history,
            'best_val_metric': self.best_val_metric,
            'final_epoch': self.current_epoch
        }
        
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Close tensorboard writer
        self.writer.close()
        
        logger.info("Training complete!")


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
        models_to_run = ["Simple3DCNN", "ResNet3D", "EfficientNet3D"]

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
            train_dataset = SPECTDataset(csv_path=temp_train_csv, data_root=args.data_root)
            val_dataset = SPECTDataset(csv_path=temp_val_csv, data_root=args.data_root)
            test_dataset = SPECTDataset(csv_path=temp_test_csv, data_root=args.data_root)
            
            # Initialize model for this fold
            unique_labels = sorted(train_dataset.df['label'].unique())
            num_classes = len(unique_labels)
            print(f"[INFO] Model initialized with {num_classes} classes for labels: {unique_labels}")
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            print(f"[INFO] Label mapping: {label_mapping}")
            
            # Create model based on type
            if model_name == "Simple3DCNN":
                model = get_spect_model(
                    model_type='simple',
                    num_classes=num_classes,
                    base_channels=args.base_channels
                )
            elif model_name == "ResNet3D":
                model = get_spect_model(
                    model_type='resnet',
                    num_classes=num_classes,
                    base_channels=args.base_channels
                )
            elif model_name == "EfficientNet3D":
                model = get_spect_model(
                    model_type='efficient',
                    num_classes=num_classes,
                    base_channels=args.base_channels
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Move model to device
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Create trainer for this fold
            config = {
                'data_root': args.data_root,
                'output_dir': model_dir,
                'model_type': model_name.lower().replace('3d', ''),
                'num_classes': num_classes,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'use_gpu': args.device == 'cuda',
                'random_seed': args.random_seed,
                'early_stopping_patience': args.early_stopping_patience,
                'gradient_clip': args.grad_clip_max_norm,
                'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio
            }
            
    trainer = SPECTTrainer(config)
            trainer.model = model
            trainer.device = device
            
            # Train the model
            try:
                trainer.train()
                
                # Evaluate on test set
                test_metrics = trainer._validate_epoch(DataLoader(
                    test_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=False, 
                    num_workers=args.num_workers
                ))
                
                fold_results.append({
                    'fold': fold_idx,
                    'test_metrics': test_metrics,
                    'model_name': model_name
                })
                
                print(f"Fold {fold_idx} completed. Test AUC: {test_metrics.get('auc', 0):.4f}")
                
            except Exception as e:
                print(f"Error in fold {fold_idx}: {e}")
                continue

        # Save fold results for this model
        if fold_results:
            results_file = os.path.join(model_dir, f"{model_name}_fold_results.json")
            with open(results_file, 'w') as f:
                json.dump(fold_results, f, indent=2, default=str)
            
            # Calculate summary statistics
            test_aucs = [r['test_metrics'].get('auc', 0) for r in fold_results]
            test_accs = [r['test_metrics'].get('accuracy', 0) for r in fold_results]
            
            summary_stats = {
                'model_name': model_name,
                'num_folds': len(fold_results),
                'test_auc_stats': compute_summary_stats(test_aucs),
                'test_acc_stats': compute_summary_stats(test_accs),
                'fold_results': fold_results
            }
            
            all_model_results.append(summary_stats)
            
            print(f"\n{model_name} Summary:")
            print(f"  Test AUC: {summary_stats['test_auc_stats']['mean']:.4f} ± {summary_stats['test_auc_stats']['std']:.4f}")
            print(f"  Test Acc: {summary_stats['test_acc_stats']['mean']:.2f}% ± {summary_stats['test_acc_stats']['std']:.2f}%")

    # Save overall results
    overall_results_file = os.path.join(run_dir, "overall_results.json")
    with open(overall_results_file, 'w') as f:
        json.dump(all_model_results, f, indent=2, default=str)

    # Clean up temporary files
    print(f"\nCleaning up {len(temp_files_this_run)} temporary CSV files...")
    for temp_file in temp_files_this_run:
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"Warning: Could not remove {temp_file}: {e}")

    print(f"\nTraining completed! Results saved to: {run_dir}")
    return all_model_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SPECT models with k-fold cross-validation')
    
    # Data arguments
    parser.add_argument("--master_csv", type=str, required=True,
                        help="Path to master CSV file with subject IDs and labels")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing preprocessed SPECT data")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory to save model checkpoints and results")
    
    # Model arguments
    parser.add_argument("--labels", nargs='+', type=int, required=True,
                        help="List of label values to use for classification")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to train (alternative to --models)")
    parser.add_argument("--models", nargs='+', type=str, default=None,
                        choices=['Simple3DCNN', 'ResNet3D', 'EfficientNet3D'],
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
                        help="Balance dataset before k-fold CV (undersampling; leftovers discarded)")
    
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
    available_models = ["Simple3DCNN", "ResNet3D", "EfficientNet3D"]
    
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
