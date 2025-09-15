#!/usr/bin/env python3
"""
SPECT Deep Learning Training Script
Optimized for training CN vs PD classification models on preprocessed SPECT data

Features:
- Automatic data loading and validation
- Multiple model architectures (Simple3DCNN, ResNet3D, EfficientNet3D)
- Comprehensive training pipeline with validation
- Automatic checkpointing and model saving
- Performance monitoring and logging
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import SPECTDataset, SPECTDatasetBalanced, split_spect_dataset
from models_spect import get_spect_model, get_model_summary

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


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration for SPECT training."""
    return {
        'data_root': '/Volumes/reseng202500013-ndd-ml/data/Final_SPECT',
        'output_dir': '/Volumes/reseng202500013-ndd-ml/data/Final_SPECT/training_output',
        'model_type': 'simple',
        'model_params': {
            'base_channels': 16,
            'dropout_rate': 0.5
        },
        'num_classes': 2,
        'batch_size': 4,
        'num_workers': 2,
        'epochs': 100,
        'learning_rate': 1e-4,
        'optimizer': 'adam',
        'weight_decay': 1e-5,
        'scheduler': 'step',
        'lr_step_size': 30,
        'lr_gamma': 0.1,
        'use_class_weights': True,
        'gradient_clip': 1.0,
        'early_stopping_patience': 20,
        'log_interval': 10,
        'use_gpu': True,
        'random_seed': 42,
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1
    }


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SPECT deep learning models')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--data_root', type=str, help='Path to SPECT data directory')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--model_type', type=str, choices=['simple', 'resnet', 'efficient'], help='Model type')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.data_root:
        config['data_root'] = args.data_root
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.model_type:
        config['model_type'] = args.model_type
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # Save configuration
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / 'config.yaml')
    
    # Create trainer and start training
    trainer = SPECTTrainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
