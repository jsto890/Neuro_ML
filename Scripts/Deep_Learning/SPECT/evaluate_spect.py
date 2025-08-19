#!/usr/bin/env python3
"""
SPECT Model Evaluation and Inference Script
Comprehensive evaluation of trained SPECT models with detailed performance analysis

Features:
- Model loading and inference
- Performance metrics calculation
- Visualization of results
- Prediction generation
- Model comparison
- Error analysis
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
import nibabel as nib

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import SPECTDataset
from models_spect import get_spect_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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


class SPECTEvaluator:
    """
    Comprehensive evaluator for SPECT deep learning models.
    Handles model evaluation, inference, and performance analysis.
    """
    
    def __init__(self, 
                 model_path: str,
                 data_root: str,
                 output_dir: str,
                 device: Optional[str] = None):
        """
        Initialize SPECT evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            data_root: Path to SPECT data directory
            output_dir: Directory to save evaluation results
            device: Device to use for evaluation (auto-detect if None)
        """
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        set_random_seeds(42)
        
        # Initialize components
        self.model = None
        self.config = None
        self.test_dataset = None
        self.test_loader = None
        
        logger.info(f"SPECT Evaluator initialized with device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_model(self) -> nn.Module:
        """Load trained model from checkpoint."""
        logger.info("Loading trained model...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract configuration
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            logger.info("Configuration loaded from checkpoint")
        else:
            logger.warning("No configuration found in checkpoint, using defaults")
            self.config = self._create_default_config()
        
        # Create model
        model_type = self.config.get('model_type', 'simple')
        model_params = self.config.get('model_params', {})
        
        self.model = get_spect_model(
            model_type=model_type,
            num_classes=self.config.get('num_classes', 2),
            **model_params
        )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded successfully")
        else:
            raise ValueError("No model state dict found in checkpoint")
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Print model info
        model_info = self.model.get_model_info()
        logger.info(f"Model loaded: {model_info}")
        
        return self.model
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration if none exists."""
        return {
            'model_type': 'simple',
            'num_classes': 2,
            'model_params': {
                'base_channels': 16,
                'dropout_rate': 0.5
            }
        }
    
    def setup_test_data(self, labels_csv: Optional[str] = None) -> DataLoader:
        """Setup test dataset and data loader."""
        logger.info("Setting up test data...")
        
        # Find test labels CSV
        if labels_csv:
            test_csv = Path(labels_csv)
        else:
            # Look for test labels in output directory
            test_csv = self.output_dir / 'labels' / 'spect_labels_test.csv'
            if not test_csv.exists():
                # Look in data root
                test_csv = self.data_root / 'labels' / 'spect_labels_test.csv'
        
        if not test_csv.exists():
            raise FileNotFoundError(f"Test labels CSV not found. Please provide path to test labels.")
        
        # Create test dataset
        self.test_dataset = SPECTDataset(
            data_root=str(self.data_root),
            labels_csv=str(test_csv),
            validate_data=True
        )
        
        # Create data loader
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,  # Use batch size 1 for evaluation
            shuffle=False,
            num_workers=0,  # No multiprocessing for evaluation
            pin_memory=False
        )
        
        logger.info(f"Test dataset created with {len(self.test_dataset)} samples")
        return self.test_loader
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate model on test dataset."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.test_loader is None:
            raise RuntimeError("Test data not setup. Call setup_test_data() first.")
        
        logger.info("Starting model evaluation...")
        
        # Initialize metrics
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_subject_ids = []
        
        # Evaluation loop
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Get subject ID
                subject_id = self.test_dataset.subjects[batch_idx]
                all_subject_ids.append(subject_id)
                
                # Forward pass
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(output, dim=1)
                
                # Store results
                all_predictions.append(prediction.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                
                # Progress logging
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"Evaluated {batch_idx + 1}/{len(self.test_loader)} samples")
        
        # Convert to numpy arrays
        all_predictions = np.concatenate(all_predictions).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        all_probabilities = np.concatenate(all_probabilities, axis=0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        # Create results dictionary
        results = {
            'metrics': metrics,
            'predictions': all_predictions.tolist(),
            'targets': all_targets.tolist(),
            'probabilities': all_probabilities.tolist(),
            'subject_ids': all_subject_ids,
            'model_path': str(self.model_path),
            'config': self.config
        }
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        logger.info("Model evaluation complete!")
        return results
    
    def _calculate_metrics(self, 
                          targets: np.ndarray, 
                          predictions: np.ndarray, 
                          probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating performance metrics...")
        
        # Basic classification metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets, predictions, average=None
        )
        
        # ROC AUC (for binary classification)
        if len(np.unique(targets)) == 2:
            try:
                auc = roc_auc_score(targets, probabilities[:, 1])
            except:
                auc = 0.0
        else:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'balanced_accuracy': float(balanced_accuracy),
            'precision_cn': float(precision_per_class[0]) if len(precision_per_class) > 0 else 0,
            'precision_pd': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0,
            'recall_cn': float(recall_per_class[0]) if len(recall_per_class) > 0 else 0,
            'recall_pd': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0,
            'f1_cn': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0,
            'f1_pd': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0
        }
        
        logger.info("Performance metrics calculated:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        logger.info("Saving evaluation results...")
        
        # Save metrics as JSON
        metrics_file = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2, default=str)
        
        # Save detailed results as JSON
        results_file = self.output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'subject_id': results['subject_ids'],
            'true_label': results['targets'],
            'predicted_label': results['predictions'],
            'cn_probability': [p[0] for p in results['probabilities']],
            'pd_probability': [p[1] for p in results['probabilities']]
        })
        
        # Add diagnosis labels
        predictions_df['true_diagnosis'] = predictions_df['true_label'].map({0: 'CN', 1: 'PD'})
        predictions_df['predicted_diagnosis'] = predictions_df['predicted_label'].map({0: 'CN', 1: 'PD'})
        
        # Save predictions CSV
        predictions_file = self.output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_file, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate comprehensive visualizations of results."""
        logger.info("Generating visualizations...")
        
        targets = np.array(results['targets'])
        predictions = np.array(results['predictions'])
        probabilities = np.array(results['probabilities'])
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SPECT Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = confusion_matrix(targets, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['CN', 'PD'], yticklabels=['CN', 'PD'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # 2. ROC Curve
        if len(np.unique(targets)) == 2:
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            auc = roc_auc_score(targets, probabilities[:, 1])
            
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {auc:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend(loc="lower right")
            axes[0, 1].grid(True)
        
        # 3. Precision-Recall Curve
        if len(np.unique(targets)) == 2:
            precision, recall, _ = precision_recall_curve(targets, probabilities[:, 1])
            axes[1, 0].plot(recall, precision, color='blue', lw=2)
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve')
            axes[1, 0].grid(True)
        
        # 4. Prediction Distribution
        prediction_df = pd.DataFrame({
            'True Label': ['CN' if t == 0 else 'PD' for t in targets],
            'Predicted Label': ['CN' if p == 0 else 'PD' for p in predictions],
            'PD Probability': probabilities[:, 1]
        })
        
        sns.histplot(data=prediction_df, x='PD Probability', hue='True Label', 
                    bins=20, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title('Distribution of PD Probabilities by True Label')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Metrics Summary
        metrics = results['metrics']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1_score'], metrics['auc']]
        
        bars = ax.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 
                                                         'gold', 'plum'])
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations generated and saved")
    
    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """Predict on a single SPECT image."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Predicting on single image: {image_path}")
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = nib.load(image_path)
        data = img.get_fdata()
        
        # Validate dimensions
        if data.shape != (91, 109, 91):
            raise ValueError(f"Expected image shape (91, 109, 91), got {data.shape}")
        
        # Convert to tensor
        tensor_data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
        tensor_data = tensor_data.to(self.device)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_data)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1)
        
        # Format results
        result = {
            'image_path': image_path,
            'predicted_label': int(prediction.cpu().numpy()[0]),
            'predicted_diagnosis': 'CN' if prediction.cpu().numpy()[0] == 0 else 'PD',
            'cn_probability': float(probabilities.cpu().numpy()[0, 0]),
            'pd_probability': float(probabilities.cpu().numpy()[0, 1]),
            'confidence': float(torch.max(probabilities).cpu().numpy())
        }
        
        logger.info(f"Prediction: {result['predicted_diagnosis']} "
                   f"(CN: {result['cn_probability']:.3f}, PD: {result['pd_probability']:.3f})")
        
        return result
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        metrics = results['metrics']
        
        report = f"""
# SPECT Model Evaluation Report

## Model Information
- **Model Path**: {results['model_path']}
- **Model Type**: {results['config'].get('model_type', 'Unknown')}
- **Number of Classes**: {results['config'].get('num_classes', 'Unknown')}

## Dataset Information
- **Total Test Samples**: {len(results['targets'])}
- **CN Samples**: {sum(1 for t in results['targets'] if t == 0)}
- **PD Samples**: {sum(1 for t in results['targets'] if t == 1)}

## Performance Metrics

### Overall Performance
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
- **AUC**: {metrics['auc']:.4f}

### Per-Class Performance

#### CN (Control) Class
- **Precision**: {metrics['precision_cn']:.4f}
- **Recall**: {metrics['recall_cn']:.4f}
- **F1-Score**: {metrics['f1_cn']:.4f}

#### PD (Parkinson's Disease) Class
- **Precision**: {metrics['precision_pd']:.4f}
- **Recall**: {metrics['recall_pd']:.4f}
- **F1-Score**: {metrics['f1_pd']:.4f}

### Additional Metrics
- **Specificity**: {metrics['specificity']:.4f}
- **Sensitivity**: {metrics['sensitivity']:.4f}
- **Balanced Accuracy**: {metrics['balanced_accuracy']:.4f}

## Prediction Summary
- **Correct Predictions**: {sum(1 for t, p in zip(results['targets'], results['predictions']) if t == p)}
- **Incorrect Predictions**: {sum(1 for t, p in zip(results['targets'], results['predictions']) if t != p)}
- **Overall Error Rate**: {1 - metrics['accuracy']:.4f}

## Files Generated
- **Metrics**: evaluation_metrics.json
- **Detailed Results**: evaluation_results.json
- **Predictions**: predictions.csv
- **Visualizations**: evaluation_plots.png, performance_metrics.png

## Recommendations
"""
        
        # Add recommendations based on performance
        if metrics['accuracy'] >= 0.9:
            report += "- **Excellent Performance**: Model shows excellent classification ability\n"
        elif metrics['accuracy'] >= 0.8:
            report += "- **Good Performance**: Model shows good classification ability with room for improvement\n"
        elif metrics['accuracy'] >= 0.7:
            report += "- **Fair Performance**: Model shows fair classification ability, consider hyperparameter tuning\n"
        else:
            report += "- **Poor Performance**: Model needs significant improvement, consider architecture changes\n"
        
        if metrics['f1_score'] < metrics['accuracy']:
            report += "- **Class Imbalance**: Consider class balancing techniques if not already implemented\n"
        
        if metrics['auc'] < 0.8:
            report += "- **Low AUC**: Model may benefit from threshold optimization\n"
        
        # Save report
        report_file = self.output_dir / 'evaluation_report.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_file}")
        return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained SPECT models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Path to SPECT data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--test_labels', type=str, help='Path to test labels CSV')
    parser.add_argument('--single_image', type=str, help='Path to single image for prediction')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SPECTEvaluator(
        model_path=args.model_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Load model
    evaluator.load_model()
    
    if args.single_image:
        # Single image prediction
        result = evaluator.predict_single_image(args.single_image)
        print(f"Prediction: {result['predicted_diagnosis']}")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        # Full evaluation
        evaluator.setup_test_data(args.test_labels)
        results = evaluator.evaluate_model()
        
        # Generate report
        report = evaluator.generate_report(results)
        print(report)


if __name__ == "__main__":
    main()
