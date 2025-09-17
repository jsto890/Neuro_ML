#!/usr/bin/env python3
"""
Model Evaluation Script
=======================

Loads a trained model from a .pth file and generates evaluation plots
by running inference on test data.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, auc, average_precision_score
)
from torch.utils.data import DataLoader
import json
from pathlib import Path
import pandas as pd
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize_scalar

from dataset import SPECTDataset
from models_pet import Simple3DCNN

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def temperature_scaling(logits, temperature):
    """
    Apply temperature scaling to logits.
    
    Args:
        logits: Raw model outputs [N, num_classes]
        temperature: Temperature parameter (T > 0)
    
    Returns:
        Calibrated probabilities [N, num_classes]
    """
    return nn.Softmax(dim=1)(logits / temperature)

def find_optimal_temperature(logits, labels, max_iter=1000):
    """
    Find optimal temperature parameter for calibration using validation set.
    
    Args:
        logits: Raw model outputs [N, num_classes]
        labels: Ground truth labels [N]
        max_iter: Maximum iterations for optimization
    
    Returns:
        optimal_temperature: Optimal temperature value
        calibrated_probs: Probabilities after temperature scaling
    """
    def objective(t):
        """Objective function: negative log-likelihood."""
        if t <= 0:
            return 1e10  # Penalty for invalid temperature
        
        # Apply temperature scaling
        probs = temperature_scaling(torch.from_numpy(logits), t).numpy()
        
        # Calculate negative log-likelihood
        nll = 0
        for i, label in enumerate(labels):
            nll -= np.log(probs[i, label] + 1e-15)  # Add small epsilon for numerical stability
        
        return nll
    
    # Find optimal temperature using scipy optimization
    result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded', options={'maxiter': max_iter})
    
    if result.success:
        optimal_temperature = result.x
        calibrated_probs = temperature_scaling(torch.from_numpy(logits), optimal_temperature).numpy()
        return optimal_temperature, calibrated_probs
    else:
        print(f"Warning: Temperature optimization failed. Using default temperature 1.0")
        return 1.0, nn.Softmax(dim=1)(torch.from_numpy(logits)).numpy()

def evaluate_model_with_temperature_scaling(model, test_loader, device, val_loader=None, label_mapping=None, threshold: float | None = None):
    """
    Evaluate model with optional temperature scaling for better calibration.
    
    Args:
        model: trained model
        test_loader: test data loader
        device: device to run inference on
        val_loader: validation data loader for temperature calibration (optional)
        label_mapping: label mapping if provided
        threshold: threshold for binary classification (ignored for multiclass)
    
    Returns:
        tuple: (predictions, probabilities, labels, temperature_info)
    """
    model.eval()
    all_logits = []
    all_probabilities = []
    all_predictions = []
    all_labels = []
    
    # First pass: collect logits and probabilities
    with torch.no_grad():
        for smri, labels in test_loader:
            smri = smri.to(device)
            
            # Apply label mapping if provided
            if label_mapping is not None:
                labels = torch.tensor([label_mapping[label.item()] for label in labels], device=device)
            
            logits = model(smri)
            probabilities = nn.Softmax(dim=1)(logits)
            
            all_logits.append(logits.cpu())
            all_probabilities.append(probabilities.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_probabilities = torch.cat(all_probabilities, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Temperature scaling if validation data is provided
    temperature_info = {'temperature': 1.0, 'calibrated': False}
    
    if val_loader is not None and all_logits.shape[1] > 2:  # Only for multiclass
        print("Applying temperature scaling for multiclass calibration...")
        
        # Collect validation logits and labels
        val_logits = []
        val_labels = []
        
        with torch.no_grad():
            for smri, labels in val_loader:
                smri = smri.to(device)
                
                if label_mapping is not None:
                    labels = torch.tensor([label_mapping[label.item()] for label in labels], device=device)
                
                logits = model(smri)
                val_logits.append(logits.cpu())
                val_labels.append(labels.cpu())
        
        val_logits = torch.cat(val_logits, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        
        # Find optimal temperature
        optimal_temperature, calibrated_probs = find_optimal_temperature(val_logits, val_labels)
        
        # Apply temperature scaling to test set
        test_calibrated_probs = temperature_scaling(torch.from_numpy(all_logits), optimal_temperature).numpy()
        
        # Update probabilities and temperature info
        all_probabilities = test_calibrated_probs
        temperature_info = {
            'temperature': float(optimal_temperature),
            'calibrated': True,
            'validation_logits': val_logits,
            'validation_labels': val_labels
        }
        
        print(f"Temperature scaling applied: T = {optimal_temperature:.3f}")
    
    # Make predictions
    if threshold is not None and all_probabilities.shape[1] == 2:
        # Binary classification with threshold
        pred_pos = (all_probabilities[:, 1] >= threshold).astype(int)
        all_predictions = pred_pos
    else:
        # Multiclass or binary without threshold: use argmax
        all_predictions = np.argmax(all_probabilities, axis=1)
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels), temperature_info

def load_model(model_path, num_classes=2, device='cpu'):
    """Load a trained model from .pth file."""
    model = Simple3DCNN(num_classes=num_classes)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Extract the actual input size from the saved classifier weight
    classifier_weight = state_dict['classifier.0.weight']
    actual_input_size = classifier_weight.shape[1]
    
    # Update the classifier with the correct input size
    model.classifier[0] = nn.Linear(actual_input_size, 256)
    model._initialized = True
    
    # Now load the state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device='cpu', label_mapping=None, threshold: float | None = None, use_temperature_scaling=False, val_loader=None):
    """
    Evaluate model and return predictions and metrics.
    
    Args:
        model: trained model
        test_loader: test data loader
        device: device to run inference on
        label_mapping: label mapping if provided
        threshold: threshold for binary classification
        use_temperature_scaling: whether to apply temperature scaling for multiclass
        val_loader: validation data loader for temperature calibration (required if use_temperature_scaling=True)
    
    Returns:
        tuple: (predictions, probabilities, labels) or (predictions, probabilities, labels, temperature_info)
    """
    if use_temperature_scaling and val_loader is not None:
        return evaluate_model_with_temperature_scaling(
            model, test_loader, device, val_loader, label_mapping, threshold
        )
    else:
        # Original evaluation logic
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for smri, labels in test_loader:
                smri = smri.to(device)
                
                # Apply label mapping if provided
                if label_mapping is not None:
                    labels = torch.tensor([label_mapping[label.item()] for label in labels], device=device)
                
                logits = model(smri)
                probabilities = nn.Softmax(dim=1)(logits)
                if threshold is not None and probabilities.shape[1] == 2:
                    # Use provided threshold on positive class prob for binary case
                    pred_pos = (probabilities[:, 1] >= threshold).long()
                    predictions = pred_pos
                else:
                    predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)

def calculate_metrics(predictions, probabilities, labels):
    """Calculate comprehensive evaluation metrics."""
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # AUC for binary classification
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(labels, probabilities[:, 1])
    else:
        auc = roc_auc_score(labels, probabilities, multi_class='ovr')
    
    # Confusion matrix (ensure all classes appear, even if zero count)
    n_classes = probabilities.shape[1] if len(probabilities.shape) > 1 else int(np.max(predictions)) + 1
    all_class_labels = list(range(n_classes))
    cm = confusion_matrix(labels, predictions, labels=all_class_labels)
    
    # Classification report (ensure all classes included)
    report = classification_report(labels, predictions, labels=all_class_labels, output_dict=True, zero_division=0)
    
    mcc = matthews_corrcoef(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'mcc': mcc,
        'confusion_matrix': cm,
        'classification_report': report
    }

def create_evaluation_plots(predictions, probabilities, labels, metrics, output_dir):
    """Create comprehensive evaluation plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Disease label mapping - Updated for proper CN, AD, PD ordering
    # This assumes labels are in order: [CN, AD, PD] or [0, 1, 2]
    n_classes = probabilities.shape[1]
    if n_classes == 3:
        # For 3-class: CN, AD, PD
        label_to_disease = {0: 'CN', 1: 'AD', 2: 'PD'}
        disease_names = ['CN', 'AD', 'PD']
    elif n_classes == 2:
        # For binary classification
        label_to_disease = {0: 'Class 0', 1: 'Class 1'}
        disease_names = ['Class 0', 'Class 1']
    else:
        # For other multiclass scenarios
        label_to_disease = {i: f'Class {i}' for i in range(n_classes)}
        disease_names = [f'Class {i}' for i in range(n_classes)]
    
    # Create comprehensive plot
    if n_classes == 2:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    else:
        # For multiclass, we need more space for ROC curves
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    fig.suptitle('Model Evaluation Results - Disease Classification', fontsize=16, fontweight='bold')
    
    # 1. ROC Curve
    ax1 = axes[0, 0]
    if n_classes == 2:
        # Binary classification
        try:
            pos_probs = probabilities[:, 1]
        except Exception:
            # Handle shape (N,)
            pos_probs = probabilities
        fpr, tpr, _ = roc_curve(labels, pos_probs)
        ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', alpha=0.8)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve (Binary)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        # Multiclass: One-vs-Rest ROC curves
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i in range(n_classes):
            # One-vs-rest: class i vs all others
            y_true_binary = (labels == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                    label=f'{disease_names[i]} vs Rest (AUC = {roc_auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.8)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves (One-vs-Rest)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax2 = axes[0, 1]
    if n_classes == 2:
        # Binary classification
        try:
            pos_probs = probabilities[:, 1]
        except Exception:
            pos_probs = probabilities
        precision_curve, recall_curve, _ = precision_recall_curve(labels, pos_probs)
        ax2.plot(recall_curve, precision_curve, color='green', lw=2)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve (Binary)')
        ax2.grid(True, alpha=0.3)
    else:
        # Multiclass: One-vs-Rest Precision-Recall curves
        
        for i in range(n_classes):
            y_true_binary = (labels == i).astype(int)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, probabilities[:, i])
            avg_precision = average_precision_score(y_true_binary, probabilities[:, i])
            
            ax2.plot(recall_curve, precision_curve, color=colors[i % len(colors)], lw=2,
                    label=f'{disease_names[i]} vs Rest (AP = {avg_precision:.3f})')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves (One-vs-Rest)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = axes[0, 2]
    cm = metrics['confusion_matrix']
    # Get disease labels for the confusion matrix
    n_classes_actual = len(cm)
    disease_labels = [label_to_disease.get(i, f'Class {i}') for i in range(n_classes_actual)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
               xticklabels=disease_labels, yticklabels=disease_labels)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix')
    
    # 4. Metrics Bar Chart
    ax4 = axes[1, 0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'MCC']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                    metrics['recall'], metrics['f1_score'], metrics['auc'], metrics['mcc']]
    
    bars = ax4.bar(metric_names, metric_values, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_ylabel('Score')
    ax4.set_title('Model Performance Metrics')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    ax4.grid(True, alpha=0.3)
    
    # 5. Prediction Distribution
    ax5 = axes[1, 1]
    unique_labels, counts = np.unique(predictions, return_counts=True)
    # Convert numeric labels to disease names for x-axis
    disease_labels = [label_to_disease.get(label, f'Class {label}') for label in unique_labels]
    
    ax5.bar(range(len(unique_labels)), counts, alpha=0.7, color='lightcoral', edgecolor='black')
    ax5.set_xlabel('Predicted Class')
    ax5.set_ylabel('Count')
    ax5.set_title('Prediction Distribution')
    ax5.set_xticks(range(len(unique_labels)))
    ax5.set_xticklabels(disease_labels)
    ax5.grid(True, alpha=0.3)
    
    # 6. Probability Distribution
    ax6 = axes[1, 2]
    if n_classes == 2:
        # For binary classification, show probability distribution
        try:
            positive_probs = probabilities[:, 1]
        except Exception:
            positive_probs = probabilities
        ax6.hist(positive_probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax6.set_xlabel('Probability of Positive Class')
        ax6.set_ylabel('Count')
        ax6.set_title('Probability Distribution (Binary)')
        ax6.grid(True, alpha=0.3)
    else:
        # For multi-class, show max probability distribution
        max_probs = np.max(probabilities, axis=1)
        ax6.hist(max_probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax6.set_xlabel('Maximum Probability')
        ax6.set_ylabel('Count')
        ax6.set_title('Probability Distribution (Multiclass)')
        ax6.grid(True, alpha=0.3)
    
    # 7. Per-class Performance (only for multiclass)
    if n_classes > 2:
        ax7 = axes[2, 0]
        # Calculate per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        x = np.arange(len(disease_names))
        width = 0.25
        
        ax7.bar(x - width, precision_per_class, width, label='Precision', alpha=0.7)
        ax7.bar(x, recall_per_class, width, label='Recall', alpha=0.7)
        ax7.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.7)
        
        ax7.set_xlabel('Classes')
        ax7.set_ylabel('Score')
        ax7.set_title('Per-Class Performance Metrics')
        ax7.set_xticks(x)
        ax7.set_xticklabels(disease_names)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(0, 1)
        
        # 8. Class Balance Analysis
        ax8 = axes[2, 1]
        unique_labels_actual, counts_actual = np.unique(labels, return_counts=True)
        disease_labels_actual = [label_to_disease.get(label, f'Class {label}') for label in unique_labels_actual]
        
        ax8.bar(range(len(unique_labels_actual)), counts_actual, alpha=0.7, color='lightgreen', edgecolor='black')
        ax8.set_xlabel('Actual Class')
        ax8.set_ylabel('Count')
        ax8.set_title('Class Distribution in Test Set')
        ax8.set_xticks(range(len(unique_labels_actual)))
        ax8.set_xticklabels(disease_labels_actual)
        ax8.grid(True, alpha=0.3)
        
        # 9. Confidence Analysis
        ax9 = axes[2, 2]
        # Show confidence distribution for each class
        for i in range(n_classes):
            class_probs = probabilities[:, i]
            ax9.hist(class_probs, bins=20, alpha=0.5, label=disease_names[i], density=True)
        
        ax9.set_xlabel('Predicted Probability')
        ax9.set_ylabel('Density')
        ax9.set_title('Class-wise Confidence Distribution')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'model_evaluation_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plot saved to: {plot_path}")
    
    # Save metrics
    metrics_path = output_path / 'evaluation_metrics.json'
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'auc': float(metrics['auc']),
        'mcc': float(metrics['mcc']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'classification_report': metrics['classification_report']
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"MCC:       {metrics['mcc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(classification_report(labels, predictions, zero_division=0))
    print("="*60)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model from .pth file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model .pth file")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to test labels CSV file")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./model_evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu/cuda)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes in the model")
    parser.add_argument("--use_temperature_scaling", action="store_true",
                        help="Whether to apply temperature scaling for multiclass calibration")
    parser.add_argument("--val_csv", type=str,
                        help="Path to validation labels CSV file for temperature scaling (required if --use_temperature_scaling is True)")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, args.num_classes, args.device)
    
    # Create test dataset
    print(f"Loading test data from: {args.test_csv}")
    test_dataset = SPECTDataset(csv_path=args.test_csv, data_root=args.data_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Create validation dataset if temperature scaling is used
    val_dataset = None
    val_loader = None
    if args.use_temperature_scaling and args.val_csv:
        print(f"Loading validation data from: {args.val_csv}")
        val_dataset = SPECTDataset(csv_path=args.val_csv, data_root=args.data_root)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Evaluate model
    print("Running evaluation...")
    predictions, probabilities, labels, temperature_info = evaluate_model(
        model, test_loader, args.device, label_mapping=None, threshold=None,
        use_temperature_scaling=args.use_temperature_scaling, val_loader=val_loader
    )
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, probabilities, labels)
    
    # Create plots
    print("Generating evaluation plots...")
    create_evaluation_plots(predictions, probabilities, labels, metrics, args.output_dir)
    
    print(f"\n✅ Evaluation completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 