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
    matthews_corrcoef
)
from torch.utils.data import DataLoader
import json
from pathlib import Path
import pandas as pd

from dataset import PETDataset
from models_pet import Simple3DCNN

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

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

def evaluate_model(model, test_loader, device='cpu', label_mapping=None):
    """Evaluate model and return predictions and metrics."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for pet, labels in test_loader:
            pet = pet.to(device)
            
            # Apply label mapping if provided
            if label_mapping is not None:
                labels = torch.tensor([label_mapping[label.item()] for label in labels], device=device)
            
            logits = model(pet)
            probabilities = nn.Softmax(dim=1)(logits)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)

def calculate_metrics(predictions, probabilities, labels):
    """Calculate comprehensive evaluation metrics."""
    # Ensure labels are in the correct format for binary classification
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2 and probabilities.shape[1] == 2:
        # For binary classification, ensure labels are in {0, 1} format
        if not (0 in unique_labels and 1 in unique_labels):
            # Map labels to {0, 1} if they're not already
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
            mapped_labels = np.array([label_mapping[label] for label in labels])
            mapped_predictions = np.array([label_mapping[pred] for pred in predictions])
        else:
            mapped_labels = labels
            mapped_predictions = predictions
    else:
        mapped_labels = labels
        mapped_predictions = predictions
    
    # Basic metrics
    accuracy = accuracy_score(mapped_labels, mapped_predictions)
    precision = precision_score(mapped_labels, mapped_predictions, average='weighted', zero_division=0)
    recall = recall_score(mapped_labels, mapped_predictions, average='weighted', zero_division=0)
    f1 = f1_score(mapped_labels, mapped_predictions, average='weighted', zero_division=0)
    
    # AUC for binary classification
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(mapped_labels, probabilities[:, 1])
    else:
        auc = roc_auc_score(mapped_labels, probabilities, multi_class='ovr')
    
    # Confusion matrix
    cm = confusion_matrix(mapped_labels, mapped_predictions)
    
    # Classification report
    report = classification_report(mapped_labels, mapped_predictions, output_dict=True, zero_division=0)
    
    mcc = matthews_corrcoef(mapped_labels, mapped_predictions)
    
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
    
    # Disease label mapping
    label_to_disease = {0: 'CN', 1: 'AD', 2: 'PD'}
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Evaluation Results - Disease Classification (PET)', fontsize=16, fontweight='bold')
    
    # 1. ROC Curve
    ax1 = axes[0, 0]
    if probabilities.shape[1] == 2:
        # Ensure labels are in {0, 1} format for ROC curve
        unique_labels = np.unique(labels)
        if len(unique_labels) == 2 and not (0 in unique_labels and 1 in unique_labels):
            # Map labels to {0, 1} if they're not already
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
            mapped_labels = np.array([label_mapping[label] for label in labels])
        else:
            mapped_labels = labels
            
        fpr, tpr, _ = roc_curve(mapped_labels, probabilities[:, 1])
        ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', alpha=0.8)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'ROC Curve\n(Not available for multi-class)', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('ROC Curve')
    
    # 2. Precision-Recall Curve
    ax2 = axes[0, 1]
    if probabilities.shape[1] == 2:
        # Use the same mapped labels for consistency
        precision_curve, recall_curve, _ = precision_recall_curve(mapped_labels, probabilities[:, 1])
        ax2.plot(recall_curve, precision_curve, color='green', lw=2)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Precision-Recall Curve\n(Not available for multi-class)', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Precision-Recall Curve')
    
    # 3. Confusion Matrix
    ax3 = axes[0, 2]
    cm = metrics['confusion_matrix']
    # Get disease labels for the confusion matrix
    n_classes = len(cm)
    disease_labels = [label_to_disease.get(i, f'Class {i}') for i in range(n_classes)]
    
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
    if probabilities.shape[1] == 2:
        # For binary classification, show probability distribution
        positive_probs = probabilities[:, 1]
        ax6.hist(positive_probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax6.set_xlabel('Probability of Positive Class')
        ax6.set_ylabel('Count')
        ax6.set_title('Probability Distribution')
        ax6.grid(True, alpha=0.3)
    else:
        # For multi-class, show max probability distribution
        max_probs = np.max(probabilities, axis=1)
        ax6.hist(max_probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax6.set_xlabel('Maximum Probability')
        ax6.set_ylabel('Count')
        ax6.set_title('Probability Distribution')
        ax6.grid(True, alpha=0.3)
    
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
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, args.num_classes, args.device)
    
    # Create test dataset
    print(f"Loading test data from: {args.test_csv}")
    test_dataset = PETDataset(csv_path=args.test_csv, data_root=args.data_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Evaluate model
    print("Running evaluation...")
    predictions, probabilities, labels = evaluate_model(model, test_loader, args.device)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, probabilities, labels)
    
    # Create plots
    print("Generating evaluation plots...")
    create_evaluation_plots(predictions, probabilities, labels, metrics, args.output_dir)
    
    print(f"\n✅ Evaluation completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 