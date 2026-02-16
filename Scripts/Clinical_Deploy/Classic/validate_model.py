#!/usr/bin/env python3
"""
Model Validation Script (Multiclass CN/AD/PD) for Classical Models
=================================================================

Enhancements:
- Multiclass-aware metrics (accuracy, macro/weighted precision/recall/F1)
- Confusion matrix with disease labels (CN/AD/PD)
- ROC-AUC OvR for multiclass (if probabilities available)
- JSON summary report written to output directory
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
import argparse
import warnings
warnings.filterwarnings('ignore')

class ModelValidator:
    """Validate and interpret a classical model (supports multiclass)."""
    
    def __init__(self, model_path, scaler_path, feature_importance_path, label_map=None):
        """
        Initialize the model validator.
        
        Args:
            model_path (str): Path to optimised SVM model
            scaler_path (str): Path to fitted scaler
            feature_importance_path (str): Path to feature importance CSV
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_importance_path = feature_importance_path
        self.label_map = label_map or {0: 'AD', 1: 'CN', 2: 'PD'}
        
        # Load model and scaler
        self.load_model()
        self.load_feature_importance()
        
    def load_model(self):
        """Load the optimised SVM model and scaler."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f" Loaded SVM model: {self.model}")
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f" Loaded scaler: {type(self.scaler).__name__}")
            
        except Exception as e:
            print(f" Error loading model: {e}")
            sys.exit(1)
    
    def load_feature_importance(self):
        """Load feature importance rankings."""
        try:
            self.feature_importance = pd.read_csv(self.feature_importance_path)
            print(f" Loaded feature importance for {len(self.feature_importance)} features")
        except Exception as e:
            print(f" Error loading feature importance: {e}")
            self.feature_importance = None
    
    def predict_with_confidence(self, X, threshold=0.8):
        """
        Make predictions with confidence scores.
        
        Args:
            X (array): Input features
            threshold (float): Confidence threshold for high-confidence predictions
            
        Returns:
            dict: Predictions, probabilities, and confidence levels
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            # Create a dummy probability array if not available
            probabilities = np.zeros((X_scaled.shape[0], len(getattr(self.model, 'classes_', [0, 1]))))
        
        # Calculate confidence scores
        max_probs = np.max(probabilities, axis=1)
        confidence_levels = []
        
        for prob in max_probs:
            if prob >= threshold:
                confidence_levels.append("High")
            elif prob >= 0.6:
                confidence_levels.append("Medium")
            else:
                confidence_levels.append("Low")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': max_probs,
            'confidence_levels': confidence_levels,
            'high_confidence_mask': max_probs >= threshold
        }
    
    def interpret_prediction(self, features, prediction, probability, feature_names=None):
        """
        Provide clinical interpretation of a prediction.
        
        Args:
            features (array): Input features
            prediction (int): Model prediction (0 or 1)
            probability (array): Prediction probabilities
            feature_names (list): Feature names for interpretation
            
        Returns:
            dict: Clinical interpretation
        """
        # Get top contributing features
        if self.feature_importance is not None and feature_names is not None:
            # Map feature names to importance scores
            feature_contributions = {}
            for i, feature in enumerate(feature_names):
                if feature in self.feature_importance['feature'].values:
                    importance = self.feature_importance[
                        self.feature_importance['feature'] == feature
                    ]['importance'].iloc[0]
                    feature_contributions[feature] = importance * features[i]
            
            # Sort by absolute contribution
            sorted_contributions = sorted(
                feature_contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:5]
        else:
            sorted_contributions = []
        
        # Clinical interpretation
        # Map prediction to disease label if possible
        disease = self.label_map.get(int(prediction), str(prediction))
        if disease in {"AD", "PD"}:
            diagnosis = f"Predicted {disease}"
            recommendation = "Consider further clinical evaluation"
        else:
            diagnosis = f"Predicted {disease}"
            recommendation = "Routine monitoring"
        
        confidence = max(probability)
        if confidence >= 0.9:
            confidence_text = "Very High"
        elif confidence >= 0.8:
            confidence_text = "High"
        elif confidence >= 0.7:
            confidence_text = "Moderate"
        else:
            confidence_text = "Low"
        
        return {
            'diagnosis': diagnosis,
            'confidence': confidence_text,
            'probability': confidence,
            'recommendation': recommendation,
            'top_contributing_features': sorted_contributions
        }
    
    def validate_on_test_data(self, test_data_path, target_column='label'):
        """
        Validate model performance on test data.
        
        Args:
            test_data_path (str): Path to test data
            target_column (str): Name of target column
        """
        try:
            # Load test data
            test_data = pd.read_csv(test_data_path)
            print(f" Loaded test data: {test_data.shape}")
            
            # Separate features and target
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            
            # Make predictions
            results = self.predict_with_confidence(X_test.values)
            
            # Calculate metrics
            accuracy = np.mean(results['predictions'] == y_test)
            # ROC-AUC handling
            auc = 0.0
            try:
                probs = results['probabilities']
                # If multiclass, use OvR; if binary, use positive class
                unique = np.unique(y_test)
                if len(unique) == 2:
                    # Map labels to {0,1} if not already
                    uniq_sorted = sorted(unique)
                    if set(uniq_sorted) != {0, 1}:
                        y_bin = (y_test == uniq_sorted[-1]).astype(int)
                    else:
                        y_bin = y_test
                    if probs.ndim > 1 and probs.shape[1] >= 2:
                        auc = roc_auc_score(y_bin, probs[:, 1])
                else:
                    auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
            except Exception:
                auc = 0.0
            
            print(f"\n=== Model Validation Results ===")
            print(f"Test Accuracy: {accuracy:.3f}")
            print(f"Test AUC: {auc:.3f}")
            print(f"High Confidence Predictions: {np.mean(results['high_confidence_mask']):.1%}")
            
            # Classification report
            print(f"\n=== Classification Report ===")
            print(classification_report(y_test, results['predictions']))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, results['predictions'])
            print(f"\n=== Confusion Matrix ===")
            print(cm)
            
            # Per-class metrics
            prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, results['predictions'], average='weighted', zero_division=0)
            prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, results['predictions'], average='macro', zero_division=0)

            return {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'precision_weighted': float(prec_w),
                'recall_weighted': float(rec_w),
                'f1_weighted': float(f1_w),
                'precision_macro': float(prec_m),
                'recall_macro': float(rec_m),
                'f1_macro': float(f1_m),
                'high_confidence_rate': float(np.mean(results['high_confidence_mask'])),
                'predictions': results['predictions'],
                'probabilities': results['probabilities']
            }
            
        except Exception as e:
            print(f" Error validating model: {e}")
            return None
    
    def generate_validation_report(self, output_dir, metrics: dict, cm: np.ndarray, labels: list):
        """Write a concise JSON report and a markdown summary."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON summary
        summary_json = {
            "model_type": type(self.model).__name__,
            "scaler_type": type(self.scaler).__name__,
            "label_map": self.label_map,
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "labels": labels,
        }
        json_path = output_dir / 'validation_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary_json, f, indent=2)
        print(f" Validation JSON saved to: {json_path}")

        # Minimal markdown
        report_md = f"""
# Classical Model Validation (Multiclass)

Model: {type(self.model).__name__}
Scaler: {type(self.scaler).__name__}

- Accuracy: {metrics.get('accuracy', 0.0):.3f}
- AUC (OvR if multiclass): {metrics.get('auc', 0.0):.3f}
- F1 (weighted/macro): {metrics.get('f1_weighted', 0.0):.3f} / {metrics.get('f1_macro', 0.0):.3f}

Labels: {labels}
Confusion Matrix (rows=actual, cols=pred):
{cm}
"""
        md_path = output_dir / 'validation_report.md'
        with open(md_path, 'w') as f:
            f.write(report_md)
        print(f" Validation report saved to: {md_path}
")

def main():
    parser = argparse.ArgumentParser(description="Validate classical model (multiclass CN/AD/PD)")
    parser.add_argument('--model-path', default='~/reseng202500013-ndd-ml/data/optimised_classical_results/optimised_svm_model.pkl')
    parser.add_argument('--scaler-path', default='~/reseng202500013-ndd-ml/data/optimised_classical_results/optimised_scaler.pkl')
    parser.add_argument('--feature-importance', default='~/reseng202500013-ndd-ml/data/optimised_classical_results/optimised_feature_importance.csv')
    parser.add_argument('--test-data', default='~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv')
    parser.add_argument('--output-dir', default='~/reseng202500013-ndd-ml/clinical_outputs/classical_validation')
    parser.add_argument('--label-map-json', required=False, help='Optional JSON mapping of numeric labels to names')
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    scaler_path = os.path.expanduser(args.scaler_path)
    feature_importance_path = os.path.expanduser(args.feature_importance)
    test_data_path = os.path.expanduser(args.test_data)
    output_dir = os.path.expanduser(args.output_dir)

    # Label map if provided
    label_map = None
    if args.label_map_json:
        try:
            with open(os.path.expanduser(args.label_map_json), 'r') as f:
                raw = json.load(f)
            label_map = {int(k): str(v) for k, v in raw.items()}
        except Exception:
            label_map = None

    print("Starting Model Validation...")
    print(f"Model: {model_path}")
    print(f"Test Data: {test_data_path}")
    print("=" * 50)

    # Initialize validator
    validator = ModelValidator(model_path, scaler_path, feature_importance_path, label_map)

    # Validate on test data
    results = validator.validate_on_test_data(test_data_path)

    # Confusion matrix and labels
    test_df = pd.read_csv(test_data_path)
    y_true = test_df['label'].values.astype(int)
    y_pred = results['predictions']
    cm = confusion_matrix(y_true, y_pred)
    unique_labels_sorted = sorted(list(set(y_true) | set(y_pred)))
    disease_labels = [validator.label_map.get(int(l), str(l)) for l in unique_labels_sorted]

    # Build metrics payload
    metrics_payload = {
        'accuracy': results.get('accuracy', 0.0),
        'auc': results.get('auc', 0.0),
        'precision_weighted': results.get('precision_weighted', 0.0),
        'recall_weighted': results.get('recall_weighted', 0.0),
        'f1_weighted': results.get('f1_weighted', 0.0),
        'precision_macro': results.get('precision_macro', 0.0),
        'recall_macro': results.get('recall_macro', 0.0),
        'f1_macro': results.get('f1_macro', 0.0),
        'high_confidence_rate': results.get('high_confidence_rate', 0.0),
    }

    # Generate validation report files
    validator.generate_validation_report(output_dir, metrics_payload, cm, disease_labels)

    print(f"\n Validation completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 