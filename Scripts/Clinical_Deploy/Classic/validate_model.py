#!/usr/bin/env python3
"""
Model Validation Script for Optimized SVM
========================================

This script validates the optimized SVM model and provides:
- Prediction confidence scores
- Clinical interpretation
- Model performance on new data
- Feature importance analysis
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelValidator:
    """Validate and interpret the optimized SVM model."""
    
    def __init__(self, model_path, scaler_path, feature_importance_path):
        """
        Initialize the model validator.
        
        Args:
            model_path (str): Path to optimized SVM model
            scaler_path (str): Path to fitted scaler
            feature_importance_path (str): Path to feature importance CSV
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_importance_path = feature_importance_path
        
        # Load model and scaler
        self.load_model()
        self.load_feature_importance()
        
    def load_model(self):
        """Load the optimized SVM model and scaler."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Loaded SVM model: {self.model}")
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Loaded scaler: {type(self.scaler).__name__}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def load_feature_importance(self):
        """Load feature importance rankings."""
        try:
            self.feature_importance = pd.read_csv(self.feature_importance_path)
            print(f"✓ Loaded feature importance for {len(self.feature_importance)} features")
        except Exception as e:
            print(f"✗ Error loading feature importance: {e}")
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
        probabilities = self.model.predict_proba(X_scaled)
        
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
        if prediction == 1:
            diagnosis = "Positive for neurodegenerative disease"
            recommendation = "Consider further clinical evaluation"
        else:
            diagnosis = "Negative for neurodegenerative disease"
            recommendation = "Continue routine monitoring"
        
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
            print(f"✓ Loaded test data: {test_data.shape}")
            
            # Separate features and target
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            
            # Make predictions
            results = self.predict_with_confidence(X_test.values)
            
            # Calculate metrics
            accuracy = np.mean(results['predictions'] == y_test)
            auc = roc_auc_score(y_test, results['probabilities'][:, 1])
            
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
            
            return {
                'accuracy': accuracy,
                'auc': auc,
                'high_confidence_rate': np.mean(results['high_confidence_mask']),
                'predictions': results['predictions'],
                'probabilities': results['probabilities']
            }
            
        except Exception as e:
            print(f"✗ Error validating model: {e}")
            return None
    
    def generate_validation_report(self, output_dir):
        """Generate a comprehensive validation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create validation report
        report = f"""
# Optimized SVM Model Validation Report

## Model Information
- Model Type: {type(self.model).__name__}
- Model Parameters: {self.model.get_params()}
- Scaler Type: {type(self.scaler).__name__}

## Feature Importance Summary
- Total Features: {len(self.feature_importance) if self.feature_importance is not None else 'Unknown'}
- Top 5 Features: {list(self.feature_importance['feature'].head()) if self.feature_importance is not None else 'Unknown'}

## Clinical Usage Guidelines

### High Confidence Predictions (≥80%)
- Use for clinical decision making
- High reliability for diagnosis

### Medium Confidence Predictions (60-80%)
- Use with caution
- Consider additional clinical context

### Low Confidence Predictions (<60%)
- Do not use for clinical decisions
- Recommend additional testing

## Model Limitations
- Trained on binary classification (0/1)
- Requires same feature preprocessing
- Performance may vary on different populations

## Recommendations
1. Validate on independent dataset
2. Monitor performance over time
3. Update model with new data periodically
4. Use ensemble model as backup for critical decisions
        """
        
        # Save report
        with open(output_dir / 'validation_report.md', 'w') as f:
            f.write(report)
        
        print(f"✓ Validation report saved to: {output_dir / 'validation_report.md'}")

def main():
    """Main validation function."""
    
    # Paths
    model_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_svm_model.pkl")
    scaler_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_scaler.pkl")
    feature_importance_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_feature_importance.csv")
    test_data_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv")
    output_dir = os.path.expanduser("~/reseng202500013-ndd-ml/data/model_validation")
    
    print("Starting Model Validation...")
    print(f"Model: {model_path}")
    print(f"Test Data: {test_data_path}")
    print("=" * 50)
    
    # Initialize validator
    validator = ModelValidator(model_path, scaler_path, feature_importance_path)
    
    # Validate on test data
    results = validator.validate_on_test_data(test_data_path)
    
    # Generate validation report
    validator.generate_validation_report(output_dir)
    
    print(f"\n✓ Validation completed!")
    print(f"Results saved to: {output_dir}")
    
    # Example prediction
    print(f"\n=== Example Prediction ===")
    # Load some test data for example
    test_data = pd.read_csv(test_data_path)
    sample_features = test_data.drop(columns=['label']).iloc[0:1].values
    sample_names = test_data.drop(columns=['label']).columns.tolist()
    
    prediction_results = validator.predict_with_confidence(sample_features)
    interpretation = validator.interpret_prediction(
        sample_features[0], 
        prediction_results['predictions'][0],
        prediction_results['probabilities'][0],
        sample_names
    )
    
    print(f"Sample Prediction: {interpretation['diagnosis']}")
    print(f"Confidence: {interpretation['confidence']} ({interpretation['probability']:.3f})")
    print(f"Recommendation: {interpretation['recommendation']}")

if __name__ == "__main__":
    main() 