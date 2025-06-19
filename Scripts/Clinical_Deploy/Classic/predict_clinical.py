#!/usr/bin/env python3
"""
Clinical Prediction Script for Optimized SVM
===========================================

Simple script for making clinical predictions using the optimized SVM model.
This is the main script you'll use for clinical applications.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

class ClinicalPredictor:
    """Simple clinical predictor using optimized SVM model."""
    
    def __init__(self):
        """Initialize the clinical predictor."""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the optimized SVM model and scaler."""
        try:
            # Load model
            model_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_svm_model.pkl")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_scaler.pkl")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✓ Model loaded successfully")
            print(f"  Model: {type(self.model).__name__}")
            print(f"  Scaler: {type(self.scaler).__name__}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def predict_patient(self, features, patient_id="Unknown"):
        """
        Predict for a single patient.
        
        Args:
            features (array): Patient features (must match training features)
            patient_id (str): Patient identifier
            
        Returns:
            dict: Prediction results with clinical interpretation
        """
        try:
            # Ensure features is 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Get confidence
            confidence = max(probability)
            
            # Clinical interpretation
            if prediction == 1:
                diagnosis = "POSITIVE - Signs of neurodegenerative disease detected"
                recommendation = "Recommend further clinical evaluation and specialist consultation"
                urgency = "High" if confidence > 0.8 else "Medium"
            else:
                diagnosis = "NEGATIVE - No significant signs of neurodegenerative disease"
                recommendation = "Continue routine monitoring and follow-up"
                urgency = "Low"
            
            # Confidence level
            if confidence >= 0.9:
                confidence_level = "Very High"
            elif confidence >= 0.8:
                confidence_level = "High"
            elif confidence >= 0.7:
                confidence_level = "Moderate"
            else:
                confidence_level = "Low"
            
            return {
                'patient_id': patient_id,
                'prediction': int(prediction),
                'probability': float(confidence),
                'confidence_level': confidence_level,
                'diagnosis': diagnosis,
                'recommendation': recommendation,
                'urgency': urgency,
                'probabilities': {
                    'negative': float(probability[0]),
                    'positive': float(probability[1])
                }
            }
            
        except Exception as e:
            print(f"✗ Error making prediction: {e}")
            return None
    
    def predict_batch(self, data_path, output_path=None):
        """
        Predict for multiple patients from CSV file.
        
        Args:
            data_path (str): Path to CSV with patient features
            output_path (str): Path to save results (optional)
        """
        try:
            # Load data
            data = pd.read_csv(data_path)
            print(f"✓ Loaded {len(data)} patients from {data_path}")
            
            # Check if 'patient_id' column exists, otherwise use index
            if 'patient_id' in data.columns:
                patient_ids = data['patient_id'].tolist()
                features_data = data.drop(columns=['patient_id'])
            else:
                patient_ids = [f"Patient_{i}" for i in range(len(data))]
                features_data = data
            
            # Make predictions
            results = []
            for i, (patient_id, features) in enumerate(zip(patient_ids, features_data.values)):
                result = self.predict_patient(features, patient_id)
                if result:
                    results.append(result)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(data)} patients")
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Save results
            if output_path:
                results_df.to_csv(output_path, index=False)
                print(f"✓ Results saved to {output_path}")
            
            # Print summary
            print(f"\n=== Batch Prediction Summary ===")
            print(f"Total patients: {len(results)}")
            print(f"Positive predictions: {sum(results_df['prediction'])}")
            print(f"Negative predictions: {len(results_df) - sum(results_df['prediction'])}")
            print(f"Average confidence: {results_df['probability'].mean():.3f}")
            
            return results_df
            
        except Exception as e:
            print(f"✗ Error in batch prediction: {e}")
            return None

def main():
    """Main function for clinical predictions."""
    
    print("Clinical Prediction System")
    print("=========================")
    print("Using Optimized SVM Model for Neurodegenerative Disease Detection")
    print()
    
    # Initialize predictor
    predictor = ClinicalPredictor()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Batch prediction mode
        data_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "clinical_predictions.csv"
        
        print(f"Batch prediction mode:")
        print(f"  Input: {data_path}")
        print(f"  Output: {output_path}")
        print()
        
        results = predictor.predict_batch(data_path, output_path)
        
        if results is not None:
            print(f"\n✓ Batch prediction completed!")
            print(f"Results saved to: {output_path}")
    
    else:
        # Interactive mode
        print("Interactive prediction mode")
        print("Enter patient features (comma-separated) or 'quit' to exit")
        print("Expected features: 30 radiomics features")
        print()
        
        while True:
            try:
                # Get input
                user_input = input("Enter features (comma-separated) or 'quit': ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                # Parse features
                features = [float(x.strip()) for x in user_input.split(',')]
                
                if len(features) != 30:
                    print(f"✗ Expected 30 features, got {len(features)}")
                    continue
                
                # Make prediction
                result = predictor.predict_patient(np.array(features))
                
                if result:
                    print(f"\n=== Prediction Results ===")
                    print(f"Patient ID: {result['patient_id']}")
                    print(f"Diagnosis: {result['diagnosis']}")
                    print(f"Confidence: {result['confidence_level']} ({result['probability']:.3f})")
                    print(f"Urgency: {result['urgency']}")
                    print(f"Recommendation: {result['recommendation']}")
                    print(f"Probabilities: Negative={result['probabilities']['negative']:.3f}, Positive={result['probabilities']['positive']:.3f}")
                    print()
                
            except ValueError:
                print("✗ Invalid input. Please enter comma-separated numbers.")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"✗ Error: {e}")

if __name__ == "__main__":
    main() 