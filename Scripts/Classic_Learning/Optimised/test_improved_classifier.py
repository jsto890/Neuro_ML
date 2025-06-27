#!/usr/bin/env python3
"""
Test script for the improved optimized classifier.
Compares performance with the original optimized classifier.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_optimized_classifier import ImprovedOptimizedRadiomicsClassifier as ImprovedClassifier
from optimized_classifier import ImprovedOptimizedRadiomicsClassifier as OriginalClassifier

def test_improved_classifier(input_path, output_dir):
    """Test the improved classifier and compare with original."""
    
    print("=" * 60)
    print("TESTING IMPROVED OPTIMIZED CLASSIFIER")
    print("=" * 60)
    
    # Create output directory for improved classifier
    improved_output = Path(output_dir) / "improved_results"
    improved_output.mkdir(parents=True, exist_ok=True)
    
    # Test improved classifier
    print("\n1. Running Improved Optimized Classifier...")
    print("-" * 40)
    
    improved_classifier = ImprovedClassifier(
        input_path=input_path,
        output_dir=str(improved_output),
        random_state=42,
        binary_only=True
    )
    
    # Run the improved pipeline
    improved_success = improved_classifier.run_improved_pipeline()
    
    if improved_success:
        print("\n✅ Improved classifier completed successfully!")
        
        # Display key results
        if hasattr(improved_classifier, 'results') and improved_classifier.results:
            print("\nImproved Classifier Results:")
            print("-" * 30)
            
            for model_name, results in improved_classifier.results.items():
                print(f"\n{model_name}:")
                if 'test' in results:
                    test_results = results['test']
                    print(f"  Test Accuracy: {test_results.get('accuracy', 'N/A'):.4f}")
                    print(f"  Test AUC: {test_results.get('auc', 'N/A'):.4f}")
                    print(f"  Test F1: {test_results.get('f1', 'N/A'):.4f}")
                
                if 'train' in results:
                    train_results = results['train']
                    print(f"  Train Accuracy: {train_results.get('accuracy', 'N/A'):.4f}")
                    print(f"  Train AUC: {train_results.get('auc', 'N/A'):.4f}")
        
        # Display feature engineering results
        if hasattr(improved_classifier, 'feature_engineering_results'):
            print("\nFeature Engineering Results:")
            print("-" * 30)
            
            fe_results = improved_classifier.feature_engineering_results
            if 'feature_selection' in fe_results:
                fs_info = fe_results['feature_selection']
                print(f"Final features: {fs_info.get('n_features', 'N/A')}")
                print(f"Selection method: {fs_info.get('method', 'N/A')}")
            
            if 'outlier_detection' in fe_results:
                od_info = fe_results['outlier_detection']
                print(f"Outliers removed: {od_info.get('n_outliers_removed', 'N/A')}")
                print(f"Method: {od_info.get('method', 'N/A')}")
        
        # Display ensemble information
        if hasattr(improved_classifier, 'ensemble_info'):
            print("\nEnsemble Information:")
            print("-" * 30)
            ensemble_info = improved_classifier.ensemble_info
            print(f"Base models: {ensemble_info.get('base_models', [])}")
            print(f"Meta-learner: {ensemble_info.get('meta_learner', 'N/A')}")
            print(f"Total models: {ensemble_info.get('total_models', 'N/A')}")
            print(f"Regularization: {ensemble_info.get('regularization', 'N/A')}")
        
    else:
        print("\n❌ Improved classifier failed!")
        return False
    
    # Test original classifier for comparison
    print("\n" + "=" * 60)
    print("TESTING ORIGINAL OPTIMIZED CLASSIFIER")
    print("=" * 60)
    
    # Create output directory for original classifier
    original_output = Path(output_dir) / "original_results"
    original_output.mkdir(parents=True, exist_ok=True)
    
    print("\n2. Running Original Optimized Classifier...")
    print("-" * 40)
    
    original_classifier = OriginalClassifier(
        input_path=input_path,
        output_dir=str(original_output),
        random_state=42,
        binary_only=True
    )
    
    # Run the original pipeline
    original_success = original_classifier.run_pipeline()
    
    if original_success:
        print("\n✅ Original classifier completed successfully!")
        
        # Display key results
        if hasattr(original_classifier, 'results') and original_classifier.results:
            print("\nOriginal Classifier Results:")
            print("-" * 30)
            
            for model_name, results in original_classifier.results.items():
                print(f"\n{model_name}:")
                if 'test' in results:
                    test_results = results['test']
                    print(f"  Test Accuracy: {test_results.get('accuracy', 'N/A'):.4f}")
                    print(f"  Test AUC: {test_results.get('auc', 'N/A'):.4f}")
                    print(f"  Test F1: {test_results.get('f1', 'N/A'):.4f}")
                
                if 'train' in results:
                    train_results = results['train']
                    print(f"  Train Accuracy: {train_results.get('accuracy', 'N/A'):.4f}")
                    print(f"  Train AUC: {train_results.get('auc', 'N/A'):.4f}")
    else:
        print("\n❌ Original classifier failed!")
        return False
    
    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if improved_success and original_success:
        print("\nComparing test performance:")
        print("-" * 30)
        
        # Compare SVM results
        if ('svm' in improved_classifier.results and 
            'svm' in original_classifier.results):
            
            improved_svm = improved_classifier.results['svm']['test']
            original_svm = original_classifier.results['svm']['test']
            
            print(f"\nSVM Comparison:")
            print(f"  Original Test Accuracy: {original_svm.get('accuracy', 0):.4f}")
            print(f"  Improved Test Accuracy:  {improved_svm.get('accuracy', 0):.4f}")
            print(f"  Original Test AUC: {original_svm.get('auc', 0):.4f}")
            print(f"  Improved Test AUC:  {improved_svm.get('auc', 0):.4f}")
            
            # Calculate improvement
            acc_improvement = improved_svm.get('accuracy', 0) - original_svm.get('accuracy', 0)
            auc_improvement = improved_svm.get('auc', 0) - original_svm.get('auc', 0)
            
            print(f"  Accuracy Improvement: {acc_improvement:+.4f}")
            print(f"  AUC Improvement: {auc_improvement:+.4f}")
        
        # Compare ensemble results
        if ('ensemble' in improved_classifier.results and 
            'ensemble' in original_classifier.results):
            
            improved_ensemble = improved_classifier.results['ensemble']['test']
            original_ensemble = original_classifier.results['ensemble']['test']
            
            print(f"\nEnsemble Comparison:")
            print(f"  Original Test Accuracy: {original_ensemble.get('accuracy', 0):.4f}")
            print(f"  Improved Test Accuracy:  {improved_ensemble.get('accuracy', 0):.4f}")
            print(f"  Original Test AUC: {original_ensemble.get('auc', 0):.4f}")
            print(f"  Improved Test AUC:  {improved_ensemble.get('auc', 0):.4f}")
            
            # Calculate improvement
            acc_improvement = improved_ensemble.get('accuracy', 0) - original_ensemble.get('accuracy', 0)
            auc_improvement = improved_ensemble.get('auc', 0) - original_ensemble.get('auc', 0)
            
            print(f"  Accuracy Improvement: {acc_improvement:+.4f}")
            print(f"  AUC Improvement: {auc_improvement:+.4f}")
    
    print(f"\nResults saved to:")
    print(f"  Improved: {improved_output}")
    print(f"  Original: {original_output}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test improved optimized classifier')
    parser.add_argument('--input', required=True, help='Path to radiomics CSV file')
    parser.add_argument('--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run the test
    success = test_improved_classifier(args.input, args.output)
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 