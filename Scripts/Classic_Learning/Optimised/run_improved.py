#!/usr/bin/env python3
"""
Improved Optimized Radiomics Classification Pipeline Runner
==========================================================

This script runs an improved version of the optimized pipeline that addresses:
1. Overfitting issues in ensemble models
2. SVM convergence problems
3. Data leakage in feature engineering
4. Aggressive outlier removal
5. Complex polynomial features

Key Improvements:
- Conservative outlier detection (3x IQR)
- Simplified polynomial features (degree 2 only)
- Mutual information feature selection
- Regularized ensemble models
- Improved SVM parameter ranges
- Data leakage prevention
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function to run the improved optimized pipeline."""
    parser = argparse.ArgumentParser(description='Improved Optimized Radiomics Classification Pipeline')
    parser.add_argument('--config', type=str, default='config_improved.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, 
                       help='Override input file path from config')
    parser.add_argument('--output', type=str, 
                       help='Override output directory from config')
    parser.add_argument('--random_state', type=int, 
                       help='Override random state from config')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Override config with command line arguments
    if args.input:
        config['data']['input_file'] = args.input
    if args.output:
        config['data']['output_dir'] = args.output
    if args.random_state:
        config['data']['random_state'] = args.random_state
    
    # Expand user paths
    config['data']['input_file'] = os.path.expanduser(config['data']['input_file'])
    config['data']['output_dir'] = os.path.expanduser(config['data']['output_dir'])
    
    # Validate input file
    if not os.path.exists(config['data']['input_file']):
        print(f"Input file not found: {config['data']['input_file']}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Improved Optimized Radiomics Classification Pipeline")
    print("="*60)
    print(f"Input: {config['data']['input_file']}")
    print(f"Output: {config['data']['output_dir']}")
    print(f"Random seed: {config['data']['random_state']}")
    print(f"Classification: {'Binary' if config['data']['binary_only'] else 'Multi-class'}")
    print("="*60)
    
    # Import the improved classifier
    try:
        from improved_optimized_classifier import ImprovedOptimizedRadiomicsClassifier
    except ImportError:
        print("Error: improved_optimized_classifier.py not found")
        print("Please ensure the improved classifier file is in the same directory")
        sys.exit(1)
    
    # Create classifier and run pipeline
    classifier = ImprovedOptimizedRadiomicsClassifier(
        input_path=config['data']['input_file'],
        output_dir=config['data']['output_dir'],
        random_state=config['data']['random_state'],
        binary_only=config['data']['binary_only']
    )
    
    # Run the improved pipeline
    success = classifier.run_improved_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("Improved pipeline completed successfully!")
        print(f"Results saved to: {config['data']['output_dir']}")
        print("="*60)
        
        print("\nGenerated files:")
        print("  • improved_svm_model.pkl - Optimized SVM model")
        print("  • improved_ensemble_model.pkl - Improved ensemble model")
        print("  • improved_scaler.pkl - Feature scaler")
        print("  • improved_feature_importance.csv - Feature importance")
        print("  • improved_feature_engineering_results.json - Engineering details")
        print("  • improved_results_summary.json - Detailed results")
        print("  • improved_optimized_pipeline.log - Execution log")
        
        print("\nKey Improvements Applied:")
        print("  • Conservative outlier detection (3x IQR)")
        print("  • Simplified polynomial features (degree 2 only)")
        print("  • Mutual information feature selection")
        print("  • Regularized ensemble models")
        print("  • Improved SVM parameter ranges")
        print("  • Data leakage prevention")
        print("  • Reduced overfitting through regularization")
        
        print("\nClinical Recommendations:")
        print("  • Use improved_svm_model.pkl for primary predictions")
        print("  • Review improved_feature_importance.csv for key biomarkers")
        print("  • Check improved_feature_engineering_results.json for insights")
        print("  • Ensemble model provides robust backup predictions")
        print("  • Monitor train vs test performance for overfitting")
        
        print("\nPerformance Expectations:")
        print("  • Reduced overfitting compared to original pipeline")
        print("  • Better generalization to unseen data")
        print("  • More stable SVM convergence")
        print("  • Improved clinical interpretability")
        
    else:
        print("Pipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 