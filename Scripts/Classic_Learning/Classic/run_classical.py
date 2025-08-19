#!/usr/bin/env python3
"""
Simple Runner for Radiomics Classical Learning Pipeline
=======================================================

This script provides a simple interface to run the complete classical learning pipeline.

Usage:
    python run_classical.py
    python run_classical.py --input path/to/radiomics.csv --output results/
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from Scripts.Classic_Learning.Classic.radiomics_classifier import RadiomicsClassifier

def main():
    parser = argparse.ArgumentParser(description='Run Radiomics Classical Learning Pipeline')
    parser.add_argument('--input', 
                       default='~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv',
                       help='Path to radiomics CSV file')
    parser.add_argument('--output-dir', 
                       default='~/reseng202500013-ndd-ml/data/classical_results',
                       help='Output directory for results')
    parser.add_argument('--config',
                       default='config_classical.yaml',
                       help='Path to configuration file')
    parser.add_argument('--random-state', 
                       type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--binary-only', 
                       action='store_true', default=True,
                       help='Use only binary classification (labels 0 and 1)')
    parser.add_argument('--multi-class', 
                       action='store_true', default=False,
                       help='Use multi-class classification (all labels)')
    parser.add_argument('--outer-k-folds', type=int, default=0,
                        help='If >1, run outer Stratified K-Fold with this many folds (e.g., 5 for ~80/20 Test per fold)')
    parser.add_argument('--val-ratio', type=float, default=0.0,
                        help='Validation ratio within the training pool per outer fold (0.0 to disable)')
    
    args = parser.parse_args()
    
    # Handle binary vs multi-class
    if args.multi_class:
        binary_only = False
    else:
        binary_only = True
    
    # Expand user paths
    input_path = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    config_path = os.path.expanduser(args.config) if args.config else None
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        print("Please run the radiomics extraction first:")
        print("cd Scripts/Feature_Extraction/pyRadioMics/")
        print("python3 simple_radiomics.py --labels ~/reseng202500013-ndd-ml/data/mri_labels.csv --output-dir ~/reseng202500013-ndd-ml/data/ --config ~/reseng202500013-ndd-ml/P4P/config.yaml")
        sys.exit(1)
    
    print("Starting Radiomics Classical Learning Pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Random seed: {args.random_state}")
    print(f"Classification: {'Binary (0,1)' if binary_only else 'Multi-class'}")
    print("=" * 60)
    
    # Initialize classifier
    classifier = RadiomicsClassifier(input_path, output_dir, args.random_state, binary_only)

    # Choose standard pipeline or outer CV
    if args.outer_k_folds and args.outer_k_folds > 1:
        print(f"Running Outer Stratified K-Fold: {args.outer_k_folds} folds | Val ratio: {args.val_ratio}")
        success = classifier.run_outer_cv(k_folds=args.outer_k_folds, val_ratio=args.val_ratio)
    else:
        success = classifier.run_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  • random_forest_model.pkl - Trained model")
        print(f"  • scaler.pkl - Feature scaler")
        print(f"  • feature_importance.csv - Feature importance rankings")
        print(f"  • evaluation_plots.png - Performance plots")
        print(f"  • results_summary.json - Detailed results")
        print(f"  • pipeline.log - Execution log")
        print("\nNext steps:")
        print("  • Check evaluation_plots.png for performance visualization")
        print("  • Review feature_importance.csv for top features")
        print("  • Use random_forest_model.pkl for predictions on new data")
    else:
        print("\nPipeline failed! Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 