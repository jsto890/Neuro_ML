#!/usr/bin/env python3
"""
Optimized Radiomics Classification Pipeline Runner
=================================================

This script runs the optimized radiomics classification pipeline with:
- SVM as primary model with fine-tuned hyperparameters
- Advanced feature engineering based on cross-model importance
- RFECV feature selection optimized for SVM
- Ensemble methods with SVM as base
- Clinical interpretability and robustness focus
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))


def main():
    """Run the optimized radiomics classification pipeline."""
    
    parser = argparse.ArgumentParser(description='Optimized Radiomics Classification Pipeline')
    parser.add_argument('--input', default='~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv',
                        help='Path to radiomics CSV file')
    parser.add_argument('--output-dir', default='~/reseng202500013-ndd-ml/data/optimized_classical_results',
                        help='Output directory for results')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--binary-only', action='store_true', default=True,
                        help='Use only binary classification (labels 0 and 1)')
    parser.add_argument('--multi-class', action='store_true', default=False,
                        help='Use multi-class classification (all labels)')
    parser.add_argument('--outer-k-folds', type=int, default=0,
                        help='If >1, run outer Stratified K-Fold with this many folds (e.g., 5)')
    parser.add_argument('--val-ratio', type=float, default=0.0,
                        help='Validation ratio within the training pool per outer fold (0.0 to disable)')

    args = parser.parse_args()

    # Expand user paths
    input_path = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    random_state = args.random_state
    binary_only = not args.multi_class
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        print("Please run the radiomics extraction first:")
        print("cd Scripts/Feature_Extraction/pyRadioMics/")
        print("python3 simple_radiomics.py --labels ~/reseng202500013-ndd-ml/data/mri_labels.csv --output-dir ~/reseng202500013-ndd-ml/data/ --config ~/reseng202500013-ndd-ml/P4P/config.yaml")
        sys.exit(1)
    
    print("Starting Optimized Radiomics Classification Pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Random seed: {random_state}")
    print(f"Classification: {'Binary (0,1)' if binary_only else 'Multi-class'}")
    print("=" * 60)
    
    # Initialize and run pipeline
    # Note: the exported class in optimized_classifier.py is ImprovedOptimizedRadiomicsClassifier
    from optimized_classifier import ImprovedOptimizedRadiomicsClassifier as OptimizedRadiomicsClassifierActual
    classifier = OptimizedRadiomicsClassifierActual(input_path, output_dir, random_state, binary_only)

    if args.outer_k_folds and args.outer_k_folds > 1:
        print(f"Running Outer Stratified K-Fold: {args.outer_k_folds} folds | Val ratio: {args.val_ratio}")
        success = classifier.run_outer_cv(k_folds=args.outer_k_folds, val_ratio=args.val_ratio)
    else:
        success = classifier.run_improved_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("Optimized pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  • optimized_svm_model.pkl - Fine-tuned SVM model")
        print(f"  • optimized_ensemble_model.pkl - Ensemble model")
        print(f"  • optimized_scaler.pkl - Feature scaler")
        print(f"  • optimized_feature_importance.csv - Feature importance")
        print(f"  • feature_engineering_results.json - Engineering details")
        print(f"  • optimized_evaluation_plots.png - Performance plots")
        print(f"  • optimized_results_summary.json - Detailed results")
        print(f"  • optimized_pipeline.log - Execution log")
        print("\nKey Optimizations:")
        print(f"  • Bayesian optimization for hyperparameter tuning")
        print(f"  • Advanced models: XGBoost, LightGBM")
        print(f"  • Advanced feature engineering with polynomial features")
        print(f"  • Stacking ensemble with diverse base models")
        print(f"  • Clinical interpretability focus")
        print("\nClinical Recommendations:")
        print(f"  • Use optimized_svm_model.pkl for primary predictions")
        print(f"  • Review optimized_feature_importance.csv for key biomarkers")
        print(f"  • Check feature_engineering_results.json for feature insights")
        print(f"  • Ensemble model provides robust backup predictions")
    else:
        print("\nPipeline failed! Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 