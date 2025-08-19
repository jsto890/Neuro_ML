#!/usr/bin/env python3
"""
Enhanced Radiomics Classification Pipeline Runner
===============================================

This script runs the enhanced radiomics classification pipeline with:
- Multiple algorithms (Random Forest, SVM, Logistic Regression, Gradient Boosting)
- Advanced feature engineering and selection
- Better regularization to prevent overfitting
- Ensemble methods
- Comprehensive model comparison
"""

import os
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from Scripts.Classic_Learning.Enhanced.enhanced_classifier import EnhancedRadiomicsClassifier

def main():
    """Run the enhanced radiomics classification pipeline."""
    
    parser = argparse.ArgumentParser(description='Enhanced Radiomics Classification Pipeline')
    parser.add_argument('--input', default='~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv',
                        help='Path to radiomics CSV file')
    parser.add_argument('--output-dir', default='~/reseng202500013-ndd-ml/data/enhanced_classical_results',
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

    # Binary vs multi-class handling
    binary_only = not args.multi_class

    # Expand user paths
    input_path = os.path.expanduser(args.input)
    base_output_dir = Path(os.path.expanduser(args.output_dir))
    # Create timestamped run directory
    run_dir = base_output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    random_state = args.random_state
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        print("Please run the radiomics extraction first:")
        print("cd Scripts/Feature_Extraction/pyRadioMics/")
        print("python3 simple_radiomics.py --labels ~/reseng202500013-ndd-ml/data/mri_labels.csv --output-dir ~/reseng202500013-ndd-ml/data/ --config ~/reseng202500013-ndd-ml/P4P/config.yaml")
        sys.exit(1)
    
    print("Starting Enhanced Radiomics Classification Pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {str(run_dir)}")
    print(f"Random seed: {random_state}")
    print(f"Classification: {'Binary (0,1)' if binary_only else 'Multi-class'}")
    print("=" * 60)
    
    # Initialize and run pipeline
    classifier = EnhancedRadiomicsClassifier(input_path, str(run_dir), random_state, binary_only)

    if args.outer_k_folds and args.outer_k_folds > 1:
        print(f"Running Outer Stratified K-Fold: {args.outer_k_folds} folds | Val ratio: {args.val_ratio}")
        success = classifier.run_outer_cv(k_folds=args.outer_k_folds, val_ratio=args.val_ratio)
    else:
        success = classifier.run_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("Enhanced pipeline completed successfully!")
        print(f"Results saved to: {str(run_dir)}")
        print("\nGenerated files:")
        print(f"  • randomforest_model.pkl - Random Forest model")
        print(f"  • logisticregression_model.pkl - Logistic Regression model")
        print(f"  • svm_model.pkl - SVM model")
        print(f"  • gradientboosting_model.pkl - Gradient Boosting model")
        print(f"  • scaler.pkl - Feature scaler")
        print(f"  • feature_importance_comparison.csv - Feature importance across models")
        print(f"  • enhanced_evaluation_plots.png - Comprehensive performance plots")
        print(f"  • enhanced_results_summary.json - Detailed results")
        print(f"  • enhanced_pipeline.log - Execution log")
        print("\nKey Improvements:")
        print(f"  • Multiple algorithms compared")
        print(f"  • Advanced feature selection")
        print(f"  • Regularization to prevent overfitting")
        print(f"  • Ensemble model for better performance")
        print(f"  • Comprehensive evaluation metrics")
    else:
        print("\nPipeline failed! Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 