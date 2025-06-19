#!/usr/bin/env python3
"""
Enhanced Radiomics Classification Pipeline Runner
================================================

This script runs the enhanced radiomics classification pipeline with:
- Multiple algorithms (Random Forest, SVM, Logistic Regression, Gradient Boosting)
- Advanced feature engineering and selection
- Better regularization to prevent overfitting
- Ensemble methods
- Comprehensive model comparison
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from Scripts.Classic_Learning.Enhanced.enhanced_classifier import EnhancedRadiomicsClassifier

def main():
    """Run the enhanced radiomics classification pipeline."""
    
    # Default paths
    input_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv")
    output_dir = os.path.expanduser("~/reseng202500013-ndd-ml/data/enhanced_classical_results")
    random_state = 42
    binary_only = True
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        print("Please run the radiomics extraction first:")
        print("cd Scripts/Feature_Extraction/pyRadioMics/")
        print("python3 simple_radiomics.py --labels ~/reseng202500013-ndd-ml/data/mri_labels.csv --output-dir ~/reseng202500013-ndd-ml/data/ --config ~/reseng202500013-ndd-ml/P4P/config.yaml")
        sys.exit(1)
    
    print("Starting Enhanced Radiomics Classification Pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Random seed: {random_state}")
    print(f"Classification: {'Binary (0,1)' if binary_only else 'Multi-class'}")
    print("=" * 60)
    
    # Initialize and run pipeline
    classifier = EnhancedRadiomicsClassifier(input_path, output_dir, random_state, binary_only)
    success = classifier.run_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("Enhanced pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
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