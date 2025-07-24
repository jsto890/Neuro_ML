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
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from optimized_classifier import OptimizedRadiomicsClassifier

def main():
    """Run the optimized radiomics classification pipeline."""
    
    # Default paths
    input_path = os.path.expanduser("~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv")
    output_dir = os.path.expanduser("~/reseng202500013-ndd-ml/data/optimized_classical_results")
    random_state = 42
    binary_only = True
    
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
    classifier = OptimizedRadiomicsClassifier(input_path, output_dir, random_state, binary_only)
    success = classifier.run_optimized_pipeline()
    
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