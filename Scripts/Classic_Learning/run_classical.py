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

from radiomics_classifier import RadiomicsClassifier

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
    
    args = parser.parse_args()
    
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
    print("=" * 60)
    
    # Initialize and run pipeline
    classifier = RadiomicsClassifier(input_path, output_dir, args.random_state)
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