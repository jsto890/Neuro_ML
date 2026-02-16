#!/usr/bin/env python3
"""
FDR Feature Selection Comparison Runner
======================================

This script runs the enhanced FDR classifier to compare three feature selection approaches:
1. FDR-based feature selection (False Discovery Rate correction)
2. Current selection (MutualInfo + RFECV)
3. No feature selection (all features)

Usage:
    python run_fdr_comparison.py --input radiomics_features.csv --output results/
"""

import sys
import os
from pathlib import Path

# Add the Optimised directory to the path
sys.path.append(str(Path(__file__).parent / "Optimised"))

try:
    from enhanced_fdr_classifier import EnhancedFDRRadiomicsClassifier
except ImportError as e:
    print(f"Error importing EnhancedFDRRadiomicsClassifier: {e}")
    print("Please ensure you're running this from the Classic_Learning directory")
    sys.exit(1)

def main():
    """Main function to run the FDR comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FDR Feature Selection Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_fdr_comparison.py --input radiomics_features.csv --output results/
  
  # Run with custom FDR alpha
  python run_fdr_comparison.py --input radiomics_features.csv --output results/ --fdr-alpha 0.01
  
  # Run with custom random seed
  python run_fdr_comparison.py --input radiomics_features.csv --output results/ --random-state 123
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to radiomics CSV file (must contain subject_id, label columns)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--fdr-alpha', '-a',
        type=float,
        default=0.05,
        help='FDR significance level (default: 0.05)'
    )
    
    parser.add_argument(
        '--random-state', '-r',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--binary-only', '-b',
        action='store_true',
        help='Use only binary classification (labels 0 and 1) (default: True)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f" Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(" Starting FDR Feature Selection Comparison")
    print("=" * 50)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"FDR alpha: {args.fdr_alpha}")
    print(f"Random state: {args.random_state}")
    print(f"Binary only: {args.binary_only}")
    print("=" * 50)
    
    try:
        # Create classifier and run pipeline
        classifier = EnhancedFDRRadiomicsClassifier(
            input_path=args.input,
            output_dir=args.output,
            random_state=args.random_state,
            binary_only=True,  # Always use binary classification for now
            fdr_alpha=args.fdr_alpha
        )
        
        # Set verbose logging if requested
        if args.verbose:
            classifier.logger.setLevel('DEBUG')
        
        # Run the complete pipeline
        success = classifier.run_complete_pipeline()
        
        if success:
            print("\n FDR Comparison completed successfully!")
            print("=" * 50)
            
            # Print summary of results
            if hasattr(classifier, 'comparison_results') and classifier.comparison_results:
                print("\n COMPARISON SUMMARY:")
                print("-" * 30)
                
                for approach, results in classifier.comparison_results.items():
                    n_features = results['n_features']
                    svm_acc = results['svm_test'].get('accuracy', 0) if results['svm_test'] else 0
                    svm_mcc = results['svm_test'].get('mcc', 0) if results['svm_test'] else 0
                    ensemble_acc = results['ensemble_test'].get('accuracy', 0) if results['ensemble_test'] else 0
                    ensemble_mcc = results['ensemble_test'].get('mcc', 0) if results['ensemble_test'] else 0
                    
                    print(f"{approach.replace('_', ' ').title()}:")
                    print(f"  Features: {n_features}")
                    print(f"  SVM - Accuracy: {svm_acc:.3f}, MCC: {svm_mcc:.3f}")
                    print(f"  Ensemble - Accuracy: {ensemble_acc:.3f}, MCC: {ensemble_mcc:.3f}")
                    print()
            
            print(" Generated files:")
            print("  • comparison_report.txt - Detailed comparison report")
            print("  • comparison_results.json - Comparison results")
            print("  • detailed_results.json - Detailed results for each approach")
            print("  • feature_engineering_results.json - Feature engineering details")
            print("  • [approach]_[model]_model.pkl - Trained models")
            print("  • scaler.pkl - Feature scaler")
            print("  • enhanced_fdr_pipeline.log - Execution log")
            
            print("\n Key Findings:")
            print("  • FDR selection uses statistical correction for multiple testing")
            print("  • Current selection uses MutualInfo + RFECV")
            print("  • No selection uses all features after preprocessing")
            print("  • MCC (Matthews Correlation Coefficient) is the key metric for comparison")
            
        else:
            print("\n FDR Comparison failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n Error running FDR comparison: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 