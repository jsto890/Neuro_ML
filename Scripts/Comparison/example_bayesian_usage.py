#!/usr/bin/env python3
"""
Example usage of Bayesian model comparison for P4P models.

This script demonstrates how to run comprehensive Bayesian analysis
on your deep learning model outputs.
"""

import os
import sys
from pathlib import Path

# Add the Scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))

from bayesian_model_comparison import BayesianModelComparison


def example_analysis():
    """
    Example of running Bayesian analysis on your model outputs.
    
    This assumes you have model outputs in the structure:
    run_directory/
    ├── ModelName1/
    │   ├── test_evaluation_plots_fold_1/
    │   │   ├── predictions.npy
    │   │   ├── probabilities.npy
    │   │   ├── labels.npy
    │   │   └── evaluation_metrics.json
    │   ├── test_evaluation_plots_fold_2/
    │   └── ...
    └── ModelName2/
        └── ...
    """
    
    # Example paths - modify these to match your actual data
    run_dirs = [
        # Add your run directories here
        "/path/to/your/checkpoints_multi_mri",
        "/path/to/your/checkpoints_multi_pet", 
        "/path/to/your/checkpoints_multi_spect",
    ]
    
    # Optional: specify which models to include
    models_to_analyze = [
        "Simple3DCNN",
        "ResNet3D", 
        "DenseNet3D",
        # Add other model names as needed
    ]
    
    # Output directory
    output_dir = os.path.expanduser("~/P4P_results/bayesian_analysis_example")
    
    # Create comparator and run analysis
    print("Starting Bayesian model comparison...")
    comparator = BayesianModelComparison(output_dir, random_seed=42)
    
    results = comparator.run_complete_analysis(
        run_dirs=run_dirs,
        models=models_to_analyze
    )
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    
    # Access specific results
    if results.accuracy_results:
        print("\n=== Accuracy Results ===")
        models = results.accuracy_results['models']
        means = results.accuracy_results['accuracy_means']
        
        print("Model Accuracy Estimates:")
        for model, acc in zip(models, means):
            print(f"  {model}: {acc:.4f}")
        
        # Model comparison probabilities
        if 'model_comparisons' in results.accuracy_results:
            print("\nModel Comparison Probabilities:")
            comp_df = results.accuracy_results['model_comparisons']
            for _, row in comp_df.iterrows():
                print(f"  P({row['model_a']} > {row['model_b']}) = {row['prob_a_better']:.4f}")
    
    if results.stacking_results:
        print("\n=== Ensemble Results ===")
        ensemble_acc = results.stacking_results.get('ensemble_accuracy', 0)
        print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
        
        individual_accs = results.stacking_results.get('individual_accuracies', {})
        print("Individual Model Accuracies:")
        for model, acc in individual_accs.items():
            print(f"  {model}: {acc:.4f}")


def analyze_specific_run():
    """
    Analyze a specific run directory with known structure.
    """
    
    # Example: analyze your specific run
    run_dir = "/Users/josephstorey/reseng202500013-ndd-ml/data/checkpoints_multi_mri/run_20250918_143555"
    
    if not os.path.exists(run_dir):
        print(f"Run directory {run_dir} does not exist")
        print("Please update the path to match your actual data location")
        return
    
    output_dir = os.path.expanduser("~/P4P_results/bayesian_analysis_specific")
    
    print(f"Analyzing run: {run_dir}")
    print(f"Output directory: {output_dir}")
    
    comparator = BayesianModelComparison(output_dir, random_seed=42)
    
    results = comparator.run_complete_analysis(
        run_dirs=[run_dir],
        models=None  # Include all models found
    )
    
    print(f"\nAnalysis complete! Check results in: {output_dir}")


if __name__ == "__main__":
    # Run the example
    print("Bayesian Model Comparison Example")
    print("=" * 50)
    
    # Uncomment the function you want to run:
    
    # example_analysis()  # Generic example
    analyze_specific_run()  # Analyze your specific run
