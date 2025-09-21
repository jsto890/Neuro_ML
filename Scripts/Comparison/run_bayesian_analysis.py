#!/usr/bin/env python3
"""
Convenience script to run Bayesian analysis on P4P model outputs.

This script integrates with your existing model comparison workflow
and provides an easy way to run Bayesian analysis on your sMRI/PET/SPECT models.
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from bayesian_model_comparison import BayesianModelComparison


def find_model_outputs(base_dir: str) -> list:
    """
    Find model output directories in the typical P4P structure.
    
    Looks for directories containing model outputs like:
    - checkpoints_multi_mri/
    - checkpoints_multi_pet/
    - checkpoints_multi_spect/
    """
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: Base directory {base_dir} does not exist")
        return []
    
    # Look for common checkpoint directories
    checkpoint_patterns = [
        "checkpoints_multi_mri",
        "checkpoints_multi_pet", 
        "checkpoints_multi_spect",
        "checkpoints_multi_dspect",
        "checkpoints_*"
    ]
    
    found_dirs = []
    
    for pattern in checkpoint_patterns:
        matches = list(base_path.glob(pattern))
        found_dirs.extend(matches)
    
    # Also look for run directories directly
    run_dirs = list(base_path.glob("run_*"))
    found_dirs.extend(run_dirs)
    
    # Filter to only directories that contain model outputs
    valid_dirs = []
    for dir_path in found_dirs:
        if dir_path.is_dir():
            # Check if it contains model subdirectories
            model_dirs = [d for d in dir_path.iterdir() if d.is_dir()]
            eval_dirs = []
            for model_dir in model_dirs:
                eval_dirs.extend(list(model_dir.glob("test_evaluation_plots_fold_*")))
            
            if eval_dirs:
                valid_dirs.append(str(dir_path))
                print(f"Found model outputs in: {dir_path}")
    
    return valid_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian analysis on P4P model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all models in a directory
  python run_bayesian_analysis.py --base-dir ~/reseng202500013-ndd-ml/data
  
  # Analyze specific run
  python run_bayesian_analysis.py --run-dirs ~/reseng202500013-ndd-ml/data/checkpoints_multi_mri/run_20250918_143555
  
  # Analyze specific models only
  python run_bayesian_analysis.py --base-dir ~/reseng202500013-ndd-ml/data --models Simple3DCNN ResNet3D
  
  # Custom output directory
  python run_bayesian_analysis.py --base-dir ~/reseng202500013-ndd-ml/data --output-dir ~/my_bayesian_results
        """
    )
    
    parser.add_argument(
        "--base-dir", 
        help="Base directory containing model checkpoints (e.g., ~/reseng202500013-ndd-ml/data)"
    )
    parser.add_argument(
        "--run-dirs", nargs="*",
        help="Specific run directories to analyze"
    )
    parser.add_argument(
        "--models", nargs="*",
        help="Specific models to include (e.g., Simple3DCNN ResNet3D)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for results (default: ~/P4P_results/bayesian_analysis/<timestamp>)"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--auto-find", action="store_true", default=True,
        help="Automatically find model output directories (default: True)"
    )
    
    args = parser.parse_args()
    
    # Determine run directories
    run_dirs = []
    
    if args.run_dirs:
        run_dirs.extend(args.run_dirs)
    
    if args.base_dir and args.auto_find:
        found_dirs = find_model_outputs(args.base_dir)
        run_dirs.extend(found_dirs)
    
    # Remove duplicates and check existence
    run_dirs = list(set(run_dirs))
    run_dirs = [d for d in run_dirs if os.path.exists(d)]
    
    if not run_dirs:
        print("❌ No valid model output directories found!")
        print("\nPlease specify:")
        print("  --base-dir /path/to/your/data/directory")
        print("  OR")
        print("  --run-dirs /path/to/specific/run/directory")
        print("\nExample:")
        print("  python run_bayesian_analysis.py --base-dir ~/reseng202500013-ndd-ml/data")
        return 1
    
    print(f"Found {len(run_dirs)} run directories:")
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  {i}. {run_dir}")
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.expanduser(f"~/P4P_results/bayesian_analysis/{timestamp}")
    
    print(f"\nOutput directory: {output_dir}")
    
    # Run Bayesian analysis
    try:
        print("\n🚀 Starting Bayesian model comparison...")
        comparator = BayesianModelComparison(output_dir, args.random_seed)
        
        results = comparator.run_complete_analysis(
            run_dirs=run_dirs,
            models=args.models
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Results saved to: {output_dir}")
        print(f"📈 Plots saved to: {output_dir}/plots")
        print(f"📋 Data saved to: {output_dir}/data")
        print(f"📄 Results saved to: {output_dir}/results")
        
        # Print summary
        if results.accuracy_results:
            print(f"\n📊 Model Performance Summary:")
            models = results.accuracy_results['models']
            means = results.accuracy_results['accuracy_means']
            
            for model, acc in zip(models, means):
                print(f"  {model}: {acc:.4f}")
        
        if results.stacking_results and 'ensemble_accuracy' in results.stacking_results:
            ensemble_acc = results.stacking_results['ensemble_accuracy']
            print(f"\n🤝 Ensemble Accuracy: {ensemble_acc:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that model output directories contain the expected structure")
        print("2. Ensure all required packages are installed:")
        print("   pip install -r requirements_bayesian.txt")
        print("3. Run the setup test:")
        print("   python test_bayesian_setup.py")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
