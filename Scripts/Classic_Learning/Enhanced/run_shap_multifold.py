"""
Multi-Fold SHAP Analysis Script
================================

This script runs SHAP analysis across multiple CV folds and compares
feature importance stability across folds.

Usage:
    python run_shap_multifold.py \
        --cv_dir /path/to/enhanced_run_SPECT/run_20251010_171321 \
        --data /path/to/radiomics_spect.csv \
        --output /path/to/shap_multifold_results \
        --model_type randomforest
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from shap_interpretability import SHAPInterpreter, SHAP_AVAILABLE
from run_shap_analysis import load_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_fold_directories(cv_dir: Path) -> List[Path]:
    """
    Find all outer CV fold directories.
    
    Args:
        cv_dir: Directory containing outercv_fold_* subdirectories
    
    Returns:
        List of fold directory paths
    """
    fold_dirs = sorted(cv_dir.glob("outercv_fold_*"))
    logger.info(f"Found {len(fold_dirs)} CV fold directories")
    return fold_dirs


def run_shap_for_fold(fold_dir: Path, model_type: str, data: Dict, 
                      output_dir: Path, fold_num: int,
                      class_names: Optional[List[str]] = None) -> Optional[Dict]:
    """
    Run SHAP analysis for a single fold.
    
    Args:
        fold_dir: Directory containing fold models
        model_type: Type of model to analyze (e.g., 'randomforest', 'svm')
        data: Dictionary with train/test data
        output_dir: Output directory
        fold_num: Fold number
        class_names: Class names (optional)
    
    Returns:
        Dictionary with SHAP results or None if failed
    """
    # Find model file
    model_files = list(fold_dir.glob(f"{model_type}_model.pkl"))
    
    if not model_files:
        logger.error(f"No {model_type}_model.pkl found in {fold_dir}")
        return None
    
    model_path = model_files[0]
    logger.info(f"Analyzing Fold {fold_num}: {model_path.name}")
    
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Create fold-specific output directory
        fold_output_dir = output_dir / f"fold_{fold_num}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create interpreter
        interpreter = SHAPInterpreter(
            model=model,
            X_train=data['X_train'],
            feature_names=data['feature_names'],
            output_dir=fold_output_dir,
            model_name=f"{model_type}_fold{fold_num}",
            class_names=class_names
        )
        
        # Compute SHAP values
        shap_values = interpreter.compute_shap_values(data['X_test'])
        
        # Create basic plots
        interpreter.plot_summary(data['X_test'], max_display=20)
        interpreter.plot_bar(data['X_test'], max_display=20)
        
        # Export SHAP values
        shap_df = interpreter.export_shap_values(data['X_test'], data['y_test'])
        
        # Get mean absolute SHAP values for feature importance
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shap_vals = shap_values
        
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        logger.info(f"✓ Fold {fold_num} completed successfully")
        
        return {
            'fold_num': fold_num,
            'model_path': str(model_path),
            'shap_values': shap_vals,
            'mean_abs_shap': mean_abs_shap,
            'feature_names': data['feature_names']
        }
    
    except Exception as e:
        logger.error(f"✗ Error analyzing Fold {fold_num}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_folds(fold_results: List[Dict], output_dir: Path, model_type: str):
    """
    Compare SHAP feature importance across folds.
    
    Args:
        fold_results: List of fold result dictionaries
        output_dir: Output directory
        model_type: Model type name
    """
    logger.info("=" * 80)
    logger.info("Comparing Feature Importance Across Folds")
    logger.info("=" * 80)
    
    # Extract feature names and mean SHAP values
    feature_names = fold_results[0]['feature_names']
    n_features = len(feature_names)
    n_folds = len(fold_results)
    
    # Create matrix of mean absolute SHAP values (features x folds)
    shap_matrix = np.zeros((n_features, n_folds))
    
    for i, result in enumerate(fold_results):
        shap_matrix[:, i] = result['mean_abs_shap']
    
    # Calculate statistics across folds
    mean_shap = shap_matrix.mean(axis=1)
    std_shap = shap_matrix.std(axis=1)
    cv_shap = std_shap / (mean_shap + 1e-10)  # Coefficient of variation
    
    # Create DataFrame with results
    comparison_df = pd.DataFrame({
        'feature': feature_names,
        'mean_shap': mean_shap,
        'std_shap': std_shap,
        'cv_shap': cv_shap
    })
    
    # Add individual fold values
    for i, result in enumerate(fold_results):
        comparison_df[f'fold_{result["fold_num"]}_shap'] = result['mean_abs_shap']
    
    # Sort by mean SHAP value
    comparison_df = comparison_df.sort_values('mean_shap', ascending=False)
    
    # Save to CSV
    csv_path = output_dir / f"{model_type}_feature_importance_across_folds.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison to {csv_path}")
    
    # Plot 1: Top features with error bars
    plt.figure(figsize=(12, 8))
    top_n = 20
    top_features = comparison_df.head(top_n)
    
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features['mean_shap'], xerr=top_features['std_shap'], 
             alpha=0.7, capsize=5)
    plt.yticks(y_pos, top_features['feature'])
    plt.xlabel('Mean Absolute SHAP Value')
    plt.title(f'Top {top_n} Features: {model_type.title()} (Mean ± Std across {n_folds} folds)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_top_features_with_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created top features plot")
    
    # Plot 2: Heatmap of feature importance across folds
    plt.figure(figsize=(14, 10))
    top_n_heatmap = 30
    top_features_heatmap = comparison_df.head(top_n_heatmap)
    
    # Get fold columns
    fold_cols = [col for col in comparison_df.columns if col.startswith('fold_')]
    heatmap_data = top_features_heatmap[fold_cols].values
    
    # Normalize each row (feature) for better visualization
    heatmap_data_norm = (heatmap_data - heatmap_data.min(axis=1, keepdims=True)) / \
                        (heatmap_data.max(axis=1, keepdims=True) - heatmap_data.min(axis=1, keepdims=True) + 1e-10)
    
    sns.heatmap(heatmap_data_norm, 
                xticklabels=[f"Fold {i+1}" for i in range(n_folds)],
                yticklabels=top_features_heatmap['feature'].values,
                cmap='YlOrRd', cbar_kws={'label': 'Normalized SHAP Value'})
    plt.title(f'Feature Importance Across Folds: {model_type.title()}\n(Row-normalized)')
    plt.xlabel('CV Fold')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_heatmap_across_folds.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created heatmap")
    
    # Plot 3: Coefficient of variation (stability metric)
    plt.figure(figsize=(12, 8))
    top_n_cv = 25
    top_features_cv = comparison_df.head(top_n_cv)
    
    colors = ['green' if cv < 0.3 else 'orange' if cv < 0.5 else 'red' 
              for cv in top_features_cv['cv_shap']]
    
    y_pos = np.arange(len(top_features_cv))
    plt.barh(y_pos, top_features_cv['cv_shap'], color=colors, alpha=0.7)
    plt.yticks(y_pos, top_features_cv['feature'])
    plt.xlabel('Coefficient of Variation (CV)')
    plt.title(f'Feature Stability Across Folds: {model_type.title()}\n' + 
              'Green: Stable (CV<0.3), Orange: Moderate (0.3≤CV<0.5), Red: Variable (CV≥0.5)')
    plt.gca().invert_yaxis()
    plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_feature_stability.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created stability plot")
    
    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("Feature Stability Summary:")
    logger.info("=" * 80)
    
    stable_features = comparison_df[comparison_df['cv_shap'] < 0.3]
    moderate_features = comparison_df[(comparison_df['cv_shap'] >= 0.3) & (comparison_df['cv_shap'] < 0.5)]
    variable_features = comparison_df[comparison_df['cv_shap'] >= 0.5]
    
    logger.info(f"Stable features (CV < 0.3): {len(stable_features)} ({len(stable_features)/n_features*100:.1f}%)")
    logger.info(f"Moderate features (0.3 ≤ CV < 0.5): {len(moderate_features)} ({len(moderate_features)/n_features*100:.1f}%)")
    logger.info(f"Variable features (CV ≥ 0.5): {len(variable_features)} ({len(variable_features)/n_features*100:.1f}%)")
    
    logger.info("\nTop 10 Most Stable Features:")
    stable_top = comparison_df.nsmallest(10, 'cv_shap')[['feature', 'mean_shap', 'cv_shap']]
    for i, row in stable_top.iterrows():
        logger.info(f"  {row['feature']:40s} | Mean SHAP: {row['mean_shap']:.4f} | CV: {row['cv_shap']:.4f}")
    
    logger.info("\nTop 10 Most Important Features (by mean SHAP):")
    important_top = comparison_df.head(10)[['feature', 'mean_shap', 'cv_shap']]
    for i, row in important_top.iterrows():
        logger.info(f"  {row['feature']:40s} | Mean SHAP: {row['mean_shap']:.4f} | CV: {row['cv_shap']:.4f}")
    
    # Save summary to JSON
    summary = {
        'model_type': model_type,
        'n_folds': n_folds,
        'n_features': n_features,
        'stability_summary': {
            'stable_count': int(len(stable_features)),
            'moderate_count': int(len(moderate_features)),
            'variable_count': int(len(variable_features))
        },
        'top_10_stable': stable_top.to_dict('records'),
        'top_10_important': important_top.to_dict('records')
    }
    
    json_path = output_dir / f"{model_type}_multifold_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-fold SHAP analysis for cross-validation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_shap_multifold.py \\
      --cv_dir ~/data/classic_results/enhanced_run_SPECT/run_20251010_171321 \\
      --data ~/data/radiomics_spect.csv \\
      --output ~/data/shap_multifold_results \\
      --model_type randomforest \\
      --class_names CN PD
        """
    )
    
    parser.add_argument('--cv_dir', type=str, required=True,
                       help='Directory containing outercv_fold_* subdirectories')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file with radiomics features')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--model_type', type=str, required=True,
                       help='Model type to analyze (e.g., randomforest, svm, xgboost)')
    parser.add_argument('--class_names', nargs='+', type=str,
                       help='Class names (e.g., --class_names CN AD PD)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Check SHAP availability
    if not SHAP_AVAILABLE:
        logger.error("SHAP library not found. Install with: pip install shap")
        sys.exit(1)
    
    # Setup paths
    cv_dir = Path(args.cv_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find fold directories
    fold_dirs = find_fold_directories(cv_dir)
    
    if not fold_dirs:
        logger.error(f"No outercv_fold_* directories found in {cv_dir}")
        sys.exit(1)
    
    # Load data
    logger.info("Loading data...")
    data = load_data(args.data, test_size=args.test_size, random_state=args.random_state)
    
    # Run SHAP for each fold
    logger.info("=" * 80)
    logger.info(f"Running SHAP analysis for {args.model_type} across {len(fold_dirs)} folds")
    logger.info("=" * 80)
    
    fold_results = []
    for i, fold_dir in enumerate(fold_dirs, 1):
        result = run_shap_for_fold(
            fold_dir=fold_dir,
            model_type=args.model_type,
            data=data,
            output_dir=output_dir,
            fold_num=i,
            class_names=args.class_names
        )
        
        if result is not None:
            fold_results.append(result)
    
    if not fold_results:
        logger.error("No successful SHAP analyses! Check errors above.")
        sys.exit(1)
    
    # Compare across folds
    compare_folds(fold_results, output_dir, args.model_type)
    
    # Final summary
    logger.info("=" * 80)
    logger.info(f"Multi-fold SHAP analysis completed!")
    logger.info(f"Successfully analyzed {len(fold_results)}/{len(fold_dirs)} folds")
    logger.info(f"Results saved to: {output_dir.absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

