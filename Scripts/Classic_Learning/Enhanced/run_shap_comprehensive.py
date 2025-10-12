"""
Comprehensive Multi-Fold, Multi-Model SHAP Analysis
====================================================

This script performs exhaustive SHAP analysis across:
- All CV folds
- All model types
- All classes (multi-class support)

It generates:
1. Per-model cross-fold averaged SHAP values
2. Cross-model comparison of feature importance
3. Ensemble-based feature importance
4. Comprehensive visualizations and rankings

Usage:
    # Analyze all models (default)
    python run_shap_comprehensive.py \
        --cv_dir /path/to/enhanced_run_SPECT/run_20251010_171321 \
        --data /path/to/radiomics_spect.csv \
        --output /path/to/shap_comprehensive_results \
        --class_names CN PD
    
    # Or specify specific models
    python run_shap_comprehensive.py \
        --cv_dir /path/to/enhanced_run_SPECT/run_20251010_171321 \
        --data /path/to/radiomics_spect.csv \
        --output /path/to/shap_comprehensive_results \
        --class_names CN PD \
        --model_types randomforest xgboost
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from shap_interpretability import SHAPInterpreter, SHAP_AVAILABLE
from run_shap_analysis import load_data, load_selected_features, filter_data_to_selected_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_fold_directories(cv_dir: Path) -> List[Path]:
    """Find all outer CV fold directories."""
    fold_dirs = sorted(cv_dir.glob("outercv_fold_*"))
    logger.info(f"Found {len(fold_dirs)} CV fold directories")
    return fold_dirs


def get_shap_for_model_fold(fold_dir: Path, model_type: str, data: Dict,
                            class_names: List[str]) -> Optional[Dict]:
    """
    Compute SHAP values for a single model in a single fold.
    
    Returns:
        Dictionary with SHAP values and metadata, or None if failed
    """
    # Find model file
    model_files = list(fold_dir.glob(f"{model_type}_model.pkl"))
    
    if not model_files:
        logger.warning(f"No {model_type}_model.pkl found in {fold_dir}")
        return None
    
    model_path = model_files[0]
    fold_num = int(fold_dir.name.split('_')[-1])
    
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Determine expected feature count
        expected_features = None
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
        elif hasattr(model, 'coef_'):
            expected_features = model.coef_.shape[1] if model.coef_.ndim > 1 else model.coef_.shape[0]
        
        # Prepare data
        current_data = data.copy()
        data_features = current_data['X_train'].shape[1]
        
        # Handle feature mismatch
        if expected_features is not None and data_features != expected_features:
            selected_features = load_selected_features(fold_dir)
            
            if selected_features and len(selected_features) == expected_features:
                current_data = filter_data_to_selected_features(current_data, selected_features)
            else:
                logger.error(f"Fold {fold_num}, {model_type}: Could not resolve feature mismatch")
                return None
        
        # Create interpreter
        interpreter = SHAPInterpreter(
            model=model,
            X_train=current_data['X_train'],
            feature_names=current_data['feature_names'],
            output_dir=Path('/tmp'),  # Temporary, not saving individual plots
            model_name=f"{model_type}_fold{fold_num}",
            class_names=class_names
        )
        
        # Compute SHAP values
        shap_values = interpreter.compute_shap_values(current_data['X_test'])
        
        # Handle different SHAP formats
        if isinstance(shap_values, list):
            # Multi-class: use class 1 (positive class for binary, or specific class)
            shap_vals = shap_values[1] if len(shap_values) == 2 else shap_values[-1]
        else:
            shap_vals = shap_values
        
        # Handle 3D arrays
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]  # Use class 1
        
        # Compute mean absolute SHAP
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        logger.info(f"✓ Fold {fold_num}, {model_type}: SHAP computed successfully")
        
        return {
            'fold_num': fold_num,
            'model_type': model_type,
            'shap_values': shap_vals,
            'mean_abs_shap': mean_abs_shap,
            'feature_names': current_data['feature_names'],
            'X_test': current_data['X_test'],
            'y_test': current_data['y_test']
        }
    
    except Exception as e:
        logger.error(f"✗ Fold {fold_num}, {model_type}: {e}")
        return None


def aggregate_shap_across_folds(fold_results: List[Dict], model_type: str) -> Dict:
    """
    Aggregate SHAP values across folds for a single model type.
    
    Handles varying feature sets across folds by finding common features.
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not fold_results:
        return None
    
    n_folds = len(fold_results)
    
    # Find common features across all folds
    feature_sets = [set(r['feature_names']) for r in fold_results]
    common_features = feature_sets[0].intersection(*feature_sets[1:])
    
    if not common_features:
        logger.warning(f"{model_type}: No common features across all folds!")
        return None
    
    common_features = sorted(list(common_features))
    n_features = len(common_features)
    
    logger.info(f"  Using {n_features} common features (folds had {[len(r['feature_names']) for r in fold_results]} features)")
    
    # Build SHAP matrix using only common features
    shap_matrix = np.zeros((n_folds, n_features))
    
    for fold_idx, result in enumerate(fold_results):
        # Map common features to this fold's feature indices
        for feat_idx, feat_name in enumerate(common_features):
            if feat_name in result['feature_names']:
                orig_idx = result['feature_names'].index(feat_name)
                shap_matrix[fold_idx, feat_idx] = result['mean_abs_shap'][orig_idx]
    
    # Compute statistics
    mean_shap = shap_matrix.mean(axis=0)
    std_shap = shap_matrix.std(axis=0)
    median_shap = np.median(shap_matrix, axis=0)
    cv_shap = std_shap / (mean_shap + 1e-10)  # Coefficient of variation
    
    # Rank consistency: how often is each feature in top 10 across folds
    # (only among common features)
    top_10_counts = np.zeros(n_features)
    for result in fold_results:
        # Get top 10 from this fold's common features
        fold_common_shap = []
        for feat_name in common_features:
            if feat_name in result['feature_names']:
                orig_idx = result['feature_names'].index(feat_name)
                fold_common_shap.append(result['mean_abs_shap'][orig_idx])
            else:
                fold_common_shap.append(0.0)
        
        fold_common_shap = np.array(fold_common_shap)
        top_10_indices = np.argsort(fold_common_shap)[-10:]
        top_10_counts[top_10_indices] += 1
    
    top_10_frequency = top_10_counts / n_folds
    
    return {
        'model_type': model_type,
        'n_folds': n_folds,
        'feature_names': common_features,
        'n_common_features': n_features,
        'mean_shap': mean_shap,
        'std_shap': std_shap,
        'median_shap': median_shap,
        'cv_shap': cv_shap,
        'top_10_frequency': top_10_frequency,
        'fold_results': fold_results
    }


def compare_models(model_aggregates: Dict[str, Dict], output_dir: Path):
    """
    Compare feature importance across different model types.
    
    Finds common features across all models for fair comparison.
    
    Creates comparison plots and rankings.
    """
    logger.info("=" * 80)
    logger.info("Comparing Feature Importance Across Models")
    logger.info("=" * 80)
    
    # Find common features across ALL models
    valid_models = {k: v for k, v in model_aggregates.items() if v is not None}
    
    if not valid_models:
        logger.error("No valid model data found")
        return
    
    # Get intersection of features across all models
    feature_sets = [set(v['feature_names']) for v in valid_models.values()]
    common_features = feature_sets[0].intersection(*feature_sets[1:]) if len(feature_sets) > 1 else feature_sets[0]
    feature_names = sorted(list(common_features))
    n_features = len(feature_names)
    
    logger.info(f"Using {n_features} features common to all models")
    for model_type, data in valid_models.items():
        logger.info(f"  {model_type}: {data['n_common_features']} features (across folds)")
    model_types = list(valid_models.keys())
    
    # Create comparison DataFrame using only common features
    comparison_data = {'feature': feature_names}
    
    for model_type in model_types:
        data = valid_models[model_type]
        
        # Map common features to this model's indices
        model_mean_shap = []
        model_cv = []
        model_top10 = []
        
        for feat_name in feature_names:
            if feat_name in data['feature_names']:
                idx = data['feature_names'].index(feat_name)
                model_mean_shap.append(data['mean_shap'][idx])
                model_cv.append(data['cv_shap'][idx])
                model_top10.append(data['top_10_frequency'][idx])
            else:
                model_mean_shap.append(0.0)
                model_cv.append(1.0)
                model_top10.append(0.0)
        
        comparison_data[f'{model_type}_mean'] = model_mean_shap
        comparison_data[f'{model_type}_cv'] = model_cv
        comparison_data[f'{model_type}_top10_freq'] = model_top10
    
    df = pd.DataFrame(comparison_data)
    
    # Compute consensus score: average across models
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    df['consensus_importance'] = df[mean_cols].mean(axis=1)
    
    # Compute stability score: how consistent across models
    df['cross_model_std'] = df[mean_cols].std(axis=1)
    df['cross_model_cv'] = df['cross_model_std'] / (df['consensus_importance'] + 1e-10)
    
    # Sort by consensus importance
    df = df.sort_values('consensus_importance', ascending=False)
    
    # Save comparison table
    csv_path = output_dir / "model_comparison_feature_importance.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved model comparison to {csv_path}")
    
    # Plot 1: Heatmap of feature importance across models
    plt.figure(figsize=(14, 12))
    top_n = 30
    top_features = df.head(top_n)
    
    heatmap_data = top_features[mean_cols].values
    
    # Normalize for visualization
    heatmap_norm = (heatmap_data - heatmap_data.min(axis=0)) / \
                   (heatmap_data.max(axis=0) - heatmap_data.min(axis=0) + 1e-10)
    
    sns.heatmap(heatmap_norm.T,
                xticklabels=top_features['feature'].values,
                yticklabels=[m.replace('_mean', '').title() for m in mean_cols],
                cmap='YlOrRd',
                cbar_kws={'label': 'Normalized SHAP Importance'})
    
    plt.title(f'Top {top_n} Features: Model Comparison\n(Column-normalized)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Feature')
    plt.ylabel('Model')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created model comparison heatmap")
    
    # Plot 2: Consensus features with model agreement
    plt.figure(figsize=(14, 10))
    top_n_consensus = 25
    top_consensus = df.head(top_n_consensus)
    
    x = np.arange(len(top_consensus))
    width = 0.8 / len(model_types)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for i, model_type in enumerate(model_types):
        if model_aggregates[model_type] is not None:
            col_name = f'{model_type}_mean'
            if col_name in top_consensus.columns:
                offset = width * (i - len(model_types)/2 + 0.5)
                ax.bar(x + offset, top_consensus[col_name], width, 
                      label=model_type.title(), alpha=0.8)
    
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Mean Absolute SHAP Value', fontsize=12)
    ax.set_title(f'Top {top_n_consensus} Consensus Features Across Models', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_consensus['feature'], rotation=90, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "consensus_features_grouped.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created consensus features plot")
    
    # Plot 3: Feature stability across models
    plt.figure(figsize=(12, 8))
    top_n_stable = 25
    
    # Sort by cross-model CV (lower = more stable)
    df_stable = df.sort_values('cross_model_cv')
    top_stable = df_stable.head(top_n_stable)
    
    colors = ['green' if cv < 0.3 else 'orange' if cv < 0.5 else 'red' 
              for cv in top_stable['cross_model_cv']]
    
    y_pos = np.arange(len(top_stable))
    plt.barh(y_pos, top_stable['cross_model_cv'], color=colors, alpha=0.7)
    plt.yticks(y_pos, top_stable['feature'])
    plt.xlabel('Cross-Model Coefficient of Variation')
    plt.title(f'Top {top_n_stable} Most Stable Features Across Models\n' +
              'Green: Stable (CV<0.3), Orange: Moderate (0.3≤CV<0.5), Red: Variable (CV≥0.5)',
              fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "cross_model_stability.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created cross-model stability plot")
    
    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("Cross-Model Feature Analysis Summary:")
    logger.info("=" * 80)
    
    logger.info(f"\nTop 10 Consensus Features (averaged across all models):")
    for i, row in df.head(10).iterrows():
        logger.info(f"  {row['feature']:50s} | Importance: {row['consensus_importance']:.4f} | CV: {row['cross_model_cv']:.4f}")
    
    logger.info(f"\nTop 10 Most Stable Features (low cross-model variance):")
    for i, row in df_stable.head(10).iterrows():
        logger.info(f"  {row['feature']:50s} | Importance: {row['consensus_importance']:.4f} | CV: {row['cross_model_cv']:.4f}")
    
    # Save summary JSON
    summary = {
        'n_models': len(model_types),
        'n_features': n_features,
        'model_types': model_types,
        'top_10_consensus': df.head(10)[['feature', 'consensus_importance', 'cross_model_cv']].to_dict('records'),
        'top_10_stable': df_stable.head(10)[['feature', 'consensus_importance', 'cross_model_cv']].to_dict('records')
    }
    
    json_path = output_dir / "model_comparison_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to {json_path}")


def create_ensemble_importance(model_aggregates: Dict[str, Dict], output_dir: Path):
    """
    Create ensemble-based feature importance by voting/averaging across all models and folds.
    
    Uses common features across all models for fair comparison.
    
    This simulates what an ensemble model would learn.
    """
    logger.info("=" * 80)
    logger.info("Computing Ensemble Feature Importance")
    logger.info("=" * 80)
    
    # Find common features across all models
    valid_models = {k: v for k, v in model_aggregates.items() if v is not None}
    
    if not valid_models:
        logger.error("No valid model data found")
        return
    
    # Get intersection of features
    feature_sets = [set(v['feature_names']) for v in valid_models.values()]
    common_features = feature_sets[0].intersection(*feature_sets[1:]) if len(feature_sets) > 1 else feature_sets[0]
    feature_names = sorted(list(common_features))
    n_features = len(feature_names)
    
    logger.info(f"Using {n_features} features common to all models")
    
    # Strategy 1: Simple average across models (using common features)
    all_mean_shaps = []
    model_weights = {}
    
    for model_type, data in valid_models.items():
        # Map common features to this model's SHAP values
        model_shap = []
        for feat_name in feature_names:
            if feat_name in data['feature_names']:
                idx = data['feature_names'].index(feat_name)
                model_shap.append(data['mean_shap'][idx])
            else:
                model_shap.append(0.0)
        
        all_mean_shaps.append(np.array(model_shap))
        model_weights[model_type] = 1.0  # Equal weighting
    
    if not all_mean_shaps:
        logger.error("No model data available for ensemble")
        return
    
    # Equal-weighted ensemble
    ensemble_equal = np.mean(all_mean_shaps, axis=0)
    
    # Strategy 2: Stability-weighted (weight by inverse of CV)
    weighted_shaps = []
    weights = []
    
    for model_type, data in valid_models.items():
        # Map common features
        model_shap = []
        model_cv = []
        for feat_name in feature_names:
            if feat_name in data['feature_names']:
                idx = data['feature_names'].index(feat_name)
                model_shap.append(data['mean_shap'][idx])
                model_cv.append(data['cv_shap'][idx])
            else:
                model_shap.append(0.0)
                model_cv.append(1.0)
        
        model_shap = np.array(model_shap)
        model_cv = np.array(model_cv)
        
        # Weight inversely proportional to CV (more stable = higher weight)
        cv_mean = model_cv.mean()
        weight = 1.0 / (cv_mean + 0.1)  # Add small constant to avoid division by zero
        weighted_shaps.append(model_shap * weight)
        weights.append(weight)
        model_weights[model_type] = weight
    
    weights = np.array(weights)
    ensemble_weighted = np.sum(weighted_shaps, axis=0) / weights.sum()
    
    # Strategy 3: Voting - count how many models put feature in top 20
    top_k = 20
    vote_counts = np.zeros(n_features)
    
    for model_type, data in valid_models.items():
        # Map common features
        model_shap = []
        for feat_name in feature_names:
            if feat_name in data['feature_names']:
                idx = data['feature_names'].index(feat_name)
                model_shap.append(data['mean_shap'][idx])
            else:
                model_shap.append(0.0)
        
        model_shap = np.array(model_shap)
        top_k_indices = np.argsort(model_shap)[-top_k:]
        vote_counts[top_k_indices] += 1
    
    vote_frequency = vote_counts / len(valid_models)
    
    # Create ensemble DataFrame
    ensemble_df = pd.DataFrame({
        'feature': feature_names,
        'ensemble_equal': ensemble_equal,
        'ensemble_weighted': ensemble_weighted,
        'vote_frequency': vote_frequency,
        'vote_count': vote_counts
    })
    
    # Add individual model importances
    for model_type, data in valid_models.items():
        # Map common features
        model_shap = []
        model_cv = []
        for feat_name in feature_names:
            if feat_name in data['feature_names']:
                idx = data['feature_names'].index(feat_name)
                model_shap.append(data['mean_shap'][idx])
                model_cv.append(data['cv_shap'][idx])
            else:
                model_shap.append(0.0)
                model_cv.append(1.0)
        
        ensemble_df[f'{model_type}_importance'] = model_shap
        ensemble_df[f'{model_type}_cv'] = model_cv
    
    # Sort by ensemble importance
    ensemble_df = ensemble_df.sort_values('ensemble_weighted', ascending=False)
    
    # Save ensemble table
    csv_path = output_dir / "ensemble_feature_importance.csv"
    ensemble_df.to_csv(csv_path, index=False)
    logger.info(f"Saved ensemble importance to {csv_path}")
    
    # Plot 1: Ensemble importance (weighted)
    plt.figure(figsize=(12, 10))
    top_n = 30
    top_ensemble = ensemble_df.head(top_n)
    
    y_pos = np.arange(len(top_ensemble))
    plt.barh(y_pos, top_ensemble['ensemble_weighted'], alpha=0.8, color='purple')
    plt.yticks(y_pos, top_ensemble['feature'])
    plt.xlabel('Ensemble SHAP Importance (Stability-Weighted)')
    plt.title(f'Top {top_n} Features: Ensemble Importance\n(Weighted by model stability across folds)',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_importance_weighted.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created ensemble importance plot (weighted)")
    
    # Plot 2: Voting frequency
    plt.figure(figsize=(12, 10))
    top_voted = ensemble_df.nlargest(30, 'vote_frequency')
    
    y_pos = np.arange(len(top_voted))
    colors = plt.cm.RdYlGn(top_voted['vote_frequency'])
    plt.barh(y_pos, top_voted['vote_frequency'], color=colors, alpha=0.8)
    plt.yticks(y_pos, top_voted['feature'])
    plt.xlabel(f'Voting Frequency (proportion of models ranking in top {top_k})')
    plt.title(f'Top 30 Features: Ensemble Voting\n(How often each feature ranks in top {top_k} across models)',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.xlim([0, 1.0])
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_voting_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created ensemble voting plot")
    
    # Plot 3: Comparison of ensemble strategies
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    
    top_n_comp = 20
    
    # Equal-weighted
    top_equal = ensemble_df.nlargest(top_n_comp, 'ensemble_equal')
    axes[0].barh(range(len(top_equal)), top_equal['ensemble_equal'], color='steelblue', alpha=0.8)
    axes[0].set_yticks(range(len(top_equal)))
    axes[0].set_yticklabels(top_equal['feature'])
    axes[0].set_xlabel('SHAP Importance')
    axes[0].set_title('Equal-Weighted Ensemble')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Stability-weighted
    top_weighted = ensemble_df.nlargest(top_n_comp, 'ensemble_weighted')
    axes[1].barh(range(len(top_weighted)), top_weighted['ensemble_weighted'], color='darkgreen', alpha=0.8)
    axes[1].set_yticks(range(len(top_weighted)))
    axes[1].set_yticklabels(top_weighted['feature'])
    axes[1].set_xlabel('SHAP Importance')
    axes[1].set_title('Stability-Weighted Ensemble')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    # Voting
    top_voted_comp = ensemble_df.nlargest(top_n_comp, 'vote_frequency')
    axes[2].barh(range(len(top_voted_comp)), top_voted_comp['vote_frequency'], color='darkorange', alpha=0.8)
    axes[2].set_yticks(range(len(top_voted_comp)))
    axes[2].set_yticklabels(top_voted_comp['feature'])
    axes[2].set_xlabel('Vote Frequency')
    axes[2].set_title('Voting-Based Ensemble')
    axes[2].invert_yaxis()
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Top {top_n_comp} Features: Ensemble Strategy Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created ensemble strategy comparison")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Ensemble Feature Importance Summary:")
    logger.info("=" * 80)
    
    logger.info(f"\nModel weights (based on stability):")
    for model_type, weight in sorted(model_weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {model_type:20s}: {weight:.4f}")
    
    logger.info(f"\nTop 10 Ensemble Features (stability-weighted):")
    for i, row in ensemble_df.head(10).iterrows():
        logger.info(f"  {row['feature']:50s} | Importance: {row['ensemble_weighted']:.4f} | Votes: {int(row['vote_count'])}/{len(valid_models)}")
    
    # Save ensemble summary
    summary = {
        'ensemble_strategy': 'stability_weighted',
        'n_models': len(valid_models),
        'n_common_features': n_features,
        'model_weights': model_weights,
        'top_10_features': ensemble_df.head(10)[['feature', 'ensemble_weighted', 'vote_frequency']].to_dict('records'),
        'top_20_by_voting': ensemble_df.nlargest(20, 'vote_frequency')[['feature', 'vote_frequency', 'vote_count']].to_dict('records')
    }
    
    json_path = output_dir / "ensemble_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nEnsemble summary saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive multi-fold, multi-model SHAP analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--cv_dir', type=str, required=True,
                       help='Directory containing outercv_fold_* subdirectories')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file with radiomics features')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for comprehensive results')
    parser.add_argument('--model_types', nargs='+', type=str,
                       default=['randomforest', 'extratrees', 'gradientboosting', 
                                'xgboost', 'lightgbm', 'svm', 'logisticregression', 'knn'],
                       help='Model types to analyze (space-separated). Default: all 8 models')
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
    
    # Run comprehensive analysis
    logger.info("=" * 80)
    logger.info("Starting Comprehensive Multi-Fold, Multi-Model SHAP Analysis")
    logger.info(f"Folds: {len(fold_dirs)} | Models: {len(args.model_types)}")
    logger.info("=" * 80)
    
    # Collect SHAP results for all models across all folds
    all_results = defaultdict(list)  # model_type -> list of fold results
    
    for fold_dir in fold_dirs:
        fold_num = int(fold_dir.name.split('_')[-1])
        logger.info(f"\nProcessing Fold {fold_num}...")
        
        for model_type in args.model_types:
            result = get_shap_for_model_fold(
                fold_dir=fold_dir,
                model_type=model_type,
                data=data,
                class_names=args.class_names or ['Class 0', 'Class 1']
            )
            
            if result is not None:
                all_results[model_type].append(result)
    
    # Check if we got any results
    if not any(all_results.values()):
        logger.error("No successful SHAP analyses! Check errors above.")
        sys.exit(1)
    
    # Aggregate across folds for each model
    logger.info("\n" + "=" * 80)
    logger.info("Aggregating Results Across Folds")
    logger.info("=" * 80)
    
    model_aggregates = {}
    for model_type in args.model_types:
        if all_results[model_type]:
            logger.info(f"\nAggregating {model_type}: {len(all_results[model_type])} folds")
            model_aggregates[model_type] = aggregate_shap_across_folds(
                all_results[model_type], model_type
            )
        else:
            logger.warning(f"No results for {model_type}")
            model_aggregates[model_type] = None
    
    # Compare models
    compare_models(model_aggregates, output_dir)
    
    # Create ensemble importance
    create_ensemble_importance(model_aggregates, output_dir)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Comprehensive SHAP Analysis Complete!")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {output_dir.absolute()}")
    logger.info("\nGenerated files:")
    logger.info("  - model_comparison_feature_importance.csv")
    logger.info("  - model_comparison_heatmap.png")
    logger.info("  - consensus_features_grouped.png")
    logger.info("  - cross_model_stability.png")
    logger.info("  - ensemble_feature_importance.csv")
    logger.info("  - ensemble_importance_weighted.png")
    logger.info("  - ensemble_voting_frequency.png")
    logger.info("  - ensemble_strategy_comparison.png")
    logger.info("  - model_comparison_summary.json")
    logger.info("  - ensemble_summary.json")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

