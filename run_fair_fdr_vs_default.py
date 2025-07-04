#!/usr/bin/env python3
"""
Fair FDR vs Default Feature Selection Comparison (Full Pipeline)
==============================================================

This script performs a fair, automated comparison between FDR feature selection
and the default (MutualInfo+RFECV) feature selection using the full improved optimised pipeline
and the same train/val/test splits.

Usage:
    python run_fair_fdr_vs_default.py --input radiomics_features.csv --output results/
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add the Optimised directory to the path
sys.path.append(str(Path(__file__).parent / "Optimised"))
from improved_optimized_classifier import ImprovedOptimizedRadiomicsClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.feature_selection import VarianceThreshold

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fair_fdr_vs_default")


def load_and_split_data(input_path, binary_only=True, test_size=0.2, val_size=0.2, random_state=42):
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    # Remove diagnostic columns
    diagnostic_cols = [col for col in df.columns if col.startswith('diagnostics_')]
    if diagnostic_cols:
        df = df.drop(columns=diagnostic_cols)
        logger.info(f"Removed {len(diagnostic_cols)} diagnostic columns")
    # Filter for binary classification
    if binary_only:
        initial_count = len(df)
        df = df[df['label'].isin([0, 1])]
        final_count = len(df)
        logger.info(f"Filtered to binary classification: {initial_count} → {final_count} samples")
        if final_count == 0:
            raise ValueError("No samples remaining after binary filtering")
        unique_labels = df['label'].unique()
        if len(unique_labels) != 2 or not all(label in [0, 1] for label in unique_labels):
            raise ValueError(f"Expected binary labels [0, 1], got: {unique_labels}")
    subject_ids = df['subject_id'].values
    y = df['label'].values
    feature_names = [col for col in df.columns if col not in ['subject_id', 'label']]
    X = df[feature_names].values
    # Split
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, subject_ids, test_size=test_size, random_state=random_state, stratify=y
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    # Log label counts
    logger.info(f"Label counts - Train: {dict(enumerate(np.bincount(y_train)))}, Val: {dict(enumerate(np.bincount(y_val)))}, Test: {dict(enumerate(np.bincount(y_test)))}")
    logger.info(f"Data splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, y_test, ids_test, feature_names, df)


def basic_preprocessing(X_train, X_val, X_test):
    variance_selector = VarianceThreshold(threshold=0.01)
    scaler = RobustScaler()
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)
    X_test_var = variance_selector.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_var)
    X_val_scaled = scaler.transform(X_val_var)
    X_test_scaled = scaler.transform(X_test_var)
    return X_train_scaled, X_val_scaled, X_test_scaled, variance_selector, scaler


def fdr_feature_selection(X_train, y_train, feature_names, fdr_alpha=0.05):
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available, skipping FDR selection")
        return feature_names, []
    f_scores, p_values = f_classif(X_train, y_train)
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=fdr_alpha, method='fdr_bh')
    selected_indices = np.where(rejected)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    logger.info(f"FDR selection: {len(feature_names)} → {len(selected_features)} features (alpha={fdr_alpha})")
    return selected_features, p_corrected.tolist()


def default_feature_selection(X_train, y_train, X_val, X_test, feature_names):
    k_best = min(50, X_train.shape[1] // 2)
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
    X_train_mi = mi_selector.fit_transform(X_train, y_train)
    X_val_mi = mi_selector.transform(X_val)
    X_test_mi = mi_selector.transform(X_test)
    mi_mask = mi_selector.get_support()
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if mi_mask[i]]
    estimator = LogisticRegression(random_state=42, max_iter=1000)
    rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc', min_features_to_select=10, n_jobs=-1)
    X_train_final = rfecv.fit_transform(X_train_mi, y_train)
    X_val_final = rfecv.transform(X_val_mi)
    X_test_final = rfecv.transform(X_test_mi)
    rfecv_mask = rfecv.get_support()
    final_feature_names = [selected_feature_names[i] for i in range(len(selected_feature_names)) if rfecv_mask[i]]
    logger.info(f"Default selection: {len(feature_names)} → {len(final_feature_names)} features (MutualInfo+RFECV)")
    return X_train_final, X_val_final, X_test_final, final_feature_names


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, random_state=42):
    results = {}
    # SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=random_state, max_iter=10000, tol=1e-3)
    svm.fit(X_train, y_train)
    results['svm'] = evaluate_model(svm, X_train, y_train, X_val, y_val, X_test, y_test, 'SVM')
    # Ensemble
    base_models = {
        'svm_linear': SVC(kernel='linear', probability=True, random_state=random_state, max_iter=10000),
        'logistic': LogisticRegression(random_state=random_state, max_iter=1000)
    }
    ensemble = VotingClassifier(estimators=[(name, model) for name, model in base_models.items()], voting='soft')
    ensemble.fit(X_train, y_train)
    results['ensemble'] = evaluate_model(ensemble, X_train, y_train, X_val, y_val, X_test, y_test, 'Ensemble')
    return results


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    out = {}
    for split, (X, y) in zip(['train', 'val', 'test'], [(X_train, y_train), (X_val, y_val), (X_test, y_test)]):
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        out[split] = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'auc': roc_auc_score(y, y_pred_proba),
            'mcc': matthews_corrcoef(y, y_pred)
        }
        logger.info(f"{model_name} {split} - Accuracy: {out[split]['accuracy']:.4f}, AUC: {out[split]['auc']:.4f}, MCC: {out[split]['mcc']:.4f}")
    return out


def main():
    parser = argparse.ArgumentParser(description='Fair FDR vs Default Feature Selection Comparison (Full Pipeline)')
    parser.add_argument('--input', type=str, required=True, help='Path to radiomics CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--fdr_alpha', type=float, default=0.05, help='FDR significance level')
    parser.add_argument('--binary_only', action='store_true', help='Use only binary classification')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and split data
    X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, y_test, ids_test, feature_names, df = load_and_split_data(
        args.input, binary_only=args.binary_only, random_state=args.random_state)

    # 2. Preprocessing
    X_train_proc, X_val_proc, X_test_proc, variance_selector, scaler = basic_preprocessing(X_train, X_val, X_test)
    processed_feature_names = [f for i, f in enumerate(feature_names) if variance_selector.get_support()[i]]

    # 3. FDR feature selection
    selected_fdr_features, fdr_pvals = fdr_feature_selection(X_train_proc, y_train, processed_feature_names, fdr_alpha=args.fdr_alpha)

    # 4. Default feature selection
    X_train_def, X_val_def, X_test_def, def_features = default_feature_selection(
        X_train_proc, y_train, X_val_proc, X_test_proc, processed_feature_names)

    # 5. Train and evaluate on both feature sets
    logger.info("\n=== FDR Feature Selection Results ===")
    fdr_results = train_and_evaluate(X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test, random_state=args.random_state)
    logger.info("\n=== Default Feature Selection Results ===")
    def_results = train_and_evaluate(X_train_def, y_train, X_val_def, y_val, X_test_def, y_test, random_state=args.random_state)

    # 6. Save results
    results = {
        'fdr': fdr_results,
        'default': def_results,
        'fdr_features': selected_fdr_features,
        'default_features': def_features,
        'fdr_pvals': fdr_pvals
    }
    with open(output_dir / 'fair_fdr_vs_default_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(output_dir / 'fair_fdr_vs_default_report.txt', 'w') as f:
        f.write("FAIR FDR VS DEFAULT FEATURE SELECTION COMPARISON\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Input File: {args.input}\n")
        f.write(f"FDR Alpha: {args.fdr_alpha}\n\n")
        for method, res in [('FDR', fdr_results), ('Default', def_results)]:
            f.write(f"{method} Feature Selection:\n")
            for model, splits in res.items():
                f.write(f"  {model} Model:\n")
                for split, metrics in splits.items():
                    f.write(f"    {split.capitalize()}: ")
                    f.write(", ".join([f"{k}: {metrics[k]:.4f}" for k in ['accuracy','auc','mcc']]))
                    f.write("\n")
            f.write("\n")
        f.write(f"FDR Features ({len(selected_fdr_features)}): {selected_fdr_features}\n\n")
        f.write(f"Default Features ({len(def_features)}): {def_features}\n\n")
    # Save models
    with open(output_dir / 'fdr_svm_model.pkl', 'wb') as f:
        pickle.dump(fdr_results['svm'], f)
    with open(output_dir / 'default_svm_model.pkl', 'wb') as f:
        pickle.dump(def_results['svm'], f)
    logger.info("\nComparison complete. Results saved to: %s", output_dir)

    # 7. Save split indices for reproducibility
    split_info = {
        'train_ids': ids_train.tolist(),
        'val_ids': ids_val.tolist(),
        'test_ids': ids_test.tolist()
    }
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    # 8. Save FDR features
    with open(output_dir / 'fdr_features.json', 'w') as f:
        json.dump({'fdr_features': selected_fdr_features, 'fdr_pvals': fdr_pvals}, f, indent=2)

    # 9. Run improved pipeline with default feature selection
    logger.info("\n=== Running Improved Optimised Pipeline (Default Feature Selection) ===")
    default_outdir = output_dir / 'default_features_run'
    default_outdir.mkdir(exist_ok=True)
    clf_default = ImprovedOptimizedRadiomicsClassifier(
        input_path=args.input,
        output_dir=default_outdir,
        random_state=args.random_state,
        binary_only=args.binary_only,
        selected_features=None
    )
    # Overwrite splits for fair comparison
    clf_default.load_data()
    clf_default.splits = {
        'train': (X_train, y_train, ids_train),
        'val': (X_val, y_val, ids_val),
        'test': (X_test, y_test, ids_test)
    }
    clf_default.run_improved_pipeline()

    # 10. Run improved pipeline with FDR features
    logger.info("\n=== Running Improved Optimised Pipeline (FDR Feature Selection) ===")
    fdr_outdir = output_dir / 'fdr_features_run'
    fdr_outdir.mkdir(exist_ok=True)
    clf_fdr = ImprovedOptimizedRadiomicsClassifier(
        input_path=args.input,
        output_dir=fdr_outdir,
        random_state=args.random_state,
        binary_only=args.binary_only,
        selected_features=selected_fdr_features
    )
    clf_fdr.load_data()
    clf_fdr.splits = {
        'train': (X_train[:, [feature_names.index(f) for f in selected_fdr_features]], y_train, ids_train),
        'val': (X_val[:, [feature_names.index(f) for f in selected_fdr_features]], y_val, ids_val),
        'test': (X_test[:, [feature_names.index(f) for f in selected_fdr_features]], y_test, ids_test)
    }
    clf_fdr.run_improved_pipeline()

    # 11. Save summary report
    with open(output_dir / 'fair_full_fdr_vs_default_report.txt', 'w') as f:
        f.write("FAIR FDR VS DEFAULT FEATURE SELECTION COMPARISON (FULL PIPELINE)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Input File: {args.input}\n")
        f.write(f"FDR Alpha: {args.fdr_alpha}\n\n")
        f.write("Default Feature Selection Results in: default_features_run/\n")
        f.write("FDR Feature Selection Results in: fdr_features_run/\n\n")
        f.write("Splits and features are identical for both runs.\n")
        f.write("See each subdirectory for full model outputs, metrics, and logs.\n")
    logger.info("\nComparison complete. Results saved to: %s", output_dir)

if __name__ == "__main__":
    main() 