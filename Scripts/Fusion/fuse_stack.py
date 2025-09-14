#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold


def load_and_merge(radiomics_csv: str, deep_csv: str, master_csv: Optional[str] = None) -> pd.DataFrame:
    rad = pd.read_csv(radiomics_csv)
    deep = pd.read_csv(deep_csv)

    # Keep only necessary columns from radiomics if accidental extras
    if 'image_path' in rad.columns:
        rad = rad.drop(columns=['image_path'])

    # Inner join on subject_id to ensure alignment
    df = pd.merge(rad, deep, on=['subject_id', 'label'], how='inner')

    if master_csv:
        master = pd.read_csv(master_csv)
        if 'subject_id' not in master.columns or 'label' not in master.columns:
            # Try headerless
            master = pd.read_csv(master_csv, header=None, names=['subject_id', 'label'])
        # sanity: keep only subjects present in master
        df = pd.merge(df, master[['subject_id', 'label']], on=['subject_id', 'label'], how='inner')

    # Drop duplicates and NaNs
    df = df.drop_duplicates(subset=['subject_id']).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return df


def pick_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Radiomics: everything not subject_id/label/emb_*/logit_*
    exclude = { 'subject_id', 'label' }
    deep_cols = [c for c in df.columns if c.startswith('emb_') or c.startswith('logit_')]
    rad_cols = [c for c in df.columns if c not in exclude and c not in deep_cols]
    return rad_cols, deep_cols


def compute_macro_ap(y_true: np.ndarray, probas: np.ndarray) -> float:
    # One-vs-rest AP, macro average
    classes = np.unique(y_true)
    aps = []
    for c in classes:
        y_bin = (y_true == c).astype(int)
        ap = average_precision_score(y_bin, probas[:, c])
        aps.append(ap)
    return float(np.mean(aps)) if aps else float('nan')


def main():
    ap = argparse.ArgumentParser(description='Stacking fusion of deep features + radiomics (k-fold)')
    ap.add_argument('--radiomics_csv', required=True)
    ap.add_argument('--deep_csv', required=True, help='CSV exported by export_deep_features.py')
    ap.add_argument('--master_csv', default=None)
    ap.add_argument('--k_folds', type=int, default=5)
    ap.add_argument('--random_seed', type=int, default=42)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', choices=['logreg', 'optimized_ensemble'], default='optimized_ensemble')
    ap.add_argument('--use', choices=['embeddings', 'logits', 'both'], default='both')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = load_and_merge(args.radiomics_csv, args.deep_csv, args.master_csv)
    rad_cols, deep_cols = pick_feature_columns(df)
    if args.use == 'embeddings':
        deep_cols = [c for c in deep_cols if c.startswith('emb_')]
    elif args.use == 'logits':
        deep_cols = [c for c in deep_cols if c.startswith('logit_')]

    X_rad = df[rad_cols].to_numpy(dtype=np.float32)
    X_deep = df[deep_cols].to_numpy(dtype=np.float32) if deep_cols else None
    y = df['label'].to_numpy(dtype=int)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)

    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_rad, y), start=1):
        Xr_tr, Xr_va = X_rad[tr_idx], X_rad[va_idx]
        yr, yv = y[tr_idx], y[va_idx]
        if X_deep is not None:
            Xd_tr, Xd_va = X_deep[tr_idx], X_deep[va_idx]
        else:
            Xd_tr = Xd_va = None

        rad_scaler = StandardScaler().fit(Xr_tr)
        Xr_tr_s = rad_scaler.transform(Xr_tr)
        Xr_va_s = rad_scaler.transform(Xr_va)

        if Xd_tr is not None:
            deep_scaler = StandardScaler(with_mean=True, with_std=True).fit(Xd_tr)
            Xd_tr_s = deep_scaler.transform(Xd_tr)
            Xd_va_s = deep_scaler.transform(Xd_va)
            Xtr = np.hstack([Xr_tr_s, Xd_tr_s])
            Xva = np.hstack([Xr_va_s, Xd_va_s])
        else:
            deep_scaler = None
            Xtr = Xr_tr_s
            Xva = Xr_va_s

        # Build meta-learner
        if args.model == 'logreg':
            classes, counts = np.unique(yr, return_counts=True)
            total = counts.sum()
            class_weight = {int(c): float(total / (len(classes) * n)) for c, n in zip(classes, counts)}
            clf = LogisticRegression(
                max_iter=5000, multi_class='multinomial', solver='saga', n_jobs=4,
                class_weight=class_weight, verbose=0
            )
        else:
            # Attempt to include XGBoost and LightGBM if available
            estimators = []
            try:
                import xgboost as xgb  # type: ignore
                estimators.append(('xgb', xgb.XGBClassifier(
                    objective='multi:softprob', num_class=len(np.unique(yr)),
                    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    tree_method='hist', reg_lambda=1.0, random_state=args.random_seed)))
            except Exception:
                pass
            try:
                from lightgbm import LGBMClassifier  # type: ignore
                estimators.append(('lgbm', LGBMClassifier(
                    objective='multiclass', num_class=len(np.unique(yr)),
                    n_estimators=400, max_depth=-1, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=args.random_seed)))
            except Exception:
                pass
            estimators.append(('rf', RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=4, class_weight='balanced_subsample', random_state=args.random_seed)))
            estimators.append(('gb', GradientBoostingClassifier(random_state=args.random_seed)))
            estimators.append(('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=args.random_seed)))

            clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=4, flatten_transform=True)

        clf.fit(Xtr, yr)
        prob = clf.predict_proba(Xva)
        pred = prob.argmax(axis=1)

        acc = accuracy_score(yv, pred)
        cm = confusion_matrix(yv, pred).tolist()
        try:
            auc = roc_auc_score(yv, prob, multi_class='ovr')
        except Exception:
            auc = float('nan')
        ap_macro = compute_macro_ap(yv, prob)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(yv, pred, average='macro', zero_division=0)
        report = classification_report(yv, pred, output_dict=True, zero_division=0)

        fold_dir = Path(args.out_dir) / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': clf,
            'rad_scaler': rad_scaler,
            'deep_scaler': deep_scaler,
            'rad_cols': rad_cols,
            'deep_cols': deep_cols,
        }, fold_dir / 'stack_model.joblib')

        metrics = {
            'fold': fold,
            'val_acc': float(acc),
            'val_auc_ovr': float(auc),
            'val_ap_macro': float(ap_macro),
            'val_precision_macro': float(prec_m),
            'val_recall_macro': float(rec_m),
            'val_f1_macro': float(f1_m),
            'confusion_matrix': cm,
            'classification_report': report,
        }
        with open(fold_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        fold_metrics.append(metrics)
        print(f"Fold {fold}: acc={acc:.4f} auc={auc:.4f} ap_macro={ap_macro:.4f}")

    # Summary
    def avg(key):
        vals = [m[key] for m in fold_metrics if np.isfinite(m[key])]
        return float(np.mean(vals)) if vals else float('nan')

    summary = {
        'folds': len(fold_metrics),
        'val_acc_mean': avg('val_acc'),
        'val_auc_ovr_mean': avg('val_auc_ovr'),
        'val_ap_macro_mean': avg('val_ap_macro'),
        'val_f1_macro_mean': avg('val_f1_macro'),
    }
    with open(Path(args.out_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Summary:", summary)


if __name__ == '__main__':
    main()


