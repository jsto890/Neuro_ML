#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import re
import json
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix


def find_checkpoints(run_dir: Path, k_folds: int) -> dict:
    patterns = [
        re.compile(r"best_.*_fold_(\d+)\.pth$"),
        re.compile(r"best_(?:smri|pet)_model_fold_(\d+)\.pth$"),
    ]
    found = {}
    for p in run_dir.rglob("*.pth"):
        for pat in patterns:
            m = pat.search(p.name)
            if m:
                fold = int(m.group(1))
                if 1 <= fold <= k_folds:
                    found[fold] = p
                break
    return found


def export_features_for_split(split_csv: Path, ckpt_path: Path, args, tag: str, base_channels: int) -> Path:
    out_csv = Path(args.out_dir) / f"deep_features_{args.modality.lower()}_{tag}.csv"
    cmd = [
        sys.executable, str(Path(__file__).parent / 'export_deep_features.py'),
        '--modality', args.modality,
        '--model', args.backbone,
        '--checkpoint', str(ckpt_path),
        '--csv', str(split_csv),
        '--data_root', args.data_root,
        '--out_csv', str(out_csv),
        '--base_channels', str(base_channels),
        '--export', args.export,
        '--device', args.device,
    ]
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out_csv


def fuse_with_radiomics(rad_csv: Path, deep_csv: Path) -> pd.DataFrame:
    rad = pd.read_csv(rad_csv)
    deep = pd.read_csv(deep_csv)
    fused = pd.merge(rad, deep, on=['subject_id', 'label'], how='inner')
    return fused


def train_enhanced_on_train_only(fused_train_csv: Path, out_dir: Path, threads: int, multi_class: bool, random_state: int):
    # Train Enhanced on training data only (no outer CV); saves models and scaler
    cmd = [
        sys.executable, str(Path(__file__).resolve().parents[2] / 'Scripts/Classic_Learning/Enhanced/run_enhanced.py'),
        '--input', str(fused_train_csv),
        '--output-dir', str(out_dir),
        '--random-state', str(random_state),
        '--ml-threads', str(threads),
        '--outer-k-folds', '0',
    ]
    if multi_class:
        cmd.append('--multi-class')
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def enhanced_predict_proba(enhanced_dir: Path, X: pd.DataFrame) -> np.ndarray:
    # Load scaler and models saved by Enhanced pipeline
    import joblib
    # Handle timestamped run subdirectory created by run_enhanced.py
    base_dir = enhanced_dir
    # If models not in enhanced_dir, pick the newest run_* subdir
    if not any((enhanced_dir / name).exists() for name in ['randomforest_model.pkl', 'svm_model.pkl', 'logisticregression_model.pkl', 'gradientboosting_model.pkl', 'scaler.pkl']):
        candidates = [d for d in enhanced_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        if candidates:
            base_dir = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    scaler_path = base_dir / 'scaler.pkl'
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    # Gather model pkls
    model_paths = []
    for pat in ['randomforest_model.pkl', 'svm_model.pkl', 'logisticregression_model.pkl', 'gradientboosting_model.pkl']:
        p = base_dir / pat
        if p.exists():
            model_paths.append(p)
    if not model_paths:
        raise RuntimeError(f"No models found in {base_dir}")
    models = [joblib.load(p) for p in model_paths]
    feats = X.copy()
    if scaler is not None:
        feats[feats.columns] = scaler.transform(feats)
    # Average predict_proba across models
    probs = None
    for m in models:
        p = m.predict_proba(feats)
        probs = p if probs is None else (probs + p)
    probs /= len(models)
    return probs


def score_fold(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = prob.argmax(axis=1)
    acc = accuracy_score(y_true, pred)
    try:
        auc = roc_auc_score(y_true, prob, multi_class='ovr')
    except Exception:
        auc = float('nan')
    ap_macro = float(np.mean([average_precision_score((y_true == c).astype(int), prob[:, c]) for c in np.unique(y_true)]))
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, pred).tolist()
    return {
        'acc': float(acc), 'auc_ovr': float(auc), 'ap_macro': float(ap_macro),
        'precision_macro': float(prec_m), 'recall_macro': float(rec_m), 'f1_macro': float(f1_m),
        'confusion_matrix': cm,
    }


def main():
    ap = argparse.ArgumentParser(description='Automated fusion: export deep features across folds and run enhanced radiomics ensemble')
    ap.add_argument('--modality', required=True, choices=['PET', 'MRI'])
    ap.add_argument('--backbone', required=True, help='e.g., Simple3DCNN or DenseNet121_3D')
    ap.add_argument('--dl_run_dir', required=True, help='Root run directory containing fold checkpoints')
    ap.add_argument('--master_csv', required=True)
    ap.add_argument('--radiomics_csv', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--export', choices=['embeddings', 'logits', 'both'], default='embeddings')
    ap.add_argument('--k_folds', type=int, default=5)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--multi_class', action='store_true')
    ap.add_argument('--ml_threads', type=int, default=4)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Find fold checkpoints
    ckpts = find_checkpoints(Path(args.dl_run_dir), args.k_folds)
    if len(ckpts) == 0:
        print(f"No fold checkpoints found in {args.dl_run_dir}")
        sys.exit(1)
    print('Found checkpoints:', {k: str(v) for k, v in sorted(ckpts.items())})

    # 2) Deterministically recreate outer folds using StratifiedKFold with the same seed
    master = pd.read_csv(args.master_csv)
    if 'subject_id' not in master.columns or 'label' not in master.columns:
        master = pd.read_csv(args.master_csv, header=None, names=['subject_id', 'label'])
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)

    fold_metrics: List[Dict] = []
    for fold_idx, (train_pool_idx, test_idx) in enumerate(skf.split(master['subject_id'], master['label']), start=1):
        if fold_idx not in ckpts:
            print(f"[WARN] No checkpoint found for fold {fold_idx}; skipping")
            continue
        fold_dir = out_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Split train pool into train/val deterministically
        train_pool = master.iloc[train_pool_idx].copy()
        test_df = master.iloc[test_idx].copy()
        train_df, val_df = train_test_split(
            train_pool, test_size=args.val_ratio, stratify=train_pool['label'], random_state=args.random_state
        )
        # Save fold CSVs for reference
        train_csv = fold_dir / 'train.csv'
        val_csv   = fold_dir / 'val.csv'
        test_csv  = fold_dir / 'test.csv'
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        # 3) Export deep features per split using the fold checkpoint
        ckpt_path = ckpts[fold_idx]
        deep_train = export_features_for_split(train_csv, ckpt_path, args, f'fold{fold_idx}_train', args.base_channels)
        deep_val   = export_features_for_split(val_csv,   ckpt_path, args, f'fold{fold_idx}_val', args.base_channels)
        deep_test  = export_features_for_split(test_csv,  ckpt_path, args, f'fold{fold_idx}_test', args.base_channels)

        # 4) Fuse with radiomics for classic training (train+val) and evaluation (test)
        fused_train = pd.concat([
            fuse_with_radiomics(Path(args.radiomics_csv), Path(deep_train)),
            fuse_with_radiomics(Path(args.radiomics_csv), Path(deep_val)),
        ], ignore_index=True).drop_duplicates(subset=['subject_id'])
        fused_test  = fuse_with_radiomics(Path(args.radiomics_csv), Path(deep_test))

        # 5) Train Enhanced on fused_train only (saves models)
        fused_train_csv = fold_dir / 'fused_train.csv'
        fused_test_csv  = fold_dir / 'fused_test.csv'
        fused_train.to_csv(fused_train_csv, index=False)
        fused_test.to_csv(fused_test_csv, index=False)
        enhanced_train_dir = fold_dir / 'enhanced_train'
        enhanced_train_dir.mkdir(exist_ok=True)
        train_enhanced_on_train_only(fused_train_csv, enhanced_train_dir, args.ml_threads, args.multi_class, args.random_state)

        # 6) Evaluate Enhanced ensemble on fused_test with saved artifacts
        # Build features only (drop id/label)
        X_test = fused_test.drop(columns=['subject_id', 'label'])
        y_test = fused_test['label'].to_numpy(dtype=int)
        prob = enhanced_predict_proba(enhanced_train_dir, X_test)
        metrics = score_fold(y_test, prob)
        with open(fold_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        fold_metrics.append(metrics)
        print(f"Fold {fold_idx}: acc={metrics['acc']:.4f} auc={metrics['auc_ovr']:.4f} ap={metrics['ap_macro']:.4f}")

    # Summary across folds
    def avg(key):
        vals = [m[key] for m in fold_metrics if np.isfinite(m[key])]
        return float(np.mean(vals)) if vals else float('nan')
    summary = {
        'folds': len(fold_metrics),
        'val_acc_mean': avg('acc'),
        'val_auc_ovr_mean': avg('auc_ovr'),
        'val_ap_macro_mean': avg('ap_macro'),
        'val_f1_macro_mean': avg('f1_macro'),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Fusion summary:', summary)


if __name__ == '__main__':
    main()


