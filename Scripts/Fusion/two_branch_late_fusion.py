#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

import importlib.util

PROJ_ROOT = Path(__file__).resolve().parents[2]

def _import_from(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class RadiomicsTable:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if 'subject_id' not in df.columns:
            raise ValueError('radiomics_csv must include subject_id column')
        self.df = df.set_index('subject_id')
        self.feature_cols = [c for c in df.columns if c not in ('subject_id', 'label')]

    def get(self, sid: str) -> np.ndarray:
        row = self.df.loc[sid]
        return row[self.feature_cols].to_numpy(dtype=np.float32)

    def dim(self) -> int:
        return len(self.feature_cols)


class FusionDataset(Dataset):
    def __init__(self, modality: str, csv_path: str, data_root: str, radiomics: RadiomicsTable):
        self.modality = modality.upper()
        self.radiomics = radiomics
        if self.modality == 'PET':
            pet_ds_mod = _import_from(PROJ_ROOT / 'Scripts/Deep_Learning/PET/dataset.py')
            self.base = pet_ds_mod.PETDataset(csv_path=csv_path, data_root=data_root)
        else:
            mri_ds_mod = _import_from(PROJ_ROOT / 'Scripts/Deep_Learning/MRI/dataset.py')
            self.base = mri_ds_mod.SMRIDataset(csv_path=csv_path, data_root=data_root)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        sid = self.base.subjects[idx]
        rad = torch.from_numpy(self.radiomics.get(sid))
        return img.float(), rad.float(), label


class FusionNet(nn.Module):
    def __init__(self, cnn: nn.Module, cnn_embed_dim: int, rad_dim: int, num_classes: int, rad_hidden: int = 256, fuse_hidden: int = 256):
        super().__init__()
        self.cnn = cnn
        self.rad_mlp = nn.Sequential(
            nn.Linear(rad_dim, rad_hidden),
            nn.BatchNorm1d(rad_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(rad_hidden, rad_hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(cnn_embed_dim + rad_hidden, fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fuse_hidden, num_classes),
        )

    def forward(self, x_img: torch.Tensor, x_rad: torch.Tensor):
        logits, fmap = self.cnn(x_img)
        cnn_embed = F.adaptive_avg_pool3d(fmap, 1).flatten(1)
        rad_embed = self.rad_mlp(x_rad)
        fused = torch.cat([cnn_embed, rad_embed], dim=1)
        out = self.head(fused)
        return out


def get_backbone(modality: str, name: str, num_classes: int) -> Tuple[nn.Module, int]:
    if modality.upper() == 'PET':
        pet_models_mod = _import_from(PROJ_ROOT / 'Scripts/Deep_Learning/PET/models_pet.py')
        model = pet_models_mod.get_3d_model(name, in_channels=1, num_classes=num_classes)
    else:
        mri_models_mod = _import_from(PROJ_ROOT / 'Scripts/Deep_Learning/MRI/models_smri.py')
        model = mri_models_mod.get_3d_model(name, in_channels=1, num_classes=num_classes)
    # Infer embed dim from last conv channels if provided on model
    cnn_embed_dim = getattr(model, 'embed_dim', None)
    if cnn_embed_dim is None:
        # heuristic: try to forward one dummy to get fmap shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 96, 112, 96)
            logits, fmap = model(dummy)
            cnn_embed_dim = fmap.shape[1]
    return model, int(cnn_embed_dim)


def compute_scores(y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    acc = accuracy_score(y_true, pred)
    try:
        auc = roc_auc_score(y_true, prob, multi_class='ovr')
    except Exception:
        auc = float('nan')
    ap_macro = np.mean([average_precision_score((y_true == c).astype(int), prob[:, c]) for c in np.unique(y_true)])
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, pred).tolist()
    return {
        'acc': float(acc), 'auc_ovr': float(auc), 'ap_macro': float(ap_macro),
        'precision_macro': float(prec_m), 'recall_macro': float(rec_m), 'f1_macro': float(f1_m),
        'confusion_matrix': cm,
    }


def main():
    ap = argparse.ArgumentParser(description='End-to-end late fusion (CNN image + MLP radiomics)')
    ap.add_argument('--modality', required=True, choices=['PET', 'MRI'])
    ap.add_argument('--backbone', required=True, help='e.g., Simple3DCNN or DenseNet121_3D')
    ap.add_argument('--master_csv', required=True)
    ap.add_argument('--radiomics_csv', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--k_folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--learning_rate', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--label_smoothing', type=float, default=0.05)
    ap.add_argument('--random_seed', type=int, default=42)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master_csv)
    if 'subject_id' not in master.columns or 'label' not in master.columns:
        master = pd.read_csv(args.master_csv, header=None, names=['subject_id', 'label'])
    y = master['label'].to_numpy(dtype=int)

    radiomics = RadiomicsTable(args.radiomics_csv)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
    fold_summaries = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(master['subject_id'], y), start=1):
        train_csv = Path(args.out_dir) / f'temp_train_fold_{fold}.csv'
        val_csv   = Path(args.out_dir) / f'temp_val_fold_{fold}.csv'
        master.iloc[tr_idx][['subject_id', 'label']].to_csv(train_csv, index=False)
        master.iloc[va_idx][['subject_id', 'label']].to_csv(val_csv, index=False)

        train_ds = FusionDataset(args.modality, str(train_csv), args.data_root, radiomics)
        val_ds   = FusionDataset(args.modality, str(val_csv), args.data_root, radiomics)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        num_classes = len(np.unique(y))
        backbone, cnn_embed_dim = get_backbone(args.modality, args.backbone, num_classes)
        model = FusionNet(backbone, cnn_embed_dim, radiomics.dim(), num_classes).to(args.device)

        # Class weights
        classes, counts = np.unique(y[tr_idx], return_counts=True)
        total = counts.sum()
        class_weight = torch.tensor([total / (len(classes) * n) for n in counts], dtype=torch.float32, device=args.device)

        criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best = { 'auc': -1.0 }
        for epoch in range(1, args.epochs + 1):
            model.train()
            for xb, xr, yb in train_loader:
                xb = xb.to(args.device)
                xr = xr.to(args.device)
                yb = yb.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb, xr)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            all_prob, all_true = [], []
            with torch.no_grad():
                for xb, xr, yb in val_loader:
                    logits = model(xb.to(args.device), xr.to(args.device))
                    prob = torch.softmax(logits, dim=1).cpu().numpy()
                    all_prob.append(prob)
                    all_true.append(yb.numpy())
            prob = np.concatenate(all_prob, axis=0)
            yv = np.concatenate(all_true, axis=0)
            scores = compute_scores(yv, prob)
            if scores['auc_ovr'] > best['auc']:
                best = scores.copy()
                # save checkpoint
                fold_dir = Path(args.out_dir) / f'fold_{fold}'
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), fold_dir / 'fusion_best.pth')
                with open(fold_dir / 'val_metrics.json', 'w') as f:
                    json.dump(best, f, indent=2)
            print(f"Fold {fold} Epoch {epoch}: acc={scores['acc']:.4f} auc={scores['auc_ovr']:.4f} ap={scores['ap_macro']:.4f}")

        fold_summaries.append(best)

    # Summary
    def avg(key):
        vals = [m[key] for m in fold_summaries if np.isfinite(m[key])]
        return float(np.mean(vals)) if vals else float('nan')
    summary = {
        'folds': len(fold_summaries),
        'val_acc_mean': avg('acc'),
        'val_auc_ovr_mean': avg('auc_ovr'),
        'val_ap_macro_mean': avg('ap_macro'),
        'val_f1_macro_mean': avg('f1_macro'),
    }
    with open(Path(args.out_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Late fusion summary:', summary)


if __name__ == '__main__':
    main()


