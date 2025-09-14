#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch

PROJ_ROOT = Path(__file__).resolve().parents[2]


def _import_from(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def load_dataset(modality: str, csv_path: str, data_root: str):
    if modality.upper() == "PET":
        pet_ds_mod = _import_from(PROJ_ROOT / 'Scripts/Deep_Learning/PET/dataset.py')
        ds = pet_ds_mod.PETDataset(csv_path=csv_path, data_root=data_root)
    elif modality.upper() == "MRI":
        mri_ds_mod = _import_from(PROJ_ROOT / 'Scripts/Deep_Learning/MRI/dataset.py')
        ds = mri_ds_mod.SMRIDataset(csv_path=csv_path, data_root=data_root)
    else:
        raise ValueError("modality must be PET or MRI")
    return ds


def load_backbone(modality: str, backbone_name: str, num_classes: int, base_channels: int):
    if modality.upper() == "PET":
        pet_models_mod = _import_from(PROJ_ROOT / 'Scripts/Deep_Learning/PET/models_pet.py')
        model = pet_models_mod.get_3d_model(backbone_name, in_channels=1, num_classes=num_classes, base_channels=base_channels)
    else:
        mri_models_mod = _import_from(PROJ_ROOT / 'Scripts/Deep_Learning/MRI/models_smri.py')
        model = mri_models_mod.get_3d_model(backbone_name, in_channels=1, num_classes=num_classes, base_channels=base_channels)
    return model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Export deep embeddings/logits per subject")
    p.add_argument("--modality", required=True, choices=["PET", "MRI"])
    p.add_argument("--model", required=True, help="Backbone model name, e.g., Simple3DCNN or DenseNet121_3D")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True, help="Master CSV with subject_id,label for the split to export")
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--export", choices=["embeddings", "logits", "both"], default="embeddings")
    args = p.parse_args()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_dataset(args.modality, args.csv, args.data_root)
    num_classes = len(sorted(set(ds.labels)))

    # DataLoader (no shuffle)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    # Load model
    model = load_backbone(args.modality, args.model, num_classes=num_classes, base_channels=args.base_channels)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    # Filter to load only matching shapes to avoid classifier/head mismatches
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            skipped.append(k)
    if skipped:
        print(f"[INFO] Skipping {len(skipped)} keys due to shape mismatch or absence (e.g., {skipped[:3]}...)")
    model.load_state_dict(filtered, strict=False)
    model.eval().to(args.device)

    # Prepare optional hook to capture last conv feature map if model does not return it
    fmap_container = {}
    hook_handle = None
    if not hasattr(model, 'forward_returns_fmap'):
        try:
            target_module = getattr(model, 'features', None)
            if isinstance(target_module, torch.nn.Module):
                def _hook(_m, _i, o):
                    fmap_container['fmap'] = o.detach()
                hook_handle = target_module.register_forward_hook(_hook)
        except Exception:
            hook_handle = None

    # Forward and collect
    rows = []
    idx_offset = 0
    for images, labels in loader:
        images = images.to(args.device)
        out = model(images)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            logits, fmap = out
        else:
            logits = out
            fmap = fmap_container.get('fmap')
        batch_size = images.shape[0]

        # Embedding from fmap
        if fmap is not None and fmap.ndim == 5:
            embed = torch.nn.functional.adaptive_avg_pool3d(fmap, 1).flatten(1)
        else:
            # Fallback: use logits as embedding if fmap unavailable
            embed = logits

        for i in range(batch_size):
            sid = ds.subjects[idx_offset + i]
            row = {"subject_id": sid, "label": int(ds.labels[idx_offset + i])}
            if args.export in ("embeddings", "both"):
                vec = embed[i].detach().cpu().numpy()
                for j, v in enumerate(vec):
                    row[f"emb_{j}"] = float(v)
            if args.export in ("logits", "both"):
                logit_vec = logits[i].detach().cpu().numpy()
                for j, v in enumerate(logit_vec):
                    row[f"logit_{j}"] = float(v)
            rows.append(row)
        idx_offset += batch_size

    # Cleanup hook
    if hook_handle is not None:
        try:
            hook_handle.remove()
        except Exception:
            pass

    # Save CSV
    if rows:
        # Align columns
        cols = ["subject_id", "label"]
        if args.export in ("embeddings", "both"):
            emb_dim = len([k for k in rows[0].keys() if k.startswith("emb_")])
            cols += [f"emb_{j}" for j in range(emb_dim)]
        if args.export in ("logits", "both"):
            cols += [f"logit_{j}" for j in range(num_classes)]
        df = pd.DataFrame(rows)[cols]
        df.to_csv(args.out_csv, index=False)
        print(f"Saved {len(df)} rows to {args.out_csv}")
    else:
        print("No rows exported.")


if __name__ == "__main__":
    main()


