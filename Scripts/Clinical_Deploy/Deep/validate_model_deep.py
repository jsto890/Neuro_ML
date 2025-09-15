#!/usr/bin/env python3
"""
Deep Model Validation (Multiclass CN/AD/PD) with Optional 3D Interpretability
=============================================================================

- Evaluates a CSV containing NIfTI image paths and labels
- Computes multiclass metrics (accuracy, macro/weighted P/R/F1, OvR AUC)
- Saves confusion matrix and concise JSON + markdown reports
- Optionally saves per-subject 3D interpretability volumes (Grad-CAM, saliency, occlusion, GradientSHAP)

CSV requirements:
- Default columns: subject_id, image_path, label (configurable via CLI)
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

try:
    import nibabel as nib  # type: ignore
except Exception as e:
    print("Please install nibabel: pip install nibabel")
    raise


def expand_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def load_repo_modules() -> Tuple[object, object]:
    """Dynamically import gradcam and models_smri from the repo by path."""
    import importlib.util

    this_dir = Path(__file__).resolve().parent
    scripts_dir = this_dir.parent.parent  # .../Scripts
    repo_root = scripts_dir.parent        # .../P4P

    gradcam_path = repo_root / 'Scripts' / 'Deep_Learning' / 'MRI' / 'gradcam.py'
    models_path = repo_root / 'Scripts' / 'Deep_Learning' / 'MRI' / 'models_smri.py'

    # gradcam
    spec_g = importlib.util.spec_from_file_location('gradcam3d', str(gradcam_path))
    gradcam = importlib.util.module_from_spec(spec_g)
    assert spec_g and spec_g.loader
    spec_g.loader.exec_module(gradcam)  # type: ignore

    # models
    spec_m = importlib.util.spec_from_file_location('models_smri', str(models_path))
    models_smri = importlib.util.module_from_spec(spec_m)
    assert spec_m and spec_m.loader
    spec_m.loader.exec_module(models_smri)  # type: ignore

    return gradcam, models_smri


def load_nifti(image_path: str):
    img = nib.load(expand_path(image_path))
    data = img.get_fdata().astype(np.float32)
    return data, img.affine, img.header


def normalize_volume(volume: np.ndarray, method: str = 'zscore') -> np.ndarray:
    if method == 'zscore':
        mean = float(volume.mean())
        std = float(volume.std())
        if std < 1e-6:
            return np.zeros_like(volume)
        return (volume - mean) / std
    elif method == 'minmax':
        vmin = float(volume.min())
        vmax = float(volume.max())
        if vmax - vmin < 1e-6:
            return np.zeros_like(volume)
        return (volume - vmin) / (vmax - vmin)
    else:
        return volume


def to_model_tensor(volume: np.ndarray, device: str, resize_dims: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
    t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    if resize_dims is not None:
        t = F.interpolate(t, size=resize_dims, mode='trilinear', align_corners=False)
    return t.to(device)


def get_device(preferred: Optional[str] = None) -> str:
    if preferred:
        if preferred == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        if preferred == 'mps' and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_model(arch: str, num_classes: int, in_channels: int, weights_path: str, device: str):
    gradcam, models_smri = load_repo_modules()
    arch_l = arch.lower()

    if arch_l in ['smri_gradcam_3dcnn', 'smri-gradcam-3dcnn']:
        model = models_smri.SMRI_GradCAM_3DCNN(in_channels=in_channels, num_classes=num_classes)
    elif arch_l in ['simple3dcnn', 'simple_3dcnn']:
        model = models_smri.get_3d_model('simple3dcnn', num_classes=num_classes, in_channels=in_channels)
    elif arch_l in ['resnet18_3d', 'resnet18']:
        model = models_smri.get_3d_model('resnet18_3d', num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unsupported model architecture: {arch}")

    model.to(device)
    model.eval()

    if weights_path:
        state = torch.load(expand_path(weights_path), map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            sd = state['state_dict']
        else:
            sd = state
        clean_sd = {}
        for k, v in sd.items():
            nk = k.replace('module.', '') if isinstance(k, str) and k.startswith('module.') else k
            clean_sd[nk] = v
        model.load_state_dict(clean_sd, strict=False)

    return model, gradcam


def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        p = F.softmax(logits, dim=1)
    return p.squeeze(0).cpu().numpy()


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
    return logits


def predict_one(model, x: torch.Tensor) -> Tuple[int, np.ndarray]:
    logits = forward_logits(model, x)
    probs = softmax_probs(logits)
    pred = int(np.argmax(probs))
    return pred, probs


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < 1e-8:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)


def compute_gradcam_volume(gradcam_module, model, input_tensor: torch.Tensor, target_class: int, arch: str, device: str) -> np.ndarray:
    arch_l = arch.lower()
    if arch_l in ['smri_gradcam_3dcnn', 'smri-gradcam-3dcnn']:
        cam = gradcam_module.compute_gradcam_3d(model, input_tensor, target_class, device=device)
    elif arch_l in ['simple3dcnn', 'simple_3dcnn']:
        cam = gradcam_module.compute_gradcam_simple3d(model, input_tensor, target_class, device=device)
    else:
        raise ValueError(f"Grad-CAM not implemented for architecture: {arch}")
    return normalize_map(cam)


def compute_saliency_volume(model, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    x = input_tensor.clone().detach().requires_grad_(True)
    model.zero_grad()
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    score = F.softmax(logits, dim=1)[0, target_class]
    score.backward()
    grad = x.grad.detach().squeeze(0).squeeze(0).cpu().numpy()
    sal = np.abs(grad)
    return normalize_map(sal)


def compute_occlusion_volume(model, input_tensor: torch.Tensor, target_class: int, ksize: int = 16, stride: Optional[int] = None, baseline: float = 0.0) -> np.ndarray:
    stride = stride or ksize
    model.eval()
    with torch.no_grad():
        out0 = model(input_tensor)
        logits0 = out0[0] if isinstance(out0, tuple) else out0
        p0 = F.softmax(logits0, dim=1)[0, target_class].item()

    vol = input_tensor.clone().detach()
    _, _, D, H, W = vol.shape
    occ_map = np.zeros((D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    for z in range(0, D, stride):
        z2 = min(z + ksize, D)
        for y in range(0, H, stride):
            y2 = min(y + ksize, H)
            for x in range(0, W, stride):
                x2 = min(x + ksize, W)
                occluded = vol.clone()
                occluded[:, :, z:z2, y:y2, x:x2] = baseline
                with torch.no_grad():
                    out = model(occluded)
                    logits = out[0] if isinstance(out, tuple) else out
                    p = F.softmax(logits, dim=1)[0, target_class].item()
                drop = max(0.0, p0 - p)
                occ_map[z:z2, y:y2, x:x2] += drop
                count_map[z:z2, y:y2, x:x2] += 1.0

    mask = count_map > 0
    occ_map[mask] = occ_map[mask] / count_map[mask]
    return normalize_map(occ_map)


def compute_gradient_shap(model, input_tensor: torch.Tensor, target_class: int) -> Optional[np.ndarray]:
    try:
        from captum.attr import GradientShap  # type: ignore
    except Exception:
        return None

    model.eval()
    baseline = torch.zeros_like(input_tensor)
    gs = GradientShap(model)
    attributions = gs.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        n_samples=50,
        stdevs=0.001,
    )
    attr = attributions.detach().squeeze(0).squeeze(0).cpu().numpy()
    attr = np.abs(attr)
    return normalize_map(attr)


def save_nifti(volume: np.ndarray, affine: np.ndarray, header, out_path: Path):
    img = nib.Nifti1Image(volume.astype(np.float32), affine, header)
    nib.save(img, str(out_path))


def resolve_image_path(subject_id: str, image_root: str, search_pattern: str) -> Optional[str]:
    """Resolve an image path by searching under image_root with a glob pattern.

    search_pattern can reference {sid} or {subject_id} placeholders.
    Example: "**/*{sid}*.nii*" → matches recursively.
    """
    pattern = search_pattern.format(sid=subject_id, subject_id=subject_id)
    query = os.path.join(expand_path(image_root), pattern)
    matches = glob.glob(query, recursive=True)
    if not matches:
        return None
    # Prefer .nii.gz if multiple
    matches_sorted = sorted(matches, key=lambda p: (0 if p.endswith('.nii.gz') else 1, len(p)))
    return matches_sorted[0]


def main():
    parser = argparse.ArgumentParser(description='Validate deep model with multiclass metrics and optional interpretability')
    parser.add_argument('--data-csv', required=True, help='CSV with either [image_path,label] or [subject_id,label]')
    parser.add_argument('--image-column', default='image_path', help='If present in CSV, used directly. Otherwise image_root + search_pattern resolves.')
    parser.add_argument('--label-column', default='label')
    parser.add_argument('--subject-column', default='subject_id')
    parser.add_argument('--image-root', default=None, help='Root directory to recursively search for images when image_column is not provided')
    parser.add_argument('--search-pattern', default='**/*{sid}*.nii*', help='Glob pattern (recursive) to find images; placeholders: {sid} or {subject_id}')
    parser.add_argument('--fail-missing', action='store_true', help='Fail if any subject image cannot be resolved')
    parser.add_argument('--model-arch', default='SMRI_GradCAM_3DCNN')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--label-map-json', type=str, help='Optional JSON mapping of numeric labels to names')
    parser.add_argument('--normalize', choices=['zscore', 'minmax', 'none'], default='zscore')
    parser.add_argument('--resize-dims', type=int, nargs=3, metavar=('D', 'H', 'W'), help='Optional resize to D H W before inference')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default=None)
    parser.add_argument('--output-dir', default='~/reseng202500013-ndd-ml/clinical_outputs/deep_validation')

    # Interpretability options
    parser.add_argument('--save-interpretability', action='store_true', help='Save Grad-CAM, saliency, occlusion, GradientSHAP per subject')
    parser.add_argument('--interpret-all-classes', action='store_true', help='Grad-CAM for all classes instead of predicted only')
    parser.add_argument('--max-interpret', type=int, default=0, help='Max number of subjects to save interpretability for (0 = all)')
    parser.add_argument('--occ-ksize', type=int, default=16)
    parser.add_argument('--occ-stride', type=int, default=None)
    parser.add_argument('--occ-baseline', type=float, default=0.0)

    args = parser.parse_args()

    device = get_device(args.device)

    # Label mapping (default: 0->AD, 1->CN, 2->PD)
    label_map: Dict[int, str] = {0: 'AD', 1: 'CN', 2: 'PD'}
    if args.label_map_json:
        try:
            with open(expand_path(args.label_map_json), 'r') as f:
                raw = json.load(f)
            label_map = {int(k): str(v) for k, v in raw.items()}
        except Exception as e:
            print(f"Warning: failed to load label map JSON ({e}). Using default mapping.")

    # Load model and CSV
    model, gradcam_module = load_model(args.model_arch, args.num_classes, in_channels=1, weights_path=args.weights, device=device)
    df = pd.read_csv(expand_path(args.data_csv))

    # Determine mode: image paths present OR resolve via subject ids
    has_image_col = args.image_column in df.columns
    has_subject_col = args.subject_column in df.columns
    if not has_image_col:
        if not has_subject_col:
            print(f"CSV must contain either '{args.image_column}' or '{args.subject_column}'")
            sys.exit(1)
        if args.image_root is None:
            print("--image-root is required when image paths are not provided in the CSV")
            sys.exit(1)
    # Validate labels
    if args.label_column not in df.columns:
        print(f"Missing required column: {args.label_column}")
        sys.exit(1)

    out_dir = Path(expand_path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects: List[dict] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    proba_rows: List[np.ndarray] = []

    interpret_saved = 0
    limit_interpret = (args.max_interpret > 0)

    for idx, row in df.iterrows():
        label = int(row[args.label_column])
        subject_id = str(row[args.subject_column]) if has_subject_col else f"subj_{idx:04d}"
        if has_image_col:
            img_path = str(row[args.image_column])
        else:
            resolved = resolve_image_path(subject_id, args.image_root, args.search_pattern)
            if resolved is None:
                msg = f"Could not resolve image for subject {subject_id} under {args.image_root} with pattern {args.search_pattern}"
                if args.fail_missing:
                    raise FileNotFoundError(msg)
                else:
                    print(msg)
                    continue
            img_path = resolved

        try:
            vol_np, affine, header = load_nifti(img_path)
            vol_np = normalize_volume(vol_np, method=args.normalize)
            input_tensor = to_model_tensor(vol_np, device=device, resize_dims=tuple(args.resize_dims) if args.resize_dims else None)

            pred_idx, probs = predict_one(model, input_tensor)
            prob_dict = {label_map.get(i, str(i)): float(p) for i, p in enumerate(probs)}

            y_true.append(label)
            y_pred.append(pred_idx)
            proba_rows.append(probs)

            subject_entry = {
                'subject_id': subject_id,
                'image': expand_path(img_path),
                'true_label_index': label,
                'true_label_name': label_map.get(int(label), str(label)),
                'pred_label_index': pred_idx,
                'pred_label_name': label_map.get(int(pred_idx), str(pred_idx)),
                'confidence': float(np.max(probs)),
                'probabilities': prob_dict,
                'interpretability': {}
            }

            # Optional interpretability
            if args.save_interpretability and (not limit_interpret or interpret_saved < args.max_interpret):
                sid = Path(img_path).stem.replace('.nii', '').replace('.gz', '')
                case_dir = out_dir / f"{sid}_maps"
                case_dir.mkdir(parents=True, exist_ok=True)

                # Grad-CAM
                classes_to_compute = range(args.num_classes) if args.interpret_all_classes else [pred_idx]
                gradcam_paths = {}
                for c in classes_to_compute:
                    try:
                        cam = compute_gradcam_volume(gradcam_module, model, input_tensor, c, args.model_arch, device)
                        cam_path = case_dir / f"{sid}_gradcam_class{c}.nii.gz"
                        save_nifti(cam, affine, header, cam_path)
                        gradcam_paths[str(c)] = str(cam_path)
                    except Exception as e:
                        print(f"Grad-CAM failed for {subject_id} class {c}: {e}")

                # Saliency
                try:
                    sal = compute_saliency_volume(model, input_tensor, pred_idx)
                    sal_path = case_dir / f"{sid}_saliency.nii.gz"
                    save_nifti(sal, affine, header, sal_path)
                except Exception as e:
                    print(f"Saliency failed for {subject_id}: {e}")
                    sal_path = None

                # Occlusion
                try:
                    occ = compute_occlusion_volume(
                        model, input_tensor, pred_idx,
                        ksize=int(args.occ_ksize), stride=args.occ_stride, baseline=float(args.occ_baseline)
                    )
                    occ_path = case_dir / f"{sid}_occlusion.nii.gz"
                    save_nifti(occ, affine, header, occ_path)
                except Exception as e:
                    print(f"Occlusion failed for {subject_id}: {e}")
                    occ_path = None

                # GradientSHAP
                try:
                    gshap = compute_gradient_shap(model, input_tensor, pred_idx)
                    if gshap is not None:
                        gshap_path = case_dir / f"{sid}_gradientshap.nii.gz"
                        save_nifti(gshap, affine, header, gshap_path)
                    else:
                        gshap_path = None
                except Exception as e:
                    print(f"GradientSHAP failed for {subject_id}: {e}")
                    gshap_path = None

                subject_entry['interpretability'] = {
                    'gradcam': gradcam_paths,
                    'saliency': str(sal_path) if sal_path else None,
                    'occlusion': str(occ_path) if occ_path else None,
                    'gradientshap': str(gshap_path) if gshap_path else None,
                }
                interpret_saved += 1

            subjects.append(subject_entry)

        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            continue

    # Aggregate metrics
    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    proba_mat = np.vstack(proba_rows) if len(proba_rows) > 0 else None

    metrics = {
        'n_subjects': int(len(y_true_np)),
        'accuracy': float(accuracy_score(y_true_np, y_pred_np)) if len(y_true_np) > 0 else 0.0,
    }
    try:
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average='weighted', zero_division=0)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average='macro', zero_division=0)
        metrics.update({
            'precision_weighted': float(prec_w),
            'recall_weighted': float(rec_w),
            'f1_weighted': float(f1_w),
            'precision_macro': float(prec_m),
            'recall_macro': float(rec_m),
            'f1_macro': float(f1_m),
        })
    except Exception:
        pass

    # AUC
    try:
        uniq = np.unique(y_true_np)
        if proba_mat is not None and len(uniq) >= 2:
            if len(uniq) == 2:
                uniq_sorted = sorted(uniq)
                y_bin = (y_true_np == uniq_sorted[-1]).astype(int) if set(uniq_sorted) != {0, 1} else y_true_np
                auc_val = roc_auc_score(y_bin, proba_mat[:, 1] if proba_mat.shape[1] >= 2 else proba_mat[:, 0])
            else:
                auc_val = roc_auc_score(y_true_np, proba_mat, multi_class='ovr', average='weighted')
            metrics['auc'] = float(auc_val)
        else:
            metrics['auc'] = 0.0
    except Exception:
        metrics['auc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np)
    unique_labels_sorted = sorted(list(set(y_true_np) | set(y_pred_np)))
    disease_labels = [label_map.get(int(l), str(l)) for l in unique_labels_sorted]

    # Reports
    summary = {
        'model': {
            'architecture': args.model_arch,
            'weights': expand_path(args.weights),
            'num_classes': int(args.num_classes),
        },
        'data': {
            'csv': expand_path(args.data_csv),
            'image_column': args.image_column,
            'label_column': args.label_column,
            'subject_column': args.subject_column,
        },
        'label_map': label_map,
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'labels': disease_labels,
        'subjects': subjects,
    }

    out_dir = Path(expand_path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'deep_validation_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Validation JSON saved to: {json_path}")

    md = f"""
# Deep Model Validation (Multiclass)

Model: {args.model_arch}
Weights: {expand_path(args.weights)}
Dataset: {expand_path(args.data_csv)}

- N: {metrics.get('n_subjects', 0)}
- Accuracy: {metrics.get('accuracy', 0.0):.3f}
- AUC (OvR if multiclass): {metrics.get('auc', 0.0):.3f}
- F1 (weighted/macro): {metrics.get('f1_weighted', 0.0):.3f} / {metrics.get('f1_macro', 0.0):.3f}

Labels: {disease_labels}
Confusion Matrix (rows=actual, cols=pred):
{cm}
"""
    md_path = out_dir / 'deep_validation_report.md'
    with open(md_path, 'w') as f:
        f.write(md)
    print(f"✓ Validation report saved to: {md_path}")


if __name__ == '__main__':
    main()


