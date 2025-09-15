#!/usr/bin/env python3
"""
Deep Clinical Prediction (Multiclass CN/AD/PD) with 3D Interpretability Volumes
===============================================================================

- Accepts a single-subject 3D NIfTI image
- Loads a specified 3D model architecture with weights
- Produces:
  - Prediction (CN/AD/PD), confidence, probability distribution (JSON)
  - 3D Grad-CAM NIfTI for predicted class
  - 3D Saliency map (|d score / d input|)
  - 3D Occlusion sensitivity map
  - 3D GradientSHAP attribution map (if captum is available)

Outputs default to a directory OUTSIDE the repo per user preference.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def load_nifti(image_path: str) -> Tuple[np.ndarray, np.ndarray, object]:
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


def load_model(arch: str, num_classes: int, in_channels: int, weights_path: str, device: str):
    gradcam, models_smri = load_repo_modules()
    arch_l = arch.lower()

    # Helper to load and clean a checkpoint
    def _load_clean_sd(path: str):
        state = torch.load(expand_path(path), map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            sd = state['state_dict']
        else:
            sd = state
        clean_sd = {}
        for k, v in sd.items():
            nk = k.replace('module.', '') if isinstance(k, str) and k.startswith('module.') else k
            clean_sd[nk] = v
        return clean_sd

    # Supported architectures
    if arch_l in ['smri_gradcam_3dcnn', 'smri-gradcam-3dcnn']:
        model = models_smri.SMRI_GradCAM_3DCNN(in_channels=in_channels, num_classes=num_classes)
        model.to(device)
        model.eval()
        if weights_path:
            clean_sd = _load_clean_sd(weights_path)
            model.load_state_dict(clean_sd, strict=False)
        return model, gradcam

    elif arch_l in ['simple3dcnn', 'simple_3dcnn']:
        # Auto-detect base_channels and classifier input from checkpoint if available
        base_channels = 16
        classifier_in = None
        clean_sd = None
        if weights_path:
            clean_sd = _load_clean_sd(weights_path)
            try:
                if 'features.0.weight' in clean_sd and hasattr(clean_sd['features.0.weight'], 'shape'):
                    base_channels = int(clean_sd['features.0.weight'].shape[0])
            except Exception:
                pass
            try:
                if 'classifier.0.weight' in clean_sd and hasattr(clean_sd['classifier.0.weight'], 'shape'):
                    classifier_in = int(clean_sd['classifier.0.weight'].shape[1])
            except Exception:
                pass

        base_model = models_smri.get_3d_model('simple3dcnn', num_classes=num_classes, in_channels=in_channels, base_channels=base_channels)
        base_model.to(device)
        base_model.eval()

        # If classifier input size is known from checkpoint, set it before loading
        if classifier_in is not None:
            base_model.classifier[0] = nn.Linear(classifier_in, 256).to(device)
            base_model._initialized = True  # skip lazy init

        if clean_sd is not None:
            base_model.load_state_dict(clean_sd, strict=False)

        # Wrap to expose feature maps for Grad-CAM
        class GradCAMWrapper(nn.Module):
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model
                self.features = model.features
                self.classifier = model.classifier
            def forward(self, x: torch.Tensor):
                feats = self.features(x)
                x_flat = feats.view(feats.size(0), -1)
                logits = self.classifier(x_flat)
                return logits, feats

        wrapped = GradCAMWrapper(base_model).to(device)
        wrapped.eval()
        return wrapped, gradcam

    elif arch_l in ['resnet18_3d', 'resnet18']:
        model = models_smri.get_3d_model('resnet18_3d', num_classes=num_classes, in_channels=in_channels)
        model.to(device)
        model.eval()
        if weights_path:
            clean_sd = _load_clean_sd(weights_path)
            model.load_state_dict(clean_sd, strict=False)
        return model, gradcam

    else:
        raise ValueError(f"Unsupported model architecture: {arch}")


def get_device(preferred: Optional[str] = None) -> str:
    if preferred:
        if preferred == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        if preferred == 'mps' and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    # Auto
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        p = F.softmax(logits, dim=1)
    return p.squeeze(0).cpu().numpy()


def predict(model, input_tensor: torch.Tensor) -> Tuple[int, np.ndarray, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)
        if isinstance(out, tuple) and len(out) >= 1:
            logits = out[0]
        else:
            logits = out
    probs = softmax_probs(logits)
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs, logits


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
        # Requires a wrapper returning (logits, fmap); not guaranteed. Best-effort fallback to generic.
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

    # Average where counted
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
    # 50 samples, small noise
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


def main():
    parser = argparse.ArgumentParser(description='Deep clinical prediction with 3D interpretability')
    parser.add_argument('--image', required=True, help='Path to input NIfTI (.nii or .nii.gz)')
    parser.add_argument('--model-arch', default='SMRI_GradCAM_3DCNN', help='Model architecture (SMRI_GradCAM_3DCNN, Simple3DCNN, ResNet18_3D)')
    parser.add_argument('--weights', required=True, help='Path to model weights (.pt/.pth) state_dict or checkpoint')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--label-map-json', type=str, help='Optional JSON mapping of numeric labels to names')
    parser.add_argument('--normalize', choices=['zscore', 'minmax', 'none'], default='zscore')
    parser.add_argument('--resize-dims', type=int, nargs=3, metavar=('D', 'H', 'W'), help='Optional resize to D H W before inference')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default=None)
    parser.add_argument('--output-dir', default='~/reseng202500013-ndd-ml/clinical_outputs/deep', help='Directory to write outputs (JSON + NIfTI maps)')
    parser.add_argument('--occ-ksize', type=int, default=16)
    parser.add_argument('--occ-stride', type=int, default=None)
    parser.add_argument('--occ-baseline', type=float, default=0.0)
    parser.add_argument('--all-classes-interpret', action='store_true', help='Produce Grad-CAM for all classes (default: predicted only)')

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

    # Load model + gradcam util
    model, gradcam_module = load_model(args.model_arch, args.num_classes, in_channels=1, weights_path=args.weights, device=device)

    # Load and preprocess image
    vol_np, affine, header = load_nifti(args.image)
    vol_np = normalize_volume(vol_np, method=args.normalize)
    input_tensor = to_model_tensor(vol_np, device=device, resize_dims=tuple(args.resize_dims) if args.resize_dims else None)

    # Predict
    pred_idx, probs, logits = predict(model, input_tensor)
    pred_name = label_map.get(pred_idx, str(pred_idx))
    confidence = float(np.max(probs))
    prob_dict = {label_map.get(i, str(i)): float(p) for i, p in enumerate(probs)}

    # Prepare output directory
    out_dir = Path(expand_path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    sid = Path(args.image).stem.replace('.nii', '').replace('.gz', '')

    # Interpretability maps
    gradcam_paths = {}
    classes_to_compute = range(args.num_classes) if args.all_classes_interpret else [pred_idx]
    for c in classes_to_compute:
        try:
            cam = compute_gradcam_volume(gradcam_module, model, input_tensor, c, args.model_arch, device)
            cam_path = out_dir / f"{sid}_gradcam_class{c}.nii.gz"
            save_nifti(cam, affine, header, cam_path)
            gradcam_paths[str(c)] = str(cam_path)
        except Exception as e:
            print(f"Grad-CAM failed for class {c}: {e}")

    # Saliency (absolute gradient)
    try:
        sal = compute_saliency_volume(model, input_tensor, pred_idx)
        sal_path = out_dir / f"{sid}_saliency.nii.gz"
        save_nifti(sal, affine, header, sal_path)
    except Exception as e:
        print(f"Saliency failed: {e}")
        sal_path = None

    # Occlusion sensitivity
    try:
        occ = compute_occlusion_volume(
            model, input_tensor, pred_idx, ksize=int(args.occ_ksize), stride=args.occ_stride, baseline=float(args.occ_baseline)
        )
        occ_path = out_dir / f"{sid}_occlusion.nii.gz"
        save_nifti(occ, affine, header, occ_path)
    except Exception as e:
        print(f"Occlusion failed: {e}")
        occ_path = None

    # GradientSHAP (if captum available)
    try:
        gshap = compute_gradient_shap(model, input_tensor, pred_idx)
        if gshap is not None:
            gshap_path = out_dir / f"{sid}_gradientshap.nii.gz"
            save_nifti(gshap, affine, header, gshap_path)
        else:
            gshap_path = None
    except Exception as e:
        print(f"GradientSHAP failed: {e}")
        gshap_path = None

    # JSON report
    report = {
        'modality': 'MRI',
        'model_type': 'deep',
        'architecture': args.model_arch,
        'weights': expand_path(args.weights),
        'image': expand_path(args.image),
        'prediction': {
            'label_index': pred_idx,
            'label_name': pred_name,
            'confidence': confidence,
            'probabilities': prob_dict,
        },
        'interpretability': {
            'gradcam': gradcam_paths,
            'saliency': str(sal_path) if sal_path else None,
            'occlusion': str(occ_path) if occ_path else None,
            'gradientshap': str(gshap_path) if gshap_path else None,
        },
    }

    json_path = out_dir / f"{sid}_clinical_prediction_deep.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Prediction: {pred_name} (conf {confidence:.3f})")
    print(f"✓ JSON report: {json_path}")


if __name__ == '__main__':
    main()


