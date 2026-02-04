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
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob as glob_mod

try:
    import nibabel as nib  # type: ignore
except Exception as e:
    print("Please install nibabel: pip install nibabel")
    raise

# Optional SciPy (for overlay-time mask erosion). If unavailable, we skip erosion.
try:
    from scipy.ndimage import distance_transform_edt  # type: ignore
except Exception:
    distance_transform_edt = None  # type: ignore


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

        # IMPORTANT:
        # Simple3DCNN initialises classifier[0] as Linear(1,256) and normally fixes it during forward()
        # via _initialize_classifier(x). However, our Grad-CAM wrapper bypasses base_model.forward(),
        # so we must align classifier[0] to the checkpoint *before* load_state_dict, otherwise PyTorch
        # will error on size mismatch even with strict=False.
        if classifier_in is not None:
            try:
                out_feats = int(getattr(base_model.classifier[0], "out_features", 256))
            except Exception:
                out_feats = 256
            try:
                base_model.classifier[0] = nn.Linear(int(classifier_in), int(out_feats)).to(device)
                # Prevent base_model.forward() from re-initialising and overwriting the loaded classifier
                if hasattr(base_model, "_initialized"):
                    base_model._initialized = True  # type: ignore
            except Exception:
                pass

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
    """
    Resolve device string. Accepts 'cpu', 'mps', 'cuda', or 'cuda:N'.
    Falls back to CPU if unavailable.
    """
    if preferred:
        pref = str(preferred).lower()
        if pref.startswith('cuda'):
            if not torch.cuda.is_available():
                return 'cpu'
            # Allow 'cuda' or 'cuda:N'
            return pref
        if pref == 'mps':
            return 'mps' if torch.backends.mps.is_available() else 'cpu'
        if pref == 'cpu':
            return 'cpu'
        # Unknown string → best effort
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


def set_dropout_enabled_only(module: nn.Module, enable: bool) -> None:
    """
    Enable training mode for Dropout layers only to approximate MC Dropout at inference,
    leaving other modules (e.g., BatchNorm) in eval mode.
    """
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train(enable)
        else:
            # Keep non-dropout submodules in eval
            try:
                m.eval()
            except Exception:
                pass


def aggregate_predictions(models: List[nn.Module], input_tensor: torch.Tensor, avg_method: str,
                         tta_n: int = 0, tta_noise_std: float = 0.0, mc_dropout: bool = False,
                         use_amp: bool = False, device_str: str = 'cpu',
                         vote_weight: Optional[float] = None, vote_majority_thresh: Optional[float] = None
                         ) -> Tuple[np.ndarray, torch.Tensor, Dict[str, object]]:
    """
    Run ensemble with optional TTA and return probabilities and aggregated logits.
    Returns (probs_np, logits_tensor, meta)
    meta contains per-replicate votes and counts for diagnostics.
    """
    if tta_n is None or tta_n < 0:
        tta_n = 0
    replicates = max(1, int(tta_n))

    # Prepare models
    for m in models:
        m.eval()
        if mc_dropout:
            set_dropout_enabled_only(m, True)

    all_logits: List[torch.Tensor] = []
    votes: List[int] = []
    use_cuda_amp = use_amp and isinstance(device_str, str) and device_str.startswith('cuda') and torch.cuda.is_available()
    with torch.no_grad():
        for m in models:
            for r in range(replicates):
                xt = input_tensor
                if tta_noise_std and tta_noise_std > 0.0 and tta_n > 0:
                    xt = xt + torch.randn_like(xt) * float(tta_noise_std)
                if use_cuda_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = m(xt)
                else:
                    out = m(xt)
                lg = out[0] if isinstance(out, tuple) else out
                all_logits.append(lg)
                v = int(torch.argmax(F.softmax(lg, dim=1), dim=1).item())
                votes.append(v)

    vote_hist = {int(i): int(votes.count(i)) for i in set(votes)}
    total_votes = max(1, len(votes))
    vote_frac = {k: v / float(total_votes) for k, v in vote_hist.items()}
    meta = {
        'tta_replicates': replicates,
        'votes': votes,
        'vote_hist': vote_hist,
        'vote_frac': vote_frac,
    }

    if avg_method == 'probs':
        probs_stack = torch.stack([F.softmax(lg, dim=1) for lg in all_logits], dim=0)
        probs_t = torch.mean(probs_stack, dim=0)  # [1, C]
        logits = torch.log(probs_t + 1e-12)
        probs = probs_t.squeeze(0).cpu().numpy()
    else:
        logits_t = torch.mean(torch.stack(all_logits, dim=0), dim=0)
        logits = logits_t
        probs = softmax_probs(logits_t)

    # Blend probabilities with vote fractions to form a consensus probability vector
    try:
        p = np.asarray(probs, dtype=np.float32)
        vf = np.zeros_like(p)
        for i in range(p.shape[0]):
            vf[i] = float(vote_frac.get(i, 0.0))
        # Majority snap if requested
        if vote_majority_thresh is not None and len(vf) > 0:
            top_v = float(np.max(vf))
            top_i = int(np.argmax(vf))
            if top_v >= float(vote_majority_thresh):
                p = np.zeros_like(p); p[top_i] = 1.0
                probs = p.astype(np.float32)
                return probs, logits, meta
        # Confidence-adaptive or user-forced blend
        if vote_weight is None:
            ent = -np.sum(np.clip(p, 1e-9, 1.0) * np.log(np.clip(p, 1e-9, 1.0)))
            max_ent = np.log(max(1, p.shape[0]))
            flatness = float(ent / max(max_ent, 1e-8))  # 0=peaked, 1=flat
            alpha = 0.5 * flatness + 0.2  # in [0.2, 0.7]
        else:
            alpha = float(min(1.0, max(0.0, vote_weight)))
        consensus = (1.0 - alpha) * p + alpha * vf
        s = float(consensus.sum())
        if s > 0:
            consensus = consensus / s
        probs = consensus.astype(np.float32)
    except Exception:
        pass

    return probs, logits, meta


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < 1e-8:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)


def robust_normalize_map(arr: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """
    Normalize using robust percentiles to enhance contrast when most values are near-constant.
    """
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, low_percentile)
    hi = np.percentile(arr, high_percentile)
    if hi - lo < 1e-8:
        return normalize_map(arr)
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


def robust_normalize_within_mask(arr: np.ndarray, mask: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """
    Normalize using percentiles computed ONLY over voxels where mask > 0, but
    apply the resulting scaling to the WHOLE array. This preserves out-of-mask
    highlights while ensuring the dynamic range is determined by in-brain voxels.
    """
    arr = arr.astype(np.float32)
    mask = (mask > 0).astype(bool)
    if not np.any(mask):
        return robust_normalize_map(arr, low_percentile, high_percentile)
    vals = arr[mask]
    lo = np.percentile(vals, low_percentile)
    hi = np.percentile(vals, high_percentile)
    if hi - lo < 1e-8:
        return normalize_map(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def compute_gradcam_volume(gradcam_module, model, input_tensor: torch.Tensor, target_class: int, arch: str, device: str, cam_layer: str = 'prepool') -> np.ndarray:
    arch_l = arch.lower()
    if arch_l in ['smri_gradcam_3dcnn', 'smri-gradcam-3dcnn']:
        cam = gradcam_module.compute_gradcam_3d(model, input_tensor, target_class, device=device)
        return robust_normalize_map(cam)
    elif arch_l in ['simple3dcnn', 'simple_3dcnn']:
        # Use vanilla Grad-CAM by default
        try:
            cam = compute_gradcam_simple3d_local(model, input_tensor, target_class, device, which_layer=cam_layer)
            return robust_normalize_map(cam)
        except Exception:
            # Fallback to module implementation
            cam = gradcam_module.compute_gradcam_simple3d(model, input_tensor, target_class, device=device)
            return robust_normalize_map(cam)
    else:
        raise ValueError(f"Grad-CAM not implemented for architecture: {arch}")
    # Unreachable
    # return normalize_map(cam)


def compute_gradcam_plusplus_volume(gradcam_module, model, input_tensor: torch.Tensor, target_class: int, arch: str, device: str, cam_layer: str = 'prepool') -> np.ndarray:
    """Grad-CAM++ version of compute_gradcam_volume"""
    arch_l = arch.lower()
    if arch_l in ['smri_gradcam_3dcnn', 'smri-gradcam-3dcnn']:
        # For SMRI_GradCAM_3DCNN, fall back to vanilla Grad-CAM
        cam = gradcam_module.compute_gradcam_3d(model, input_tensor, target_class, device=device)
        return robust_normalize_map(cam)
    elif arch_l in ['simple3dcnn', 'simple_3dcnn']:
        # Use Grad-CAM++ local implementation
        try:
            cam = compute_gradcam_plusplus_simple3d_local(model, input_tensor, target_class, device, which_layer=cam_layer)
            return robust_normalize_map(cam)
        except Exception:
            # Fallback to vanilla Grad-CAM if Grad-CAM++ fails
            try:
                cam = compute_gradcam_simple3d_local(model, input_tensor, target_class, device, which_layer=cam_layer)
                return robust_normalize_map(cam)
            except Exception:
                # Final fallback to module implementation
                cam = gradcam_module.compute_gradcam_simple3d(model, input_tensor, target_class, device=device)
                return robust_normalize_map(cam)
    else:
        raise ValueError(f"Grad-CAM++ not implemented for architecture: {arch}")


def _select_conv_for_cam_simple3d(model: nn.Module, which: str) -> nn.Module:
    # features = [Conv3d, BN, ReLU, (Dropout), MaxPool, Conv3d, BN, ReLU, (Dropout), MaxPool, Conv3d, BN, ReLU, (Dropout), MaxPool]
    convs = [m for m in model.features.modules() if isinstance(m, nn.Conv3d)]
    if not convs:
        raise RuntimeError("No Conv3d layers found in features")
    if which == 'last':
        return convs[-1]
    if which == 'mid':
        return convs[len(convs)//2]
    # 'prepool' → pick the last conv before final MaxPool (third block conv)
    return convs[-1]


def compute_gradcam_simple3d_local(model: nn.Module, smri_tensor: torch.Tensor, target_class: int, device: str = "cpu", which_layer: str = 'prepool') -> np.ndarray:
    """
    Grad-CAM for Simple3DCNN wrapper using raw logits and hooks on the last Conv3d.
    More robust when softmax-based CAM yields near-constant maps.
    Expects model(x) -> (logits, features).
    """
    model.to(device)
    model.eval()

    last_conv = _select_conv_for_cam_simple3d(model, which_layer)

    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None

    def fwd_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def bwd_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(bwd_hook)

    smri_tensor = smri_tensor.to(device).requires_grad_(True)
    logits, _ = model(smri_tensor)
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=False)

    if activations is None or gradients is None:
        h1.remove(); h2.remove()
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

    weights = torch.mean(gradients, dim=(2, 3, 4)).squeeze(0)  # [C]
    cam = torch.zeros_like(activations[0, 0])
    for i, w in enumerate(weights):
        cam += w * activations[0, i]
    cam = F.relu(cam)
    cam = cam.unsqueeze(0).unsqueeze(0)
    target_size = smri_tensor.shape[-3:]
    cam_upsampled = F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)
    cam_np = cam_upsampled.squeeze().cpu().numpy()

    h1.remove(); h2.remove()
    return cam_np.astype(np.float32)


def compute_gradcam_plusplus_simple3d_local(model: nn.Module, smri_tensor: torch.Tensor, target_class: int, device: str = "cpu", which_layer: str = 'prepool') -> np.ndarray:
    """
    Grad-CAM++ for Simple3DCNN wrapper using the improved weighting scheme.
    Implements the Grad-CAM++ paper: https://arxiv.org/abs/1710.11063
    """
    model.to(device)
    model.eval()

    last_conv = _select_conv_for_cam_simple3d(model, which_layer)

    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None
    grad_squared: Optional[torch.Tensor] = None
    grad_cubed: Optional[torch.Tensor] = None

    def fwd_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def bwd_hook(module, grad_input, grad_output):
        nonlocal gradients, grad_squared, grad_cubed
        grad = grad_output[0].detach()
        gradients = grad
        grad_squared = grad ** 2
        grad_cubed = grad ** 3

    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(bwd_hook)

    smri_tensor = smri_tensor.to(device).requires_grad_(True)
    logits, _ = model(smri_tensor)
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=False)

    if activations is None or gradients is None:
        h1.remove(); h2.remove()
        raise RuntimeError("Grad-CAM++ hooks did not capture activations/gradients")

    # Grad-CAM++ weighting scheme
    # w_k^c = ReLU(Σ_i Σ_j α_ij^k^c * A_ij^k^c) where α_ij^k^c = (∂²y^c/∂A_ij^k²) / (2 * ∂²y^c/∂A_ij^k² + Σ_a Σ_b A_ab^k * ∂³y^c/∂A_ab^k³)
    
    # For numerical stability, we use a simplified version:
    # w_k = Σ_i Σ_j (grad_ij^k * A_ij^k) / (Σ_i Σ_j A_ij^k + ε)
    # This approximates the higher-order terms
    
    eps = 1e-8
    weights = torch.zeros(activations.size(1), device=device)
    
    for k in range(activations.size(1)):
        # Numerator: Σ_i Σ_j (grad_ij^k * A_ij^k)
        numerator = torch.sum(gradients[0, k] * activations[0, k])
        
        # Denominator: Σ_i Σ_j A_ij^k + ε
        denominator = torch.sum(activations[0, k]) + eps
        
        # Additional term for Grad-CAM++: include higher-order information
        if grad_squared is not None and grad_cubed is not None:
            # Simplified higher-order term
            grad2_sum = torch.sum(grad_squared[0, k])
            grad3_sum = torch.sum(grad_cubed[0, k])
            higher_order = grad2_sum / (2 * grad2_sum + grad3_sum + eps)
            weights[k] = numerator / denominator * higher_order
        else:
            weights[k] = numerator / denominator
    
    # Apply ReLU to keep only positive weights
    weights = F.relu(weights)
    
    # Weighted combination of feature maps
    cam = torch.zeros_like(activations[0, 0])
    for k, w in enumerate(weights):
        cam += w * activations[0, k]
    
    # Suppress thin edge responses by subtracting per-slice percentile baseline then ReLU
    cam = cam - torch.quantile(cam, 0.85)
    cam = F.relu(cam)
    cam = cam.unsqueeze(0).unsqueeze(0)
    target_size = smri_tensor.shape[-3:]
    cam_upsampled = F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)
    cam_np = cam_upsampled.squeeze().cpu().numpy()

    h1.remove(); h2.remove()
    return cam_np.astype(np.float32)


def compute_saliency_volume(model, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    x = input_tensor.clone().detach().requires_grad_(True)
    model.zero_grad()
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    score = F.softmax(logits, dim=1)[0, target_class]
    score.backward()
    grad = x.grad.detach().squeeze(0).squeeze(0).cpu().numpy()
    sal = np.abs(grad)
    return sal.astype(np.float32)


def compute_smoothgrad_volume(model, input_tensor: torch.Tensor, target_class: int, n_samples: int = 40, noise_std: float = 0.1) -> np.ndarray:
    """
    SmoothGrad: Average gradients over noisy versions of the input.
    Reduces noise and provides more stable saliency maps.
    """
    model.eval()
    saliency_maps = []
    
    # Get in-mask intensity std for noise scaling
    mask = (input_tensor != 0).float()
    if mask.sum() > 0:
        in_mask_std = input_tensor[mask > 0].std().item()
        noise_scale = noise_std * in_mask_std
    else:
        noise_scale = noise_std * input_tensor.std().item()
    
    for _ in range(n_samples):
        # Add noise to input
        noise = torch.randn_like(input_tensor) * noise_scale
        noisy_input = input_tensor + noise
        noisy_input = noisy_input.requires_grad_(True)
        
        model.zero_grad()
        out = model(noisy_input)
        logits = out[0] if isinstance(out, tuple) else out
        score = F.softmax(logits, dim=1)[0, target_class]
        score.backward()
        
        grad = noisy_input.grad.detach().squeeze(0).squeeze(0).cpu().numpy()
        saliency_maps.append(np.abs(grad))
    
    # Average over all samples
    smooth_sal = np.mean(saliency_maps, axis=0)
    return smooth_sal.astype(np.float32)


def compute_integrated_gradients_volume(model, input_tensor: torch.Tensor, target_class: int, n_steps: int = 64, baseline_type: str = 'zeros') -> np.ndarray:
    """
    Integrated Gradients: Integrate gradients along path from baseline to input.
    Provides more faithful attributions than vanilla gradients.
    """
    model.eval()
    
    # Create baseline (zeros or mean)
    if baseline_type == 'zeros':
        baseline = torch.zeros_like(input_tensor)
    elif baseline_type == 'mean':
        mask = (input_tensor != 0).float()
        if mask.sum() > 0:
            mean_val = input_tensor[mask > 0].mean()
            baseline = torch.zeros_like(input_tensor)
            baseline[mask > 0] = mean_val
        else:
            baseline = torch.zeros_like(input_tensor)
    else:
        baseline = torch.zeros_like(input_tensor)
    
    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, n_steps + 1, device=input_tensor.device)
    integrated_grads = torch.zeros_like(input_tensor)
    
    for alpha in alphas[1:]:  # Skip alpha=0
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated = interpolated.requires_grad_(True)
        
        model.zero_grad()
        out = model(interpolated)
        logits = out[0] if isinstance(out, tuple) else out
        score = F.softmax(logits, dim=1)[0, target_class]
        score.backward()
        
        grad = interpolated.grad.detach()
        integrated_grads += grad / n_steps
    
    # Multiply by input difference
    integrated_grads *= (input_tensor - baseline)
    ig_sal = integrated_grads.squeeze(0).squeeze(0).cpu().numpy()
    return np.abs(ig_sal).astype(np.float32)


def compute_fused_saliency_volume(model, input_tensor: torch.Tensor, target_class: int, 
                                 sg_samples: int = 40, ig_steps: int = 64, 
                                 noise_std: float = 0.1, baseline_type: str = 'zeros') -> np.ndarray:
    """
    Fused SmoothGrad + Integrated Gradients using geometric mean.
    Combines benefits of both methods for more robust saliency.
    """
    # Compute both methods
    sg_sal = compute_smoothgrad_volume(model, input_tensor, target_class, sg_samples, noise_std)
    ig_sal = compute_integrated_gradients_volume(model, input_tensor, target_class, ig_steps, baseline_type)
    
    # Z-score normalize within brain mask
    mask = (input_tensor.squeeze(0).squeeze(0).cpu().numpy() != 0)
    
    def zscore_within_mask(arr, mask):
        if mask.sum() == 0:
            return arr
        masked_vals = arr[mask]
        mean_val = masked_vals.mean()
        std_val = masked_vals.std()
        if std_val < 1e-8:
            return arr
        zscored = (arr - mean_val) / std_val
        return zscored
    
    sg_norm = zscore_within_mask(sg_sal, mask)
    ig_norm = zscore_within_mask(ig_sal, mask)
    
    # Clamp negatives to 0 and take geometric mean
    sg_pos = np.maximum(sg_norm, 0)
    ig_pos = np.maximum(ig_norm, 0)
    
    # Geometric mean (avoid zeros)
    eps = 1e-8
    fused = np.sqrt((sg_pos + eps) * (ig_pos + eps))
    
    return fused.astype(np.float32)


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


def save_overlay_pngs(anat: np.ndarray, heat: np.ndarray, out_path: Path, title: str = "", mask: Optional[np.ndarray] = None, alpha: float = 0.4, zero_outside: bool = False):
    """
    Save quick axial/coronal/sagittal overlays for visual sanity check.
    """
    anat = anat.astype(np.float32)
    # Robust normalize anat and heat for visibility
    a_lo, a_hi = np.percentile(anat, 2.0), np.percentile(anat, 98.0)
    if a_hi - a_lo < 1e-6:
        anat_n = anat
    else:
        anat_n = np.clip((anat - a_lo) / (a_hi - a_lo), 0.0, 1.0)
    if mask is not None and np.any(mask > 0):
        sel = heat[mask > 0]
        h_lo, h_hi = np.percentile(sel, 90.0), np.percentile(sel, 99.5)
    else:
        h_lo, h_hi = np.percentile(heat, 90.0), np.percentile(heat, 99.5)
    heat = np.clip((heat - h_lo) / (h_hi - h_lo + 1e-8), 0.0, 1.0)
    if bool(zero_outside) and mask is not None:
        heat = heat * (mask > 0).astype(np.float32)
    D, H, W = anat.shape
    za, ya, xa = D // 2, H // 2, W // 2
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(anat_n[za].T, cmap='gray', origin='lower')
    axs[0].imshow(heat[za].T, cmap='hot', alpha=float(alpha), origin='lower')
    axs[0].set_title('Sagittal')
    axs[0].axis('off')
    axs[1].imshow(anat_n[:, ya, :].T, cmap='gray', origin='lower')
    axs[1].imshow(heat[:, ya, :].T, cmap='hot', alpha=float(alpha), origin='lower')
    axs[1].set_title('Coronal')
    axs[1].axis('off')
    axs[2].imshow(anat_n[:, :, xa].T, cmap='gray', origin='lower')
    axs[2].imshow(heat[:, :, xa].T, cmap='hot', alpha=float(alpha), origin='lower')
    axs[2].set_title('Axial')
    axs[2].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _erode_mask_mm(mask: np.ndarray, header, erode_mm: float) -> np.ndarray:
    """
    Erode a boolean mask by a physical margin in millimeters using a distance transform.
    If SciPy is unavailable or erode_mm <= 0, returns the original mask.
    """
    try:
        mm = float(erode_mm)
    except Exception:
        mm = 0.0
    if distance_transform_edt is None or mm <= 0.0:
        return (mask > 0).astype(np.float32)
    try:
        zooms = None
        try:
            zooms = header.get_zooms()[:3] if header is not None else None
        except Exception:
            zooms = None
        if zooms is None or any((z is None or not np.isfinite(z) or float(z) <= 0) for z in zooms):
            zooms = (1.0, 1.0, 1.0)
        # Distance inside mask to nearest background, in mm
        dist = distance_transform_edt(mask > 0, sampling=zooms).astype(np.float32)
        eroded = dist >= float(mm)
        return eroded.astype(np.float32)
    except Exception:
        return (mask > 0).astype(np.float32)


def _mm_to_voxels_per_axis(mm: float, header) -> Tuple[int, int, int]:
    try:
        mm_val = float(mm)
    except Exception:
        mm_val = 0.0
    try:
        zooms = header.get_zooms()[:3] if header is not None else (1.0, 1.0, 1.0)
    except Exception:
        zooms = (1.0, 1.0, 1.0)
    zv = []
    for z in zooms:
        try:
            zv.append(int(max(0, round(mm_val / float(z) if float(z) > 0 else 0.0))))
        except Exception:
            zv.append(0)
    return int(zv[0]), int(zv[1]), int(zv[2])


def _expand_and_clip_bbox(bbox: Tuple[int, int, int, int, int, int], pad_xyz: Tuple[int, int, int], shape: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox
    pz, py, px = pad_xyz
    z0 = max(0, z0 - pz); z1 = min(shape[0], z1 + pz)
    y0 = max(0, y0 - py); y1 = min(shape[1], y1 + py)
    x0 = max(0, x0 - px); x1 = min(shape[2], x1 + px)
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def _embed_focus_into_full(focus_vol: np.ndarray, bbox: Tuple[int, int, int, int, int, int], full_shape: Tuple[int, int, int]) -> np.ndarray:
    full = np.zeros(full_shape, dtype=np.float32)
    z0, z1, y0, y1, x0, x1 = bbox
    sz = min(focus_vol.shape[0], max(0, z1 - z0))
    sy = min(focus_vol.shape[1], max(0, y1 - y0))
    sx = min(focus_vol.shape[2], max(0, x1 - x0))
    if sz > 0 and sy > 0 and sx > 0:
        full[z0:z0+sz, y0:y0+sy, x0:x0+sx] = focus_vol[:sz, :sy, :sx].astype(np.float32)
    return full


def _inside_distance_mm(mask: np.ndarray, header) -> np.ndarray:
    if distance_transform_edt is None:
        return (mask > 0).astype(np.float32)
    try:
        zooms = header.get_zooms()[:3] if header is not None else (1.0, 1.0, 1.0)
    except Exception:
        zooms = (1.0, 1.0, 1.0)
    dist = distance_transform_edt(mask > 0, sampling=zooms).astype(np.float32)
    return dist


def resize_volume_to_shape(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Trilinear resize of a 3D volume to target shape.
    """
    if tuple(volume.shape) == tuple(target_shape):
        return volume.astype(np.float32)
    t = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=tuple(target_shape), mode='trilinear', align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def compute_nonzero_bbox(volume: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Compute bounding box of non-zero voxels in [D,H,W] volume. Returns (z0,z1,y0,y1,x0,x1) or None if empty.
    """
    nz = np.argwhere(volume != 0)
    if nz.size == 0:
        return None
    z0, y0, x0 = nz.min(axis=0)
    z1, y1, x1 = nz.max(axis=0) + 1
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def main():
    parser = argparse.ArgumentParser(description='Deep clinical prediction with 3D interpretability')
    parser.add_argument('--image', required=True, help='Path to input NIfTI (.nii or .nii.gz)')
    parser.add_argument('--model-arch', default='SMRI_GradCAM_3DCNN', help='Model architecture (SMRI_GradCAM_3DCNN, Simple3DCNN, ResNet18_3D)')
    parser.add_argument('--weights', required=True, help='Path to model weights (.pt/.pth) state_dict or checkpoint')
    parser.add_argument('--weights-list', nargs='+', type=str, help='Additional weight files to ensemble (logits averaged)')
    parser.add_argument('--weights-glob', type=str, help='Glob to collect multiple weight files for ensembling')
    parser.add_argument('--ensemble-avg-method', choices=['logits', 'probs'], default='logits', help='Average logits (recommended) or probabilities across models')
    # Risk-aware prediction and TTA
    parser.add_argument('--tta-n', type=int, default=0, help='Number of TTA replicates (Gaussian noise)')
    parser.add_argument('--tta-noise-std', type=float, default=0.005, help='Stddev of Gaussian noise for TTA')
    parser.add_argument('--mc-dropout', action='store_true', help='Enable MC Dropout by activating Dropout layers during inference')
    parser.add_argument('--risk-weights', type=float, nargs='+', help='Multipliers per class to bias decisions, length = num-classes (e.g., 1 1.2 1.2)')
    parser.add_argument('--cn-min-prob', type=float, default=None, help='If predicted CN prob is below this, choose best non-CN class')
    parser.add_argument('--interop-threads', type=int, default=0, help='Torch set_num_interop_threads (0=leave default)')
    parser.add_argument('--cudnn-benchmark', action='store_true', help='Enable cudnn.benchmark for faster convs on fixed sizes (CUDA only)')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision (CUDA autocast) for prediction phase')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--known-label', type=int, default=None, help='Optional known/true class index for this subject (0..num-classes-1). If set, interpretability targets (and overlays) use this label instead of the predicted class.')
    parser.add_argument('--label-map-json', type=str, help='Optional JSON mapping of numeric labels to names')
    parser.add_argument('--normalize', choices=['zscore', 'minmax', 'none'], default='zscore')
    parser.add_argument('--resize-dims', type=int, nargs=3, metavar=('D', 'H', 'W'), help='Optional resize to D H W before inference')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output-dir', default='~/reseng202500013-ndd-ml/clinical_outputs/deep', help='Directory to write outputs (JSON + NIfTI maps)')
    parser.add_argument('--occ-ksize', type=int, default=16)
    parser.add_argument('--occ-stride', type=int, default=None)
    parser.add_argument('--occ-baseline', type=float, default=0.0)
    parser.add_argument('--gshap-samples', type=int, default=50, help='GradientSHAP samples per attribution')
    parser.add_argument('--gshap-std', type=float, default=0.001, help='GradientSHAP noise stdev')
    parser.add_argument('--num-threads', type=int, default=0, help='Torch set_num_threads (0=leave default)')
    parser.add_argument('--all-classes-interpret', action='store_true', help='Produce Grad-CAM for all classes (default: predicted only)')
    parser.add_argument('--save-overlay-pngs', action='store_true', help='Also save 2D slice overlays (axial/coronal/sagittal) as PNGs')
    parser.add_argument('--cam-layer', type=str, default='prepool', choices=['last', 'prepool', 'mid'], help='Which conv layer to use for Grad-CAM (Simple3DCNN)')
    parser.add_argument('--cam-single-model', action='store_true', help='If using an ensemble, compute CAMs using only the first model (much faster); prediction can still use the full ensemble.')
    parser.add_argument('--cam-tta-n', type=int, default=0, help='Number of TTA replicates when computing CAMs (averaged for stability). If 0, falls back to --tta-n.')
    parser.add_argument('--cam-tta-noise-std', type=float, default=0.0, help='Gaussian noise std for CAM TTA (fraction of input scale). If 0, falls back to --tta-noise-std.')
    parser.add_argument('--cam-tta-filter-by-pred', action='store_true', help='When averaging CAMs over TTA, include only replicates that predict the target class')
    # Interpretability selection
    parser.add_argument('--run-all', action='store_true', help='Run all interpretability maps (default if neither --run nor --run-all provided)')
    parser.add_argument('--run', nargs='+', choices=['gradcam', 'gradcam_plusplus', 'saliency', 'smoothgrad', 'integrated_gradients', 'fused_saliency', 'occlusion', 'gradientshap'], help='One or more interpretability methods to run, e.g. --run gradcam smoothgrad fused_saliency')
    parser.add_argument('--cam-classes', type=int, nargs='+', default=None, help='Optional explicit class indices to generate CAMs for (overrides --all-classes-interpret / predicted class selection).')
    # Advanced saliency parameters
    parser.add_argument('--sg-samples', type=int, default=40, help='SmoothGrad samples (40=fast, 80=HQ)')
    parser.add_argument('--sg-noise-std', type=float, default=0.1, help='SmoothGrad noise std (fraction of in-mask intensity std)')
    parser.add_argument('--ig-steps', type=int, default=64, help='Integrated Gradients steps (64=fast, 128=HQ)')
    parser.add_argument('--ig-baseline', type=str, default='zeros', choices=['zeros', 'mean'], help='Integrated Gradients baseline type')
    # Overlay display controls (masking without reprocessing/training)
    parser.add_argument('--overlay-erode-mm', type=float, default=2.0, help='Erode the nonzero brain mask by this many millimeters for overlay normalization (0 disables)')
    parser.add_argument('--overlay-zero-outside', action='store_true', help='Zero CAM values outside the eroded brain mask when rendering PNG overlays')
    parser.add_argument('--overlay-alpha', type=float, default=0.4, help='Alpha transparency for overlays in PNGs')
    # Focused-ROI inference (crop input before prediction/interpretability)
    parser.add_argument('--focus-input', action='store_true', help='Run prediction and interpretability on a cropped, eroded brain ROI to suppress edge/rim artifacts')
    parser.add_argument('--focus-erode-mm', type=float, default=2.0, help='Mask erosion in mm used to define the focused ROI (before padding)')
    parser.add_argument('--focus-pad-mm', type=float, default=2.0, help='Extra mm of padding added around the ROI after erosion when cropping the input')
    parser.add_argument('--focus-taper-mm', type=float, default=3.0, help='Soft window thickness (mm) applied to input inside the brain mask to reduce border attention (0 disables)')
    # Grad-CAM++ edge suppression (does not affect vanilla Grad-CAM)
    parser.add_argument('--gcpp-edge-taper-mm', type=float, default=3.0, help='If >0, multiply Grad-CAM++ by an interior distance ramp of this thickness (mm) to suppress rim focus')
    # Voting emphasis for TTA
    parser.add_argument('--vote-weight', type=float, default=None, help='Blend weight in [0,1] for TTA vote fractions when combining with averaged probabilities. Higher focuses more on votes. If not set, uses adaptive weighting.')
    parser.add_argument('--vote-majority-thresh', type=float, default=None, help='If the top vote fraction ≥ this threshold, snap final probabilities to that class (majority rule).')

    args = parser.parse_args()

    # Determine which interpretability methods to run
    if args.run and len(args.run) > 0:
        run_set = set([str(m).lower() for m in args.run])
    else:
        # Default or --run-all → run everything
        run_set = {'gradcam', 'saliency', 'occlusion', 'gradientshap'}

    device = get_device(args.device)
    # Collect method-specific errors to export in JSON and a sidecar log
    interpret_errors: Dict[str, List[str]] = {
        'gradcam': [],
        'saliency': [],
        'occlusion': [],
        'gradientshap': [],
    }

    if isinstance(args.num_threads, int) and args.num_threads > 0:
        try:
            torch.set_num_threads(int(args.num_threads))
        except Exception:
            pass
    if isinstance(args.interop_threads, int) and args.interop_threads > 0:
        try:
            torch.set_num_interop_threads(int(args.interop_threads))
        except Exception:
            pass
    try:
        if bool(args.cudnn_benchmark) and device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    # Label mapping (default: 0->CN, 1->AD, 2->PD)
    label_map: Dict[int, str] = {0: 'CN', 1: 'AD', 2: 'PD'}
    if args.label_map_json:
        try:
            with open(expand_path(args.label_map_json), 'r') as f:
                raw = json.load(f)
            label_map = {int(k): str(v) for k, v in raw.items()}
        except Exception as e:
            print(f"Warning: failed to load label map JSON ({e}). Using default mapping.")

    # Collect weight paths (deduplicated, preserve order)
    candidates: List[str] = []
    if args.weights:
        candidates.append(expand_path(args.weights))
    if args.weights_list:
        candidates.extend([expand_path(w) for w in args.weights_list])
    if args.weights_glob:
        candidates.extend(sorted(glob_mod.glob(expand_path(args.weights_glob))))
    seen = set()
    weight_paths: List[str] = []
    for p in candidates:
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            weight_paths.append(p)
    is_ensemble = len(weight_paths) > 1

    # Load model(s)
    if not is_ensemble:
        model, gradcam_module = load_model(args.model_arch, args.num_classes, in_channels=1, weights_path=weight_paths[0], device=device)
        models: List[object] = [model]
    else:
        models = []
        gradcam_module = None
        for w in weight_paths:
            m, gm = load_model(args.model_arch, args.num_classes, in_channels=1, weights_path=w, device=device)
            if gradcam_module is None:
                gradcam_module = gm
            models.append(m)

    # Load and preprocess image
    vol_np, affine, header = load_nifti(args.image)
    # Preserve a brain mask BEFORE any normalization to avoid non-zero background from re-normalization
    brain_mask_orig = (vol_np != 0).astype(np.float32)
    
    # Optional focused-ROI inference: crop to an eroded, padded brain bbox to remove bright rim
    use_focus = bool(getattr(args, 'focus_input', False))
    focus_bbox = (0, vol_np.shape[0], 0, vol_np.shape[1], 0, vol_np.shape[2])
    if use_focus:
        eroded_for_bbox = _erode_mask_mm(brain_mask_orig, header, float(getattr(args, 'focus_erode_mm', 2.0)))
        bbox0 = compute_nonzero_bbox(eroded_for_bbox)
        if bbox0 is None:
            bbox0 = compute_nonzero_bbox(brain_mask_orig)
        if bbox0 is None:
            bbox0 = focus_bbox
        pad_xyz = _mm_to_voxels_per_axis(float(getattr(args, 'focus_pad_mm', 2.0)), header)
        focus_bbox = _expand_and_clip_bbox(bbox0, pad_xyz, vol_np.shape)
        z0, z1, y0, y1, x0, x1 = focus_bbox
        vol_focus = vol_np[z0:z1, y0:y1, x0:x1]
        brain_mask_focus = brain_mask_orig[z0:z1, y0:y1, x0:x1]
    else:
        vol_focus = vol_np
        brain_mask_focus = brain_mask_orig
    
    # For overlays, optionally erode the MASK used for normalization
    brain_mask_eroded_focus = _erode_mask_mm(brain_mask_focus, header, float(getattr(args, 'overlay_erode_mm', 0.0)))
    # And embed to full shape for PNG overlays
    brain_mask_eroded = _embed_focus_into_full(brain_mask_eroded_focus, focus_bbox, vol_np.shape) if use_focus else brain_mask_eroded_focus
    
    # Normalize ONLY the focus crop for model input, then apply optional soft interior window (taper) to down-weight border voxels for attention
    vol_focus_n = normalize_volume(vol_focus, method=args.normalize)
    try:
        taper_mm = float(getattr(args, 'focus_taper_mm', 0.0))
    except Exception:
        taper_mm = 0.0
    if taper_mm > 0.0:
        # Build a smooth ramp [0..1] inside the eroded focus mask and multiply the input
        dist_mm_focus = _inside_distance_mm(brain_mask_focus, header)
        ramp_focus = np.clip(dist_mm_focus / max(taper_mm, 1e-6), 0.0, 1.0).astype(np.float32)
        vol_focus_n = vol_focus_n * ramp_focus
    input_tensor = to_model_tensor(vol_focus_n, device=device, resize_dims=tuple(args.resize_dims) if args.resize_dims else None)

    # Predict with optional ensemble + TTA
    if not is_ensemble and int(args.tta_n) <= 0:
        pred_idx, probs, logits = predict(models[0], input_tensor)
        meta_pred = {'tta_replicates': 1, 'votes': [pred_idx], 'vote_hist': {pred_idx: 1}}
    else:
        probs, logits, meta_pred = aggregate_predictions(
            models, input_tensor, avg_method=args.ensemble_avg_method,
            tta_n=int(args.tta_n), tta_noise_std=float(args.tta_noise_std), mc_dropout=bool(args.mc_dropout),
            use_amp=bool(args.amp), device_str=device,
            vote_weight=getattr(args, 'vote_weight', None), vote_majority_thresh=getattr(args, 'vote_majority_thresh', None)
        )
        pred_idx = int(np.argmax(probs))

    # Raw prediction snapshot
    pred_name_raw = label_map.get(pred_idx, str(pred_idx))
    confidence_raw = float(np.max(probs))
    prob_dict_raw = {label_map.get(i, str(i)): float(p) for i, p in enumerate(probs)}

    # Risk-aware adjustment (applied after consensus blending)
    adjusted_probs = probs.copy()
    applied_risk = None
    if args.risk_weights and len(args.risk_weights) == int(args.num_classes):
        rw = np.asarray(args.risk_weights, dtype=np.float32)
        adjusted_probs = adjusted_probs * rw
        applied_risk = rw.tolist()
    # CN thresholding (assumes class 0 = CN by default mapping)
    final_idx = int(np.argmax(adjusted_probs))
    if args.cn_min_prob is not None and final_idx == 0:
        if float(adjusted_probs[0]) < float(args.cn_min_prob):
            # pick best non-CN class
            non_cn_idx = int(np.argmax(adjusted_probs[1:]) + 1)
            final_idx = non_cn_idx

    pred_idx = final_idx
    pred_name = label_map.get(pred_idx, str(pred_idx))
    # For confidence, report the adjusted winning score (not renormalized)
    confidence = float(adjusted_probs[pred_idx])
    prob_dict = {label_map.get(i, str(i)): float(adjusted_probs[i]) for i in range(len(adjusted_probs))}

    # Choose which class index to use for "default" interpretability targets and overlays.
    # For cohort analyses, a known/true label is often preferred to avoid missing overlays when misclassified.
    known_label = getattr(args, 'known_label', None)
    interpret_idx = pred_idx
    if known_label is not None:
        try:
            known_i = int(known_label)
            if known_i < 0 or known_i >= int(args.num_classes):
                raise ValueError(f"--known-label out of range [0, {int(args.num_classes)-1}]")
            interpret_idx = known_i
        except Exception as e:
            print(f"Warning: ignoring --known-label ({e}); using predicted class for interpretability.")
            interpret_idx = pred_idx

    # Prepare output directory
    out_dir = Path(expand_path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    sid = Path(args.image).stem.replace('.nii', '').replace('.gz', '')

    # Interpretability maps (conditionally executed)
    gradcam_paths = {}
    sal_path = None
    occ_path = None
    gshap_path = None

    # Vanilla Grad-CAM
    if 'gradcam' in run_set:
        if getattr(args, 'cam_classes', None) is not None and len(getattr(args, 'cam_classes')) > 0:
            classes_to_compute = sorted(set([int(c) for c in getattr(args, 'cam_classes')]))
        else:
            classes_to_compute = range(args.num_classes) if args.all_classes_interpret else [interpret_idx]
        overlay_saved = False
        for c in classes_to_compute:
            try:
                if int(c) < 0 or int(c) >= int(args.num_classes):
                    raise ValueError(f"Requested class {c} out of range [0, {int(args.num_classes)-1}]")
                def _cam_for_model(m):
                    # CAM TTA: average multiple noisy passes if requested (defaults to global TTA when unset)
                    reps = int(getattr(args, 'cam_tta_n', 0))
                    if reps <= 0:
                        reps = int(getattr(args, 'tta_n', 0))
                    reps = max(1, reps)
                    noise_std = float(getattr(args, 'cam_tta_noise_std', 0.0))
                    if noise_std <= 0.0:
                        noise_std = float(getattr(args, 'tta_noise_std', 0.0))
                    cams = []
                    used = 0
                    for r in range(reps):
                        xt = input_tensor
                        if reps > 1 and noise_std > 0.0:
                            xt = xt + torch.randn_like(xt) * noise_std
                        # Optionally filter by replicate's predicted class
                        if bool(getattr(args, 'cam_tta_filter_by_pred', False)):
                            with torch.no_grad():
                                out = m(xt)
                                lg = out[0] if isinstance(out, tuple) else out
                                pred_r = int(torch.argmax(F.softmax(lg, dim=1), dim=1).item())
                            if pred_r != int(c):
                                continue
                        cams.append(compute_gradcam_volume(gradcam_module, m, xt, c, args.model_arch, device, cam_layer=args.cam_layer))
                        used += 1
                    if len(cams) == 0:
                        # Fallback: use single pass on original input to avoid empty average
                        return compute_gradcam_volume(gradcam_module, m, input_tensor, c, args.model_arch, device, cam_layer=args.cam_layer)
                    return np.mean(np.stack(cams, axis=0), axis=0).astype(np.float32)

                if not is_ensemble or bool(getattr(args, 'cam_single_model', False)):
                    cam = _cam_for_model(models[0])
                else:
                    cam_list = [_cam_for_model(m) for m in models]
                    cam = np.mean(np.stack(cam_list, axis=0), axis=0).astype(np.float32)
                # Map CAM to full image shape (respecting focused ROI if enabled)
                if use_focus:
                    cam_local = resize_volume_to_shape(cam, vol_focus_n.shape)
                    cam_full = _embed_focus_into_full(cam_local, focus_bbox, vol_np.shape)
                else:
                    cam_full = resize_volume_to_shape(cam, vol_np.shape)
                cam_path = out_dir / f"{sid}_gradcam_class{c}.nii.gz"
                save_nifti(cam_full, affine, header, cam_path)
                gradcam_paths[str(c)] = str(cam_path)
                # Optional overlays for predicted class
                if args.save_overlay_pngs and int(c) == int(interpret_idx) and not overlay_saved:
                    try:
                        class_name = label_map.get(int(c), str(c))
                        save_overlay_pngs(
                            vol_np,
                            cam_full,
                            out_dir / f"{sid}_gradcam_overlay.png",
                            title=f"{sid} • Grad-CAM • {class_name}",
                            mask=brain_mask_eroded,
                            alpha=float(getattr(args, 'overlay_alpha', 0.4)),
                            zero_outside=bool(getattr(args, 'overlay_zero_outside', False))
                        )
                        overlay_saved = True
                    except Exception:
                        pass
            except Exception as e:
                msg = f"Grad-CAM failed for class {c}: {e}"
                print(msg)
                interpret_errors['gradcam'].append(msg)

    # Grad-CAM++
    if 'gradcam_plusplus' in run_set:
        if getattr(args, 'cam_classes', None) is not None and len(getattr(args, 'cam_classes')) > 0:
            classes_to_compute = sorted(set([int(c) for c in getattr(args, 'cam_classes')]))
        else:
            classes_to_compute = range(args.num_classes) if args.all_classes_interpret else [interpret_idx]
        overlay_saved_pp = False
        for c in classes_to_compute:
            try:
                if int(c) < 0 or int(c) >= int(args.num_classes):
                    raise ValueError(f"Requested class {c} out of range [0, {int(args.num_classes)-1}]")
                def _campp_for_model(m):
                    reps = int(getattr(args, 'cam_tta_n', 0))
                    if reps <= 0:
                        reps = int(getattr(args, 'tta_n', 0))
                    reps = max(1, reps)
                    noise_std = float(getattr(args, 'cam_tta_noise_std', 0.0))
                    if noise_std <= 0.0:
                        noise_std = float(getattr(args, 'tta_noise_std', 0.0))
                    cams = []
                    used = 0
                    for r in range(reps):
                        xt = input_tensor
                        if reps > 1 and noise_std > 0.0:
                            xt = xt + torch.randn_like(xt) * noise_std
                        if bool(getattr(args, 'cam_tta_filter_by_pred', False)):
                            with torch.no_grad():
                                out = m(xt)
                                lg = out[0] if isinstance(out, tuple) else out
                                pred_r = int(torch.argmax(F.softmax(lg, dim=1), dim=1).item())
                            if pred_r != int(c):
                                continue
                        cams.append(compute_gradcam_plusplus_volume(gradcam_module, m, xt, c, args.model_arch, device, cam_layer=args.cam_layer))
                        used += 1
                    if len(cams) == 0:
                        return compute_gradcam_plusplus_volume(gradcam_module, m, input_tensor, c, args.model_arch, device, cam_layer=args.cam_layer)
                    return np.mean(np.stack(cams, axis=0), axis=0).astype(np.float32)

                if not is_ensemble or bool(getattr(args, 'cam_single_model', False)):
                    cam = _campp_for_model(models[0])
                else:
                    cam_list = [_campp_for_model(m) for m in models]
                    cam = np.mean(np.stack(cam_list, axis=0), axis=0).astype(np.float32)
                # Map CAM++ to full image shape
                if use_focus:
                    cam_local = resize_volume_to_shape(cam, vol_focus_n.shape)
                    cam_full = _embed_focus_into_full(cam_local, focus_bbox, vol_np.shape)
                else:
                    cam_full = resize_volume_to_shape(cam, vol_np.shape)

                # Optional edge tapering for Grad-CAM++ to avoid mask rims
                try:
                    taper_mm = float(getattr(args, 'gcpp_edge_taper_mm', 0.0))
                except Exception:
                    taper_mm = 0.0
                if taper_mm > 0.0:
                    dist_mm = _inside_distance_mm(brain_mask_orig, header)
                    ramp = np.clip(dist_mm / max(taper_mm, 1e-6), 0.0, 1.0).astype(np.float32)
                    cam_full = cam_full * ramp

                cam_path = out_dir / f"{sid}_gradcam_plusplus_class{c}.nii.gz"
                save_nifti(cam_full, affine, header, cam_path)
                gradcam_paths[f"plusplus_{c}"] = str(cam_path)
                # Optional overlays for predicted class
                if args.save_overlay_pngs and int(c) == int(interpret_idx) and not overlay_saved_pp:
                    try:
                        class_name = label_map.get(int(c), str(c))
                        save_overlay_pngs(
                            vol_np,
                            cam_full,
                            out_dir / f"{sid}_gradcam_plusplus_overlay.png",
                            title=f"{sid} • Grad-CAM++ • {class_name}",
                            mask=brain_mask_eroded,
                            alpha=float(getattr(args, 'overlay_alpha', 0.4)),
                            zero_outside=bool(getattr(args, 'overlay_zero_outside', False))
                        )
                        overlay_saved_pp = True
                    except Exception:
                        pass
            except Exception as e:
                msg = f"Grad-CAM++ failed for class {c}: {e}"
                print(msg)
                interpret_errors['gradcam'].append(msg)

    # Saliency (absolute gradient)
    if 'saliency' in run_set:
        try:
            if not is_ensemble:
                sal = compute_saliency_volume(models[0], input_tensor, interpret_idx)
            else:
                sal_accum = None
                for m in models:
                    sal_i = compute_saliency_volume(m, input_tensor, interpret_idx)
                    sal_accum = sal_i if sal_accum is None else (sal_accum + sal_i)
                sal = sal_accum / float(len(models))
            if use_focus:
                sal_local = resize_volume_to_shape(sal, vol_focus_n.shape)
                sal_full = _embed_focus_into_full(sal_local, focus_bbox, vol_np.shape)
            else:
                sal_full = resize_volume_to_shape(sal, vol_np.shape)
            # Normalize using in-brain voxels only
            sal_full = robust_normalize_within_mask(sal_full, brain_mask_orig)
            sal_path = out_dir / f"{sid}_saliency.nii.gz"
            save_nifti(sal_full, affine, header, sal_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        sal_full,
                        out_dir / f"{sid}_saliency_overlay.png",
                        title=f"{sid} • Saliency • Target={label_map.get(int(interpret_idx), str(interpret_idx))}",
                        mask=brain_mask_eroded,
                        alpha=float(getattr(args, 'overlay_alpha', 0.4)),
                        zero_outside=bool(getattr(args, 'overlay_zero_outside', False))
                    )
                except Exception:
                    pass
        except Exception as e:
            msg = f"Saliency failed: {e}"
            print(msg)
            interpret_errors['saliency'].append(msg)
            sal_path = None

    # SmoothGrad
    if 'smoothgrad' in run_set:
        try:
            if not is_ensemble:
                sg_sal = compute_smoothgrad_volume(models[0], input_tensor, interpret_idx, 
                                                 n_samples=int(args.sg_samples), 
                                                 noise_std=float(args.sg_noise_std))
            else:
                sg_accum = None
                for m in models:
                    sg_i = compute_smoothgrad_volume(m, input_tensor, interpret_idx, 
                                                   n_samples=int(args.sg_samples), 
                                                   noise_std=float(args.sg_noise_std))
                    sg_accum = sg_i if sg_accum is None else (sg_accum + sg_i)
                sg_sal = sg_accum / float(len(models))
            if use_focus:
                sg_local = resize_volume_to_shape(sg_sal, vol_focus_n.shape)
                sg_full = _embed_focus_into_full(sg_local, focus_bbox, vol_np.shape)
            else:
                sg_full = resize_volume_to_shape(sg_sal, vol_np.shape)
            sg_full = robust_normalize_within_mask(sg_full, brain_mask_orig)
            sg_path = out_dir / f"{sid}_smoothgrad.nii.gz"
            save_nifti(sg_full, affine, header, sg_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        sg_full,
                        out_dir / f"{sid}_smoothgrad_overlay.png",
                        title=f"{sid} • SmoothGrad • Target={label_map.get(int(interpret_idx), str(interpret_idx))}",
                        mask=brain_mask_eroded,
                        alpha=float(getattr(args, 'overlay_alpha', 0.4)),
                        zero_outside=bool(getattr(args, 'overlay_zero_outside', False))
                    )
                except Exception:
                    pass
        except Exception as e:
            msg = f"SmoothGrad failed: {e}"
            print(msg)
            interpret_errors['saliency'].append(msg)

    # Integrated Gradients
    if 'integrated_gradients' in run_set:
        try:
            if not is_ensemble:
                ig_sal = compute_integrated_gradients_volume(models[0], input_tensor, interpret_idx, 
                                                           n_steps=int(args.ig_steps), 
                                                           baseline_type=args.ig_baseline)
            else:
                ig_accum = None
                for m in models:
                    ig_i = compute_integrated_gradients_volume(m, input_tensor, interpret_idx, 
                                                             n_steps=int(args.ig_steps), 
                                                             baseline_type=args.ig_baseline)
                    ig_accum = ig_i if ig_accum is None else (ig_accum + ig_i)
                ig_sal = ig_accum / float(len(models))
            if use_focus:
                ig_local = resize_volume_to_shape(ig_sal, vol_focus_n.shape)
                ig_full = _embed_focus_into_full(ig_local, focus_bbox, vol_np.shape)
            else:
                ig_full = resize_volume_to_shape(ig_sal, vol_np.shape)
            ig_full = robust_normalize_within_mask(ig_full, brain_mask_orig)
            ig_path = out_dir / f"{sid}_integrated_gradients.nii.gz"
            save_nifti(ig_full, affine, header, ig_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        ig_full,
                        out_dir / f"{sid}_integrated_gradients_overlay.png",
                        title=f"{sid} • Integrated Gradients • Target={label_map.get(int(interpret_idx), str(interpret_idx))}",
                        mask=brain_mask_eroded,
                        alpha=float(getattr(args, 'overlay_alpha', 0.4)),
                        zero_outside=bool(getattr(args, 'overlay_zero_outside', False))
                    )
                except Exception:
                    pass
        except Exception as e:
            msg = f"Integrated Gradients failed: {e}"
            print(msg)
            interpret_errors['saliency'].append(msg)

    # Fused Saliency (SmoothGrad + Integrated Gradients)
    if 'fused_saliency' in run_set:
        try:
            if not is_ensemble:
                fused_sal = compute_fused_saliency_volume(models[0], input_tensor, interpret_idx, 
                                                        sg_samples=int(args.sg_samples), 
                                                        ig_steps=int(args.ig_steps),
                                                        noise_std=float(args.sg_noise_std), 
                                                        baseline_type=args.ig_baseline)
            else:
                fused_accum = None
                for m in models:
                    fused_i = compute_fused_saliency_volume(m, input_tensor, interpret_idx, 
                                                          sg_samples=int(args.sg_samples), 
                                                          ig_steps=int(args.ig_steps),
                                                          noise_std=float(args.sg_noise_std), 
                                                          baseline_type=args.ig_baseline)
                    fused_accum = fused_i if fused_accum is None else (fused_accum + fused_i)
                fused_sal = fused_accum / float(len(models))
            if use_focus:
                fused_local = resize_volume_to_shape(fused_sal, vol_focus_n.shape)
                fused_full = _embed_focus_into_full(fused_local, focus_bbox, vol_np.shape)
            else:
                fused_full = resize_volume_to_shape(fused_sal, vol_np.shape)
            fused_full = robust_normalize_within_mask(fused_full, brain_mask_orig)
            fused_path = out_dir / f"{sid}_fused_saliency.nii.gz"
            save_nifti(fused_full, affine, header, fused_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        fused_full,
                        out_dir / f"{sid}_fused_saliency_overlay.png",
                        title=f"{sid} • Fused Saliency • Target={label_map.get(int(interpret_idx), str(interpret_idx))}",
                        mask=brain_mask_eroded,
                        alpha=float(getattr(args, 'overlay_alpha', 0.4)),
                        zero_outside=bool(getattr(args, 'overlay_zero_outside', False))
                    )
                except Exception:
                    pass
        except Exception as e:
            msg = f"Fused Saliency failed: {e}"
            print(msg)
            interpret_errors['saliency'].append(msg)

    # Occlusion sensitivity
    if 'occlusion' in run_set:
        try:
            if not is_ensemble:
                # Default to stride = ksize//2 if not provided
                stride_val = args.occ_stride if args.occ_stride is not None else max(1, int(args.occ_ksize)//2)
                occ_full = compute_occlusion_volume(
                    models[0], input_tensor, interpret_idx, ksize=int(args.occ_ksize), stride=stride_val, baseline=float(args.occ_baseline)
                )
            else:
                occ_accum = None
                for m in models:
                    stride_val = args.occ_stride if args.occ_stride is not None else max(1, int(args.occ_ksize)//2)
                    occ_i = compute_occlusion_volume(
                        m, input_tensor, interpret_idx, ksize=int(args.occ_ksize), stride=stride_val, baseline=float(args.occ_baseline)
                    )
                    occ_accum = occ_i if occ_accum is None else (occ_accum + occ_i)
                occ_full = occ_accum / float(len(models))
            if use_focus:
                occ_local = resize_volume_to_shape(occ_full, vol_focus_n.shape)
                occ_map = _embed_focus_into_full(occ_local, focus_bbox, vol_np.shape)
            else:
                occ_map = resize_volume_to_shape(occ_full, vol_np.shape)
            occ_map = robust_normalize_within_mask(occ_map, brain_mask_orig)
            occ_path = out_dir / f"{sid}_occlusion.nii.gz"
            save_nifti(occ_map, affine, header, occ_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        occ_map,
                        out_dir / f"{sid}_occlusion_overlay.png",
                        title=f"{sid} • Occlusion • Target={label_map.get(int(interpret_idx), str(interpret_idx))}",
                        mask=brain_mask_eroded,
                        alpha=float(getattr(args, 'overlay_alpha', 0.4)),
                        zero_outside=bool(getattr(args, 'overlay_zero_outside', False))
                    )
                except Exception:
                    pass
        except Exception as e:
            msg = f"Occlusion failed: {e}"
            print(msg)
            interpret_errors['occlusion'].append(msg)
            occ_path = None

    # GradientSHAP (if captum available)
    if 'gradientshap' in run_set:
        try:
            def _gshap_one(m):
                try:
                    from captum.attr import GradientShap  # type: ignore
                except Exception:
                    return None
                # Wrap model to ensure we return logits only (handles wrappers that return (logits, feats))
                class LogitOnly(nn.Module):
                    def __init__(self, base: nn.Module):
                        super().__init__()
                        self.base = base
                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        out = self.base(x)
                        return out[0] if isinstance(out, tuple) else out
                attr_model = LogitOnly(m).to(input_tensor.device)
                attr_model.eval()
                # Ensure inputs require gradients for Captum
                attr_input = input_tensor.clone().detach().requires_grad_(True)
                baseline = torch.zeros_like(attr_input)
                gs = GradientShap(attr_model)
                attributions = gs.attribute(
                    attr_input,
                    baselines=baseline,
                    target=interpret_idx,
                    n_samples=int(args.gshap_samples),
                    stdevs=float(args.gshap_std),
                )
                attr = attributions.detach().squeeze(0).squeeze(0).cpu().numpy()
                attr = np.abs(attr).astype(np.float32)
                return attr

            if not is_ensemble:
                gattr = _gshap_one(models[0])
            else:
                g_list = []
                for m in models:
                    try:
                        g = _gshap_one(m)
                        if g is not None:
                            g_list.append(g)
                        else:
                            interpret_errors['gradientshap'].append("Captum not available or initialisation failed for one model.")
                    except Exception as _e:
                        # Skip models that fail attribution but continue with others
                        warn = f"GradientSHAP warning (one model skipped): {_e}"
                        print(warn)
                        interpret_errors['gradientshap'].append(warn)
                gattr = None if len(g_list) == 0 else np.mean(np.stack(g_list, axis=0), axis=0)

            if gattr is not None:
                if use_focus:
                    g_local = resize_volume_to_shape(gattr, vol_focus_n.shape)
                    g_full = _embed_focus_into_full(g_local, focus_bbox, vol_np.shape)
                else:
                    g_full = resize_volume_to_shape(gattr, vol_np.shape)
                g_full = robust_normalize_within_mask(g_full, brain_mask_orig)
                gshap_path = out_dir / f"{sid}_gradientshap.nii.gz"
                save_nifti(g_full, affine, header, gshap_path)
            else:
                gshap_path = None
        except Exception as e:
            msg = f"GradientSHAP failed: {e}"
            print(msg)
            interpret_errors['gradientshap'].append(msg)
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
            'risk_weights_applied': applied_risk,
            'cn_min_prob': float(args.cn_min_prob) if args.cn_min_prob is not None else None,
        },
        'known_label': {
            'label_index': int(known_label) if known_label is not None else None,
            'label_name': label_map.get(int(known_label), str(known_label)) if known_label is not None else None,
            'interpretability_target_index': int(interpret_idx),
            'interpretability_target_name': label_map.get(int(interpret_idx), str(interpret_idx)),
        },
        'prediction_raw': {
            'label_index': int(np.argmax(list(prob_dict_raw.values()))),
            'label_name': pred_name_raw,
            'confidence': confidence_raw,
            'probabilities': prob_dict_raw,
            'tta': meta_pred,
        },
        'interpretability': {
            'gradcam': gradcam_paths,
            'saliency': str(sal_path) if sal_path else None,
            'occlusion': str(occ_path) if occ_path else None,
            'gradientshap': str(gshap_path) if gshap_path else None,
        },
        'interpretability_errors': interpret_errors,
    }

    json_path = out_dir / f"{sid}_clinical_prediction_deep.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Prediction: {pred_name} (conf {confidence:.3f})")
    print(f"✓ JSON report: {json_path}")

    # Also export a simple text error log if any errors occurred
    if any(len(v) > 0 for v in interpret_errors.values()):
        log_path = out_dir / f"{sid}_interpretability_errors.log"
        try:
            with open(log_path, 'w') as lf:
                for k, msgs in interpret_errors.items():
                    lf.write(f"[{k}]\n")
                    if msgs:
                        for msg in msgs:
                            lf.write(f"- {msg}\n")
                    else:
                        lf.write("- none\n")
                    lf.write("\n")
            print(f"✓ Error log: {log_path}")
        except Exception as e:
            print(f"Warning: failed to write error log: {e}")


if __name__ == '__main__':
    main()


