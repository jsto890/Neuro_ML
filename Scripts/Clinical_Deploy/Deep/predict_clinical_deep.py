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
                          use_amp: bool = False, device_str: str = 'cpu'
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

    meta = {
        'tta_replicates': replicates,
        'votes': votes,
        'vote_hist': {int(i): int(votes.count(i)) for i in set(votes)}
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


def save_overlay_pngs(anat: np.ndarray, heat: np.ndarray, out_path: Path, title: str = "", mask: Optional[np.ndarray] = None):
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
    D, H, W = anat.shape
    za, ya, xa = D // 2, H // 2, W // 2
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(anat_n[za].T, cmap='gray', origin='lower', interpolation='nearest', resample=False)
    axs[0].imshow(heat[za].T, cmap='hot', alpha=0.4, origin='lower', interpolation='nearest', resample=False)
    axs[0].set_title('Axial')
    axs[0].axis('off')
    axs[1].imshow(anat_n[:, ya, :].T, cmap='gray', origin='lower', interpolation='nearest', resample=False)
    axs[1].imshow(heat[:, ya, :].T, cmap='hot', alpha=0.4, origin='lower', interpolation='nearest', resample=False)
    axs[1].set_title('Coronal')
    axs[1].axis('off')
    axs[2].imshow(anat_n[:, :, xa].T, cmap='gray', origin='lower', interpolation='nearest', resample=False)
    axs[2].imshow(heat[:, :, xa].T, cmap='hot', alpha=0.4, origin='lower', interpolation='nearest', resample=False)
    axs[2].set_title('Sagittal')
    axs[2].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)


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
    # Interpretability selection
    parser.add_argument('--run-all', action='store_true', help='Run all interpretability maps (default if neither --run nor --run-all provided)')
    parser.add_argument('--run', nargs='+', choices=['gradcam', 'gradcam_plusplus', 'saliency', 'smoothgrad', 'integrated_gradients', 'fused_saliency', 'occlusion', 'gradientshap'], help='One or more interpretability methods to run, e.g. --run gradcam smoothgrad fused_saliency')
    # Advanced saliency parameters
    parser.add_argument('--sg-samples', type=int, default=40, help='SmoothGrad samples (40=fast, 80=HQ)')
    parser.add_argument('--sg-noise-std', type=float, default=0.1, help='SmoothGrad noise std (fraction of in-mask intensity std)')
    parser.add_argument('--ig-steps', type=int, default=64, help='Integrated Gradients steps (64=fast, 128=HQ)')
    parser.add_argument('--ig-baseline', type=str, default='zeros', choices=['zeros', 'mean'], help='Integrated Gradients baseline type')
    # Probability shaping
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature for logits before risk adjustment (T<1 sharpens, T>1 flattens)')

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
    vol_np = normalize_volume(vol_np, method=args.normalize)
    input_tensor = to_model_tensor(vol_np, device=device, resize_dims=tuple(args.resize_dims) if args.resize_dims else None)

    # Predict with optional ensemble + TTA
    if not is_ensemble and int(args.tta_n) <= 0:
        pred_idx, probs, logits = predict(models[0], input_tensor)
        meta_pred = {'tta_replicates': 1, 'votes': [pred_idx], 'vote_hist': {pred_idx: 1}}
    else:
        probs, logits, meta_pred = aggregate_predictions(
            models, input_tensor, avg_method=args.ensemble_avg_method,
            tta_n=int(args.tta_n), tta_noise_std=float(args.tta_noise_std), mc_dropout=bool(args.mc_dropout),
            use_amp=bool(args.amp), device_str=device
        )
        pred_idx = int(np.argmax(probs))

    # Optional temperature scaling on logits to sharpen/flatten probabilities
    try:
        T = float(args.temperature)
    except Exception:
        T = 1.0
    if T != 1.0:
        try:
            if isinstance(logits, torch.Tensor):
                with torch.no_grad():
                    probs_t = F.softmax(logits / T, dim=1)
                probs = probs_t.squeeze(0).cpu().numpy()
                pred_idx = int(np.argmax(probs))
            else:
                # Fallback: power transform on probs
                p = np.asarray(probs, dtype=np.float32)
                gamma = 1.0 / max(T, 1e-8)
                p = np.power(np.maximum(p, 1e-12), gamma)
                p = p / np.sum(p)
                probs = p
                pred_idx = int(np.argmax(probs))
        except Exception:
            pass

    # Raw prediction snapshot
    pred_name_raw = label_map.get(pred_idx, str(pred_idx))
    confidence_raw = float(np.max(probs))
    prob_dict_raw = {label_map.get(i, str(i)): float(p) for i, p in enumerate(probs)}

    # Risk-aware adjustment
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
        classes_to_compute = range(args.num_classes) if args.all_classes_interpret else [pred_idx]
        overlay_saved = False
        for c in classes_to_compute:
            try:
                if not is_ensemble:
                    cam = compute_gradcam_volume(gradcam_module, models[0], input_tensor, c, args.model_arch, device, cam_layer=args.cam_layer)
                else:
                    cam_accum = None
                    for m in models:
                        cam_i = compute_gradcam_volume(gradcam_module, m, input_tensor, c, args.model_arch, device, cam_layer=args.cam_layer)
                        cam_accum = cam_i if cam_accum is None else (cam_accum + cam_i)
                    cam = cam_accum / float(len(models))
                # Resize CAM back to the anatomy's original shape for NIfTI and overlays
                cam_resized = resize_volume_to_shape(cam, vol_np.shape)
                cam_path = out_dir / f"{sid}_gradcam_class{c}.nii.gz"
                save_nifti(cam_resized, affine, header, cam_path)
                gradcam_paths[str(c)] = str(cam_path)
                # Optional overlays for predicted class
                if args.save_overlay_pngs and c == pred_idx and not overlay_saved:
                    try:
                        class_name = label_map.get(int(c), str(c))
                        save_overlay_pngs(
                            vol_np,
                            cam_resized,
                            out_dir / f"{sid}_gradcam_overlay.png",
                            title=f"{sid} • Grad-CAM • {class_name}",
                            mask=brain_mask_orig
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
        classes_to_compute = range(args.num_classes) if args.all_classes_interpret else [pred_idx]
        overlay_saved_pp = False
        for c in classes_to_compute:
            try:
                if not is_ensemble:
                    cam = compute_gradcam_plusplus_volume(gradcam_module, models[0], input_tensor, c, args.model_arch, device, cam_layer=args.cam_layer)
                else:
                    cam_accum = None
                    for m in models:
                        cam_i = compute_gradcam_plusplus_volume(gradcam_module, m, input_tensor, c, args.model_arch, device, cam_layer=args.cam_layer)
                        cam_accum = cam_i if cam_accum is None else (cam_accum + cam_i)
                    cam = cam_accum / float(len(models))
                # Resize CAM back to the anatomy's original shape for NIfTI and overlays
                cam_resized = resize_volume_to_shape(cam, vol_np.shape)
                cam_path = out_dir / f"{sid}_gradcam_plusplus_class{c}.nii.gz"
                save_nifti(cam_resized, affine, header, cam_path)
                gradcam_paths[f"plusplus_{c}"] = str(cam_path)
                # Optional overlays for predicted class
                if args.save_overlay_pngs and c == pred_idx and not overlay_saved_pp:
                    try:
                        class_name = label_map.get(int(c), str(c))
                        save_overlay_pngs(
                            vol_np,
                            cam_resized,
                            out_dir / f"{sid}_gradcam_plusplus_overlay.png",
                            title=f"{sid} • Grad-CAM++ • {class_name}",
                            mask=brain_mask_orig
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
                sal = compute_saliency_volume(models[0], input_tensor, pred_idx)
            else:
                sal_accum = None
                for m in models:
                    sal_i = compute_saliency_volume(m, input_tensor, pred_idx)
                    sal_accum = sal_i if sal_accum is None else (sal_accum + sal_i)
                sal = sal_accum / float(len(models))
            sal_resized = resize_volume_to_shape(sal, vol_np.shape)
            # Do NOT remove out-of-brain values; normalize using in-brain voxels only
            sal_resized = robust_normalize_within_mask(sal_resized, brain_mask_orig)
            sal_path = out_dir / f"{sid}_saliency.nii.gz"
            save_nifti(sal_resized, affine, header, sal_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        sal_resized,
                        out_dir / f"{sid}_saliency_overlay.png",
                        title=f"{sid} • Saliency • Pred={pred_name}",
                        mask=brain_mask_orig
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
                sg_sal = compute_smoothgrad_volume(models[0], input_tensor, pred_idx, 
                                                 n_samples=int(args.sg_samples), 
                                                 noise_std=float(args.sg_noise_std))
            else:
                sg_accum = None
                for m in models:
                    sg_i = compute_smoothgrad_volume(m, input_tensor, pred_idx, 
                                                   n_samples=int(args.sg_samples), 
                                                   noise_std=float(args.sg_noise_std))
                    sg_accum = sg_i if sg_accum is None else (sg_accum + sg_i)
                sg_sal = sg_accum / float(len(models))
            sg_resized = resize_volume_to_shape(sg_sal, vol_np.shape)
            sg_resized = robust_normalize_within_mask(sg_resized, brain_mask_orig)
            sg_path = out_dir / f"{sid}_smoothgrad.nii.gz"
            save_nifti(sg_resized, affine, header, sg_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        sg_resized,
                        out_dir / f"{sid}_smoothgrad_overlay.png",
                        title=f"{sid} • SmoothGrad • Pred={pred_name}",
                        mask=brain_mask_orig
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
                ig_sal = compute_integrated_gradients_volume(models[0], input_tensor, pred_idx, 
                                                           n_steps=int(args.ig_steps), 
                                                           baseline_type=args.ig_baseline)
            else:
                ig_accum = None
                for m in models:
                    ig_i = compute_integrated_gradients_volume(m, input_tensor, pred_idx, 
                                                             n_steps=int(args.ig_steps), 
                                                             baseline_type=args.ig_baseline)
                    ig_accum = ig_i if ig_accum is None else (ig_accum + ig_i)
                ig_sal = ig_accum / float(len(models))
            ig_resized = resize_volume_to_shape(ig_sal, vol_np.shape)
            ig_resized = robust_normalize_within_mask(ig_resized, brain_mask_orig)
            ig_path = out_dir / f"{sid}_integrated_gradients.nii.gz"
            save_nifti(ig_resized, affine, header, ig_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        ig_resized,
                        out_dir / f"{sid}_integrated_gradients_overlay.png",
                        title=f"{sid} • Integrated Gradients • Pred={pred_name}",
                        mask=brain_mask_orig
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
                fused_sal = compute_fused_saliency_volume(models[0], input_tensor, pred_idx, 
                                                        sg_samples=int(args.sg_samples), 
                                                        ig_steps=int(args.ig_steps),
                                                        noise_std=float(args.sg_noise_std), 
                                                        baseline_type=args.ig_baseline)
            else:
                fused_accum = None
                for m in models:
                    fused_i = compute_fused_saliency_volume(m, input_tensor, pred_idx, 
                                                          sg_samples=int(args.sg_samples), 
                                                          ig_steps=int(args.ig_steps),
                                                          noise_std=float(args.sg_noise_std), 
                                                          baseline_type=args.ig_baseline)
                    fused_accum = fused_i if fused_accum is None else (fused_accum + fused_i)
                fused_sal = fused_accum / float(len(models))
            fused_resized = resize_volume_to_shape(fused_sal, vol_np.shape)
            fused_resized = robust_normalize_within_mask(fused_resized, brain_mask_orig)
            fused_path = out_dir / f"{sid}_fused_saliency.nii.gz"
            save_nifti(fused_resized, affine, header, fused_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        fused_resized,
                        out_dir / f"{sid}_fused_saliency_overlay.png",
                        title=f"{sid} • Fused Saliency • Pred={pred_name}",
                        mask=brain_mask_orig
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
                    models[0], input_tensor, pred_idx, ksize=int(args.occ_ksize), stride=stride_val, baseline=float(args.occ_baseline)
                )
            else:
                occ_accum = None
                for m in models:
                    stride_val = args.occ_stride if args.occ_stride is not None else max(1, int(args.occ_ksize)//2)
                    occ_i = compute_occlusion_volume(
                        m, input_tensor, pred_idx, ksize=int(args.occ_ksize), stride=stride_val, baseline=float(args.occ_baseline)
                    )
                    occ_accum = occ_i if occ_accum is None else (occ_accum + occ_i)
                occ_full = occ_accum / float(len(models))
            occ_resized = resize_volume_to_shape(occ_full, vol_np.shape)
            # Keep out-of-brain values but normalize contrast based on brain voxels only
            occ_resized = robust_normalize_within_mask(occ_resized, brain_mask_orig)
            occ_path = out_dir / f"{sid}_occlusion.nii.gz"
            save_nifti(occ_resized, affine, header, occ_path)
            if args.save_overlay_pngs:
                try:
                    save_overlay_pngs(
                        vol_np,
                        occ_resized,
                        out_dir / f"{sid}_occlusion_overlay.png",
                        title=f"{sid} • Occlusion • Pred={pred_name}",
                        mask=brain_mask_orig
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
                    target=pred_idx,
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
                            interpret_errors['gradientshap'].append("Captum not available or initialization failed for one model.")
                    except Exception as _e:
                        # Skip models that fail attribution but continue with others
                        warn = f"GradientSHAP warning (one model skipped): {_e}"
                        print(warn)
                        interpret_errors['gradientshap'].append(warn)
                gattr = None if len(g_list) == 0 else np.mean(np.stack(g_list, axis=0), axis=0)

            if gattr is not None:
                gattr_resized = resize_volume_to_shape(gattr, vol_np.shape)
                gattr_resized = robust_normalize_within_mask(gattr_resized, brain_mask_orig)
                gshap_path = out_dir / f"{sid}_gradientshap.nii.gz"
                save_nifti(gattr_resized, affine, header, gshap_path)
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
        'temperature': float(T),
        'prediction': {
            'label_index': pred_idx,
            'label_name': pred_name,
            'confidence': confidence,
            'probabilities': prob_dict,
            'risk_weights_applied': applied_risk,
            'cn_min_prob': float(args.cn_min_prob) if args.cn_min_prob is not None else None,
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

    # Probability visualization (bar chart)
    try:
        labels = list(prob_dict.keys())
        values = [prob_dict[k] for k in labels]
        # Normalize values to sum 1 for display if risk weights modified
        s = sum(values)
        disp_vals = [v / s if s > 0 else 0.0 for v in values]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, disp_vals, color=['#2ca02c' if l == 'CN' else '#1f77b4' if l == 'AD' else '#d62728' for l in labels])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel('Probability')
        ax.set_title(f"{sid} • Prediction: {pred_name} ({confidence:.0%})")
        for rect, v in zip(bars, disp_vals):
            ax.annotate(f"{v*100:.0f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        prob_png = out_dir / f"{sid}_probabilities.png"
        plt.savefig(str(prob_png), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Probabilities plot: {prob_png}")
    except Exception as _e:
        print(f"Warning: failed to save probabilities plot: {_e}")

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


