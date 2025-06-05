# scripts/gradcam.py

import torch
import torch.nn.functional as F

def compute_gradcam_3d(model, smri_tensor, target_class, device="cuda"):
    """
    Compute a 3D Grad-CAM heatmap for a single sMRI input.

    Args:
      model         : an instance of SMRI_GradCAM_3DCNN, already loaded with weights.
      smri_tensor   : torch.Tensor of shape [1, 1, D, H, W], single‐subject volume.
      target_class  : integer (0 or 1), the class index you want to visualize.
      device        : "cuda" or "cpu"

    Returns:
      cam_norm (np.ndarray) : 3D array [D, H, W], normalized to [0,1].
    """
    model.to(device)
    model.eval()

    fmap_grad = None

    # 1) Register a backward hook on the final conv block (model.features) to capture gradients
    def save_grad(module, grad_input, grad_output):
        nonlocal fmap_grad
        fmap_grad = grad_output[0].detach()  # shape: [1, C, d, h, w]

    handle_bw = model.features.register_backward_hook(save_grad)

    # 2) Forward pass
    smri_tensor = smri_tensor.to(device).requires_grad_(True)
    logits, fmap = model(smri_tensor)   # fmap shape: [1, C, d, h, w]
    probs = F.softmax(logits, dim=1)
    score = probs[0, target_class]      # scalar

    # 3) Backpropagate the class score
    model.zero_grad()
    score.backward(retain_graph=False)

    # 4) Grab gradients and activations
    grads = fmap_grad                  # [1, C, d, h, w]
    activations = fmap.detach()        # [1, C, d, h, w]

    # 5) Compute channel‐wise weights: global average over (d, h, w)
    weights = torch.mean(grads, dim=(2, 3, 4)).squeeze(0)  # [C]

    # 6) Weighted sum of feature maps → coarse CAM
    cam = torch.zeros_like(activations[0, 0])  # [d, h, w]
    for i, w in enumerate(weights):
        cam += w * activations[0, i]          # weighted combination

    # 7) ReLU and upsample to original volume size
    cam = F.relu(cam)                           # [d, h, w]
    cam = cam.unsqueeze(0).unsqueeze(0)         # [1, 1, d, h, w]
    target_size = smri_tensor.shape[-3:]        # (D, H, W)
    cam_upsampled = F.interpolate(cam,
                                  size=target_size,
                                  mode="trilinear",
                                  align_corners=False)  # [1,1,D,H,W]
    cam_upsampled = cam_upsampled.squeeze().cpu().numpy()  # [D, H, W]

    # 8) Normalize to [0,1]
    cam_min = cam_upsampled.min()
    cam_max = cam_upsampled.max()
    cam_norm = (cam_upsampled - cam_min) / (cam_max - cam_min + 1e-8)

    # 9) Remove hook
    handle_bw.remove()

    return cam_norm
