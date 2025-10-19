# scripts/gradcam.py

import torch
import torch.nn.functional as F

def compute_gradcam_3d(model, smri_tensor, target_class, device="cuda"):
    """
    Compute a 3D Grad-CAM heatmap for a single sMRI input.

    Args:
      model         : an instance of SMRI_GradCAM_3DCNN, already loaded with weights.
      smri_tensor   : torch.Tensor of shape [1, 1, D, H, W], single‐subject volume.
      target_class  : integer (0 or 1), the class index you want to visualise.
      device        : "cuda" or "cpu"

    Returns:
      cam_norm (np.ndarray) : 3D array [D, H, W], normalized to [0,1].
      
    Authors:
      Joseph Storey
      Jackson Schofield
    """
    model.to(device)
    model.eval()

    fmap_grad = None

    # 1) Register a backward hook on the final conv block (model.features) to capture gradients
    def save_grad(module, grad_input, grad_output):
        nonlocal fmap_grad
        fmap_grad = grad_output[0].detach()  # shape: [1, C, d, h, w]

    handle_bw = model.features.register_full_backward_hook(save_grad)

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

def compute_gradcam_simple3d(model, smri_tensor, target_class, device="cuda"):
    """
    Compute a 3D Grad-CAM heatmap for Simple3DCNN architecture.
    
    Args:
      model         : Wrapped Simple3DCNN model with exposed features
      smri_tensor   : torch.Tensor of shape [1, 1, D, H, W], single‐subject volume.
      target_class  : integer (0 or 1), the class index you want to visualise.
      device        : "cuda" or "cpu"

    Returns:
      cam_norm (np.ndarray) : 3D array [D, H, W], normalized to [0,1].
    """
    model.to(device)
    model.eval()

    fmap_grad = None

    # 1) Register a backward hook on the features to capture gradients
    def save_grad(module, grad_input, grad_output):
        nonlocal fmap_grad
        fmap_grad = grad_output[0].detach()

    handle_bw = model.features.register_full_backward_hook(save_grad)

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


def compute_gradcam_swinunetr(model, smri_tensor, target_class, device="cuda"):
    """
    Compute a 3D Grad-CAM heatmap for SwinUNETR architecture.
    
    Args:
      model         : SwinUNETRClassifierGradCAM model
      smri_tensor   : torch.Tensor of shape [1, 1, D, H, W], single‐subject volume.
      target_class  : integer (0 or 1), the class index you want to visualise.
      device        : "cuda" or "cpu"

    Returns:
      cam_norm (np.ndarray) : 3D array [D, H, W], normalized to [0,1].
    """
    model.to(device)
    model.eval()

    fmap_grad = None

    # 1) Register a backward hook on the encoder to capture gradients
    def save_grad(module, grad_input, grad_output):
        nonlocal fmap_grad
        # For SwinUNETR, we need to capture gradients from the deepest layer
        if isinstance(grad_output, tuple):
            fmap_grad = grad_output[0].detach()
        else:
            fmap_grad = grad_output.detach()

    # Register hook on the encoder's output
    handle_bw = model.encoder.register_full_backward_hook(save_grad)

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


def compute_gradcam_visiontransformer(model, smri_tensor, target_class, device="cuda"):
    """
    Compute a 3D Grad-CAM heatmap for Vision Transformer architecture.
    Note: This is a simplified version that uses the last transformer block's output.
    
    Args:
      model         : VisionTransformer3D model
      smri_tensor   : torch.Tensor of shape [1, 1, D, H, W], single‐subject volume.
      target_class  : integer (0 or 1), the class index you want to visualise.
      device        : "cuda" or "cpu"

    Returns:
      cam_norm (np.ndarray) : 3D array [D, H, W], normalized to [0,1].
    """
    model.to(device)
    model.eval()

    # For Vision Transformer, we'll use attention weights from the last layer
    # This is a simplified approach - in practice, you might want to use attention rollout
    
    # 1) Forward pass to get attention weights
    smri_tensor = smri_tensor.to(device).requires_grad_(True)
    
    # Get patch embeddings
    B = smri_tensor.shape[0]
    x = model.patch_embed(smri_tensor)  # [B, embed_dim, D//patch_size, H//patch_size, W//patch_size]
    x = x.flatten(2).transpose(1, 2)    # [B, num_patches, embed_dim]
    
    # Add class token
    cls_tokens = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.dropout(x)
    
    # Get attention weights from the last transformer block
    for i, block in enumerate(model.blocks):
        if i == len(model.blocks) - 1:  # Last block
            # Get attention weights
            attn_weights = block.attn(F.normalize(block.norm1(x), dim=-1))
            break
        x = block(x)
    
    # 2) Create attention map
    # Use attention weights from class token to patches
    cls_attention = attn_weights[0, :, 0, 1:]  # [num_heads, num_patches]
    cls_attention = cls_attention.mean(dim=0)   # [num_patches]
    
    # 3) Reshape to 3D
    patch_size = model.patch_size
    img_size = model.img_size
    num_patches_per_dim = [img_size[i] // patch_size for i in range(3)]
    
    attention_3d = cls_attention.reshape(num_patches_per_dim[0], 
                                        num_patches_per_dim[1], 
                                        num_patches_per_dim[2])
    
    # 4) Upsample to original size
    attention_3d = attention_3d.unsqueeze(0).unsqueeze(0)  # [1, 1, d, h, w]
    target_size = smri_tensor.shape[-3:]  # (D, H, W)
    cam_upsampled = F.interpolate(attention_3d,
                                  size=target_size,
                                  mode="trilinear",
                                  align_corners=False)  # [1,1,D,H,W]
    cam_upsampled = cam_upsampled.squeeze().cpu().numpy()  # [D, H, W]

    # 5) Normalize to [0,1]
    cam_min = cam_upsampled.min()
    cam_max = cam_upsampled.max()
    cam_norm = (cam_upsampled - cam_min) / (cam_max - cam_min + 1e-8)

    return cam_norm
