# transformer_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# Import transformer-specific libraries
try:
    from monai.networks.nets import SwinUNETR
    from monai.networks.blocks import PatchEmbed, UnetOutBlock
    from monai.networks.nets import ViT
except ImportError:
    SwinUNETR = None
    ViT = None

try:
    import einops
except ImportError:
    einops = None

class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer for medical image classification.
    Adapted from ViT architecture for 3D volumes.
    
    Input:  [B, 1, D, H, W]  single-channel PET
    Output: [B, num_classes] logits
    """
    
    def __init__(self, 
                 img_size: Tuple[int, int, int] = (97, 115, 97),
                 patch_size: int = 16,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: Optional[nn.Module] = None,
                 act_layer: Optional[nn.Module] = None):
        
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.patch_dim = (patch_size ** 3) * in_channels
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer or nn.LayerNorm,
                act_layer=act_layer or nn.GELU
            ) for _ in range(depth)
        ])
        
        # Classification head
        self.norm = (norm_layer or nn.LayerNorm)(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape [B, 1, D, H, W]
        Returns:
            logits: Classification logits of shape [B, num_classes]
        """
        B = x.shape[0]
        # Crop or pad to [96, 112, 96] as needed
        target_shape = (96, 112, 96)
        current_shape = x.shape[-3:]
        # Center-crop if any dimension is too large
        slices = []
        for curr, tgt in zip(current_shape, target_shape):
            if curr > tgt:
                start = (curr - tgt) // 2
                end = start + tgt
                slices.append(slice(start, end))
            else:
                slices.append(slice(0, curr))
        x = x[..., slices[0], slices[1], slices[2]]
        # Pad if any dimension is too small
        new_shape = x.shape[-3:]
        pad = []
        for curr, tgt in zip(reversed(new_shape), reversed(target_shape)):
            total = max(tgt - curr, 0)
            pad.extend([total // 2, total - total // 2])
        if any(pad):
            x = F.pad(x, pad)
        # Patch embedding: [B, 1, D, H, W] -> [B, embed_dim, D//patch_size, H//patch_size, W//patch_size]
        x = self.patch_embed(x)
        # Flatten spatial dimensions: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        # Classification
        x = self.norm(x)
        cls_token = x[:, 0]  # Take the class token
        logits = self.head(cls_token)
        return logits


class TransformerBlock3D(nn.Module):
    """
    3D Transformer block with multi-head self-attention and MLP.
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention3D(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiHeadAttention3D(nn.Module):
    """
    Multi-head self-attention for 3D sequences.
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    MLP block for transformer.
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SwinUNETRClassifier(nn.Module):
    """
    Swin UNETR adapted for classification tasks.
    Uses the encoder part of Swin UNETR with a classification head.
    
    Input:  [B, 1, D, H, W]  single-channel PET
    Output: [B, num_classes] logits
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 2,
                 feature_size: int = 36,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3):
        
        super().__init__()
        
        if SwinUNETR is None:
            raise ImportError("MONAI is required for SwinUNETR. Install with 'pip install monai'.")
        
        # Create Swin UNETR model
        self.swin_unetr = SwinUNETR(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,  # We'll override this
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint
        )
        
        # Remove the decoder and output blocks, keep only encoder
        self.encoder = self.swin_unetr.swinViT
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_size * 8, 512),  # feature_size * 8 from deepest layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self._initialized = False
    
    def _initialize_classifier(self, x):
        """Initialize classifier with correct input size."""
        with torch.no_grad():
            # Get features from encoder
            hidden_states = self.encoder(x)
            # Use the deepest layer for classification
            deepest_features = hidden_states[-1]  # [B, C, D, H, W]
            pooled = self.global_pool(deepest_features)  # [B, C, 1, 1, 1]
            flattened = pooled.view(pooled.size(0), -1)  # [B, C]
            n_features = flattened.size(1)
        
        # Update classifier input size
        device = x.device
        self.classifier[0] = nn.Linear(n_features, 512).to(device)
        for layer in self.classifier:
            layer.to(device)
        
        self._initialized = True
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape [B, 1, D, H, W]
        Returns:
            logits: Classification logits of shape [B, num_classes]
        """
        if not self._initialized:
            self._initialize_classifier(x)
        
        # Get features from encoder
        hidden_states = self.encoder(x)
        
        # Use the deepest layer for classification
        deepest_features = hidden_states[-1]  # [B, C, D, H, W]
        
        # Global average pooling
        pooled = self.global_pool(deepest_features)  # [B, C, 1, 1, 1]
        flattened = pooled.view(pooled.size(0), -1)  # [B, C]
        
        # Classification
        logits = self.classifier(flattened)
        
        return logits


class SwinUNETRClassifierGradCAM(SwinUNETRClassifier):
    """
    Swin UNETR Classifier with Grad-CAM support.
    Returns both logits and feature maps for visualization.
    """
    
    def forward(self, x):
        """
        Forward pass with feature map output for Grad-CAM.
        Args:
            x: Input tensor of shape [B, 1, D, H, W]
        Returns:
            tuple: (logits, feature_maps) where feature_maps is [B, C, D, H, W]
        """
        if not self._initialized:
            self._initialize_classifier(x)
        
        # Get features from encoder
        hidden_states = self.encoder(x)
        
        # Use the deepest layer for classification
        deepest_features = hidden_states[-1]  # [B, C, D, H, W]
        
        # Global average pooling
        pooled = self.global_pool(deepest_features)  # [B, C, 1, 1, 1]
        flattened = pooled.view(pooled.size(0), -1)  # [B, C]
        
        # Classification
        logits = self.classifier(flattened)
        
        return logits, deepest_features


class FullSwinUNETRClassifier(nn.Module):
    """
    Full Swin UNETR adapted for classification tasks.
    Uses the complete Swin UNETR architecture (encoder + decoder) with a classification head.
    
    Input:  [B, 1, D, H, W]  single-channel PET
    Output: [B, num_classes] logits
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 2,
                 feature_size: int = 36,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3):
        
        super().__init__()
        
        if SwinUNETR is None:
            raise ImportError("MONAI is required for SwinUNETR. Install with 'pip install monai'.")
        
        # Create full Swin UNETR model
        self.swin_unetr = SwinUNETR(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size * 8,  # Use all features from decoder
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_size * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self._initialized = False
    
    def _initialize_classifier(self, x):
        """Initialize classifier with correct input size."""
        with torch.no_grad():
            # Preprocess input to meet Swin UNETR requirements (divisible by 32)
            # Target size: [96, 128, 96] - crop 97->96, pad 115->128
            target_shape = (96, 128, 96)
            current_shape = x.shape[-3:]
            
            # Center-crop if any dimension is too large
            slices = []
            for curr, tgt in zip(current_shape, target_shape):
                if curr > tgt:
                    start = (curr - tgt) // 2
                    end = start + tgt
                    slices.append(slice(start, end))
                else:
                    slices.append(slice(0, curr))
            
            x_processed = x[..., slices[0], slices[1], slices[2]]
            
            # Pad if any dimension is too small
            new_shape = x_processed.shape[-3:]
            pad = []
            for curr, tgt in zip(reversed(new_shape), reversed(target_shape)):
                total = max(tgt - curr, 0)
                pad.extend([total // 2, total - total // 2])
            
            if any(pad):
                x_processed = F.pad(x_processed, pad)
            
            # Get full output from Swin UNETR
            full_output = self.swin_unetr(x_processed)  # [B, C, D, H, W]
            pooled = self.global_pool(full_output)  # [B, C, 1, 1, 1]
            flattened = pooled.view(pooled.size(0), -1)  # [B, C]
            n_features = flattened.size(1)
        
        # Update classifier input size
        device = x.device
        self.classifier[0] = nn.Linear(n_features, 512).to(device)
        for layer in self.classifier:
            layer.to(device)
        
        self._initialized = True
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape [B, 1, D, H, W]
        Returns:
            logits: Classification logits of shape [B, num_classes]
        """
        if not self._initialized:
            self._initialize_classifier(x)
        
        # Preprocess input to meet Swin UNETR requirements (divisible by 32)
        # Target size: [96, 128, 96] - crop 97->96, pad 115->128
        target_shape = (96, 128, 96)
        current_shape = x.shape[-3:]
        
        # Center-crop if any dimension is too large
        slices = []
        for curr, tgt in zip(current_shape, target_shape):
            if curr > tgt:
                start = (curr - tgt) // 2
                end = start + tgt
                slices.append(slice(start, end))
            else:
                slices.append(slice(0, curr))
        
        x = x[..., slices[0], slices[1], slices[2]]
        
        # Pad if any dimension is too small
        new_shape = x.shape[-3:]
        pad = []
        for curr, tgt in zip(reversed(new_shape), reversed(target_shape)):
            total = max(tgt - curr, 0)
            pad.extend([total // 2, total - total // 2])
        
        if any(pad):
            x = F.pad(x, pad)
        
        # Get full output from Swin UNETR
        full_output = self.swin_unetr(x)  # [B, C, D, H, W]
        
        # Global average pooling
        pooled = self.global_pool(full_output)  # [B, C, 1, 1, 1]
        flattened = pooled.view(pooled.size(0), -1)  # [B, C]
        
        # Classification
        logits = self.classifier(flattened)
        
        return logits


class FullSwinUNETRClassifierGradCAM(FullSwinUNETRClassifier):
    """
    Full Swin UNETR Classifier with Grad-CAM support.
    Returns both logits and feature maps for visualization.
    """
    
    def forward(self, x):
        """
        Forward pass with feature map output for Grad-CAM.
        Args:
            x: Input tensor of shape [B, 1, D, H, W]
        Returns:
            tuple: (logits, feature_maps) where feature_maps is [B, C, D, H, W]
        """
        if not self._initialized:
            self._initialize_classifier(x)
        
        # Preprocess input to meet Swin UNETR requirements (divisible by 32)
        # Target size: [96, 128, 96] - crop 97->96, pad 115->128
        target_shape = (96, 128, 96)
        current_shape = x.shape[-3:]
        
        # Center-crop if any dimension is too large
        slices = []
        for curr, tgt in zip(current_shape, target_shape):
            if curr > tgt:
                start = (curr - tgt) // 2
                end = start + tgt
                slices.append(slice(start, end))
            else:
                slices.append(slice(0, curr))
        
        x = x[..., slices[0], slices[1], slices[2]]
        
        # Pad if any dimension is too small
        new_shape = x.shape[-3:]
        pad = []
        for curr, tgt in zip(reversed(new_shape), reversed(target_shape)):
            total = max(tgt - curr, 0)
            pad.extend([total // 2, total - total // 2])
        
        if any(pad):
            x = F.pad(x, pad)
        
        # Get full output from Swin UNETR
        full_output = self.swin_unetr(x)  # [B, C, D, H, W]
        
        # Global average pooling
        pooled = self.global_pool(full_output)  # [B, C, 1, 1, 1]
        flattened = pooled.view(pooled.size(0), -1)  # [B, C]
        
        # Classification
        logits = self.classifier(flattened)
        
        return logits, full_output


# Model factory function for transformer models
def get_transformer_model(model_name, num_classes=2, in_channels=1, **kwargs):
    """
    Returns a transformer model instance by name.
    Supported: 'VisionTransformer3D', 'SwinUNETRClassifier', 'FullSwinUNETRClassifier'
    """
    model_name = model_name.lower()
    
    if model_name == "visiontransformer3d":
        return VisionTransformer3D(
            num_classes=num_classes,
            in_channels=in_channels,
            **kwargs
        )
    elif model_name == "swinunetrclassifier":
        return SwinUNETRClassifier(
            num_classes=num_classes,
            in_channels=in_channels,
            **kwargs
        )
    elif model_name == "fullswinunetrclassifier":
        return FullSwinUNETRClassifier(
            num_classes=num_classes,
            in_channels=in_channels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown transformer model name: {model_name}") 