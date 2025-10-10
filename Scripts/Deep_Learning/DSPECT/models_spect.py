#!/usr/bin/env python3
"""
SPECT-Optimized Deep Learning Models
Designed specifically for 91x109x91 SPECT images and CN vs PD classification

Features:
- Models optimized for SPECT dimensions (91x109x91)
- Memory-efficient architectures for various hardware capabilities
- Built-in data validation and error handling
- Support for both training and inference
- Automatic model initialization based on input dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

# Import MONAI models
try:
    from monai.networks.nets import DenseNet121
except ImportError:
    DenseNet121 = None

logger = logging.getLogger(__name__)

class Simple3DCNN_SPECT(nn.Module):
    """
    Lightweight 3D CNN specifically designed for SPECT images.
    Optimized for 91x109x91 input dimensions and binary classification.
    
    Architecture:
    - 3 convolutional blocks with batch norm and ReLU
    - Adaptive pooling to handle variable input sizes
    - Dropout for regularization
    - Binary classification head
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 base_channels: int = 16,
                 dropout_rate: float = 0.5,
                 input_shape: Tuple[int, int, int] = (91, 109, 91)):
        super().__init__()
        
        self.input_shape = input_shape
        self.base_channels = base_channels
        self.num_classes = num_classes
        
        # Calculate expected feature dimensions after convolutions
        # Each conv block reduces dimensions by factor of 2 (due to MaxPool3d(2))
        self.feature_shape = self._calculate_feature_shape(input_shape)
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block: 91x109x91 -> 45x54x45
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Second conv block: 45x54x45 -> 22x27x22
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Third conv block: 22x27x22 -> 11x13x11
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Fourth conv block: 11x13x11 -> 5x6x5
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        
        # Adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Simple3DCNN_SPECT initialized with {self._count_parameters()} parameters")
        logger.info(f"Expected input shape: {input_shape}")
        logger.info(f"Feature shape after conv layers: {self.feature_shape}")
    
    def _calculate_feature_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate the expected feature dimensions after convolutional layers."""
        d, h, w = input_shape
        
        # After 4 MaxPool3d(2) operations
        d = d // 16
        h = h // 16
        w = w // 16
        
        return (d, h, w)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [B, 1, D, H, W]
            
        Returns:
            Output tensor of shape [B, num_classes]
        """
        # Input validation
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input tensor, got {x.dim()}D")
        if x.size(1) != 1:
            raise ValueError(f"Expected 1 channel, got {x.size(1)}")
        
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class ResNet3D_SPECT(nn.Module):
    """
    3D ResNet architecture adapted for SPECT images.
    Based on ResNet-18 architecture with 3D convolutions.
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 base_channels: int = 64,
                 dropout_rate: float = 0.5):
        super().__init__()
        
        self.base_channels = base_channels
        
        # Initial convolution
        self.conv1 = nn.Conv3d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Adaptive pooling and classifier
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
        logger.info(f"ResNet3D_SPECT initialized with {self._count_parameters()} parameters")
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        """Create a ResNet layer with residual connections."""
        layers = []
        
        # First block with potential downsampling
        layers.append(self._BasicBlock3D(in_channels, out_channels, stride))
        
        # Additional blocks without downsampling
        for _ in range(1, blocks):
            layers.append(self._BasicBlock3D(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if x.dim() != 5 or x.size(1) != 1:
            raise ValueError(f"Expected 5D input tensor with 1 channel, got {x.shape}")
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling and classification
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    class _BasicBlock3D(nn.Module):
        """3D Basic ResNet block."""
        
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm3d(out_channels)
            
            # Shortcut connection
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(out_channels)
                )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            
            out += self.shortcut(residual)
            out = F.relu(out)
            
            return out


class EfficientNet3D_SPECT(nn.Module):
    """
    Lightweight 3D CNN inspired by EfficientNet architecture.
    Designed for memory efficiency while maintaining performance.
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 base_channels: int = 32,
                 dropout_rate: float = 0.3,
                 input_shape: Tuple[int, int, int] = (91, 109, 91)):
        super().__init__()
        
        self.input_shape = input_shape
        
        # Efficient feature extraction with depthwise separable convolutions
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            
            # Efficient blocks
            self._make_efficient_block(base_channels, base_channels * 2, 1),
            self._make_efficient_block(base_channels * 2, base_channels * 4, 2),
            self._make_efficient_block(base_channels * 4, base_channels * 8, 2),
            self._make_efficient_block(base_channels * 8, base_channels * 16, 2),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 16, base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 8, num_classes)
        )
        
        self._initialize_weights()
        logger.info(f"EfficientNet3D_SPECT initialized with {self._count_parameters()} parameters")
    
    def _make_efficient_block(self, in_channels: int, out_channels: int, stride: int):
        """Create an efficient block with depthwise separable convolutions."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # Squeeze-and-excitation
            self._SELayer(out_channels)
        )
    
    class _SELayer(nn.Module):
        """Squeeze-and-Excitation layer for channel attention."""
        
        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, _, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1, 1)
            return x * y.expand_as(x)
    
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if x.dim() != 5 or x.size(1) != 1:
            raise ValueError(f"Expected 5D input tensor with 1 channel, got {x.shape}")
        
        # Feature extraction
        x = self.features(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class SPECTClassifier(nn.Module):
    """
    Unified SPECT classifier that can use different backbone architectures.
    Provides a consistent interface for all SPECT models.
    """
    
    def __init__(self, 
                 model_type: str = "simple",
                 num_classes: int = 2,
                 **kwargs):
        super().__init__()
        
        self.model_type = model_type
        
        # Select backbone architecture
        if model_type == "simple":
            self.backbone = Simple3DCNN_SPECT(num_classes=num_classes, **kwargs)
        elif model_type == "resnet":
            self.backbone = ResNet3D_SPECT(num_classes=num_classes, **kwargs)
        elif model_type == "efficient":
            self.backbone = EfficientNet3D_SPECT(num_classes=num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Available: simple, resnet, efficient")
        
        logger.info(f"SPECTClassifier initialized with {model_type} backbone")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': (1, 91, 109, 91),
            'output_shape': (2,),  # CN vs PD
            'architecture': str(self.backbone.__class__.__name__)
        }


def get_spect_model(model_type: str = "simple", 
                   num_classes: int = 2,
                   **kwargs) -> SPECTClassifier:
    """
    Factory function to create SPECT models.
    
    Args:
        model_type: Type of model ("simple", "resnet", "efficient")
        num_classes: Number of output classes (default: 2 for CN vs PD)
        **kwargs: Additional arguments for the model
        
    Returns:
        SPECTClassifier instance
    """
    return SPECTClassifier(model_type=model_type, num_classes=num_classes, **kwargs)


# Backwards-compatible alias for training script imports
class Simple3DCNN(Simple3DCNN_SPECT):
    pass


def get_3d_model(model_name, num_classes: int = 2, in_channels: int = 1, base_channels: int = 16,
                 use_pretrained: bool = False, dropout_p: float = 0.0,
                 vit_drop_rate: float = 0.0, vit_attn_drop_rate: float = 0.0, vit_drop_path_rate: float = 0.0):
    """
    DSPECT model factory mapping legacy names to SPECT-optimized backbones.
    Supported names:
      - "Simple3DCNN"        -> Simple3DCNN_SPECT
      - "ResNet18_3D"        -> ResNet3D_SPECT
      - "ResNet50_3D"        -> ResNet3D_SPECT
      - "DenseNet121_3D"     -> MONAI DenseNet121
      - "EfficientNetB0_3D"  -> EfficientNet3D_SPECT
      - Transformer models   -> See transformer_models.py

    Note: DenseNet121 requires MONAI to be installed.
    """
    name = str(model_name).lower()
    if name == "simple3dcnn":
        return Simple3DCNN_SPECT(
            num_classes=num_classes,
            base_channels=base_channels,
            dropout_rate=(dropout_p if dropout_p > 0 else 0.5)
        )
    if name in ("resnet18_3d", "resnet50_3d"):
        return ResNet3D_SPECT(
            num_classes=num_classes,
            base_channels=max(base_channels, 32)
        )
    if name == "densenet121_3d":
        if DenseNet121 is None:
            raise ImportError("MONAI is required for 3D DenseNet. Install with 'pip install monai'.")
        
        if use_pretrained:
            print("Warning: DenseNet121_3D does not support pretrained weights for 3D spatial dimensions.")
            print("Creating DenseNet121_3D from scratch...")
            model = DenseNet121(pretrained=False, spatial_dims=3, in_channels=in_channels, out_channels=num_classes)
        else:
            print("Creating DenseNet121_3D from scratch...")
            model = DenseNet121(pretrained=False, spatial_dims=3, in_channels=in_channels, out_channels=num_classes)
        return model
    if name == "efficientnetb0_3d":
        return EfficientNet3D_SPECT(
            num_classes=num_classes,
            base_channels=max(base_channels, 16)
        )

    raise ValueError(f"Unsupported model for DSPECT: {model_name}")


def get_model_summary(model: nn.Module, input_shape: Tuple[int, int, int, int, int] = (1, 1, 91, 109, 91)) -> str:
    """
    Generate a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, depth, height, width)
        
    Returns:
        String summary of the model
    """
    from collections import OrderedDict
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["output_shape"] = list(output.size())
            summary[m_key]["num_params"] = sum(p.numel() for p in module.parameters())
            
        hooks.append(module.register_forward_hook(hook))
    
    # Create summary
    summary = OrderedDict()
    hooks = []
    
    # Register hooks
    model.apply(register_hook)
    
    # Make a forward pass
    x = torch.zeros(input_shape)
    model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Format summary
    summary_str = "Model Summary:\n"
    summary_str += "=" * 80 + "\n"
    summary_str += f"{'Layer':<25} {'Output Shape':<25} {'Param #':<15}\n"
    summary_str += "=" * 80 + "\n"
    
    total_params = 0
    for layer in summary:
        summary_str += f"{layer:<25} {str(summary[layer]['output_shape']):<25} {summary[layer]['num_params']:<15}\n"
        total_params += summary[layer]['num_params']
    
    summary_str += "=" * 80 + "\n"
    summary_str += f"Total params: {total_params:,}\n"
    summary_str += f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
    
    return summary_str


if __name__ == "__main__":
    # Test model creation
    from collections import OrderedDict
    
    print("Testing SPECT model creation...")
    
    # Test Simple3DCNN
    model1 = get_spect_model("simple", base_channels=16)
    print(f"Simple3DCNN: {model1.get_model_info()}")
    
    # Test ResNet3D
    model2 = get_spect_model("resnet", base_channels=64)
    print(f"ResNet3D: {model2.get_model_info()}")
    
    # Test EfficientNet3D
    model3 = get_spect_model("efficient", base_channels=32)
    print(f"EfficientNet3D: {model3.get_model_info()}")
    
    # Test forward pass
    x = torch.randn(2, 1, 91, 109, 91)
    
    for name, model in [("Simple3DCNN", model1), ("ResNet3D", model2), ("EfficientNet3D", model3)]:
        try:
            output = model(x)
            print(f"{name}: Input {x.shape} -> Output {output.shape}")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    print("\nModel creation test complete!")
