#!/bin/bash

# Installation script for Transformer-based 3D Medical Imaging Models
# This script installs all required dependencies for Vision Transformers and Swin UNETR
# in the current environment

echo "=========================================="
echo "Installing Transformer Model Dependencies"
echo "Installing in current environment..."
echo "=========================================="

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (matching your CUDA 12.6)
echo "Installing PyTorch with CUDA 12.6..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install MONAI (Medical Open Network for AI)
echo "Installing MONAI..."
pip install monai

# Install Vision Transformer dependencies
echo "Installing Vision Transformer dependencies..."
pip install timm
pip install einops

# Install Swin Transformer dependencies
echo "Installing Swin Transformer dependencies..."
pip install swin-transformer-unetr

# Install additional utilities
echo "Installing additional utilities..."
pip install transformers
pip install accelerate
pip install pyyaml

# Install scientific computing libraries
echo "Installing scientific computing libraries..."
pip install numpy scipy scikit-learn
pip install matplotlib seaborn pandas
pip install nibabel

# Install optional dependencies for advanced features
echo "Installing optional dependencies..."
pip install tensorboard
pip install wandb  # for experiment tracking
pip install albumentations  # for data augmentation

# Verify installation
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')

try:
    import monai
    print(f'MONAI version: {monai.__version__}')
except ImportError:
    print('MONAI not installed')

try:
    import timm
    print(f'timm version: {timm.__version__}')
except ImportError:
    print('timm not installed')

try:
    import einops
    print(f'einops version: {einops.__version__}')
except ImportError:
    print('einops not installed')

try:
    from transformer_models import VisionTransformer3D, SwinUNETRClassifier
    print('Transformer models imported successfully')
except ImportError as e:
    print(f'Transformer models import failed: {e}')

print('Installation verification completed!')
"

echo "=========================================="
echo "Installation completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test the models: python -c 'from transformer_models import *; print(\"Models ready!\")'"
echo "2. Run training: python train_transformers.py --help"
echo ""
echo "Hardware requirements:"
echo "- GPU with at least 8GB VRAM (16GB recommended)"
echo "- 32GB RAM recommended"
echo "- CUDA 12.6 or later"
echo ""
echo "For more information, see the README and configuration files." 