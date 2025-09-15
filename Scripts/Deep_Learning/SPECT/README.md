# SPECT Deep Learning Pipeline

A comprehensive deep learning pipeline specifically designed for SPECT (Single-Photon Emission Computed Tomography) image classification, optimized for CN (Control) vs PD (Parkinson's Disease) classification.

## 🚀 **Quick Start**

### **1. Local MacBook Training (CPU)**
```bash
# Navigate to SPECT directory
cd Scripts/Deep_Learning/SPECT

# Create labels and start training
python3 train_spect.py --data_root "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT" \
    --output_dir "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT/training_output" \
                       --model_type "simple" \
                       --epochs 50 \
                       --batch_size 2
```

### **2. HPC GPU Training**
```bash
# Submit to HPC queue
sbatch --gres=gpu:1 --mem=32G --cpus-per-task=4 train_spect_hpc.sh
```

## 📁 **File Structure**

```
SPECT/
├── dataset.py              # SPECT-optimized dataset loader
├── models_spect.py         # SPECT-specific model architectures
├── train_spect.py          # Main training script
├── evaluate_spect.py       # Model evaluation and inference
├── config_spect.yaml       # Comprehensive configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 **Core Components**

### **1. Dataset Loader (`dataset.py`)**
- **SPECTDataset**: Main dataset class for loading preprocessed SPECT images
- **SPECTDatasetBalanced**: Balanced version for handling class imbalance
- **Automatic label generation**: Creates CN=0, PD=1 labels from folder structure
- **Built-in validation**: Checks data integrity, dimensions, and quality
- **Memory optimization**: Optional caching for faster training

**Key Features:**
- Expects: `CN_SPECT_PPMI_postprocessed/` and `PD_SPECT_PPMI_postprocessed/` folders
- Each subject folder contains: `6. postprocessed.nii.gz`
- Output: Images shape `[1, 91, 109, 91]`, Labels: `0` (CN) or `1` (PD)

### **2. Model Architectures (`models_spect.py`)**
- **Simple3DCNN_SPECT**: Lightweight CNN optimized for SPECT dimensions
- **ResNet3D_SPECT**: 3D ResNet architecture with residual connections
- **EfficientNet3D_SPECT**: Memory-efficient model with depthwise convolutions
- **SPECTClassifier**: Unified interface for all model types

**Model Comparison:**
| Model | Parameters | Memory | Speed | Best For |
|-------|------------|---------|-------|----------|
| Simple3DCNN | ~500K | Low | Fast | Quick experiments |
| ResNet3D | ~2M | Medium | Medium | Best performance |
| EfficientNet3D | ~1M | Low | Fast | Limited resources |

### **3. Training Script (`train_spect.py`)**
- **Automatic setup**: Data loading, model creation, training components
- **Comprehensive logging**: TensorBoard integration, checkpointing
- **Flexible configuration**: YAML config files, command-line overrides
- **Error handling**: Graceful failure recovery, early stopping
- **Performance monitoring**: Real-time metrics, validation curves

### **4. Evaluation Script (`evaluate_spect.py`)**
- **Model evaluation**: Comprehensive performance metrics
- **Visualization**: Confusion matrices, ROC curves, prediction distributions
- **Single image inference**: Predict on individual SPECT images
- **Report generation**: Detailed markdown reports with recommendations

## 📊 **Configuration Options**

### **Basic Training Configuration**
```yaml
# config_spect.yaml
model_type: "simple"           # simple, resnet, efficient
batch_size: 4                  # Adjust based on GPU memory
epochs: 100                    # Training epochs
learning_rate: 1e-4           # Initial learning rate
optimizer: "adam"             # adam, sgd
scheduler: "step"             # step, cosine, plateau
```

### **Environment-Specific Configs**
```yaml
# Local MacBook (CPU)
environments:
  local:
    batch_size: 2
    num_workers: 1
    use_gpu: false

# HPC GPU Cluster
environments:
  hpc:
    batch_size: 8
    num_workers: 4
    use_gpu: true
    mixed_precision: true
```

### **Model Presets**
```yaml
# Quick development
model_presets:
  lightweight:
    model_type: "simple"
    base_channels: 8
    batch_size: 8
    epochs: 20

# Best performance
model_presets:
  high_performance:
    model_type: "resnet"
    base_channels: 64
    batch_size: 2
    epochs: 200
```

## 🚀 **Usage Examples**

### **Example 1: Quick Local Training**
```bash
# Train lightweight model on MacBook
python3 train_spect.py \
    --data_root "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT" \
    --output_dir "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT/training_output" \
    --model_type "simple" \
    --epochs 20 \
    --batch_size 2
```

### **Example 2: Full Training with Custom Config**
```bash
# Use custom configuration
python3 train_spect.py \
    --config "config_spect.yaml" \
    --data_root "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT" \
    --output_dir "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT/training_output"
```

### **Example 3: Resume Training**
```bash
# Resume from checkpoint
python3 train_spect.py \
    --config "config_spect.yaml" \
    --resume "training_output/checkpoint_latest.pth"
```

### **Example 4: Model Evaluation**
```bash
# Evaluate trained model
python3 evaluate_spect.py \
    --model_path "training_output/checkpoint_best.pth" \
    --data_root "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT" \
    --output_dir "evaluation_results"
```

### **Example 5: Single Image Prediction**
```bash
# Predict on single image
python3 evaluate_spect.py \
    --model_path "training_output/checkpoint_best.pth" \
    --data_root "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT" \
    --output_dir "evaluation_results" \
    --single_image "CN_SPECT_PPMI_postprocessed/Subject_100001/6. postprocessed.nii.gz"
```

## 🔍 **Data Requirements**

### **Input Data Structure**
```
SPECT/
├── CN_SPECT_PPMI_postprocessed/
│   ├── Subject_100001/
│   │   └── 6. postprocessed.nii.gz
│   ├── Subject_100002/
│   │   └── 6. postprocessed.nii.gz
│   └── ...
├── PD_SPECT_PPMI_postprocessed/
│   ├── Subject_200001/
│   │   └── 6. postprocessed.nii.gz
│   ├── Subject_200002/
│   │   └── 6. postprocessed.nii.gz
│   └── ...
```

### **Image Specifications**
- **Format**: NIfTI (.nii.gz)
- **Dimensions**: 91 × 109 × 91 voxels
- **Orientation**: RAS (Right-Anterior-Superior)
- **Data type**: Float32, z-score normalized
- **Value range**: Typically -5 to +5 standard deviations

## 💻 **Hardware Requirements**

### **Local MacBook Training**
- **CPU**: Any modern multi-core processor
- **RAM**: 16GB+ recommended
- **Storage**: SSD for faster data loading
- **Training time**: 2-4 hours for 100 epochs (simple model)

### **HPC GPU Training**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **CPU**: Multi-core processor (32+ cores optimal)
- **RAM**: 32GB+ recommended
- **Training time**: 30-60 minutes for 100 epochs (simple model)

### **Cloud GPU Training**
- **Google Colab**: Free GPU (limited time)
- **AWS/Azure**: Pay-per-use GPU instances
- **Recommended**: p3.2xlarge or equivalent

## 📈 **Performance Expectations**

### **Model Performance (Typical)**
| Model | Accuracy | Training Time (Local) | Training Time (HPC) |
|-------|----------|----------------------|---------------------|
| Simple3DCNN | 75-85% | 2-4 hours | 30-60 min |
| ResNet3D | 80-90% | 4-8 hours | 1-2 hours |
| EfficientNet3D | 78-88% | 3-6 hours | 45-90 min |

### **Optimization Tips**
1. **Start simple**: Begin with Simple3DCNN for quick experiments
2. **Batch size**: Use largest batch size that fits in memory
3. **Learning rate**: Start with 1e-4, adjust based on convergence
4. **Early stopping**: Use patience of 20-30 epochs
5. **Data augmentation**: Enable for small datasets

## 🐛 **Troubleshooting**

### **Common Issues**

#### **1. Out of Memory (OOM)**
```bash
# Reduce batch size
--batch_size 2

# Use gradient accumulation
gradient_accumulation_steps: 2
```

#### **2. Slow Training**
```bash
# Increase batch size if memory allows
--batch_size 8

# Use more workers
--num_workers 4

# Enable mixed precision (GPU only)
mixed_precision: true
```

#### **3. Poor Performance**
```bash
# Check data quality
python3 dataset.py --validate_only

# Use class weights
use_class_weights: true

# Try different model architecture
--model_type "resnet"
```

#### **4. Data Loading Errors**
```bash
# Verify data structure
ls -la /Volumes/reseng202500013-ndd-ml/data/Final_SPECT/

# Check file permissions
chmod -R 755 /Volumes/reseng202500013-ndd-ml/data/Final_SPECT/

# Validate NIfTI files
python3 -c "import nibabel as nib; img = nib.load('path/to/image.nii.gz'); print(img.shape)"
```

## 🔄 **Workflow Integration**

### **Complete Pipeline**
1. **Data Preparation**: Run SPECT preprocessing pipeline
2. **Label Creation**: Automatic from folder structure
3. **Model Training**: Choose architecture and train
4. **Evaluation**: Comprehensive performance analysis
5. **Deployment**: Use trained model for inference

### **Integration with Existing Workflows**
```python
# Import SPECT components
from dataset import SPECTDataset
from models_spect import get_spect_model
from train_spect import SPECTTrainer

# Use in custom scripts
dataset = SPECTDataset("/path/to/spect/data")
model = get_spect_model("simple")
trainer = SPECTTrainer(config)
```

## 📚 **Advanced Features**

### **Cross-Validation**
```yaml
cross_validation:
  enabled: true
  n_folds: 5
  stratified: true
```

### **Hyperparameter Optimization**
```yaml
hyperopt:
  enabled: true
  trials: 100
  optimization_metric: "val_accuracy"
```

### **Model Ensemble**
```yaml
ensemble:
  enabled: true
  models: ["simple", "resnet", "efficient"]
  voting_method: "soft"
```

### **Transfer Learning**
```yaml
transfer_learning:
  enabled: true
  pretrained_model_path: "/path/to/pretrained/model.pth"
  freeze_backbone: false
```

## 📊 **Monitoring and Logging**

### **TensorBoard Integration**
```bash
# Start TensorBoard
tensorboard --logdir training_output/tensorboard

# View in browser: http://localhost:6006
```

### **Log Files**
- **Training log**: `spect_training.log`
- **Checkpoints**: `checkpoint_latest.pth`, `checkpoint_best.pth`
- **Results**: `training_results.json`
- **Configuration**: `config.yaml`

## 🚀 **Deployment**

### **Model Export**
```python
# Export to ONNX
torch.onnx.export(model, dummy_input, "spect_model.onnx")

# Export to TorchScript
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("spect_model.pt")
```

### **Production Inference**
```python
# Load trained model
model = torch.load("checkpoint_best.pth")
model.eval()

# Single prediction
with torch.no_grad():
    prediction = model(input_tensor)
```

## 🤝 **Contributing**

### **Adding New Models**
1. Create model class in `models_spect.py`
2. Add to `SPECTClassifier` factory
3. Update configuration options
4. Add tests and documentation

### **Adding New Features**
1. Follow existing code structure
2. Add comprehensive logging
3. Include error handling
4. Update documentation

## 📞 **Support**

### **Getting Help**
1. Check this README for common solutions
2. Review error logs and training output
3. Verify data structure and file paths
4. Check hardware requirements

### **Reporting Issues**
Include:
- Error messages and logs
- System specifications
- Data structure details
- Steps to reproduce

## 📄 **License**

This project is part of the P4P (Parkinson's for Parkinson's) initiative.

## 🙏 **Acknowledgments**

- Built on PyTorch ecosystem
- Optimized for SPECT neuroimaging data
- Designed for both research and clinical applications
