# P4P: Neurodegenerative Disease Detection Pipeline

A comprehensive machine learning pipeline for neurodegenerative disease detection using structural MRI (sMRI), PET, and SPECT imaging data. The P4P project provides both classical machine learning (radiomics-based) and deep learning approaches (CNN and ViT) for classifying patients with Cognitive Normal (CN), Alzheimer's Disease (AD), and Parkinson's Disease (PD).

## 🎯 Project Overview

The P4P pipeline is designed to:
- **Preprocess** medical imaging data (DICOM → NIfTI → ML-ready format)
- **Extract features** using radiomics analysis
- **Train models** using both classical ML and deep learning approaches
- **Deploy models** for clinical prediction and interpretability
- **Compare approaches** using Bayesian statistical methods

## 📁 Project Structure

```
P4P/
├── config.yaml                    # Main configuration file
├── environment.yml                 # Conda environment specification
├── requirements.txt               # Python dependencies
├── README.md                      
├── generate_mri_labels.py         # Generate MRI subject labels
├── generate_pet_labels.py         # Generate PET subject labels
├── generate_spect_labels.py       # Generate SPECT subject labels
├── Templates/                     # Template images and masks
│   ├── README.md                  # Template documentation
│   ├── MRI/                       # MRI templates and masks
│   ├── PET/                       # PET templates and masks
│   └── SPECT/                     # SPECT templates and masks
└── Scripts/                       # Main pipeline scripts
    ├── README.md                  # Scripts documentation
    ├── Preprocessing/             # Data preprocessing pipelines
    ├── Feature_Extraction/        # Radiomics feature extraction
    ├── Classic_Learning/          # Classical ML approaches
    ├── Deep_Learning/             # Deep learning approaches
    ├── Clinical_Deploy/           # Clinical deployment tools
    ├── Comparison/                # Model comparison methods
    └── Visualise/                 # Visualization tools
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd P4P

# Create conda environment
conda env create -f environment.yml
conda activate p4p

# Or install with pip
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.yaml` to specify your data paths:

```yaml
raw_data:
  smri: ~/path/to/raw/MRI
  pet: ~/path/to/raw/PET
  spect: ~/path/to/raw/SPECT

preprocessed_data:
  smri_p: ~/path/to/preprocessed/MRI
  pet_p: ~/path/to/preprocessed/PET
  spect_p: ~/path/to/preprocessed/SPECT
```

### 3. Generate Labels

```bash
# Generate labels for each modality
python generate_mri_labels.py
python generate_pet_labels.py
python generate_spect_labels.py
```

## 🔄 Complete Workflow

### 1. Data Preprocessing

#### DICOM to NIfTI Conversion
```bash
cd Scripts/Preprocessing/01_DCM_TO_NIFTI
python ultimate_converter_fixed.py --input ~/path/to/dicom --output ~/path/to/nifti
```

#### SPECT Preprocessing Pipeline
```bash
cd Scripts/Preprocessing/DSPECT
python run_pipeline.py --diagnosis CN  # or PD
```

#### PET Preprocessing
```bash
cd Scripts/Preprocessing/PET
python 02_norm_stand.py --input ~/path/to/pet --output ~/path/to/processed
```

#### MRI Preprocessing
```bash
cd Scripts/Preprocessing/MRI
python 02_smriprep_run.py --input ~/path/to/mri --output ~/path/to/processed
```

### 2. Feature Extraction

#### Radiomics Feature Extraction
```bash
cd Scripts/Feature_Extraction/pyRadioMics
python radiomics_extractor.py --input ~/path/to/images --output ~/path/to/features
```

### 3. Model Training

#### Classical Machine Learning
```bash
cd Scripts/Classic_Learning
python complete_workflow.py --input ~/path/to/radiomics.csv --output ~/path/to/results
```

#### Deep Learning
```bash
# MRI
cd Scripts/Deep_Learning/MRI
python train_smri.py --config config_hardware_optimized.yaml

# PET
cd Scripts/Deep_Learning/PET
python train_pet.py --config config_hardware_optimized.yaml

# SPECT
cd Scripts/Deep_Learning/DSPECT
python train_spect.py --config config_hardware_optimized.yaml
```

### 4. Model Evaluation and Comparison

#### Bayesian Model Comparison
```bash
cd Scripts/Comparison
python run_bayesian_analysis.py --input ~/path/to/results --output ~/path/to/comparison
```

### 5. Clinical Deployment

#### Classical Model Prediction
```bash
cd Scripts/Clinical_Deploy/Classic
python predict_clinical.py --model ~/path/to/model.pkl --input ~/path/to/features.csv
```

#### Deep Model Prediction
```bash
cd Scripts/Clinical_Deploy/Deep
python predict_clinical_deep.py --model ~/path/to/model.pth --input ~/path/to/image.nii.gz
```

## 🧠 Supported Modalities

### Structural MRI (sMRI)
- **Preprocessing**: sMRIprep pipeline, skull stripping, z-score normalization
- **Features**: Radiomics features, deep learning features
- **Models**: 3D CNN, Vision Transformers, classical ML

### PET Imaging
- **Preprocessing**: SUVR normalization, skull stripping, registration
- **Features**: Radiomics features, deep learning features
- **Models**: 3D CNN, Vision Transformers, classical ML

### SPECT Imaging
- **Preprocessing**: Reorientation, normalization, registration, masking
- **Features**: Radiomics features, deep learning features
- **Models**: 3D CNN, Vision Transformers, classical ML

## 📊 Model Performance

The pipeline supports multiple evaluation metrics:
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC, PR-AUC**
- **Matthews Correlation Coefficient**
- **Confusion Matrix Analysis**
- **Calibration Analysis**

## 🔍 Interpretability

### SHAP Analysis (Classical ML)
```bash
cd Scripts/Clinical_Deploy/Classic
python run_shap_analysis.py --model ~/path/to/model.pkl --data ~/path/to/features.csv
```

### Grad-CAM (Deep Learning)
```bash
cd Scripts/Deep_Learning/MRI
python visualise_gradcam.py --model ~/path/to/model.pth --input ~/path/to/image.nii.gz
```

## 🎨 Visualization Tools

```bash
# Interactive visualization
cd Scripts/Visualise
python interactive_visualise.py --input ~/path/to/image.nii.gz

# Middle slice visualization
python visualise_middle_slice.py --input ~/path/to/image.nii.gz --output ~/path/to/plot.png

# Grad-CAM animation
python animate_gradcam_overlay.py --input ~/path/to/image.nii.gz --gradcam ~/path/to/gradcam.nii.gz
```

## 🔧 Configuration

The pipeline uses YAML configuration files for flexible parameter tuning:

- `config.yaml`: Main project configuration
- `config_hardware_optimized.yaml`: Hardware-optimized settings
- `config_transformers.yaml`: Transformer-specific settings
- `config_enhanced.yaml`: Enhanced classifier settings

## 📈 Output Structure

Results are organized as follows:
```
results/
├── models/                    # Trained models (.pkl, .pth)
├── plots/                     # Visualization plots (.png)
├── metrics/                   # Performance metrics (.json)
├── features/                  # Extracted features (.csv)
├── logs/                      # Training logs (.log)
└── reports/                   # Summary reports (.txt, .json)
```

## 🚨 System Requirements

- **Python**: 3.8+
- **RAM**: 16GB+ (32GB+ recommended)
- **GPU**: CUDA-compatible (recommended for deep learning)
- **Storage**: 1TB+ for preprocessing and results
- **External Tools**: dcm2niix, smriprep, gdcmsan (for DICOM conversion)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Documentation

- [Templates Documentation](Templates/README.md)
- [Scripts Documentation](Scripts/README.md)
- [Preprocessing Guide](Scripts/Preprocessing/README.md)
- [Classical Learning Guide](Scripts/Classic_Learning/README.md)
- [Deep Learning Guide](Scripts/Deep_Learning/README.md)

## 🆘 Support

For questions and support:
- Check the documentation in each subdirectory
- Review the example scripts and configurations
- Open an issue on the repository

## 🔬 Research Applications

This pipeline has been designed for research applications in:
- Neurodegenerative disease detection
- Medical imaging analysis
- Radiomics research
- Deep learning in medical imaging
- Model interpretability studies

## ⚠️ Clinical Disclaimer

This software is intended for research purposes only. It should not be used for clinical diagnosis or treatment decisions without proper validation and regulatory approval.
