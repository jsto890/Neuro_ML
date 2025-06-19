# P4P Research Project: Early Detection of Neurodegenerative Diseases

## 🧠 Project Overview
This research project focuses on developing machine learning models for early detection of Alzheimer's Disease (AD) and Parkinson's Disease (PD) using multimodal medical imaging data. The project leverages PET, MRI, and DaTSPECT scans to identify early biomarkers of these neurodegenerative conditions through both classical machine learning and deep learning approaches.

## 📋 Key Features
- **Multimodal Data Processing**: Integrated pipeline for PET, MRI, and DaTSPECT image processing
- **Standardized Preprocessing**: Automated data standardization and quality control
- **Radiomics Feature Extraction**: Advanced feature extraction using pyRadiomics
- **Classical Machine Learning**: Random Forest pipeline with comprehensive evaluation
- **Deep Learning Pipeline**: CNN-based models for neuroimaging classification
- **Quality Control**: Comprehensive QC reports and validation metrics
- **Reproducible Research**: Complete pipeline with configuration management

## 📁 Project Structure
```
P4P/
├── Scripts/                    # Main processing scripts
│   ├── Preprocessing/          # Data preprocessing pipelines
│   │   ├── 01_DCM_TO_NIFTI/   # DICOM to NIfTI conversion
│   │   ├── MRI/                # MRI preprocessing (smriprep)
│   │   └── PET/                # PET preprocessing and standardization
│   ├── Feature_Extraction/     # Feature extraction using pyRadiomics
│   │   └── pyRadioMics/        # Radiomics extraction scripts
│   │       ├── simple_radiomics.py      # ✅ Working MRI radiomics extractor
│   │       ├── radiomics_extractor.py   # Multi-modality extractor
│   │       ├── test_paths.py            # Path validation tool
│   │       └── debug_paths.py           # Path debugging tool
│   ├── Classic_Learning/       # Classical machine learning pipeline
│   │   ├── radiomics_classifier.py      # Main Random Forest pipeline
│   │   ├── preprocessing.py             # Data preprocessing utilities
│   │   ├── run_classical.py             # Simple runner script
│   │   └── config_classical.yaml        # Classical learning configuration
│   ├── Deep_Learning/          # Deep learning models and training
│   │   ├── MRI/                # MRI-specific deep learning
│   │   │   ├── dataset.py              # PyTorch dataset for MRI
│   │   │   ├── models_smri.py          # CNN model architectures
│   │   │   ├── train_smri.py           # Training script
│   │   │   ├── gradcam.py              # Grad-CAM visualization
│   │   │   └── visualise_gradcam.py    # Grad-CAM plotting
│   │   ├── PET/                # PET-specific deep learning
│   │   └── SPECT/              # SPECT-specific deep learning
│   └── Visualise/              # Visualization tools
│       ├── interactive_visualise.py    # Interactive visualization
│       └── visualise_middle_slice.py   # Slice visualization
├── Script_Bin/                 # Utility scripts and tools
│   ├── convert_draft.py        # DaTSPECT preprocessing
│   ├── pet_standardise.py      # PET standardization
│   ├── DCM_to_NIfTI_SPECT.py   # SPECT DICOM conversion
│   └── testers/                # Testing utilities
├── Templates/                  # Reference templates
│   ├── PET/                    # PET templates and masks
│   ├── MRI/                    # MRI templates (MNI152)
│   └── SPECT/                  # SPECT templates
├── Labels/                     # Label files for training
│   ├── train_labels.csv        # Training set labels
│   └── val_labels.csv          # Validation set labels
├── requirements.txt            # Python package requirements
├── environment.yml             # Conda environment configuration
├── config.yaml                 # Global configuration file
└── generate_mri_labels.py      # Label generation script
```

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- Git
- (Optional) Virtual environment manager (conda/venv)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jackson-Schofield/P4P.git
   cd P4P
   ```

2. Choose your preferred installation method:

   #### Option 1: Using pip (requirements.txt)
   ```bash
   # Create and activate virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

   #### Option 2: Using conda (environment.yml)
   ```bash
   # Create and activate conda environment
   conda env create -f environment.yml
   conda activate p4p
   ```

### 🎯 Complete Workflow

#### 1. Extract Radiomics Features (MRI)
```bash
cd Scripts/Feature_Extraction/pyRadioMics/
python3 simple_radiomics.py --labels ~/reseng202500013-ndd-ml/data/mri_labels.csv --output-dir ~/reseng202500013-ndd-ml/data/ --config ~/reseng202500013-ndd-ml/P4P/config.yaml
```

#### 2. Run Classical Learning Pipeline
```bash
cd Scripts/Classic_Learning/
python3 run_classical.py
```

#### 3. Train Deep Learning Model (MRI)
```bash
cd Scripts/Deep_Learning/MRI/
python3 train_smri.py
```

## 📊 Data Processing Pipeline

### 1. Preprocessing
- **DICOM to NIfTI**: Automated conversion with quality checks
- **MRI Processing**: smriprep-based preprocessing pipeline
- **PET Standardization**: Motion correction, intensity normalization, SUVR calculation
- **Quality Control**: Comprehensive QC reports and validation metrics

### 2. Feature Extraction
- **Radiomics Features**: ~1000+ features using pyRadiomics
  - First Order: Intensity statistics
  - Shape: 3D morphological features
  - Texture: GLCM, GLRLM, GLSZM, GLDM, NGTDM
- **Feature Selection**: Automated selection of most informative features
- **Feature Validation**: Cross-validation and stability analysis

### 3. Classical Machine Learning
- **Random Forest Pipeline**: Complete training and evaluation
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Performance Metrics**: ROC AUC, accuracy, precision, recall, F1-score
- **Feature Importance**: Automated identification of key biomarkers
- **Model Interpretation**: SHAP values and feature rankings

### 4. Deep Learning
- **CNN Architectures**: Custom 3D CNN models for neuroimaging
- **Transfer Learning**: Pre-trained model adaptation
- **Grad-CAM Visualization**: Model interpretability
- **Multi-class Classification**: AD, PD, and control classification

## 🔧 Configuration

The project uses a centralized configuration system:

### Global Configuration (`config.yaml`)
```yaml
# Data paths
raw_data:
  smri: /path/to/raw/MRI
  pet: /path/to/raw/PET
  spect: /path/to/raw/SPECT

preprocessed_data:
  smri_p: /path/to/preprocessed/MRI/smriprep
  pet_p: /path/to/preprocessed/PET
  spect_p: /path/to/preprocessed/SPECT

# Template paths
templates:
  MRI_brain_mask: /path/to/templates/MRI/MNI152_T1_1mm_brain.nii.gz
  PET_template: /path/to/templates/PET/FDG-PET-template_padded.nii.gz
```

### Classical Learning Configuration (`Scripts/Classic_Learning/config_classical.yaml`)
- Preprocessing parameters
- Model hyperparameters
- Evaluation settings
- Output configurations

## 📈 Expected Results

### Radiomics Extraction
- **Processing Time**: 10-30 minutes for 474 subjects
- **Features**: ~1000+ radiomics features per subject
- **Output**: CSV file with features and labels

### Classical Learning
- **Training Time**: 5-15 minutes
- **Performance**: 70-85% accuracy, 0.75-0.90 ROC AUC
- **Output**: Trained model, feature importance, evaluation plots

### Deep Learning
- **Training Time**: 1-4 hours (depending on hardware)
- **Performance**: 75-90% accuracy
- **Output**: Trained model, training curves, Grad-CAM visualizations

## 🐛 Troubleshooting

### Common Issues

1. **NumPy Compatibility (PyRadiomics)**
   ```bash
   conda install numpy=1.24.3 -y
   ```

2. **Missing Data Paths**
   - Check `config.yaml` for correct paths
   - Use `test_paths.py` to validate paths

3. **Memory Issues**
   - Reduce batch size in deep learning
   - Use feature selection in classical learning

### Debug Tools
- `Scripts/Feature_Extraction/pyRadioMics/test_paths.py` - Path validation
- `Scripts/Feature_Extraction/pyRadioMics/debug_paths.py` - Path debugging
- Comprehensive logging in all pipelines

## 📚 Documentation

- **Feature Extraction**: `Scripts/Feature_Extraction/pyRadioMics/README.md`
- **Classical Learning**: `Scripts/Classic_Learning/README.md`
- **Deep Learning**: `Scripts/Deep_Learning/MRI/README.md`

## 🔬 Research Applications

This pipeline is designed for:
- **Early Disease Detection**: Identifying biomarkers before clinical symptoms
- **Differential Diagnosis**: Distinguishing between AD, PD, and controls
- **Prognostic Modeling**: Predicting disease progression
- **Biomarker Discovery**: Finding novel imaging biomarkers

## 👥 Collaborators
- **Joseph Storey**
- **Jackson Schofield**

## 📞 Support
For questions and support, please open an issue in the GitHub repository.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
