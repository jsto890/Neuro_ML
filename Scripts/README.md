# Scripts Directory

This directory contains all the main pipeline scripts for the P4P (Alzheimer's and Parkinson's Disease Detection Pipeline) project. The scripts are organised into logical modules covering the complete workflow from data preprocessing to model deployment.

##  Directory Structure

```
Scripts/
├── README.md                     # This file
├── Preprocessing/                # Data preprocessing pipelines
│   ├── README.md                # Preprocessing documentation
│   ├── 01_DCM_TO_NIFTI/         # DICOM to NIfTI conversion
│   ├── MRI/                     # MRI preprocessing
│   ├── PET/                     # PET preprocessing
│   └── DSPECT/                  # SPECT preprocessing
├── Feature_Extraction/           # Radiomics feature extraction
│   ├── README.md                # Feature extraction documentation
│   └── pyRadioMics/             # Radiomics analysis
├── Classic_Learning/             # Classical machine learning
│   ├── README.md                # Classical ML documentation
│   ├── Enhanced/                # Enhanced classifiers
│   └── [other classical ML scripts]
├── Deep_Learning/                # Deep learning approaches
│   ├── README.md                # Deep learning documentation
│   ├── MRI/                     # MRI deep learning
│   ├── PET/                     # PET deep learning
│   └── DSPECT/                  # SPECT deep learning
├── Clinical_Deploy/              # Clinical deployment tools
│   ├── README.md                # Clinical deployment documentation
│   ├── Classic/                 # Classical model deployment
│   └── Deep/                    # Deep model deployment
├── Comparison/                   # Model comparison methods
│   ├── README.md                # Comparison documentation
│   └── [comparison scripts]
└── Visualise/                    # Visualization tools
    ├── README.md                # Visualization documentation
    └── [visualisation scripts]
```

##  Complete Pipeline Workflow

### 1. Data Preprocessing (`Preprocessing/`)

#### DICOM to NIfTI Conversion
- **Location**: `Preprocessing/01_DCM_TO_NIFTI/`
- **Purpose**: Convert DICOM files to NIfTI format
- **Key Scripts**:
  - `ultimate_converter_fixed.py`: Multi-tool DICOM conversion
  - `dcm_convert.py`: Basic DICOM conversion
  - `test_dicom.py`: DICOM file validation

#### MRI Preprocessing
- **Location**: `Preprocessing/MRI/`
- **Purpose**: Structural MRI preprocessing pipeline
- **Key Scripts**:
  - `02_smriprep_run.py`: sMRIprep pipeline execution
  - `03_zscore_skull_strip.py`: Z-score normalization and skull stripping

#### PET Preprocessing
- **Location**: `Preprocessing/PET/`
- **Purpose**: PET image preprocessing
- **Key Scripts**:
  - `02_norm_stand.py`: PET normalization and standardisation
  - `03_skullstrip.py`: PET skull stripping

#### SPECT Preprocessing
- **Location**: `Preprocessing/DSPECT/`
- **Purpose**: SPECT image preprocessing pipeline
- **Key Scripts**:
  - `run_pipeline.py`: Complete SPECT preprocessing pipeline
  - `1_reorient.py`: Image reorientation
  - `2_normalise.py`: SPECT normalization
  - `3_register.py`: Image registration
  - `4_masking.py`: Brain masking
  - `5_padding.py`: Image padding and finalization
  - `6_postprocess.py`: Postprocessing steps

### 2. Feature Extraction (`Feature_Extraction/`)

#### Radiomics Analysis
- **Location**: `Feature_Extraction/pyRadioMics/`
- **Purpose**: Extract radiomics features from medical images
- **Key Scripts**:
  - `radiomics_extractor.py`: Comprehensive radiomics extraction
  - `simple_radiomics.py`: Basic radiomics features

### 3. Model Training

#### Classical Machine Learning (`Classic_Learning/`)
- **Purpose**: Traditional ML approaches using radiomics features
- **Key Scripts**:
  - `complete_workflow.py`: Complete radiomics classification workflow
  - `run_best_model.py`: Run the best performing model
  - `run_fdr_comparison.py`: False discovery rate comparison
  - `Enhanced/enhanced_classifier.py`: Advanced ML classifiers
  - `Enhanced/run_enhanced.py`: Enhanced classifier pipeline

#### Deep Learning (`Deep_Learning/`)
- **Purpose**: Deep learning approaches using raw image data
- **Key Scripts**:
  - `MRI/train_smri.py`: MRI deep learning training
  - `PET/train_pet.py`: PET deep learning training
  - `DSPECT/train_spect.py`: SPECT deep learning training
  - `MRI/evaluate_model.py`: Model evaluation
  - `MRI/visualise_gradcam.py`: Grad-CAM visualisation

### 4. Model Comparison (`Comparison/`)

#### Bayesian Analysis
- **Purpose**: Statistical model comparison and evaluation
- **Key Scripts**:
  - `bayesian_model_comparison.py`: Comprehensive Bayesian analysis
  - `compare_models.py`: Model comparison utilities
  - `run_bayesian_analysis.py`: Bayesian analysis pipeline

### 5. Clinical Deployment (`Clinical_Deploy/`)

#### Classical Model Deployment
- **Location**: `Clinical_Deploy/Classic/`
- **Purpose**: Deploy classical ML models for clinical use
- **Key Scripts**:
  - `predict_clinical.py`: Clinical prediction interface
  - `validate_model.py`: Model validation
  - `run_shap_analysis.py`: SHAP interpretability analysis
  - `run_shap_comprehensive.py`: Comprehensive SHAP analysis
  - `run_shap_multifold.py`: Multi-fold SHAP analysis
  - `shap_interpretability.py`: SHAP interpretability module

#### Deep Model Deployment
- **Location**: `Clinical_Deploy/Deep/`
- **Purpose**: Deploy deep learning models for clinical use
- **Key Scripts**:
  - `predict_clinical_deep.py`: Deep learning clinical prediction
  - `validate_model_deep.py`: Deep model validation

### 6. Visualization (`Visualise/`)

#### Visualization Tools
- **Purpose**: Visualize results and intermediate steps
- **Key Scripts**:
  - `interactive_visualise.py`: Interactive image visualisation
  - `visualise_middle_slice.py`: Middle slice visualisation
  - `animate_gradcam_overlay.py`: Grad-CAM animation

##  Quick Start Examples

### Complete Radiomics Workflow

```bash
# Navigate to Classic Learning
cd Scripts/Classic_Learning

# Run complete workflow
python complete_workflow.py \
    --input ~/path/to/radiomics_features.csv \
    --output ~/path/to/results/ \
    --config config.yaml
```

### SPECT Preprocessing Pipeline

```bash
# Navigate to SPECT preprocessing
cd Scripts/Preprocessing/DSPECT

# Run complete SPECT pipeline
python run_pipeline.py --diagnosis CN --force
```

### Deep Learning Training

```bash
# Navigate to MRI deep learning
cd Scripts/Deep_Learning/MRI

# Train MRI model
python train_smri.py --config config_hardware_optimised.yaml
```

### Clinical Prediction

```bash
# Navigate to clinical deployment
cd Scripts/Clinical_Deploy/Classic

# Run clinical prediction
python predict_clinical.py \
    --model ~/path/to/model.pkl \
    --input ~/path/to/features.csv \
    --output ~/path/to/predictions.json
```

##  Configuration Files

### Hardware-Optimized Configuration
- **File**: `config_hardware_optimised.yaml`
- **Purpose**: Optimized settings for different hardware configurations
- **Usage**: Automatic hardware detection and optimisation

### Transformer Configuration
- **File**: `config_transformers.yaml`
- **Purpose**: Vision Transformer specific settings
- **Usage**: Transformer model training and evaluation

### Enhanced Configuration
- **File**: `config_enhanced.yaml`
- **Purpose**: Enhanced classifier settings
- **Usage**: Advanced ML pipeline configuration

##  Output Structure

### Training Results
```
results/
├── models/                      # Trained models
│   ├── *.pkl                   # Classical ML models
│   ├── *.pth                   # Deep learning models
│   └── *.json                  # Model configurations
├── plots/                       # Visualization plots
│   ├── *.png                   # Performance plots
│   ├── *.svg                   # Vector graphics
│   └── *.html                  # Interactive plots
├── metrics/                     # Performance metrics
│   ├── *.json                  # Detailed metrics
│   ├── *.csv                   # Tabular results
│   └── *.txt                   # Summary reports
└── logs/                        # Training logs
    ├── *.log                   # Training logs
    └── *.out                   # Output logs
```

### Clinical Deployment Results
```
clinical_results/
├── predictions/                 # Model predictions
│   ├── *.json                  # Prediction results
│   ├── *.csv                   # Tabular predictions
│   └── *.nii.gz                # Prediction maps
├── interpretability/            # Model interpretability
│   ├── shap_*.png              # SHAP plots
│   ├── gradcam_*.nii.gz        # Grad-CAM maps
│   └── saliency_*.nii.gz       # Saliency maps
└── reports/                     # Clinical reports
    ├── *.html                  # Clinical reports
    └── *.pdf                   # PDF reports
```

##  Supported Modalities

### Structural MRI (sMRI)
- **Preprocessing**: sMRIprep, skull stripping, normalization
- **Features**: Radiomics, deep learning features
- **Models**: 3D CNN, Vision Transformers, classical ML

### PET Imaging
- **Preprocessing**: SUVR normalization, registration
- **Features**: Radiomics, deep learning features
- **Models**: 3D CNN, Vision Transformers, classical ML

### SPECT Imaging
- **Preprocessing**: Reorientation, normalization, registration
- **Features**: Radiomics, deep learning features
- **Models**: 3D CNN, Vision Transformers, classical ML

##  Model Interpretability

### SHAP Analysis (Classical ML)
- **Location**: `Clinical_Deploy/Classic/`
- **Purpose**: Explain classical ML model predictions
- **Features**: Feature importance, dependence plots, force plots

### Grad-CAM (Deep Learning)
- **Location**: `Deep_Learning/*/visualise_gradcam.py`
- **Purpose**: Explain deep learning model predictions
- **Features**: Gradient-weighted class activation maps

### Saliency Maps
- **Location**: `Clinical_Deploy/Deep/`
- **Purpose**: Visualize input sensitivity
- **Features**: Gradient-based saliency analysis

##  Performance Monitoring

### Training Metrics
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC, PR-AUC**
- **Confusion Matrix Analysis**
- **Learning Curves**

### Validation Metrics
- **Cross-validation Performance**
- **Holdout Test Performance**
- **Calibration Analysis**
- **Bias Detection**

##  Error Handling

### Common Issues
1. **File Path Errors**: Check configuration files and paths
2. **Memory Issues**: Use hardware-optimised configurations
3. **GPU Issues**: Check CUDA compatibility and memory
4. **Data Format Issues**: Validate input data formats

### Debugging
- **Log Files**: Check `logs/` directory for detailed error messages
- **Validation Scripts**: Use validation scripts to check data integrity
- **Test Scripts**: Run test scripts to verify pipeline components

##  Customization

### Adding New Modalities
1. Create new preprocessing scripts in `Preprocessing/`
2. Add feature extraction scripts in `Feature_Extraction/`
3. Implement training scripts in `Deep_Learning/`
4. Add deployment scripts in `Clinical_Deploy/`

### Adding New Models
1. Implement model architecture in appropriate directory
2. Add training script with configuration
3. Implement evaluation and deployment scripts
4. Add to comparison pipeline

##  Documentation

Each subdirectory contains detailed README files:
- [Preprocessing Documentation](Preprocessing/README.md)
- [Feature Extraction Documentation](Feature_Extraction/README.md)
- [Classical Learning Documentation](Classic_Learning/README.md)
- [Deep Learning Documentation](Deep_Learning/README.md)
- [Clinical Deployment Documentation](Clinical_Deploy/README.md)
- [Comparison Documentation](Comparison/README.md)
- [Visualization Documentation](Visualise/README.md)

##  Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive documentation for new scripts
3. Include example usage and configuration files
4. Add appropriate error handling and logging
5. Test scripts with sample data before submission

##  Support

For script-related issues:
- Check the specific subdirectory README files
- Review example configurations and usage
- Check log files for detailed error messages
- Validate input data formats and paths
