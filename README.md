# P4P Research Project: Early Detection of Neurodegenerative Diseases

## 🧠 Project Overview
This research project focuses on developing machine learning models for early detection of Alzheimer's Disease (AD) and Parkinson's Disease (PD) using multimodal medical imaging data. The project leverages PET, MRI, and DaTSPECT scans to identify early biomarkers of these neurodegenerative conditions.

## 📋 Key Features
- **Multimodal Data Processing**: Integrated pipeline for PET, MRI, and DaTSPECT image processing
- **Standardized Preprocessing**: Automated data standardization and quality control
- **Feature Extraction**: Advanced feature extraction using pyRadiomics
- **Machine Learning Pipeline**: End-to-end ML workflow for disease classification
- **Quality Control**: Comprehensive QC reports and validation metrics

## 📁 Project Structure
```
P4P/
├── Scripts/                    # Main processing scripts
│   ├── Preprocessing/          # Data preprocessing pipelines
│   │   ├── PET/                # PET image processing
│   │   ├── MRI/                # MRI processing
│   │   └── DaTSPECT/           # DaTSPECT processing
│   ├── Feature_Extraction/     # Feature extraction using pyRadiomics
│   ├── Deep_Learning/          # Deep learning models and training
│   │   └── MRI/                # MRI-specific deep learning
│   └── Visualise/              # Visualization tools
├── Templates/                  # Reference templates
│   ├── PET/                    # PET templates
│   ├── MRI/                    # MRI templates
│   └── SPECT/                  # SPECT templates
├── requirements.txt            # Python package requirements
├── environment.yml             # Conda environment configuration
└── config.yaml                 # Global configuration file
```

## 🚀 Getting Started

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

### Usage

## 📊 Data Processing Pipeline

### 1. Preprocessing
- DICOM to NIfTI conversion
- Motion correction
- Intensity normalization
- Quality control checks

### 2. Feature Extraction
- Radiomics feature calculation
- Feature selection and reduction
- Feature validation

### 3. Machine Learning
- Data splitting and validation
- Model training and evaluation
- Performance metrics calculation

## 📝 Documentation
- Detailed documentation is available in the `docs/` directory
- API documentation can be generated using Sphinx
- Example notebooks are provided in `docs/examples/`

## 👥 Collaborators
- Joseph Storey
- Jackson Schofield

## 📞 Support
For questions and support, please open an issue in the GitHub repository.
