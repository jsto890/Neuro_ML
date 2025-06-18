# Radiomics Feature Extraction for P4P Project

This directory contains scripts to extract radiomics features from preprocessed neuroimaging data (MRI, PET, SPECT).

## Files

- `jo_pyradio.py` - Simple radiomics extractor for MRI data
- `radiomics_extractor.py` - Comprehensive extractor supporting all modalities
- `test_paths.py` - Test script to verify data paths and structure
- `test_pyRadioMics.py` - Original test script (for reference)

## Prerequisites

Install required packages:
```bash
pip install pyradiomics SimpleITK pyyaml pandas
```

## Data Structure

The scripts expect the following data structure:

### MRI Data
```
/home/jsto890/reseng202500013-ndd-ml/data/preprocessed/smriprep/
├── {subject_id}/
│   └── anat/
│       └── {subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz
```

### Labels
```
/home/jsto890/reseng202500013-ndd-ml/data/
├── train_labels.csv
└── val_labels.csv
```

Each CSV should have columns: `subject_id, label`

## Usage

### 1. Test Your Setup
First, verify that your data paths are correct:

```bash
cd Scripts/Feature_Extraction/pyRadioMics/
python test_paths.py
```

### 2. Extract MRI Radiomics Features

#### Simple version (jo_pyradio.py):
```bash
python jo_pyradio.py
```

#### Comprehensive version (radiomics_extractor.py):
```bash
python3 radiomics_extractor.py --modality MRI --labels Labels/train_labels.csv
python3 radiomics_extractor.py --modality MRI --labels Labels/val_labels.csv
```

### 3. Extract PET Radiomics Features
```bash
python3 radiomics_extractor.py --modality PET --labels Labels/train_labels.csv
python3 radiomics_extractor.py --modality PET --labels Labels/val_labels.csv
```

### 4. Extract SPECT Radiomics Features
```bash
python3 radiomics_extractor.py --modality SPECT --labels Labels/train_labels.csv
python3 radiomics_extractor.py --modality SPECT --labels Labels/val_labels.csv
```

## Output

The scripts will generate CSV files with radiomics features:
- `radiomics_MRI_train_labels.csv`
- `radiomics_MRI_val_labels.csv`
- `radiomics_PET_train_labels.csv`
- etc.

Each CSV contains:
- `subject_id`: Subject identifier
- `label`: Class label (0, 1, 2, etc.)
- `modality`: Imaging modality
- Plus ~1000+ radiomics features

## Configuration

The scripts use `config.yaml` in the project root to find data paths. Make sure this file exists and contains the correct paths for your system.

## Troubleshooting

1. **Missing images**: Check that your preprocessed data exists and follows the expected directory structure
2. **Permission errors**: Ensure you have read access to the data directories
3. **Memory issues**: For large datasets, consider processing in batches
4. **PyRadiomics errors**: Make sure your NIfTI files are valid and not corrupted

## Features Extracted

The scripts extract all default PyRadiomics features:
- **First Order**: Statistics of intensity values
- **Shape**: 3D shape features
- **GLCM**: Gray Level Co-occurrence Matrix
- **GLRLM**: Gray Level Run Length Matrix
- **GLSZM**: Gray Level Size Zone Matrix
- **GLDM**: Gray Level Dependence Matrix
- **NGTDM**: Neighboring Gray Tone Difference Matrix

## Customization

To modify feature extraction parameters, edit the `RadiomicsFeatureExtractor()` initialization in the scripts. See the [PyRadiomics documentation](https://pyradiomics.readthedocs.io/) for available options. 