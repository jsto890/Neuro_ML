# Radiomics Feature Extraction for P4P Project

This directory contains scripts to extract radiomics features from preprocessed neuroimaging data (MRI, PET, SPECT).

## 📁 Files Overview

- `simple_radiomics.py` - **RECOMMENDED**: Working MRI radiomics extractor (avoids pandas/NumPy compatibility issues)
- `jo_pyradio.py` - Original simple radiomics extractor for MRI data
- `radiomics_extractor.py` - Comprehensive extractor supporting all modalities (may have compatibility issues)
- `test_paths.py` - Test script to verify data paths and structure
- `debug_paths.py` - Debug script to troubleshoot path issues
- `test_pyRadioMics.py` - Original test script (for reference)

## 🚀 Quick Start

### Prerequisites

Install required packages:
```bash
pip install pyradiomics SimpleITK pyyaml pandas
```

**Note**: If you encounter NumPy compatibility issues, you may need to downgrade NumPy:
```bash
conda install numpy=1.24.3 -y
```

### Step 1: Test Your Setup

First, verify that your data paths are correct:

```bash
cd Scripts/Feature_Extraction/pyRadioMics/
python3 test_paths.py
```

### Step 2: Extract MRI Radiomics Features

#### **RECOMMENDED: Simple Radiomics (Working Version)**
```bash
python3 simple_radiomics.py --labels ~/reseng202500013-ndd-ml/data/mri_labels.csv --output-dir ~/reseng202500013-ndd-ml/data/ --config ~/reseng202500013-ndd-ml/P4P/config.yaml
```

#### Alternative: Comprehensive Version (may have compatibility issues)
```bash
python3 radiomics_extractor.py --modality MRI --labels ~/reseng202500013-ndd-ml/data/mri_labels.csv --output-dir ~/reseng202500013-ndd-ml/data/ --config ~/reseng202500013-ndd-ml/P4P/config.yaml
```

## 📊 Data Structure

The scripts expect the following data structure:

### MRI Data
```
/home/jsto890/reseng202500013-ndd-ml/data/preprocessed/MRI/smriprep/
├── {subject_id}/
│   └── anat/
│       └── {subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz
```

### Labels
```
/home/jsto890/reseng202500013-ndd-ml/data/
├── mri_labels.csv
├── train_labels.csv
└── val_labels.csv
```

Each CSV should have columns: `subject_id, label`

## 📈 Output

The scripts will generate CSV files with radiomics features:
- `radiomics_MRI_mri_labels.csv` (from simple_radiomics.py)
- `radiomics_MRI_train_labels.csv`
- `radiomics_MRI_val_labels.csv`

Each CSV contains:
- `subject_id`: Subject identifier (e.g., sub-CLB00151)
- `label`: Class label (0, 1, 2, etc.)
- Plus ~1000+ radiomics features

## 🔧 Configuration

The scripts use `config.yaml` in the project root to find data paths. The key path is:
```yaml
preprocessed_data:
  smri_p: /home/jsto890/reseng202500013-ndd-ml/data/preprocessed/MRI/smriprep
```

## 🐛 Troubleshooting

### Common Issues

1. **NumPy Compatibility Error**
   ```
   ImportError: numpy.core.multiarray failed to import
   ```
   **Solution**: Downgrade NumPy
   ```bash
   conda install numpy=1.24.3 -y
   ```

2. **"Image not found" Errors**
   - Check that your config.yaml has the correct path
   - Verify the MRI data directory structure
   - Use `debug_paths.py` to test paths

3. **"Config file not found"**
   - Use the full path to config.yaml:
   ```bash
   --config ~/reseng202500013-ndd-ml/P4P/config.yaml
   ```

4. **Pandas/NumPy Issues**
   - Use `simple_radiomics.py` instead of `radiomics_extractor.py`
   - This version avoids pandas dependency issues

### Debug Tools

**Test Paths:**
```bash
python3 test_paths.py
```

**Debug Specific Paths:**
```bash
python3 debug_paths.py
```

## 🧬 Features Extracted

The scripts extract all default PyRadiomics features:
- **First Order** (~20 features): Mean, variance, skewness, kurtosis, etc.
- **Shape** (~15 features): Volume, surface area, sphericity, etc.
- **GLCM** (~24 features): Gray Level Co-occurrence Matrix features
- **GLRLM** (~16 features): Gray Level Run Length Matrix features
- **GLSZM** (~16 features): Gray Level Size Zone Matrix features
- **GLDM** (~14 features): Gray Level Dependence Matrix features
- **NGTDM** (~5 features): Neighboring Gray Tone Difference Matrix features

## 📋 Script Comparison

| Script | Status | Use Case | Dependencies |
|--------|--------|----------|--------------|
| `simple_radiomics.py` | ✅ **Working** | Production MRI extraction | Minimal (no pandas) |
| `radiomics_extractor.py` | ⚠️ May have issues | Multi-modality extraction | Full stack |
| `test_paths.py` | ✅ Working | Path validation | Standard |
| `debug_paths.py` | ✅ Working | Path debugging | Standard |

## 🔄 Next Steps

After extracting radiomics features, you can:

1. **Run Classical Learning Pipeline:**
   ```bash
   cd ../Classic_Learning/
   python3 run_classical.py
   ```

2. **Analyze Feature Importance:**
   - Check the generated CSV for feature rankings
   - Use the features in machine learning models

3. **Extract Other Modalities:**
   - PET radiomics (when PET data is available)
   - SPECT radiomics (when SPECT data is available)

## 🎯 Recommended Workflow

1. **Test Setup**: `python3 test_paths.py`
2. **Extract Features**: `python3 simple_radiomics.py --labels ~/reseng202500013-ndd-ml/data/mri_labels.csv --output-dir ~/reseng202500013-ndd-ml/data/ --config ~/reseng202500013-ndd-ml/P4P/config.yaml`
3. **Run Classical Learning**: `cd ../Classic_Learning/ && python3 run_classical.py`

## 📚 References

- [PyRadiomics Documentation](https://pyradiomics.readthedocs.io/)
- [SimpleITK Documentation](https://simpleitk.org/)
- [Radiomics Feature Definitions](https://pyradiomics.readthedocs.io/en/latest/features.html) 