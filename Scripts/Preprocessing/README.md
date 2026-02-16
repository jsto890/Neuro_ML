# Preprocessing Directory

This directory contains all preprocessing pipelines for converting raw medical imaging data into machine learning-ready formats. The preprocessing pipelines handle DICOM to NIfTI conversion, image registration, normalization, and quality control.

##  Directory Structure

```
Preprocessing/
├── README.md                     # This file
├── DCM_TO_NIFTI/             # DICOM to NIfTI conversion
│   ├── README.md                # DICOM conversion documentation
│   ├── ultimate_converter_fixed.py
│   ├── ultimate_converter.py
│   ├── dcm_convert_improved.py
│   ├── dcm_convert.py
│   ├── spect_dcm.py
│   ├── test_dicom.py
│   └── install_alternative_tools.py
├── MRI/                          # MRI preprocessing
│   ├── 02_smriprep_run.py
│   ├── 03_zscore_skull_strip.py
│   ├── delete_duplicate_smriprep_subjects.py
│   └── find_missing_mni_subjects.py
├── PET/                          # PET preprocessing
│   ├── 02_norm_stand.py
│   └── 03_skullstrip.py
└── DSPECT/                       # SPECT preprocessing
    ├── README_FIXES.md
    ├── run_pipeline.py
    ├── 1_reorient.py
    ├── 2_normalise.py
    ├── 3_register.py
    ├── 4_masking.py
    ├── 5_padding.py
    ├── 6_postprocess.py
    └── testing/
        ├── validate_pipeline.py
        └── [other test scripts]
```

##  Preprocessing Workflow

### 1. DICOM to NIfTI Conversion (`01_DCM_TO_NIFTI/`)

#### Purpose
Convert DICOM files to NIfTI format for further processing.

#### Key Scripts
- **`ultimate_converter_fixed.py`**: Multi-tool DICOM conversion with fallback options
- **`dcm_convert.py`**: Basic DICOM to NIfTI conversion
- **`test_dicom.py`**: Validate DICOM file accessibility and format

#### Usage
```bash
cd 01_DCM_TO_NIFTI

# Test DICOM access
python test_dicom.py

# Convert DICOM files
python ultimate_converter_fixed.py --input ~/path/to/dicom --output ~/path/to/nifti
```

#### Features
- **Multi-tool support**: dcm2niix, gdcmsan, pydicom+nibabel
- **Automatic fallback**: If one tool fails, tries alternatives
- **Metadata preservation**: Extracts and preserves DICOM metadata
- **Organized output**: Creates structured output directories

### 2. MRI Preprocessing (`MRI/`)

#### Purpose
Preprocess structural MRI data using sMRIprep and additional steps.

#### Key Scripts
- **`02_smriprep_run.py`**: Run sMRIprep pipeline for MRI preprocessing
- **`03_zscore_skull_strip.py`**: Z-score normalization and skull stripping
- **`delete_duplicate_smriprep_subjects.py`**: Clean up duplicate subjects
- **`find_missing_mni_subjects.py`**: Identify missing MNI-registered subjects

#### Usage
```bash
cd MRI

# Run sMRIprep pipeline
python 02_smriprep_run.py --input ~/path/to/raw/mri --output ~/path/to/processed

# Z-score normalization
python 03_zscore_skull_strip.py --input ~/path/to/smriprep --output ~/path/to/zscore
```

#### Features
- **sMRIprep integration**: Uses standardised preprocessing pipeline
- **Quality control**: Automated quality assessment
- **Z-score normalization**: Standardizes intensity values
- **Skull stripping**: Removes non-brain tissue

### 3. PET Preprocessing (`PET/`)

#### Purpose
Preprocess PET imaging data for SUVR calculation and analysis.

#### Key Scripts
- **`02_norm_stand.py`**: PET normalization and standardisation
- **`03_skullstrip.py`**: PET skull stripping

#### Usage
```bash
cd PET

# PET normalization
python 02_norm_stand.py --input ~/path/to/pet --output ~/path/to/processed

# Skull stripping
python 03_skullstrip.py --input ~/path/to/normalized --output ~/path/to/stripped
```

#### Features
- **SUVR calculation**: Standardized Uptake Value Ratio computation
- **Reference region**: Cerebellar reference region masking
- **Intensity normalization**: Standardizes PET intensity values
- **Quality control**: Validates preprocessing results

### 4. SPECT Preprocessing (`DSPECT/`)

#### Purpose
Complete SPECT preprocessing pipeline for FP-CIT imaging.

#### Key Scripts
- **`run_pipeline.py`**: Complete SPECT preprocessing pipeline
- **`1_reorient.py`**: Image reorientation to standard space
- **`2_normalise.py`**: SPECT normalization using reference regions
- **`3_register.py`**: Image registration to template
- **`4_masking.py`**: Brain masking and region extraction
- **`5_padding.py`**: Image padding and finalization
- **`6_postprocess.py`**: Final postprocessing steps

#### Usage
```bash
cd DSPECT

# Run complete pipeline
python run_pipeline.py --diagnosis CN --force

# Run individual steps
python 1_reorient.py --diagnosis CN
python 2_normalise.py --diagnosis CN --method reference
python 3_register.py --diagnosis CN
python 4_masking.py --diagnosis CN --mask_type occipital
python 5_padding.py --diagnosis CN --shape 91 109 91
python 6_postprocess.py --diagnosis CN
```

#### Features
- **Complete pipeline**: End-to-end SPECT preprocessing
- **Reference normalization**: Uses occipital region for SUVR
- **Template registration**: Registers to symmetric FP-CIT template
- **Quality control**: Comprehensive validation and testing
- **Flexible masking**: Supports different masking strategies

##  Configuration

### SPECT Pipeline Configuration
```bash
# Basic usage
python run_pipeline.py --diagnosis CN

# Advanced options
python run_pipeline.py \
    --diagnosis CN \
    --force \
    --shape 91 109 91 \
    --intensity_norm \
    --mask_type occipital \
    --isotropic
```

### Parameters
- **`--diagnosis`**: Disease group (CN, PD)
- **`--force`**: Force reprocessing even if output exists
- **`--shape`**: Target shape for finalization (default: 91 109 91)
- **`--intensity_norm`**: Apply intensity normalization
- **`--mask_type`**: Mask type (occipital, whole_brain)
- **`--isotropic`**: Resample to isotropic 1mm voxels

##  Output Structure

### DICOM Conversion Output
```
output/
├── Subject_3000/
│   └── Scan_20110120_NM/
│       ├── 3000_20110120_NM_[series].nii.gz
│       ├── 3000_20110120_NM_[series].json
│       └── Original_DICOM/
│           └── [original DICOM files]
└── [other subjects...]
```

### SPECT Pipeline Output
```
SPECT/
├── CN_SPECT_PPMI_reoriented/
├── CN_SPECT_PPMI_normalised/
├── CN_SPECT_PPMI_registered/
├── CN_SPECT_PPMI_masked/
├── CN_SPECT_PPMI_finalised/
└── CN_SPECT_PPMI_postprocessed/
```

##  Testing and Validation

### SPECT Pipeline Testing
```bash
cd DSPECT/testing

# Validate complete pipeline
python validate_pipeline.py --diagnosis CN

# Test individual components
python 1_test.py  # Test reorientation
python 2_test.py  # Test normalization
python 3_test.py  # Test registration
python 4_test_visualise.py  # Test masking
python 5_test.py  # Test padding
python 6_test.py  # Test postprocessing
```

### Quality Control
- **File validation**: Checks file integrity and format
- **Spatial validation**: Verifies image dimensions and spacing
- **Intensity validation**: Checks intensity ranges and distributions
- **Registration validation**: Validates registration quality

##  Common Issues

### DICOM Conversion Issues
1. **File access**: Check file permissions and paths
2. **Format compatibility**: Some DICOM files may not convert properly
3. **Memory issues**: Large files may require more memory

### SPECT Preprocessing Issues
1. **Registration failures**: Check template alignment
2. **Masking issues**: Verify mask quality and coverage
3. **Normalization problems**: Check reference region selection

### MRI Preprocessing Issues
1. **sMRIprep failures**: Check input data quality
2. **Skull stripping**: Verify brain extraction quality
3. **Normalization**: Check intensity standardisation

##  Debugging

### Log Files
- **Conversion logs**: Check DICOM conversion output
- **Preprocessing logs**: Review pipeline execution logs
- **Error logs**: Identify specific failure points

### Validation Scripts
- **File validation**: Use test scripts to verify outputs
- **Visual inspection**: Check intermediate results visually
- **Statistical validation**: Verify preprocessing statistics

##  Dependencies

### External Tools
- **dcm2niix**: DICOM to NIfTI conversion
- **gdcmsan**: Alternative DICOM conversion
- **sMRIprep**: MRI preprocessing pipeline
- **ANTs**: Image registration and normalization

### Python Libraries
- **nibabel**: NIfTI file handling
- **pydicom**: DICOM file handling
- **numpy**: Numerical operations
- **scipy**: Scientific computing
- **pandas**: Data manipulation

##  Performance Optimization

### Memory Management
- **Batch processing**: Process files in batches
- **Memory monitoring**: Monitor memory usage during processing
- **Cleanup**: Remove temporary files after processing

### Parallel Processing
- **Multi-threading**: Use multiple threads for file processing
- **GPU acceleration**: Use GPU for registration when available
- **Distributed processing**: Distribute processing across multiple machines

##  Support

For preprocessing issues:
- Check log files for detailed error messages
- Validate input data formats and quality
- Review configuration parameters
- Test with sample data first
- Check external tool installations
