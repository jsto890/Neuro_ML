# Label Generation Directory

This directory contains scripts for generating and validating subject labels for machine learning training and evaluation. The scripts automatically scan preprocessed medical imaging data and create standardized label files that map subject IDs to disease classifications.

## 📁 Directory Structure

```
Label_Generation/
├── README.md                     # This file
├── generate_mri_labels.py        # Generate MRI subject labels
├── generate_pet_labels.py        # Generate PET subject labels
└── generate_spect_labels.py      # Generate SPECT subject labels
```

## 🎯 Purpose

The label generation scripts serve several critical functions:

1. **Automated Label Creation**: Scan preprocessed imaging data and automatically generate subject labels
2. **Data Validation**: Validate existing labels against imaging records
3. **Standardization**: Ensure consistent label formats across modalities
4. **Quality Control**: Identify mismatches and missing subjects
5. **Incremental Updates**: Update labels when new data becomes available

## 🏷️ Label Mapping

### Standard Disease Labels
All scripts use a canonical label mapping for consistency:

- **CN (Cognitive Normal)**: `0`
- **AD (Alzheimer's Disease)**: `1` 
- **PD (Parkinson's Disease)**: `2`

### Modality-Specific Mappings

#### MRI Labels (`generate_mri_labels.py`)
- **Full multiclass**: `0=CN, 1=AD, 2=PD`
- **Binary option**: `0=CN, 1=AD+PD` (configurable)

#### PET Labels (`generate_pet_labels.py`)
- **Full multiclass**: `0=CN, 1=AD, 2=PD`
- **Binary option**: `0=CN, 1=AD+PD` (configurable)

#### SPECT Labels (`generate_spect_labels.py`)
- **Binary classification**: `0=CN, 1=PD`
- **Note**: AD subjects are excluded from SPECT analysis

## 🔄 Scripts Overview

### MRI Label Generation (`generate_mri_labels.py`)

#### Purpose
Generate labels for structural MRI subjects based on sMRIprep preprocessed data.

#### Required File Pattern
```
{sub-XXXX}/anat/{sub-XXXX}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz
```

#### Usage
```bash
# Generate labels
python generate_mri_labels.py

# Validate existing labels
python generate_mri_labels.py --validate-only

# Validate with detailed reports
python generate_mri_labels.py --validate-only --write-reports --out-dir ~/path/to/reports
```

#### Features
- **Automatic scanning**: Scans sMRIprep output directory
- **File validation**: Ensures required zscore files exist
- **Label validation**: Validates against imaging records
- **Incremental updates**: Updates existing labels when new data is added
- **Mismatch detection**: Identifies label inconsistencies

### PET Label Generation (`generate_pet_labels.py`)

#### Purpose
Generate labels for PET subjects based on SUVR preprocessed data.

#### Required File Patterns
```
{sub-XXXX}_*_PET_{disease}_SUVR_s2_brain_soft4.nii.gz
{sub-XXXX}_*_PET_{disease}_SUVR_s2_brain_soft4.nii
{sub-XXXX}_*_PET_{disease}_SUVR.nii.gz
{sub-XXXX}_*_PET_{disease}_SUVR.nii
```

#### Usage
```bash
# Generate labels
python generate_pet_labels.py

# Validate existing labels
python generate_pet_labels.py --validate-only

# Validate with detailed reports
python generate_pet_labels.py --validate-only --write-reports --out-dir ~/path/to/reports
```

#### Features
- **Multiple file patterns**: Supports various SUVR file naming conventions
- **Disease folder scanning**: Scans organized disease-specific folders
- **File preference**: Prefers newer file formats (soft4, .nii.gz)
- **Subject validation**: Ensures subjects exist in imaging records
- **Automatic cleanup**: Removes subjects without valid PET files

### SPECT Label Generation (`generate_spect_labels.py`)

#### Purpose
Generate labels for SPECT subjects based on postprocessed DSPECT data.

#### Directory Structure
```
SPECT/
├── CN_SPECT_PPMI_postprocessed/
│   └── [subject folders with SPECT files]
└── PD_SPECT_PPMI_postprocessed/
    └── [subject folders with SPECT files]
```

#### Usage
```bash
# Generate labels with default paths
python generate_spect_labels.py

# Generate labels with custom data root
python generate_spect_labels.py --data-root ~/path/to/spect/data

# Generate labels with custom output path
python generate_spect_labels.py --output-path ~/path/to/spect_labels.csv

# Validate existing labels
python generate_spect_labels.py --validate-only

# Validate with detailed reports
python generate_spect_labels.py --validate-only --write-reports --out-dir ~/path/to/reports
```

#### Features
- **Binary classification**: CN vs PD (AD excluded)
- **Flexible data paths**: Configurable input and output paths
- **Postprocessed data**: Works with DSPECT pipeline output
- **Subject scanning**: Automatically finds subjects in disease folders
- **Validation reports**: Detailed mismatch analysis

## 📊 Output Format

### Label CSV Structure
All scripts generate CSV files with the following structure:

```csv
subject_id,label
sub-001,0
sub-002,1
sub-003,2
```

### Validation Reports
When using `--write-reports`, additional CSV files are generated:

#### Mismatch Report (`*_mismatches.csv`)
```csv
subject_id,label_csv,label_csv_name,expected_label,expected_name
sub-001,1,AD,0,CN
sub-002,0,CN,2,PD
```

#### Missing Records Report (`*_subjects_not_in_records.csv`)
```csv
subject_id,label_csv,records
sub-999,0,None
```

## 🔧 Configuration

### Default Paths
All scripts use consistent default paths:

```python
DATA_DIR = Path.home() / "reseng202500013-ndd-ml" / "data"

# MRI
PREPROCESSED_MRI_DIR = DATA_DIR / "preprocessed" / "MRI" / "smriprep"
OUTPUT_LABELS_PATH = DATA_DIR / "mri_labels.csv"

# PET
PREPROCESSED_PET_DIR = DATA_DIR / "preprocessed" / "PET"
OUTPUT_LABELS_PATH = DATA_DIR / "pet_labels.csv"

# SPECT
SPECT_ROOT = DATA_DIR / "preprocessed" / "SPECT"
OUTPUT_LABELS_PATH = SPECT_ROOT / "spect_labels.csv"

# Common
IMAGING_RECORDS_PATH = DATA_DIR / "imaging_records.csv"
```

### Custom Paths
All scripts support custom path configuration via command-line arguments:

```bash
# Custom data directories
python generate_mri_labels.py --data-dir ~/custom/mri/path

# Custom output paths
python generate_pet_labels.py --output ~/custom/pet_labels.csv

# Custom records file
python generate_spect_labels.py --records ~/custom/imaging_records.csv
```

## 📈 Validation Process

### Label Validation
All scripts include comprehensive validation:

1. **File existence**: Check if required files exist
2. **Format validation**: Validate CSV structure
3. **Label consistency**: Compare labels against imaging records
4. **Mismatch detection**: Identify inconsistencies
5. **Missing subjects**: Find subjects not in records

### Validation Output
```
[VALIDATION] MRI labels vs imaging_records (0=CN,1=AD,2=PD)
  Total in labels: 150
  Not found in records: 2
  Mismatches: 3

AD->CN 1
AD->PD 0
CN->AD 1
CN->PD 1
PD->AD 0
PD->CN 0

AD_correct 45
CN_correct 50
PD_correct 52
AD_wrong 1
CN_wrong 2
PD_wrong 0
```

## 🚨 Common Issues

### File Path Issues
1. **Missing directories**: Ensure preprocessed data directories exist
2. **Permission errors**: Check file and directory permissions
3. **Path configuration**: Verify default paths match your setup

### Data Issues
1. **Missing files**: Ensure required files exist for each subject
2. **File naming**: Check file naming conventions
3. **Directory structure**: Verify expected directory organization

### Label Issues
1. **Mismatches**: Review imaging records for accuracy
2. **Missing subjects**: Check if subjects are in imaging records
3. **Format issues**: Ensure CSV format is correct

## 🔍 Debugging

### Validation Mode
Use `--validate-only` to check existing labels without regenerating:

```bash
# Check existing labels
python generate_mri_labels.py --validate-only

# Check with detailed reports
python generate_mri_labels.py --validate-only --write-reports
```

### Verbose Output
All scripts provide detailed logging:
- File scanning progress
- Subject discovery
- Label validation results
- Mismatch details
- Summary statistics

### Common Debugging Steps
1. **Check file paths**: Verify all paths exist and are accessible
2. **Validate data structure**: Ensure expected file patterns exist
3. **Review imaging records**: Check imaging_records.csv format
4. **Test with sample data**: Use small datasets for testing
5. **Check permissions**: Ensure read/write access to all directories

## 📚 Dependencies

### Required Libraries
- **pandas**: Data manipulation and CSV handling
- **pathlib**: Path operations
- **argparse**: Command-line argument parsing
- **os**: Operating system interface

### Input Requirements
- **imaging_records.csv**: Master subject and disease information
- **Preprocessed data**: Modality-specific preprocessed imaging data
- **File patterns**: Specific file naming conventions for each modality

## 🚀 Usage Examples

### Complete Workflow
```bash
# 1. Generate MRI labels
cd Scripts/Label_Generation
python generate_mri_labels.py

# 2. Generate PET labels
python generate_pet_labels.py

# 3. Generate SPECT labels
python generate_spect_labels.py

# 4. Validate all labels
python generate_mri_labels.py --validate-only --write-reports
python generate_pet_labels.py --validate-only --write-reports
python generate_spect_labels.py --validate-only --write-reports
```

### Custom Configuration
```bash
# Custom paths for SPECT
python generate_spect_labels.py \
    --data-root ~/custom/spect/data \
    --output-path ~/custom/spect_labels.csv \
    --records-path ~/custom/imaging_records.csv
```

### Validation Only
```bash
# Validate existing labels with reports
python generate_mri_labels.py \
    --validate-only \
    --write-reports \
    --out-dir ~/path/to/validation_reports
```

## 📞 Support

For label generation issues:
- Check file paths and directory structure
- Validate imaging_records.csv format
- Review file naming conventions
- Check script output for detailed error messages
- Test with sample data first
- Verify permissions and access rights
