# DICOM to NIfTI Converter for CN_SPECT_PPMI

This script converts all DICOM files in the CN_SPECT_PPMI folder to NIfTI format with proper organization.

## Features

- **Automatic DICOM discovery**: Recursively finds all DICOM files in the specified folder
- **Smart organization**: Groups files by scan series and organizes output by subject ID and scan date
- **Metadata preservation**: Extracts and preserves DICOM metadata in BIDS JSON format
- **Backup**: Keeps original DICOM files as backup
- **Compression**: Outputs compressed NIfTI files (.nii.gz)

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pydicom dcm2niix nibabel
```

## Usage

### 1. Test DICOM Access

First, test if the script can access your DICOM files:

```bash
python3 test_dicom.py
```

This will verify:
- The DICOM file path exists
- The file can be read
- Basic metadata can be extracted

### 2. Run Conversion

Convert all DICOM files to NIfTI:

```bash
python3 dcm_convert.py
```

## Input Structure

The script expects DICOM files in this structure:
```
/Users/jacksonschofield/Desktop/CN_SPECT_PPMI/
├── 3000/
│   └── Raw_Data/
│       └── 2011-01-20_16_28_47.0/
│           └── I248908/
│               └── PPMI_3000_NM_Raw_Data_br_raw_20110805101009028_1_S117534_I248908.dcm
└── [other subjects...]
```

## Output Structure

The script creates organized output on your Desktop:

```
~/Desktop/CN_SPECT_PPMI_NIfTI/
├── Subject_3000/
│   └── Scan_20110120_NM/
│       ├── 3000_20110120_NM_[series_description].nii.gz
│       ├── 3000_20110120_NM_[series_description].json
│       └── Original_DICOM/
│           └── [original DICOM files]
└── [other subjects...]
```

## What Gets Preserved

- **NIfTI files**: Compressed 3D/4D image data
- **BIDS JSON**: Complete DICOM metadata in standardized format
- **Original DICOM**: Backup copies of source files
- **Folder structure**: Organized by subject, date, and modality

## Troubleshooting

### Common Issues

1. **"dcm2niix not found"**: The script will attempt to install it automatically
2. **Permission errors**: Ensure you have read access to the DICOM folder
3. **Memory issues**: For very large datasets, the script processes one series at a time

### Manual Installation

If automatic installation fails:

```bash
# For macOS
brew install dcm2niix

# For pip
pip install dcm2niix

# For conda
conda install -c conda-forge dcm2niix
```

## Notes

- The script automatically detects scan series and groups related DICOM files
- Each series gets its own output folder with descriptive naming
- Original DICOM files are preserved as backup
- Output is compressed to save disk space
- Progress is displayed for each conversion step
