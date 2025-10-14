# Templates Directory

This directory contains template images, masks, and reference files used throughout the P4P pipeline for medical image preprocessing, registration, and analysis.

## 📁 Directory Structure

```
Templates/
├── README.md                     # This file
├── MRI/                          # MRI templates and masks
│   ├── MNI152_T1_1mm_brain.nii.gz
│   └── MNI152_T1_1mm_brain_mask.nii.gz
├── PET/                          # PET templates and masks
│   ├── cereb_mask25_bin.nii.gz
│   ├── FDG_PET_brainmask.nii.gz
│   ├── FDG_PET.nii.gz
│   └── HO_sub_maxprob25.nii.gz
└── SPECT/                        # SPECT templates and masks
    ├── occipital_mask.nii.gz
    └── symFPCITtemplate_MNI_norm.nii
```

## 🧠 MRI Templates

### Files in `MRI/`

#### `MNI152_T1_1mm_brain.nii.gz`
- **Purpose**: Standard MNI152 T1-weighted brain template at 1mm resolution
- **Usage**: Reference template for MRI preprocessing and registration
- **Source**: Montreal Neurological Institute (MNI) standard space
- **Format**: NIfTI compressed (.nii.gz)
- **Dimensions**: 182×218×182 voxels, 1mm isotropic resolution

#### `MNI152_T1_1mm_brain_mask.nii.gz`
- **Purpose**: Brain mask in MNI152 space
- **Usage**: Skull stripping and brain extraction for MRI data
- **Format**: NIfTI compressed (.nii.gz)
- **Content**: Binary mask (0=background, 1=brain tissue)

### Usage Examples

```python
import nibabel as nib
import numpy as np

# Load MNI152 template
template = nib.load('Templates/MRI/MNI152_T1_1mm_brain.nii.gz')
template_data = template.get_fdata()

# Load brain mask
mask = nib.load('Templates/MRI/MNI152_T1_1mm_brain_mask.nii.gz')
mask_data = mask.get_fdata()

# Apply mask to template
brain_only = template_data * mask_data
```

## 🧬 PET Templates

### Files in `PET/`

#### `FDG_PET.nii.gz`
- **Purpose**: FDG-PET template for PET image preprocessing
- **Usage**: Reference template for PET registration and normalization
- **Format**: NIfTI compressed (.nii.gz)
- **Tracer**: Fluorodeoxyglucose (FDG)

#### `FDG_PET_brainmask.nii.gz`
- **Purpose**: Brain mask for FDG-PET images
- **Usage**: Skull stripping and brain extraction for PET data
- **Format**: NIfTI compressed (.nii.gz)
- **Content**: Binary mask (0=background, 1=brain tissue)

#### `cereb_mask25_bin.nii.gz`
- **Purpose**: Cerebellar mask for PET normalization
- **Usage**: Reference region for SUVR (Standardized Uptake Value Ratio) calculation
- **Format**: NIfTI compressed (.nii.gz)
- **Content**: Binary mask (0=background, 1=cerebellar region)

#### `HO_sub_maxprob25.nii.gz`
- **Purpose**: Harvard-Oxford subcortical atlas with 25% probability threshold
- **Usage**: Subcortical region identification and analysis
- **Format**: NIfTI compressed (.nii.gz)
- **Content**: Probabilistic atlas with subcortical regions

### Usage Examples

```python
import nibabel as nib
import numpy as np

# Load FDG-PET template
pet_template = nib.load('Templates/PET/FDG_PET.nii.gz')
pet_data = pet_template.get_fdata()

# Load cerebellar mask for SUVR calculation
cereb_mask = nib.load('Templates/PET/cereb_mask25_bin.nii.gz')
cereb_data = cereb_mask.get_fdata()

# Calculate SUVR (example)
cereb_uptake = np.mean(pet_data[cereb_data > 0])
suvr_image = pet_data / cereb_uptake
```

## 🔬 SPECT Templates

### Files in `SPECT/`

#### `symFPCITtemplate_MNI_norm.nii`
- **Purpose**: Symmetric FP-CIT SPECT template in MNI space
- **Usage**: Reference template for SPECT registration and normalization
- **Format**: NIfTI (.nii)
- **Tracer**: FP-CIT (Ioflupane)
- **Note**: Symmetric version for better registration accuracy

#### `occipital_mask.nii.gz`
- **Purpose**: Occipital region mask for SPECT normalization
- **Usage**: Reference region for SPECT SUVR calculation
- **Format**: NIfTI compressed (.nii.gz)
- **Content**: Binary mask (0=background, 1=occipital region)
- **Rationale**: Occipital cortex typically shows minimal dopaminergic uptake

### Usage Examples

```python
import nibabel as nib
import numpy as np

# Load SPECT template
spect_template = nib.load('Templates/SPECT/symFPCITtemplate_MNI_norm.nii')
spect_data = spect_template.get_fdata()

# Load occipital mask for SUVR calculation
occipital_mask = nib.load('Templates/SPECT/occipital_mask.nii.gz')
occipital_data = occipital_mask.get_fdata()

# Calculate SUVR using occipital reference
occipital_uptake = np.mean(spect_data[occipital_data > 0])
suvr_image = spect_data / occipital_uptake
```

## 🔧 Configuration

### Template Paths in `config.yaml`

```yaml
templates:
  MRI_brain_mask: ~/reseng202500013-ndd-ml/P4P/Templates/MRI/MNI152_T1_1mm_brain.nii.gz
  PET_template: ~/reseng202500013-ndd-ml/P4P/Templates/PET/FDG-PET-template_padded.nii.gz
  PET_brain_mask: ~/reseng202500013-ndd-ml/P4P/Templates/PET/brain_in_petspace.nii.gz
  PET_cereb_mask: ~/reseng202500013-ndd-ml/P4P/Templates/PET/cereb_in_petspace.nii.gz
  SPECT_occipital: ~/reseng202500013-ndd-ml/P4P/Templates/SPECT/occipital_mask.nii.gz
  SPECT_template: ~/reseng202500013-ndd-ml/P4P/Templates/SPECT/symFPCITtemplate_MNI_norm.nii
```

## 📊 Template Specifications

### Spatial Properties

| Template | Dimensions | Voxel Size | Space | Format |
|----------|------------|------------|-------|--------|
| MNI152_T1_1mm_brain | 182×218×182 | 1×1×1 mm | MNI152 | .nii.gz |
| FDG_PET | Variable | Variable | PET Space | .nii.gz |
| symFPCITtemplate | Variable | Variable | MNI Space | .nii |

### Mask Properties

| Mask | Purpose | Content | Threshold |
|------|---------|---------|-----------|
| MNI152_brain_mask | Brain extraction | Binary | 0.5 |
| FDG_PET_brainmask | Brain extraction | Binary | 0.5 |
| cereb_mask25_bin | Cerebellar reference | Binary | 0.25 |
| occipital_mask | Occipital reference | Binary | 0.5 |

## 🚀 Usage in Pipeline

### Preprocessing Scripts

Templates are automatically loaded in preprocessing scripts:

```python
# Example from SPECT preprocessing
from pathlib import Path
import nibabel as nib

# Load template
template_path = Path("Templates/SPECT/symFPCITtemplate_MNI_norm.nii")
template = nib.load(str(template_path))

# Load mask
mask_path = Path("Templates/SPECT/occipital_mask.nii.gz")
mask = nib.load(str(mask_path))
```

### Registration

Templates serve as reference spaces for image registration:

```python
# Example registration workflow
from nibabel.processing import resample_from_to

# Resample image to template space
resampled_image = resample_from_to(
    source_image, 
    template, 
    order=1,  # Linear interpolation
    mode='constant',
    cval=0
)
```

## 🔍 Quality Control

### Template Validation

```python
def validate_template(template_path):
    """Validate template file integrity"""
    try:
        img = nib.load(template_path)
        data = img.get_fdata()
        
        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False, "Contains NaN or infinite values"
        
        # Check dimensions
        if len(data.shape) != 3:
            return False, f"Expected 3D, got {len(data.shape)}D"
        
        return True, "Valid template"
        
    except Exception as e:
        return False, f"Error loading template: {e}"
```

## 📚 References

### MNI152 Template
- **Source**: Montreal Neurological Institute
- **Citation**: Fonov, V., Evans, A.C., Botteron, K., et al. (2011). Unbiased average age-appropriate atlases for pediatric studies. NeuroImage, 54(1), 313-327.

### FDG-PET Template
- **Source**: ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Usage**: Standard FDG-PET reference for Alzheimer's disease studies

### FP-CIT SPECT Template
- **Source**: PPMI (Parkinson's Progression Markers Initiative)
- **Usage**: Standard FP-CIT reference for Parkinson's disease studies

## ⚠️ Important Notes

1. **File Formats**: Ensure compatibility with your processing pipeline
2. **Space Consistency**: All templates should be in the same reference space
3. **Resolution**: Check voxel sizes match your data requirements
4. **Updates**: Templates may need updates for different study populations
5. **Validation**: Always validate template integrity before use

## 🛠️ Maintenance

### Adding New Templates

1. Place new template files in appropriate subdirectory
2. Update `config.yaml` with new paths
3. Document template specifications in this README
4. Add validation functions if needed

### Template Updates

1. Backup existing templates
2. Test new templates with sample data
3. Update configuration files
4. Document changes in this README

## 📞 Support

For template-related issues:
- Check file paths in `config.yaml`
- Validate template file integrity
- Ensure proper file permissions
- Review preprocessing script logs
