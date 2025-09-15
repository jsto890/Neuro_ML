# Labeling System Documentation

## Overview

The P4P project uses a consistent labeling system across all deep learning and classical machine learning scripts. This document explains the labeling scheme and where it's implemented.

## Label Mapping

**Current Label System:**
- **0 = AD (Alzheimer's Disease)**
- **1 = CN (Control/Healthy)**
- **2 = PD (Parkinson's Disease)**

## Where Labels Are Used

### 1. **Deep Learning Training Scripts**

#### **MRI Training (`Scripts/Deep_Learning/MRI/train_smri.py`)**
- **Label Mapping**: `{0: 'AD', 1: 'CN', 2: 'PD'}` (line 267)
- **Usage**: 
  - Binary classification: `--labels 0 1` (AD vs CN)
  - Multi-class: `--labels 0 1 2` (AD vs CN vs PD)
  - Help text: "Labels to include in training (e.g., 0 1 for AD vs CN)"

#### **PET Training (`Scripts/Deep_Learning/PET/train_pet.py`)**
- **Label Mapping**: `{0: 'AD', 1: 'CN', 2: 'PD'}` (line 267)
- **Usage**: Same as MRI training

#### **Transformer Training Scripts**
- **MRI**: `Scripts/Deep_Learning/MRI/train_transformers.py`
- **PET**: `Scripts/Deep_Learning/PET/train_transformers.py`
- **Help text**: "Labels to include in training (e.g., 0 1 for AD vs CN)"

### 2. **Configuration Files**

#### **Transformer Configs**
- `Scripts/Deep_Learning/MRI/config_transformers.yaml`
- `Scripts/Deep_Learning/MRI/config_hardware_optimized.yaml`
- `Scripts/Deep_Learning/PET/config_transformers.yaml`
- `Scripts/Deep_Learning/PET/config_hardware_optimized.yaml`

**All contain:**
```yaml
num_classes: 2  # Binary classification (AD vs CN)
```

### 3. **Evaluation Scripts**

#### **Model Evaluation**
- `Scripts/Deep_Learning/MRI/evaluate_model.py`
- `Scripts/Deep_Learning/PET/evaluate_model.py`
- **Plot titles**: "Model Evaluation Results - AD vs CN Classification"

### 4. **Classical Learning Scripts**

#### **Binary Classification Checks**
- `Scripts/Classic_Learning/Optimised/enhanced_fdr_classifier.py`
- `Scripts/Classic_Learning/Optimised/improved_optimized_classifier.py`
- `run_fair_fdr_vs_default.py`

**All contain validation:**
```python
if len(unique_labels) != 2 or not all(label in [0, 1] for label in unique_labels):
    raise ValueError(f"Expected binary labels [0, 1], got: {unique_labels}")
```

### 5. **Documentation Files**

#### **README Files**
- `Scripts/Deep_Learning/MRI/README_TRANSFORMERS.md`
- `Scripts/Deep_Learning/PET/README_TRANSFORMERS.md`
- `Scripts/Deep_Learning/MRI/TEST_EVALUATION_STRATEGIES.md`

**All examples use:**
```bash
--labels 0 1  # AD vs CN
--labels 0 1 2  # AD vs CN vs PD
```

## Usage Examples

### **Binary Classification (AD vs CN)**
```bash
# MRI Training
python train_smri.py --master_csv mri_labels.csv --data_root /path/to/data --labels 0 1

# PET Training  
python train_pet.py --master_csv pet_labels.csv --data_root /path/to/data --labels 0 1

# Transformer Training
python train_transformers.py --master_csv mri_labels.csv --data_root /path/to/data --labels 0 1 --model VisionTransformer3D
```

### **Multi-class Classification (AD vs CN vs PD)**
```bash
# MRI Training
python train_smri.py --master_csv mri_labels.csv --data_root /path/to/data --labels 0 1 2

# PET Training
python train_pet.py --master_csv pet_labels.csv --data_root /path/to/data --labels 0 1 2
```

### **Classical Learning**
```bash
# Binary classification only
python radiomics_classifier.py --labels labels.csv --binary_only True
```

## Data Format

### **CSV Label Files**
Expected format:
```csv
subject_id,label
sub-001,0
sub-002,1
sub-003,2
```

Where:
- `0` = AD (Alzheimer's Disease)
- `1` = CN (Control/Healthy)  
- `2` = PD (Parkinson's Disease)

## Important Notes

1. **Consistency**: All scripts now use the same labeling scheme (AD=0, CN=1, PD=2)

2. **Binary vs Multi-class**: 
   - Binary: Use `--labels 0 1` (AD vs CN)
   - Multi-class: Use `--labels 0 1 2` (AD vs CN vs PD)

3. **Model Output**: 
   - Binary: Output shape `[batch_size, 2]` (probabilities for AD vs CN)
   - Multi-class: Output shape `[batch_size, 3]` (probabilities for AD vs CN vs PD)

4. **Evaluation Metrics**:
   - Binary: AUC, accuracy, precision, recall, F1-score
   - Multi-class: Macro-averaged metrics across all classes

5. **Threshold Optimization**: 
   - Binary: Optimizes threshold for AD vs CN classification
   - Multi-class: Uses max probability for classification

## Troubleshooting

### **Common Issues**

1. **"Expected binary labels [0, 1]" error**:
   - Ensure you're using `--labels 0 1` for binary classification
   - Check that your CSV file only contains labels 0 and 1

2. **Model loading errors**:
   - Ensure the number of classes matches your label selection
   - Binary: `num_classes=2`
   - Multi-class: `num_classes=3`

3. **Incorrect predictions**:
   - Verify label mapping in your data
   - Check that model was trained with the same label scheme

### **Validation Commands**

```bash
# Check label distribution in your CSV
python -c "
import pandas as pd
df = pd.read_csv('your_labels.csv')
print('Label distribution:')
print(df['label'].value_counts().sort_index())
print('\nUnique labels:', sorted(df['label'].unique()))
"
```

## Migration from Old Labeling

If you have data with different labeling schemes, you'll need to update your CSV files to match the current system:

- **Old**: CN=0, AD=1 → **New**: AD=0, CN=1
- **Old**: CN=0, AD=1, PD=2 → **New**: AD=0, CN=1, PD=2

Use pandas to remap labels:
```python
import pandas as pd

# Read your CSV
df = pd.read_csv('old_labels.csv')

# Remap labels (example: old CN=0, AD=1 to new AD=0, CN=1)
label_mapping = {0: 1, 1: 0}  # CN=0→1, AD=1→0
df['label'] = df['label'].map(label_mapping)

# Save updated CSV
df.to_csv('new_labels.csv', index=False)
``` 