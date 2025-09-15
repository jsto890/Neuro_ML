# New Data Balancing Strategy for Deep Learning

## Overview

The deep learning training scripts have been updated with a new data balancing strategy that addresses the limitations of the previous approach. The new strategy ensures better utilization of available data while maintaining balanced training and validation sets.

## Previous Strategy (Old)

The previous strategy performed undersampling on **all** sets (train/val/test):

1. Undersample the entire dataset to balance classes
2. Split the balanced dataset into train/val/test sets
3. **Problem**: All unused subjects were discarded, leading to data waste

## New Strategy

The new strategy implements a more efficient approach:

1. **Undersample** to balance classes (keeping track of removed subjects)
2. **Split** the balanced dataset into 70/20/10 (train/val/test)
3. **Add** the remaining unused subjects to the test set

### Benefits

- **No data waste**: All subjects are used in training
- **Balanced training/validation**: Ensures fair model evaluation
- **Larger test set**: More comprehensive final evaluation
- **Better generalization**: Test set includes both balanced and original distribution samples

## Implementation

### Function: `balance_and_split_dataset()`

```python
def balance_and_split_dataset(df, val_ratio=0.2, test_ratio=0.1, random_state=None):
    """
    New data balancing strategy:
    1. Undersample to balance classes (keeping track of removed subjects)
    2. Split balanced dataset into 70/20/10 (train/val/test)
    3. Add removed subjects to test set
    """
```

### Parameters

- `df`: DataFrame with 'subject_id' and 'label' columns
- `val_ratio`: Proportion of balanced data for validation (default: 0.2)
- `test_ratio`: Proportion of balanced data for test (default: 0.1)
- `random_state`: Random seed for reproducibility

### Returns

- `train_df`: Training set (70% of balanced data)
- `val_df`: Validation set (20% of balanced data)
- `test_df`: Test set (10% of balanced data + all removed subjects)
- `removed_subjects_df`: Subjects removed during undersampling

## Usage

### Command Line Arguments

The following scripts now support the new balancing strategy:

- `Scripts/Deep_Learning/MRI/train_smri.py`
- `Scripts/Deep_Learning/PET/train_pet.py`
- `Scripts/Deep_Learning/MRI/train_transformers.py`
- `Scripts/Deep_Learning/PET/train_transformers.py`

### Example Usage

```bash
# Use new balancing strategy with default 70/20/10 split
python train_smri.py --master_csv data/mri_labels.csv --data_root data/preprocessed/MRI \
    --labels 0 1 --balance_dataset --random_seed 42

# Use custom split ratios
python train_smri.py --master_csv data/mri_labels.csv --data_root data/preprocessed/MRI \
    --labels 0 1 --balance_dataset --val_ratio 0.15 --test_ratio 0.15 --random_seed 42

# Use original strategy (no balancing)
python train_smri.py --master_csv data/mri_labels.csv --data_root data/preprocessed/MRI \
    --labels 0 1 --random_seed 42
```

### Default Values

- `--val_ratio`: 0.2 (20% of balanced data)
- `--test_ratio`: 0.1 (10% of balanced data)
- `--balance_dataset`: Flag to enable new strategy

## Example Output

When using the new strategy, you'll see output like:

```
Original class distribution:
  Label 0: 100 subjects
  Label 1: 200 subjects
  Label 2: 150 subjects

Balanced class distribution (undersampled):
  Label 0: 100 subjects
  Label 1: 100 subjects
  Label 2: 100 subjects

Removed subjects: 150 subjects
  Label 1: 100 subjects
  Label 2: 50 subjects

Splitting balanced dataset (train: 70.0%, val: 20.0%, test: 10.0%)

Final dataset splits:
Training set: 210 subjects
Validation set: 60 subjects
Test set: 210 subjects (balanced: 30, added: 180)
```

## Testing

A test script is available to demonstrate the new strategy:

```bash
python Scripts/Deep_Learning/test_new_balancing.py
```

This script creates a synthetic imbalanced dataset and shows how the new strategy works step-by-step.

## Migration from Old Strategy

### For Existing Scripts

If you were using the old `--balance_dataset` flag, the behavior has changed:

- **Old**: Undersampled all sets, discarded unused subjects
- **New**: Undersamples train/val sets, adds unused subjects to test set

### Backward Compatibility

- The `--balance_dataset` flag still exists but now uses the new strategy
- Original behavior is preserved when `--balance_dataset` is not used
- All other arguments remain the same

## Files Modified

The following files have been updated with the new balancing strategy:

1. `Scripts/Deep_Learning/MRI/train_smri.py`
2. `Scripts/Deep_Learning/PET/train_pet.py`
3. `Scripts/Deep_Learning/MRI/train_transformers.py`
4. `Scripts/Deep_Learning/PET/train_transformers.py`
5. `Scripts/Deep_Learning/test_new_balancing.py` (new test script)

## Key Changes

1. **New function**: `balance_and_split_dataset()` added to all training scripts
2. **Updated argument defaults**: `--val_ratio=0.2`, `--test_ratio=0.1`
3. **Updated help text**: Clarifies the new strategy
4. **Enhanced logging**: Shows detailed breakdown of splits and removed subjects
5. **Test script**: Demonstrates the new strategy with synthetic data 