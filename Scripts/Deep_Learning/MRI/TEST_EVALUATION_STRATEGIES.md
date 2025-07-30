# Test Evaluation Strategies

## 🎯 **Problem Solved**

Previously, the training pipeline had an issue where **only the last fold's model was used for test evaluation**, even though each fold saved its best model. This meant:

- Fold 1 saves best model → `best_smri_model.pth`
- Fold 2 saves best model → **overwrites** `best_smri_model.pth`
- Fold 3 saves best model → **overwrites** `best_smri_model.pth`
- ...
- Fold 5 saves best model → **overwrites** `best_smri_model.pth`
- Test evaluation uses **only Fold 5's model**

## 🔧 **Solutions Implemented**

### **1. Fold-Specific Model Saving**
- Each fold now saves its best model as `best_smri_model_fold_{fold_num}.pth`
- Also saves general `best_smri_model.pth` for backward compatibility
- Models are saved based on validation AUC performance

### **2. Multiple Test Evaluation Strategies**

#### **Strategy 1: `best_fold` (Default)**
```bash
python train_smri.py --test_strategy best_fold
```
- **How it works**: Finds the fold with the highest validation AUC
- **Model used**: `best_smri_model_fold_{best_fold}.pth`
- **Advantage**: Uses the most promising model from cross-validation
- **Output**: `Using best fold: 3 (AUC: 0.8567)`

#### **Strategy 2: `last_fold`**
```bash
python train_smri.py --test_strategy last_fold
```
- **How it works**: Uses the last fold's model (original behavior)
- **Model used**: `best_smri_model.pth`
- **Advantage**: Consistent with previous behavior
- **Disadvantage**: May not be the best performing model

#### **Strategy 3: `ensemble` (Future)**
```bash
python train_smri.py --test_strategy ensemble
```
- **How it works**: Combines predictions from all fold models
- **Model used**: All `best_smri_model_fold_{fold_num}.pth` files
- **Advantage**: More robust predictions through ensemble voting
- **Status**: Not yet implemented

## 📊 **Example Output**

### **With `best_fold` strategy:**
```
Evaluating Simple3DCNN on the test set...
Using best fold: 3 (AUC: 0.8567)
Model file size: 45.23 MB
Test set evaluation for Simple3DCNN saved to: .../test_evaluation_plots
```

### **Fold Results Summary:**
```
Fold 1: AUC: 0.8234, Acc: 0.7891
Fold 2: AUC: 0.8456, Acc: 0.8123
Fold 3: AUC: 0.8567, Acc: 0.8234  ← Best fold (selected for test)
Fold 4: AUC: 0.8345, Acc: 0.8012
Fold 5: AUC: 0.8478, Acc: 0.8156
```

## 🚀 **Usage Examples**

### **Default (best fold):**
```bash
python train_smri.py \
    --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv \
    --data_root ~/reseng202500013-ndd-ml/data/preprocessed/MRI \
    --labels 0 1 \
    --run_all \
    --balance_dataset
```

### **Explicit best fold:**
```bash
python train_smri.py \
    --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv \
    --data_root ~/reseng202500013-ndd-ml/data/preprocessed/MRI \
    --labels 0 1 \
    --run_all \
    --balance_dataset \
    --test_strategy best_fold
```

### **Use last fold (original behavior):**
```bash
python train_smri.py \
    --master_csv ~/reseng202500013-ndd-ml/data/mri_labels.csv \
    --data_root ~/reseng202500013-ndd-ml/data/preprocessed/MRI \
    --labels 0 1 \
    --run_all \
    --balance_dataset \
    --test_strategy last_fold
```

## 📁 **File Structure**

After training, each model directory contains:
```
model_dir/
├── best_smri_model.pth                    # General model (last fold)
├── best_smri_model_fold_1.pth            # Fold 1 best model
├── best_smri_model_fold_2.pth            # Fold 2 best model
├── best_smri_model_fold_3.pth            # Fold 3 best model
├── best_smri_model_fold_4.pth            # Fold 4 best model
├── best_smri_model_fold_5.pth            # Fold 5 best model
├── test_metrics.json                     # Test evaluation results
├── test_evaluation_plots/                # Test evaluation plots
└── ...
```

## 🎯 **Recommendations**

1. **Use `best_fold` (default)**: This ensures you're using the most promising model from cross-validation
2. **Monitor fold performance**: Check the fold results to ensure consistent performance across folds
3. **Consider ensemble**: For production, consider implementing ensemble evaluation for more robust predictions

## 🔍 **Troubleshooting**

### **"Warning: Fold-specific model not found"**
- This happens if the training was interrupted or models weren't saved properly
- The system falls back to the general `best_smri_model.pth`
- Check that all folds completed successfully

### **"ERROR: Model file not found"**
- No model files were saved during training
- Check training logs for errors
- Ensure sufficient disk space for model saving 