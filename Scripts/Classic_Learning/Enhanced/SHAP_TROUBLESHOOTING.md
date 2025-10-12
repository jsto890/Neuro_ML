# SHAP Troubleshooting Guide

## Issues Fixed

### Issue 1: "could not convert string to float: 'v3.0.1'"

**Problem:** Your radiomics CSV file contains non-numeric columns (like version strings 'v3.0.1') that were being passed to SHAP, causing conversion errors.

**Solution:** Updated `run_shap_analysis.py` to:
- Filter out non-numeric columns automatically
- Check column data types before loading
- Log warnings when non-numeric columns are skipped
- Convert data to float64 explicitly

**Fixed in:** `run_shap_analysis.py` (load_data function)

### Issue 2: Scaler.pkl being analyzed as a model

**Problem:** The script was trying to run SHAP on `scaler.pkl`, which is a preprocessing artifact, not a model.

**Solution:** Updated `find_model_files()` to exclude files matching patterns:
- scaler
- selector
- imputer  
- encoder
- preprocessor

**Fixed in:** `run_shap_analysis.py` (find_model_files function)

### Issue 3: Models in nested fold directories

**Problem:** Your models are in `outercv_fold_*/` subdirectories, not directly in the main directory.

**Solution:** You need to specify the fold directory directly:
```bash
--model_dir ~/path/to/run_20251010_171321/outercv_fold_5
```

Or use the new multi-fold script to analyze all folds at once (see below).

---

## Updated Usage

### Analyze Single Fold

```bash
cd Scripts/Classic_Learning/Enhanced/

python run_shap_analysis.py \
    --model_dir ~/reseng202500013-ndd-ml/data/classic_results/enhanced_run_SPECT/run_20251010_171321/outercv_fold_5 \
    --data ~/reseng202500013-ndd-ml/data/radiomics_spect.csv \
    --output ~/reseng202500013-ndd-ml/data/shap_results \
    --class_names CN PD \
    --all
```

This will analyze all models in fold 5 (RandomForest, SVM, XGBoost, LightGBM, etc.)

### Analyze Specific Model Across All Folds

```bash
python run_shap_multifold.py \
    --cv_dir ~/reseng202500013-ndd-ml/data/classic_results/enhanced_run_SPECT/run_20251010_171321 \
    --data ~/reseng202500013-ndd-ml/data/radiomics_spect.csv \
    --output ~/reseng202500013-ndd-ml/data/shap_multifold_randomforest \
    --model_type randomforest \
    --class_names CN PD
```

This will:
- Analyze RandomForest models in all 5 folds
- Compare feature importance across folds
- Generate stability metrics
- Create comparison plots

Repeat for other model types:
```bash
# SVM
python run_shap_multifold.py ... --model_type svm

# XGBoost
python run_shap_multifold.py ... --model_type xgboost

# LightGBM
python run_shap_multifold.py ... --model_type lightgbm

# Logistic Regression
python run_shap_multifold.py ... --model_type logisticregression

# Gradient Boosting
python run_shap_multifold.py ... --model_type gradientboosting
```

---

## Multi-Fold Analysis Outputs

The `run_shap_multifold.py` script generates:

### Individual Fold Results
- `fold_1/` - SHAP plots for fold 1
- `fold_2/` - SHAP plots for fold 2
- ... (one directory per fold)

### Cross-Fold Comparison
- **`<model>_feature_importance_across_folds.csv`** - Complete comparison table with:
  - Mean SHAP value across folds
  - Standard deviation
  - Coefficient of variation (stability metric)
  - Individual fold values

- **`<model>_top_features_with_variance.png`** - Bar plot of top features with error bars

- **`<model>_heatmap_across_folds.png`** - Heatmap showing how feature importance varies across folds

- **`<model>_feature_stability.png`** - Coefficient of variation plot
  - Green bars: Stable features (CV < 0.3)
  - Orange bars: Moderate stability (0.3 ≤ CV < 0.5)
  - Red bars: Variable features (CV ≥ 0.5)

- **`<model>_multifold_summary.json`** - JSON summary with:
  - Stability statistics
  - Top 10 most stable features
  - Top 10 most important features

---

## Interpreting Stability

### Coefficient of Variation (CV)

CV = Standard Deviation / Mean

- **CV < 0.3** = **Stable** ✅
  - Feature importance is consistent across folds
  - Reliable biomarker
  - Good for clinical interpretation

- **0.3 ≤ CV < 0.5** = **Moderate** ⚠️
  - Some variability across folds
  - Still useful but with caution

- **CV ≥ 0.5** = **Variable** ❌
  - High variability across folds
  - May be fold-dependent or overfitting
  - Use with caution for interpretation

### What This Tells You

1. **Stable + Important** = Best biomarkers
   - High mean SHAP value
   - Low coefficient of variation
   - Consistent across all folds

2. **Stable + Unimportant** = Consistently irrelevant
   - Low mean SHAP value
   - Low CV
   - Can safely ignore

3. **Variable + Important** = Investigate further
   - High mean SHAP value but high CV
   - May indicate:
     - Data heterogeneity
     - Model instability
     - Interaction effects
     - Fold-specific patterns

---

## About Ensemble Models

The Enhanced pipeline creates ensemble models through **voting**, but these are typically created dynamically during cross-validation and not always saved as separate `.pkl` files.

### Where to Find Ensemble Results

1. **Check `enhanced_results_summary.json`** in each fold directory
   - Contains performance metrics for the voting ensemble
   - Shows which models were included

2. **To Generate Ensemble SHAP:**
   - Analyze individual models (RF, SVM, XGBoost, etc.)
   - Compare their SHAP values
   - Features important across multiple models = robust ensemble features
   - Use the multi-fold comparison to see consistency

3. **Manual Ensemble Creation:**
   If you want to create an ensemble model for SHAP analysis:
   ```python
   from sklearn.ensemble import VotingClassifier
   import pickle
   
   # Load individual models
   rf = pickle.load(open('randomforest_model.pkl', 'rb'))
   svm = pickle.load(open('svm_model.pkl', 'rb'))
   xgb = pickle.load(open('xgboost_model.pkl', 'rb'))
   
   # Create ensemble
   ensemble = VotingClassifier(
       estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)],
       voting='soft'
   )
   
   # Note: This won't work directly with SHAP TreeExplainer
   # Use KernelExplainer instead (slower)
   ```

---

## Recommended Workflow

### Step 1: Single Fold Analysis (Quick Check)
```bash
# Analyze fold 5 (your best fold)
python run_shap_analysis.py \
    --model_dir .../outercv_fold_5 \
    --data .../radiomics_spect.csv \
    --output .../shap_fold5 \
    --class_names CN PD \
    --all
```

Review the plots to verify everything works.

### Step 2: Multi-Fold Analysis (Comprehensive)
```bash
# Analyze RandomForest across all folds
python run_shap_multifold.py \
    --cv_dir .../run_20251010_171321 \
    --data .../radiomics_spect.csv \
    --output .../shap_multifold_rf \
    --model_type randomforest \
    --class_names CN PD
```

Review stability metrics and identify consistent features.

### Step 3: Compare Model Types
```bash
# Repeat for other models
python run_shap_multifold.py ... --model_type svm
python run_shap_multifold.py ... --model_type xgboost
python run_shap_multifold.py ... --model_type lightgbm
```

### Step 4: Identify Consensus Features
Look for features that are:
- Important across multiple model types
- Stable across folds (low CV)
- These are your most reliable biomarkers!

---

## Data Format Requirements

Your CSV should have:
- **One label column**: `label`, `Label`, `diagnosis`, `Diagnosis`, `class`, or `Class`
- **Numeric feature columns only**: All other columns should be numeric
- **Optional ID columns**: `subject_id`, `Subject_ID`, etc. (will be automatically excluded)

### Bad CSV Example:
```
subject_id,pyradiomics_version,feature1,feature2,label
SUB001,v3.0.1,0.45,0.67,0
SUB002,v3.0.1,0.52,0.71,1
```
❌ The `pyradiomics_version` column will cause errors

### Good CSV Example:
```
subject_id,feature1,feature2,label
SUB001,0.45,0.67,0
SUB002,0.52,0.71,1
```
✅ Only numeric features (+ ID and label columns)

### Auto-Fix:
The updated script now automatically:
1. Detects and skips non-numeric columns
2. Logs warnings about skipped columns
3. Only uses numeric data for SHAP

---

## Common Errors and Solutions

### Error: "Cannot use mean strategy with non-numeric data"
**Cause:** Non-numeric columns in CSV
**Solution:** Fixed in updated script (auto-filters numeric columns)

### Error: "No model files found"
**Cause:** Looking in wrong directory
**Solution:** Point to fold directory: `.../outercv_fold_5`, not parent directory

### Error: "Model type X not found"
**Cause:** Model name doesn't match file name
**Solution:** Check actual file names in fold directory
- File: `randomforest_model.pkl` → Use: `--model_type randomforest`
- File: `svm_model.pkl` → Use: `--model_type svm`

### Warning: "InconsistentVersionWarning"
**Not an error!** Models were trained with sklearn 1.7.0, you have 1.7.1
- Safe to ignore for minor version differences
- Only problematic for major version differences

---

## Performance Notes

### Speed by Model Type:
- **Very Fast** (~seconds): RandomForest, XGBoost, LightGBM, GradientBoosting
- **Fast** (~seconds to minutes): Logistic Regression
- **Slow** (~minutes to hours): SVM, KNN

### Tips for Large Datasets:
1. Test with single fold first
2. Use `--test_size 0.1` for faster testing
3. Tree models are fastest - start with RandomForest
4. For SVM/KNN, consider using a smaller subset of data

---

## Questions?

See the main `README_SHAP.md` for:
- Detailed SHAP interpretation guide
- Programmatic usage examples
- Clinical interpretation guidelines
- Additional troubleshooting

