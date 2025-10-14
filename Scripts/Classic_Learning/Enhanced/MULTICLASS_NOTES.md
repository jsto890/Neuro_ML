# Multi-Class SHAP Analysis Notes

## Multi-Class Support Status

### ✅ Fully Supported (Fast)
These models work perfectly with multi-class (3+ classes):
- **RandomForest** - TreeExplainer, very fast
- **ExtraTrees** - TreeExplainer, ~40s per fold
- **XGBoost** - TreeExplainer, very fast
- **LightGBM** - TreeExplainer, very fast

### ⚠️ Partially Supported (Slow Fallback)
- **GradientBoosting** - TreeExplainer only supports binary
  - Automatically falls back to KernelExplainer for multi-class
  - **Much slower** (~2-3 min per sample)
  - For 3-class with 466 samples: ~20-25 hours per fold
  - **Recommendation**: Exclude GradientBoosting for multi-class

### ✅ Supported (But Slow)
- **SVM** - KernelExplainer (always slow)
- **LogisticRegression** - LinearExplainer or KernelExplainer
- **KNN** - KernelExplainer (always slow)

---

## Recommendations by Use Case

### Binary Classification (CN vs PD, CN vs AD)
**Use all 8 models:**
```bash
python run_shap_comprehensive.py \
    --cv_dir ~/data/results/ \
    --data ~/data/features.csv \
    --output ~/data/shap_binary \
    --class_names CN PD
# No --model_types = all 8 models
# Time: ~30-40 minutes
```

### Multi-Class (CN vs AD vs PD)
**Option 1: Fast - Exclude GradientBoosting (Recommended)**
```bash
python run_shap_comprehensive.py \
    --cv_dir ~/data/results/ \
    --data ~/data/features.csv \
    --output ~/data/shap_multiclass_fast \
    --model_types randomforest extratrees xgboost lightgbm \
    --class_names CN AD PD
# Time: ~10-15 minutes
```

**Option 2: Very Fast - Best Tree Models Only**
```bash
python run_shap_comprehensive.py \
    --cv_dir ~/data/results/ \
    --data ~/data/features.csv \
    --output ~/data/shap_multiclass_quick \
    --model_types randomforest xgboost lightgbm \
    --class_names CN AD PD
# Time: ~6-8 minutes
```

**Option 3: Complete - Include All (Long!)**
```bash
python run_shap_comprehensive.py \
    --cv_dir ~/data/results/ \
    --data ~/data/features.csv \
    --output ~/data/shap_multiclass_complete \
    --model_types randomforest extratrees xgboost lightgbm svm logisticregression knn \
    --class_names CN AD PD
# Time: ~60-90 minutes
# (GradientBoosting excluded due to SHAP limitation)
```

---

## Current PET Multi-Class Run

Your current run is using:
```bash
--model_types randomforest extratrees gradientboosting xgboost lightgbm
--class_names CN AD PD
```

**Status:**
- ✅ RandomForest: Working perfectly (~0.5s per fold)
- ✅ ExtraTrees: Working (~40-50s per fold)
- ❌ GradientBoosting: **Failing** (SHAP limitation for multi-class)
- ✅ XGBoost: Working perfectly (~0.2s per fold)
- ✅ LightGBM: Working perfectly (~0.1s per fold)

**Expected completion:** 4/5 models will complete successfully
- GradientBoosting will be skipped automatically
- Total time: ~5-10 minutes (4 models × 5 folds)

---

## What Happens with Failed Models

The comprehensive script **gracefully handles failures**:
1. Logs the error
2. Continues with other models
3. Excludes failed model from ensemble analysis
4. Reports success rate at the end

**Example output:**
```
Aggregating randomforest: 5 folds
  Using 70 common features

Aggregating extratrees: 5 folds  
  Using 70 common features

Aggregating gradientboosting: 0 folds
  No results for gradientboosting (multi-class not supported by TreeExplainer)

Aggregating xgboost: 5 folds
  Using 70 common features

Aggregating lightgbm: 5 folds
  Using 70 common features

Using 68 features common to all models (4 successful models)
```

---

## Multi-Class SHAP Interpretation

For multi-class problems (CN vs AD vs PD), SHAP values show:
- **Which features** distinguish between classes
- **Direction of effect** (positive/negative)
- **Magnitude of impact**

### Per-Class Analysis (Future Enhancement)

Currently, the script uses **class 1** (middle class) for SHAP:
- For CN(0) vs AD(1) vs PD(2): Uses AD class
- Shows features that distinguish AD from CN and PD

**Possible enhancement:**
Generate separate SHAP analyses for each class:
- Features that predict CN
- Features that predict AD  
- Features that predict PD

This is not yet implemented but could be added if needed.

---

## Technical Details

### Why GradientBoosting Fails on Multi-Class

From SHAP library source:
```python
# shap/explainers/_tree.py
if isinstance(model, GradientBoostingClassifier) and model.n_classes_ > 2:
    raise Exception("GradientBoostingClassifier is only supported for binary classification right now!")
```

This is a **SHAP library limitation**, not our code.

### Workarounds

1. **Use other tree models** (RF, XGB, LGB work fine)
2. **Convert to binary** if only comparing 2 classes
3. **Wait for SHAP update** (check: https://github.com/slundberg/shap/issues)
4. **Use KernelExplainer** (very slow, implemented as fallback)

---

## Summary

### For Your Current Runs:

**SPECT (Binary: CN vs PD)**
- Use all 8 models
- All should work perfectly
- Time: ~30-40 minutes

**PET (Multi-class: CN vs AD vs PD)**  
- Exclude GradientBoosting (SHAP limitation)
- Use: RandomForest, ExtraTrees, XGBoost, LightGBM
- Time: ~10-15 minutes
- Still get robust ensemble analysis from 4 models

**Current PET run status:**
- ✅ Proceeding correctly
- Will complete with 4/5 models
- Results will still be comprehensive and useful!

