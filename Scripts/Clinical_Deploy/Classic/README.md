# Clinical Deployment - Classical Learning Models

This directory contains clinical deployment scripts for the optimized classical learning models for early detection of neurodegenerative diseases.

## 🎯 **Chosen Models for Clinical Use**

### **Primary Model: Optimized SVM**
- **File**: `~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_svm_model.pkl`
- **Performance**: 78.0% test accuracy, 0.839 AUC
- **Use Case**: Primary clinical predictions
- **Advantages**: Best performance, good interpretability, robust

### **Backup Model: Stacking Ensemble**
- **File**: `~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_ensemble_model.pkl`
- **Performance**: 70.7% test accuracy, 0.790 AUC
- **Use Case**: Critical decisions, validation, backup predictions
- **Advantages**: Ensemble diversity, robust, reduces variance

### **Required Components**
- **Scaler**: `~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_scaler.pkl`
- **Feature Importance**: `~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_feature_importance.csv`

## 🏥 **Clinical Scripts**

### **1. Clinical Predictor** (`predict_clinical.py`) ⭐ **MAIN SCRIPT**

#### **Interactive Mode**
```bash
python predict_clinical.py
```
- Enter patient features manually
- Get immediate clinical interpretation
- Real-time confidence scores

#### **Batch Mode**
```bash
python predict_clinical.py patient_data.csv results.csv
```
- Process multiple patients from CSV
- Generate comprehensive results file
- Summary statistics

#### **Features**
- **Clinical Interpretation**: Automatic diagnosis and recommendations
- **Confidence Scoring**: High/Medium/Low confidence levels
- **Urgency Assessment**: Based on prediction confidence
- **Feature Analysis**: Top contributing features for interpretation

### **2. Model Validator** (`validate_model.py`)

#### **Usage**
```bash
python validate_model.py
```

#### **Features**
- **Performance Validation**: Test accuracy, AUC, classification report
- **Confidence Analysis**: High-confidence prediction rates
- **Clinical Guidelines**: Usage recommendations based on confidence
- **Validation Report**: Comprehensive markdown report

## 📊 **Clinical Performance**

### **Confidence Guidelines**
- **High Confidence (≥80%)**: Use for clinical decision making
- **Medium Confidence (60-80%)**: Use with caution, consider additional context
- **Low Confidence (<60%)**: Do not use for clinical decisions, recommend additional testing

## 🚀 **Quick Start**

### **1. Validate Your Model**
```bash
cd Scripts/Clinical_Deploy/Classic
python validate_model.py
```

### **2. Test Clinical Predictions**
```bash
# Test on your data
python predict_clinical.py ~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv test_results.csv
```

### **3. Use for New Patients**
```bash
# Interactive mode for single patients
python predict_clinical.py
```

## 📋 **Input Requirements**

### **Feature Format**
- **Number of Features**: 30 radiomics features
- **Format**: Comma-separated values
- **Order**: Must match training data order
- **Missing Values**: Not supported (use median imputation)

### **Expected Features**
The model expects these 30 features in order:
1. `original_firstorder_Kurtosis`
2. `original_firstorder_MeanAbsoluteDeviation`
3. `original_firstorder_RobustMeanAbsoluteDeviation`
4. `original_gldm_DependenceNonUniformity`
5. `original_gldm_DependenceVariance`
6. `original_gldm_LargeDependenceEmphasis`
7. `original_glrlm_GrayLevelNonUniformity`
8. `original_glrlm_LongRunEmphasis`
9. `original_glrlm_LongRunHighGrayLevelEmphasis`
10. `original_glrlm_LongRunLowGrayLevelEmphasis`
11. `original_glrlm_RunLengthNonUniformity`
12. `original_glrlm_RunVariance`
13. `original_glszm_HighGrayLevelZoneEmphasis`
14. `original_glszm_LowGrayLevelZoneEmphasis`
15. `original_glszm_SmallAreaLowGrayLevelEmphasis`
16. `original_ngtdm_Busyness`
17. `poly2_11` (Polynomial feature)
18. `poly2_14` (Polynomial feature)
19. `poly2_4` (Polynomial feature)
20. `poly2_6` (Polynomial feature)
21. `poly2_19` (Polynomial feature)
22. `poly2_5` (Polynomial feature)
23. `poly2_21` (Polynomial feature)
24. `poly2_0` (Polynomial feature)
25. `poly2_20` (Polynomial feature)
26. `poly2_8` (Polynomial feature)
27. `poly2_3` (Polynomial feature)
28. `poly2_2` (Polynomial feature)
29. `texture_mean`
30. `texture_std`

## 📤 **Output Format**

### **Single Patient Prediction**
```json
{
  "patient_id": "Patient_001",
  "prediction": 1,
  "probability": 0.823,
  "confidence_level": "High",
  "diagnosis": "POSITIVE - Signs of neurodegenerative disease detected",
  "recommendation": "Recommend further clinical evaluation and specialist consultation",
  "urgency": "High",
  "probabilities": {
    "negative": 0.177,
    "positive": 0.823
  }
}
```

### **Batch Prediction Results**
CSV file with columns:
- `patient_id`: Patient identifier
- `prediction`: 0 (negative) or 1 (positive)
- `probability`: Confidence score (0-1)
- `confidence_level`: High/Medium/Low
- `diagnosis`: Clinical interpretation
- `recommendation`: Clinical recommendation
- `urgency`: High/Medium/Low

## 🏥 **Clinical Workflow**

### **1. Pre-processing**
- Extract radiomics features from MRI scans
- Ensure 30 features in correct order
- Handle missing values (median imputation)

### **2. Prediction**
```python
# Load model and scaler
with open('optimized_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('optimized_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make prediction
scaled_features = scaler.transform(patient_features)
prediction = model.predict(scaled_features)
probability = model.predict_proba(scaled_features)
```

### **3. Clinical Interpretation**
- **High Confidence (≥80%)**: Use for clinical decisions
- **Medium Confidence (60-80%)**: Use with caution
- **Low Confidence (<60%)**: Recommend additional testing

### **4. Follow-up**
- Use ensemble model for critical decisions
- Review feature importance for biomarker analysis
- Monitor performance over time

## ⚠️ **Clinical Limitations**

### **Model Limitations**
- **Binary Classification**: Only distinguishes between 0 (healthy) and 1 (disease)
- **Feature Dependence**: Requires exact same 30 features as training
- **Population Specific**: Performance may vary on different populations
- **No Disease Staging**: Cannot determine disease severity or progression

### **Clinical Considerations**
- **Not a Replacement**: Should not replace clinical judgment
- **Additional Testing**: Always consider other clinical factors
- **Population Differences**: May not generalize to all populations
- **Temporal Changes**: Model performance may change over time

## 🔧 **Technical Requirements**

### **Dependencies**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy
```

### **File Structure**
```
Scripts/Clinical_Deploy/Classic/
├── predict_clinical.py    # Main clinical prediction script
├── validate_model.py      # Model validation script
└── README.md             # This file

~/reseng202500013-ndd-ml/data/optimized_classical_results/
├── optimized_svm_model.pkl          # Primary model
├── optimized_ensemble_model.pkl     # Backup model
├── optimized_scaler.pkl             # Feature scaler
└── optimized_feature_importance.csv # Feature importance
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **File Not Found**
   ```bash
   # Check if model files exist
   ls ~/reseng202500013-ndd-ml/data/optimized_classical_results/
   ```

2. **Feature Mismatch**
   - Ensure exactly 30 features
   - Check feature order matches training data
   - Verify no missing values

3. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x predict_clinical.py validate_model.py
   ```

4. **Memory Issues**
   - Close other applications
   - Process patients in smaller batches

## 📞 **Support**

### **For Technical Issues**
- Check the troubleshooting section above
- Review the model validation results
- Ensure all dependencies are installed

### **For Clinical Questions**
- Consult with clinical experts
- Review feature importance for biological interpretation
- Consider additional clinical validation

## 📄 **Documentation**

### **Related Files**
- `~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_results_summary.json` - Detailed results
- `~/reseng202500013-ndd-ml/data/optimized_classical_results/feature_engineering_results.json` - Feature engineering details
- `~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_evaluation_plots.png` - Performance visualization

### **Research Context**
- Model trained on MRI radiomics features
- Binary classification for early disease detection
- Optimized using Bayesian hyperparameter tuning
- Ensemble methods for robustness

---

**⚠️ Clinical Disclaimer**: This model is for research and clinical decision support only. It should not replace clinical judgment or professional medical advice. Always consider additional clinical factors and consult with healthcare professionals for patient care decisions. 