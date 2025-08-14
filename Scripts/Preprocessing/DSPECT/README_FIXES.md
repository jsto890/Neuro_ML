# DSPECT Preprocessing Pipeline - Fixes Applied

## Issues Fixed

### 1. **Step 2: Normalization** 
**Problem**: Using global z-score normalization on all voxels, which is inappropriate for SPECT data.

**Fix**: Implemented proper SPECT-specific normalization methods:
- **Reference Region Normalization**: Uses occipital mask as reference region (standard for DAT SPECT)
- **Percentile Clipping**: Alternative method for intensity normalization
- Added proper error handling for edge cases

**Usage**:
```bash
# Reference region normalization (recommended)
python 2_normalise.py --diagnosis CN --method reference

# Percentile clipping
python 2_normalise.py --diagnosis CN --method percentile
```

### 2. **Step 4: Masking**
**Problem**: Using MRI brain mask instead of SPECT-specific mask.

**Fix**: Changed to use SPECT-specific occipital mask:
- More appropriate for DAT SPECT imaging
- Better preserves SPECT-specific brain regions
- Improved validation of masking results

### 3. **Enhanced Testing**
**Problem**: Tests didn't properly validate SPECT-specific preprocessing.

**Fixes**:
- Enhanced `2_test.py` to validate reference region normalization
- Enhanced `4_test_visulise.py` with better visualization and statistics
- Added comprehensive pipeline validation script

### 4. **New Pipeline Runner**
**Added**: `run_pipeline.py` - Automated pipeline runner with proper error handling and validation.

## How to Use the Fixed Pipeline

### Option 1: Run Individual Steps
```bash
# Step 1: Reorientation
python 1_reorient.py --diagnosis CN --force

# Step 2: SPECT-specific normalization
python 2_normalise.py --diagnosis CN --method reference

# Step 3: Registration
python 3_register.py --diagnosis CN

# Step 4: SPECT-specific masking
python 4_masking.py --diagnosis CN

# Step 5: Finalization
python 5_padding.py --diagnosis CN --shape 91 109 91

# Step 6: Postprocessing
python 6_postprocess.py --diagnosis CN --isHasel
```

### Option 2: Run Complete Pipeline
```bash
# Run entire pipeline with validation
python run_pipeline.py --diagnosis CN --isHasel

# With custom parameters
python run_pipeline.py --diagnosis PD --shape 91 109 91 --intensity_norm
```

### Option 3: Test Individual Steps
```bash
# Test normalization
python testing/2_test.py --isHasel

# Test masking visualization
python testing/4_test_visulise.py --isHasel

# Validate entire pipeline
python testing/validate_pipeline.py --diagnosis CN --isHasel
```

## Key Improvements

### 1. **SPECT-Specific Normalization**
- Reference region normalization using occipital mask
- Proper handling of edge cases (empty reference regions)
- Fallback to global stats if reference region fails

### 2. **Better Error Handling**
- Graceful handling of missing files
- Proper validation of preprocessing results
- Clear error messages and warnings

### 3. **Enhanced Validation**
- Comprehensive pipeline validation
- SPECT-specific quality checks
- Statistical validation of normalization results

### 4. **Improved Documentation**
- Clear usage instructions
- Better error messages
- Validation feedback

## Quality Assurance

The fixed pipeline now includes:

1. **Reference Region Validation**: Checks that normalization produces reasonable values
2. **Coverage Validation**: Ensures masking produces appropriate brain coverage
3. **Statistical Validation**: Validates intensity distributions
4. **Pipeline Validation**: Comprehensive end-to-end testing

## Expected Results

After running the fixed pipeline, you should see:

- **Normalization**: Mean values around 1.0 (reference region) or 0.5 (percentile)
- **Masking**: Brain coverage between 5-50% (typical for SPECT)
- **Final Data**: Consistent shapes and intensity ranges across subjects
- **Validation**: All steps passing quality checks

## Troubleshooting

### Common Issues:

1. **"No voxels in reference region"**: Check that occipital mask is properly aligned
2. **"Low brain coverage"**: Verify registration quality in step 3
3. **"Suspicious normalization values"**: Check input data quality

### Validation Failures:

Run the validation script to identify specific issues:
```bash
python testing/validate_pipeline.py --diagnosis CN --isHasel
```

This will provide detailed feedback on which steps need attention.

## Machine Learning Readiness

The fixed pipeline produces data that is:
- ✅ Properly normalized for SPECT imaging
- ✅ Consistently masked and shaped
- ✅ Validated for quality
- ✅ Ready for machine learning models

Your DSPECT data will now follow best practices for neurodegenerative disease detection using DAT SPECT imaging. 