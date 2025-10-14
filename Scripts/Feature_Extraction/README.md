# Feature Extraction Directory

This directory contains tools for extracting radiomics features from medical images. Radiomics analysis involves the extraction of quantitative features from medical images to characterize tumor or tissue properties for machine learning applications.

## 📁 Directory Structure

```
Feature_Extraction/
├── README.md                     # This file
└── pyRadioMics/                  # Radiomics analysis tools
    ├── radiomics_extractor.py    # Comprehensive radiomics extraction
    └── simple_radiomics.py       # Basic radiomics features
```

## 🔬 Radiomics Analysis

### Comprehensive Radiomics Extractor (`radiomics_extractor.py`)

#### Purpose
Extract comprehensive radiomics features from medical images using the pyRadiomics library.

#### Features
- **Complete feature set**: Extract all available radiomics features
- **Multiple image types**: Support for various medical image modalities
- **Batch processing**: Process multiple images efficiently
- **Quality control**: Validate feature extraction quality
- **Export options**: Multiple output formats (CSV, JSON, HDF5)
- **Configuration**: Flexible parameter configuration

#### Usage
```bash
cd pyRadioMics

# Basic extraction
python radiomics_extractor.py \
    --input ~/path/to/images/ \
    --mask ~/path/to/masks/ \
    --output ~/path/to/features.csv

# Advanced extraction with configuration
python radiomics_extractor.py \
    --input ~/path/to/images/ \
    --mask ~/path/to/masks/ \
    --output ~/path/to/features.csv \
    --config ~/path/to/radiomics_config.yaml \
    --format csv \
    --verbose
```

#### Supported Feature Classes
1. **First-Order Statistics**
   - Mean, standard deviation, variance
   - Skewness, kurtosis
   - Energy, entropy
   - Percentiles, range

2. **Shape Features**
   - Volume, surface area
   - Sphericity, compactness
   - Surface-to-volume ratio
   - Maximum 3D diameter

3. **Gray Level Co-occurrence Matrix (GLCM)**
   - Contrast, correlation
   - Energy, homogeneity
   - Entropy, dissimilarity
   - Autocorrelation

4. **Gray Level Run Length Matrix (GLRLM)**
   - Short run emphasis
   - Long run emphasis
   - Gray level non-uniformity
   - Run length non-uniformity

5. **Gray Level Size Zone Matrix (GLSZM)**
   - Small zone emphasis
   - Large zone emphasis
   - Gray level non-uniformity
   - Zone size non-uniformity

6. **Gray Level Distance Zone Matrix (GLDZM)**
   - Small distance emphasis
   - Large distance emphasis
   - Gray level non-uniformity
   - Distance non-uniformity

7. **Neighboring Gray Tone Difference Matrix (NGTDM)**
   - Coarseness
   - Contrast
   - Busyness
   - Complexity

8. **Gray Level Dependence Matrix (GLDM)**
   - Small dependence emphasis
   - Large dependence emphasis
   - Gray level non-uniformity
   - Dependence non-uniformity

#### Configuration File (`radiomics_config.yaml`)
```yaml
imageType:
  Original: {}
  Wavelet:
    - LLH
    - LHL
    - LHH
    - HLL
    - HLH
    - HHL
    - HHH
  LoG:
    - sigma: [1.0, 2.0, 3.0]
  Square:
    - {}
  SquareRoot:
    - {}
  Logarithm:
    - {}
  Exponential:
    - {}

featureClass:
  shape:
    - {}
  firstorder:
    - {}
  glcm:
    - {}
  glrlm:
    - {}
  glszm:
    - {}
  gldzm:
    - {}
  ngtdm:
    - {}
  gldm:
    - {}

settings:
  binWidth: 25
  normalize: true
  normalizeScale: 100
  removeOutliers: 3
  resampledPixelSpacing: [1, 1, 1]
  interpolator: sitkBSpline
  label: 1
```

### Simple Radiomics (`simple_radiomics.py`)

#### Purpose
Extract basic radiomics features for quick analysis and testing.

#### Features
- **Essential features**: Core radiomics features
- **Fast extraction**: Optimized for speed
- **Simple interface**: Easy-to-use interface
- **Basic validation**: Essential quality control

#### Usage
```bash
python simple_radiomics.py \
    --input ~/path/to/image.nii.gz \
    --mask ~/path/to/mask.nii.gz \
    --output ~/path/to/simple_features.csv
```

#### Extracted Features
- **Basic statistics**: Mean, std, min, max
- **Shape features**: Volume, surface area
- **Texture features**: GLCM, GLRLM basics
- **Intensity features**: Histogram features

## 📊 Feature Extraction Process

### Input Requirements
- **Image files**: NIfTI format (.nii, .nii.gz)
- **Mask files**: Binary masks defining ROI
- **Configuration**: Feature extraction parameters
- **Labels**: Optional class labels for supervised learning

### Processing Steps
1. **Image loading**: Load NIfTI images and masks
2. **Preprocessing**: Apply image preprocessing filters
3. **Feature extraction**: Extract radiomics features
4. **Quality control**: Validate extracted features
5. **Export**: Save features in specified format

### Quality Control
- **Feature validation**: Check for NaN and infinite values
- **Statistical validation**: Validate feature distributions
- **Correlation analysis**: Identify highly correlated features
- **Outlier detection**: Detect and handle outliers

## 🔧 Configuration

### Feature Selection
```yaml
feature_selection:
  firstorder: true
  shape: true
  glcm: true
  glrlm: true
  glszm: true
  gldzm: true
  ngtdm: true
  gldm: true
  
  wavelet_filters:
    - LLH
    - LHL
    - LHH
    - HLL
    - HLH
    - HHL
    - HHH
```

### Image Preprocessing
```yaml
preprocessing:
  binWidth: 25
  normalize: true
  normalizeScale: 100
  removeOutliers: 3
  resampledPixelSpacing: [1, 1, 1]
  interpolator: sitkBSpline
  label: 1
```

### Output Settings
```yaml
output:
  format: csv  # csv, json, hdf5
  include_metadata: true
  include_image_info: true
  compression: true
  verbose: true
```

## 📈 Feature Analysis

### Feature Statistics
- **Descriptive statistics**: Mean, std, min, max, percentiles
- **Distribution analysis**: Histograms, Q-Q plots
- **Correlation analysis**: Feature correlation matrices
- **Outlier analysis**: Outlier detection and analysis

### Feature Selection
- **Variance threshold**: Remove low-variance features
- **Correlation filtering**: Remove highly correlated features
- **Statistical testing**: Univariate feature selection
- **Mutual information**: Information-theoretic selection

### Feature Engineering
- **Normalization**: Z-score, min-max, robust scaling
- **Transformation**: Log, square root, Box-Cox
- **Polynomial features**: Feature interactions
- **Dimensionality reduction**: PCA, ICA, feature selection

## 📊 Output Formats

### CSV Format
```csv
subject_id,feature_1,feature_2,feature_3,...
sub-001,0.123,0.456,0.789,...
sub-002,0.234,0.567,0.890,...
```

### JSON Format
```json
{
  "features": {
    "sub-001": {
      "firstorder": {
        "Mean": 0.123,
        "StdDev": 0.456
      },
      "shape": {
        "Volume": 1000,
        "SurfaceArea": 500
      }
    }
  },
  "metadata": {
    "extraction_date": "2024-01-01",
    "pyradiomics_version": "3.1.0"
  }
}
```

### HDF5 Format
- **Hierarchical structure**: Organized feature storage
- **Compression**: Efficient storage
- **Metadata**: Rich metadata support
- **Accessibility**: Easy programmatic access

## 🚨 Common Issues

### Image Loading Issues
1. **File format**: Ensure NIfTI format compatibility
2. **File corruption**: Validate file integrity
3. **Memory issues**: Check available memory
4. **Path issues**: Verify file paths

### Feature Extraction Issues
1. **Mask quality**: Ensure proper mask quality
2. **Image quality**: Check image preprocessing
3. **Parameter settings**: Verify configuration parameters
4. **Memory constraints**: Monitor memory usage

### Quality Control Issues
1. **NaN values**: Handle missing or invalid features
2. **Infinite values**: Check for infinite values
3. **Feature correlation**: Identify highly correlated features
4. **Outlier detection**: Handle extreme values

## 🔍 Debugging

### Feature Extraction Debugging
- **Check inputs**: Validate input images and masks
- **Review configuration**: Check parameter settings
- **Monitor memory**: Track memory usage
- **Test with sample data**: Use known good data

### Quality Control Debugging
- **Validate features**: Check feature distributions
- **Review statistics**: Analyze feature statistics
- **Check correlations**: Review feature correlations
- **Handle outliers**: Identify and handle outliers

## 📚 Dependencies

### Core Libraries
- **pyRadiomics**: Radiomics feature extraction
- **SimpleITK**: Medical image processing
- **nibabel**: NIfTI file I/O
- **numpy**: Numerical operations
- **pandas**: Data manipulation

### Optional Libraries
- **h5py**: HDF5 file support
- **scikit-learn**: Feature selection and analysis
- **matplotlib**: Visualization
- **seaborn**: Statistical visualization

## 🚀 Performance Optimization

### Memory Management
- **Batch processing**: Process images in batches
- **Memory mapping**: Use memory-mapped files
- **Garbage collection**: Clean up unused objects
- **Memory monitoring**: Track memory usage

### Computational Optimization
- **Parallel processing**: Use multiple CPU cores
- **Vectorization**: Use vectorized operations
- **Caching**: Cache intermediate results
- **Optimization**: Optimize feature extraction

### Storage Optimization
- **Compression**: Use compressed output formats
- **Efficient formats**: Choose appropriate file formats
- **Metadata optimization**: Optimize metadata storage
- **Cleanup**: Remove temporary files

## 📞 Support

For feature extraction issues:
- Check input image and mask quality
- Validate configuration parameters
- Review pyRadiomics documentation
- Test with sample data first
- Check system requirements and dependencies
- Monitor memory usage and performance
