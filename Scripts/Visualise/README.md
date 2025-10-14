# Visualization Directory

This directory contains visualization tools for medical imaging data, model results, and interpretability analysis. The visualization tools support interactive exploration, static plotting, and animation of medical images and analysis results.

## 📁 Directory Structure

```
Visualise/
├── README.md                     # This file
├── interactive_visualise.py      # Interactive image visualization
├── visualise_middle_slice.py     # Middle slice visualization
└── animate_gradcam_overlay.py    # Grad-CAM animation
```

## 🎨 Visualization Tools

### Interactive Visualization (`interactive_visualise.py`)

#### Purpose
Interactive 3D medical image visualization with slice navigation, overlay display, and analysis tools.

#### Features
- **3D slice navigation**: Navigate through 3D image slices
- **Overlay display**: Display activation maps and masks
- **Interactive controls**: Sliders, buttons, and radio buttons
- **Multi-modal display**: Display multiple image modalities
- **Analysis tools**: ROI analysis and measurements
- **Export functionality**: Export images and analysis results

#### Usage
```bash
python interactive_visualise.py \
    --input ~/path/to/image.nii.gz \
    --overlay ~/path/to/overlay.nii.gz \
    --output ~/path/to/output/
```

#### Interactive Controls
- **Slice sliders**: Navigate through axial, sagittal, and coronal slices
- **Intensity adjustment**: Adjust image contrast and brightness
- **Overlay controls**: Toggle overlay display and transparency
- **ROI selection**: Select regions of interest
- **Measurement tools**: Measure distances and areas

#### Supported Formats
- **NIfTI files**: .nii, .nii.gz
- **DICOM files**: .dcm
- **Numpy arrays**: .npy
- **Image files**: .png, .jpg, .tiff

### Middle Slice Visualization (`visualise_middle_slice.py`)

#### Purpose
Generate static visualizations of the middle slice of 3D medical images with optional overlays.

#### Features
- **Middle slice extraction**: Automatically find and display middle slice
- **Overlay support**: Display activation maps and masks
- **Multiple orientations**: Axial, sagittal, and coronal views
- **Customizable styling**: Colors, transparency, and annotations
- **Batch processing**: Process multiple images
- **Export options**: PNG, SVG, PDF formats

#### Usage
```bash
# Single image
python visualise_middle_slice.py \
    --input ~/path/to/image.nii.gz \
    --output ~/path/to/middle_slice.png

# With overlay
python visualise_middle_slice.py \
    --input ~/path/to/image.nii.gz \
    --overlay ~/path/to/activation.nii.gz \
    --output ~/path/to/middle_slice_with_overlay.png

# Multiple orientations
python visualise_middle_slice.py \
    --input ~/path/to/image.nii.gz \
    --orientations axial sagittal coronal \
    --output ~/path/to/multi_orientation.png
```

#### Parameters
- **`--input`**: Input NIfTI image file
- **`--overlay`**: Optional overlay NIfTI file
- **`--output`**: Output image file
- **`--orientations`**: Image orientations to display
- **`--colormap`**: Colormap for overlay
- **`--alpha`**: Overlay transparency (0-1)
- **`--title`**: Image title
- **`--dpi`**: Output image DPI

### Grad-CAM Animation (`animate_gradcam_overlay.py`)

#### Purpose
Create animated visualizations of Grad-CAM activation maps overlaid on medical images.

#### Features
- **Animation creation**: Generate animated overlays
- **Multiple formats**: MP4, GIF, and frame sequences
- **Customizable timing**: Control animation speed and duration
- **Quality options**: High-quality rendering options
- **Batch processing**: Process multiple image pairs
- **Interactive preview**: Preview animations before saving

#### Usage
```bash
# Create MP4 animation
python animate_gradcam_overlay.py \
    --input ~/path/to/image.nii.gz \
    --gradcam ~/path/to/gradcam.nii.gz \
    --output ~/path/to/animation.mp4 \
    --format mp4

# Create GIF animation
python animate_gradcam_overlay.py \
    --input ~/path/to/image.nii.gz \
    --gradcam ~/path/to/gradcam.nii.gz \
    --output ~/path/to/animation.gif \
    --format gif

# Custom animation settings
python animate_gradcam_overlay.py \
    --input ~/path/to/image.nii.gz \
    --gradcam ~/path/to/gradcam.nii.gz \
    --output ~/path/to/animation.mp4 \
    --format mp4 \
    --fps 10 \
    --duration 5 \
    --quality high
```

#### Parameters
- **`--input`**: Input NIfTI image file
- **`--gradcam`**: Grad-CAM activation map
- **`--output`**: Output animation file
- **`--format`**: Animation format (mp4, gif, frames)
- **`--fps`**: Frames per second
- **`--duration`**: Animation duration in seconds
- **`--quality`**: Rendering quality (low, medium, high)
- **`--colormap`**: Colormap for Grad-CAM
- **`--alpha`**: Overlay transparency

## 🎯 Visualization Types

### Medical Image Visualization
- **3D slice display**: Display 3D images as 2D slices
- **Multi-planar reconstruction**: Show multiple orientations
- **Intensity adjustment**: Adjust contrast and brightness
- **Colormap selection**: Choose appropriate colormaps
- **Annotation support**: Add text and shape annotations

### Activation Map Visualization
- **Grad-CAM display**: Show gradient-weighted activation maps
- **Saliency maps**: Display input sensitivity maps
- **Occlusion sensitivity**: Show occlusion-based sensitivity
- **Overlay display**: Overlay activation maps on original images
- **Transparency control**: Adjust overlay transparency

### Statistical Visualization
- **Performance plots**: ROC curves, confusion matrices
- **Feature importance**: Bar plots and heatmaps
- **Distribution plots**: Histograms and box plots
- **Correlation plots**: Correlation matrices and scatter plots
- **Time series**: Training curves and performance over time

## 🔧 Configuration

### Visualization Settings
```yaml
visualization:
  default_colormap: "viridis"
  overlay_alpha: 0.6
  figure_size: [12, 8]
  dpi: 300
  
  medical_images:
    slice_thickness: 1.0
    interpolation: "nearest"
    contrast_limits: "auto"
    
  activation_maps:
    colormap: "jet"
    alpha: 0.7
    threshold: 0.1
    
  animation:
    default_fps: 10
    default_duration: 5
    quality: "high"
```

### Interactive Controls
```yaml
interactive:
  slice_navigation: true
  intensity_adjustment: true
  overlay_controls: true
  roi_selection: true
  measurement_tools: true
  
  controls:
    slice_sliders: true
    intensity_sliders: true
    overlay_sliders: true
    button_controls: true
    keyboard_shortcuts: true
```

## 📊 Output Formats

### Static Images
- **PNG**: High-quality raster images
- **SVG**: Scalable vector graphics
- **PDF**: Print-ready documents
- **TIFF**: High-quality raster images
- **JPEG**: Compressed raster images

### Animated Content
- **MP4**: High-quality video format
- **GIF**: Animated GIF format
- **Frame sequences**: Individual frame images
- **HTML5**: Interactive web animations

### Interactive Content
- **HTML**: Interactive web visualizations
- **Jupyter notebooks**: Interactive notebook outputs
- **WebGL**: 3D web visualizations
- **Plotly**: Interactive plotly visualizations

## 🎨 Styling and Customization

### Color Schemes
- **Medical colormaps**: Optimized for medical imaging
- **Accessibility**: Colorblind-friendly palettes
- **Custom colormaps**: User-defined color schemes
- **Transparency**: Alpha channel support

### Layout Options
- **Grid layouts**: Multiple image arrangements
- **Single image**: Focused single image display
- **Side-by-side**: Comparison layouts
- **Overlay layouts**: Superimposed images

### Annotation Options
- **Text annotations**: Titles, labels, and descriptions
- **Shape annotations**: Arrows, circles, and rectangles
- **Scale bars**: Spatial reference scales
- **Color bars**: Intensity scale references

## 🚀 Performance Optimization

### Memory Management
- **Lazy loading**: Load images on demand
- **Memory mapping**: Use memory-mapped files
- **Caching**: Cache processed images
- **Cleanup**: Automatic memory cleanup

### Rendering Optimization
- **GPU acceleration**: Use GPU for rendering
- **Parallel processing**: Multi-threaded rendering
- **Quality settings**: Adjustable quality levels
- **Compression**: Optimize output file sizes

### Interactive Performance
- **Responsive controls**: Smooth interactive response
- **Efficient updates**: Minimize redraw operations
- **Caching**: Cache interactive states
- **Optimization**: Optimize for real-time interaction

## 🔍 Quality Control

### Image Quality
- **Resolution**: Maintain image resolution
- **Contrast**: Optimize contrast and brightness
- **Artifacts**: Detect and handle imaging artifacts
- **Noise**: Noise reduction and filtering

### Visualization Quality
- **Clarity**: Ensure clear and readable visualizations
- **Accuracy**: Maintain data accuracy
- **Consistency**: Consistent styling and formatting
- **Accessibility**: Ensure accessibility compliance

## 📚 Dependencies

### Core Libraries
- **matplotlib**: Plotting and visualization
- **nibabel**: Medical image I/O
- **numpy**: Numerical operations
- **scipy**: Scientific computing
- **pandas**: Data manipulation

### Interactive Libraries
- **ipywidgets**: Interactive widgets
- **plotly**: Interactive plotting
- **bokeh**: Interactive visualization
- **streamlit**: Web app framework

### Animation Libraries
- **matplotlib.animation**: Animation support
- **PIL**: Image processing
- **imageio**: Image I/O and animation
- **ffmpeg**: Video encoding

## 🚨 Common Issues

### Image Loading Issues
1. **File format**: Check supported file formats
2. **File corruption**: Validate file integrity
3. **Memory issues**: Check available memory
4. **Path issues**: Verify file paths

### Visualization Issues
1. **Display problems**: Check display settings
2. **Performance issues**: Optimize rendering settings
3. **Quality issues**: Adjust quality parameters
4. **Layout problems**: Check layout configuration

### Animation Issues
1. **Frame rate**: Adjust FPS settings
2. **File size**: Optimize output file size
3. **Quality**: Balance quality and performance
4. **Format compatibility**: Check format support

## 🔍 Debugging

### Visualization Debugging
- **Check data**: Validate input data
- **Test with sample data**: Use known good data
- **Review parameters**: Check visualization parameters
- **Monitor performance**: Track rendering performance

### Interactive Debugging
- **Test controls**: Verify interactive controls
- **Check responsiveness**: Monitor response times
- **Validate outputs**: Check output quality
- **Review user feedback**: Gather user feedback

## 📞 Support

For visualization issues:
- Check input data format and quality
- Validate visualization parameters
- Test with sample data first
- Review output quality and format
- Check system requirements and dependencies
- Monitor performance and memory usage
