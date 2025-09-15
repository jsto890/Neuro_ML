#!/usr/bin/env python3
"""
Interactive SPECT Visualization Tool
Displays processed SPECT images with interactive controls
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import argparse

nii_file = "/Volumes/reseng202500013-ndd-ml/data/interpret/sub-I1624206_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore_gradcam_class0.nii.gz"

def find_spect_data():
    """Find available SPECT data in the Desktop SPECT directory"""
    base_dir = "/Users/jacksonschofield/Desktop/SPECT"
    
    available_data = []
    
    # Check for processed data
    for diagnosis in ['CN', 'PD']:
        for step in ['reoriented', 'normalised', 'registered', 'masked', 'finalised', 'postprocessed']:
            step_dir = os.path.join(base_dir, f"{diagnosis}_SPECT_PPMI_{step}")
            if os.path.exists(step_dir):
                subjects = [d for d in os.listdir(step_dir) if d.startswith('Subject_')]
                if subjects:
                    available_data.append({
                        'diagnosis': diagnosis,
                        'step': step,
                        'path': step_dir,
                        'subjects': subjects[:5]  # Limit to first 5 for display
                    })
    
    return available_data

def load_spect_image(file_path):
    """Load and validate SPECT image"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Basic validation
        if np.all(data == 0):
            print(f"⚠️ Warning: Image {file_path} contains only zeros")
            return None, None
        
        if np.any(np.isnan(data)):
            print(f"⚠️ Warning: Image {file_path} contains NaN values")
            data = np.nan_to_num(data, nan=0.0)
        
        if np.any(np.isinf(data)):
            print(f"⚠️ Warning: Image {file_path} contains infinite values")
            data = np.nan_to_num(data, posinf=0.0, neginf=0.0)
        
        return img, data
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None, None

def create_interactive_viewer(img, data, title="SPECT Image"):
    """Create interactive matplotlib viewer with slice controls"""
    
    # Get image dimensions
    nx, ny, nz = data.shape
    print(f"Image dimensions: {nx} x {ny} x {nz}")
    print(f"Data range: {np.min(data):.3f} to {np.max(data):.3f}")
    print(f"Non-zero voxels: {np.count_nonzero(data):,}")
    
    # Create figure with subplots for different views
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Interactive SPECT Viewer: {title}", fontsize=16)
    
    # Initial slice indices
    x_idx = nx // 2
    y_idx = ny // 2
    z_idx = nz // 2
    
    # Create initial plots
    vmin, vmax = np.percentile(data[data > 0], [5, 95]) if np.any(data > 0) else (0, 1)
    
    # Axial view (XY plane)
    im1 = ax1.imshow(data[:, :, z_idx].T, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Axial View (Z={z_idx})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Sagittal view (YZ plane)
    im2 = ax2.imshow(data[x_idx, :, :].T, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Sagittal View (X={x_idx})')
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    
    # Coronal view (XZ plane)
    im3 = ax3.imshow(data[:, y_idx, :].T, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    ax3.set_title(f'Coronal View (Y={y_idx})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    
    # Histogram
    non_zero_data = data[data > 0]
    if len(non_zero_data) > 0:
        ax4.hist(non_zero_data.flatten(), bins=50, alpha=0.7, color='red')
        ax4.set_title('Intensity Distribution (Non-zero voxels)')
        ax4.set_xlabel('Intensity')
        ax4.set_ylabel('Frequency')
        ax4.axvline(np.mean(non_zero_data), color='blue', linestyle='--', label=f'Mean: {np.mean(non_zero_data):.3f}')
        ax4.axvline(np.median(non_zero_data), color='green', linestyle='--', label=f'Median: {np.median(non_zero_data):.3f}')
        ax4.legend()
    
    # Add sliders for interactive control
    plt.subplots_adjust(bottom=0.2)
    
    # Z-slice slider (Axial view)
    ax_z = plt.axes([0.2, 0.1, 0.6, 0.03])
    z_slider = Slider(ax_z, 'Z Slice', 0, nz-1, valinit=z_idx, valstep=1)
    
    # X-slice slider (Sagittal view)
    ax_x = plt.axes([0.2, 0.05, 0.6, 0.03])
    x_slider = Slider(ax_x, 'X Slice', 0, nx-1, valinit=x_idx, valstep=1)
    
    # Y-slice slider (Coronal view)
    ax_y = plt.axes([0.2, 0.0, 0.6, 0.03])
    y_slider = Slider(ax_y, 'Y Slice', 0, ny-1, valinit=y_idx, valstep=1)
    
    def update_z(val):
        z_idx = int(val)
        im1.set_array(data[:, :, z_idx].T)
        ax1.set_title(f'Axial View (Z={z_idx})')
        fig.canvas.draw_idle()
    
    def update_x(val):
        x_idx = int(val)
        im2.set_array(data[x_idx, :, :].T)
        ax2.set_title(f'Sagittal View (X={x_idx})')
        fig.canvas.draw_idle()
    
    def update_y(val):
        y_idx = int(val)
        im3.set_array(data[:, y_idx, :].T)
        ax3.set_title(f'Coronal View (Y={y_idx})')
        fig.canvas.draw_idle()
    
    # Connect sliders to update functions
    z_slider.on_changed(update_z)
    x_slider.on_changed(update_x)
    y_slider.on_changed(update_y)
    
    # Add contrast adjustment buttons
    ax_contrast = plt.axes([0.02, 0.7, 0.1, 0.1])
    btn_auto = Button(ax_contrast, 'Auto\nContrast')
    
    def auto_contrast(event):
        new_vmin, new_vmax = np.percentile(data[data > 0], [5, 95]) if np.any(data > 0) else (0, 1)
        im1.set_clim(new_vmin, new_vmax)
        im2.set_clim(new_vmin, new_vmax)
        im3.set_clim(new_vmin, new_vmax)
        fig.canvas.draw_idle()
    
    btn_auto.on_clicked(auto_contrast)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Interactive SPECT Visualization Tool")
    parser.add_argument("--file", type=str, help="Path to specific NIfTI file to visualize")
    parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], help="Diagnosis group to browse")
    parser.add_argument("--step", type=str, choices=['reoriented', 'normalised', 'registered', 'masked', 'finalised', 'postprocessed'], 
                       help="Processing step to browse")
    args = parser.parse_args()
    
    if args.file:
        # Load specific file
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return
        
        img, data = load_spect_image(args.file)
        if img is not None:
            create_interactive_viewer(img, data, os.path.basename(args.file))
        return
    
    # Find available data
    print("🔍 Searching for available SPECT data...")
    available_data = find_spect_data()
    
    if not available_data:
        print("❌ No SPECT data found in /Users/jacksonschofield/Desktop/SPECT/")
        print("Please ensure you have run the preprocessing pipeline first.")
        return
    
    print(f"\n📊 Found {len(available_data)} data directories:")
    for i, data_info in enumerate(available_data):
        print(f"  {i+1}. {data_info['diagnosis']} - {data_info['step']} ({len(data_info['subjects'])} subjects)")
    
    # Filter by diagnosis and step if specified
    if args.diagnosis:
        available_data = [d for d in available_data if d['diagnosis'] == args.diagnosis]
    
    if args.step:
        available_data = [d for d in available_data if d['step'] == args.step]
    
    if not available_data:
        print(f"❌ No data found matching diagnosis={args.diagnosis}, step={args.step}")
        return
    
    # Select data to visualize
    selected_data = available_data[0]  # Use first available
    print(f"\n🎯 Using: {selected_data['diagnosis']} - {selected_data['step']}")
    
    # Find a subject with valid data
    valid_subject = None
    for subject in selected_data['subjects']:
        subject_dir = os.path.join(selected_data['path'], subject)
        
        # Look for the appropriate file based on step
        step_files = {
            'reoriented': '1. reorient.nii.gz',
            'normalised': '2. normalised.nii.gz',
            'registered': '3. registered.nii.gz',
            'masked': '4. masked.nii.gz',
            'finalised': '5. finalised.nii.gz',
            'postprocessed': '6. postprocessed.nii.gz'
        }
        
        expected_file = step_files.get(selected_data['step'])
        if expected_file:
            file_path = os.path.join(subject_dir, expected_file)
            if os.path.exists(file_path):
                img, data = load_spect_image(file_path)
                if img is not None and data is not None and np.any(data > 0):
                    valid_subject = subject
                    break
    
    if not valid_subject:
        print("❌ No valid subjects found with non-zero data")
        return
    
    print(f"✅ Found valid subject: {valid_subject}")
    
    # Load and display the image
    file_path = os.path.join(selected_data['path'], valid_subject, expected_file)
    img, data = load_spect_image(file_path)
    
    if img is not None and data is not None:
        title = f"{valid_subject} - {selected_data['step']}"
        create_interactive_viewer(img, data, title)
    else:
        print("❌ Failed to load image data")

if __name__ == "__main__":
    main()
