#!/usr/bin/env python3
"""
Interactive Visualization Tool
- Browse SPECT pipeline folders (original mode)
- OR overlay a heatmap on a base 3D volume with slice sliders
  Usage:
    python interactive_visualise.py --base /path/base.nii.gz --overlay /path/heat.nii.gz [--overlay-alpha 0.4]
"""

import os
import sys
import re
import json
import glob as glob_mod
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import argparse

nii_file = "/Volumes/reseng202500013-ndd-ml/data/interpret/sub-I1624206_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore_gradcam_class0.nii.gz"

def _robust_normalize(arr: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, lo_p)
    hi = np.percentile(arr, hi_p)
    if hi - lo < 1e-6:
        return arr
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

def _normalize_overlay_within_mask(overlay: np.ndarray, base: np.ndarray, lo_p: float = 90.0, hi_p: float = 99.5) -> np.ndarray:
    mask = (base != 0)
    vals = overlay[mask] if np.any(mask) else overlay
    lo = np.percentile(vals, lo_p)
    hi = np.percentile(vals, hi_p)
    if hi - lo < 1e-6:
        return _robust_normalize(overlay, 2.0, 98.0)
    out = (overlay - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

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


def create_overlay_viewer(base_img, base_data, overlay_data, title="Overlay Viewer", init_alpha: float = 0.4):
    """Interactive viewer with overlays and slice sliders for a 3D volume.
    - base_data: anatomical volume (grayscale)
    - overlay_data: heatmap volume, same shape as base_data
    """
    if base_data.shape != overlay_data.shape:
        print(f"⚠️ Base and overlay shapes differ: {base_data.shape} vs {overlay_data.shape}. Proceeding without resample.")

    base = _robust_normalize(base_data)
    heat = _normalize_overlay_within_mask(overlay_data, base_data)

    nx, ny, nz = base.shape
    x_idx, y_idx, z_idx = nx // 2, ny // 2, nz // 2

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # Initial plots
    im1 = ax1.imshow(base[:, :, z_idx].T, cmap='gray', origin='lower')
    hm1 = ax1.imshow(heat[:, :, z_idx].T, cmap='hot', origin='lower', alpha=init_alpha)
    ax1.set_title(f'Axial (Z={z_idx})')
    ax1.axis('off')

    im2 = ax2.imshow(base[x_idx, :, :].T, cmap='gray', origin='lower')
    hm2 = ax2.imshow(heat[x_idx, :, :].T, cmap='hot', origin='lower', alpha=init_alpha)
    ax2.set_title(f'Sagittal (X={x_idx})')
    ax2.axis('off')

    im3 = ax3.imshow(base[:, y_idx, :].T, cmap='gray', origin='lower')
    hm3 = ax3.imshow(heat[:, y_idx, :].T, cmap='hot', origin='lower', alpha=init_alpha)
    ax3.set_title(f'Coronal (Y={y_idx})')
    ax3.axis('off')

    plt.subplots_adjust(bottom=0.05)

    # Crosshair lines
    vline1 = ax1.axvline(x_idx, color='cyan', linewidth=1, alpha=0.8)
    hline1 = ax1.axhline(y_idx, color='cyan', linewidth=1, alpha=0.8)
    
    vline2 = ax2.axvline(y_idx, color='cyan', linewidth=1, alpha=0.8)
    hline2 = ax2.axhline(z_idx, color='cyan', linewidth=1, alpha=0.8)
    
    vline3 = ax3.axvline(x_idx, color='cyan', linewidth=1, alpha=0.8)
    hline3 = ax3.axhline(z_idx, color='cyan', linewidth=1, alpha=0.8)

    def update_views():
        # Update images
        im1.set_array(base[:, :, z_idx].T)
        hm1.set_array(heat[:, :, z_idx].T)
        ax1.set_title(f'Axial (Z={z_idx})')

        im2.set_array(base[x_idx, :, :].T)
        hm2.set_array(heat[x_idx, :, :].T)
        ax2.set_title(f'Sagittal (X={x_idx})')

        im3.set_array(base[:, y_idx, :].T)
        hm3.set_array(heat[:, y_idx, :].T)
        ax3.set_title(f'Coronal (Y={y_idx})')
        
        # Update crosshairs
        vline1.set_xdata([x_idx, x_idx])
        hline1.set_ydata([y_idx, y_idx])
        
        vline2.set_xdata([y_idx, y_idx])
        hline2.set_ydata([z_idx, z_idx])
        
        vline3.set_xdata([x_idx, x_idx])
        hline3.set_ydata([z_idx, z_idx])
        
        fig.canvas.draw_idle()

    def on_motion(event):
        nonlocal x_idx, y_idx, z_idx
        if event.inaxes == ax1:  # Axial view - Y position sets Y, X position sets X
            if event.xdata is not None and event.ydata is not None:
                x_idx = max(0, min(nx-1, int(event.xdata)))
                y_idx = max(0, min(ny-1, int(event.ydata)))
                update_views()
        elif event.inaxes == ax2:  # Sagittal view - Y position sets Z, X position sets Y
            if event.xdata is not None and event.ydata is not None:
                y_idx = max(0, min(ny-1, int(event.xdata)))
                z_idx = max(0, min(nz-1, int(event.ydata)))
                update_views()
        elif event.inaxes == ax3:  # Coronal view - Y position sets Z, X position sets X
            if event.xdata is not None and event.ydata is not None:
                x_idx = max(0, min(nx-1, int(event.xdata)))
                z_idx = max(0, min(nz-1, int(event.ydata)))
                update_views()

    def on_key(event):
        nonlocal x_idx, y_idx, z_idx
        if event.key == 'up':
            z_idx = min(nz-1, z_idx + 1)
            update_views()
        elif event.key == 'down':
            z_idx = max(0, z_idx - 1)
            update_views()
        elif event.key == 'left':
            x_idx = max(0, x_idx - 1)
            update_views()
        elif event.key == 'right':
            x_idx = min(nx-1, x_idx + 1)
            update_views()
        elif event.key == 'pageup':
            y_idx = min(ny-1, y_idx + 1)
            update_views()
        elif event.key == 'pagedown':
            y_idx = max(0, y_idx - 1)
            update_views()

    # Connect mouse motion and keyboard events
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Interactive Visualization Tool")
    parser.add_argument("--file", type=str, help="Path to specific NIfTI file to visualize")
    parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], help="Diagnosis group to browse (SPECT mode)")
    parser.add_argument("--step", type=str, choices=['reoriented', 'normalised', 'registered', 'masked', 'finalised', 'postprocessed'], 
                       help="Processing step to browse (SPECT mode)")
    # Overlay mode
    parser.add_argument("--base", type=str, help="Path to base anatomical NIfTI for overlay mode")
    parser.add_argument("--overlay", type=str, help="Path to overlay heatmap NIfTI for overlay mode")
    parser.add_argument("--overlay-alpha", type=float, default=0.4, help="Initial overlay alpha (0-1)")
    args = parser.parse_args()
    
    # Overlay mode takes precedence if both paths provided
    if args.base and args.overlay:
        if not os.path.exists(args.base):
            print(f"❌ Base file not found: {args.base}")
            return
        if not os.path.exists(args.overlay):
            print(f"❌ Overlay file not found: {args.overlay}")
            return
        try:
            b_img = nib.load(args.base); b_data = b_img.get_fdata().astype(np.float32)
            o_img = nib.load(args.overlay); o_data = o_img.get_fdata().astype(np.float32)
        except Exception as e:
            print(f"❌ Failed to load NIfTI: {e}")
            return
        if b_data.ndim != 3:
            print("❌ Base volume must be 3D")
            return
        if o_data.ndim == 4:
            o_data = o_data.mean(axis=-1)
        if o_data.ndim != 3:
            print("❌ Overlay must be 3D or 4D")
            return
        
        # Extract subject ID and predicted disease from JSON if available
        sub_id = "Unknown"
        predicted_disease = "Unknown"
        
        # Try to find corresponding JSON file
        base_dir = os.path.dirname(args.base)
        base_name = os.path.basename(args.base)
        json_pattern = base_name.replace('.nii.gz', '').replace('.nii', '') + '*clinical_prediction_deep.json'
        json_files = glob_mod.glob(os.path.join(base_dir, json_pattern))
        
        if json_files:
            try:
                import json
                with open(json_files[0], 'r') as f:
                    json_data = json.load(f)
                # Extract subject ID from image path
                if 'image' in json_data:
                    img_path = json_data['image']
                    sub_match = re.search(r'sub-([A-Za-z0-9]+)', img_path)
                    if sub_match:
                        sub_id = sub_match.group(1)
                # Extract predicted disease
                if 'prediction' in json_data and 'label_name' in json_data['prediction']:
                    predicted_disease = json_data['prediction']['label_name']
            except Exception as e:
                print(f"⚠️ Could not parse JSON: {e}")
        
        title = f"{sub_id} • {predicted_disease}"
        create_overlay_viewer(b_img, b_data, o_data, title=title, init_alpha=float(args.overlay_alpha))
        return

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
