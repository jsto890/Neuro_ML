#!/usr/bin/env python3
"""
Test script to verify reorientation fix
Compares before and after reorientation to check for distortion
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def fix_path(path):
    return os.path.expanduser(path)

parser = argparse.ArgumentParser(description="Test reorientation fix by comparing before/after images")
parser.add_argument("--subject", type=str, default="sub-I10256370_PPMI_SPECT_CN", 
                    help="Subject ID to test")
args = parser.parse_args()

# Paths
raw_path = f"/Volumes/reseng202500013-ndd-ml/data/raw/SPECT/PPMI/CN/{args.subject}/{args.subject}.nii"
reoriented_path = f"/Volumes/reseng202500013-ndd-ml/data/preprocessed/SPECT/reoriented/CN/{args.subject}/{args.subject}_RAS.nii.gz"

print(f"Testing reorientation fix for {args.subject}")
print(f"Raw: {raw_path}")
print(f"Reoriented: {reoriented_path}")

# Load images
raw_img = nib.load(raw_path)
reoriented_img = nib.load(reoriented_path)

raw_data = raw_img.get_fdata()
reoriented_data = reoriented_img.get_fdata()

# Get middle slices
raw_middle = raw_data.shape[2] // 2
reoriented_middle = reoriented_data.shape[2] // 2

# Create comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Raw image - middle slice
ax1.imshow(raw_data[:, :, raw_middle].T, cmap='gray', origin='lower')
ax1.set_title(f"Raw - Middle Slice\nShape: {raw_data.shape}\nOrientation: {nib.orientations.aff2axcodes(raw_img.affine)}")
ax1.axis('off')

# Reoriented image - middle slice
ax2.imshow(reoriented_data[:, :, reoriented_middle].T, cmap='gray', origin='lower')
ax2.set_title(f"Reoriented - Middle Slice\nShape: {reoriented_data.shape}\nOrientation: {nib.orientations.aff2axcodes(reoriented_img.affine)}")
ax2.axis('off')

# Raw image - sagittal slice
raw_sagittal = raw_data.shape[0] // 2
ax3.imshow(raw_data[raw_sagittal, :, :].T, cmap='gray', origin='lower')
ax3.set_title(f"Raw - Sagittal Slice")
ax3.axis('off')

# Reoriented image - sagittal slice
reoriented_sagittal = reoriented_data.shape[0] // 2
ax4.imshow(reoriented_data[reoriented_sagittal, :, :].T, cmap='gray', origin='lower')
ax4.set_title(f"Reoriented - Sagittal Slice")
ax4.axis('off')

plt.tight_layout()
plt.savefig(f"reorientation_test_{args.subject}.png", dpi=150, bbox_inches='tight')
plt.show()

# Print statistics
print(f"\n=== Statistics ===")
print(f"Raw shape: {raw_data.shape}")
print(f"Reoriented shape: {reoriented_data.shape}")
print(f"Raw orientation: {nib.orientations.aff2axcodes(raw_img.affine)}")
print(f"Reoriented orientation: {nib.orientations.aff2axcodes(reoriented_img.affine)}")
print(f"Raw non-zero voxels: {np.count_nonzero(raw_data)}")
print(f"Reoriented non-zero voxels: {np.count_nonzero(reoriented_data)}")

# Check for distortion
raw_aspect = raw_data.shape[0] / raw_data.shape[1]
reoriented_aspect = reoriented_data.shape[0] / reoriented_data.shape[1]

print(f"\n=== Aspect Ratio Check ===")
print(f"Raw aspect ratio (width/height): {raw_aspect:.3f}")
print(f"Reoriented aspect ratio (width/height): {reoriented_aspect:.3f}")

if abs(raw_aspect - reoriented_aspect) < 0.1:
    print("✅ Aspect ratios are similar - no major distortion")
else:
    print("❌ Aspect ratios differ significantly - possible distortion") 