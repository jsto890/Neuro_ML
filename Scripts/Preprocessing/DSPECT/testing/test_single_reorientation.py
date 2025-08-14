#!/usr/bin/env python3
"""
Test reorientation fix on a single image
"""

import nibabel as nib
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

def reorient_to_RAS(nifti_path, output_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine

    print("\n--- Reorientation Debug ---")
    print(f"Processing file: {nifti_path}")
    print("Original affine:")
    print(affine)
    print("Original shape:", data.shape)
    orig_ornt_codes = nib.orientations.aff2axcodes(affine)
    print("Original orientation codes:", orig_ornt_codes)
    target_ornt_codes = ('R', 'A', 'S')
    print("Target orientation codes:", target_ornt_codes)

    if orig_ornt_codes != target_ornt_codes:
        print("Reorienting to RAS...")
        
        # Get current and target orientations
        current_ornt = io_orientation(affine)
        target_ornt = axcodes2ornt(target_ornt_codes)
        
        # Calculate the transformation
        transform = ornt_transform(current_ornt, target_ornt)
        
        # Use nibabel's built-in reorientation which handles affine correctly
        reoriented_img = nib.as_closest_canonical(img)
        data = reoriented_img.get_fdata()
        new_affine = reoriented_img.affine
        
        print("New data shape after reorient:", data.shape)
        print("New affine matrix:")
        print(new_affine)
    else:
        print("Image is already RAS. No reorientation needed.")
        new_affine = affine

    # Always print final orientation codes
    final_ornt_codes = nib.orientations.aff2axcodes(new_affine)
    print("Final orientation codes:", final_ornt_codes)
    print("Final data shape:", data.shape)
    print("--- End Debug ---\n")

    reoriented_img = nib.Nifti1Image(data, new_affine)
    nib.save(reoriented_img, output_path)

def test_reorientation_fix():
    # Test subject
    subject = "sub-I359637_PPMI_SPECT_CN"
    
    # Paths
    raw_path = f"/Volumes/reseng202500013-ndd-ml/data/raw/SPECT/PPMI/CN/{subject}/{subject}.nii"
    test_output_path = f"/tmp/test_reorientation_{subject}_RAS.nii.gz"
    
    print(f"Testing reorientation fix for {subject}")
    print(f"Raw: {raw_path}")
    print(f"Test output: {test_output_path}")
    
    # Test the reorientation
    try:
        reorient_to_RAS(raw_path, test_output_path)
        
        # Load and compare
        raw_img = nib.load(raw_path)
        test_img = nib.load(test_output_path)
        
        raw_data = raw_img.get_fdata()
        test_data = test_img.get_fdata()
        
        # Calculate aspect ratios
        raw_aspect = raw_data.shape[0] / raw_data.shape[1]
        test_aspect = test_data.shape[0] / test_data.shape[1]
        
        print(f"\n=== Results ===")
        print(f"Raw shape: {raw_data.shape}")
        print(f"Test shape: {test_data.shape}")
        print(f"Raw orientation: {nib.orientations.aff2axcodes(raw_img.affine)}")
        print(f"Test orientation: {nib.orientations.aff2axcodes(test_img.affine)}")
        print(f"Raw aspect ratio: {raw_aspect:.3f}")
        print(f"Test aspect ratio: {test_aspect:.3f}")
        
        # Check if reorientation was successful
        raw_ornt = nib.orientations.aff2axcodes(raw_img.affine)
        test_ornt = nib.orientations.aff2axcodes(test_img.affine)
        target_ornt = ('R', 'A', 'S')
        
        # Check if we achieved RAS orientation
        if test_ornt == target_ornt:
            print("✅ SUCCESS: Achieved RAS orientation")
            
            # Check if data integrity is preserved (same number of non-zero voxels)
            raw_nonzero = np.count_nonzero(raw_data)
            test_nonzero = np.count_nonzero(test_data)
            
            if abs(raw_nonzero - test_nonzero) < 100:  # Allow small differences due to interpolation
                print("✅ SUCCESS: Data integrity preserved")
                print(f"   Raw non-zero voxels: {raw_nonzero}")
                print(f"   Test non-zero voxels: {test_nonzero}")
                return True
            else:
                print(f"❌ FAILED: Data integrity compromised")
                print(f"   Raw non-zero voxels: {raw_nonzero}")
                print(f"   Test non-zero voxels: {test_nonzero}")
                return False
        else:
            print(f"❌ FAILED: Did not achieve RAS orientation")
            print(f"   Expected: {target_ornt}")
            print(f"   Got: {test_ornt}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_reorientation_fix()
    if success:
        print("\n🎉 Reorientation fix test PASSED! Ready to run on all images.")
    else:
        print("\n💥 Reorientation fix test FAILED! Need to fix the code.") 