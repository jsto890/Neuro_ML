#!/usr/bin/env python3
"""
Test script to verify the new Desktop SPECT folder structure
"""

import os
from pathlib import Path

def test_desktop_paths():
    """Test if the new Desktop SPECT paths exist and contain data"""
    
    base_dir = "/Users/jacksonschofield/Desktop/SPECT"
    
    print("🔍 Testing Desktop SPECT folder structure...")
    print(f"Base directory: {base_dir}")
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return False
    
    # Check for CN and PD folders
    cn_dir = os.path.join(base_dir, "CN_SPECT_PPMI_NIfTI")
    pd_dir = os.path.join(base_dir, "PD_SPECT_PPMI_NIfTI")
    
    print(f"\n📁 CN directory: {cn_dir}")
    print(f"📁 PD directory: {pd_dir}")
    
    # Test CN directory
    if os.path.exists(cn_dir):
        cn_subjects = [d for d in os.listdir(cn_dir) if d.startswith('sub-')]
        print(f"✅ CN directory found with {len(cn_subjects)} subjects")
        
        if cn_subjects:
            # Check first subject for NIfTI files
            first_subject = cn_subjects[0]
            subject_dir = os.path.join(cn_dir, first_subject)
            nii_files = [f for f in os.listdir(subject_dir) if f.endswith('.nii.gz')]
            print(f"   📊 First subject '{first_subject}' has {len(nii_files)} NIfTI files")
            
            if nii_files:
                print(f"   📄 Files: {nii_files[:5]}...")  # Show first 5 files
            else:
                print(f"   ⚠️ No NIfTI files found in {first_subject}")
    else:
        print(f"❌ CN directory not found: {cn_dir}")
    
    # Test PD directory
    if os.path.exists(pd_dir):
        pd_subjects = [d for d in os.listdir(pd_dir) if d.startswith('sub-')]
        print(f"✅ PD directory found with {len(pd_subjects)} subjects")
        
        if pd_subjects:
            # Check first subject for NIfTI files
            first_subject = pd_subjects[0]
            subject_dir = os.path.join(pd_dir, first_subject)
            nii_files = [f for f in os.listdir(subject_dir) if f.endswith('.nii.gz')]
            print(f"   📊 First subject '{first_subject}' has {len(nii_files)} NIfTI files")
            
            if nii_files:
                print(f"   📄 Files: {nii_files[:5]}...")  # Show first 5 files
            else:
                print(f"   ⚠️ No NIfTI files found in {first_subject}")
    else:
        print(f"❌ PD directory not found: {pd_dir}")
    
    print("\n🎯 Path structure test completed!")
    return True

if __name__ == "__main__":
    test_desktop_paths()
