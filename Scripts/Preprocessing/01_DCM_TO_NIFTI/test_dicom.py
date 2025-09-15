#!/usr/bin/env python3
"""
Test script to verify DICOM file access and basic functionality
"""

import pydicom
from pathlib import Path

def test_dicom_access():
    """Test if we can access the DICOM file"""
    dicom_path = Path("/Users/jacksonschofield/Desktop/CN_SPECT_PPMI/3000/Raw_Data/2011-01-20_16_28_47.0/I248908/PPMI_3000_NM_Raw_Data_br_raw_20110805101009028_1_S117534_I248908.dcm")
    
    print(f"Testing DICOM file access...")
    print(f"File path: {dicom_path}")
    print(f"File exists: {dicom_path.exists()}")
    
    if not dicom_path.exists():
        print("❌ DICOM file not found!")
        return False
    
    try:
        ds = pydicom.dcmread(dicom_path)
        print("✅ Successfully read DICOM file!")
        
        # Print basic info
        print(f"Patient ID: {getattr(ds, 'PatientID', 'Unknown')}")
        print(f"Patient Name: {getattr(ds, 'PatientName', 'Unknown')}")
        print(f"Study Date: {getattr(ds, 'StudyDate', 'Unknown')}")
        print(f"Modality: {getattr(ds, 'Modality', 'Unknown')}")
        print(f"Series Description: {getattr(ds, 'SeriesDescription', 'Unknown')}")
        print(f"Image dimensions: {ds.pixel_array.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading DICOM file: {e}")
        return False

def test_folder_structure():
    """Test if the CN_SPECT_PPMI folder structure exists"""
    base_path = Path("/Users/jacksonschofield/Desktop/CN_SPECT_PPMI")
    
    print(f"\nTesting folder structure...")
    print(f"Base path: {base_path}")
    print(f"Base path exists: {base_path.exists()}")
    
    if not base_path.exists():
        print("❌ Base folder not found!")
        return False
    
    # Count DICOM files
    dicom_count = 0
    for file_path in base_path.rglob("*.dcm"):
        dicom_count += 1
    
    print(f"Found {dicom_count} DICOM files in total")
    
    return dicom_count > 0

if __name__ == "__main__":
    print("=== DICOM Access Test ===")
    
    # Test individual file
    file_ok = test_dicom_access()
    
    # Test folder structure
    folder_ok = test_folder_structure()
    
    if file_ok and folder_ok:
        print("\n✅ All tests passed! Ready to run conversion.")
    else:
        print("\n❌ Some tests failed. Please check the paths and permissions.")
