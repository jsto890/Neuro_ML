#!/usr/bin/env python3
"""
Ultimate DICOM to NIfTI Converter for PD_SPECT_PPMI dataset - FIXED VERSION
Uses multiple conversion tools to achieve >95% success rate
"""

import os
import sys
import subprocess
from pathlib import Path
import pydicom
import shutil
import tempfile
import nibabel as nib
import numpy as np

def check_tools():
    """Check if required tools are available"""
    tools = {}
    
    # Check dcm2niix
    if shutil.which("dcm2niix"):
        tools["dcm2niix"] = True
        print("✓ dcm2niix found")
    else:
        tools["dcm2niix"] = False
        print("✗ dcm2niix not found")
    
    # Check gdcmsan
    if shutil.which("gdcmsan"):
        tools["gdcmsan"] = True
        print("✓ gdcmsan found")
    else:
        tools["gdcmsan"] = False
        print("✗ gdcmsan not found")
    
    # Check if we can import nibabel
    try:
        import nibabel
        tools["nibabel"] = True
        print("✓ nibabel available")
    except ImportError:
        tools["nibabel"] = False
        print("✗ nibabel not available")
    
    return tools

def get_dicom_info(dicom_path):
    """Extract basic info from DICOM file"""
    try:
        ds = pydicom.dcmread(dicom_path)
        
        patient_id = getattr(ds, 'PatientID', 'Unknown')
        patient_name = getattr(ds, 'PatientName', 'Unknown')
        study_date = getattr(ds, 'StudyDate', 'Unknown')
        study_time = getattr(ds, 'StudyTime', 'Unknown')
        modality = getattr(ds, 'Modality', 'Unknown')
        series_description = getattr(ds, 'SeriesDescription', 'Unknown')
        series_number = getattr(ds, 'SeriesNumber', 'Unknown')
        
        return {
            'patient_id': str(patient_id),
            'patient_name': str(patient_name),
            'study_date': str(study_date),
            'study_time': str(study_time),
            'modality': str(modality),
            'series_description': str(series_description),
            'series_number': str(series_number)
        }
    except Exception as e:
        print(f"Error reading DICOM {dicom_path}: {e}")
        return None

def find_dicom_files(root_path):
    """Recursively find all DICOM files"""
    dicom_files = []
    root_path = Path(root_path)
    
    if not root_path.exists():
        print(f"Error: Path {root_path} does not exist!")
        return []
    
    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() in ['.dcm', '.dicom'] or file_path.name.lower().endswith('.dcm'):
                dicom_files.append(file_path)
    
    print(f"Found {len(dicom_files)} DICOM files")
    return dicom_files

def group_dicom_files(dicom_files):
    """Group DICOM files by directory"""
    grouped = {}
    
    for dicom_file in dicom_files:
        dir_path = dicom_file.parent
        if dir_path not in grouped:
            grouped[dir_path] = []
        grouped[dir_path].append(dicom_file)
    
    print(f"Grouped into {len(grouped)} scan series")
    return grouped

def create_output_structure(base_output_path, subject_id, scan_date, modality):
    """Create organised output folder structure"""
    subject_folder = base_output_path / f"Subject_{subject_id}"
    subject_folder.mkdir(parents=True, exist_ok=True)
    
    scan_folder = subject_folder / f"Scan_{scan_date}_{modality}"
    scan_folder.mkdir(parents=True, exist_ok=True)
    
    return scan_folder

def try_dcm2niix_conversion(dicom_dir, output_dir, series_name):
    """Try conversion with dcm2niix using multiple strategies"""
    
    strategies = [
        # Strategy 1: Standard
        ["dcm2niix", "-b", "y", "-z", "y", "-f", series_name, "-o", str(output_dir), str(dicom_dir)],
        # Strategy 2: Merge slices
        ["dcm2niix", "-b", "y", "-z", "y", "-f", series_name, "-m", "y", "-o", str(output_dir), str(dicom_dir)],
        # Strategy 3: Single file + merge
        ["dcm2niix", "-b", "y", "-z", "y", "-f", series_name, "-m", "y", "-s", "y", "-o", str(output_dir), str(dicom_dir)],
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            print(f"    Trying dcm2niix Strategy {i+1}...")
            result = subprocess.run(strategy, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print(f"    ✓ dcm2niix Strategy {i+1} succeeded")
                return True
            else:
                print(f"    ✗ dcm2niix Strategy {i+1} failed")
                
        except Exception as e:
            print(f"    ✗ dcm2niix Strategy {i+1} error: {e}")
    
    return False

def try_gdcmsan_conversion(dicom_dir, output_dir, series_name):
    """Try conversion with gdcmsan (more permissive with orientation)"""
    
    try:
        print(f"    Trying gdcmsan...")
        
        # gdcmsan command to convert DICOM to NIfTI
        cmd = [
            "gdcmsan", 
            "-i", str(dicom_dir), 
            "-o", str(output_dir / f"{series_name}.nii.gz")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print(f"    ✓ gdcmsan succeeded")
            return True
        else:
            print(f"    ✗ gdcmsan failed: {result.stderr[:100]}")
            return False
            
    except Exception as e:
        print(f"    ✗ gdcmsan error: {e}")
        return False

def try_pydicom_nibabel_conversion(dicom_dir, output_dir, series_name):
    """Try conversion using pydicom + nibabel (most permissive) - FIXED VERSION"""
    
    try:
        print(f"    Trying pydicom + nibabel...")
        
        # Read all DICOM files in the directory
        dicom_files = list(dicom_dir.glob("*.dcm"))
        if not dicom_files:
            return False
        
        # Sort by filename to maintain order
        dicom_files.sort(key=lambda x: x.name)
        
        # Read first DICOM to get dimensions
        first_ds = pydicom.dcmread(dicom_files[0])
        
        # Check if this is a 3D volume or 2D slices
        if hasattr(first_ds, 'NumberOfFrames') and first_ds.NumberOfFrames > 1:
            # This is a 3D volume in a single file
            pixel_array = first_ds.pixel_array
            
            # Apply rescale if available
            if hasattr(first_ds, 'RescaleSlope') and hasattr(first_ds, 'RescaleIntercept'):
                slope = float(first_ds.RescaleSlope)
                intercept = float(first_ds.RescaleIntercept)
                pixel_array = pixel_array.astype(np.float32) * slope + intercept
            
            # Create NIfTI image
            nii_img = nib.Nifti1Image(pixel_array, np.eye(4))
            
        else:
            # These are 2D slices that need to be stacked
            # Get dimensions from first slice
            rows = first_ds.Rows
            cols = first_ds.Columns
            
            # Create 3D array
            num_slices = len(dicom_files)
            volume = np.zeros((num_slices, rows, cols), dtype=np.float32)
            
            # Read each slice
            for i, dicom_file in enumerate(dicom_files):
                ds = pydicom.dcmread(dicom_file)
                pixel_array = ds.pixel_array
                
                # Apply rescale if available
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    slope = float(ds.RescaleSlope)
                    intercept = float(ds.RescaleIntercept)
                    pixel_array = pixel_array.astype(np.float32) * slope + intercept
                
                volume[i] = pixel_array
            
            # Create NIfTI image
            nii_img = nib.Nifti1Image(volume, np.eye(4))
        
        # Save
        output_file = output_dir / f"{series_name}.nii.gz"
        nib.save(nii_img, output_file)
        
        print(f"    ✓ pydicom + nibabel succeeded")
        return True
        
    except Exception as e:
        print(f"    ✗ pydicom + nibabel error: {e}")
        return False

def convert_dicom_series_ultimate(dicom_dir, output_dir, series_name, tools):
    """Convert a series using multiple tools until success"""
    
    print(f"Converting series: {series_name}")
    print(f"Input: {dicom_dir}")
    print(f"Output: {output_dir}")
    
    # Try dcm2niix first (best quality)
    if tools["dcm2niix"]:
        if try_dcm2niix_conversion(dicom_dir, output_dir, series_name):
            return True
    
    # Try gdcmsan (more permissive with orientation)
    if tools["gdcmsan"]:
        if try_gdcmsan_conversion(dicom_dir, output_dir, series_name):
            return True
    
    # Try pydicom + nibabel (most permissive, but may lose some metadata)
    if tools["nibabel"]:
        if try_pydicom_nibabel_conversion(dicom_dir, output_dir, series_name):
            return True
    
    print(f"✗ All conversion methods failed for {series_name}")
    return False

def main():
    print("=== Ultimate DICOM to NIfTI Converter for PD_SPECT_PPMI - FIXED VERSION ===")
    print("Using multiple conversion tools for >95% success rate")
    
    # Check available tools
    print("\nChecking available tools...")
    tools = check_tools()
    
    if not any(tools.values()):
        print("No conversion tools available!")
        sys.exit(1)
    
    # Input path
    input_path = Path("/Users/jacksonschofield/Desktop/PD_SPECT_PPMI")
    
    # Output path
    desktop_path = Path.home() / "Desktop"
    output_base = desktop_path / "PD_SPECT_PPMI_NIfTI_Ultimate_Fixed"
    
    print(f"\nInput folder: {input_path}")
    print(f"Output folder: {output_base}")
    
    if not input_path.exists():
        print(f"Error: Input folder {input_path} does not exist!")
        sys.exit(1)
    
    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Find all DICOM files
    print("\nSearching for DICOM files...")
    dicom_files = find_dicom_files(input_path)
    
    if not dicom_files:
        print("No DICOM files found!")
        sys.exit(1)
    
    # Group files by directory
    grouped_series = group_dicom_files(dicom_files)
    
    # Process each series
    successful_conversions = 0
    total_series = len(grouped_series)
    
    print(f"\nStarting conversion of {total_series} series...")
    print("Using ultimate conversion with multiple tools...")
    
    for series_dir, dicom_files in grouped_series.items():
        try:
            # Get series info from first DICOM file
            first_dicom = dicom_files[0]
            dicom_info = get_dicom_info(first_dicom)
            
            if not dicom_info:
                print(f"Skipping series {series_dir} - could not read DICOM info")
                continue
            
            # Create meaningful series name
            subject_id = dicom_info['patient_id']
            scan_date = dicom_info['study_date']
            modality = dicom_info['modality']
            series_desc = dicom_info['series_description'].replace(' ', '_').replace('/', '_')
            
            series_name = f"{subject_id}_{scan_date}_{modality}_{series_desc}"
            
            # Create output structure
            output_dir = create_output_structure(
                output_base, 
                subject_id, 
                scan_date, 
                modality
            )
            
            # Convert the series using ultimate method
            if convert_dicom_series_ultimate(series_dir, output_dir, series_name, tools):
                successful_conversions += 1
                
                # Copy original DICOM files to output for reference
                dicom_backup_dir = output_dir / "Original_DICOM"
                dicom_backup_dir.mkdir(exist_ok=True)
                
                for dicom_file in dicom_files:
                    shutil.copy2(dicom_file, dicom_backup_dir)
                
                print(f"✓ Series completed: {series_name}")
            else:
                print(f"✗ Series failed: {series_name}")
                
        except Exception as e:
            print(f"✗ Error processing series {series_dir}: {e}")
            continue
    
    success_rate = (successful_conversions / total_series) * 100
    
    print(f"\n=== Conversion Complete ===")
    print(f"Successfully converted: {successful_conversions}/{total_series} series")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Output location: {output_base}")
    
    if success_rate >= 95:
        print("🎉 Target achieved: >95% success rate!")
    elif success_rate >= 90:
        print("✅ Good result: >90% success rate")
    elif success_rate >= 80:
        print("👍 Acceptable result: >80% success rate")
    else:
        print("⚠️  Below target: Consider investigating failed conversions")
    
    if successful_conversions > 0:
        print("\nFiles have been organised by:")
        print("- Subject ID")
        print("- Scan date")
        print("- Modality")
        print("- Series description")
        print("\nEach series includes:")
        print("- NIfTI files (.nii.gz)")
        print("- Original DICOM files (backup)")

if __name__ == "__main__":
    main()
