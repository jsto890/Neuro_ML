#!/usr/bin/env python3
"""
Improved DICOM to NIfTI Converter for CN_SPECT_PPMI dataset
Uses multiple conversion strategies to achieve >95% success rate
"""

import os
import sys
import subprocess
from pathlib import Path
import pydicom
import shutil
import tempfile

def check_dcm2niix():
    """Check if dcm2niix is available"""
    if shutil.which("dcm2niix") is None:
        print("dcm2niix not found. Please install it first:")
        print("brew install dcm2niix")
        sys.exit(1)
    print("✓ dcm2niix found!")

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
    """Create organized output folder structure"""
    subject_folder = base_output_path / f"Subject_{subject_id}"
    subject_folder.mkdir(parents=True, exist_ok=True)
    
    scan_folder = subject_folder / f"Scan_{scan_date}_{modality}"
    scan_folder.mkdir(parents=True, exist_ok=True)
    
    return scan_folder

def try_conversion_strategy(dicom_dir, output_dir, series_name, strategy_num):
    """Try different conversion strategies with increasing flexibility"""
    
    strategies = [
        # Strategy 1: Standard conversion with strict orientation
        [
            "dcm2niix",
            "-b", "y",           # Write BIDS JSON sidecar
            "-z", "y",           # Compress output
            "-f", series_name,   # Output filename prefix
            "-o", str(output_dir),  # Output directory
            str(dicom_dir)       # Input directory
        ],
        
        # Strategy 2: More flexible orientation handling
        [
            "dcm2niix",
            "-b", "y",           # Write BIDS JSON sidecar
            "-z", "y",           # Compress output
            "-f", series_name,   # Output filename prefix
            "-m", "y",           # Merge 2D slices
            "-o", str(output_dir),  # Output directory
            str(dicom_dir)       # Input directory
        ],
        
        # Strategy 3: Very permissive conversion
        [
            "dcm2niix",
            "-b", "y",           # Write BIDS JSON sidecar
            "-z", "y",           # Compress output
            "-f", series_name,   # Output filename prefix
            "-m", "y",           # Merge 2D slices
            "-s", "y",           # Single file output
            "-o", str(output_dir),  # Output directory
            str(dicom_dir)       # Input directory
        ],
        
        # Strategy 4: Force conversion ignoring orientation warnings
        [
            "dcm2niix",
            "-b", "y",           # Write BIDS JSON sidecar
            "-z", "y",           # Compress output
            "-f", series_name,   # Output filename prefix
            "-m", "y",           # Merge 2D slices
            "-s", "y",           # Single file output
            "-v", "n",           # Verbose off
            "-o", str(output_dir),  # Output directory
            str(dicom_dir)       # Input directory
        ]
    ]
    
    if strategy_num >= len(strategies):
        return False, "No more strategies to try"
    
    strategy = strategies[strategy_num]
    
    try:
        print(f"  Trying Strategy {strategy_num + 1}...")
        
        result = subprocess.run(
            strategy, 
            capture_output=True, 
            text=True, 
            check=False  # Don't raise exception on non-zero exit
        )
        
        if result.returncode == 0:
            return True, f"Strategy {strategy_num + 1} succeeded"
        
        # Check if it's an orientation issue
        if "slice orientation varies" in result.stderr or "No valid DICOM images were found" in result.stdout:
            return False, f"Strategy {strategy_num + 1} failed: orientation issue"
        
        return False, f"Strategy {strategy_num + 1} failed: {result.stderr[:100]}"
        
    except Exception as e:
        return False, f"Strategy {strategy_num + 1} error: {str(e)}"

def convert_dicom_series_robust(dicom_dir, output_dir, series_name):
    """Convert a series using multiple strategies until success"""
    
    print(f"Converting series: {series_name}")
    print(f"Input: {dicom_dir}")
    print(f"Output: {output_dir}")
    
    # Try each strategy until one works
    for strategy_num in range(4):
        success, message = try_conversion_strategy(dicom_dir, output_dir, series_name, strategy_num)
        
        if success:
            print(f"✓ {message}")
            return True
        else:
            print(f"✗ {message}")
            if strategy_num < 3:  # Don't print "trying next strategy" for last attempt
                print("  Trying next strategy...")
    
    print(f"✗ All conversion strategies failed for {series_name}")
    return False

def main():
    print("=== Improved DICOM to NIfTI Converter for CN_SPECT_PPMI ===")
    print("Using multiple conversion strategies for >95% success rate")
    
    # Check dependencies
    check_dcm2niix()
    
    # Input path
    input_path = Path("/Users/jacksonschofield/Desktop/CN_SPECT_PPMI")
    
    # Output path
    desktop_path = Path.home() / "Desktop"
    output_base = desktop_path / "CN_SPECT_PPMI_NIfTI_Improved"
    
    print(f"Input folder: {input_path}")
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
    print("Using robust conversion with multiple strategies...")
    
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
            
            # Convert the series using robust method
            if convert_dicom_series_robust(series_dir, output_dir, series_name):
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
    else:
        print("⚠️  Below target: Consider investigating failed conversions")
    
    if successful_conversions > 0:
        print("\nFiles have been organized by:")
        print("- Subject ID")
        print("- Scan date")
        print("- Modality")
        print("- Series description")
        print("\nEach series includes:")
        print("- NIfTI files (.nii.gz)")
        print("- BIDS JSON metadata (.json)")
        print("- Original DICOM files (backup)")

if __name__ == "__main__":
    main()
