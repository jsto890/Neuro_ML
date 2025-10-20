import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import pydicom
import nibabel as nib
import numpy as np

class DicomConverter:
    """Utility class for converting DICOM files to NIFTI format"""
    
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.conversion_dir = self.upload_dir / "conversions"
        self.conversion_dir.mkdir(exist_ok=True)
    
    def detect_file_type(self, file_path: Path) -> str:
        """Detect if file is DICOM or NIFTI"""
        file_ext = file_path.suffix.lower()
        
        if file_ext in ['.nii', '.gz']:
            return 'NIFTI'
        elif file_ext in ['.dcm', '.dicom']:
            return 'DICOM'
        else:
            return 'UNKNOWN'
    
    def is_dicom_file(self, file_path: Path) -> bool:
        """Check if file is a valid DICOM file"""
        try:
            pydicom.dcmread(str(file_path))
            return True
        except:
            return False
    
    def convert_dicom_to_nifti(self, dicom_files: List[Path], output_name: str) -> Dict[str, Any]:
        """Convert DICOM files to NIFTI using multiple strategies"""
        
        # Create temporary directory for conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy DICOM files to temp directory
            temp_dicom_dir = temp_path / "dicom"
            temp_dicom_dir.mkdir()
            
            for dicom_file in dicom_files:
                shutil.copy2(dicom_file, temp_dicom_dir)
            
            # Try different conversion methods
            conversion_result = self._try_conversion_methods(temp_dicom_dir, output_name)
            
            if conversion_result['success']:
                # Move converted file to conversion directory
                final_path = self.conversion_dir / f"{output_name}.nii.gz"
                shutil.move(conversion_result['output_path'], final_path)
                conversion_result['output_path'] = str(final_path)
            
            return conversion_result
    
    def _try_conversion_methods(self, dicom_dir: Path, output_name: str) -> Dict[str, Any]:
        """Try different DICOM to NIFTI conversion methods"""
        
        # Method 1: Try dcm2niix (best quality)
        if self._check_dcm2niix():
            result = self._try_dcm2niix_conversion(dicom_dir, output_name)
            if result['success']:
                return result
        
        # Method 2: Try pydicom + nibabel (fallback)
        result = self._try_pydicom_nibabel_conversion(dicom_dir, output_name)
        if result['success']:
            return result
        
        return {
            'success': False,
            'error': 'All conversion methods failed',
            'output_path': None
        }
    
    def _check_dcm2niix(self) -> bool:
        """Check if dcm2niix is available"""
        try:
            subprocess.run(['dcm2niix', '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _try_dcm2niix_conversion(self, dicom_dir: Path, output_name: str) -> Dict[str, Any]:
        """Try conversion using dcm2niix"""
        try:
            output_dir = dicom_dir.parent / "nifti_output"
            output_dir.mkdir(exist_ok=True)
            
            cmd = [
                "dcm2niix",
                "-b", "y",   # write JSON sidecar
                "-z", "y",   # gzip
                "-m", "y",   # merge 2D slices/frames into 3D
                "-x", "n",   # do NOT crop
                "-f", output_name,  # filename template
                "-o", str(output_dir),
                str(dicom_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find the generated NIFTI file
            nifti_files = list(output_dir.glob(f"{output_name}*.nii.gz"))
            if nifti_files:
                return {
                    'success': True,
                    'method': 'dcm2niix',
                    'output_path': nifti_files[0],
                    'message': f'Successfully converted using dcm2niix'
                }
            else:
                return {
                    'success': False,
                    'error': 'dcm2niix completed but no NIFTI file found'
                }
                
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'error': f'dcm2niix failed: {e.stderr}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'dcm2niix error: {str(e)}'
            }
    
    def _try_pydicom_nibabel_conversion(self, dicom_dir: Path, output_name: str) -> Dict[str, Any]:
        """Try conversion using pydicom + nibabel (fallback method)"""
        try:
            # Read all DICOM files
            dicom_files = list(dicom_dir.glob("*.dcm"))
            if not dicom_files:
                return {
                    'success': False,
                    'error': 'No DICOM files found'
                }
            
            # Sort files by name
            dicom_files.sort(key=lambda x: x.name)
            
            # Read first DICOM to get dimensions
            first_ds = pydicom.dcmread(dicom_files[0])
            
            # Check if this is a 3D volume or 2D slices
            if hasattr(first_ds, 'NumberOfFrames') and first_ds.NumberOfFrames > 1:
                # 3D volume in single file
                pixel_array = first_ds.pixel_array
                
                # Apply rescale if available
                if hasattr(first_ds, 'RescaleSlope') and hasattr(first_ds, 'RescaleIntercept'):
                    slope = float(first_ds.RescaleSlope)
                    intercept = float(first_ds.RescaleIntercept)
                    pixel_array = pixel_array.astype(np.float32) * slope + intercept
                
                # Create NIfTI image
                nii_img = nib.Nifti1Image(pixel_array, np.eye(4))
                
            else:
                # 2D slices that need to be stacked
                rows = first_ds.Rows
                cols = first_ds.Columns
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
            
            # Save NIFTI file
            output_path = dicom_dir.parent / f"{output_name}.nii.gz"
            nib.save(nii_img, output_path)
            
            return {
                'success': True,
                'method': 'pydicom+nibabel',
                'output_path': output_path,
                'message': f'Successfully converted using pydicom+nibabel'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'pydicom+nibabel conversion failed: {str(e)}'
            }
    
    def get_conversion_info(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Analyze uploaded files and determine conversion needs"""
        dicom_files = []
        nifti_files = []
        unknown_files = []
        
        for file_path in file_paths:
            file_type = self.detect_file_type(file_path)
            
            if file_type == 'DICOM' and self.is_dicom_file(file_path):
                dicom_files.append(file_path)
            elif file_type == 'NIFTI':
                nifti_files.append(file_path)
            else:
                unknown_files.append(file_path)
        
        return {
            'dicom_files': dicom_files,
            'nifti_files': nifti_files,
            'unknown_files': unknown_files,
            'needs_conversion': len(dicom_files) > 0,
            'total_files': len(file_paths)
        }
