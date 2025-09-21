#!/usr/bin/env python3
"""
Simple Flask backend for P4P Project
Handles image uploads and returns success responses
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from datetime import datetime
import logging
from pathlib import Path
from dicom_converter import DicomConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize DICOM converter
dicom_converter = DicomConverter(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.nii', '.nii.gz', '.dcm', '.dicom'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def get_file_type(filename):
    """Determine file type based on extension"""
    if filename.lower().endswith('.dcm') or filename.lower().endswith('.dicom'):
        return 'DICOM'
    elif filename.lower().endswith('.nii') or filename.lower().endswith('.nii.gz'):
        return 'NIFTI'
    else:
        return 'Unknown'

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'P4P Backend is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test-analyze', methods=['GET'])
def test_analyze():
    """Test endpoint to check if analyze-files is working"""
    return jsonify({
        'success': True,
        'message': 'Test endpoint working',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Process each file
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                # Generate unique filename
                file_id = str(uuid.uuid4())
                file_extension = os.path.splitext(file.filename)[1]
                if file.filename.lower().endswith('.nii.gz'):
                    file_extension = '.nii.gz'
                
                filename = f"{file_id}{file_extension}"
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                
                # Save file
                file.save(file_path)
                
                # Get file info
                file_size = os.path.getsize(file_path)
                file_type = get_file_type(file.filename)
                
                # Log the upload
                logger.info(f"Image received: {file.filename} -> {filename}")
                logger.info(f"File type: {file_type}, Size: {file_size} bytes")
                
                # Create result
                result = {
                    'id': file_id,
                    'original_name': file.filename,
                    'saved_name': filename,
                    'file_type': file_type,
                    'size_bytes': file_size,
                    'upload_time': datetime.now().isoformat(),
                    'status': 'received'
                }
                results.append(result)
                
            else:
                logger.warning(f"Invalid file type: {file.filename}")
                results.append({
                    'id': str(uuid.uuid4()),
                    'original_name': file.filename,
                    'error': 'Invalid file type',
                    'status': 'rejected'
                })
        
        # Return success response
        return jsonify({
            'success': True,
            'message': f'Successfully received {len([r for r in results if r.get("status") == "received"])} file(s)',
            'files': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500

@app.route('/upload-status/<file_id>', methods=['GET'])
def get_upload_status(file_id):
    """Get status of uploaded file"""
    # For now, just return a simple status
    return jsonify({
        'file_id': file_id,
        'status': 'received',
        'message': 'File successfully received and saved',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/list-files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                files.append({
                    'filename': filename,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
        
        return jsonify({
            'success': True,
            'files': files,
            'count': len(files)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analyze-files', methods=['POST'])
def analyze_files():
    """Analyze uploaded files and determine conversion needs"""
    logger.info("analyze_files endpoint called")
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")
        file_ids = data.get('file_ids', [])
        
        if not file_ids:
            logger.warning("No file IDs provided")
            return jsonify({
                'success': False,
                'error': 'No file IDs provided'
            }), 400
        
        # Get file paths for the provided IDs
        file_paths = []
        for file_id in file_ids:
            file_path = os.path.join(UPLOAD_FOLDER, file_id)
            if os.path.exists(file_path):
                file_paths.append(Path(file_path))
        
        if not file_paths:
            return jsonify({
                'success': False,
                'error': 'No valid files found'
            }), 404
        
        # Analyze files
        analysis = dicom_converter.get_conversion_info(file_paths)
        
        return jsonify({
            'success': True,
            'analysis': {
                'dicom_count': len(analysis['dicom_files']),
                'nifti_count': len(analysis['nifti_files']),
                'unknown_count': len(analysis['unknown_files']),
                'needs_conversion': analysis['needs_conversion'],
                'total_files': analysis['total_files'],
                'dicom_files': [str(f) for f in analysis['dicom_files']],
                'nifti_files': [str(f) for f in analysis['nifti_files']],
                'unknown_files': [str(f) for f in analysis['unknown_files']]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing files: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/convert-dicom', methods=['POST'])
def convert_dicom():
    """Convert DICOM files to NIFTI"""
    try:
        data = request.get_json()
        file_ids = data.get('file_ids', [])
        output_name = data.get('output_name', 'converted')
        
        if not file_ids:
            return jsonify({
                'success': False,
                'error': 'No file IDs provided'
            }), 400
        
        # Get DICOM file paths
        dicom_files = []
        for file_id in file_ids:
            file_path = os.path.join(UPLOAD_FOLDER, file_id)
            if os.path.exists(file_path):
                dicom_files.append(Path(file_path))
        
        if not dicom_files:
            return jsonify({
                'success': False,
                'error': 'No DICOM files found'
            }), 404
        
        # Convert DICOM to NIFTI
        conversion_result = dicom_converter.convert_dicom_to_nifti(dicom_files, output_name)
        
        if conversion_result['success']:
            # Get file info
            output_path = Path(conversion_result['output_path'])
            file_size = output_path.stat().st_size
            
            return jsonify({
                'success': True,
                'conversion': {
                    'method': conversion_result['method'],
                    'message': conversion_result['message'],
                    'output_file': output_path.name,
                    'output_path': str(output_path),
                    'file_size_bytes': file_size,
                    'file_size_mb': round(file_size / (1024 * 1024), 2)
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': conversion_result['error']
            }), 500
        
    except Exception as e:
        logger.error(f"Error converting DICOM: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting P4P Backend...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /upload - Upload files")
    print("  GET  /upload-status/<file_id> - Get file status")
    print("  GET  /list-files - List uploaded files")
    print("  POST /analyze-files - Analyze file types")
    print("  POST /convert-dicom - Convert DICOM to NIFTI")
    print("\nStarting server on http://localhost:5001")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
