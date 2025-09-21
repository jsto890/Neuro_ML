# P4P Backend

Simple Flask backend for handling medical image uploads.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5001`

## Endpoints

- `GET /health` - Health check
- `POST /upload` - Upload medical images (NIFTI/DICOM)
- `GET /upload-status/<file_id>` - Get upload status
- `GET /list-files` - List all uploaded files

## Features

- Accepts NIFTI (.nii, .nii.gz) and DICOM (.dcm, .dicom) files
- Saves files to `uploads/` directory
- Generates unique file IDs
- CORS enabled for frontend communication
- File type validation
- Detailed logging

## File Structure

```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── uploads/           # Uploaded files directory
└── README.md          # This file
```
