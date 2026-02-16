# Backend

Flask API for file upload, conversion checks, and analysis orchestration.

## Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Default API URL: `http://localhost:5001`.

## Core endpoints

- `GET /health` for service health checks
- `POST /upload` for NIfTI and DICOM uploads
- `GET /upload-status/<file_id>` for upload status
- `GET /list-files` for uploaded file listing
- `POST /analyze-files` for post-upload analysis flow

## Notes

- Uploaded files are written to `backend/uploads/`.
- `backend/uploads/` is excluded from version control.
- CORS is enabled for local frontend integration.
