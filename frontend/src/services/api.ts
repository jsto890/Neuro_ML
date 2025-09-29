// API service for communicating with the backend

const API_BASE_URL = 'http://localhost:5001';

export interface UploadResponse {
  success: boolean;
  message: string;
  files: Array<{
    id: string;
    original_name: string;
    saved_name: string;
    file_type: string;
    size_bytes: number;
    upload_time: string;
    status: string;
  }>;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  message: string;
  timestamp: string;
}

export class ApiService {
  static async checkHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    return response.json();
  }

  static async uploadFiles(files: FileList): Promise<UploadResponse> {
    const formData = new FormData();
    
    // Add all files to FormData
    Array.from(files).forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Upload failed');
    }

    return response.json();
  }

  static async getUploadStatus(fileId: string) {
    const response = await fetch(`${API_BASE_URL}/upload-status/${fileId}`);
    if (!response.ok) {
      throw new Error('Failed to get upload status');
    }
    return response.json();
  }

  static async listFiles() {
    const response = await fetch(`${API_BASE_URL}/list-files`);
    if (!response.ok) {
      throw new Error('Failed to list files');
    }
    return response.json();
  }

  static async analyzeFiles(fileIds: string[]) {
    console.log('API: Analyzing files with IDs:', fileIds);
    const response = await fetch(`${API_BASE_URL}/analyze-files`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file_ids: fileIds }),
    });

    console.log('API: Response status:', response.status);
    if (!response.ok) {
      const errorText = await response.text();
      console.error('API: Error response:', errorText);
      throw new Error(`Failed to analyze files: ${response.status} - ${errorText}`);
    }
    return response.json();
  }

  static async convertDicom(fileIds: string[], outputName: string = 'converted') {
    const response = await fetch(`${API_BASE_URL}/convert-dicom`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        file_ids: fileIds, 
        output_name: outputName 
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to convert DICOM files');
    }
    return response.json();
  }

  static async preprocessFiles(fileIds: string[], imageType: string) {
    const response = await fetch(`${API_BASE_URL}/preprocess`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        file_ids: fileIds, 
        image_type: imageType 
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to preprocess files');
    }
    return response.json();
  }

  static async predictParkinson(fileIds: string[], imageType: string) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        file_ids: fileIds, 
        image_type: imageType 
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to run model prediction');
    }
    return response.json();
  }
}
