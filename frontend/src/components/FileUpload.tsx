import React, { useCallback, useState } from 'react';
import { Upload, FileImage, X, Loader2 } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (files: FileList) => void;
  isAnalyzing: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, isAnalyzing }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const isValidFile = (fileName: string): boolean => {
    const name = fileName.toLowerCase();
    return name.endsWith('.nii') || 
           name.endsWith('.nii.gz') ||
           name.endsWith('.dcm') ||
           name.endsWith('.dicom');
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const files = Array.from(e.dataTransfer.files).filter(file => isValidFile(file.name));
      setSelectedFiles(files);
    }
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files).filter(file => isValidFile(file.name));
      setSelectedFiles(files);
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = () => {
    if (selectedFiles.length > 0) {
      const fileList = selectedFiles.reduce((dataTransfer, file) => {
        dataTransfer.items.add(file);
        return dataTransfer;
      }, new DataTransfer());
      
      onFileUpload(fileList.files);
      setSelectedFiles([]);
    }
  };

  const getFileType = (fileName: string): string => {
    const name = fileName.toLowerCase();
    if (name.includes('spect')) return 'SPECT';
    if (name.includes('mri')) return 'MRI';
    if (name.includes('pet')) return 'PET';
    if (name.endsWith('.dcm') || name.endsWith('.dicom')) return 'DICOM';
    if (name.endsWith('.nii') || name.endsWith('.nii.gz')) return 'NIFTI';
    return 'Unknown';
  };

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-lg font-medium text-gray-900 mb-1">
          Upload Medical Images
        </h2>
        <p className="text-sm text-gray-600">
          Drag and drop NIFTI or DICOM files or click to browse
        </p>
      </div>

      {/* Drop Zone */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-6 transition-colors ${
          dragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          multiple
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="text-center">
          <Upload className="w-8 h-8 text-gray-400 mx-auto mb-3" />
          <p className="text-base font-medium text-gray-900 mb-1">
            {dragActive ? 'Drop files here' : 'Choose files or drag and drop'}
          </p>
          <p className="text-xs text-gray-500">
            NIFTI (.nii, .nii.gz) or DICOM (.dcm, .dicom) files up to 100MB each
          </p>
        </div>
      </div>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-medium text-gray-700">Selected Files:</h3>
          <div className="space-y-1">
            {selectedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 bg-gray-50 rounded"
              >
                <div className="flex items-center space-x-2">
                  <FileImage className="w-4 h-4 text-gray-400" />
                  <div>
                    <p className="text-xs font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">
                      {getFileType(file.name)} • {(file.size / 1024 / 1024).toFixed(1)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload Button */}
      <div className="flex justify-center">
        <button
          onClick={handleUpload}
          disabled={selectedFiles.length === 0 || isAnalyzing}
          className={`px-6 py-2 rounded font-medium transition-colors ${
            selectedFiles.length === 0 || isAnalyzing
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-primary-600 text-white hover:bg-primary-700'
          }`}
        >
          {isAnalyzing ? (
            <div className="flex items-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Analyzing...</span>
            </div>
          ) : (
            `Analyze ${selectedFiles.length} File${selectedFiles.length !== 1 ? 's' : ''}`
          )}
        </button>
      </div>
    </div>
  );
};

export default FileUpload;
