import React, { useCallback, useState } from 'react';
import { Upload, FileImage, X, Loader2, FolderOpen } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (files: FileList) => void;
  isAnalyzing: boolean;
  uploadType: 'dicom' | 'nifti';
  imageType: 'smri' | 'pet' | 'dat-spect';
  onBack: () => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, isAnalyzing, uploadType, imageType, onBack }) => {
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

  const isValidFile = useCallback((fileName: string): boolean => {
    const name = fileName.toLowerCase();
    if (uploadType === 'dicom') {
      return name.endsWith('.dcm') || name.endsWith('.dicom');
    } else {
      return name.endsWith('.nii') || name.endsWith('.nii.gz');
    }
  }, [uploadType]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const files = Array.from(e.dataTransfer.files).filter(file => isValidFile(file.name));
      setSelectedFiles(files);
    }
  }, [isValidFile]);

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
    if (uploadType === 'dicom') {
      return 'DICOM';
    } else {
      return 'NIFTI';
    }
  };

  const getImageTypeDisplay = (): string => {
    switch (imageType) {
      case 'smri': return 'sMRI';
      case 'pet': return 'PET';
      case 'dat-spect': return 'DAT-SPECT';
      default: return 'Unknown';
    }
  };

  const getUploadTypeDisplay = (): string => {
    return uploadType === 'dicom' ? 'DICOM Folder' : 'NIFTI File';
  };

  return (
    <div className="space-y-6">
      {/* Header with back button and type info */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center text-gray-600 hover:text-gray-800 hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors"
        >
          ← Back to Image Type
        </button>
      </div>

      <div className="text-center">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">
          Upload {getImageTypeDisplay()} Images
        </h2>
        <p className="text-gray-600 mb-4">
          {uploadType === 'dicom' 
            ? 'Select a folder containing DICOM slices or drag and drop DICOM files'
            : 'Select a NIFTI file or drag and drop it here'
          }
        </p>
        
        {/* Type indicators */}
        <div className="flex justify-center space-x-4">
          <div className="flex items-center space-x-2 px-3 py-1 bg-gray-100 rounded-full">
            <span className="text-sm font-medium text-gray-700">
              {getUploadTypeDisplay()}
            </span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1 bg-gray-100 rounded-full">
            <span className="text-sm font-medium text-gray-700">
              {getImageTypeDisplay()}
            </span>
          </div>
        </div>
      </div>

      {/* Drop Zone */}
      <div
        className={`relative border-2 border-dashed rounded-xl p-8 transition-colors ${
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
          multiple={uploadType === 'dicom'}
          {...(uploadType === 'dicom' && { webkitdirectory: 'true' })}
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="text-center">
          {uploadType === 'dicom' ? (
            <FolderOpen className="w-12 h-12 text-orange-400 mx-auto mb-4" />
          ) : (
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          )}
          <p className="text-lg font-medium text-gray-900 mb-2">
            {dragActive 
              ? 'Drop files here' 
              : uploadType === 'dicom' 
                ? 'Choose folder or drag and drop DICOM files'
                : 'Choose file or drag and drop NIFTI file'
            }
          </p>
          <p className="text-sm text-gray-500">
            {uploadType === 'dicom' 
              ? 'DICOM files (.dcm, .dicom) - select folder or individual files'
              : 'NIFTI files (.nii, .nii.gz) up to 100MB'
            }
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
          className={`px-8 py-3 rounded-lg font-medium transition-colors ${
            selectedFiles.length === 0 || isAnalyzing
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg hover:shadow-xl'
          }`}
        >
          {isAnalyzing ? (
            <div className="flex items-center space-x-2">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Processing {getImageTypeDisplay()} Images...</span>
            </div>
          ) : (
            <div className="flex items-center space-x-2">
              <span>Upload {selectedFiles.length} {getImageTypeDisplay()} File{selectedFiles.length !== 1 ? 's' : ''}</span>
              <Upload className="w-4 h-4" />
            </div>
          )}
        </button>
      </div>
    </div>
  );
};

export default FileUpload;
