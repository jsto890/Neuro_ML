import React from 'react';
import { FolderOpen, FileImage, ArrowRight } from 'lucide-react';

interface UploadTypeSelectorProps {
  onSelectType: (type: 'dicom' | 'nifti') => void;
}

const UploadTypeSelector: React.FC<UploadTypeSelectorProps> = ({ onSelectType }) => {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">
          Choose Upload Type
        </h2>
        <p className="text-gray-600">
          Select how you want to upload your medical images
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* DICOM Folder Option */}
        <div
          onClick={() => onSelectType('dicom')}
          className="group cursor-pointer bg-white rounded-xl border-2 border-gray-200 hover:border-primary-500 hover:shadow-lg transition-all duration-200 p-8"
        >
          <div className="text-center">
            <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-orange-200 transition-colors">
              <FolderOpen className="w-8 h-8 text-orange-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              DICOM Folder
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Upload a folder containing DICOM slices (.dcm files)
            </p>
            <div className="flex items-center justify-center text-primary-600 group-hover:text-primary-700">
              <span className="text-sm font-medium">Select Folder</span>
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>

        {/* NIFTI File Option */}
        <div
          onClick={() => onSelectType('nifti')}
          className="group cursor-pointer bg-white rounded-xl border-2 border-gray-200 hover:border-primary-500 hover:shadow-lg transition-all duration-200 p-8"
        >
          <div className="text-center">
            <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-indigo-200 transition-colors">
              <FileImage className="w-8 h-8 text-indigo-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              NIFTI File
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Upload a single NIFTI file (.nii or .nii.gz)
            </p>
            <div className="flex items-center justify-center text-primary-600 group-hover:text-primary-700">
              <span className="text-sm font-medium">Select File</span>
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>
      </div>

      <div className="text-center">
        <p className="text-xs text-gray-500">
          Choose the option that matches your data format
        </p>
      </div>
    </div>
  );
};

export default UploadTypeSelector;
