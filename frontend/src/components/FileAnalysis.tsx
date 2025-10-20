import React, { useState, useEffect, useCallback } from 'react';
import { CheckCircle, AlertCircle, Loader2, FileImage, Download, Play } from 'lucide-react';
import { ApiService } from '../services/api';

interface FileAnalysisProps {
  uploadedFiles: Array<{
    id: string;
    fileName: string;
    fileType: string;
    size: number;
  }>;
  onContinue: (conversionResults: any) => void;
  onCancel: () => void;
}

interface AnalysisResult {
  dicom_count: number;
  nifti_count: number;
  unknown_count: number;
  needs_conversion: boolean;
  total_files: number;
  dicom_files: string[];
  nifti_files: string[];
  unknown_files: string[];
}

interface ConversionResult {
  method: string;
  message: string;
  output_file: string;
  output_path: string;
  file_size_bytes: number;
  file_size_mb: number;
}

const FileAnalysis: React.FC<FileAnalysisProps> = ({ 
  uploadedFiles, 
  onContinue, 
  onCancel 
}) => {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [conversionResults, setConversionResults] = useState<ConversionResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isConverting, setIsConverting] = useState(false);
  const [conversionComplete, setConversionComplete] = useState(false);

  const analyzeFiles = useCallback(async () => {
    setIsAnalyzing(true);
    try {
      const fileIds = uploadedFiles.map(file => file.id);
      console.log('Analyzing files with IDs:', fileIds);
      const response = await ApiService.analyzeFiles(fileIds);
      console.log('Analysis response:', response);
      
      if (response.success) {
        setAnalysis(response.analysis);
      } else {
        console.error('Analysis failed:', response.error);
        // Set a default analysis to prevent UI errors
        setAnalysis({
          dicom_count: 0,
          nifti_count: uploadedFiles.length,
          unknown_count: 0,
          needs_conversion: false,
          total_files: uploadedFiles.length,
          dicom_files: [],
          nifti_files: uploadedFiles.map(f => f.fileName),
          unknown_files: []
        });
      }
    } catch (error) {
      console.error('Analysis error:', error);
      // Set a default analysis to prevent UI errors
      setAnalysis({
        dicom_count: 0,
        nifti_count: uploadedFiles.length,
        unknown_count: 0,
        needs_conversion: false,
        total_files: uploadedFiles.length,
        dicom_files: [],
        nifti_files: uploadedFiles.map(f => f.fileName),
        unknown_files: []
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [uploadedFiles]);

  useEffect(() => {
    analyzeFiles();
  }, [analyzeFiles]);

  const convertDicomFiles = async () => {
    if (!analysis) return;
    
    setIsConverting(true);
    try {
      const dicomFileIds = uploadedFiles
        .filter(file => file.fileType === 'DICOM')
        .map(file => file.id);
      
      const response = await ApiService.convertDicom(dicomFileIds, 'converted');
      
      if (response.success) {
        setConversionResults(prev => [...prev, response.conversion]);
        setConversionComplete(true);
      } else {
        console.error('Conversion failed:', response.error);
      }
    } catch (error) {
      console.error('Conversion error:', error);
    } finally {
      setIsConverting(false);
    }
  };

  const handleContinue = () => {
    onContinue({
      analysis,
      conversionResults,
      hasConversion: conversionComplete
    });
  };

  const getFileTypeIcon = (fileType: string) => {
    switch (fileType) {
      case 'DICOM': return <FileImage className="w-4 h-4 text-orange-500" />;
      case 'NIFTI': return <FileImage className="w-4 h-4 text-indigo-500" />;
      default: return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getFileTypeColor = (fileType: string) => {
    switch (fileType) {
      case 'DICOM': return 'bg-orange-100 text-orange-800';
      case 'NIFTI': return 'bg-indigo-100 text-indigo-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (isAnalyzing) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="text-center">
          <Loader2 className="w-8 h-8 text-primary-500 animate-spin mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Analyzing Files</h3>
          <p className="text-sm text-gray-600">Detecting file types and conversion requirements...</p>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="text-center">
          <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Analysis Failed</h3>
          <p className="text-sm text-gray-600 mb-4">Could not analyze the uploaded files.</p>
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Analysis Results */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">File Analysis Results</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{analysis.total_files}</div>
            <div className="text-sm text-blue-800">Total Files</div>
          </div>
          <div className="text-center p-3 bg-orange-50 rounded-lg">
            <div className="text-2xl font-bold text-orange-600">{analysis.dicom_count}</div>
            <div className="text-sm text-orange-800">DICOM Files</div>
          </div>
          <div className="text-center p-3 bg-indigo-50 rounded-lg">
            <div className="text-2xl font-bold text-indigo-600">{analysis.nifti_count}</div>
            <div className="text-sm text-indigo-800">NIFTI Files</div>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-600">{analysis.unknown_count}</div>
            <div className="text-sm text-gray-800">Unknown Files</div>
          </div>
        </div>

        {/* File List */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700">Uploaded Files:</h4>
          {uploadedFiles.map((file) => (
            <div key={file.id} className="flex items-center justify-between p-2 bg-gray-50 rounded">
              <div className="flex items-center space-x-2">
                {getFileTypeIcon(file.fileType)}
                <div>
                  <p className="text-sm font-medium text-gray-900">{file.fileName}</p>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 text-xs rounded ${getFileTypeColor(file.fileType)}`}>
                      {file.fileType}
                    </span>
                    <span className="text-xs text-gray-500">
                      {(file.size / 1024 / 1024).toFixed(1)} MB
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Conversion Section */}
      {analysis.needs_conversion && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">DICOM Conversion Required</h3>
          
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 mr-3" />
              <div>
                <h4 className="text-sm font-medium text-yellow-800">Conversion Needed</h4>
                <p className="text-sm text-yellow-700 mt-1">
                  {analysis.dicom_count} DICOM file{analysis.dicom_count !== 1 ? 's' : ''} need{analysis.dicom_count === 1 ? 's' : ''} to be converted to NIFTI format before processing.
                </p>
              </div>
            </div>
          </div>

          {!conversionComplete ? (
            <button
              onClick={convertDicomFiles}
              disabled={isConverting}
              className={`px-6 py-2 rounded font-medium transition-colors ${
                isConverting
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-orange-600 text-white hover:bg-orange-700'
              }`}
            >
              {isConverting ? (
                <div className="flex items-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Converting...</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <Download className="w-4 h-4" />
                  <span>Convert DICOM to NIFTI</span>
                </div>
              )}
            </button>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center text-green-600">
                <CheckCircle className="w-5 h-5 mr-2" />
                <span className="font-medium">Conversion Complete!</span>
              </div>
              
              {conversionResults.map((result, index) => (
                <div key={index} className="bg-green-50 border border-green-200 rounded-lg p-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-green-800">{result.output_file}</p>
                      <p className="text-xs text-green-600">
                        {result.method} • {result.file_size_mb} MB
                      </p>
                    </div>
                    <CheckCircle className="w-5 h-5 text-green-600" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-between">
        <button
          onClick={onCancel}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors"
        >
          Cancel
        </button>
        
        <button
          onClick={handleContinue}
          disabled={analysis.needs_conversion && !conversionComplete}
          className={`px-6 py-2 rounded font-medium transition-colors ${
            analysis.needs_conversion && !conversionComplete
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-primary-600 text-white hover:bg-primary-700'
          }`}
        >
          <div className="flex items-center space-x-2">
            <Play className="w-4 h-4" />
            <span>Continue to Preprocessing</span>
          </div>
        </button>
      </div>
    </div>
  );
};

export default FileAnalysis;
