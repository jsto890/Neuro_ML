import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import FileAnalysis from './components/FileAnalysis';
import AnalysisResults from './components/AnalysisResults';
import Footer from './components/Footer';
import { ApiService } from './services/api';

interface AnalysisResult {
  id: string;
  fileName: string;
  prediction: string;
  confidence: number;
  timestamp: Date;
  imageType: 'SPECT' | 'MRI' | 'PET' | 'DICOM' | 'NIFTI';
}

function App() {
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [showFileAnalysis, setShowFileAnalysis] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<Array<{
    id: string;
    fileName: string;
    fileType: string;
    size: number;
  }>>([]);

  // Check backend health on component mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        await ApiService.checkHealth();
        setBackendStatus('connected');
      } catch (error) {
        console.error('Backend connection failed:', error);
        setBackendStatus('disconnected');
      }
    };
    
    checkBackend();
  }, []);

  const handleFileUpload = async (files: FileList) => {
    setIsAnalyzing(true);
    
    try {
      // Upload files to backend
      const uploadResponse = await ApiService.uploadFiles(files);
      
      if (uploadResponse.success) {
        // Store uploaded files for analysis
        const filesData = uploadResponse.files.map((file) => ({
          id: file.id,
          fileName: file.original_name,
          fileType: file.file_type,
          size: file.size_bytes // Use size_bytes from API response
        }));
        
        setUploadedFiles(filesData);
        setShowFileAnalysis(true);
      } else {
        console.error('Upload failed:', uploadResponse.message);
      }
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalysisContinue = (conversionResults: any) => {
    // Create analysis results from uploaded files
    const newResults: AnalysisResult[] = uploadedFiles.map((file) => {
      let imageType: 'SPECT' | 'MRI' | 'PET' | 'DICOM' | 'NIFTI' = 'NIFTI';
      
      if (file.fileName.toLowerCase().includes('spect')) imageType = 'SPECT';
      else if (file.fileName.toLowerCase().includes('mri')) imageType = 'MRI';
      else if (file.fileName.toLowerCase().includes('pet')) imageType = 'PET';
      else if (file.fileType === 'DICOM') imageType = 'DICOM';
      else if (file.fileType === 'NIFTI') imageType = 'NIFTI';
      
      return {
        id: file.id,
        fileName: file.fileName,
        prediction: conversionResults.hasConversion ? 'Ready for preprocessing' : 'Ready for analysis',
        confidence: 100,
        timestamp: new Date(),
        imageType: imageType
      };
    });
    
    setAnalysisResults(prev => [...newResults, ...prev]);
    setShowFileAnalysis(false);
    setUploadedFiles([]);
  };

  const handleAnalysisCancel = () => {
    setShowFileAnalysis(false);
    setUploadedFiles([]);
  };

  const clearResults = () => {
    setAnalysisResults([]);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header backendStatus={backendStatus} />
      
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-8">
            <h1 className="text-2xl font-semibold text-gray-900 mb-2">
              Medical Image Analysis
            </h1>
            <p className="text-gray-600">
              Upload NIFTI or DICOM images for analysis
            </p>
          </div>

          {showFileAnalysis ? (
            <div className="bg-white rounded-lg shadow p-6">
              <FileAnalysis 
                uploadedFiles={uploadedFiles}
                onContinue={handleAnalysisContinue}
                onCancel={handleAnalysisCancel}
              />
            </div>
          ) : (
            <>
              {/* Upload Section */}
              <div className="bg-white rounded-lg shadow p-6 mb-6">
                <FileUpload 
                  onFileUpload={handleFileUpload}
                  isAnalyzing={isAnalyzing}
                />
              </div>

              {/* Results Section */}
              {analysisResults.length > 0 && (
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-lg font-medium text-gray-900">
                      Results
                    </h2>
                    <button
                      onClick={clearResults}
                      className="px-3 py-1 text-sm text-red-600 hover:text-red-700 hover:bg-red-50 rounded transition-colors"
                    >
                      Clear All
                    </button>
                  </div>
                  <AnalysisResults results={analysisResults} />
                </div>
              )}
            </>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;