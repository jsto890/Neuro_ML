import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import UploadTypeSelector from './components/UploadTypeSelector';
import ImageTypeSelector from './components/ImageTypeSelector';
import FileUpload from './components/FileUpload';
import FileAnalysis from './components/FileAnalysis';
import TwoStepAnalysis from './components/TwoStepAnalysis';
import AnalysisResults from './components/AnalysisResults';
import Footer from './components/Footer';
import ErrorBoundary from './components/ErrorBoundary';
import { ApiService } from './services/api';

interface AnalysisResult {
  id: string;
  fileName: string;
  prediction: string;
  confidence: number;
  timestamp: Date;
  imageType: 'sMRI' | 'PET' | 'DAT-SPECT';
  uploadType: 'dicom' | 'nifti';
  riskLevel?: string;
  modelName?: string;
  additionalMetrics?: {
    dopamine_transporter_binding?: number;
    striatal_uptake_ratio?: number;
    asymmetry_index?: number;
    hippocampal_volume?: number;
    cortical_thickness?: number;
    amyloid_burden?: number;
  };
}

type UploadStep = 'type-select' | 'image-type-select' | 'file-upload' | 'analysis' | 'results';

function App() {
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [currentStep, setCurrentStep] = useState<UploadStep>('type-select');
  const [selectedUploadType, setSelectedUploadType] = useState<'dicom' | 'nifti' | null>(null);
  const [selectedImageType, setSelectedImageType] = useState<'smri' | 'pet' | 'dat-spect' | null>(null);
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

  // Step navigation functions
  const handleUploadTypeSelect = (type: 'dicom' | 'nifti') => {
    setSelectedUploadType(type);
    setCurrentStep('image-type-select');
  };

  const handleImageTypeSelect = (type: 'smri' | 'pet' | 'dat-spect') => {
    setSelectedImageType(type);
    setCurrentStep('file-upload');
  };

  const handleBackToImageType = () => {
    setCurrentStep('image-type-select');
  };

  const handleBackToUploadType = () => {
    setCurrentStep('type-select');
    setSelectedImageType(null);
  };

  const handleBackToTypeSelect = () => {
    setCurrentStep('type-select');
    setSelectedUploadType(null);
    setSelectedImageType(null);
  };

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
        setCurrentStep('analysis');
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
    setCurrentStep('analysis');
  };

  const handleTwoStepAnalysisComplete = (results: any) => {
    try {
      // Create analysis results from prediction results
      const newResults: AnalysisResult[] = (results?.predictions || []).map((prediction: any) => {
        const file = uploadedFiles.find(f => f.id === prediction.file_id);
        
        return {
          id: prediction.file_id || 'unknown',
          fileName: file?.fileName || 'Unknown',
          prediction: prediction.prediction || 'Unknown',
          confidence: Math.round((prediction.confidence || 0) * 100),
          timestamp: new Date(),
          imageType: getImageTypeDisplay(selectedImageType!) as 'sMRI' | 'PET' | 'DAT-SPECT',
          uploadType: selectedUploadType!,
          riskLevel: prediction.risk_level,
          modelName: prediction.model_name,
          additionalMetrics: prediction.additional_metrics
        };
      });
      
      setAnalysisResults(prev => [...newResults, ...prev]);
      setUploadedFiles([]);
      setCurrentStep('results');
    } catch (error) {
      console.error('Error in handleTwoStepAnalysisComplete:', error);
    }
  };

  const getImageTypeDisplay = (type: string) => {
    switch (type) {
      case 'smri': return 'sMRI';
      case 'pet': return 'PET';
      case 'dat-spect': return 'DAT-SPECT';
      default: return 'Unknown';
    }
  };

  const handleAnalysisCancel = () => {
    setShowFileAnalysis(false);
    setUploadedFiles([]);
    setCurrentStep('file-upload');
  };

  const clearResults = () => {
    setAnalysisResults([]);
    setCurrentStep('type-select');
    setSelectedUploadType(null);
    setSelectedImageType(null);
  };

  const startNewUpload = () => {
    setCurrentStep('type-select');
    setSelectedUploadType(null);
    setSelectedImageType(null);
    setUploadedFiles([]);
  };

  const handleLogoClick = () => {
    setCurrentStep('type-select');
    setSelectedUploadType(null);
    setSelectedImageType(null);
    setUploadedFiles([]);
    setAnalysisResults([]);
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'type-select':
        return (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <UploadTypeSelector onSelectType={handleUploadTypeSelect} />
          </div>
        );

      case 'image-type-select':
        return (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <ImageTypeSelector 
              onSelectType={handleImageTypeSelect}
              onBack={handleBackToUploadType}
            />
          </div>
        );

      case 'file-upload':
        return (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <FileUpload 
              onFileUpload={handleFileUpload}
              isAnalyzing={isAnalyzing}
              uploadType={selectedUploadType!}
              imageType={selectedImageType!}
              onBack={handleBackToImageType}
            />
          </div>
        );

      case 'analysis':
        return (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <TwoStepAnalysis 
              uploadedFiles={uploadedFiles}
              imageType={selectedImageType!}
              onComplete={handleTwoStepAnalysisComplete}
              onCancel={handleAnalysisCancel}
            />
          </div>
        );

      case 'results':
        return (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-8">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-semibold text-gray-900">
                  Analysis Results
                </h2>
                <div className="flex space-x-3">
                  <button
                    onClick={startNewUpload}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                  >
                    Upload More
                  </button>
                  <button
                    onClick={clearResults}
                    className="px-4 py-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors"
                  >
                    Clear All
                  </button>
                </div>
              </div>
              <AnalysisResults results={analysisResults} />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        <Header backendStatus={backendStatus} onLogoClick={handleLogoClick} />
        
        <main className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            {/* Hero Section */}
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Medical Image Analysis
              </h1>
              <p className="text-gray-600">
                Upload and analyze sMRI, PET, or DAT-SPECT images
              </p>
            </div>

            {/* Progress Indicator */}
            {currentStep !== 'results' && (
              <div className="mb-8">
                <div className="flex items-center justify-center space-x-4">
                  {[
                    { step: 'type-select', label: 'Upload Type', icon: '📁' },
                    { step: 'image-type-select', label: 'Image Type', icon: '🧠' },
                    { step: 'file-upload', label: 'Upload Files', icon: '⬆️' },
                    { step: 'analysis', label: 'Analysis', icon: '⚡' }
                  ].map((item, index) => {
                    const isActive = currentStep === item.step;
                    const isCompleted = [
                      'type-select',
                      'image-type-select', 
                      'file-upload',
                      'analysis'
                    ].indexOf(currentStep) > index;
                    
                    return (
                      <div key={item.step} className="flex items-center">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                          isActive 
                            ? 'bg-primary-600 text-white' 
                            : isCompleted 
                              ? 'bg-green-500 text-white' 
                              : 'bg-gray-200 text-gray-500'
                        }`}>
                          {isCompleted ? '✓' : item.icon}
                        </div>
                        <span className={`ml-2 text-sm font-medium ${
                          isActive ? 'text-primary-600' : 'text-gray-500'
                        }`}>
                          {item.label}
                        </span>
                        {index < 3 && (
                          <div className={`w-8 h-0.5 mx-4 ${
                            isCompleted ? 'bg-green-500' : 'bg-gray-200'
                          }`} />
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {renderCurrentStep()}
          </div>
        </main>

        <Footer />
      </div>
    </ErrorBoundary>
  );
}

export default App;