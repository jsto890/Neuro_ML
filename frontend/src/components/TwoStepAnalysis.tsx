import React, { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle, Loader2, Brain, Zap, Clock, TrendingUp, Activity } from 'lucide-react';
import { ApiService } from '../services/api';

interface TwoStepAnalysisProps {
  uploadedFiles: Array<{
    id: string;
    fileName: string;
    fileType: string;
    size: number;
  }>;
  imageType: 'smri' | 'pet' | 'dat-spect';
  onComplete: (results: any) => void;
  onCancel: () => void;
}

interface PreprocessingResult {
  file_id: string;
  preprocessing_status: string;
  steps_completed: number;
  total_steps: number;
  preprocessing_steps: string[];
  output_path: string;
  quality_score: number;
  processing_time: number;
  image_type: string;
}

interface PredictionResult {
  file_id: string;
  prediction: string;
  confidence: number;
  risk_level: string;
  model_name: string;
  model_version: string;
  prediction_time: number;
  image_type: string;
  additional_metrics: {
    dopamine_transporter_binding?: number;
    striatal_uptake_ratio?: number;
    asymmetry_index?: number;
    hippocampal_volume?: number;
    cortical_thickness?: number;
    amyloid_burden?: number;
  };
}

const TwoStepAnalysis: React.FC<TwoStepAnalysisProps> = ({ 
  uploadedFiles, 
  imageType, 
  onComplete, 
  onCancel 
}) => {
  const [currentStep, setCurrentStep] = useState<'preprocessing' | 'prediction' | 'complete'>('preprocessing');
  const [preprocessingResults, setPreprocessingResults] = useState<PreprocessingResult[]>([]);
  const [predictionResults, setPredictionResults] = useState<PredictionResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentPreprocessingStep, setCurrentPreprocessingStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const getImageTypeDisplay = (type: string) => {
    switch (type) {
      case 'smri': return 'sMRI';
      case 'pet': return 'PET';
      case 'dat-spect': return 'DAT-SPECT';
      default: return 'Unknown';
    }
  };

  const getImageTypeIcon = (type: string) => {
    switch (type) {
      case 'smri': return <Brain className="w-5 h-5 text-purple-600" />;
      case 'pet': return <Activity className="w-5 h-5 text-green-600" />;
      case 'dat-spect': return <Zap className="w-5 h-5 text-blue-600" />;
      default: return <Brain className="w-5 h-5 text-gray-600" />;
    }
  };

  const runPreprocessing = async () => {
    setIsProcessing(true);
    setError(null);
    
    try {
      const fileIds = uploadedFiles?.map(file => file.id) || [];
      if (fileIds.length === 0) {
        setError('No files to preprocess');
        return;
      }
      
      const response = await ApiService.preprocessFiles(fileIds, imageType);
      
      if (response?.success) {
        setPreprocessingResults(response.preprocessing_results || []);
        
        // Simulate step-by-step progress
        const steps = response.preprocessing_results?.[0]?.preprocessing_steps || [];
        for (let i = 0; i <= steps.length; i++) {
          setCurrentPreprocessingStep(i);
          await new Promise(resolve => setTimeout(resolve, 300));
        }
        
        console.log('Setting currentStep to prediction');
        setCurrentStep('prediction');
      } else {
        setError(response?.error || 'Preprocessing failed');
      }
    } catch (error) {
      console.error('Preprocessing error:', error);
      setError('Preprocessing failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const runPrediction = async () => {
    console.log('runPrediction called');
    setIsProcessing(true);
    setError(null);
    
    try {
      const fileIds = uploadedFiles?.map(file => file.id) || [];
      console.log('File IDs for prediction:', fileIds);
      console.log('Image type:', imageType);
      
      if (fileIds.length === 0) {
        setError('No files to predict');
        return;
      }
      
      console.log('Calling API service...');
      const response = await ApiService.predictParkinson(fileIds, imageType);
      console.log('Prediction response:', response);
      
      if (response?.success) {
        setPredictionResults(response.predictions || []);
        setCurrentStep('complete');
        
        // Call onComplete with all results
        onComplete({
          preprocessing: preprocessingResults,
          predictions: response.predictions || [],
          imageType: imageType
        });
      } else {
        setError(response?.error || 'Prediction failed');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      setError('Prediction failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      case 'moderate': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.8) return 'text-yellow-600';
    return 'text-red-600';
  };

  useEffect(() => {
    console.log('TwoStepAnalysis useEffect - currentStep:', currentStep);
    if (currentStep === 'preprocessing') {
      runPreprocessing();
    }
  }, [currentStep]);

  console.log('TwoStepAnalysis render - currentStep:', currentStep, 'isProcessing:', isProcessing);
  
  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">
          {getImageTypeDisplay(imageType)} Analysis
        </h2>
        <p className="text-gray-600">
          Processing {uploadedFiles.length} file{uploadedFiles.length !== 1 ? 's' : ''} with Simple3DCNN
        </p>
      </div>

      {/* Progress Steps */}
      <div className="flex items-center justify-center space-x-8">
        <div className={`flex items-center space-x-2 ${currentStep === 'preprocessing' ? 'text-primary-600' : currentStep === 'prediction' || currentStep === 'complete' ? 'text-green-600' : 'text-gray-400'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            currentStep === 'preprocessing' ? 'bg-primary-600 text-white' : 
            currentStep === 'prediction' || currentStep === 'complete' ? 'bg-green-500 text-white' : 
            'bg-gray-200 text-gray-500'
          }`}>
            {currentStep === 'preprocessing' ? <Loader2 className="w-4 h-4 animate-spin" /> : 
             currentStep === 'prediction' || currentStep === 'complete' ? <CheckCircle className="w-4 h-4" /> : 
             '1'}
          </div>
          <span className="font-medium">Preprocessing</span>
        </div>
        
        <div className={`w-12 h-0.5 ${currentStep === 'prediction' || currentStep === 'complete' ? 'bg-green-500' : 'bg-gray-200'}`} />
        
        <div className={`flex items-center space-x-2 ${currentStep === 'prediction' ? 'text-primary-600' : currentStep === 'complete' ? 'text-green-600' : 'text-gray-400'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            currentStep === 'prediction' ? 'bg-primary-600 text-white' : 
            currentStep === 'complete' ? 'bg-green-500 text-white' : 
            'bg-gray-200 text-gray-500'
          }`}>
            {currentStep === 'prediction' ? <Loader2 className="w-4 h-4 animate-spin" /> : 
             currentStep === 'complete' ? <CheckCircle className="w-4 h-4" /> : 
             '2'}
          </div>
          <span className="font-medium">Model Prediction</span>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertCircle className="w-5 h-5 text-red-600 mr-3" />
            <div>
              <h3 className="text-sm font-medium text-red-800">Processing Error</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Preprocessing Step */}
      {currentStep === 'preprocessing' && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Preprocessing Files</h3>
            <div className="flex items-center space-x-2">
              {getImageTypeIcon(imageType)}
              <span className="text-sm text-gray-600">{getImageTypeDisplay(imageType)}</span>
            </div>
          </div>

          <div className="space-y-4">
            {preprocessingResults.length > 0 && preprocessingResults[0].preprocessing_steps.map((step, index) => (
              <div key={index} className="flex items-center space-x-3">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                  index < currentPreprocessingStep ? 'bg-green-500 text-white' : 
                  index === currentPreprocessingStep ? 'bg-primary-500 text-white' : 
                  'bg-gray-200 text-gray-500'
                }`}>
                  {index < currentPreprocessingStep ? <CheckCircle className="w-4 h-4" /> : 
                   index === currentPreprocessingStep ? <Loader2 className="w-4 h-4 animate-spin" /> : 
                   index + 1}
                </div>
                <span className={`text-sm ${
                  index < currentPreprocessingStep ? 'text-green-700' : 
                  index === currentPreprocessingStep ? 'text-primary-700' : 
                  'text-gray-500'
                }`}>
                  {step}
                </span>
              </div>
            ))}

            {preprocessingResults.length > 0 && (
              <div className="mt-6 p-4 bg-green-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-sm font-medium text-green-800">Preprocessing Complete</h4>
                    <p className="text-sm text-green-700">
                      Quality Score: {(preprocessingResults[0].quality_score * 100).toFixed(1)}%
                    </p>
                  </div>
                  <CheckCircle className="w-6 h-6 text-green-600" />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Prediction Step */}
      {currentStep === 'prediction' && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="text-center mb-6">
            <h3 className="text-lg font-medium text-gray-900 mb-2">Running Model Prediction</h3>
            <p className="text-sm text-gray-600">Simple3DCNN analyzing preprocessed data...</p>
          </div>

          <div className="flex justify-center">
            <button
              onClick={() => {
                console.log('Button clicked, isProcessing:', isProcessing, 'currentStep:', currentStep);
                if (!isProcessing) {
                  runPrediction();
                }
              }}
              disabled={isProcessing}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                isProcessing
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-primary-600 text-white hover:bg-primary-700'
              }`}
            >
              {isProcessing ? (
                <div className="flex items-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Running Prediction...</span>
                </div>
              ) : (
                'Start Model Prediction'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Results Step */}
      {currentStep === 'complete' && predictionResults.length > 0 && (
        <div className="space-y-4">
          {predictionResults.map((result, index) => (
            <div key={result.file_id} className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-medium text-gray-900">
                    {uploadedFiles.find(f => f.id === result.file_id)?.fileName}
                  </h3>
                  <div className="flex items-center space-x-2 mt-1">
                    {getImageTypeIcon(imageType)}
                    <span className="text-sm text-gray-600">{getImageTypeDisplay(imageType)}</span>
                    <span className="text-sm text-gray-500">•</span>
                    <span className="text-sm text-gray-500">{result.model_name} v{result.model_version}</span>
                  </div>
                </div>
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskLevelColor(result.risk_level)}`}>
                  {result.risk_level} Risk
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="text-sm font-medium text-gray-700">Prediction</label>
                  <div className="mt-1 p-3 bg-gray-50 rounded-lg">
                    <p className="text-lg font-semibold text-gray-900">{result.prediction}</p>
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700">Confidence</label>
                  <div className="mt-1 flex items-center space-x-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    <span className={`text-sm font-medium ${getConfidenceColor(result.confidence)}`}>
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-3">GRADCAM Results</h4>
                <p className="text-xs text-gray-600 mb-3">
                  Heat map showing where the model focused its attention during prediction
                </p>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="flex items-center justify-center">
                    <img 
                      src={`${window.location.origin}/grad-cam-test.png`}
                      alt="GRADCAM Heat Map" 
                      className="max-w-full h-auto rounded border border-gray-200 shadow-sm"
                      style={{ maxHeight: '200px' }}
                      onError={(e) => {
                        console.error('Failed to load GRADCAM image:', e);
                        console.error('Image src:', e.currentTarget.src);
                        e.currentTarget.style.display = 'none';
                      }}
                      onLoad={() => {
                        console.log('GRADCAM image loaded successfully');
                      }}
                    />
                  </div>
                  <div className="mt-2 text-center">
                    <div className="inline-flex items-center space-x-3 text-xs text-gray-600">
                      <div className="flex items-center space-x-1">
                        <div className="w-2 h-2 bg-red-500 rounded"></div>
                        <span>High</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <div className="w-2 h-2 bg-yellow-500 rounded"></div>
                        <span>Medium</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <div className="w-2 h-2 bg-blue-500 rounded"></div>
                        <span>Low</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-between">
        <button
          onClick={onCancel}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
        >
          Cancel
        </button>
        
        {currentStep === 'complete' && (
          <button
            onClick={() => onComplete({ preprocessing: preprocessingResults, predictions: predictionResults, imageType })}
            className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            View Results
          </button>
        )}
      </div>
    </div>
  );
};

export default TwoStepAnalysis;
