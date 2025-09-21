import React from 'react';
import { CheckCircle, XCircle, Clock, FileImage, TrendingUp } from 'lucide-react';

interface AnalysisResult {
  id: string;
  fileName: string;
  prediction: string;
  confidence: number;
  timestamp: Date;
  imageType: 'SPECT' | 'MRI' | 'PET' | 'DICOM' | 'NIFTI';
}

interface AnalysisResultsProps {
  results: AnalysisResult[];
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ results }) => {
  const getPredictionColor = (prediction: string) => {
    if (prediction.includes('CN') || prediction.includes('Control')) {
      return 'text-green-600 bg-green-50 border-green-200';
    } else if (prediction.includes('PD') || prediction.includes('Parkinson')) {
      return 'text-orange-600 bg-orange-50 border-orange-200';
    }
    return 'text-gray-600 bg-gray-50 border-gray-200';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600';
    if (confidence >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getImageTypeColor = (type: string) => {
    switch (type) {
      case 'SPECT': return 'bg-blue-100 text-blue-800';
      case 'MRI': return 'bg-purple-100 text-purple-800';
      case 'PET': return 'bg-green-100 text-green-800';
      case 'DICOM': return 'bg-orange-100 text-orange-800';
      case 'NIFTI': return 'bg-indigo-100 text-indigo-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(timestamp);
  };

  return (
    <div className="space-y-4">
      {results.map((result) => (
        <div
          key={result.id}
          className="border border-gray-200 rounded p-4 hover:shadow-sm transition-shadow"
        >
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center space-x-2">
              <FileImage className="w-4 h-4 text-gray-400" />
              <div>
                <h3 className="text-sm font-medium text-gray-900">
                  {result.fileName}
                </h3>
                <div className="flex items-center space-x-2 mt-1">
                  <span className={`px-2 py-1 text-xs font-medium rounded ${getImageTypeColor(result.imageType)}`}>
                    {result.imageType}
                  </span>
                  <span className="text-xs text-gray-500">
                    {formatTimestamp(result.timestamp)}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              {result.prediction.includes('CN') || result.prediction.includes('Control') ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <XCircle className="w-4 h-4 text-orange-500" />
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Prediction */}
            <div className="space-y-1">
              <label className="text-xs font-medium text-gray-700">Prediction</label>
              <div className={`px-2 py-1 rounded border ${getPredictionColor(result.prediction)}`}>
                <span className="text-sm font-medium">{result.prediction}</span>
              </div>
            </div>

            {/* Confidence */}
            <div className="space-y-1">
              <label className="text-xs font-medium text-gray-700">Confidence</label>
              <div className="flex items-center space-x-2">
                <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                  <div
                    className="bg-primary-500 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${result.confidence}%` }}
                  />
                </div>
                <span className={`text-xs font-medium ${getConfidenceColor(result.confidence)}`}>
                  {result.confidence}%
                </span>
              </div>
            </div>
          </div>

          {/* Additional Info */}
          <div className="mt-3 pt-2 border-t border-gray-200">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1">
                  <TrendingUp className="w-3 h-3" />
                  <span>AI Analysis</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>~2s</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}

      {results.length === 0 && (
        <div className="text-center py-8">
          <FileImage className="w-8 h-8 text-gray-300 mx-auto mb-3" />
          <p className="text-sm text-gray-500">No analysis results yet</p>
          <p className="text-xs text-gray-400">Upload some NIFTI files to get started</p>
        </div>
      )}
    </div>
  );
};

export default AnalysisResults;
