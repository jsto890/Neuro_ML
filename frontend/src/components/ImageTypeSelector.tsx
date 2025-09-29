import React from 'react';
import { Brain, Activity, Zap, ArrowRight } from 'lucide-react';

interface ImageTypeSelectorProps {
  onSelectType: (type: 'smri' | 'pet' | 'dat-spect') => void;
  onBack: () => void;
}

const ImageTypeSelector: React.FC<ImageTypeSelectorProps> = ({ onSelectType, onBack }) => {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">
          Select Image Type
        </h2>
        <p className="text-gray-600">
          Choose the type of medical image you're uploading
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* sMRI Option */}
        <div
          onClick={() => onSelectType('smri')}
          className="group cursor-pointer bg-white rounded-xl border-2 border-gray-200 hover:border-purple-500 hover:shadow-lg transition-all duration-200 p-6"
        >
          <div className="text-center">
            <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-purple-200 transition-colors">
              <Brain className="w-8 h-8 text-purple-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              sMRI
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Structural Magnetic Resonance Imaging
            </p>
            <div className="flex items-center justify-center text-purple-600 group-hover:text-purple-700">
              <span className="text-sm font-medium">Select sMRI</span>
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>

        {/* PET Option */}
        <div
          onClick={() => onSelectType('pet')}
          className="group cursor-pointer bg-white rounded-xl border-2 border-gray-200 hover:border-green-500 hover:shadow-lg transition-all duration-200 p-6"
        >
          <div className="text-center">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-green-200 transition-colors">
              <Activity className="w-8 h-8 text-green-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              PET
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Positron Emission Tomography
            </p>
            <div className="flex items-center justify-center text-green-600 group-hover:text-green-700">
              <span className="text-sm font-medium">Select PET</span>
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>

        {/* DAT-SPECT Option */}
        <div
          onClick={() => onSelectType('dat-spect')}
          className="group cursor-pointer bg-white rounded-xl border-2 border-gray-200 hover:border-blue-500 hover:shadow-lg transition-all duration-200 p-6"
        >
          <div className="text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-blue-200 transition-colors">
              <Zap className="w-8 h-8 text-blue-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              DAT-SPECT
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Dopamine Transporter SPECT
            </p>
            <div className="flex items-center justify-center text-blue-600 group-hover:text-blue-700">
              <span className="text-sm font-medium">Select DAT-SPECT</span>
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-center">
        <button
          onClick={onBack}
          className="px-6 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
        >
          ← Back to Upload Type
        </button>
      </div>
    </div>
  );
};

export default ImageTypeSelector;
