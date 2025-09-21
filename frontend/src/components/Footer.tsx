import React from 'react';
import { Brain } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-white border-t border-gray-200 mt-8">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-gradient-to-r from-primary-500 to-medical-500 rounded flex items-center justify-center">
              <Brain className="w-3 h-3 text-white" />
            </div>
            <span className="text-sm font-medium text-gray-900">P4P Project</span>
          </div>
          <p className="text-xs text-gray-500">
            Joseph Storey & Jackson Schofield
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
