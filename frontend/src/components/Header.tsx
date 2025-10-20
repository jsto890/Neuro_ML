import React from 'react';
import { Brain, Wifi, WifiOff } from 'lucide-react';

interface HeaderProps {
  backendStatus?: 'connected' | 'disconnected' | 'checking';
  onLogoClick?: () => void;
}

const Header: React.FC<HeaderProps> = ({ backendStatus = 'checking', onLogoClick }) => {
  const getStatusColor = () => {
    switch (backendStatus) {
      case 'connected': return 'text-green-600';
      case 'disconnected': return 'text-red-600';
      case 'checking': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = () => {
    switch (backendStatus) {
      case 'connected': return <Wifi className="w-3 h-3" />;
      case 'disconnected': return <WifiOff className="w-3 h-3" />;
      case 'checking': return <div className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />;
      default: return <WifiOff className="w-3 h-3" />;
    }
  };

  const getStatusText = () => {
    switch (backendStatus) {
      case 'connected': return 'Backend Connected';
      case 'disconnected': return 'Backend Disconnected';
      case 'checking': return 'Checking...';
      default: return 'Unknown';
    }
  };

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div 
            className="flex items-center space-x-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={onLogoClick}
          >
            <div className="flex items-center justify-center w-8 h-8 bg-gradient-to-r from-primary-500 to-medical-500 rounded-md">
              <Brain className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">P4P Project</h1>
              <p className="text-xs text-gray-500">Joseph Storey & Jackson Schofield</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-1 text-xs">
            {getStatusIcon()}
            <span className={getStatusColor()}>{getStatusText()}</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
