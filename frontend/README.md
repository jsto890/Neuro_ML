# P4P Project Frontend

A modern React application for medical image analysis, supporting both NIFTI and DICOM image processing with AI-powered classification.

**GitHub Repository:** [https://github.com/Jackson-Schofield/P4P](https://github.com/Jackson-Schofield/P4P)

## Features

- **Modern UI/UX**: Clean, responsive design built with React and Tailwind CSS
- **File Upload**: Drag-and-drop interface for NIFTI (.nii, .nii.gz) and DICOM (.dcm, .dicom) files
- **Real-time Analysis**: Simulated AI analysis with confidence scoring
- **Multiple Image Types**: Support for SPECT, MRI, PET, DICOM, and NIFTI images
- **Results Display**: Comprehensive analysis results with visual indicators
- **TypeScript**: Full type safety and better development experience

## Tech Stack

- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **Headless UI** for accessible components

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Available Scripts

- `npm start` - Runs the app in development mode
- `npm run build` - Builds the app for production
- `npm test` - Launches the test runner
- `npm run eject` - Ejects from Create React App (one-way operation)

## Project Structure

```
src/
├── components/
│   ├── Header.tsx          # Navigation header
│   ├── FileUpload.tsx      # File upload component
│   ├── AnalysisResults.tsx # Results display
│   └── Footer.tsx          # Footer component
├── App.tsx                 # Main application component
├── index.css              # Global styles with Tailwind
└── index.tsx              # Application entry point
```

## Features Overview

### File Upload
- Drag-and-drop interface
- Support for multiple file selection
- File type validation (.nii, .nii.gz, .dcm, .dicom)
- File size display
- Image type detection (SPECT/MRI/PET/DICOM/NIFTI)

### Analysis Results
- Prediction display with color coding
- Confidence percentage with progress bar
- Timestamp and processing time
- File type indicators
- Clear all results functionality

### Design System
- Custom color palette (primary, medical)
- Inter font family
- Responsive grid layouts
- Hover states and transitions
- Accessibility considerations

## Future Enhancements

- [ ] Real API integration
- [ ] Image preview functionality
- [ ] Batch processing
- [ ] Export results
- [ ] User authentication
- [ ] Advanced filtering and sorting
- [ ] 3D image visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Authors

- **Joseph Storey** - Research and Development
- **Jackson Schofield** - Research and Development

## License

This project is part of the P4P medical imaging research project.

## Repository

**GitHub:** [https://github.com/Jackson-Schofield/P4P](https://github.com/Jackson-Schofield/P4P)