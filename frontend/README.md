# Frontend

React and TypeScript interface for image upload and classification result display.

## Purpose

The frontend provides a demo lightweight workflow to:

- upload MRI, PET, SPECT in both NIfTI, and DICOM file form
- submit files to the backend API
- present prediction and confidence outputs with individualised GradCAM attention visualisation

## Stack

- React 18
- TypeScript
- Tailwind CSS

## Local development

```bash
cd frontend
npm install
npm start
```

Default development URL: `http://localhost:3000`.

## Related components

- API integration: `src/services/api.ts`
- Upload UI: `src/components/FileUpload.tsx`
- Result views: `src/components/AnalysisResults.tsx`, `src/components/TwoStepAnalysis.tsx`

## Notes

This module is maintained as part of the main `Neuro_ML` repository.