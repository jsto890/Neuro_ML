#!/usr/bin/env python3
"""
Radiomics Feature Extractor for P4P Project
===========================================

Extracts radiomics features from preprocessed neuroimaging data.
Supports MRI, PET, and SPECT modalities.

Usage:
    python radiomics_extractor.py --modality MRI --labels Labels/train_labels.csv
    python radiomics_extractor.py --modality PET --labels Labels/val_labels.csv
    python radiomics_extractor.py --modality SPECT --labels Labels/train_labels.csv
"""

import os
import yaml
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RadiomicsExtractor:
    def __init__(self, config_path='config.yaml'):
        """Initialize the radiomics extractor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize radiomics extractor with default settings
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Define modality-specific paths and file patterns
        self.modality_configs = {
            'MRI': {
                'data_path': self.config['preprocessed_data']['smri_p'],
                'file_pattern': '{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz',
                'subdir': 'smriprep/{subject_id}/anat'
            },
            'PET': {
                'data_path': self.config['preprocessed_data']['pet_p'],
                'file_pattern': 'pet_mni_crop.nii.gz',  # Final preprocessed PET
                'subdir': '{site}/{dx}/{subject_id}'
            },
            'SPECT': {
                'data_path': self.config['preprocessed_data']['spect_p'],
                'file_pattern': 'dspect_preproc.nii.gz',  # Preprocessed SPECT
                'subdir': '{subject_id}'
            }
        }
    
    def create_mask_from_image(self, image_path):
        """Create a mask from non-zero regions of the image"""
        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.NotEqual(image, 0)  # Non-zero region = ROI
            return image, mask
        except Exception as e:
            logger.error(f"Error creating mask from {image_path}: {e}")
            return None, None
    
    def find_image_path(self, subject_id, modality):
        """Find the image path for a given subject and modality"""
        config = self.modality_configs[modality]
        data_path = config['data_path']
        file_pattern = config['file_pattern']
        subdir = config['subdir']
        
        # Handle different subdirectory patterns
        if '{site}' in subdir and '{dx}' in subdir:
            # For PET data, we need to find the site and diagnosis
            # This is a simplified approach - you might need to adjust based on your actual structure
            for site_dir in Path(data_path).iterdir():
                if site_dir.is_dir():
                    for dx_dir in site_dir.iterdir():
                        if dx_dir.is_dir():
                            subject_dir = dx_dir / subject_id
                            if subject_dir.exists():
                                image_path = subject_dir / file_pattern
                                if image_path.exists():
                                    return str(image_path)
        else:
            # For MRI and SPECT data
            subdir_path = Path(data_path) / subdir.format(subject_id=subject_id)
            image_path = subdir_path / file_pattern.format(subject_id=subject_id)
            if image_path.exists():
                return str(image_path)
        
        return None
    
    def extract_features(self, labels_path, modality, output_dir='.'):
        """Extract radiomics features from all subjects in the dataset"""
        
        # Load labels
        df = pd.read_csv(labels_path)
        if 'subject_id' not in df.columns or 'label' not in df.columns:
            df = pd.read_csv(labels_path, header=None, names=['subject_id', 'label'])
        
        # Clean the dataframe
        df = df[~df['subject_id'].isin(['subject_id', ''])]
        df = df[~df['label'].isin(['label', ''])]
        df['label'] = df['label'].astype(int)
        
        logger.info(f"📊 Found {len(df)} subjects in {labels_path}")
        logger.info(f"🔬 Processing {modality} data")
        
        # Store results
        all_features = []
        successful = 0
        failed = 0
        
        # Loop through all subjects
        for idx, row in df.iterrows():
            subject_id = row['subject_id']
            label = row['label']
            
            # Find the image path
            image_path = self.find_image_path(subject_id, modality)
            
            if not image_path:
                logger.warning(f"❌ Image not found for {subject_id}")
                failed += 1
                continue
                
            logger.info(f"🔍 Processing: {subject_id} (label: {label})")
            
            try:
                image, mask = self.create_mask_from_image(image_path)
                if image is None or mask is None:
                    logger.error(f"❌ Failed to create mask for {subject_id}")
                    failed += 1
                    continue
                
                result = self.extractor.execute(image, mask)
                
                # Add subject info to results
                result['subject_id'] = subject_id
                result['label'] = label
                result['modality'] = modality
                
                all_features.append(result)
                successful += 1
                
                feature_count = len([k for k in result.keys() if k not in ['subject_id', 'label', 'modality']])
                logger.info(f"✅ Extracted {feature_count} features from {subject_id}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {subject_id}: {e}")
                failed += 1
        
        # Save results
        if all_features:
            results_df = pd.DataFrame(all_features)
            output_filename = f"radiomics_{modality}_{Path(labels_path).stem}.csv"
            output_path = Path(output_dir) / output_filename
            results_df.to_csv(output_path, index=False)
            
            logger.info(f"\n💾 Saved {len(all_features)} feature sets to {output_path}")
            logger.info(f"📈 Feature matrix shape: {results_df.shape}")
            logger.info(f"✅ Successful: {successful}, ❌ Failed: {failed}")
            
            return output_path
        else:
            logger.error("❌ No features were successfully extracted")
            return None

def main():
    parser = argparse.ArgumentParser(description='Extract radiomics features from neuroimaging data')
    parser.add_argument('--modality', choices=['MRI', 'PET', 'SPECT'], required=True,
                       help='Imaging modality to process')
    parser.add_argument('--labels', required=True,
                       help='Path to CSV file with subject_id and label columns')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor and run
    extractor = RadiomicsExtractor(args.config)
    output_path = extractor.extract_features(args.labels, args.modality, args.output_dir)
    
    if output_path:
        logger.info(f"🎉 Feature extraction completed successfully!")
        logger.info(f"📁 Results saved to: {output_path}")
    else:
        logger.error("💥 Feature extraction failed!")

if __name__ == "__main__":
    main() 