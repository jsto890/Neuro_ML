#!/usr/bin/env python3
"""
Simple Radiomics Extractor for MRI Data
=======================================

A simplified version that avoids pandas/NumPy compatibility issues.
"""

import os
import yaml
import csv
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_labels_simple(labels_path):
    """Load labels using csv module instead of pandas"""
    subjects = []
    labels = []
    
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row['subject_id'].strip()
            label = int(row['label'].strip())
            subjects.append(subject_id)
            labels.append(label)
    
    return subjects, labels

def create_mask_from_image(image_path):
    """Create a mask from non-zero regions of the image"""
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.NotEqual(image, 0)  # Non-zero region = ROI
        return image, mask
    except Exception as e:
        logger.error(f"Error creating mask from {image_path}: {e}")
        return None, None

def find_mri_path(data_root, subject_id):
    """Find the MRI image path for a given subject"""
    image_path = os.path.join(
        data_root,
        "smriprep",
        subject_id,
        "anat",
        f"{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz"
    )
    
    if os.path.exists(image_path):
        return image_path
    return None

def extract_mri_radiomics(config_path, labels_path, output_dir):
    """Extract radiomics features from MRI data"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_root = config['preprocessed_data']['smri_p']
    
    # Load labels
    subjects, labels = load_labels_simple(labels_path)
    logger.info(f"📊 Found {len(subjects)} subjects in {labels_path}")
    
    # Initialize radiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    
    # Store results
    all_features = []
    successful = 0
    failed = 0
    
    # Process each subject
    for i, (subject_id, label) in enumerate(zip(subjects, labels)):
        logger.info(f"🔍 Processing {i+1}/{len(subjects)}: {subject_id} (label: {label})")
        
        # Find image path
        image_path = find_mri_path(data_root, subject_id)
        
        if not image_path:
            logger.warning(f"❌ Image not found for {subject_id}")
            failed += 1
            continue
        
        try:
            # Create mask and extract features
            image, mask = create_mask_from_image(image_path)
            if image is None or mask is None:
                logger.error(f"❌ Failed to create mask for {subject_id}")
                failed += 1
                continue
            
            result = extractor.execute(image, mask)
            
            # Add subject info
            result['subject_id'] = subject_id
            result['label'] = label
            
            all_features.append(result)
            successful += 1
            
            feature_count = len([k for k in result.keys() if k not in ['subject_id', 'label']])
            logger.info(f"✅ Extracted {feature_count} features from {subject_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {subject_id}: {e}")
            failed += 1
    
    # Save results
    if all_features:
        output_filename = f"radiomics_MRI_{Path(labels_path).stem}.csv"
        output_path = Path(output_dir) / output_filename
        
        # Write CSV manually to avoid pandas issues
        with open(output_path, 'w', newline='') as f:
            if all_features:
                fieldnames = all_features[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_features)
        
        logger.info(f"\n💾 Saved {len(all_features)} feature sets to {output_path}")
        logger.info(f"✅ Successful: {successful}, ❌ Failed: {failed}")
        
        return output_path
    else:
        logger.error("❌ No features were successfully extracted")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract radiomics features from MRI data')
    parser.add_argument('--labels', required=True, help='Path to CSV file with subject_id and label columns')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run extraction
    output_path = extract_mri_radiomics(args.config, args.labels, args.output_dir)
    
    if output_path:
        logger.info(f"🎉 Feature extraction completed successfully!")
        logger.info(f"📁 Results saved to: {output_path}")
    else:
        logger.error("💥 Feature extraction failed!")

if __name__ == "__main__":
    main() 