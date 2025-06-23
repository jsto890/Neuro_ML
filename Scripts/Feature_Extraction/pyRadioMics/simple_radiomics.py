#!/usr/bin/env python3
"""
Simple Radiomics Extractor for MRI Data
=======================================

A simplified version that avoids pandas/NumPy compatibility issues.
Now includes incremental processing to avoid reprocessing already analyzed images.
"""

import os
import yaml
import csv
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import Path
import logging
import json
from datetime import datetime

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
        subject_id,
        "anat",
        f"{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz"
    )
    
    if os.path.exists(image_path):
        return image_path
    return None

def load_existing_results(output_path):
    """Load existing results to avoid reprocessing"""
    if not os.path.exists(output_path):
        return {}, set()
    
    existing_results = {}
    processed_subjects = set()
    
    try:
        with open(output_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject_id = row['subject_id']
                processed_subjects.add(subject_id)
                # Store the entire row for potential use
                existing_results[subject_id] = row
        
        logger.info(f"📋 Found {len(processed_subjects)} already processed subjects")
        return existing_results, processed_subjects
    except Exception as e:
        logger.warning(f"⚠️ Could not load existing results: {e}")
        return {}, set()

def save_progress_checkpoint(checkpoint_path, processed_subjects, failed_subjects, total_subjects):
    """Save progress to a checkpoint file"""
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'processed_subjects': list(processed_subjects),
        'failed_subjects': list(failed_subjects),
        'total_subjects': total_subjects,
        'progress_percentage': len(processed_subjects) / total_subjects * 100
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def extract_mri_radiomics(config_path, labels_path, output_dir, force_reprocess=False):
    """Extract radiomics features from MRI data with incremental processing"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_root = config['preprocessed_data']['smri_p']
    # Expand tilde if present
    data_root = os.path.expanduser(data_root)
    
    # Load labels
    subjects, labels = load_labels_simple(labels_path)
    logger.info(f"📊 Found {len(subjects)} subjects in {labels_path}")
    
    # Determine output file path
    output_filename = f"radiomics_MRI_{Path(labels_path).stem}.csv"
    output_path = Path(output_dir) / output_filename
    checkpoint_path = Path(output_dir) / f"checkpoint_{Path(labels_path).stem}.json"
    
    # Load existing results if not forcing reprocess
    if not force_reprocess:
        existing_results, processed_subjects = load_existing_results(output_path)
    else:
        existing_results, processed_subjects = {}, set()
        logger.info("🔄 Force reprocess mode: will reprocess all subjects")
    
    # Initialize radiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    
    # Store results
    all_features = []
    successful = 0
    failed = 0
    skipped = 0
    failed_subjects = set()
    
    # Process each subject
    for i, (subject_id, label) in enumerate(zip(subjects, labels)):
        logger.info(f"🔍 Processing {i+1}/{len(subjects)}: {subject_id} (label: {label})")
        
        # Check if already processed
        if subject_id in processed_subjects and not force_reprocess:
            logger.info(f"⏭️ Skipping {subject_id} (already processed)")
            all_features.append(existing_results[subject_id])
            skipped += 1
            continue
        
        # Find image path
        image_path = find_mri_path(data_root, subject_id)
        
        if not image_path:
            logger.warning(f"❌ Image not found for {subject_id}")
            failed += 1
            failed_subjects.add(subject_id)
            continue
        
        try:
            # Create mask and extract features
            image, mask = create_mask_from_image(image_path)
            if image is None or mask is None:
                logger.error(f"❌ Failed to create mask for {subject_id}")
                failed += 1
                failed_subjects.add(subject_id)
                continue
            
            result = extractor.execute(image, mask)
            
            # Add subject info
            result['subject_id'] = subject_id
            result['label'] = label
            
            all_features.append(result)
            successful += 1
            
            feature_count = len([k for k in result.keys() if k not in ['subject_id', 'label']])
            logger.info(f"✅ Extracted {feature_count} features from {subject_id}")
            
            # Save checkpoint every 10 successful extractions
            if successful % 10 == 0:
                save_progress_checkpoint(
                    checkpoint_path, 
                    {f['subject_id'] for f in all_features}, 
                    failed_subjects, 
                    len(subjects)
                )
                logger.info(f"💾 Progress checkpoint saved ({successful}/{len(subjects)} completed)")
            
        except Exception as e:
            logger.error(f"❌ Error processing {subject_id}: {e}")
            failed += 1
            failed_subjects.add(subject_id)
    
    # Save final results
    if all_features:
        # Write CSV manually to avoid pandas issues
        with open(output_path, 'w', newline='') as f:
            if all_features:
                fieldnames = all_features[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_features)
        
        # Save final checkpoint
        save_progress_checkpoint(
            checkpoint_path, 
            {f['subject_id'] for f in all_features}, 
            failed_subjects, 
            len(subjects)
        )
        
        logger.info(f"\n💾 Saved {len(all_features)} feature sets to {output_path}")
        logger.info(f"✅ Successful: {successful}, ⏭️ Skipped: {skipped}, ❌ Failed: {failed}")
        logger.info(f"📊 Progress: {len(all_features)}/{len(subjects)} subjects processed")
        
        return output_path
    else:
        logger.error("❌ No features were successfully extracted")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract radiomics features from MRI data (incremental)')
    parser.add_argument('--labels', required=True, help='Path to CSV file with subject_id and label columns')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing of all subjects (ignore existing results)')
    parser.add_argument('--check-progress', action='store_true',
                       help='Check progress without processing new subjects')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine output file path for progress check
    output_filename = f"radiomics_MRI_{Path(args.labels).stem}.csv"
    output_path = Path(args.output_dir) / output_filename
    checkpoint_path = Path(args.output_dir) / f"checkpoint_{Path(args.labels).stem}.json"
    
    if args.check_progress:
        # Just check progress without processing
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"📊 Progress Report:")
            logger.info(f"   Processed: {len(checkpoint_data['processed_subjects'])} subjects")
            logger.info(f"   Failed: {len(checkpoint_data['failed_subjects'])} subjects")
            logger.info(f"   Total: {checkpoint_data['total_subjects']} subjects")
            logger.info(f"   Progress: {checkpoint_data['progress_percentage']:.1f}%")
            logger.info(f"   Last update: {checkpoint_data['timestamp']}")
        else:
            logger.info("📊 No checkpoint found - no previous processing detected")
        return
    
    # Run extraction
    output_path = extract_mri_radiomics(
        args.config, args.labels, args.output_dir, args.force_reprocess
    )
    
    if output_path:
        logger.info(f"🎉 Feature extraction completed successfully!")
        logger.info(f"📁 Results saved to: {output_path}")
        logger.info(f"📊 To check progress: python {__file__} --labels {args.labels} --output-dir {args.output_dir} --check-progress")
    else:
        logger.error("💥 Feature extraction failed!")

if __name__ == "__main__":
    main() 