#!/usr/bin/env python3
"""
Simple Radiomics Extractor for MRI/PET Data
===========================================

• MRI: uses sMRIPrep brain mask if available; otherwise non-zero mask
• PET: finds preprocessed PET under data/preprocessed/PET/{disease}/{subject_id}/
       filename pattern: {sid}_*_PET_{disease}_SUVR_s2_brain_soft4.nii.gz (fallback to *_SUVR.nii.gz)

Outputs:
• MRI: radiomics_MRI_{labels_stem}.csv (unchanged)
• PET: radiomics_pet.csv (as requested)
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

def find_mri_mask_path(image_path: str) -> str:
    """Infer sMRIPrep brain mask path from the z-scored image path.

    Expected image filename:
      {subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz
    Corresponding mask filename:
      {subject_id}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
    """
    return image_path.replace(
        "_desc-preproc_T1w_brain_zscore.nii.gz", "_desc-brain_mask.nii.gz"
    )


def load_image_and_mask(image_path: str):
    """Load image and prefer the sMRIPrep brain mask; fallback to non-zero mask.

    Returns (image, mask) or (None, None) on failure.
    """
    try:
        image = sitk.ReadImage(image_path)

        # Prefer the sMRIPrep brain mask if present
        mask_path = find_mri_mask_path(image_path)
        if os.path.exists(mask_path):
            mask = sitk.ReadImage(mask_path)
            # Ensure mask is integer type expected by PyRadiomics
            if mask.GetPixelID() != sitk.sitkUInt8:
                mask = sitk.Cast(mask, sitk.sitkUInt8)
        else:
            # Fallback: derive a binary mask from image non-zeros
            mask = sitk.NotEqual(image, 0)

        # Sanity-check that the mask contains ROI voxels (label==1)
        try:
            import numpy as np  # local import to avoid global dependency when unused
            mask_sum = np.asarray(sitk.GetArrayViewFromImage(mask)).sum()
        except Exception:
            # Conservative fallback if numpy view fails
            stats = sitk.StatisticsImageFilter()
            stats.Execute(mask)
            mask_sum = stats.GetSum()

        if mask_sum == 0:
            logger.error(
                f"Mask appears empty for image: {image_path}. "
                f"Checked path: {mask_path}"
            )
            return None, None

        return image, mask
    except Exception as e:
        logger.error(f"Error loading image/mask from {image_path}: {e}")
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

def find_pet_path(data_root, subject_id):
    """Find the PET image path for a given subject.
    Scans data/preprocessed/PET/{disease}/{subject_dir}/ where subject_dir starts with subject_id,
    e.g., sub-XXXX_ADNI_PET_CN, and matches preferred/legacy SUVR filenames.
    """
    try:
        from pathlib import Path
        base = Path(os.path.expanduser(data_root))
        if not base.exists():
            return None
        for dx_dir in base.iterdir():
            if not dx_dir.is_dir():
                continue
            # Find subject-specific directory by prefix match (subject_id + "_")
            for subject_dir in dx_dir.iterdir():
                if not subject_dir.is_dir():
                    continue
                name = subject_dir.name
                if not name.startswith(f"{subject_id}_"):
                    continue
                parts = name.split('_')
                disease_token = parts[-1] if len(parts) >= 4 else dx_dir.name
                disease_token_upper = str(disease_token).upper()
                patterns = [
                    f"{subject_id}_*_PET_{disease_token_upper}_SUVR_s2_brain_soft4.nii.gz",
                    f"{subject_id}_*_PET_{disease_token_upper}_SUVR_s2_brain_soft4.nii",
                    f"{subject_id}_*_PET_{disease_token_upper}_SUVR.nii.gz",
                    f"{subject_id}_*_PET_{disease_token_upper}_SUVR.nii",
                ]
                for pat in patterns:
                    matches = list(subject_dir.glob(pat))
                    if matches:
                        return str(matches[0])
        return None
    except Exception:
        return None

def load_pet_image_and_mask(image_path: str):
    """Load PET image and build a non-zero mask suitable for PyRadiomics."""
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.NotEqual(image, 0)
        if mask.GetPixelID() != sitk.sitkUInt8:
            mask = sitk.Cast(mask, sitk.sitkUInt8)
        return image, mask
    except Exception as e:
        logger.error(f"Error loading PET image/mask from {image_path}: {e}")
        return None, None

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
            # Load image and a valid brain mask (prefer sMRIPrep brain mask)
            image, mask = load_image_and_mask(image_path)
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

def find_spect_path(data_root, subject_id):
    """Find the SPECT image path for a given subject.
    Searches in CN_SPECT_PPMI_postprocessed and PD_SPECT_PPMI_postprocessed folders.
    """
    try:
        from pathlib import Path
        base = Path(os.path.expanduser(data_root))
        if not base.exists():
            return None
        
        # Check both CN and PD directories
        for disease_dir in ['CN_SPECT_PPMI_postprocessed', 'PD_SPECT_PPMI_postprocessed']:
            subject_dir = base / disease_dir / subject_id
            if not subject_dir.exists():
                continue
            
            # Try primary file
            image_path = subject_dir / "6. postprocessed.nii.gz"
            if image_path.exists():
                return str(image_path)
            
            # Fallback to finalised
            image_path = subject_dir / "5. finalised.nii.gz"
            if image_path.exists():
                return str(image_path)
        
        return None
    except Exception:
        return None

def load_spect_image_and_mask(image_path: str):
    """Load SPECT image and build a non-zero mask suitable for PyRadiomics."""
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.NotEqual(image, 0)
        if mask.GetPixelID() != sitk.sitkUInt8:
            mask = sitk.Cast(mask, sitk.sitkUInt8)
        return image, mask
    except Exception as e:
        logger.error(f"Error loading SPECT image/mask from {image_path}: {e}")
        return None, None

def extract_spect_radiomics(config_path, labels_path, output_dir, force_reprocess=False):
    """Extract radiomics features from SPECT data with incremental processing."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_root = config['preprocessed_data']['spect_p']
    data_root = os.path.expanduser(data_root)

    # Load labels
    subjects, labels = load_labels_simple(labels_path)
    logger.info(f"📊 Found {len(subjects)} subjects in {labels_path}")

    # Output and checkpoint
    output_path = Path(output_dir) / "radiomics_spect.csv"
    checkpoint_path = Path(output_dir) / f"checkpoint_spect_{Path(labels_path).stem}.json"

    # Existing results
    if not force_reprocess:
        existing_results, processed_subjects = load_existing_results(output_path)
    else:
        existing_results, processed_subjects = {}, set()
        logger.info("🔄 Force reprocess mode: will reprocess all subjects")

    extractor = featureextractor.RadiomicsFeatureExtractor()

    all_features = []
    successful = 0
    failed = 0
    skipped = 0
    failed_subjects = set()

    for i, (subject_id, label) in enumerate(zip(subjects, labels)):
        logger.info(f"🔍 Processing {i+1}/{len(subjects)}: {subject_id} (label: {label})")

        if subject_id in processed_subjects and not force_reprocess:
            logger.info(f"⏭️ Skipping {subject_id} (already processed)")
            all_features.append(existing_results[subject_id])
            skipped += 1
            continue

        image_path = find_spect_path(data_root, subject_id)
        if not image_path:
            logger.warning(f"❌ SPECT image not found for {subject_id}")
            failed += 1
            failed_subjects.add(subject_id)
            continue

        try:
            image, mask = load_spect_image_and_mask(image_path)
            if image is None or mask is None:
                logger.error(f"❌ Failed to create SPECT mask for {subject_id}")
                failed += 1
                failed_subjects.add(subject_id)
                continue

            result = extractor.execute(image, mask)
            result['subject_id'] = subject_id
            result['label'] = label
            result['image_path'] = image_path

            all_features.append(result)
            successful += 1

            feature_count = len([k for k in result.keys() if k not in ['subject_id', 'label', 'image_path']])
            logger.info(f"✅ Extracted {feature_count} features from {subject_id}")

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

    if all_features:
        # Write CSV manually
        with open(output_path, 'w', newline='') as f:
            fieldnames = all_features[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_features)

        save_progress_checkpoint(
            checkpoint_path,
            {f['subject_id'] for f in all_features},
            failed_subjects,
            len(subjects)
        )

        logger.info(f"\n💾 Saved {len(all_features)} SPECT feature sets to {output_path}")
        logger.info(f"✅ Successful: {successful}, ⏭️ Skipped: {skipped}, ❌ Failed: {failed}")
        return output_path
    else:
        logger.error("❌ No SPECT features were successfully extracted")
        return None

def extract_pet_radiomics(config_path, labels_path, output_dir, force_reprocess=False):
    """Extract radiomics features from PET data with incremental processing."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_root = config['preprocessed_data']['pet_p']
    data_root = os.path.expanduser(data_root)

    # Load labels
    subjects, labels = load_labels_simple(labels_path)
    logger.info(f"📊 Found {len(subjects)} subjects in {labels_path}")

    # Output and checkpoint
    output_path = Path(output_dir) / "radiomics_pet.csv"
    checkpoint_path = Path(output_dir) / f"checkpoint_pet_{Path(labels_path).stem}.json"

    # Existing results
    if not force_reprocess:
        existing_results, processed_subjects = load_existing_results(output_path)
    else:
        existing_results, processed_subjects = {}, set()
        logger.info("🔄 Force reprocess mode: will reprocess all subjects")

    extractor = featureextractor.RadiomicsFeatureExtractor()

    all_features = []
    successful = 0
    failed = 0
    skipped = 0
    failed_subjects = set()

    for i, (subject_id, label) in enumerate(zip(subjects, labels)):
        logger.info(f"🔍 Processing {i+1}/{len(subjects)}: {subject_id} (label: {label})")

        if subject_id in processed_subjects and not force_reprocess:
            logger.info(f"⏭️ Skipping {subject_id} (already processed)")
            all_features.append(existing_results[subject_id])
            skipped += 1
            continue

        image_path = find_pet_path(data_root, subject_id)
        if not image_path:
            logger.warning(f"❌ PET image not found for {subject_id}")
            failed += 1
            failed_subjects.add(subject_id)
            continue

        try:
            image, mask = load_pet_image_and_mask(image_path)
            if image is None or mask is None:
                logger.error(f"❌ Failed to create PET mask for {subject_id}")
                failed += 1
                failed_subjects.add(subject_id)
                continue

            result = extractor.execute(image, mask)
            result['subject_id'] = subject_id
            result['label'] = label
            result['image_path'] = image_path

            all_features.append(result)
            successful += 1

            feature_count = len([k for k in result.keys() if k not in ['subject_id', 'label', 'image_path']])
            logger.info(f"✅ Extracted {feature_count} features from {subject_id}")

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

    if all_features:
        # Write CSV manually
        with open(output_path, 'w', newline='') as f:
            fieldnames = all_features[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_features)

        save_progress_checkpoint(
            checkpoint_path,
            {f['subject_id'] for f in all_features},
            failed_subjects,
            len(subjects)
        )

        logger.info(f"\n💾 Saved {len(all_features)} PET feature sets to {output_path}")
        logger.info(f"✅ Successful: {successful}, ⏭️ Skipped: {skipped}, ❌ Failed: {failed}")
        return output_path
    else:
        logger.error("❌ No PET features were successfully extracted")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract radiomics features from MRI/PET/SPECT data (incremental)')
    parser.add_argument('--labels', required=True, help='Path to CSV file with subject_id and label columns')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing of all subjects (ignore existing results)')
    parser.add_argument('--check-progress', action='store_true',
                       help='Check progress without processing new subjects')
    parser.add_argument('--modality', choices=['MRI', 'PET', 'SPECT'], default='PET',
                       help='Modality to process (default: PET)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine output file path for progress check
    if args.modality.upper() == 'PET':
        output_path = Path(args.output_dir) / "radiomics_pet.csv"
        checkpoint_path = Path(args.output_dir) / f"checkpoint_pet_{Path(args.labels).stem}.json"
    elif args.modality.upper() == 'SPECT':
        output_path = Path(args.output_dir) / "radiomics_spect.csv"
        checkpoint_path = Path(args.output_dir) / f"checkpoint_spect_{Path(args.labels).stem}.json"
    else:
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
    
    # Run extraction by modality
    if args.modality.upper() == 'PET':
        output_path = extract_pet_radiomics(
            args.config, args.labels, args.output_dir, args.force_reprocess
        )
    elif args.modality.upper() == 'SPECT':
        output_path = extract_spect_radiomics(
            args.config, args.labels, args.output_dir, args.force_reprocess
        )
    else:
        output_path = extract_mri_radiomics(
            args.config, args.labels, args.output_dir, args.force_reprocess
        )
    
    if output_path:
        logger.info(f"🎉 Feature extraction completed successfully!")
        logger.info(f"📁 Results saved to: {output_path}")
        logger.info(f"📊 To check progress: python {__file__} --modality {args.modality} --labels {args.labels} --output-dir {args.output_dir} --check-progress")
    else:
        logger.error("💥 Feature extraction failed!")

if __name__ == "__main__":
    main() 