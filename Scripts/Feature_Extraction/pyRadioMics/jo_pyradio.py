import os
import yaml
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import Path

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Paths from config
data_root = config['preprocessed_data']['smri_p']  # smriprep directory
labels_path = 'Labels/train_labels.csv'  # or val_labels.csv

# Radiomics extractor with default settings
extractor = featureextractor.RadiomicsFeatureExtractor()

def create_dummy_mask(image_path):
    """Create a mask from non-zero regions of the image"""
    image = sitk.ReadImage(image_path)
    mask = sitk.NotEqual(image, 0)  # Non-zero region = ROI
    return image, mask

def extract_radiomics_features():
    """Extract radiomics features from all subjects in the dataset"""
    
    # Load labels
    df = pd.read_csv(labels_path)
    if 'subject_id' not in df.columns or 'label' not in df.columns:
        df = pd.read_csv(labels_path, header=None, names=['subject_id', 'label'])
    
    # Drop any rows where header strings were misread as data
    df = df[~df['subject_id'].isin(['subject_id', ''])]
    df = df[~df['label'].isin(['label', ''])]
    df['label'] = df['label'].astype(int)
    
    print(f"📊 Found {len(df)} subjects in {labels_path}")
    
    # Store results
    all_features = []
    
    # Loop through all subjects
    for idx, row in df.iterrows():
        subject_id = row['subject_id']
        label = row['label']
        
        # Construct the path to the z-scored T1 brain image
        image_path = os.path.join(
            data_root,
            "smriprep",
            subject_id,
            "anat",
            f"{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz"
        )
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
            
        print(f"\n🔍 Extracting features from: {subject_id} (label: {label})")
        
        try:
            image, mask = create_dummy_mask(image_path)
            result = extractor.execute(image, mask)
            
            # Add subject info to results
            result['subject_id'] = subject_id
            result['label'] = label
            
            all_features.append(result)
            print(f"✅ Successfully extracted {len([k for k in result.keys() if k not in ['subject_id', 'label']])} features")
            
        except Exception as e:
            print(f"❌ Error with {subject_id}: {e}")
    
    # Save results to CSV
    if all_features:
        results_df = pd.DataFrame(all_features)
        output_path = f"radiomics_features_{Path(labels_path).stem}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Saved {len(all_features)} feature sets to {output_path}")
        print(f"📈 Feature matrix shape: {results_df.shape}")
    else:
        print("❌ No features were successfully extracted")

if __name__ == "__main__":
    extract_radiomics_features()
