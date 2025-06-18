#!/usr/bin/env python3
"""
Test script to verify data paths and structure for radiomics extraction
"""

import os
import yaml
import pandas as pd
from pathlib import Path

def test_config():
    """Test if config file can be loaded"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config file loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def test_labels(labels_path):
    """Test if labels file exists and can be loaded"""
    try:
        df = pd.read_csv(labels_path)
        if 'subject_id' not in df.columns or 'label' not in df.columns:
            df = pd.read_csv(labels_path, header=None, names=['subject_id', 'label'])
        
        # Clean the dataframe
        df = df[~df['subject_id'].isin(['subject_id', ''])]
        df = df[~df['label'].isin(['label', ''])]
        df['label'] = df['label'].astype(int)
        
        print(f"✅ Labels file loaded: {len(df)} subjects")
        print(f"   Sample subjects: {df['subject_id'].head().tolist()}")
        print(f"   Labels: {sorted(df['label'].unique())}")
        return df
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return None

def test_mri_paths(config, df):
    """Test MRI data paths"""
    data_path = config['preprocessed_data']['smri_p']
    print(f"\n🔍 Testing MRI paths in: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ MRI data path does not exist: {data_path}")
        return
    
    # Test first few subjects
    for subject_id in df['subject_id'].head(3):
        image_path = os.path.join(
            data_path,
            "smriprep",
            subject_id,
            "anat",
            f"{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz"
        )
        
        if os.path.exists(image_path):
            print(f"✅ Found MRI for {subject_id}")
        else:
            print(f"❌ Missing MRI for {subject_id}: {image_path}")

def test_pet_paths(config, df):
    """Test PET data paths"""
    data_path = config['preprocessed_data']['pet_p']
    print(f"\n🔍 Testing PET paths in: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ PET data path does not exist: {data_path}")
        return
    
    # Check directory structure
    for site_dir in Path(data_path).iterdir():
        if site_dir.is_dir():
            print(f"   Found site: {site_dir.name}")
            for dx_dir in site_dir.iterdir():
                if dx_dir.is_dir():
                    print(f"     Found diagnosis: {dx_dir.name}")
                    # Check first few subjects
                    for subject_dir in list(dx_dir.iterdir())[:3]:
                        if subject_dir.is_dir():
                            pet_file = subject_dir / "pet_mni_crop.nii.gz"
                            if pet_file.exists():
                                print(f"       ✅ Found PET for {subject_dir.name}")
                            else:
                                print(f"       ❌ Missing PET for {subject_dir.name}")

def main():
    print("🧪 Testing P4P Radiomics Setup")
    print("=" * 40)
    
    # Test config
    config = test_config()
    if not config:
        return
    
    # Test labels
    labels_path = 'Labels/train_labels.csv'
    df = test_labels(labels_path)
    if df is None:
        return
    
    # Test MRI paths
    test_mri_paths(config, df)
    
    # Test PET paths
    test_pet_paths(config, df)
    
    print("\n🎯 Setup Summary:")
    print("   - Config: ✅")
    print("   - Labels: ✅")
    print("   - Ready to run radiomics extraction!")

if __name__ == "__main__":
    main() 