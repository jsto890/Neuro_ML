#!/usr/bin/env python3
"""
Debug script to check MRI paths
"""

import os
import yaml

def debug_mri_paths():
    # Load config
    with open('/home/jsto890/reseng202500013-ndd-ml/P4P/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_root = config['preprocessed_data']['smri_p']
    print(f"Data root: {data_root}")
    
    # Test subjects from labels
    test_subjects = ['sub-CLB00151', 'sub-CLB00011', 'sub-I45060']
    
    for subject_id in test_subjects:
        image_path = os.path.join(
            data_root,
            "smriprep",
            subject_id,
            "anat",
            f"{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz"
        )
        
        print(f"\nSubject: {subject_id}")
        print(f"Expected path: {image_path}")
        print(f"Path exists: {os.path.exists(image_path)}")
        
        # Check if directory exists
        dir_path = os.path.dirname(image_path)
        print(f"Directory exists: {os.path.exists(dir_path)}")
        
        if os.path.exists(dir_path):
            print(f"Files in directory: {os.listdir(dir_path)[:5]}")

if __name__ == "__main__":
    debug_mri_paths() 