# dataset.py

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd

class PETDataset(Dataset):
    """
    PyTorch Dataset for loading single-channel PET volumes and labels.
    Expects:
      - A CSV file with columns 'subject_id' and 'label' (headered or headerless).
      - A data_root directory containing PET/(site)/(disease)/((sub-id)_SITE_PET_DISEASE)/(sub-id)_SITE_PET_DISEASE_SUVR.nii.gz files
      - Sites: ADNI, PPMI
      - Diseases: CN, PD, AD
      - Dynamic naming: sub-{ID}_{SITE}_PET_{DISEASE}_SUVR.nii.gz
    """

    def __init__(self, csv_path: str, data_root: str):
        """
        Args:
            csv_path  (str): path to CSV; expects columns 'subject_id' and 'label',
                             but can handle headerless files.
            data_root (str): root folder where PET/(site)/(disease)/((sub-id)_SITE_PET_DISEASE)/(sub-id)_SITE_PET_DISEASE_SUVR.nii.gz files live.
                             The script will automatically search for files in ADNI/CN, ADNI/PD, ADNI/AD, PPMI/CN, PPMI/PD, PPMI/AD directories.
                             File naming is dynamic: sub-{ID}_{SITE}_PET_{DISEASE}_SUVR.nii.gz
        """
        # Read CSV normally
        df = pd.read_csv(csv_path)
        # If it doesn't have the expected columns, re-read as headerless
        if 'subject_id' not in df.columns or 'label' not in df.columns:
            df = pd.read_csv(csv_path, header=None, names=['subject_id', 'label'])

        # Drop any rows where header strings were misread as data
        df = df[~df['subject_id'].isin(['subject_id', ''])]
        df = df[~df['label'].isin(['label', ''])]

        # Convert label column to integer type
        df['label'] = df['label'].astype(int)
        
        # Keep original labels (0, 2) but ensure they're valid
        unique_labels = sorted(df['label'].unique())
        print(f"[INFO] Original labels in dataset: {unique_labels}")

        self.df = df
        self.subjects = df['subject_id'].tolist()
        self.labels = df['label'].tolist()
        self.data_root = data_root

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx: int):
        sid = self.subjects[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Construct the path to the PET SUVR image
        # Structure: preprocessed/PET/(site)/(disease)/((sub-id)_SITE_PET_DISEASE)/(sub-id)_SITE_PET_DISEASE_SUVR.nii.gz
        # The file naming is dynamic: sub-{ID}_{SITE}_PET_{DISEASE}_SUVR.nii.gz
        possible_paths = []
        
        # Try different combinations of site and disease
        sites = ['ADNI', 'PPMI']
        diseases = ['CN', 'PD', 'AD']
        
        for site in sites:
            for disease in diseases:
                # Try the dynamic naming pattern
                possible_paths.append(os.path.join(
                    self.data_root,
                    "PET",
                    site,
                    disease,
                    f"{sid}_{site}_PET_{disease}",
                    f"{sid}_{site}_PET_{disease}_SUVR.nii.gz"
                ))
        
        # Try to find the file
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"PET file not found for subject {sid}. Tried paths: {possible_paths}")

        img = nib.load(img_path)
        data = img.get_fdata()  # shape: (D, H, W)

        # Convert to a torch.FloatTensor with shape [1, D, H, W]
        pet = torch.from_numpy(data).unsqueeze(0).float()

        return pet, label
