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
      - A data_root directory containing PET/(site)/(disease)/((sub-id)_ADNI_PET_CN)/(sub-id)_PPMI_PET_PD_SUVR.nii.gz files
      - Sites: ADNI, PPMI
      - Diseases: CN, PD, AD
    """

    def __init__(self, csv_path: str, data_root: str):
        """
        Args:
            csv_path  (str): path to CSV; expects columns 'subject_id' and 'label',
                             but can handle headerless files.
            data_root (str): root folder where PET/(site)/(disease)/((sub-id)_ADNI_PET_CN)/(sub-id)_PPMI_PET_PD_SUVR.nii.gz files live.
                             The script will automatically search for files in ADNI/CN, ADNI/PD, ADNI/AD, PPMI/CN, PPMI/PD, PPMI/AD directories.
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
        # Structure: preprocessed/PET/(site)/(disease)/((sub-id)_ADNI_PET_CN)/(sub-id)_PPMI_PET_PD_SUVR.nii.gz
        # Try different combinations of site and disease
        possible_paths = []
        
        # Try ADNI site with different diseases
        for disease in ['CN', 'PD', 'AD']:
            possible_paths.append(os.path.join(
                self.data_root,
                "PET",
                "ADNI",
                disease,
                f"{sid}_ADNI_PET_CN",
                f"{sid}_PPMI_PET_PD_SUVR.nii.gz"
            ))
        
        # Try PPMI site with different diseases
        for disease in ['CN', 'PD', 'AD']:
            possible_paths.append(os.path.join(
                self.data_root,
                "PET",
                "PPMI",
                disease,
                f"{sid}_ADNI_PET_CN",
                f"{sid}_PPMI_PET_PD_SUVR.nii.gz"
            ))
        
        # Try to find the file
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            # If not found, try the original structure as fallback
            img_path = os.path.join(
                self.data_root,
                "PET",
                f"{sid}_ADNI_PET_CN",
                f"{sid}_PPMI_PET_PD_SUVR.nii.gz"
            )
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"PET file not found for subject {sid}. Tried paths: {possible_paths}")

        img = nib.load(img_path)
        data = img.get_fdata()  # shape: (D, H, W)

        # Convert to a torch.FloatTensor with shape [1, D, H, W]
        pet = torch.from_numpy(data).unsqueeze(0).float()

        return pet, label
