# scripts/dataset.py

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd

class SMRIDataset(Dataset):
    """
    PyTorch Dataset for loading single‐channel sMRI volumes and labels.
    Expects:
      - A CSV file with columns [subject_id, label]
      - A data_root directory containing NIfTI files named <subject_id>.nii.gz
    """

    def __init__(self, csv_path: str, data_root: str):
        """
        Args:
            csv_path  (str): path to CSV with two columns: 'subject_id', 'label'.
            data_root (str): folder where <subject_id>.nii.gz lives, e.g. "data/preprocessed/sMRI".
        """
        self.df = pd.read_csv(csv_path)
        self.subjects = self.df['subject_id'].tolist()
        self.labels   = self.df['label'].tolist()
        self.data_root = data_root

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx: int):
        sid   = self.subjects[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Locate and load the NIfTI
        img_path = os.path.join(self.data_root, f"{sid}.nii.gz")
        img_nii = nib.load(img_path)
        img_np  = img_nii.get_fdata()                # shape = (D, H, W)

        # Convert to a 4D torch Tensor: [1, D, H, W]
        smri = torch.from_numpy(img_np).unsqueeze(0).float()

        return smri, label
