# dataset.py

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd

class SMRIDataset(Dataset):
    """
    PyTorch Dataset for loading single-channel sMRI volumes and labels.
    Expects:
      - A CSV file with columns 'subject_id' and 'label' (headered or headerless).
      - A data_root directory containing smriprep/<subject>/anat/<subject>_desc-preproc_T1w_brain_zscore.nii.gz
    """

    def __init__(self, csv_path: str, data_root: str, transform=None, return_index: bool = False):
        """
        Args:
            csv_path  (str): path to CSV; expects columns 'subject_id' and 'label',
                             but can handle headerless files.
            data_root (str): root folder where smriprep/<subject>/anat/... lives.
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
        self.transform = transform
        self.return_index = return_index

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx: int):
        sid = self.subjects[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Construct the path to the z-scored T1 brain image
        img_path = os.path.join(
            self.data_root,
            "smriprep",
            sid,
            "anat",
            f"{sid}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore.nii.gz"
        )

        img = nib.load(img_path)
        data = img.get_fdata()  # shape: (D, H, W)

        # Convert to a torch.FloatTensor with shape [1, D, H, W]
        smri = torch.from_numpy(data).unsqueeze(0).float()

        # Optional train-time transforms (expects and returns torch.Tensor [1, D, H, W])
        if self.transform is not None:
            smri = self.transform(smri)

        if self.return_index:
            return smri, label, idx
        return smri, label
