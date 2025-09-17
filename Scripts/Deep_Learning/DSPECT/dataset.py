# dataset.py

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import pandas as pd
import numpy as np

class SPECTDataset(Dataset):
    """
    PyTorch Dataset for loading single-channel SPECT volumes and labels (binary: CN=0, PD=1).
    Expects:
      - A CSV file with columns 'subject_id' and 'label' (headered or headerless), where label ∈ {0,1}.
      - Preprocessed DSPECT directory structure under data_root:
            CN_SPECT_PPMI_postprocessed/Subject_*/6. postprocessed.nii.gz
            PD_SPECT_PPMI_postprocessed/Subject_*/6. postprocessed.nii.gz
    """

    def __init__(self, csv_path: str, data_root: str, target_shape: tuple[int, int, int] = (91, 109, 91)):
        """
        Args:
            csv_path  (str): path to CSV; expects columns 'subject_id' and 'label',
                             but can handle headerless files.
            data_root (str): root folder where DSPECT postprocessed data lives with CN_/PD_ folders.
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
        
        # Enforce binary labels (CN=0, PD=1)
        unique_labels = sorted(df['label'].unique())
        invalid = [l for l in unique_labels if l not in [0, 1]]
        if invalid:
            raise ValueError(f"SPECT dataset expects binary labels 0 (CN) or 1 (PD). Found: {unique_labels}")
        print(f"[INFO] SPECT labels in dataset: {unique_labels}")

        self.df = df
        self.subjects = df['subject_id'].tolist()
        self.labels = df['label'].tolist()
        self.data_root = data_root
        self.target_shape = target_shape

    @staticmethod
    def _pad_or_crop_center(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
        """
        Center pad or crop a 3D volume to the target shape.
        Pads with zeros if smaller; center-crops if larger.
        """
        assert volume.ndim == 3, "Expected a 3D volume"
        d, h, w = volume.shape
        td, th, tw = target_shape

        # Pad if needed
        pad_d_before = max((td - d) // 2, 0)
        pad_h_before = max((th - h) // 2, 0)
        pad_w_before = max((tw - w) // 2, 0)

        pad_d_after = max(td - d - pad_d_before, 0)
        pad_h_after = max(th - h - pad_h_before, 0)
        pad_w_after = max(tw - w - pad_w_before, 0)

        if pad_d_before or pad_d_after or pad_h_before or pad_h_after or pad_w_before or pad_w_after:
            volume = np.pad(
                volume,
                ((pad_d_before, pad_d_after), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
                mode='constant',
                constant_values=0,
            )

        # Crop if needed (center crop)
        d, h, w = volume.shape
        start_d = max((d - td) // 2, 0)
        start_h = max((h - th) // 2, 0)
        start_w = max((w - tw) // 2, 0)
        end_d = start_d + td
        end_h = start_h + th
        end_w = start_w + tw

        volume = volume[start_d:end_d, start_h:end_h, start_w:end_w]
        return volume

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx: int):
        sid = self.subjects[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Construct the path to the SPECT image based on label
        # label: 0 -> CN, 1 -> PD
        diagnosis_dir = 'CN_SPECT_PPMI_postprocessed' if label.item() == 0 else 'PD_SPECT_PPMI_postprocessed'
        # Common DSPECT filename
        candidate_path = os.path.join(self.data_root, diagnosis_dir, sid, '6. postprocessed.nii.gz')
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"SPECT file not found for subject {sid}. Expected: {candidate_path}")

        # Load image robustly; if corrupted/unreadable, return a zero placeholder
        try:
            img = nib.load(candidate_path)
            data = img.get_fdata()  # shape: (D, H, W) (may vary by site)
        except (ImageFileError, Exception) as e:
            print(f"[WARN] Failed to load NIfTI for {sid} at {img_path}: {e}. Using zero volume placeholder.")
            data = np.zeros(self.target_shape, dtype=np.float32)

        # Standardize shape for training consistency
        data = self._pad_or_crop_center(data, self.target_shape)

        # Sanitize NaNs/Infs and normalize (similar spirit to sMRI z-scored inputs)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        nonzero_mask = data != 0
        if np.any(nonzero_mask):
            mu = float(data[nonzero_mask].mean())
            sigma = float(data[nonzero_mask].std())
            if sigma < 1e-6:
                sigma = 1.0
            data = (data - mu) / sigma
            # Optional clipping to avoid extreme tails
            data = np.clip(data, -5.0, 5.0)

        # Convert to a torch.FloatTensor with shape [1, D, H, W]
        pet = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)

        return pet, label
