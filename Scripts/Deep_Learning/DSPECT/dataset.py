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
    PyTorch Dataset for loading single-channel SPECT volumes and labels.
    Expects:
      - A CSV file with columns 'subject_id' and 'label' (headered or headerless).
      - Preprocessed DSPECT directory structure under data_root:
            CN_SPECT_PPMI_postprocessed/Subject_*/6. postprocessed.nii.gz
            PD_SPECT_PPMI_postprocessed/Subject_*/6. postprocessed.nii.gz
    """

    def __init__(self, csv_path: str, data_root: str, target_shape: tuple[int, int, int] = (91, 109, 91), transform=None):
        """
        Args:
            csv_path  (str): path to CSV; expects columns 'subject_id' and 'label',
                             but can handle headerless files.
            data_root (str): root folder where DSPECT postprocessed data lives with CN_/PD_ folders.
            transform (callable|None): optional transform applied to the image tensor [1, D, H, W]
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
        
        # Get unique labels and create mapping for binary classification (0, 2 -> 0, 1)
        unique_labels = sorted(df['label'].unique())
        print(f"[INFO] SPECT labels in dataset: {unique_labels}")
        
        # Validate that we have exactly 2 labels for binary classification
        if len(unique_labels) != 2:
            raise ValueError(f"SPECT dataset expects exactly 2 labels for binary classification. Found: {unique_labels}")
        
        # Create label mapping to convert labels to 0, 1 for binary classification
        # This handles cases like (0, 2) -> (0, 1) or (1, 2) -> (0, 1)
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        print(f"[INFO] Label mapping: {self.label_mapping}")
        
        # Apply label mapping
        df['mapped_label'] = df['label'].map(self.label_mapping)

        self.df = df
        self.subjects = df['subject_id'].tolist()
        self.labels = df['mapped_label'].tolist()
        self.data_root = data_root
        self.target_shape = target_shape
        self.transform = transform

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

        # Get the original label to determine directory
        original_label = self.df.iloc[idx]['label']
        
        # Construct the path to the SPECT image based on original label
        # Map original labels to directory names for SPECT binary classification
        if original_label == 0:
            diagnosis_dir = 'CN_SPECT_PPMI_postprocessed'
        elif original_label == 2:
            diagnosis_dir = 'PD_SPECT_PPMI_postprocessed'
        else:
            raise ValueError(f"SPECT dataset expects labels 0 (CN) or 2 (PD). Found: {original_label}")
        
        # Required DSPECT filename
        candidate_path = os.path.join(self.data_root, diagnosis_dir, sid, '6. postprocessed.nii.gz')
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(
                f"SPECT file not found for subject {sid}. Expected: {candidate_path}"
            )

        # Load image robustly; if corrupted/unreadable, return a zero placeholder
        try:
            img = nib.load(candidate_path)
            data = img.get_fdata()  # shape: (D, H, W) (may vary by site)
        except (ImageFileError, Exception) as e:
            print(f"[WARN] Failed to load NIfTI for {sid} at {candidate_path}: {e}. Using zero volume placeholder.")
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

        # Optional transform (e.g., MONAI Compose) applied on-the-fly
        if self.transform is not None:
            pet = self.transform(pet)

        return pet, label
