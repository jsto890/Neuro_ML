# dataset.py

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np

class PETDataset(Dataset):
    """
    PyTorch Dataset for loading single-channel PET volumes and labels.
    Expects:
      - A CSV file with columns 'subject_id' and 'label' (headered or headerless).
      - Supports multiple on-disk layouts under data_root/PET:
        1) PET/(site)/(disease)/((sub-id)_SITE_PET_DISEASE)/(sub-id)_SITE_PET_DISEASE_*.nii.gz
        2) PET/(disease)/((sub-id)_SITE_PET_DISEASE)/(sub-id)_SITE_PET_DISEASE_*.nii.gz
        3) PET/(disease)/(sub-id)_SITE_PET_DISEASE_*.nii.gz
         Where Sites ∈ {ADNI, PPMI} and Diseases ∈ {CN, PD, AD}
      - Preferred naming: sub-{ID}_{SITE}_PET_{DISEASE}_SUVR_s2_brain_soft4.nii.gz
      - Fallback naming:  sub-{ID}_{SITE}_PET_{DISEASE}_SUVR.nii.gz (for backward compatibility)
    """

    def __init__(self, csv_path: str, data_root: str, target_shape: tuple[int, int, int] = (96, 112, 96)):
        """
        Args:
            csv_path  (str): path to CSV; expects columns 'subject_id' and 'label',
                             but can handle headerless files.
            data_root (str): root folder where PET/(site)/(disease)/((sub-id)_SITE_PET_DISEASE)/(sub-id)_SITE_PET_DISEASE_SUVR_s2_brain_soft4.nii.gz files live.
                             The script will automatically search for files in ADNI/CN, ADNI/PD, ADNI/AD, PPMI/CN, PPMI/PD, PPMI/AD directories.
                             File naming is dynamic: sub-{ID}_{SITE}_PET_{DISEASE}_SUVR_s2_brain_soft4.nii.gz
                             Fallback naming: sub-{ID}_{SITE}_PET_{DISEASE}_SUVR.nii.gz (for backward compatibility)
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

        # Construct the path to the PET SUVR image
        # Updated to robustly handle multiple directory layouts
        possible_paths = []
        
        # Try different combinations of site and disease
        sites = ['ADNI', 'PPMI']
        diseases = ['CN', 'PD', 'AD']
        
        # 1) Original layout: PET/(site)/(disease)/subdir/file
        for site in sites:
            for disease in diseases:
                possible_paths.append(os.path.join(
                    self.data_root, "PET", site, disease,
                    f"{sid}_{site}_PET_{disease}",
                    f"{sid}_{site}_PET_{disease}_SUVR_s2_brain_soft4.nii.gz"
                ))
                possible_paths.append(os.path.join(
                    self.data_root, "PET", site, disease,
                    f"{sid}_{site}_PET_{disease}",
                    f"{sid}_{site}_PET_{disease}_SUVR.nii.gz"
                ))

        # 2) Layout: PET/(disease)/subdir/file (no site level)
        for disease in diseases:
            for site in sites:
                possible_paths.append(os.path.join(
                    self.data_root, "PET", disease,
                    f"{sid}_{site}_PET_{disease}",
                    f"{sid}_{site}_PET_{disease}_SUVR_s2_brain_soft4.nii.gz"
                ))
                possible_paths.append(os.path.join(
                    self.data_root, "PET", disease,
                    f"{sid}_{site}_PET_{disease}",
                    f"{sid}_{site}_PET_{disease}_SUVR.nii.gz"
                ))

                # 3) Layout: PET/(disease)/file (no subdir)
                possible_paths.append(os.path.join(
                    self.data_root, "PET", disease,
                    f"{sid}_{site}_PET_{disease}_SUVR_s2_brain_soft4.nii.gz"
                ))
                possible_paths.append(os.path.join(
                    self.data_root, "PET", disease,
                    f"{sid}_{site}_PET_{disease}_SUVR.nii.gz"
                ))

        # 4) Defensive: some datasets may place subdir disease under a different top-level folder
        for top_dir in diseases:
            for site in sites:
                for disease in diseases:
                    possible_paths.append(os.path.join(
                        self.data_root, "PET", top_dir,
                        f"{sid}_{site}_PET_{disease}",
                        f"{sid}_{site}_PET_{disease}_SUVR_s2_brain_soft4.nii.gz"
                    ))
                    possible_paths.append(os.path.join(
                        self.data_root, "PET", top_dir,
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
        data = img.get_fdata()  # shape: (D, H, W) (may vary by site)

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
