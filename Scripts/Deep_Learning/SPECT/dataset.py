#!/usr/bin/env python3
"""
SPECT Dataset Loader for Deep Learning
Optimized for preprocessed SPECT images from the DSPECT pipeline

Features:
- Direct loading from CN_SPECT_PPMI_postprocessed and PD_SPECT_PPMI_postprocessed folders
- Automatic label assignment (CN=0, PD=1)
- Memory-efficient loading with optional caching
- Built-in data validation and quality checks
- Support for both training and inference modes
"""

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Tuple, Optional, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SPECTDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed SPECT images.
    
    Expects:
    - CN_SPECT_PPMI_postprocessed/ folder with Subject_* subfolders
    - PD_SPECT_PPMI_postprocessed/ folder with Subject_* subfolders
    - Each subject folder contains '6. postprocessed.nii.gz'
    
    Outputs:
    - Images: torch.FloatTensor of shape [1, 91, 109, 91]
    - Labels: torch.LongTensor with CN=0, PD=1
    """
    
    def __init__(self, 
                 data_root: str,
                 labels_csv: Optional[str] = None,
                 transform: Optional[callable] = None,
                 cache_data: bool = False,
                 validate_data: bool = True):
        """
        Initialize SPECT dataset.
        
        Args:
            data_root: Path to SPECT folder containing CN_ and PD_ subfolders
            labels_csv: Optional CSV with custom labels (if None, auto-generates)
            transform: Optional data augmentation transforms
            cache_data: Whether to cache images in memory (use with caution)
            validate_data: Whether to validate data integrity on init
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.cache_data = cache_data
        self.validate_data = validate_data
        
        # Validate data root exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        # Setup paths
        self.cn_path = self.data_root / "CN_SPECT_PPMI_postprocessed"
        self.pd_path = self.data_root / "PD_SPECT_PPMI_postprocessed"
        
        if not self.cn_path.exists():
            raise FileNotFoundError(f"CN data path not found: {self.cn_path}")
        if not self.pd_path.exists():
            raise FileNotFoundError(f"PD data path not found: {self.pd_path}")
        
        # Load or generate labels
        if labels_csv and os.path.exists(labels_csv):
            self._load_labels_from_csv(labels_csv)
        else:
            self._generate_labels_automatically()
        
        # Validate data if requested
        if self.validate_data:
            self._validate_dataset()
        
        # Setup caching
        self.cached_data = {} if self.cache_data else None
        if self.cache_data:
            logger.info("Caching enabled - loading all images into memory")
            self._cache_all_data()
        
        logger.info(f"SPECT Dataset initialized with {len(self.subjects)} subjects")
        logger.info(f"Class distribution: CN={sum(1 for l in self.labels if l == 0)}, PD={sum(1 for l in self.labels if l == 1)}")
    
    def _generate_labels_automatically(self):
        """Generate labels automatically from folder structure."""
        logger.info("Generating labels automatically from folder structure...")
        
        # Get CN subjects
        cn_subjects = []
        if self.cn_path.exists():
            cn_subjects = [d.name for d in self.cn_path.iterdir() 
                          if d.is_dir() and d.name.startswith('Subject_')]
        
        # Get PD subjects
        pd_subjects = []
        if self.pd_path.exists():
            pd_subjects = [d.name for d in self.pd_path.iterdir() 
                          if d.is_dir() and d.name.startswith('Subject_')]
        
        # Create labels: CN=0, PD=1
        self.subjects = []
        self.labels = []
        self.file_paths = []
        
        # Add CN subjects
        for subject in cn_subjects:
            file_path = self.cn_path / subject / "6. postprocessed.nii.gz"
            if file_path.exists():
                self.subjects.append(subject)
                self.labels.append(0)  # CN = 0
                self.file_paths.append(str(file_path))
        
        # Add PD subjects
        for subject in pd_subjects:
            file_path = self.pd_path / subject / "6. postprocessed.nii.gz"
            if file_path.exists():
                self.subjects.append(subject)
                self.labels.append(1)  # PD = 1
                self.file_paths.append(str(file_path))
        
        logger.info(f"Found {len(cn_subjects)} CN subjects and {len(pd_subjects)} PD subjects")
    
    def _load_labels_from_csv(self, csv_path: str):
        """Load labels from CSV file."""
        logger.info(f"Loading labels from CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Handle different CSV formats
        if 'subject_id' in df.columns and 'label' in df.columns:
            # Standard format
            pass
        elif len(df.columns) >= 2:
            # Assume first two columns are subject_id and label
            df.columns = ['subject_id', 'label'] + list(df.columns[2:])
        else:
            raise ValueError(f"CSV must have at least 2 columns: subject_id and label")
        
        # Clean data
        df = df.dropna(subset=['subject_id', 'label'])
        df = df[~df['subject_id'].isin(['subject_id', ''])]
        df = df[~df['label'].isin(['label', ''])]
        
        # Convert labels to integers
        df['label'] = df['label'].astype(int)
        
        # Validate labels
        unique_labels = sorted(df['label'].unique())
        if not all(label in [0, 1] for label in unique_labels):
            raise ValueError(f"Labels must be 0 (CN) or 1 (PD), found: {unique_labels}")
        
        self.subjects = df['subject_id'].tolist()
        self.labels = df['label'].tolist()
        
        # Generate file paths
        self.file_paths = []
        for subject, label in zip(self.subjects, self.labels):
            if label == 0:  # CN
                file_path = self.cn_path / subject / "6. postprocessed.nii.gz"
            else:  # PD
                file_path = self.pd_path / subject / "6. postprocessed.nii.gz"
            
            if file_path.exists():
                self.file_paths.append(str(file_path))
            else:
                logger.warning(f"File not found for subject {subject}: {file_path}")
        
        # Remove subjects with missing files
        valid_indices = [i for i, path in enumerate(self.file_paths) if path]
        self.subjects = [self.subjects[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.file_paths = [self.file_paths[i] for i in valid_indices]
        
        logger.info(f"Loaded {len(self.subjects)} subjects from CSV")
    
    def _validate_dataset(self):
        """Validate dataset integrity and quality."""
        logger.info("Validating dataset...")
        
        valid_subjects = []
        valid_labels = []
        valid_paths = []
        
        for subject, label, file_path in zip(self.subjects, self.labels, self.file_paths):
            try:
                # Check file exists
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Load and validate image
                img = nib.load(file_path)
                data = img.get_fdata()
                
                # Check dimensions
                if data.shape != (91, 109, 91):
                    logger.warning(f"Subject {subject}: Wrong dimensions {data.shape}, expected (91, 109, 91)")
                    continue
                
                # Check for NaN/Inf values
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    logger.warning(f"Subject {subject}: Contains NaN or Inf values")
                    continue
                
                # Check data range (z-score should be roughly -5 to 5)
                if np.any(data < -10) or np.any(data > 10):
                    logger.warning(f"Subject {subject}: Extreme values detected, range [{np.min(data):.2f}, {np.max(data):.2f}]")
                
                # Check brain coverage (should be 1-15% for SPECT)
                coverage = np.count_nonzero(data) / data.size * 100
                if coverage < 0.5 or coverage > 20:
                    logger.warning(f"Subject {subject}: Unusual brain coverage {coverage:.1f}%")
                
                valid_subjects.append(subject)
                valid_labels.append(label)
                valid_paths.append(file_path)
                
            except Exception as e:
                logger.error(f"Error validating subject {subject}: {e}")
                continue
        
        # Update dataset with valid subjects
        self.subjects = valid_subjects
        self.labels = valid_labels
        self.file_paths = valid_paths
        
        logger.info(f"Dataset validation complete: {len(self.subjects)} valid subjects")
        
        # Report statistics
        if self.subjects:
            cn_count = sum(1 for l in self.labels if l == 0)
            pd_count = sum(1 for l in self.labels if l == 1)
            logger.info(f"Final distribution: CN={cn_count}, PD={pd_count}")
    
    def _cache_all_data(self):
        """Cache all images in memory for faster access."""
        logger.info("Caching all images in memory...")
        
        for i, (subject, file_path) in enumerate(zip(self.subjects, self.file_paths)):
            try:
                img = nib.load(file_path)
                data = img.get_fdata()
                
                # Convert to tensor and cache
                tensor_data = torch.from_numpy(data).unsqueeze(0).float()
                self.cached_data[i] = tensor_data
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Cached {i + 1}/{len(self.subjects)} images")
                    
            except Exception as e:
                logger.error(f"Error caching subject {subject}: {e}")
                self.cached_data[i] = None
        
        logger.info("Caching complete")
    
    def __len__(self) -> int:
        """Return number of subjects in dataset."""
        return len(self.subjects)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single SPECT image and label.
        
        Args:
            idx: Index of the subject
            
        Returns:
            tuple: (image_tensor, label_tensor)
                - image_tensor: torch.FloatTensor of shape [1, 91, 109, 91]
                - label_tensor: torch.LongTensor with value 0 (CN) or 1 (PD)
        """
        if idx >= len(self.subjects):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.subjects)}")
        
        subject = self.subjects[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Load image (from cache or file)
        if self.cache_data and idx in self.cached_data:
            if self.cached_data[idx] is not None:
                image = self.cached_data[idx]
            else:
                raise RuntimeError(f"Cached data for subject {subject} is None")
        else:
            # Load from file
            file_path = self.file_paths[idx]
            try:
                img = nib.load(file_path)
                data = img.get_fdata()
                
                # Convert to tensor with shape [1, 91, 109, 91]
                image = torch.from_numpy(data).unsqueeze(0).float()
                
            except Exception as e:
                logger.error(f"Error loading subject {subject}: {e}")
                # Return zero tensor as fallback
                image = torch.zeros(1, 91, 109, 91, dtype=torch.float32)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_subject_info(self, idx: int) -> Dict:
        """Get detailed information about a subject."""
        if idx >= len(self.subjects):
            raise IndexError(f"Index {idx} out of range")
        
        subject = self.subjects[idx]
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        
        info = {
            'subject_id': subject,
            'label': label,
            'diagnosis': 'CN' if label == 0 else 'PD',
            'file_path': file_path,
            'index': idx
        }
        
        # Add image statistics if file exists
        if os.path.exists(file_path):
            try:
                img = nib.load(file_path)
                data = img.get_fdata()
                info.update({
                    'shape': data.shape,
                    'data_type': str(data.dtype),
                    'min_value': float(np.min(data)),
                    'max_value': float(np.max(data)),
                    'mean_value': float(np.mean(data)),
                    'std_value': float(np.std(data)),
                    'coverage_percent': float(np.count_nonzero(data) / data.size * 100)
                })
            except Exception as e:
                info['error'] = str(e)
        
        return info
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        if not self.subjects:
            return torch.tensor([1.0, 1.0])
        
        # Count samples per class
        class_counts = np.bincount(self.labels)
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.subjects)
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        return torch.from_numpy(class_weights).float()
    
    def save_labels_csv(self, output_path: str):
        """Save current labels to CSV file."""
        df = pd.DataFrame({
            'subject_id': self.subjects,
            'label': self.labels,
            'diagnosis': ['CN' if l == 0 else 'PD' for l in self.labels]
        })
        df.to_csv(output_path, index=False)
        logger.info(f"Labels saved to: {output_path}")


class SPECTDatasetBalanced(SPECTDataset):
    """
    Balanced version of SPECT dataset that ensures equal representation of classes.
    Useful for training when classes are imbalanced.
    """
    
    def __init__(self, 
                 data_root: str,
                 labels_csv: Optional[str] = None,
                 transform: Optional[callable] = None,
                 balance_strategy: str = 'undersample',
                 random_seed: Optional[int] = 42):
        """
        Initialize balanced SPECT dataset.
        
        Args:
            balance_strategy: 'undersample' (reduce majority) or 'oversample' (duplicate minority)
            random_seed: Random seed for reproducibility
        """
        super().__init__(data_root, labels_csv, transform, cache_data=False, validate_data=True)
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self._balance_dataset(balance_strategy)
        logger.info(f"Balanced dataset created with {len(self.subjects)} subjects")
    
    def _balance_dataset(self, strategy: str):
        """Balance the dataset using specified strategy."""
        # Count samples per class
        cn_indices = [i for i, label in enumerate(self.labels) if label == 0]
        pd_indices = [i for i, label in enumerate(self.labels) if label == 1]
        
        cn_count = len(cn_indices)
        pd_count = len(pd_indices)
        
        logger.info(f"Original distribution: CN={cn_count}, PD={pd_count}")
        
        if strategy == 'undersample':
            # Reduce majority class to match minority
            target_count = min(cn_count, pd_count)
            
            if cn_count > target_count:
                cn_indices = np.random.choice(cn_indices, target_count, replace=False)
            if pd_count > target_count:
                pd_indices = np.random.choice(pd_indices, target_count, replace=False)
                
        elif strategy == 'oversample':
            # Duplicate minority class to match majority
            target_count = max(cn_count, pd_count)
            
            if cn_count < target_count:
                cn_indices = np.random.choice(cn_indices, target_count, replace=True)
            if pd_count < target_count:
                pd_indices = np.random.choice(pd_indices, target_count, replace=True)
        
        # Combine and shuffle
        all_indices = list(cn_indices) + list(pd_indices)
        np.random.shuffle(all_indices)
        
        # Update dataset
        self.subjects = [self.subjects[i] for i in all_indices]
        self.labels = [self.labels[i] for i in all_indices]
        self.file_paths = [self.file_paths[i] for i in all_indices]
        
        # Report final distribution
        final_cn = sum(1 for l in self.labels if l == 0)
        final_pd = sum(1 for l in self.labels if l == 1)
        logger.info(f"Balanced distribution: CN={final_cn}, PD={final_pd}")


# Utility functions for data splitting
def split_spect_dataset(data_root: str, 
                       output_dir: str,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.2,
                       test_ratio: float = 0.1,
                       random_seed: int = 42) -> Dict[str, str]:
    """
    Split SPECT dataset into train/validation/test sets.
    
    Args:
        data_root: Path to SPECT data folder
        output_dir: Directory to save split CSV files
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.2)
        test_ratio: Proportion for testing (default: 0.1)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to train, validation, and test CSV files
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset to get subjects and labels
    dataset = SPECTDataset(data_root, validate_data=True)
    
    if len(dataset) == 0:
        raise ValueError("No valid subjects found in dataset")
    
    # Create DataFrame
    df = pd.DataFrame({
        'subject_id': dataset.subjects,
        'label': dataset.labels,
        'diagnosis': ['CN' if l == 0 else 'PD' for l in dataset.labels]
    })
    
    # Split by class to ensure balanced representation
    cn_df = df[df['label'] == 0]
    pd_df = df[df['label'] == 1]
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Split each class
    def split_class_df(class_df, train_ratio, val_ratio, test_ratio):
        n = len(class_df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Shuffle
        class_df = class_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        train_df = class_df.iloc[:n_train]
        val_df = class_df.iloc[n_train:n_train + n_val]
        test_df = class_df.iloc[n_train + n_val:]
        
        return train_df, val_df, test_df
    
    # Split each class
    cn_train, cn_val, cn_test = split_class_df(cn_df, train_ratio, val_ratio, test_ratio)
    pd_train, pd_val, pd_test = split_class_df(pd_df, train_ratio, val_ratio, test_ratio)
    
    # Combine splits
    train_df = pd.concat([cn_train, pd_train], ignore_index=True).sample(frac=1, random_state=random_seed)
    val_df = pd.concat([cn_val, pd_val], ignore_index=True).sample(frac=1, random_state=random_seed)
    test_df = pd.concat([cn_test, pd_test], ignore_index=True).sample(frac=1, random_state=random_seed)
    
    # Save splits
    train_path = output_dir / "spect_labels_train.csv"
    val_path = output_dir / "spect_labels_val.csv"
    test_path = output_dir / "spect_labels_test.csv"
    all_path = output_dir / "spect_labels_all.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(all_path, index=False)
    
    # Report statistics
    logger.info(f"Dataset split complete:")
    logger.info(f"  Training: {len(train_df)} subjects (CN={sum(train_df['label']==0)}, PD={sum(train_df['label']==1)})")
    logger.info(f"  Validation: {len(val_df)} subjects (CN={sum(val_df['label']==0)}, PD={sum(val_df['label']==1)})")
    logger.info(f"  Testing: {len(test_df)} subjects (CN={sum(test_df['label']==0)}, PD={sum(test_df['label']==1)})")
    
    return {
        'train': str(train_path),
        'validation': str(val_path),
        'test': str(test_path),
        'all': str(all_path)
    }


if __name__ == "__main__":
    # Example usage
    data_root = "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT"
    
    # Create dataset
    dataset = SPECTDataset(data_root)
    
    # Split dataset
    output_dir = "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT/labels"
    split_paths = split_spect_dataset(data_root, output_dir)
    
    print("Dataset split complete!")
    for split_name, split_path in split_paths.items():
        print(f"  {split_name}: {split_path}")
