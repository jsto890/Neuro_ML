# P4P Research Project

This repository contains all code, notes, and resources for our P4P project.

## 🧠 Project Summary
We're working on a research project for the P4P course focused on detecting early stages of Alzheimer’s Disease (AD) and Parkinson’s Disease (PD) using a machine learning approach applied to PET and MRI scans.

## 📁 Project Structure
- `feature_extraction/` – pyRadiomMics testing currently 6/5/25
- `test_data/` – Python or other automation scripts
- `docs/` – Reports, posters, and notes
- `data/` – Any datasets or data files (can be .gitignored)

## 🚀 How to Get Started
1. Clone this repository:
   ```
   git clone https://github.com/YOUR_USERNAME/p4p-project.git
   ```
2. Navigate into the folder:
   ```
   cd p4p-project
   ```
3. (Optional) Create a virtual environment and install any dependencies.

## 🧑‍🤝‍🧑 Collaborators
- Jackson Schofield
- Joseph Storey


# sMRI‐Only 3D-CNN + Grad-CAM

## Overview

This mini‐project trains a simple 3D convolutional network on structural MRI (sMRI) volumes (160×192×192) for binary classification, and then uses Grad-CAM to visualize saliency maps on a few validation subjects.

- **Data location**: `data/preprocessed/sMRI/`
- **Labels**: `labels/train_labels.csv` and `labels/val_labels.csv`
- **Scripts**: everything under `scripts/`
- **Checkpoints**: best model saved to `checkpoints/best_smri_model.pth`
- **Grad-CAM outputs**: saved under `gradcam_outputs/`

## 1. Create a conda/pip environment

```bash
# Example using conda
conda create -n sMRI3d python=3.10
conda activate sMRI3d

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch

# Install other dependencies
pip install nibabel pandas scikit-learn matplotlib tqdm


Train the simple 3D-CNN on sMRI

python scripts/train_smri.py \
  --train_csv labels/train_labels.csv \
  --val_csv labels/val_labels.csv \
  --data_root data/preprocessed/sMRI \
  --epochs 30 \
  --batch_size 2 \
  --num_workers 4 \
  --checkpoint_dir checkpoints \
  --device cuda


Run Grad-CAM visualization

Once best_smri_model.pth exists, run:

python scripts/visualize_gradcam.py \
  --val_csv labels/val_labels.csv \
  --data_root data/preprocessed/sMRI \
  --checkpoint checkpoints/best_smri_model.pth \
  --device cuda \
  --output_dir gradcam_outputs
