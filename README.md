# Neuro ML

Multimodal machine learning workflows for neurodegenerative disease research using MRI, PET, and SPECT imaging.

## What this repository contains

- End-to-end preprocessing pipelines from DICOM/NIfTI to model-ready data
- Feature extraction and classical machine learning workflows
- Deep learning training and evaluation scripts
- Clinical deployment utilities and interpretability tooling
- Optional web interface (`frontend` and `backend`) for upload and inference workflows

## Repository structure

```text
Neuro_ML/
├── config.yaml
├── requirements.txt
├── environment.yml
├── Scripts/
│   ├── Preprocessing/
│   ├── Feature_Extraction/
│   ├── Classic_Learning/
│   ├── Deep_Learning/
│   ├── Clinical_Deploy/
│   ├── Comparison/
│   └── Visualise/
├── Templates/
├── backend/
└── frontend/
```

## Quick start

### 1) Clone and set up environment

```bash
git clone <your-repo-url>
cd Neuro_ML

conda env create -f environment.yml
conda activate p4p
```

Alternative with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure local paths

Edit `config.yaml` to point to your local raw and preprocessed data paths.

### 3) Run core workflows

Classical machine learning:

```bash
cd Scripts/Classic_Learning
python complete_workflow.py
```

Deep learning examples:

```bash
cd Scripts/Deep_Learning/MRI
python train_smri.py --config config_hardware_optimised.yaml
```

```bash
cd Scripts/Deep_Learning/PET
python train_pet.py --config config_hardware_optimised.yaml
```

```bash
cd Scripts/Deep_Learning/DSPECT
python train_spect.py --config config_hardware_optimised.yaml
```

## Development quality gates

This repository includes a GitHub Actions CI workflow for:

- `ruff` lint checks
- `pytest` test execution

Run these locally:

```bash
pip install -r requirements-dev.txt
ruff check tests
pytest -q tests
```

## Documentation

- `Scripts/README.md`
- `Scripts/Preprocessing/README.md`
- `Scripts/Classic_Learning/README.md`
- `Scripts/Deep_Learning/README.md`
- `Templates/README.md`

## Clinical and research disclaimer

This software is for research and engineering development only. It is not a clinical diagnostic system and must not be used for clinical decision-making without formal validation and regulatory approval.
