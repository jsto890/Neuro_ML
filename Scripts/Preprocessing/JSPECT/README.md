# JSPECT Preprocessing Pipeline

This script preprocesses DaT-SPECT NIfTI files into MNI space and performs SUVR normalization using an occipital mask.

Outputs are written under the preprocessed SPECT directory configured in `config.yaml` (not into the repo). By default, files go to `~/reseng202500013-ndd-ml/data/preprocessed/SPECT/jfinal`.

## Requirements
- SimpleITK, NumPy, PyYAML (already declared in `requirements.txt`)
- Template files defined in `config.yaml`:
  - `templates.SPECT_template` (e.g., `Templates/SPECT/symFPCITtemplate_MNI_norm.nii`)
  - `templates.SPECT_occipital` (e.g., `Templates/SPECT/occipital_mask.nii.gz`)

## Usage
```bash
python Preprocessing/JSPECT/run_jspect.py \
  --output-subdir jfinal \
  --clip-upper 10.0 \
  --skip-existing
```

Optional overrides:
- `--config /absolute/path/to/config.yaml`
- `--input-root ~/reseng202500013-ndd-ml/data/raw/SPECT`
- `--template /abs/path/to/template.nii[.gz]`
- `--occipital-mask /abs/path/to/occipital_mask.nii[.gz]`
- `--clip-lower 0.0 --clip-upper 10.0`
- `--limit 5`

The script scans for subject NIfTIs under `raw/SPECT/PPMI/{CN,PD}/sub-*/`. For each subject it saves:
- `sub-*_space-MNI.nii.gz` (registered to template, pre-SUVR)
- `sub-*_space-MNI_SUVR.nii.gz` (SUVR-normalized, optionally clipped)

If the occipital mask grid differs from the template, it is resampled to the template grid with nearest-neighbor interpolation.

