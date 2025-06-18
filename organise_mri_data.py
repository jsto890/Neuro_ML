import os
import shutil
import pandas as pd
from pathlib import Path

def organize_mri_data(
        root_dir: Path,
        mapping_csv: Path,
        output_base: Path,
        timepoint: str = "BL",
        modality: str = "MRI"
    ):
    """
    Scan through all 'anat' folders under root_dir, rename nii and json files using pattern:
    sub-<ID>_<timepoint>_<modality>_<disease>.<ext>
    and copy into output_base/<timepoint>/<disease>/<subject>/

    mapping_csv must have columns: MRI_ID (e.g. sub-CLA00045) and diagnosis (AD, CN, or PD)
    """
    # Load mapping
    df = pd.read_csv(mapping_csv)
    # Only keep relevant diagnoses
    df = df[df['diagnosis'].isin(['AD', 'CN', 'PD'])]
    mapping = dict(zip(df['MRI_ID'], df['diagnosis']))

    # Iterate over each top-level folder (e.g., CLA, COA)
    for top_level_dir in root_dir.iterdir():
        if not top_level_dir.is_dir():
            continue
        
        # Look for sub-* folders inside the top-level folder
        for subj_dir in top_level_dir.iterdir():
            if not subj_dir.is_dir() or not subj_dir.name.startswith('sub-'):
                continue
            
            subj_code = subj_dir.name  # e.g. sub-CLA00045
            disease = mapping.get(subj_code)
            if disease is None:
                print(f"Skipping {subj_code}: no valid diagnosis entry.")
                continue

            anat_dir = subj_dir / 'anat'
            if not anat_dir.exists():
                print(f"Skipping {subj_code}: no anat folder.")
                continue

            # Find nii and json files
            for ext in ('nii.gz', 'json'):
                for file in anat_dir.glob(f"*.{ext}"):
                    new_fname = f"{subj_code}_{timepoint}_{modality}_{disease}.{ext}"
                    dest_dir = output_base / timepoint / disease / subj_code
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file, dest_dir / new_fname)
                    print(f"Copied {file.name} -> {dest_dir / new_fname}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize MRI anat data by renaming and copying files."
    )
    parser.add_argument(
        'root_dir', type=Path,
        help='Root folder containing subject subdirectories'
    )
    parser.add_argument(
        'mapping_csv', type=Path,
        help='CSV file with columns MRI_ID and diagnosis'
    )
    parser.add_argument(
        'output_base', type=Path,
        help='Base output path for organized data'
    )
    parser.add_argument('--timepoint', default='BL',
                        help='Timepoint label, default BL')
    parser.add_argument('--modality', default='MRI',
                        help='Modality label, default MRI')

    args = parser.parse_args()
    organize_mri_data(
        args.root_dir,
        args.mapping_csv,
        args.output_base,
        timepoint=args.timepoint,
        modality=args.modality
    )
