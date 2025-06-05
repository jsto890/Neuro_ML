#!/usr/bin/env python3
"""
Batch‑convert DaT‑SPECT DICOM folders → NIfTI and rescale, with subject‑specific
filenames <SubjectID>_dspect.nii.gz / <SubjectID>_dspect_rescaled.nii.gz.
"""
import subprocess, sys, json
from pathlib import Path
import nibabel as nib
from nibabel import Nifti1Image
import numpy as np

# USER SETTINGS 
INPUT_DIR  = Path('/Users/josephstorey/Desktop/Part_4_Project/data/test_data/dspect/HC/raw')
OUTPUT_DIR = Path('/Users/josephstorey/Desktop/Part_4_Project/data/test_data/dspect/HC/converted')
COMPRESS   = 'y'          # gzip y|n
IGNORE_DERIVED = 1        # 0=no, 1=yes (skip localisers), 2=verbose
REORIENT   = True
APPLY_RESCALE = True

def run_dcm2niix(input_dir: Path, output_dir: Path, subj_id: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{subj_id}_dspect"                 # e.g. I1472275_dspect.nii.gz
    cmd = [
        'dcm2niix', '-z', COMPRESS, '-i', str(IGNORE_DERIVED),
        '-f', fname, '-o', str(output_dir), str(input_dir)
    ]
    subprocess.run(cmd, check=True)
    return output_dir / f"{fname}.nii.gz"       # return main NIfTI path

def pure_python_convert(input_dir: Path, output_dir: Path, subj_id: str):
    import pydicom
    files = sorted(input_dir.glob('*.dcm'),
                   key=lambda f: int(getattr(pydicom.dcmread(f, stop_before_pixels=True),
                                             'InstanceNumber', 0)))
    if not files:
        raise RuntimeError('No DICOMs for fallback')
    frames = [pydicom.dcmread(f) for f in files]
    data = np.stack([s.pixel_array for s in frames]).astype(np.float32)
    px, py = frames[0].PixelSpacing
    pz     = frames[0].SliceThickness
    affine = np.diag([px, py, pz, 1])
    out_path = output_dir / f"{subj_id}_dspect.nii.gz"
    nib = __import__('nibabel')
    nib.save(nib.nifti1.Nifti1Image(data, affine), out_path)
    return out_path

def reorient(nii_path: Path):
    img = nib.load(str(nii_path))
    nib.save(nib.as_closest_canonical(img), str(nii_path))

def rescale(nii_path: Path):
    json_path = nii_path.with_suffix('').with_suffix('.json')
    if not json_path.exists():
        return
    meta = json.loads(json_path.read_text())
    slope = meta.get('RescaleSlope', 1.0)
    icpt  = meta.get('RescaleIntercept', 0.0)
    orig = nib.load(str(nii_path))
    data = orig.get_fdata() * slope + icpt
    out = Nifti1Image(data, orig.affine)

    out_path = nii_path.with_name(nii_path.stem + '_rescaled.nii.gz')
    nib.save(out, str(out_path))

def process_subject(subj_dir: Path):
    subj_id     = subj_dir.name
    dest_folder = OUTPUT_DIR / f"subject_{subj_id}_dspect_hc"
    print(f"\n▶ {subj_id}")
    try:
        nii_path = run_dcm2niix(subj_dir, dest_folder, subj_id)
    except subprocess.CalledProcessError:
        print("dcm2niix failed – trying pure‑Python fallback…")
        nii_path = pure_python_convert(subj_dir, dest_folder, subj_id)
    if REORIENT:
        reorient(nii_path)
    if APPLY_RESCALE:
        rescale(nii_path)
    print("✓ finished", subj_id)
    
if __name__ == '__main__':
    if not INPUT_DIR.is_dir():
        sys.exit(f"INPUT_DIR not found: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    if not subs:
        sys.exit('No subject directories in INPUT_DIR')
    for s in subs:
        process_subject(s)
