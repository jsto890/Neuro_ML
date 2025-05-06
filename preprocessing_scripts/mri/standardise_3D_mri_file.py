import os
import argparse
import tempfile
from datetime import datetime

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# =============================================================================
# CONFIG DEFAULTS
# =============================================================================
DEFAULT_INPUT_DIR = "/Users/josephstorey/Desktop/Part_4_Project/data/test_data/mri/BRAINLAT/AD"
DEFAULT_OUTPUT_DIR = "/Users/josephstorey/Desktop/Part_4_Project/data/processed_data/MRI"
DEFAULT_SPACING = (1.0, 1.0, 1.0)
LOG_FILENAME = "processing_log.txt"
TARGET_SHAPE = (160, 192, 192)

# =============================================================================
# STEP FUNCTIONS
# =============================================================================

def reorient_to_RAS(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)


def bias_field_correction(img: nib.Nifti1Image) -> nib.Nifti1Image:
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(img, tmp)
    sitk_img = sitk.ReadImage(tmp)
    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    corrected = sitk.N4BiasFieldCorrectionImageFilter().Execute(sitk_img, mask)
    arr = sitk.GetArrayFromImage(corrected).transpose(2, 1, 0)
    os.remove(tmp)
    return nib.Nifti1Image(arr, img.affine)


def skull_strip(img: nib.Nifti1Image) -> nib.Nifti1Image:
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(img, tmp)
    sitk_img = sitk.ReadImage(tmp)
    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    mask_clean = sitk.BinaryMorphologicalClosingImageFilter().Execute(mask)
    stripped = sitk.Mask(sitk_img, mask_clean)
    arr = sitk.GetArrayFromImage(stripped).transpose(2, 1, 0)
    os.remove(tmp)
    return nib.Nifti1Image(arr, img.affine)


def resample_image(img: nib.Nifti1Image, spacing: tuple = DEFAULT_SPACING) -> nib.Nifti1Image:
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(img, tmp)
    sitk_img = sitk.ReadImage(tmp);
    os.remove(tmp)
    orig_size, orig_sp = sitk_img.GetSize(), sitk_img.GetSpacing()
    new_size = [int(round(o*os_/sp)) for o, os_, sp in zip(orig_size, orig_sp, spacing)]
    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(spacing)
    res.SetSize(new_size)
    res.SetOutputOrigin(sitk_img.GetOrigin())
    res.SetOutputDirection(sitk_img.GetDirection())
    res.SetInterpolator(sitk.sitkBSpline)
    out_img = res.Execute(sitk_img)
    arr = sitk.GetArrayFromImage(out_img).transpose(2,1,0)
    
    # LPS to RAS
    dir_lps = np.array(out_img.GetDirection()).reshape(3,3)
    orig_lps = np.array(out_img.GetOrigin())
    lps2ras = np.diag([-1,-1,1])
    dir_ras = lps2ras @ dir_lps
    orig_ras = lps2ras @ orig_lps
    aff = np.eye(4)
    aff[:3,:3] = dir_ras * np.array(spacing)[None,:]
    aff[:3,3] = orig_ras
    return nib.Nifti1Image(arr, aff)


def zscore_normalize(img: nib.Nifti1Image) -> nib.Nifti1Image:
    data = img.get_fdata(); mask = data>0
    return nib.Nifti1Image((data-mask*data.mean())/data[mask].std(), img.affine)


def pad_or_crop(img: nib.Nifti1Image, shape: tuple = TARGET_SHAPE) -> nib.Nifti1Image:
    data = img.get_fdata()
    aff = img.affine.copy()
    old_spacing = img.header.get_zooms()[:3]
    # compute direction cosines
    dir_cos = aff[:3,:3] @ np.linalg.inv(np.diag(old_spacing))
    # prepare out volume
    out = np.zeros(shape, dtype=data.dtype)
    shifts_mm = np.zeros(3)
    slices_in = []
    slices_out = []
    for i in range(3):
        ds, ts = data.shape[i], shape[i]
        if ds > ts:
            start = (ds - ts)//2
            slices_in.append(slice(start, start+ts))
            slices_out.append(slice(0, ts))
            shifts_mm[i] = start * old_spacing[i]
        elif ds < ts:
            start = (ts - ds)//2
            slices_in.append(slice(0, ds))
            slices_out.append(slice(start, start+ds))
            shifts_mm[i] = -start * old_spacing[i]
        else:
            slices_in.append(slice(0, ds))
            slices_out.append(slice(0, ts))
            shifts_mm[i] = 0
    out[tuple(slices_out)] = data[tuple(slices_in)]
    # adjust origin
    aff[:3,3] += dir_cos.dot(shifts_mm)
    return nib.Nifti1Image(out, aff)

# =============================================================================
# PIPELINE
# =============================================================================

def process_subject(path, out_dir, log_file):
    base = os.path.basename(path).replace('.nii.gz','')
    steps = [
        ('Reorient', reorient_to_RAS),
        ('Bias', bias_field_correction),
        ('Strip', skull_strip),
        ('Resamp', lambda im: resample_image(im, DEFAULT_SPACING)),
        ('Norm', zscore_normalize),
        ('PadCrop', pad_or_crop)
    ]
    logs=[]
    try:
        im = nib.load(path)
        for lbl,fn in tqdm(steps, desc=base, ncols=60):
            im = fn(im); logs.append(lbl)
        outp = os.path.join(out_dir, f"{base}_final.nii.gz")
        nib.save(im, outp)
        with open(log_file,'a') as f:
            f.write(f"[{datetime.now()}] OK {base}\n")
            for L in logs: f.write(f"  - {L}\n")
            f.write(f"-> {outp}\n\n")
    except Exception as e:
        with open(log_file,'a') as f: f.write(f"[{datetime.now()}] ERR {base}: {e}\n\n")

def batch_process(in_dir, out_dir, log_file):
    for f in os.listdir(in_dir):
        if f.endswith('.nii.gz'): process_subject(os.path.join(in_dir,f), out_dir, log_file)

if __name__=='__main__':
    p=argparse.ArgumentParser('Std3D')
    p.add_argument('input_dir', nargs='?', default=DEFAULT_INPUT_DIR)
    p.add_argument('-o','--output_dir', default=DEFAULT_OUTPUT_DIR)
    a=p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    logf=os.path.join(a.output_dir, LOG_FILENAME)
    print(f"In:{a.input_dir}\nOut:{a.output_dir}\nLog:{logf}")
    batch_process(a.input_dir, a.output_dir, logf)
