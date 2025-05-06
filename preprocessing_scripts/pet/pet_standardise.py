#!/usr/bin/env python3
"""
pure_pet_standardize.py  (PET-only standardisation)

Pipeline:
 1. (Optional) 4-D motion correction
 2. Static frame averaging
 3. SUV → SUVR (cerebellum)
 4. Rigid+SyN registration to an MNI PET template
 5. Brain mask + fixed crop (160×192×192)
 6. Save outputs

Author: Joseph Storey, 2025-04-23
"""

import json, logging, subprocess
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm

PET_CONVERTED_ROOT = Path(
    "/Users/josephstorey/Desktop/Part_4_Project/data/test_data/pet/AD/converted"
)
STD_OUTPUT_ROOT = Path(
    "/Users/josephstorey/Desktop/Part_4_Project/data/processed_data/PET/AD/standardized"
)

REF_DIR = Path(
    "/Users/josephstorey/Desktop/Part_4_Project/data/test_data/Templates/PET_refs"
)
MNI_PET_TEMPLATE = REF_DIR / "FDG-PET-template.nii.gz"
MNI_BRAIN_MASK   = REF_DIR / "MNI152_T1_1mm_brain_mask.nii.gz"
REF_REGION_MASK  = REF_DIR / "cereb_mask_thr25_bin.nii.gz"

STATIC_FRAMES = (50, 70)          # inclusive start, exclusive end
CROP_SHAPE    = (160, 192, 192)   # final tensor size for ML

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)

def motion_correct_pet4d(pet4d: Path) -> Path:
    logging.info(f"  • motion-correcting {pet4d.name}")
    pet  = ants.image_read(str(pet4d))
    mid  = pet.shape[-1] // 2
    ref  = ants.extract_image(pet, axis=3, idx=mid)
    aligned = []
    for i in range(pet.shape[-1]):
        frm = ants.extract_image(pet, axis=3, idx=i)
        tx  = ants.registration(fixed=ref, moving=frm, type_of_transform="Rigid")
        aligned.append(
            ants.apply_transforms(fixed=ref, moving=frm, transformlist=tx["fwdtransforms"])
        )
    mc4d = ants.concat_images(aligned, axis=3)
    out  = pet4d.parent / "pet_mc4d.nii.gz"
    ants.image_write(mc4d, str(out))
    return out

def average_frames(pet4d: Path, fr: tuple[int, int]) -> nib.Nifti1Image:
    img  = nib.load(str(pet4d))
    data = img.get_fdata(dtype=np.float32)[..., fr[0]: fr[1]].mean(axis=3)
    return nib.Nifti1Image(data, img.affine)

def suv_and_suvr(static_img: nib.Nifti1Image, json_sidecar: Path):
    """
    Compute SUVR directly from the static PET image.

    • We do **not** convert to SUV because the JSON lacks body-weight.
    • The original counts are saved as “raw” for completeness.
    """
    data = static_img.get_fdata(dtype=np.float32)

    # cerebellum mask in MNI space
    ref_mask = nib.load(str(REF_REGION_MASK)).get_fdata().astype(bool)
    mean_ref = data[ref_mask].mean()

    suvr = data / mean_ref

    raw_img  = nib.Nifti1Image(data,              static_img.affine)
    suvr_img = nib.Nifti1Image(suvr.astype(np.float32), static_img.affine)
    return raw_img, suvr_img

def register_to_mni(suvr_img: nib.Nifti1Image) -> ants.ANTsImage:
    tpl = ants.image_read(str(MNI_PET_TEMPLATE))
    mov = ants.from_nibabel(suvr_img)

    rigid = ants.registration(fixed=tpl, moving=mov, type_of_transform="Rigid")
    syn   = ants.registration(
        fixed=tpl, moving=rigid["warpedmovout"], type_of_transform="SyN"
    )
    return ants.apply_transforms(
        fixed=tpl, moving=mov, transformlist=syn["fwdtransforms"], interpolator="linear"
    )

def crop_nifti(img: nib.Nifti1Image, mask: nib.Nifti1Image,
               box: tuple[int,int,int]) -> nib.Nifti1Image:
    """
    Crop `img` to a fixed-size `box` centred on the center-of-mass of `mask`.
    Both img and mask must share the same affine and shape.
    """
    data = img.get_fdata(dtype=np.float32)
    m    = mask.get_fdata().astype(bool)
    com  = np.array(ndimage.center_of_mass(m))[[2,1,0]]  # x,y,z → i,j,k
    half = np.array(box) // 2

    start = (com - half).round().astype(int)
    start = np.clip(start, 0, np.array(data.shape) - box)

    # build slices
    sl = tuple(slice(start[d], start[d] + box[d]) for d in range(3))
    cropped = data[sl].astype(np.float32)

    return nib.Nifti1Image(cropped, img.affine)

def process_subject(subj_dir: Path):
    subj = subj_dir.name
    logging.info(f"=== {subj} ===")

    pet_nii  = next(subj_dir.glob("*.nii*"))
    pet_json = next(subj_dir.glob("*.json"))

    # 1. Motion-correct if 4-D
    if len(nib.load(str(pet_nii)).shape) == 4:
        pet_for_static = motion_correct_pet4d(pet_nii)
    else:
        pet_for_static = pet_nii

    # 2. Static average
    img = nib.load(str(pet_for_static))
    if len(img.shape) == 4:
        static_img = average_frames(pet_for_static, STATIC_FRAMES)
    else:
        static_img = img
    nib.save(static_img, subj_dir / "pet_static.nii.gz")

    # 3. Register to PET template (raw counts)
    warped = register_to_mni(static_img)
    ants.image_write(warped, str(subj_dir / "pet_mni_raw.nii.gz"))

    # 4. Compute SUVR in template space
    data_tpl = warped.to_nibabel().get_fdata(dtype=np.float32)
    ref_mask = nib.load(str(REF_REGION_MASK)).get_fdata().astype(bool)
    mean_ref = data_tpl[ref_mask].mean()
    suvr_tpl = data_tpl / mean_ref
    suvr_img = nib.Nifti1Image(suvr_tpl.astype(np.float32),
                               warped.to_nibabel().affine)
    nib.save(suvr_img, subj_dir / "pet_mni_suvr.nii.gz")

    # 5. Mask & crop the SUVR image (nibabel version)
    suvr_nib  = nib.Nifti1Image(suvr_tpl.astype(np.float32),warped.to_nibabel().affine)
    mask_nib  = nib.load(str(MNI_BRAIN_MASK))
    cropped   = crop_nifti(suvr_nib, mask_nib, CROP_SHAPE)
    nib.save(cropped, subj_dir / "pet_mni_crop.nii.gz")

    # 6. Move outputs to final dir
    final = STD_OUTPUT_ROOT / subj
    final.mkdir(parents=True, exist_ok=True)
    for f in subj_dir.glob("pet_*nii.gz"):
        f.replace(final / f.name)
    logging.info(f"→ outputs in {final}\n")

def main():
    for subj_dir in tqdm(sorted(PET_CONVERTED_ROOT.iterdir())):
        if subj_dir.is_dir():
            process_subject(subj_dir)

if __name__ == "__main__":
    main()
