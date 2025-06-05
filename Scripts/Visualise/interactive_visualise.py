from nibabel import load
from nibabel.viewers import OrthoSlicer3D

nii_file = "/Volumes/reseng202500013-ndd-ml/data/preprocessed/PET/ADNI/AD/sub-I10241161_ADNI_PET_AD/sub-I10241161_ADNI_PET_AD_SUVR_cropped.nii.gz"

img = load(nii_file)
OrthoSlicer3D(img.dataobj).show()