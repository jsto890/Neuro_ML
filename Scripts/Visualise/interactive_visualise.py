from nibabel import load
from nibabel.viewers import OrthoSlicer3D

nii_file = "/Volumes/reseng202500013-ndd-ml/data/preprocessed/PET/PPMI/sub-I372065_PPMI_PET_PD/sub-I372065_PPMI_PET_PD_SUVR_s2_maskSoft2.nii.gz"

img = load(nii_file)
OrthoSlicer3D(img.dataobj).show()