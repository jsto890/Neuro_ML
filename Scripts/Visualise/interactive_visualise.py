from nibabel import load
from nibabel.viewers import OrthoSlicer3D

nii_file = "/Volumes/reseng202500013-ndd-ml/data/preprocessed/PET/PPMI/PD/sub-I10261856_PPMI_PET_PD/sub-I10261852_PPMI_PET_PD_SUVR.nii.gz"
#nii_file = "/Users/josephstorey/P4P/Templates/PET/FDG_PET.nii.gz"

img = load(nii_file)
OrthoSlicer3D(img.dataobj).show()