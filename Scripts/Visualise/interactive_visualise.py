from nibabel import load
from nibabel.viewers import OrthoSlicer3D

nii_file = "/Volumes/reseng202500013-ndd-ml/data/raw/SPECT/PPMI/PD/sub-I248960_PPMI_SPECT_PD/sub-I248960_PPMI_SPECT_PD.nii"
#nii_file = "/Users/josephstorey/P4P/Templates/PET/FDG_PET.nii.gz"

img = load(nii_file)
OrthoSlicer3D(img.dataobj).show()