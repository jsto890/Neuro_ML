from nibabel import load
from nibabel.viewers import OrthoSlicer3D

nii_file = "/Volumes/reseng202500013-ndd-ml/data/preprocessed/SPECT/normalised/CN/sub-I246577_PPMI_SPECT_CN/sub-I246577_PPMI_SPECT_CN_RAS.nii.gz"

img = load(nii_file)
OrthoSlicer3D(img.dataobj).show()
