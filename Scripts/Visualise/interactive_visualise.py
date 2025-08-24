from nibabel import load
from nibabel.viewers import OrthoSlicer3D

nii_file = "/Volumes/reseng202500013-ndd-ml/data/Final_SPECT/CN_SPECT_PPMI_postprocessed/Subject_3204/5. finalised.nii.gz"

img = load(nii_file)
OrthoSlicer3D(img.dataobj).show()