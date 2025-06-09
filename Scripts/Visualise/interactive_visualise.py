from nibabel import load
from nibabel.viewers import OrthoSlicer3D

nii_file = "/Volumes//reseng202500013-ndd-ml/data/preprocessed/MRI/smriprep/sub-CLB00203/anat/sub-CLB00203_desc-preproc_T1w_brain_zscore.nii.gz"

img = load(nii_file)
OrthoSlicer3D(img.dataobj).show()