from nibabel import load
from nibabel.viewers import OrthoSlicer3D

nii_file = "/Volumes/reseng202500013-ndd-ml/data/interpret/sub-I1624206_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w_brain_zscore_gradcam_class0.nii.gz"

img = load(nii_file)
OrthoSlicer3D(img.dataobj).show()