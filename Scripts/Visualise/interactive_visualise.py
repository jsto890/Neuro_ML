from nibabel import load
from nibabel.viewers import OrthoSlicer3D

nii_file = '/Volumes/reseng202500013-ndd-ml/data/preprocessed/NEWPET/ADNI/sub-I10938763_ADNI_PET_AD/sub-I10938763_ADNI_PET_AD_static_brain_MNI.nii.gz'
#nii_file = "/Users/josephstorey/P4P/Templates/PET/FDG_PET.nii.gz"

img = load(nii_file)
OrthoSlicer3D(img.dataobj).show()