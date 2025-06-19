import nibabel as nib
from nibabel.orientations import aff2axcodes

# Load your original and reoriented files
orig = nib.load("/Volumes/FlaireHD/P4P/SPECT/CN/raw/sub-I246577_PPMI_SPECT_CN/sub-I246577_PPMI_SPECT_CN.nii")
reoriented = nib.load("/Volumes/FlaireHD/P4P/SPECT/CN/reoriented/sub-I246577_PPMI_SPECT_CN/sub-I246577_PPMI_SPECT_CN_RAS.nii.gz")

print("Original orientation:", aff2axcodes(orig.affine))
print("Reoriented orientation:", aff2axcodes(reoriented.affine))