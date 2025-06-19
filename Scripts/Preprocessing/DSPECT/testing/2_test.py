import nibabel as nib
import numpy as np

# File paths
before_path = "/Volumes/FlaireHD/P4P/SPECT/CN/reoriented/sub-I246577_PPMI_SPECT_CN/sub-I246577_PPMI_SPECT_CN_RAS.nii.gz"
after_path = "/Volumes/FlaireHD/P4P/SPECT/CN/normalised/sub-I246577_PPMI_SPECT_CN/sub-I246577_PPMI_SPECT_CN_RAS.nii.gz"

def get_stats(path):
    img = nib.load(path)
    data = img.get_fdata()
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "shape": data.shape
    }

before_stats = get_stats(before_path)
after_stats = get_stats(after_path)

print(f"\n📊 Intensity Comparison for sub-I246577_PPMI_SPECT_CN_RAS.nii.gz\n")
print(f"{'Metric':<10} {'Before':>12} {'After':>12}")
print("-" * 38)
for key in before_stats:
    if isinstance(before_stats[key], tuple):  # For shape
        print(f"{key:<10} {str(before_stats[key]):>12} {str(after_stats[key]):>12}")
    else:
        print(f"{key:<10} {before_stats[key]:>12.5f} {after_stats[key]:>12.5f}")