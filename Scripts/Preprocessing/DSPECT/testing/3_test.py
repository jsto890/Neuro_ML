import nibabel as nib
import numpy as np

# === CONFIG ===
subject_id = "sub-I246577_PPMI_SPECT_CN"
registered_path = f"/Volumes/reseng202500013-ndd-ml/data/preprocessed/SPECT/registered/CN/{subject_id}/{subject_id}_registered.nii.gz"
template_path = "/Users/jacksonschofield/Desktop/P4P/Templates/SPECT/symFPCITtemplate_MNI_norm.nii"

print(f"\n🧪 Registration Quality Test for {subject_id}\n")

# === HELPERS ===
def get_image_info(path):
    img = nib.load(path)
    data = img.get_fdata()
    voxel_sizes = np.round(np.sqrt((img.affine[:3, :3] ** 2).sum(0)), 3)
    return {
        "shape": data.shape,
        "voxel_sizes": voxel_sizes,
        "nonzero_voxels": np.count_nonzero(data),
        "mean": np.mean(data),
        "max": np.max(data),
        "min": np.min(data)
    }

# === RUN ===
try:
    reg_info = get_image_info(registered_path)
    tmpl_info = get_image_info(template_path)

    print(f"{'Metric':<18} {'Registered':>18} {'Template':>18}")
    print("-" * 60)
    print(f"{'Shape':<18} {str(reg_info['shape']):>18} {str(tmpl_info['shape']):>18}")
    print(f"{'Voxel Sizes':<18} {str(reg_info['voxel_sizes']):>18} {str(tmpl_info['voxel_sizes']):>18}")
    print(f"{'Non-zero Voxels':<18} {reg_info['nonzero_voxels']:>18} {'-':>18}")
    print(f"{'Mean Intensity':<18} {reg_info['mean']:>18.4f} {'-':>18}")
    print(f"{'Max Intensity':<18} {reg_info['max']:>18.4f} {'-':>18}")
    print(f"{'Min Intensity':<18} {reg_info['min']:>18.4f} {'-':>18}")

    if reg_info["nonzero_voxels"] < 1000:
        print("\n⚠️ WARNING: Too few non-zero voxels. This might indicate a failed registration.")

except FileNotFoundError:
    print(f"❌ File not found: {registered_path}")
except Exception as e:
    print(f"❌ Error: {e}")