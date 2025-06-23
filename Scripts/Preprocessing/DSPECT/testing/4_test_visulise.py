import nibabel as nib
import matplotlib.pyplot as plt

# === File paths ===
before_path = "~/reseng202500013-ndd-ml/data/preprocessed/SPECT/registered/CN/sub-I246577_PPMI_SPECT_CN/sub-I246577_PPMI_SPECT_CN_registered.nii.gz"
after_path = "~/reseng202500013-ndd-ml/data/preprocessed/SPECT/masked/CN/sub-I246577_PPMI_SPECT_CN/sub-I246577_PPMI_SPECT_CN_masked.nii.gz"

# === Load data ===
before_img = nib.load(before_path)
after_img = nib.load(after_path)
before_data = before_img.get_fdata()
after_data = after_img.get_fdata()

# === Choose slice index to view ===
z_index = before_data.shape[2] // 2  # Middle slice

# === Plot ===
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(before_data[:, :, z_index], cmap='gray')
plt.title("Before Masking")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(after_data[:, :, z_index], cmap='gray')
plt.title("After Masking")
plt.axis("off")

plt.suptitle("Brain Masking Quality Check: sub-I246577_PPMI_SPECT_CN")
plt.tight_layout()
plt.show()