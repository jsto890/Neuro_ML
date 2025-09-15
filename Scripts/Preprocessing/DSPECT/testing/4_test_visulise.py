import nibabel as nib
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Test script for SPECT brain masking visualization.")
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
args = parser.parse_args()

# --- Path Configuration ---
# Updated to use Desktop SPECT folders
data_root = "/Users/jacksonschofield/Desktop/SPECT"

print(f"INFO: Using data root: {data_root}")

# --- Find a subject to test ---
subject_id = None
try:
    # Use CN_SPECT_PPMI_NIfTI for testing
    masked_base_dir = os.path.join(data_root, "CN_SPECT_PPMI_NIfTI")
    for f in sorted(os.listdir(masked_base_dir)):
        if f.startswith('sub-'):
            subject_id = f
            break
except FileNotFoundError:
    print(f"❌ Could not find masked data directory. Looked in: {masked_base_dir}")
    exit(1)

if not subject_id:
    print(f"❌ No subject found in {masked_base_dir}")
    exit(1)

print(f"INFO: Testing with subject: {subject_id}")

# === File paths ===
# Use CN_SPECT_PPMI_NIfTI for both before and after
before_path = os.path.join(data_root, "CN_SPECT_PPMI_NIfTI", subject_id, f"{subject_id}_RAS.nii.gz")
after_path = os.path.join(masked_base_dir, subject_id, f"{subject_id}_RAS.nii.gz")

try:
    # === Load data ===
    before_img = nib.load(before_path)
    after_img = nib.load(after_path)
    before_data = before_img.get_fdata()
    after_data = after_img.get_fdata()

    # === Choose slice index to view ===
    z_index = before_data.shape[2] // 2  # Middle slice

    # === Plot ===
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    vmin, vmax = np.percentile(before_data[before_data > 0], [5, 95])

    ax1.imshow(before_data[:, :, z_index].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax1.set_title("Before Masking")
    ax1.axis("off")

    ax2.imshow(after_data[:, :, z_index].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax2.set_title("After Masking")
    ax2.axis("off")

    ax3.hist(before_data[before_data > 0].flatten(), bins=50, alpha=0.7, label='Before')
    ax3.hist(after_data[after_data > 0].flatten(), bins=50, alpha=0.7, label='After')
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.set_title('Intensity Distribution')

    before_coverage = np.count_nonzero(before_data) / before_data.size * 100
    after_coverage = np.count_nonzero(after_data) / after_data.size * 100
    
    ax4.text(0.1, 0.8, f'Before masking: {before_coverage:.2f}%', fontsize=12)
    ax4.text(0.1, 0.6, f'After masking: {after_coverage:.2f}%', fontsize=12)
    ax4.text(0.1, 0.4, f'Reduction: {before_coverage - after_coverage:.2f}%', fontsize=12)
    ax4.text(0.1, 0.2, f'Non-zero voxels: {np.count_nonzero(after_data):,}', fontsize=12)
    ax4.axis('off')
    ax4.set_title('Masking Statistics')

    fig.suptitle(f"SPECT Masking Quality Check: {subject_id}")
    plt.tight_layout()
    plt.show()

    print(f"\n✅ Masking validation:")
    if after_coverage < before_coverage and after_coverage > 5:
        print("   ✓ Masking appears successful")
    else:
        print("   ⚠️ Masking results may need review")

except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("Please ensure you have run steps 3 and 4.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")