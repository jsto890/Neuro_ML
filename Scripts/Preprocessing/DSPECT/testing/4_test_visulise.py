import nibabel as nib
import matplotlib.pyplot as plt
import os
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Test script for brain masking visualization.")
parser.add_argument("--isHasel", action="store_true", help="Set this flag if running on the Hasel server.")
args = parser.parse_args()

# --- Path Configuration ---
if args.isHasel:
    data_root = os.path.expanduser('~/reseng202500013-ndd-ml')
else:
    data_root = '/Volumes/reseng202500013-ndd-ml'

print(f"INFO: Using data root: {data_root}")

# --- Find a subject to test ---
subject_id = None
try:
    masked_base_dir = os.path.join(data_root, "data/preprocessed/SPECT/masked/CN")
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
before_path = os.path.join(data_root, "data/preprocessed/SPECT/registered/CN", subject_id, f"{subject_id}_registered.nii.gz")
after_path = os.path.join(masked_base_dir, subject_id, f"{subject_id}_masked.nii.gz")

try:
    # === Load data ===
    before_img = nib.load(before_path)
    after_img = nib.load(after_path)
    before_data = before_img.get_fdata()
    after_data = after_img.get_fdata()

    # === Choose slice index to view ===
    z_index = before_data.shape[2] // 2  # Middle slice

    # === Plot ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(before_data[:, :, z_index].T, cmap='gray', origin='lower')
    ax1.set_title("Before Masking")
    ax1.axis("off")

    ax2.imshow(after_data[:, :, z_index].T, cmap='gray', origin='lower')
    ax2.set_title("After Masking")
    ax2.axis("off")

    fig.suptitle(f"Brain Masking Quality Check: {subject_id}")
    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("Please ensure you have run steps 3 and 4.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")