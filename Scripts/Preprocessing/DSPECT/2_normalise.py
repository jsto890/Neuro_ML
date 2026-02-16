import os
import nibabel as nib
import numpy as np
import shutil
import yaml
import argparse

def fix_path(path):
    """Convert config path to actual mounted path"""
    return os.path.expanduser(path)

parser = argparse.ArgumentParser(description="Normalize SPECT images using reference region method.")
parser.add_argument("--diagnosis", type=str, choices=['CN', 'PD'], required=True, 
                    help="Diagnosis group to process (CN or PD)")
parser.add_argument("--method", type=str, choices=['reference', 'percentile', 'min_max'], default='reference')
args = parser.parse_args()

# Find config file in project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
config_path = os.path.join(project_root, 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Updated to use Desktop SPECT folder structure
input_root = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_reoriented"
output_root = f"/Users/jacksonschofield/Desktop/SPECT/{args.diagnosis}_SPECT_PPMI_normalised"
occipital_mask_path = "/Users/jacksonschofield/Desktop/P4P/Templates/SPECT/occipital_mask.nii.gz"

print(f"\n Processing {args.diagnosis} subjects")
print(f" Input directory: {input_root}")
print(f" Output directory: {output_root}")
print(f" Normalization method: {args.method}")
if args.method == 'reference':
    print(f" Reference mask: {occipital_mask_path}\n")

# Create output directory
os.makedirs(output_root, exist_ok=True)

def normalize_with_reference_region(img_img, mask_img):
    """Normalize using occipital reference region.
    img_img: nibabel Nifti1Image of the subject image (reoriented)
    mask_img: nibabel Nifti1Image of the reference region mask (template space)
    """
    from nilearn.image import resample_to_img

    try:
        # Resample mask to the image using real-world affines to ensure alignment
        resampled_mask = resample_to_img(
            mask_img,
            img_img,
            interpolation='nearest',
            copy_header=True,
            force_resample=True,
        )
        img_data = img_img.get_fdata()
        mask_data_resampled = resampled_mask.get_fdata() > 0

        reference_region = img_data[mask_data_resampled]
        if reference_region.size == 0:
            print("Warning: No voxels in reference region after resampling, using global stats (non-zero voxels)")
            non_zero = img_data[img_data > 0]
            if non_zero.size == 0:
                return img_data  # nothing to scale
            return (img_data - np.mean(non_zero)) / (np.std(non_zero) if np.std(non_zero) != 0 else 1)

        reference_mean = np.mean(reference_region)
        if reference_mean == 0:
            print("Warning: Reference region mean is zero, using global stats (non-zero voxels)")
            non_zero = img_data[img_data > 0]
            if non_zero.size == 0:
                return img_data
            return (img_data - np.mean(non_zero)) / (np.std(non_zero) if np.std(non_zero) != 0 else 1)

        return img_data / reference_mean

    except Exception as e:
        print(f"Warning: Mask resampling failed ({e}), using global stats (non-zero voxels)")
        img_data = img_img.get_fdata()
        non_zero = img_data[img_data > 0]
        if non_zero.size == 0:
            return img_data
        return (img_data - np.mean(non_zero)) / (np.std(non_zero) if np.std(non_zero) != 0 else 1)

def normalize_with_percentile(img_data, low_percentile=5, high_percentile=95):
    """Normalize using percentile clipping"""
    non_zero_data = img_data[img_data > 0]
    if len(non_zero_data) == 0:
        return img_data
    
    low_val = np.percentile(non_zero_data, low_percentile)
    high_val = np.percentile(non_zero_data, high_percentile)
    clipped = np.clip(img_data, low_val, high_val)
    return (clipped - low_val) / (high_val - low_val)

def normalize_min_max(img_data):
    non_zero_data = img_data[img_data > 0]
    if len(non_zero_data) == 0:
        return img_data
    min_val = np.min(non_zero_data)
    max_val = np.max(non_zero_data)
    return (img_data - min_val) / (max_val - min_val)

for subject in os.listdir(input_root):
    if subject.startswith("._") or subject == ".DS_Store":
        continue

    in_dir = os.path.join(input_root, subject)
    if not os.path.isdir(in_dir):
        continue
    
    out_dir = os.path.join(output_root, subject)
    os.makedirs(out_dir, exist_ok=True)

    # Copy only essential files (NIfTI + JSON, skip DICOM folders)
    print(f"[COPY] {subject}: Copying essential files")
    for item in os.listdir(in_dir):
        source_path = os.path.join(in_dir, item)
        dest_path = os.path.join(out_dir, item)
        
        if os.path.isfile(source_path):
            # Copy JSON files and the reoriented NIfTI
            if item.endswith('.json') or item == "1. reorient.nii.gz":
                shutil.copy2(source_path, dest_path)
                print(f"   [COPY] {item}")
        elif os.path.isdir(source_path) and not item.startswith('Original_DICOM'):
            # Copy only non-DICOM directories
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
            print(f"   [COPY] {item}/ (directory)")
        else:
            print(f"   [SKIP] {item} (DICOM folder - not needed)")

    # Process the reoriented NIfTI file
    input_path = os.path.join(in_dir, "1. reorient.nii.gz")
    output_path = os.path.join(out_dir, "2. normalised.nii.gz")

    if os.path.exists(input_path):
        print(f"Normalising {input_path} → {output_path}")
        try:
            img = nib.load(input_path)
            data = img.get_fdata()
            
            if args.method == 'reference':
                mask_img = nib.load(occipital_mask_path)
                norm_data = normalize_with_reference_region(img, mask_img)
            elif args.method == 'percentile':
                norm_data = normalize_with_percentile(data)
            elif args.method == 'min_max':
                norm_data = normalize_min_max(data)
            
            if np.std(norm_data) == 0:
                raise ValueError("Normalization resulted in zero variance")

            norm_img = nib.Nifti1Image(norm_data, img.affine, img.header)
            nib.save(norm_img, output_path)
            print(f" Successfully normalized {subject}")
            print(f"[STATS] {subject}: mean={np.mean(norm_data):.3f}, std={np.std(norm_data):.3f}")
        except Exception as e:
            print(f" Failed on {input_path}: {e}")
    else:
        print(f"️ {subject}: No reoriented file found at {input_path}")

print(f"\n Normalization complete! Output saved to: {output_root}")
print(f" Each subject now has a dedicated folder with normalized data and all original files.")