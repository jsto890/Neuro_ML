import os
import SimpleITK as sitk
from radiomics import featureextractor

# Paths
base_path = "/nesi/project/uoa04358/test_data/mri/BRAINLAT"
groups = ["AD", "HC"]

# Radiomics extractor with default settings
extractor = featureextractor.RadiomicsFeatureExtractor()

def create_dummy_mask(image_path):
    image = sitk.ReadImage(image_path)
    mask = sitk.NotEqual(image, 0)  # Non-zero region = ROI
    return image, mask

# Loop through both AD and HC folders
for group in groups:
    group_path = os.path.join(base_path, group)
    for fname in os.listdir(group_path):
        if fname.endswith(".nii.gz"):
            image_path = os.path.join(group_path, fname)
            print(f"\n🔍 Extracting features from: {group}/{fname}")
            
            try:
                image, mask = create_dummy_mask(image_path)
                result = extractor.execute(image, mask)
                for key, val in result.items():
                    print(f"{key}: {val}")
            except Exception as e:
                print(f"❌ Error with {fname}: {e}")
