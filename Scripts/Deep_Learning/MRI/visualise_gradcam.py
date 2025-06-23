# scripts/visualize_gradcam.py

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib

from dataset import SMRIDataset
from models_smri import SMRI_GradCAM_3DCNN
from gradcam import compute_gradcam_3d, compute_gradcam_simple3d

def overlay_heatmap_on_slice(mri_slice, cam_slice, cmap="hot", alpha=0.4):
    """
    Given a 2D anatomical slice and a 2D Cam slice, return an overlaid figure.
    """
    plt.imshow(mri_slice.T, cmap="gray", origin="lower")
    plt.imshow(cam_slice.T, cmap=cmap, alpha=alpha, origin="lower")
    plt.axis("off")

def main():
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM on sMRI volumes")
    parser.add_argument("--val_csv",     type=str, required=True,
                        help="Validation CSV with columns [subject_id,label]")
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Folder with sMRI NIfTIs (same as train script)")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to best_smri_model.pth")
    parser.add_argument("--batch_size",  type=int, default=1,
                        help="How many subjects to process at once (set to 1 for Grad-CAM)")
    parser.add_argument("--device",      type=str, default="cuda",
                        help="\"cuda\" or \"cpu\"")
    parser.add_argument("--output_dir",  type=str, default="gradcam_outputs",
                        help="Where to save any saved figures or .nii heatmaps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load dataset
    val_dataset = SMRIDataset(csv_path=args.val_csv, data_root=args.data_root)
    # We use batch_size=1 so we get one volume at a time and can run Grad-CAM on it
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2)

    # 2) Instantiate and load model
    # Try to load as Simple3DCNN first (which is what was trained)
    try:
        from models_smri import Simple3DCNN
        model = Simple3DCNN(num_classes=2)  # Binary classification
        
        # Load the state dict
        state_dict = torch.load(args.checkpoint, map_location=args.device)
        
        # Extract the actual input size from the saved classifier weight
        classifier_weight = state_dict['classifier.0.weight']
        actual_input_size = classifier_weight.shape[1]
        
        # Update the classifier with the correct input size
        model.classifier[0] = torch.nn.Linear(actual_input_size, 256)
        model._initialized = True
        
        # Now load the state dict
        model.load_state_dict(state_dict)
        print("Loaded Simple3DCNN model successfully")
        
        # For GradCAM, we need to modify the model to expose the feature maps
        # Create a wrapper that exposes the last conv layer
        class GradCAMWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.features = model.features
                self.classifier = model.classifier
                
            def forward(self, x):
                # Get features from the last conv layer
                features = self.features(x)
                # Flatten and classify
                x = features.view(features.size(0), -1)
                logits = self.classifier(x)
                return logits, features
        
        model = GradCAMWrapper(model)
        
    except Exception as e:
        print(f"Failed to load as Simple3DCNN: {e}")
        print("Trying SMRI_GradCAM_3DCNN...")
        
        # Fallback to original GradCAM model
        model = SMRI_GradCAM_3DCNN(in_channels=1, base_channels=16, num_classes=2)
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint)
    
    model.to(args.device)
    model.eval()

    # 3) Loop through a few validation subjects
    for i, (smri, label) in enumerate(val_loader):
        """
        smri shape: [1, 1, D, H, W]
        label: tensor([0]) or tensor([1])
        """
        smri = smri.to(args.device)
        true_label = label.item()

        # 4) Compute Grad-CAM for the positive class (index=1)
        # Check if we're using the wrapped model
        if hasattr(model, 'model'):  # Wrapped model
            # Use a modified GradCAM function for Simple3DCNN
            cam_3d = compute_gradcam_simple3d(model, smri, target_class=1, device=args.device)
        else:  # Original GradCAM model
            cam_3d = compute_gradcam_3d(model, smri, target_class=1, device=args.device)
        # cam_3d: numpy array [D, H, W], normalized [0,1]

        # 5) Extract original sMRI volume for overlay
        smri_np = smri.squeeze().cpu().numpy()  # [D, H, W]

        # 6) Choose a slice to visualize (e.g., mid‐axial)
        D, H, W = smri_np.shape
        mid_ax = D // 2
        anat_slice = smri_np[mid_ax]    # [H, W]
        cam_slice  = cam_3d[mid_ax]     # [H, W]

        # 7) Plot and save to disk
        plt.figure(figsize=(6, 3))
        overlay_heatmap_on_slice(anat_slice, cam_slice)
        plt.title(f"Subject: {val_dataset.subjects[i]}, True={true_label}")
        output_png = os.path.join(args.output_dir, f"{val_dataset.subjects[i]}_gradcam.png")
        plt.savefig(output_png, bbox_inches="tight", dpi=150)
        plt.close()

        # 8) Optionally save the entire 3D heatmap as a NIfTI (use the same affine as input)
        #    We need the original affine → load from nibabel:
        nifti_path = os.path.join(args.data_root, f"{val_dataset.subjects[i]}.nii.gz")
        orig_nii = nib.load(nifti_path)
        heatmap_nii = nib.Nifti1Image(cam_3d, affine=orig_nii.affine)
        nib.save(heatmap_nii, os.path.join(args.output_dir, f"{val_dataset.subjects[i]}_gradcam.nii.gz"))

        # 9) Break after a few subjects (remove this break to run on all)
        if i >= 4:
            break

if __name__ == "__main__":
    main()
