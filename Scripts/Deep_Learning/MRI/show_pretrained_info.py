#!/usr/bin/env python3
"""
Show Pretrained Model Information
=================================

Displays information about available pretrained models for sMRI classification.
"""

from models_smri import get_pretrained_model_info

def main():
    print("="*80)
    print("PRETRAINED MODEL INFORMATION FOR SMRI CLASSIFICATION")
    print("="*80)
    
    info = get_pretrained_model_info()
    
    for model_name, details in info.items():
        print(f"\n📊 {model_name}")
        print("-" * 50)
        print(f"Source:           {details['source']}")
        print(f"Pretrained on:    {details['pretrained_on']}")
        print(f"Input size:       {details['input_size']}")
        print(f"Pretrained:       {'✅ Yes' if details['pretrained_available'] else '❌ No'}")
        print(f"Notes:            {details['notes']}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    print("\n# Train all models with pretrained weights:")
    print("python train_smri.py --master_csv ~/data/mri_labels.csv \\")
    print("    --data_root ~/data/preprocessed/MRI \\")
    print("    --labels 0 1 --use_pretrained")
    
    print("\n# Train specific models with pretrained weights:")
    print("python train_smri.py --master_csv ~/data/mri_labels.csv \\")
    print("    --data_root ~/data/preprocessed/MRI \\")
    print("    --labels 0 1 --models ResNet50_3D DenseNet121_3D --use_pretrained")
    
    print("\n# Train single model with pretrained weights:")
    print("python train_smri.py --master_csv ~/data/mri_labels.csv \\")
    print("    --data_root ~/data/preprocessed/MRI \\")
    print("    --labels 0 1 --model ResNet50_3D --use_pretrained")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n🎯 For best performance:")
    print("  • Use --use_pretrained flag for ResNet, DenseNet, and EfficientNet")
    print("  • MONAI pretrained models are specifically trained on medical imaging")
    print("  • ResNet50_3D often performs better than ResNet18_3D")
    print("  • DenseNet121_3D has good feature reuse capabilities")
    
    print("\n⚡ For faster training:")
    print("  • Use ResNet18_3D instead of ResNet50_3D")
    print("  • Reduce batch size if memory is limited")
    print("  • Use fewer epochs with pretrained models")
    
    print("\n🔬 For research comparison:")
    print("  • Train both pretrained and from-scratch versions")
    print("  • Compare performance to measure transfer learning benefits")
    print("  • Use same random seed for fair comparison")

if __name__ == "__main__":
    main() 