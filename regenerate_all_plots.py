#!/usr/bin/env python3
"""
Master script to regenerate all model evaluation plots with the new box plot format.
This script will find all trained models and regenerate their evaluation plots using
the updated plotting functions that include model name, image type, and box plots.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import glob

def find_checkpoint_directories(base_dir):
    """Find all checkpoint directories containing trained models."""
    checkpoint_dirs = []
    
    # Look for directories matching pattern: run_YYYYMMDD_HHMMSS
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    
    for run_dir in run_dirs:
        if os.path.isdir(run_dir):
            # Look for model subdirectories within each run
            for item in os.listdir(run_dir):
                item_path = os.path.join(run_dir, item)
                if os.path.isdir(item_path):
                    # Check if it contains model files
                    if any(f.endswith('.pth') for f in os.listdir(item_path)):
                        checkpoint_dirs.append(item_path)
    
    return sorted(checkpoint_dirs)

def get_image_type_from_path(model_dir):
    """Determine image type from the model directory path."""
    path_str = str(model_dir).lower()
    if 'mri' in path_str:
        return 'sMRI'
    elif 'pet' in path_str:
        return 'PET'
    elif 'spect' in path_str or 'dspect' in path_str:
        return 'SPECT'
    else:
        return 'Unknown'

def get_regenerate_script_path(image_type):
    """Get the path to the appropriate regenerate_plots.py script."""
    base_dir = Path(__file__).parent
    
    if image_type == 'sMRI':
        return base_dir / "Scripts" / "Deep_Learning" / "MRI" / "regenerate_plots.py"
    elif image_type == 'PET':
        return base_dir / "Scripts" / "Deep_Learning" / "PET" / "regenerate_plots.py"
    elif image_type == 'SPECT':
        return base_dir / "Scripts" / "Deep_Learning" / "DSPECT" / "regenerate_plots.py"
    else:
        return None

def regenerate_model_plots(model_dir, image_type, labels=None, dry_run=False):
    """Regenerate plots for a single model."""
    model_name = os.path.basename(model_dir.rstrip(os.sep))
    regenerate_script = get_regenerate_script_path(image_type)
    
    if not regenerate_script or not regenerate_script.exists():
        print(f"[WARNING] No regenerate script found for {image_type} at {regenerate_script}")
        return False
    
    # Build command
    cmd = [
        sys.executable, str(regenerate_script),
        "--model_dir", str(model_dir),
        "--model_name", model_name
    ]
    
    if labels:
        cmd.extend(["--labels"] + [str(l) for l in labels])
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Regenerating plots for {model_name} ({image_type})")
    print(f"  Model directory: {model_dir}")
    print(f"  Script: {regenerate_script}")
    print(f"  Command: {' '.join(cmd)}")
    
    if not dry_run:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(regenerate_script.parent))
            if result.returncode == 0:
                print(f"  ✅ Successfully regenerated plots")
                return True
            else:
                print(f"  ❌ Failed to regenerate plots")
                print(f"  Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ❌ Exception during regeneration: {e}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Regenerate all model evaluation plots with new box plot format")
    parser.add_argument("--checkpoint_dir", required=True, 
                        help="Base directory containing checkpoint subdirectories (e.g., ~/data/checkpoints_multi)")
    parser.add_argument("--image_type", choices=['sMRI', 'PET', 'SPECT', 'all'], default='all',
                        help="Image type to process (default: all)")
    parser.add_argument("--labels", nargs="+", type=int, default=[0, 1, 2],
                        help="Label set used for plot titles (default: [0, 1, 2] for CN, AD, PD)")
    parser.add_argument("--model_name", default=None,
                        help="Only regenerate plots for models with this name (e.g., Simple3DCNN)")
    parser.add_argument("--dry_run", action='store_true',
                        help="Show what would be done without actually running")
    parser.add_argument("--force", action='store_true',
                        help="Force regeneration even if new plots already exist")
    
    args = parser.parse_args()
    
    checkpoint_base = os.path.expanduser(args.checkpoint_dir)
    if not os.path.exists(checkpoint_base):
        print(f"Error: Checkpoint directory does not exist: {checkpoint_base}")
        sys.exit(1)
    
    print(f"🔍 Searching for trained models in: {checkpoint_base}")
    print(f"📊 Image type filter: {args.image_type}")
    print(f"🏷️  Labels: {args.labels}")
    if args.model_name:
        print(f"🤖 Model name filter: {args.model_name}")
    print()
    
    # Find all model directories
    model_dirs = find_checkpoint_directories(checkpoint_base)
    
    if not model_dirs:
        print("❌ No trained models found in checkpoint directory")
        sys.exit(1)
    
    print(f"Found {len(model_dirs)} model directories:")
    for model_dir in model_dirs:
        print(f"  - {model_dir}")
    print()
    
    # Filter by image type and model name
    filtered_dirs = []
    for model_dir in model_dirs:
        image_type = get_image_type_from_path(model_dir)
        model_name = os.path.basename(model_dir.rstrip(os.sep))
        
        # Apply filters
        if args.image_type != 'all' and image_type != args.image_type:
            continue
        if args.model_name and model_name != args.model_name:
            continue
            
        filtered_dirs.append((model_dir, image_type, model_name))
    
    if not filtered_dirs:
        print(f"❌ No models found matching filters")
        sys.exit(1)
    
    print(f"📋 Will process {len(filtered_dirs)} models:")
    for model_dir, image_type, model_name in filtered_dirs:
        print(f"  - {model_name} ({image_type})")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()
    
    # Process each model
    success_count = 0
    total_count = len(filtered_dirs)
    
    for i, (model_dir, image_type, model_name) in enumerate(filtered_dirs, 1):
        print(f"[{i}/{total_count}] Processing {model_name} ({image_type})")
        
        # Check if new plots already exist (unless force is specified)
        if not args.force:
            evaluation_dir = os.path.join(model_dir, "evaluation_plots")
            if os.path.exists(evaluation_dir):
                plot_file = os.path.join(evaluation_dir, "model_evaluation_analysis.png")
                if os.path.exists(plot_file):
                    print(f"  ⏭️  Skipping - new plots already exist (use --force to regenerate)")
                    success_count += 1
                    continue
        
        # Regenerate plots
        if regenerate_model_plots(model_dir, image_type, args.labels, args.dry_run):
            success_count += 1
    
    print(f"\n📊 SUMMARY")
    print(f"✅ Successfully processed: {success_count}/{total_count} models")
    print(f"❌ Failed: {total_count - success_count}/{total_count} models")
    
    if args.dry_run:
        print(f"\n💡 To actually regenerate the plots, run without --dry_run")
    
    print(f"\n🎉 All done! New box plot format evaluation plots have been generated.")
    print(f"   Look for 'model_evaluation_analysis.png' in each model's evaluation_plots/ directory.")

if __name__ == "__main__":
    main()
