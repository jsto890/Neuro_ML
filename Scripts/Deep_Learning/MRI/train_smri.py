# scripts/train_smri.py

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import csv
from datetime import datetime
import uuid
from sklearn.model_selection import KFold
import pandas as pd

from dataset import SMRIDataset
from models_smri import Simple3DCNN

def filter_labels(csv_path, labels):
    """Filter the CSV file to only include specified labels."""
    df = pd.read_csv(csv_path)
    filtered_df = df[df.iloc[:, 1].isin(labels)]
    return filtered_df

def get_label_description(labels):
    """Convert numeric labels to descriptive names."""
    label_map = {0: 'CN', 1: 'AD', 2: 'PD'}
    return ' vs '.join([label_map[label] for label in sorted(labels)])

def log_metrics(run_id, model_name, args, best_val_auc, best_val_acc, final_train_loss, final_train_acc, notes=""):
    """Log training metrics to CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get descriptive label names
    label_description = get_label_description(args.labels)
    
    # Prepare the row data
    row = {
        'run_id': run_id,
        'timestamp': timestamp,
        'model_name': model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'data_root': args.data_root,
        'checkpoint_dir': args.checkpoint_dir,
        'labels': label_description,
        'best_val_auc': best_val_auc,
        'best_val_acc': best_val_acc,
        'final_train_loss': final_train_loss,
        'final_train_acc': final_train_acc,
        'notes': notes
    }
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile('logging.csv')
    
    with open('logging.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def train_sMRI_model(model, train_loader, val_loader, epochs, device, checkpoint_dir, args):
    """
    Trains model; saves best checkpoint by validation AUC into checkpoint_dir.
    Returns the model loaded with best weights.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )

    model.to(device)
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_state = None
    final_train_loss = 0.0
    final_train_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for smri, labels in train_loader:
            smri, labels = smri.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(smri)              # [B, 2]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * smri.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_corrects / len(train_loader.dataset)
        
        if epoch == epochs:
            final_train_loss = epoch_loss
            final_train_acc = epoch_acc

        # --- Validation phase ---
        model.eval()
        val_logits = []
        val_labels = []

        with torch.no_grad():
            for smri, labels in val_loader:
                smri = smri.to(device)
                logits = model(smri)          # [B, 2]
                val_logits.append(logits.cpu().numpy())
                val_labels.append(labels.numpy())

        val_logits = np.concatenate(val_logits, axis=0)  # [N_val, 2]
        val_labels = np.concatenate(val_labels, axis=0)  # [N_val]

        # Convert logits → probabilities for all classes
        probs = nn.Softmax(dim=1)(torch.from_numpy(val_logits)).numpy()
        val_auc = roc_auc_score(val_labels, probs, multi_class='ovr')
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = accuracy_score(val_labels, val_preds)

        # Update learning rate based on validation AUC
        scheduler.step(val_auc)

        print(f"Epoch {epoch}/{epochs}  "
              f"Train loss={epoch_loss:.4f}, acc={epoch_acc:.4f}  "
              f"Val AUC={val_auc:.4f}, acc={val_acc:.4f}  "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")

        # Checkpoint if this is the best AUC so far
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(best_state, os.path.join(checkpoint_dir, "best_smri_model.pth"))
            print(f"  [Checkpoint] Saved new best model (AUC={val_auc:.4f})")

    # Load best model weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_auc, best_val_acc, final_train_loss, final_train_acc

def k_fold_training(args, k_folds=5):
    """
    Perform k-fold cross validation training.
    """
    # Filter the datasets based on labels
    train_df = filter_labels(args.train_csv, args.labels)
    val_df = filter_labels(args.val_csv, args.labels)
    
    # Create temporary CSV files for filtered data
    temp_train_csv = 'temp_train_filtered.csv'
    temp_val_csv = 'temp_val_filtered.csv'
    train_df.to_csv(temp_train_csv, index=False)
    val_df.to_csv(temp_val_csv, index=False)
    
    # Create dataset with filtered data
    dataset = SMRIDataset(csv_path=temp_train_csv, data_root=args.data_root)
    
    # Initialize k-fold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'\nFOLD {fold + 1}/{k_folds}')
        
        # Create data samplers for this fold
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        # Create data loaders for this fold
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers
        )
        
        # Initialize model for this fold
        model = Simple3DCNN(in_channels=1, base_channels=16, num_classes=len(args.labels))
        
        # Train model for this fold
        trained_model, best_val_auc, best_val_acc, final_train_loss, final_train_acc = train_sMRI_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            device=args.device,
            checkpoint_dir=os.path.join(args.checkpoint_dir, f'fold_{fold+1}'),
            args=args
        )
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'best_val_auc': best_val_auc,
            'best_val_acc': best_val_acc,
            'final_train_loss': final_train_loss,
            'final_train_acc': final_train_acc
        })
        
        # Log metrics for this fold
        run_id = str(uuid.uuid4())[:8]
        log_metrics(
            run_id=run_id,
            model_name="Simple3DCNN",
            args=args,
            best_val_auc=best_val_auc,
            best_val_acc=best_val_acc,
            final_train_loss=final_train_loss,
            final_train_acc=final_train_acc,
            notes=f"Fold {fold+1}/{k_folds}"
        )
    
    # Clean up temporary files
    os.remove(temp_train_csv)
    os.remove(temp_val_csv)
    
    # Calculate and print average results
    avg_auc = np.mean([r['best_val_auc'] for r in fold_results])
    avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
    print(f"\nK-Fold Cross Validation Results:")
    print(f"Average Validation AUC: {avg_auc:.4f}")
    print(f"Average Validation Accuracy: {avg_acc:.4f}")
    
    return fold_results

def main():
    parser = argparse.ArgumentParser(description="Train a 3D‐CNN on sMRI volumes")
    parser.add_argument("--train_csv",   type=str, required=True,
                        help="Path to train_labels.csv")
    parser.add_argument("--val_csv",     type=str, required=True,
                        help="Path to val_labels.csv")
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Folder containing sMRI NIfTIs, e.g. data/preprocessed/sMRI")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--k_folds",     type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--labels",      type=int, nargs='+', required=True,
                        help="Labels to include in training (e.g., 0 1 for CN vs AD)")
    args = parser.parse_args()

    # Perform k-fold cross validation
    fold_results = k_fold_training(args, k_folds=args.k_folds)

if __name__ == "__main__":
    main()
