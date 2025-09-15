Fusion (Radiomics + Deep Learning)
=================================

This folder provides two fusion approaches:

1) Post-hoc stacking (embeddings/logits + radiomics → small meta-learner)
2) End-to-end late fusion (image branch + radiomics branch concatenated before the classifier)

Prereqs
-------
- Deep learning runs trained already (checkpoints available)
- Radiomics CSV available with columns: subject_id, label, <features...>
- Master CSV for the modality with columns: subject_id, label

Scripts
-------
- export_deep_features.py: Export per-subject CNN features (penultimate embedding) or logits.
- fuse_stack.py: Train a meta-learner on [deep features + radiomics] using k-fold.
- two_branch_late_fusion.py: Train an end-to-end two-branch model (CNN + MLP) with k-fold.

Usage Examples
--------------
1) Export deep features (embeddings) for PET Simple3DCNN:
```bash
python3 Scripts/Fusion/export_deep_features.py \
  --modality PET \
  --model Simple3DCNN \
  --checkpoint /home/jsto890/reseng202500013-ndd-ml/data/checkpoints_multi_pet/run_xxx/Simple3DCNN/best_pet_model_fold_1.pth \
  --csv /home/jsto890/reseng202500013-ndd-ml/data/pet_labels.csv \
  --data_root /home/jsto890/reseng202500013-ndd-ml/data/preprocessed \
  --out_csv /home/jsto890/reseng202500013-ndd-ml/data/deep_features_pet_fold1.csv
```

2) Stacking fusion (logistic regression):
```bash
python3 Scripts/Fusion/fuse_stack.py \
  --radiomics_csv /home/jsto890/reseng202500013-ndd-ml/data/radiomics_pet.csv \
  --deep_csv /home/jsto890/reseng202500013-ndd-ml/data/deep_features_pet_fold1.csv \
  --master_csv /home/jsto890/reseng202500013-ndd-ml/data/pet_labels.csv \
  --k_folds 5 --random_seed 42 \
  --out_dir /home/jsto890/reseng202500013-ndd-ml/data/fusion/stacking
```

3) End-to-end late fusion (CNN + MLP on radiomics):
```bash
python3 Scripts/Fusion/two_branch_late_fusion.py \
  --modality PET \
  --backbone Simple3DCNN \
  --master_csv /home/jsto890/reseng202500013-ndd-ml/data/pet_labels.csv \
  --radiomics_csv /home/jsto890/reseng202500013-ndd-ml/data/radiomics_pet.csv \
  --data_root /home/jsto890/reseng202500013-ndd-ml/data/preprocessed \
  --k_folds 5 --epochs 50 --batch_size 8 --device cuda:0 \
  --out_dir /home/jsto890/reseng202500013-ndd-ml/data/fusion/late
```

Notes
-----
- Stacking fusion is recommended as a first step; it is easier to validate and deploy.
- For late fusion, ensure your backbone returns (logits, fmap). Current PET/MRI backbones do this.
- All scripts expect absolute paths.


