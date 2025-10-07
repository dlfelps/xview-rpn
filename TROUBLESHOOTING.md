# Troubleshooting Guide

## Issue: RPN Not Detecting Objects After Training

### Symptoms
- Training loss doesn't improve (stays around 0.69)
- After training, the RPN produces proposals but with very low objectness scores (< 0.001)
- Visualizations show no detections

### Root Cause
The pretrained ResNet50 backbone (trained on ImageNet natural images) produces features that are poorly suited for xView satellite imagery. When the backbone is frozen, the RPN cannot learn meaningful object representations because:

1. **Domain mismatch**: Natural images (ImageNet) vs satellite imagery (xView) have very different characteristics
2. **Feature incompatibility**: The frozen backbone features don't align well with the object patterns in satellite imagery
3. **Training fails**: The RPN tries to learn on top of incompatible features, causing the loss to increase or stay flat

### Verification
You can verify this issue by checking:

1. **Objectness scores**: Trained model has scores ~0.0003 vs untrained ~0.5
2. **Loss trend**: Loss worsens or stays flat during training with frozen backbone
3. **Weight updates**: Weights update but loss doesn't improve

### Solution
**Unfreeze the backbone** during training to allow it to adapt to satellite imagery:

```bash
python -m faster.train \
    --data-dir data/train_images \
    --geojson data/xView_train.geojson \
    --epochs 10 \
    --batch-size 8 \
    --unfreeze-backbone \
    --lr 0.001 \
    --backbone-lr 0.0001 \
    --device cuda
```

Key parameters:
- `--unfreeze-backbone`: Allows backbone to adapt to satellite imagery
- `--backbone-lr 0.0001`: Lower learning rate for backbone (10x lower than RPN)
- `--lr 0.001`: Normal learning rate for RPN head

### Why This Works
- The backbone can adapt its features to satellite imagery characteristics
- The lower backbone learning rate prevents catastrophic forgetting of pretrained features
- The RPN can learn on top of features that actually represent satellite objects

### Expected Results
With unfrozen backbone:
- Loss should **decrease** during training
- Objectness scores should increase (e.g., > 0.5 for positive proposals)
- Visualizations should show meaningful proposals around objects

### Alternative Approaches
If you want to keep the backbone frozen, consider:
1. **Using a backbone pretrained on satellite imagery** (e.g., from remote sensing datasets)
2. **Pre-training the backbone** on xView in an unsupervised manner first
3. **Using a simpler backbone** designed for the target domain

## Data Verification
The data pipeline is working correctly:
- Bounding boxes align with objects
- Patch clipping handles edge cases properly
- Targets are in the correct format for PyTorch

You can verify with:
```bash
python -m faster.verify_data
```

This will visualize samples and check for data issues.
