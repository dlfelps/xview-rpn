# Full Faster R-CNN Training

## Overview
The training pipeline has been updated to train the **complete Faster R-CNN model** (backbone + RPN + ROI heads) with actual xView class labels.

## Changes Made

### 1. Data Pipeline ([data.py](faster/data.py))
- **Class labels**: Uses actual xView type_ids instead of binary labels
- **Class mapping**: Maps xView type_ids (non-sequential) to sequential 1-indexed labels
- **Caching**: Class mapping saved/loaded with cache
- **Filtering**: Complete objects only, < 64x64 pixels

### 2. Model ([model.py](faster/model.py))
- **Full training**: All components trainable (backbone, RPN, ROI heads)
- **Custom classes**: Supports xView's class count
- **Loss function**: Uses all 4 Faster R-CNN losses:
  - `loss_classifier`: Classification loss for detected objects
  - `loss_box_reg`: Bounding box regression for final detections
  - `loss_objectness`: RPN objectness (object vs background)
  - `loss_rpn_box_reg`: RPN bounding box regression

### 3. Training ([train.py](faster/train.py))
- **Optimizer**: SGD with momentum (standard for Faster R-CNN)
- **Learning rate**: Single LR for all components
- **Logging**: Displays all loss components

## Usage

### Rebuild Cache (Required)
The cache needs to be rebuilt to include class mappings:
```bash
python -m faster.train --force-rebuild-cache --data-dir data/train_images --geojson data/xView_train.geojson --epochs 1 --batch-size 2
```

### Full Training
```bash
python -m faster.train \
    --force-rebuild-cache \
    --data-dir data/train_images \
    --geojson data/xView_train.geojson \
    --epochs 10 \
    --batch-size 8 \
    --lr 0.005 \
    --device cuda
```

### Training Parameters
- `--lr 0.005`: Standard learning rate for Faster R-CNN with SGD+momentum
- `--batch-size 8`: Adjust based on GPU memory
- `--epochs 10`: Start with 10, may need more for convergence
- `--force-rebuild-cache`: Required first time to build class mappings

## Expected Output

### During Training
```
Loaded 61084 patches from cache
Loaded 62 classes
Number of classes: 63 (including background)
Training full Faster R-CNN (backbone + RPN + ROI heads)
Total trainable parameters: 41,755,286

Epoch 1/10
Train - Total: 2.456, Cls: 1.234, Box: 0.567, RPN: 0.456 + 0.199
Val   - Total: 2.123, Cls: 1.056, Box: 0.478, RPN: 0.423 + 0.166
```

### Loss Components
- **Total**: Sum of all 4 losses
- **Cls**: Classification loss (should decrease as model learns classes)
- **Box**: Box regression loss (should decrease as boxes become more accurate)
- **RPN**: Objectness + box regression (should decrease as RPN improves)

## Monitoring Training

### Good Training Signs
- All losses decreasing over epochs
- Validation loss following training loss
- Classification loss dropping significantly (learning classes)

### Bad Training Signs
- Losses increasing or exploding → Lower learning rate
- Classification loss stuck → Check class distribution
- Val loss much higher than train → Overfitting, need more data or regularization

## Extracting RPN

After training, you can extract just the RPN:
```python
# Load trained model
checkpoint = torch.load('checkpoints/best.pth')
model = create_rpn_model(num_classes=63)
model.load_state_dict(checkpoint['model_state_dict'])

# Save just RPN weights
rpn_state = {k: v for k, v in model.state_dict().items() if k.startswith('rpn.')}
torch.save({'rpn_state_dict': rpn_state}, 'rpn_only.pth')
```

## Troubleshooting

### "size mismatch for roi_heads.box_predictor"
The cache has old num_classes. Solution: use `--force-rebuild-cache`

### "CUDA out of memory"
Reduce `--batch-size` (try 4 or 2)

### "No objects found"
Check that cache was rebuilt with new filtering logic

### Losses not decreasing
- Try lower learning rate: `--lr 0.001`
- Check data quality with `python -m faster.verify_data`
- Ensure sufficient training data
