"""Test if RPN is producing proposals."""

import torch
from faster.data import XViewDataset
from faster.model import create_rpn_model

# Load dataset
print("Loading dataset...")
dataset = XViewDataset(
    image_dir='data/train_images',
    geojson_path='data/xView_train.geojson',
    patch_size=224,
)

# Load model
print("Loading model...")
device = torch.device('cpu')
model = create_rpn_model(
    pretrained=False,
    freeze_backbone=True,
    freeze_roi_heads=True,
    min_size=224,
    anchor_sizes=(16, 32, 64),
    aspect_ratios=(0.5, 1.0, 2.0),
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.to(device)
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
print(f"Train loss: {checkpoint.get('train_loss', 'N/A')}")
print(f"Val loss: {checkpoint.get('val_loss', 'N/A')}")

# Get a sample
image, target = dataset[0]
print(f"\nTest image shape: {image.shape}")
print(f"Ground truth boxes: {len(target['boxes'])}")

# Test RPN directly
print("\n=== TESTING RPN DIRECTLY ===")
with torch.no_grad():
    model.eval()

    # Transform
    images_transformed, _ = model.transform([image.to(device)], None)
    print(f"Transformed image shape: {images_transformed.tensors.shape}")

    # Get features
    features = model.backbone(images_transformed.tensors)
    print(f"Number of feature maps: {len(features)}")

    # Get RPN outputs
    objectness_logits, pred_bbox_deltas = model.rpn.head(list(features.values()))
    print(f"Objectness logits: {len(objectness_logits)} levels")

    # Get proposals
    proposals, _ = model.rpn(images_transformed, features, None)
    print(f"\nNumber of proposals from RPN: {len(proposals[0])}")

    # Check objectness scores
    from torchvision.models.detection.rpn import concat_box_prediction_layers
    objectness, _ = concat_box_prediction_layers(objectness_logits, pred_bbox_deltas)
    objectness_probs = torch.sigmoid(objectness[0])

    print(f"\nObjectness score stats:")
    print(f"  Min: {objectness_probs.min():.4f}")
    print(f"  Max: {objectness_probs.max():.4f}")
    print(f"  Mean: {objectness_probs.mean():.4f}")
    print(f"  Median: {objectness_probs.median():.4f}")
    print(f"  Scores > 0.1: {(objectness_probs > 0.1).sum()}/{len(objectness_probs)}")
    print(f"  Scores > 0.3: {(objectness_probs > 0.3).sum()}/{len(objectness_probs)}")
    print(f"  Scores > 0.5: {(objectness_probs > 0.5).sum()}/{len(objectness_probs)}")
    print(f"  Scores > 0.7: {(objectness_probs > 0.7).sum()}/{len(objectness_probs)}")

# Test with untrained model for comparison
print("\n=== TESTING UNTRAINED MODEL FOR COMPARISON ===")
untrained_model = create_rpn_model(
    pretrained=True,
    freeze_backbone=True,
    freeze_roi_heads=True,
    min_size=224,
    anchor_sizes=(16, 32, 64),
    aspect_ratios=(0.5, 1.0, 2.0),
)
untrained_model.to(device)
untrained_model.eval()

with torch.no_grad():
    images_transformed, _ = untrained_model.transform([image.to(device)], None)
    features = untrained_model.backbone(images_transformed.tensors)
    objectness_logits, pred_bbox_deltas = untrained_model.rpn.head(list(features.values()))

    objectness, _ = concat_box_prediction_layers(objectness_logits, pred_bbox_deltas)
    objectness_probs = torch.sigmoid(objectness[0])

    print(f"UNTRAINED Objectness score stats:")
    print(f"  Min: {objectness_probs.min():.4f}")
    print(f"  Max: {objectness_probs.max():.4f}")
    print(f"  Mean: {objectness_probs.mean():.4f}")
    print(f"  Median: {objectness_probs.median():.4f}")
