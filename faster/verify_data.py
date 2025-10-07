"""Verify that dataset is loading images and targets correctly."""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from faster.data import XViewDataset

# Load dataset
print("Loading dataset...")
dataset = XViewDataset(
    image_dir='data/train_images',
    geojson_path='data/xView_train.geojson',
    patch_size=224,
)

print(f"Total patches: {len(dataset)}")

# Check several samples
print("\n=== CHECKING DATA SAMPLES ===")
for i in range(5):
    image, target = dataset[i]

    print(f"\nSample {i}:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Num boxes: {len(target['boxes'])}")
    print(f"  Boxes:\n{target['boxes']}")
    print(f"  Labels: {target['labels']}")

    # Check for invalid boxes
    boxes = target['boxes']
    if len(boxes) > 0:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        print(f"  Box widths: {widths}")
        print(f"  Box heights: {heights}")

        # Check for invalid boxes (negative width/height or out of bounds)
        if (widths <= 0).any():
            print("  ⚠️  WARNING: Some boxes have width <= 0!")
        if (heights <= 0).any():
            print("  ⚠️  WARNING: Some boxes have height <= 0!")
        if (boxes[:, 0] < 0).any() or (boxes[:, 1] < 0).any():
            print("  ⚠️  WARNING: Some boxes have negative coordinates!")
        if (boxes[:, 2] > 224).any() or (boxes[:, 3] > 224).any():
            print("  ⚠️  WARNING: Some boxes extend beyond image boundaries!")

# Visualize a few samples
print("\n=== VISUALIZING SAMPLES ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    image, target = dataset[i]

    # Convert image tensor to numpy for visualization
    img_np = image.permute(1, 2, 0).numpy()

    ax = axes[i]
    ax.imshow(img_np)
    ax.set_title(f'Sample {i}: {len(target["boxes"])} objects')
    ax.axis('off')

    # Draw boxes
    boxes = target['boxes'].numpy()
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = mpatches.Rectangle((x1, y1), width, height,
                                   linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

plt.tight_layout()
plt.savefig('data_verification.png', dpi=150, bbox_inches='tight')
print("Saved visualization to data_verification.png")

# Test with model forward pass
print("\n=== TESTING MODEL FORWARD PASS ===")
from faster.model import create_rpn_model

device = torch.device('cpu')
model = create_rpn_model(
    pretrained=False,
    freeze_backbone=True,
    freeze_roi_heads=True,
    min_size=224,
    anchor_sizes=(16, 32, 64),
    aspect_ratios=(0.5, 1.0, 2.0),
)
model.to(device)
model.train()

# Get a batch
from torch.utils.data import DataLoader
from faster.data import collate_fn

loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
images, targets = next(iter(loader))

print(f"Batch size: {len(images)}")
print(f"Image 0 shape: {images[0].shape}")
print(f"Target 0 boxes shape: {targets[0]['boxes'].shape}")

# Forward pass
images_on_device = [img.to(device) for img in images]
targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

# Test RPN forward pass
model.roi_heads.eval()
images_transformed, targets_transformed = model.transform(images_on_device, targets_on_device)
features = model.backbone(images_transformed.tensors)

print(f"\nTransformed image shape: {images_transformed.tensors.shape}")
print(f"Number of feature levels: {len(features)}")

with torch.no_grad():
    proposals, rpn_losses = model.rpn(images_transformed, features, targets_transformed)

print(f"\nRPN losses:")
print(f"  loss_objectness: {rpn_losses['loss_objectness']:.4f}")
print(f"  loss_rpn_box_reg: {rpn_losses['loss_rpn_box_reg']:.4f}")

print(f"\nProposals generated: {[len(p) for p in proposals]}")

print("\n✓ Data pipeline appears to be working!")
