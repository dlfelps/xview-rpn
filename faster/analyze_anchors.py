"""Analyze anchor matching to understand training issues."""

import torch
from faster.data import XViewDataset
from faster.model import create_rpn_model

# Load dataset
dataset = XViewDataset(
    image_dir='data/train_images',
    geojson_path='data/xView_train.geojson',
    patch_size=224,
)

# Load model
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
model.train()  # Training mode to get loss computation

print("=== ANALYZING ANCHOR MATCHING ===")
print(f"Anchor sizes: {(16, 32, 64)}")
print(f"Aspect ratios: {(0.5, 1.0, 2.0)}")

# Test on several samples with objects
num_samples = 10
total_positive_anchors = 0
total_negative_anchors = 0
total_objects = 0

for i in range(num_samples):
    image, target = dataset[i]

    if len(target['boxes']) == 0:
        continue

    total_objects += len(target['boxes'])

    # Get ground truth box sizes
    boxes = target['boxes']
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    print(f"\nSample {i}:")
    print(f"  Objects: {len(boxes)}")
    print(f"  Object sizes: w={widths.min():.0f}-{widths.max():.0f}, h={heights.min():.0f}-{heights.max():.0f}")

    # Run forward pass to see anchor matching
    images = [image.to(device)]
    targets = [{k: v.to(device) for k, v in target.items()}]

    with torch.no_grad():
        loss_dict = model(images, targets)

    # The objectness loss tells us about positive/negative balance
    print(f"  RPN objectness loss: {loss_dict['loss_objectness']:.4f}")
    print(f"  RPN box reg loss: {loss_dict['loss_rpn_box_reg']:.4f}")

# Compute average object sizes across dataset
print("\n=== DATASET-WIDE STATISTICS ===")
all_widths = []
all_heights = []
all_areas = []

for i in range(min(1000, len(dataset))):
    image, target = dataset[i]
    if len(target['boxes']) > 0:
        boxes = target['boxes']
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights

        all_widths.extend(widths.tolist())
        all_heights.extend(heights.tolist())
        all_areas.extend(areas.tolist())

all_widths = torch.tensor(all_widths)
all_heights = torch.tensor(all_heights)
all_areas = torch.tensor(all_areas)

print(f"Analyzed {len(all_widths)} objects from first 1000 patches:")
print(f"\nObject widths:")
print(f"  Min: {all_widths.min():.1f}")
print(f"  25th percentile: {all_widths.quantile(0.25):.1f}")
print(f"  Median: {all_widths.median():.1f}")
print(f"  75th percentile: {all_widths.quantile(0.75):.1f}")
print(f"  Max: {all_widths.max():.1f}")

print(f"\nObject heights:")
print(f"  Min: {all_heights.min():.1f}")
print(f"  25th percentile: {all_heights.quantile(0.25):.1f}")
print(f"  Median: {all_heights.median():.1f}")
print(f"  75th percentile: {all_heights.quantile(0.75):.1f}")
print(f"  Max: {all_heights.max():.1f}")

print(f"\nObject areas:")
print(f"  Min: {all_areas.min():.1f}")
print(f"  25th percentile: {all_areas.quantile(0.25):.1f}")
print(f"  Median: {all_areas.median():.1f}")
print(f"  75th percentile: {all_areas.quantile(0.75):.1f}")
print(f"  Max: {all_areas.max():.1f}")

print("\n=== ANCHOR SIZES VS OBJECT SIZES ===")
print("Current anchor areas:")
for size in (16, 32, 64):
    for ratio in (0.5, 1.0, 2.0):
        h = size
        w = size * ratio
        area = w * h
        print(f"  {w:.0f}x{h:.0f} = {area:.0f}")

print(f"\nRecommended anchor sizes based on data:")
print(f"  Small anchors (25th percentile): ~{all_areas.quantile(0.25).sqrt():.0f}")
print(f"  Medium anchors (median): ~{all_areas.median().sqrt():.0f}")
print(f"  Large anchors (75th percentile): ~{all_areas.quantile(0.75).sqrt():.0f}")
