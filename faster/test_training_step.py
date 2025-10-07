"""Test if training actually updates the model weights."""

import torch
from torch.utils.data import DataLoader
from faster.data import XViewDataset, collate_fn
from faster.model import create_rpn_model, RPNTrainer

# Load dataset
dataset = XViewDataset(
    image_dir='data/train_images',
    geojson_path='data/xView_train.geojson',
    patch_size=224,
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

# Create model
device = torch.device('cpu')
model = create_rpn_model(
    pretrained=True,
    freeze_backbone=True,
    freeze_roi_heads=True,
    min_size=224,
    anchor_sizes=(16, 32, 64),
    aspect_ratios=(0.5, 1.0, 2.0),
)
model.to(device)

# Save initial weights
initial_cls_weight = model.rpn.head.cls_logits.weight.data.clone()
initial_bbox_weight = model.rpn.head.bbox_pred.weight.data.clone()

print("=== INITIAL STATE ===")
print(f"Initial cls weight mean: {initial_cls_weight.mean():.6f}")
print(f"Initial bbox weight mean: {initial_bbox_weight.mean():.6f}")

# Create optimizer with different learning rates
for lr in [0.001, 0.01]:
    print(f"\n=== TESTING LR={lr} ===")

    # Reset model
    test_model = create_rpn_model(
        pretrained=True,
        freeze_backbone=True,
        freeze_roi_heads=True,
        min_size=224,
        anchor_sizes=(16, 32, 64),
        aspect_ratios=(0.5, 1.0, 2.0),
    )
    test_model.to(device)

    rpn_params = [p for p in test_model.rpn.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(rpn_params, lr=lr)
    trainer = RPNTrainer(test_model, optimizer, device)

    # Train for 10 steps
    losses = []
    obj_losses = []
    box_losses = []

    for i, (images, targets) in enumerate(loader):
        if i >= 10:
            break

        loss_dict = trainer.train_step(images, targets)
        losses.append(loss_dict['rpn_loss'])
        obj_losses.append(loss_dict['loss_objectness'])
        box_losses.append(loss_dict['loss_rpn_box_reg'])

        if i == 0 or i == 9:
            print(f"  Step {i}: loss={loss_dict['rpn_loss']:.4f}, "
                  f"obj={loss_dict['loss_objectness']:.4f}, "
                  f"box={loss_dict['loss_rpn_box_reg']:.4f}")

    # Check weight changes
    final_cls_weight = test_model.rpn.head.cls_logits.weight.data
    final_bbox_weight = test_model.rpn.head.bbox_pred.weight.data

    cls_change = (final_cls_weight - initial_cls_weight).abs().mean().item()
    bbox_change = (final_bbox_weight - initial_bbox_weight).abs().mean().item()

    print(f"  Weight changes:")
    print(f"    cls_logits: {cls_change:.6f}")
    print(f"    bbox_pred: {bbox_change:.6f}")

    print(f"  Loss progression:")
    print(f"    Start: {losses[0]:.4f}")
    print(f"    End: {losses[-1]:.4f}")
    print(f"    Change: {losses[-1] - losses[0]:.4f}")
    print(f"    Trend: {'IMPROVING' if losses[-1] < losses[0] else 'WORSENING' if losses[-1] > losses[0] else 'FLAT'}")

# Test with frozen backbone vs unfrozen
print("\n=== TESTING WITH UNFROZEN BACKBONE ===")
unfrozen_model = create_rpn_model(
    pretrained=True,
    freeze_backbone=False,  # Unfreeze backbone
    freeze_roi_heads=True,
    min_size=224,
    anchor_sizes=(16, 32, 64),
    aspect_ratios=(0.5, 1.0, 2.0),
)
unfrozen_model.to(device)

# Count trainable parameters
backbone_params = sum(p.numel() for p in unfrozen_model.backbone.parameters() if p.requires_grad)
rpn_params_count = sum(p.numel() for p in unfrozen_model.rpn.parameters() if p.requires_grad)

print(f"Trainable backbone params: {backbone_params:,}")
print(f"Trainable RPN params: {rpn_params_count:,}")

# Train for a few steps
all_params = [p for p in unfrozen_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(all_params, lr=0.0001)  # Lower LR for backbone
trainer = RPNTrainer(unfrozen_model, optimizer, device)

losses = []
for i, (images, targets) in enumerate(loader):
    if i >= 5:
        break
    loss_dict = trainer.train_step(images, targets)
    losses.append(loss_dict['rpn_loss'])

print(f"With unfrozen backbone:")
print(f"  Start loss: {losses[0]:.4f}")
print(f"  End loss: {losses[-1]:.4f}")
print(f"  Change: {losses[-1] - losses[0]:.4f}")
