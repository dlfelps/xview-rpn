"""Check if RPN parameters are actually being updated during training."""

import torch
from faster.data import XViewDataset, collate_fn
from faster.model import create_rpn_model, RPNTrainer
from torch.utils.data import DataLoader

# Load dataset
dataset = XViewDataset(
    image_dir='data/train_images',
    geojson_path='data/xView_train.geojson',
    patch_size=224,
)

# Create small dataloader
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)

# Create model
device = torch.device('cpu')
model = create_rpn_model(
    pretrained=True,  # Start with pretrained to see if we can learn
    freeze_backbone=True,
    freeze_roi_heads=True,
    min_size=224,
    anchor_sizes=(16, 32, 64),
    aspect_ratios=(0.5, 1.0, 2.0),
)
model.to(device)

# Check trainable parameters
print("=== TRAINABLE PARAMETERS ===")
rpn_params = [p for p in model.rpn.parameters() if p.requires_grad]
print(f"Number of trainable RPN parameters: {len(rpn_params)}")
print(f"Total trainable RPN values: {sum(p.numel() for p in rpn_params):,}")

# Test different learning rates
for lr in [0.001, 0.01, 0.0001]:
    print(f"\n=== TESTING LR={lr} ===")

    # Reset model
    model_test = create_rpn_model(
        pretrained=True,
        freeze_backbone=True,
        freeze_roi_heads=True,
        min_size=224,
        anchor_sizes=(16, 32, 64),
        aspect_ratios=(0.5, 1.0, 2.0),
    )
    model_test.to(device)

    # Save initial weights
    initial_weight = model_test.rpn.head.cls_logits.weight.data.clone()

    # Create optimizer
    rpn_params = [p for p in model_test.rpn.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(rpn_params, lr=lr)

    # Train for a few steps
    trainer = RPNTrainer(model_test, optimizer, device)

    losses = []
    for i, (images, targets) in enumerate(loader):
        if i >= 5:  # Just 5 steps
            break

        loss_dict = trainer.train_step(images, targets)
        losses.append(loss_dict['rpn_loss'])

    # Check if weights changed
    final_weight = model_test.rpn.head.cls_logits.weight.data
    weight_change = (final_weight - initial_weight).abs().mean().item()

    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss change: {losses[-1] - losses[0]:.4f}")
    print(f"  Weight change: {weight_change:.6f}")
    print(f"  Weights updated: {'YES' if weight_change > 1e-6 else 'NO'}")
