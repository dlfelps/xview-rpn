"""Training script for RPN on xView dataset."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from faster.data import XViewDataset, collate_fn
from faster.model import create_rpn_model, RPNTrainer


def train_epoch(trainer, dataloader, epoch):
    """Train for one epoch."""
    total_loss = 0
    total_obj_loss = 0
    total_box_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        losses = trainer.train_step(images, targets)

        total_loss += losses['rpn_loss']
        total_obj_loss += losses['loss_objectness']
        total_box_loss += losses['loss_rpn_box_reg']

        pbar.set_postfix({
            'loss': f"{losses['rpn_loss']:.4f}",
            'obj': f"{losses['loss_objectness']:.4f}",
            'box': f"{losses['loss_rpn_box_reg']:.4f}",
        })

    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'loss_objectness': total_obj_loss / num_batches,
        'loss_rpn_box_reg': total_box_loss / num_batches,
    }


def validate(trainer, dataloader):
    """Validate model."""
    total_loss = 0
    total_obj_loss = 0
    total_box_loss = 0

    pbar = tqdm(dataloader, desc='Validation')
    for images, targets in pbar:
        losses = trainer.eval_step(images, targets)

        total_loss += losses['rpn_loss']
        total_obj_loss += losses['loss_objectness']
        total_box_loss += losses['loss_rpn_box_reg']

        pbar.set_postfix({
            'loss': f"{losses['rpn_loss']:.4f}",
        })

    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'loss_objectness': total_obj_loss / num_batches,
        'loss_rpn_box_reg': total_box_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train RPN on xView dataset')
    parser.add_argument('--data-dir', type=str, default='data/train_images',
                        help='Path to image directory')
    parser.add_argument('--geojson', type=str, default='data/xView_train.geojson',
                        help='Path to GeoJSON annotations')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # Create dataset
    print('Loading dataset...')
    dataset = XViewDataset(
        image_dir=args.data_dir,
        geojson_path=args.geojson,
        patch_size=224,
    )
    print(f'Total patches: {len(dataset)}')

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f'Train patches: {len(train_dataset)}, Val patches: {len(val_dataset)}')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Create model
    print('Creating model...')
    model = create_rpn_model(
        pretrained=True,
        freeze_backbone=True,
        freeze_roi_heads=True,
        min_size=224,
        anchor_sizes=(16, 32, 64),
        aspect_ratios=(0.5, 1.0, 2.0),
    )
    model.to(device)

    # Create optimizer (only for RPN parameters)
    rpn_params = [p for p in model.rpn.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(rpn_params, lr=args.lr)

    # Create trainer
    trainer = RPNTrainer(model, optimizer, device)

    # Training loop
    print('Starting training...')
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')

        # Train
        train_metrics = train_epoch(trainer, train_loader, epoch)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Obj: {train_metrics['loss_objectness']:.4f}, "
              f"Box: {train_metrics['loss_rpn_box_reg']:.4f}")

        # Validate
        val_metrics = validate(trainer, val_loader)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Obj: {val_metrics['loss_objectness']:.4f}, "
              f"Box: {val_metrics['loss_rpn_box_reg']:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
        }

        # Save latest
        torch.save(checkpoint, save_dir / 'latest.pth')

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(checkpoint, save_dir / 'best.pth')
            print(f'Saved best model with val loss: {best_val_loss:.4f}')

    print('\nTraining complete!')


if __name__ == '__main__':
    main()
