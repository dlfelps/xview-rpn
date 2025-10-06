"""Evaluation script for RPN recall on xView dataset."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from faster.data import XViewDataset, collate_fn
from faster.model import create_rpn_model, RPNTrainer


def compute_iou(box1, box2):
    """Compute IoU between two boxes [xmin, ymin, xmax, ymax]."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Compute recall: fraction of GT boxes matched by predictions.

    Args:
        gt_boxes: Ground truth boxes (N, 4)
        pred_boxes: Predicted boxes (M, 4)
        iou_threshold: IoU threshold for matching

    Returns:
        recall: Recall value
        num_matched: Number of matched GT boxes
    """
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0, 0

    if len(pred_boxes) == 0:
        return 0.0, 0

    num_matched = 0

    for gt_box in gt_boxes:
        # Check if any prediction matches this GT box
        matched = False
        for pred_box in pred_boxes:
            iou = compute_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                matched = True
                break

        if matched:
            num_matched += 1

    recall = num_matched / len(gt_boxes)
    return recall, num_matched


def evaluate_rpn(trainer, dataloader, iou_thresholds=[0.5, 0.75]):
    """
    Evaluate RPN recall.

    Args:
        trainer: RPNTrainer instance
        dataloader: DataLoader for evaluation
        iou_thresholds: List of IoU thresholds to evaluate

    Returns:
        Dictionary of recall metrics
    """
    total_gt_boxes = 0
    matched_boxes = {iou_th: 0 for iou_th in iou_thresholds}

    pbar = tqdm(dataloader, desc='Evaluating RPN')

    for images, targets in pbar:
        # Generate proposals
        outputs = trainer.generate_proposals(images)

        # Process each image in batch
        for output, target in zip(outputs, targets):
            gt_boxes = target['boxes'].cpu().numpy()
            pred_boxes = output['boxes'].cpu().numpy()

            total_gt_boxes += len(gt_boxes)

            # Compute recall at different IoU thresholds
            for iou_th in iou_thresholds:
                _, num_matched = compute_recall(gt_boxes, pred_boxes, iou_th)
                matched_boxes[iou_th] += num_matched

        # Update progress bar
        current_recalls = {
            f'R@{iou_th}': matched_boxes[iou_th] / total_gt_boxes if total_gt_boxes > 0 else 0.0
            for iou_th in iou_thresholds
        }
        pbar.set_postfix(current_recalls)

    # Compute final recalls
    recalls = {
        f'recall@{iou_th}': matched_boxes[iou_th] / total_gt_boxes if total_gt_boxes > 0 else 0.0
        for iou_th in iou_thresholds
    }
    recalls['total_gt_boxes'] = total_gt_boxes

    return recalls


def main():
    parser = argparse.ArgumentParser(description='Evaluate RPN recall on xView dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/train_images',
                        help='Path to image directory')
    parser.add_argument('--geojson', type=str, default='data/xView_train.geojson',
                        help='Path to GeoJSON annotations')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--iou-thresholds', type=float, nargs='+', default=[0.5, 0.75],
                        help='IoU thresholds for recall computation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

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

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Create model
    print('Loading model...')
    model = create_rpn_model(
        pretrained=False,
        freeze_backbone=True,
        freeze_roi_heads=True,
        min_size=224,
        anchor_sizes=(16, 32, 64),
        aspect_ratios=(0.5, 1.0, 2.0),
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create trainer (optimizer not needed for eval)
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer
    trainer = RPNTrainer(model, optimizer, device)

    # Evaluate
    print('\nEvaluating RPN recall...')
    metrics = evaluate_rpn(trainer, dataloader, args.iou_thresholds)

    # Print results
    print('\n' + '='*50)
    print('RPN Recall Results')
    print('='*50)
    print(f"Total GT boxes: {metrics['total_gt_boxes']}")
    for key, value in metrics.items():
        if key != 'total_gt_boxes':
            print(f"{key}: {value:.4f}")
    print('='*50)


if __name__ == '__main__':
    main()
