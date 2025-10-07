"""Visualize RPN proposals on image chips."""

import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from faster.data import XViewDataset
from faster.model import create_rpn_model, RPNTrainer


def draw_boxes_on_image(image_tensor, boxes, scores, score_threshold=0.5, max_boxes=100):
    """Draw bounding boxes on image tensor.

    Args:
        image_tensor: Image tensor [C, H, W]
        boxes: Proposal boxes [N, 4] in (x1, y1, x2, y2) format
        scores: Objectness scores [N]
        score_threshold: Minimum score to display
        max_boxes: Maximum number of boxes to display

    Returns:
        PIL Image with boxes drawn
    """
    # Convert tensor to PIL Image
    image = F.to_pil_image(image_tensor.cpu())

    # Filter by score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    # Sort by score and take top N
    if len(scores) > max_boxes:
        top_indices = torch.argsort(scores, descending=True)[:max_boxes]
        boxes = boxes[top_indices]
        scores = scores[top_indices]

    # Draw boxes
    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, y1 - 10), f'{score:.2f}', fill='red')

    return image


def visualize_proposals(
    checkpoint_path,
    data_dir,
    geojson_path,
    output_dir,
    num_samples=10,
    score_threshold=0.5,
    max_boxes=100,
    device='cpu',
):
    """Visualize RPN proposals on sample images.

    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to image directory
        geojson_path: Path to GeoJSON annotations
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        score_threshold: Minimum objectness score to display
        max_boxes: Maximum number of boxes per image
        device: Device to use (cpu or cuda)
    """
    # Setup device
    device = torch.device(device)
    print(f'Using device: {device}')

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load dataset
    print('Loading dataset...')
    dataset = XViewDataset(
        image_dir=data_dir,
        geojson_path=geojson_path,
        patch_size=224,
    )
    print(f'Total patches: {len(dataset)}')

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Process samples
    print(f'\nGenerating proposals for {num_samples} samples...')

    for i in range(min(num_samples, len(dataset))):
        print(f'Processing sample {i+1}/{num_samples}...')

        # Get image and targets
        image, target = dataset[i]

        # Generate RPN proposals (bypass ROI head which filters everything)
        with torch.no_grad():
            model.eval()

            # Transform and get features
            images_transformed, _ = model.transform([image.to(device)], None)
            features = model.backbone(images_transformed.tensors)

            # Get objectness scores and box predictions
            objectness_logits, pred_bbox_deltas = model.rpn.head(list(features.values()))

            # Get anchors
            anchors = model.rpn.anchor_generator(images_transformed, list(features.values()))

            # Decode proposals and get scores (mimic RPN's filter_proposals)
            from torchvision.models.detection.rpn import concat_box_prediction_layers
            objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness_logits, pred_bbox_deltas)

            num_images = len(anchors)
            num_anchors_per_level = [o[0].numel() for o in objectness_logits]
            objectness_prob = torch.sigmoid(objectness.detach())

            # Apply box decoding
            from torchvision.ops import boxes as box_ops

            for idx in range(num_images):
                objectness_prob_idx = objectness_prob[idx]
                pred_bbox_deltas_idx = pred_bbox_deltas[idx]
                anchors_idx = anchors[idx]

                # Decode boxes using box coder
                proposals_idx = model.rpn.box_coder.decode(pred_bbox_deltas_idx.unsqueeze(0), [anchors_idx])[0]

                # Clip to image
                proposals_idx = box_ops.clip_boxes_to_image(proposals_idx, images_transformed.image_sizes[idx])

                # Get top proposals before NMS
                top_n = model.rpn._pre_nms_top_n['testing']
                objectness_prob_idx, top_idx = objectness_prob_idx.topk(min(top_n, len(objectness_prob_idx)))
                proposals_idx = proposals_idx[top_idx]

                # Apply NMS
                keep = box_ops.nms(proposals_idx, objectness_prob_idx, model.rpn.nms_thresh)
                keep = keep[: model.rpn._post_nms_top_n['testing']]

                boxes = proposals_idx[keep]
                scores = objectness_prob_idx[keep]

        print(f'  Generated {len(boxes)} proposals, '
              f'{(scores >= score_threshold).sum()} above threshold {score_threshold}')

        # Create visualization with proposals
        img_with_proposals = draw_boxes_on_image(
            image, boxes, scores, score_threshold, max_boxes
        )

        # Create visualization with ground truth
        img_with_gt = F.to_pil_image(image.cpu())
        draw = ImageDraw.Draw(img_with_gt)
        gt_boxes = target['boxes'].cpu().numpy()
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)

        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(img_with_gt)
        ax1.set_title(f'Ground Truth ({len(gt_boxes)} objects)')
        ax1.axis('off')

        ax2.imshow(img_with_proposals)
        ax2.set_title(f'RPN Proposals (score ≥ {score_threshold})')
        ax2.axis('off')

        plt.tight_layout()

        # Save
        output_path = output_dir / f'sample_{i:03d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'  Saved to {output_path}')

    print(f'\nVisualization complete! Saved {num_samples} images to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Visualize RPN proposals on xView images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/train_images',
                        help='Path to image directory')
    parser.add_argument('--geojson', type=str, default='data/xView_train.geojson',
                        help='Path to GeoJSON annotations')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Minimum objectness score to display')
    parser.add_argument('--max-boxes', type=int, default=100,
                        help='Maximum number of boxes to display per image')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

    visualize_proposals(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        geojson_path=args.geojson,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        score_threshold=args.score_threshold,
        max_boxes=args.max_boxes,
        device=args.device,
    )


if __name__ == '__main__':
    main()
