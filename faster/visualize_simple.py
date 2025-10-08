"""Simple visualization of RPN proposals on image chips."""

import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from faster.data import XViewDataset
from faster.model import create_rpn_model


def get_rpn_proposals_with_scores(model, image, device):
    """Get RPN proposals with their objectness scores directly."""
    with torch.no_grad():
        model.eval()

        # Transform image
        images = [image.to(device)]
        original_image_sizes = [img.shape[-2:] for img in images]

        # Get image transformed
        images_transformed, _ = model.transform(images, None)

        # Get features from backbone
        features = model.backbone(images_transformed.tensors)

        # Get RPN outputs (objectness and bbox deltas)
        objectness_logits, pred_bbox_deltas = model.rpn.head(list(features.values()))

        # Get anchors
        anchors = model.rpn.anchor_generator(images_transformed, list(features.values()))

        # Decode boxes using the RPN's internal method
        from torchvision.ops import boxes as box_ops

        # Process each image separately (we only have 1 image in this case)
        num_images = len(images)

        # Decode boxes per level and concatenate
        proposals_list = []
        scores_list = []

        for idx in range(num_images):
            proposals_per_level = []
            scores_per_level = []

            # Process each FPN level
            for obj_logits, box_reg in zip(objectness_logits, pred_bbox_deltas):
                # obj_logits: [batch, num_anchors, H, W]
                # box_reg: [batch, num_anchors * 4, H, W]

                N, A, H, W = obj_logits.shape

                # Flatten spatial dimensions for this image
                # obj_logits: [N, A, H, W] -> [A, H, W] -> [H, W, A] -> [H*W*A]
                obj_logits_flat = obj_logits[idx].permute(1, 2, 0).reshape(-1)

                # box_reg: [N, A*4, H, W] -> [A*4, H, W] -> [A, 4, H, W] -> [H, W, A, 4] -> [H*W*A, 4]
                box_reg_flat = box_reg[idx].view(A, 4, H, W).permute(2, 3, 0, 1).reshape(-1, 4)

                proposals_per_level.append(box_reg_flat)
                scores_per_level.append(obj_logits_flat)

            # Concatenate all levels
            box_deltas_all = torch.cat(proposals_per_level, dim=0)  # [total_anchors, 4]
            objectness_all = torch.cat(scores_per_level, dim=0)  # [total_anchors]

            # Decode boxes - anchors[idx] is already [num_anchors, 4]
            # Decoder returns [num_anchors, num_images, 4], we want [num_anchors, 4]
            proposals_decoded = model.rpn.box_coder.decode(box_deltas_all, [anchors[idx]])
            proposals_idx = proposals_decoded[:, 0, :]  # [num_anchors, 4]

            # Clip to image bounds
            proposals_idx = box_ops.clip_boxes_to_image(
                proposals_idx, images_transformed.image_sizes[idx]
            )

            # Get objectness scores (sigmoid for probability)
            scores_idx = torch.sigmoid(objectness_all)

            # Filter by top N before NMS
            num_anchors = scores_idx.shape[0]
            pre_nms_top_n = min(model.rpn._pre_nms_top_n['testing'], num_anchors)
            top_scores, top_idx = scores_idx.topk(pre_nms_top_n)
            proposals_idx = proposals_idx[top_idx]

            # Apply NMS
            keep = box_ops.nms(proposals_idx, top_scores, model.rpn.nms_thresh)

            # Keep only post_nms_top_n
            keep = keep[: model.rpn._post_nms_top_n['testing']]

            proposals_list.append(proposals_idx[keep])
            scores_list.append(top_scores[keep])

        return proposals_list[0], scores_list[0]


def draw_boxes_on_image(image_tensor, boxes, scores, score_threshold=0.5, max_boxes=100):
    """Draw bounding boxes on image tensor."""
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
        draw.text((x1, max(0, y1 - 10)), f'{score:.2f}', fill='red')

    return image


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

    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load dataset
    print('Loading dataset...')
    dataset = XViewDataset(
        image_dir=args.data_dir,
        geojson_path=args.geojson,
        patch_size=224,
    )
    print(f'Total patches: {len(dataset)}')

    # Create model with correct number of classes
    print('Loading model...')
    model = create_rpn_model(
        pretrained=False,  # Don't load pretrained weights, we'll load from checkpoint
        freeze_backbone=False,  # Don't freeze so we can load trained weights
        freeze_roi_heads=False,  # Don't freeze so we can load ROI head weights
        min_size=224,
        anchor_sizes=(16, 32, 64),
        aspect_ratios=(0.5, 1.0, 2.0),
        num_classes=dataset.num_classes,  # Use dataset's number of classes
    )

    # Load checkpoint (loads full model: backbone + RPN + ROI heads)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model has {dataset.num_classes} classes")

    # Process samples
    print(f'\nGenerating proposals for {args.num_samples} samples...')

    for i in range(min(args.num_samples, len(dataset))):
        print(f'Processing sample {i+1}/{args.num_samples}...')

        # Get image and targets
        image, target = dataset[i]

        # Get RPN proposals with scores
        boxes, scores = get_rpn_proposals_with_scores(model, image, device)

        print(f'  Generated {len(boxes)} proposals, '
              f'{(scores >= args.score_threshold).sum()} above threshold {args.score_threshold}')

        # Create visualization with proposals
        img_with_proposals = draw_boxes_on_image(
            image, boxes, scores, args.score_threshold, args.max_boxes
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
        ax2.set_title(f'RPN Proposals (score ≥ {args.score_threshold})')
        ax2.axis('off')

        plt.tight_layout()

        # Save
        output_path = output_dir / f'sample_{i:03d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'  Saved to {output_path}')

    print(f'\nVisualization complete! Saved {args.num_samples} images to {output_dir}')


if __name__ == '__main__':
    main()
