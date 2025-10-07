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
    """Get RPN proposals with their objectness scores.

    This monkey-patches the RPN to capture proposals and scores.
    """
    proposals_out = []
    scores_out = []

    # Store original filter_proposals
    original_filter = model.rpn.filter_proposals

    def capture_filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level):
        """Wrapper to capture proposals and scores before filtering."""
        # Call original filter
        result = original_filter(proposals, objectness, image_shapes, num_anchors_per_level)

        # Capture the objectness scores for the filtered proposals
        # The filter_proposals returns boxes, but we need to track scores
        # Let's compute them from objectness
        for obj, prop in zip(objectness, result):
            # Get objectness probabilities
            scores = torch.sigmoid(obj).flatten()
            # Sort and take top N to match filtered proposals
            top_scores, _ = torch.sort(scores, descending=True)
            top_scores = top_scores[:len(prop)]
            scores_out.append(top_scores)

        proposals_out.extend(result)
        return result

    # Monkey patch
    model.rpn.filter_proposals = capture_filter_proposals

    # Run model
    with torch.no_grad():
        model.eval()
        _ = model([image.to(device)])

    # Restore original
    model.rpn.filter_proposals = original_filter

    if len(proposals_out) > 0 and len(scores_out) > 0:
        return proposals_out[0], scores_out[0]
    else:
        return torch.zeros((0, 4)), torch.zeros((0,))


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

    # Create model
    print('Loading model...')
    model = create_rpn_model(
        pretrained=False,  # Don't load pretrained weights, we'll load from checkpoint
        freeze_backbone=False,  # Don't freeze so we can load trained backbone weights
        freeze_roi_heads=True,
        min_size=224,
        anchor_sizes=(16, 32, 64),
        aspect_ratios=(0.5, 1.0, 2.0),
    )

    # Load checkpoint (loads both backbone and RPN weights)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Freeze after loading (for inference)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.rpn.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

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
