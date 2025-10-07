"""RPN model configuration for xView training."""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator


def create_rpn_model(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    freeze_roi_heads: bool = False,
    min_size: int = 224,
    anchor_sizes: tuple = (16, 32, 64),
    aspect_ratios: tuple = (0.5, 1.0, 2.0),
    num_classes: int = 91,  # Default COCO classes, will be replaced
):
    """
    Create Faster R-CNN model for full training (backbone + RPN + ROI heads).

    Args:
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone parameters
        freeze_roi_heads: Freeze ROI head parameters
        min_size: Minimum input image size
        anchor_sizes: RPN anchor sizes
        aspect_ratios: RPN anchor aspect ratios
        num_classes: Number of classes (including background at index 0)

    Returns:
        Faster R-CNN model for full training
    """
    # Configure custom RPN anchors
    anchor_generator = AnchorGenerator(
        sizes=tuple([anchor_sizes] * 5),  # 5 feature maps in FPN
        aspect_ratios=tuple([aspect_ratios] * 5),
    )

    # Load pretrained Faster R-CNN with custom anchor generator and num_classes
    # First create model without weights to get correct dimensions
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        min_size=min_size,
        max_size=min_size * 2,
        rpn_anchor_generator=anchor_generator,
        num_classes=num_classes,
    )

    # Load pretrained weights if requested (skip incompatible RPN head and ROI box predictor)
    if pretrained:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True, check_hash=True)

        # Remove weights that don't match our custom configuration
        keys_to_remove = []
        # Remove RPN head (custom anchors)
        keys_to_remove.extend([k for k in state_dict.keys() if k.startswith('rpn.head.')])
        # Remove ROI box predictor (custom num_classes)
        keys_to_remove.extend([k for k in state_dict.keys() if k.startswith('roi_heads.box_predictor.')])

        for key in keys_to_remove:
            del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
        print(f'Loaded pretrained weights (excluded RPN head and ROI box predictor)')

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Freeze ROI heads if requested
    if freeze_roi_heads:
        for param in model.roi_heads.parameters():
            param.requires_grad = False

    return model


def compute_rpn_loss(loss_dict: dict) -> torch.Tensor:
    """
    Compute RPN-only loss, ignoring ROI head losses.

    Args:
        loss_dict: Dictionary of losses from model forward pass

    Returns:
        Combined RPN loss
    """
    rpn_loss = loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']
    return rpn_loss


class RPNTrainer:
    """Wrapper for RPN training."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self, images, targets):
        """Execute single training step (full Faster R-CNN)."""
        self.model.train()

        # Move to device
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass - returns loss dict in training mode
        loss_dict = self.model(images, targets)

        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return {
            'total_loss': losses.item(),
            'loss_classifier': loss_dict['loss_classifier'].item(),
            'loss_box_reg': loss_dict['loss_box_reg'].item(),
            'loss_objectness': loss_dict['loss_objectness'].item(),
            'loss_rpn_box_reg': loss_dict['loss_rpn_box_reg'].item(),
        }

    @torch.no_grad()
    def eval_step(self, images, targets):
        """Execute single evaluation step (full Faster R-CNN)."""
        # Keep model in train mode to get loss dict (no gradient update due to @torch.no_grad())
        self.model.train()

        # Move to device
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass - returns loss dict in training mode
        loss_dict = self.model(images, targets)

        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())

        return {
            'total_loss': losses.item(),
            'loss_classifier': loss_dict['loss_classifier'].item(),
            'loss_box_reg': loss_dict['loss_box_reg'].item(),
            'loss_objectness': loss_dict['loss_objectness'].item(),
            'loss_rpn_box_reg': loss_dict['loss_rpn_box_reg'].item(),
        }

    @torch.no_grad()
    def generate_proposals(self, images):
        """Generate RPN proposals for evaluation."""
        self.model.eval()

        # Move to device
        images = [img.to(self.device) for img in images]

        # Get proposals (model in eval mode returns detections)
        outputs = self.model(images)

        return outputs
