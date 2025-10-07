"""RPN model configuration for xView training."""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator


def create_rpn_model(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    freeze_roi_heads: bool = True,
    min_size: int = 224,
    anchor_sizes: tuple = (16, 32, 64),
    aspect_ratios: tuple = (0.5, 1.0, 2.0),
    num_classes: int = 2,  # background + object
):
    """
    Create Faster R-CNN model configured for RPN-only training.

    Args:
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone parameters
        freeze_roi_heads: Freeze ROI head parameters
        min_size: Minimum input image size
        anchor_sizes: RPN anchor sizes
        aspect_ratios: RPN anchor aspect ratios
        num_classes: Number of classes (2 for binary object detection)

    Returns:
        Faster R-CNN model configured for RPN training
    """
    # Configure custom RPN anchors
    anchor_generator = AnchorGenerator(
        sizes=tuple([anchor_sizes] * 5),  # 5 feature maps in FPN
        aspect_ratios=tuple([aspect_ratios] * 5),
    )

    # Load pretrained Faster R-CNN with custom anchor generator
    # First create model without weights to get correct RPN head dimensions
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        min_size=min_size,
        max_size=min_size * 2,
        rpn_anchor_generator=anchor_generator,
    )

    # Load pretrained weights if requested (skip incompatible RPN head)
    if pretrained:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True, check_hash=True)

        # Remove RPN head weights that don't match our custom anchor configuration
        rpn_keys_to_remove = [k for k in state_dict.keys() if k.startswith('rpn.head.')]
        for key in rpn_keys_to_remove:
            del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

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
        """Execute single training step."""
        self.model.train()
        #  Set ROI head to eval to prevent it from computing losses
        self.model.roi_heads.eval()

        # Move to device
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass - get features and RPN outputs only
        images_transformed, targets_transformed = self.model.transform(images, targets)
        features = self.model.backbone(images_transformed.tensors)

        # Get RPN losses directly
        proposals, rpn_losses = self.model.rpn(images_transformed, features, targets_transformed)

        # Compute RPN-only loss
        rpn_loss = rpn_losses['loss_objectness'] + rpn_losses['loss_rpn_box_reg']

        # Backward pass
        self.optimizer.zero_grad()
        rpn_loss.backward()
        self.optimizer.step()

        return {
            'rpn_loss': rpn_loss.item(),
            'loss_objectness': rpn_losses['loss_objectness'].item(),
            'loss_rpn_box_reg': rpn_losses['loss_rpn_box_reg'].item(),
        }

    @torch.no_grad()
    def eval_step(self, images, targets):
        """Execute single evaluation step."""
        # Keep model in train mode to get loss dict (no gradient update due to @torch.no_grad())
        self.model.train()
        self.model.roi_heads.eval()

        # Move to device
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass - get features and RPN outputs only
        images_transformed, targets_transformed = self.model.transform(images, targets)
        features = self.model.backbone(images_transformed.tensors)

        # Get RPN losses directly
        proposals, rpn_losses = self.model.rpn(images_transformed, features, targets_transformed)

        # Compute RPN-only loss
        rpn_loss = rpn_losses['loss_objectness'] + rpn_losses['loss_rpn_box_reg']

        return {
            'rpn_loss': rpn_loss.item(),
            'loss_objectness': rpn_losses['loss_objectness'].item(),
            'loss_rpn_box_reg': rpn_losses['loss_rpn_box_reg'].item(),
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
