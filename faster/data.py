"""xView dataset wrapper for RPN training."""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import rasterio
import torch
from PIL import Image
from shapely.geometry import shape, box
from torch.utils.data import Dataset


class XViewDataset(Dataset):
    """xView dataset that clips images into 224x224 patches with annotations."""

    def __init__(
        self,
        image_dir: str,
        geojson_path: str,
        patch_size: int = 224,
        transform=None,
        min_object_size: int = 4,
    ):
        """
        Args:
            image_dir: Directory containing .tif images
            geojson_path: Path to xView_train.geojson
            patch_size: Size of image patches (default 224)
            transform: Optional torchvision transforms
            min_object_size: Minimum object size in pixels to include
        """
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.min_object_size = min_object_size

        # Load GeoJSON annotations
        with open(geojson_path, 'r') as f:
            self.geojson_data = json.load(f)

        # Build image annotations mapping
        self.image_annotations = self._build_image_annotations()

        # Generate all patches
        self.patches = self._generate_patches()

    def _build_image_annotations(self) -> Dict[str, List[Dict]]:
        """Group annotations by image_id."""
        annotations_by_image = {}

        for feature in self.geojson_data['features']:
            props = feature['properties']
            image_id = props['image_id']

            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []

            # Parse bounds_imcoords: "xmin,ymin,xmax,ymax"
            bounds = list(map(int, props['bounds_imcoords'].split(',')))
            annotations_by_image[image_id].append({
                'bounds': bounds,
                'type_id': props['type_id'],
            })

        return annotations_by_image

    def _generate_patches(self) -> List[Dict[str, Any]]:
        """Generate all 224x224 patches from images with their annotations."""
        patches = []

        for image_file in sorted(self.image_dir.glob('*.tif')):
            image_id = image_file.name

            # Skip images without annotations
            if image_id not in self.image_annotations:
                continue

            # Get image dimensions
            with rasterio.open(image_file) as src:
                height, width = src.height, src.width

            # Generate patches with stride = patch_size (no overlap)
            for y in range(0, height, self.patch_size):
                for x in range(0, width, self.patch_size):
                    # Ensure patch doesn't exceed image boundaries
                    patch_xmax = min(x + self.patch_size, width)
                    patch_ymax = min(y + self.patch_size, height)

                    # Skip patches smaller than patch_size
                    if (patch_xmax - x) < self.patch_size or (patch_ymax - y) < self.patch_size:
                        continue

                    # Find objects that intersect this patch
                    patch_box = box(x, y, patch_xmax, patch_ymax)
                    objects = []

                    for ann in self.image_annotations[image_id]:
                        xmin, ymin, xmax, ymax = ann['bounds']
                        obj_box = box(xmin, ymin, xmax, ymax)

                        if patch_box.intersects(obj_box):
                            # Calculate intersection
                            intersection = patch_box.intersection(obj_box)
                            int_bounds = intersection.bounds  # (minx, miny, maxx, maxy)

                            # Convert to patch-relative coordinates
                            rel_xmin = int_bounds[0] - x
                            rel_ymin = int_bounds[1] - y
                            rel_xmax = int_bounds[2] - x
                            rel_ymax = int_bounds[3] - y

                            # Filter out very small objects
                            obj_width = rel_xmax - rel_xmin
                            obj_height = rel_ymax - rel_ymin

                            if obj_width >= self.min_object_size and obj_height >= self.min_object_size:
                                objects.append({
                                    'bbox': [rel_xmin, rel_ymin, rel_xmax, rel_ymax],
                                    'type_id': ann['type_id'],
                                })

                    # Only include patches with at least one object
                    if objects:
                        patches.append({
                            'image_path': str(image_file),
                            'patch_coords': (x, y),
                            'objects': objects,
                        })

        return patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            image: Tensor of shape (3, 224, 224)
            target: Dict with 'boxes' (N, 4) and 'labels' (N,)
        """
        patch_info = self.patches[idx]

        # Load and crop image
        with rasterio.open(patch_info['image_path']) as src:
            x, y = patch_info['patch_coords']
            # Read patch (rasterio uses (row, col) indexing)
            window = rasterio.windows.Window(x, y, self.patch_size, self.patch_size)
            patch = src.read(window=window)  # (channels, height, width)

            # Convert to PIL Image for transforms
            # Handle different channel counts
            if patch.shape[0] == 1:
                patch = np.repeat(patch, 3, axis=0)
            elif patch.shape[0] > 3:
                patch = patch[:3]  # Take first 3 channels

            # Normalize to 0-255 range if needed
            if patch.max() > 255:
                patch = (patch / patch.max() * 255).astype(np.uint8)

            # Convert to PIL (H, W, C)
            patch = np.transpose(patch, (1, 2, 0))
            image = Image.fromarray(patch)

        # Prepare target
        boxes = []
        labels = []

        for obj in patch_info['objects']:
            boxes.append(obj['bbox'])
            # All objects get label 1 (object vs background)
            labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image, target


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return tuple(zip(*batch))
