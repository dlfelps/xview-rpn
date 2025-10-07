"""xView dataset wrapper for RPN training."""

import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF
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
        cache_dir: str = None,
        force_rebuild: bool = False,
    ):
        """
        Args:
            image_dir: Directory containing .tif images
            geojson_path: Path to xView_train.geojson
            patch_size: Size of image patches (default 224)
            transform: Optional torchvision transforms
            min_object_size: Minimum object size in pixels to include
            cache_dir: Directory to cache patch metadata (default: same as image_dir)
            force_rebuild: Force rebuild cache even if it exists
        """
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.min_object_size = min_object_size
        self.geojson_path = geojson_path

        # Build class mapping (xView type_ids to sequential labels)
        # This will be populated when loading/generating patches
        self.class_mapping = {}  # {type_id: sequential_label}
        self.num_classes = 0

        # Set cache directory
        if cache_dir is None:
            cache_dir = self.image_dir.parent / 'cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Generate cache filename based on parameters
        cache_filename = f'patches_ps{patch_size}_minsize{min_object_size}.pkl'
        self.cache_path = self.cache_dir / cache_filename

        # Load or generate patches
        if not force_rebuild and self.cache_path.exists():
            print(f'Loading cached patches from {self.cache_path}...')
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Handle old cache format (just a list) or new format (dict with base_path)
            if isinstance(cache_data, dict):
                cached_base = cache_data['base_path']
                self.patches = cache_data['patches']
                self.class_mapping = cache_data.get('class_mapping', {})
                self.num_classes = cache_data.get('num_classes', 0)
                print(f'Loaded {len(self.patches)} patches from cache (base: {cached_base})')
                print(f'Loaded {self.num_classes} classes')
            else:
                # Old format - assume patches already have absolute paths
                self.patches = cache_data
                print(f'Loaded {len(self.patches)} patches from cache (old format)')
                # Will need to rebuild to get class mapping
        else:
            print('Generating patches (this may take a while)...')
            # Load GeoJSON annotations
            with open(geojson_path, 'r') as f:
                self.geojson_data = json.load(f)

            # Build image annotations mapping
            self.image_annotations = self._build_image_annotations()

            # Generate all patches
            self.patches = self._generate_patches()

            # Build class mapping from all patches
            self._build_class_mapping()

            # Save to cache (convert to relative paths)
            print(f'Saving {len(self.patches)} patches to cache at {self.cache_path}...')
            cache_data = {
                'base_path': str(self.image_dir),
                'patches': self._convert_to_relative_paths(self.patches),
                'class_mapping': self.class_mapping,
                'num_classes': self.num_classes,
            }
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print('Cache saved successfully')

        # Convert cached relative paths to absolute paths
        self.patches = self._convert_to_absolute_paths(self.patches)

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

                        # Check if object is completely contained in patch (not partial)
                        if patch_box.contains(obj_box):
                            # Get object dimensions in original coordinates
                            orig_width = xmax - xmin
                            orig_height = ymax - ymin

                            # Skip objects that are >= 64x64 pixels
                            if orig_width >= 64 or orig_height >= 64:
                                continue

                            # Convert to patch-relative coordinates
                            rel_xmin = xmin - x
                            rel_ymin = ymin - y
                            rel_xmax = xmax - x
                            rel_ymax = ymax - y

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
                            'image_id': image_id,
                            'patch_coords': (x, y),
                            'objects': objects,
                        })

        return patches

    def _build_class_mapping(self):
        """Build mapping from xView type_ids to sequential 1-indexed labels."""
        # Collect all unique type_ids
        type_ids = set()
        for patch in self.patches:
            for obj in patch['objects']:
                type_ids.add(obj['type_id'])

        # Sort and create sequential mapping (1-indexed, 0 is background)
        sorted_type_ids = sorted(type_ids)
        self.class_mapping = {type_id: idx + 1 for idx, type_id in enumerate(sorted_type_ids)}
        self.num_classes = len(self.class_mapping) + 1  # +1 for background

        print(f'Built class mapping: {len(self.class_mapping)} classes')
        print(f'Type IDs: {sorted_type_ids[:10]}{"..." if len(sorted_type_ids) > 10 else ""}')

    def _convert_to_relative_paths(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert absolute image paths to relative paths (just image_id)."""
        relative_patches = []
        for patch in patches:
            relative_patch = patch.copy()
            # Store only the image filename, not the full path
            relative_patch['image_path'] = patch['image_id']
            relative_patches.append(relative_patch)
        return relative_patches

    def _convert_to_absolute_paths(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert relative image paths to absolute paths using current image_dir."""
        absolute_patches = []
        for patch in patches:
            absolute_patch = patch.copy()
            # Reconstruct full path using current image_dir
            image_filename = patch['image_path']
            # Extract just the filename if it's already a path
            image_filename = Path(image_filename).name
            absolute_patch['image_path'] = str(self.image_dir / image_filename)
            absolute_patches.append(absolute_patch)
        return absolute_patches

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
            # Map xView type_id to sequential label using class_mapping
            # class_mapping maps type_id -> 1-indexed label (0 is background)
            type_id = obj['type_id']
            label = self.class_mapping.get(type_id, 1)  # Default to 1 if not found
            labels.append(label)

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
            # Default: convert to tensor and normalize to [0, 1]
            # to_tensor converts PIL image (H, W, C) to tensor (C, H, W) and scales to [0, 1]
            image = TF.to_tensor(image)

        return image, target


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return tuple(zip(*batch))
