"""Test image format and normalization."""

import torch
import torchvision.transforms.functional as TF
from faster.data import XViewDataset

# Load dataset
dataset = XViewDataset(
    image_dir='data/train_images',
    geojson_path='data/xView_train.geojson',
    patch_size=224,
)

# Get a sample
image, target = dataset[0]

print("=== IMAGE FORMAT CHECK ===")
print(f"Image shape: {image.shape}")
print(f"Expected shape: torch.Size([3, 224, 224])")
print(f"Shape correct: {image.shape == torch.Size([3, 224, 224])}")

print(f"\nImage dtype: {image.dtype}")
print(f"Expected dtype: torch.float32")
print(f"Dtype correct: {image.dtype == torch.float32}")

print(f"\nImage value range: [{image.min():.4f}, {image.max():.4f}]")
print(f"Expected range: [0.0, 1.0]")
print(f"Range correct: {image.min() >= 0.0 and image.max() <= 1.0}")

print(f"\nImage mean per channel:")
print(f"  R: {image[0].mean():.4f}")
print(f"  G: {image[1].mean():.4f}")
print(f"  B: {image[2].mean():.4f}")

# Check if channel order is correct (should be RGB)
print(f"\n=== CHECKING MULTIPLE SAMPLES ===")
for i in range(5):
    img, tgt = dataset[i]
    print(f"Sample {i}: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.3f}, {img.max():.3f}]")
    if img.shape != torch.Size([3, 224, 224]):
        print(f"  ⚠️  INCORRECT SHAPE!")
    if img.dtype != torch.float32:
        print(f"  ⚠️  INCORRECT DTYPE!")
    if img.min() < 0 or img.max() > 1:
        print(f"  ⚠️  VALUES OUT OF RANGE!")

print("\n=== TESTING DIFFERENT NORMALIZATION METHODS ===")

# Load raw PIL image
patch_info = dataset.patches[0]
import rasterio
from PIL import Image
import numpy as np

with rasterio.open(patch_info['image_path']) as src:
    x, y = patch_info['patch_coords']
    window = rasterio.windows.Window(x, y, 224, 224)
    patch = src.read(window=window)

    if patch.shape[0] == 1:
        patch = np.repeat(patch, 3, axis=0)
    elif patch.shape[0] > 3:
        patch = patch[:3]

    if patch.max() > 255:
        patch = (patch / patch.max() * 255).astype(np.uint8)

    patch = np.transpose(patch, (1, 2, 0))
    pil_image = Image.fromarray(patch)

print(f"PIL image mode: {pil_image.mode}")
print(f"PIL image size: {pil_image.size}")

# Method 1: Current manual approach
img1 = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
print(f"\nMethod 1 (current manual):")
print(f"  Shape: {img1.shape}")
print(f"  Dtype: {img1.dtype}")
print(f"  Range: [{img1.min():.4f}, {img1.max():.4f}]")

# Method 2: torchvision to_tensor
img2 = TF.to_tensor(pil_image)
print(f"\nMethod 2 (torchvision.to_tensor):")
print(f"  Shape: {img2.shape}")
print(f"  Dtype: {img2.dtype}")
print(f"  Range: [{img2.min():.4f}, {img2.max():.4f}]")

# Compare
print(f"\nAre they equal? {torch.allclose(img1, img2)}")
if not torch.allclose(img1, img2):
    print(f"Max difference: {(img1 - img2).abs().max():.6f}")

print("\n=== RECOMMENDATION ===")
if torch.allclose(img1, img2):
    print("✓ Current method is correct, but using torchvision.transforms.functional.to_tensor() is recommended for clarity")
else:
    print("⚠️  Current method differs from torchvision.to_tensor()!")
    print("   Consider switching to torchvision.transforms.functional.to_tensor()")
