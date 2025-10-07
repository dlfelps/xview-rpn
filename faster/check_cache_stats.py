"""Check statistics of the cached dataset."""

from faster.data import XViewDataset
import torch

print("=== COMPARING OLD VS NEW CACHE ===\n")

# Load with old cache
print("Loading OLD cache (with partial objects and large objects)...")
dataset_old = XViewDataset(
    image_dir='data/train_images',
    geojson_path='data/xView_train.geojson',
    patch_size=224,
    force_rebuild=False,
)

print(f"Total patches: {len(dataset_old)}")

# Analyze old cache
all_widths = []
all_heights = []
partial_count = 0
large_count = 0

for i in range(min(1000, len(dataset_old))):
    image, target = dataset_old[i]
    boxes = target['boxes']

    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        all_widths.append(w.item())
        all_heights.append(h.item())

        # Check if at edge (partial)
        if x1 == 0 or y1 == 0 or x2 == 224 or y2 == 224:
            partial_count += 1

        # Check if large
        if w >= 64 or h >= 64:
            large_count += 1

all_widths = torch.tensor(all_widths)
all_heights = torch.tensor(all_heights)

print(f"\nOLD cache statistics (first 1000 patches):")
print(f"  Total objects: {len(all_widths)}")
print(f"  Partial objects (at edge): {partial_count} ({partial_count/len(all_widths)*100:.1f}%)")
print(f"  Large objects (>=64 pixels): {large_count} ({large_count/len(all_widths)*100:.1f}%)")
print(f"  Width range: [{all_widths.min():.0f}, {all_widths.max():.0f}]")
print(f"  Height range: [{all_heights.min():.0f}, {all_heights.max():.0f}]")

# Now rebuild cache with new filters
print("\n" + "="*60)
print("Rebuilding cache with NEW filters (complete objects only, < 64x64)...")
print("="*60 + "\n")

dataset_new = XViewDataset(
    image_dir='data/train_images',
    geojson_path='data/xView_train.geojson',
    patch_size=224,
    force_rebuild=True,
)

print(f"Total patches: {len(dataset_new)}")

# Analyze new cache
all_widths_new = []
all_heights_new = []
partial_count_new = 0
large_count_new = 0

for i in range(min(1000, len(dataset_new))):
    image, target = dataset_new[i]
    boxes = target['boxes']

    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        all_widths_new.append(w.item())
        all_heights_new.append(h.item())

        # Check if at edge (partial) - should be 0 now
        if x1 == 0 or y1 == 0 or x2 == 224 or y2 == 224:
            partial_count_new += 1

        # Check if large - should be 0 now
        if w >= 64 or h >= 64:
            large_count_new += 1

if len(all_widths_new) > 0:
    all_widths_new = torch.tensor(all_widths_new)
    all_heights_new = torch.tensor(all_heights_new)

    print(f"\nNEW cache statistics (first 1000 patches):")
    print(f"  Total objects: {len(all_widths_new)}")
    print(f"  Partial objects (at edge): {partial_count_new} ({'SHOULD BE 0!' if partial_count_new > 0 else 'CORRECT'})")
    print(f"  Large objects (>=64 pixels): {large_count_new} ({'SHOULD BE 0!' if large_count_new > 0 else 'CORRECT'})")
    print(f"  Width range: [{all_widths_new.min():.0f}, {all_widths_new.max():.0f}]")
    print(f"  Height range: [{all_heights_new.min():.0f}, {all_heights_new.max():.0f}]")
    print(f"  Median width: {all_widths_new.median():.1f}")
    print(f"  Median height: {all_heights_new.median():.1f}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Patches: {len(dataset_old)} -> {len(dataset_new)} ({len(dataset_new)/len(dataset_old)*100:.1f}% retained)")
    print(f"Objects (in 1000 patches): {len(all_widths)} -> {len(all_widths_new)} ({len(all_widths_new)/len(all_widths)*100:.1f}% retained)")
    print(f"Max object size: {max(all_widths.max(), all_heights.max()):.0f} -> {max(all_widths_new.max(), all_heights_new.max()):.0f} pixels")
else:
    print("\n⚠️  WARNING: No objects found in new cache!")
    print("This might indicate an issue with the filtering logic.")
