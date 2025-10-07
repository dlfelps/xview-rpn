# Data Filtering Changes

## Overview
Updated the xView dataset pipeline to focus on **small, complete objects** for improved RPN training.

## Changes Made

### 1. Complete Objects Only (No Partial Objects)
**Before**: Used `patch_box.intersects(obj_box)` - included objects partially clipped at patch boundaries
**After**: Uses `patch_box.contains(obj_box)` - only includes objects fully contained within patches

**Rationale**: Partial objects have incomplete ground truth boxes, which can confuse the RPN during training. The network would try to predict proposals for objects that are cut off at the edge.

### 2. Size Filter (< 64x64 pixels)
**Before**: No maximum size limit (objects up to 224x224 pixels included)
**After**: Excludes objects with width >= 64 or height >= 64 pixels

**Rationale**:
- Focuses training on small object detection
- Matches anchor sizes better (16, 32, 64)
- Large objects are easier to detect and don't need specialized training

## Updated Code Location
Changes in [`faster/data.py`](faster/data.py#L148-L156):
- Line 149: Changed from `intersects` to `contains`
- Lines 150-156: Added size filtering logic

## How to Rebuild Cache

The dataset caches processed patches for faster loading. After updating the filtering logic, you need to rebuild the cache:

```bash
# Option 1: Use the training script with force rebuild flag
python -m faster.train --force-rebuild-cache --data-dir data/train_images --geojson data/xView_train.geojson --epochs 10 --batch-size 8 --unfreeze-backbone --device cuda

# Option 2: Check cache statistics (also rebuilds)
python -m faster.check_cache_stats
```

## Expected Impact

### Dataset Size
- **Patches**: Will decrease (patches with only partial/large objects are excluded)
- **Objects per patch**: Will decrease on average
- **Object size distribution**: Will be more focused on small objects (4-63 pixels)

### Training Benefits
1. **Cleaner training signal**: No ambiguous partial objects
2. **Better anchor matching**: Objects match anchor sizes better
3. **Focused task**: Network specializes in small object detection
4. **Less noise**: No edge artifacts from clipped objects

## Verification

Run the cache statistics script to verify:
```bash
python -m faster.check_cache_stats
```

Expected output:
- ✅ Partial objects: 0 (down from ~15-20%)
- ✅ Large objects: 0 (down from ~10-15%)
- ✅ Max object size: <64 pixels (down from ~224)
- ✅ Median object size: 30-40 pixels

## Anchor Configuration
The current anchor sizes (16, 32, 64) with aspect ratios (0.5, 1.0, 2.0) should now align well with the filtered dataset:

| Anchor Size | Width x Height | Area |
|-------------|----------------|------|
| 16 @ 0.5    | 8 x 16        | 128  |
| 16 @ 1.0    | 16 x 16       | 256  |
| 16 @ 2.0    | 32 x 16       | 512  |
| 32 @ 0.5    | 16 x 32       | 512  |
| 32 @ 1.0    | 32 x 32       | 1024 |
| 32 @ 2.0    | 64 x 32       | 2048 |
| 64 @ 0.5    | 32 x 64       | 2048 |
| 64 @ 1.0    | 64 x 64       | 4096 |
| 64 @ 2.0    | 128 x 64      | 8192 |

With objects now < 64x64, most will be covered by the smaller anchors (16, 32) with the 64 anchors handling the larger end of the spectrum.
