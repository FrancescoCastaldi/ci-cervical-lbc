# src/data/ — Dataset Loading and Preprocessing

Module for dataset loading, creating stratified splits and transformations.

## Exam Utility

- **Shared pipeline**: all methods use the same LBCDataset — same images, same normalization
- **Stratified split**: 70/15/15 partitioning with fixed seed 42 — reproducibility
- **Consistent format**: resize 256×256, normalization [-1, 1]

## Files

| File | Content |
|---|---|
| dataset.py | LBCDataset, DegradedDataset, build_splits(), load_config() |

## Full API

### LBCDataset(split_file, image_size=256, return_path=False)

```python
ds = LBCDataset("data/splits/train.txt", image_size=256)
img = ds[0]                   # torch.Tensor, shape (3, 256, 256), range [-1, 1]
img, path = ds[0]             # with return_path=True
len(ds)                       # number of images in subset
```

The applied transform is:
```python
T.Compose([
    T.Resize((image_size, image_size)),   # bilinear interpolation
    T.ToTensor(),                         # uint8 → float [0, 1]
    T.Normalize(mean=[0.5], std=[0.5]),   # [0,1] → [-1,1]: x' = (x - 0.5) / 0.5
])
```

### DegradedDataset(degraded_dir, gt_dir)

```python
ds = DegradedDataset(
    "data/degraded/test/noise_0.05",
    "data/degraded/test/gt"
)
deg, gt = ds[0]   # two tensors (3, 256, 256), range [-1, 1]
```

The .pt files are aligned by name: same_index.pt = same image.

### build_splits(config)

1. Scans data/raw/ recursively for .png and .jpg files
2. Validates each image with PIL.Image.verify() (discards corrupted)
3. Shuffle with random.seed(42), takes subset of 962 images
4. Partition: 70% train (673), 15% val (144), 15% test (145)
5. Saves train.txt, val.txt, test.txt in data/splits/

```python
from src.data.dataset import load_config, build_splits
config = load_config()
splits = build_splits(config)
# splits: {"train": [...], "val": [...], "test": [...]}
```

### load_config(path=None)

Loads configs/experiment.yaml as a Python dict. Defines seed, paths, degradation
parameters and method hyperparameters.

## Normalization [-1, 1]

```
input_uint8 [0, 255] → ToTensor → [0, 1] → Normalize → output [-1, 1]
inverse: restored = (tensor * 0.5 + 0.5).clamp(0, 1)
```

All methods operate in range [-1, 1]. Metrics convert back to [0, 1] internally.
