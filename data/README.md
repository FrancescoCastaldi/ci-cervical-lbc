# data/ — Dataset and Processed Data

Contains the original Mendeley dataset, stratified splits, and precomputed degraded data for all methods.

## Exam Utility

- **Real dataset**: LBC cervical cytology images (962 images, 4 diagnostic classes)
- **Fixed splits**: train/val/test 70/15/15 — same split for all methods (fair comparison)
- **Identical degradation**: all methods receive the same degraded image (fair comparison)

## Mendeley Dataset Structure

```
data/raw/
├── NILM/       # Negative for Intraepithelial Lesion or Malignancy
├── HSIL/       # High-grade Squamous Intraepithelial Lesion
├── LSIL/       # Low-grade Squamous Intraepithelial Lesion
└── SCC/        # Squamous Cell Carcinoma
```

Each subfolder contains 3-channel RGB PNG images at variable resolution (~512×512).

## Class Statistics

| Class   | Count   | %     |
|---------|---------|-------|
| NILM    | ~560    | 58.2% |
| HSIL    | ~160    | 16.6% |
| LSIL    | ~142    | 14.8% |
| SCC     | ~100    | 10.4% |
| Total   | 962     | 100%  |

## Preprocessing Pipeline

1. **Validation**: PIL.Image.verify() discards corrupted images
2. **Resize**: all images resized to 256×256 pixels (bilinear)
3. **Normalization**: ToTensor() [0,1] → Normalize(0.5, 0.5) → [-1, 1]
4. **Stratified split**: 70/15/15 with seed=42, shuffle before subset

```python
from src.data.dataset import LBCDataset, DegradedDataset

# Load GT images from test set
ds = LBCDataset("data/splits/test.txt", image_size=256)
img_tensor = ds[0]  # shape: (3, 256, 256), range: [-1, 1]

# Load (degraded, GT) pairs for noise_level=0.05
ds = DegradedDataset("data/degraded/test/noise_0.05", "data/degraded/test/gt")
deg, gt = ds[0]  # two tensors (3, 256, 256)
```

## .pt File Organization

| Directory | Content |
|---|---|
| degraded/train/gt/ | Training ground truth (00000.pt ... 00672.pt) |
| degraded/train/noise_0.05/ | Training degraded (same indices) |
| degraded/val/gt/ | Validation ground truth |
| degraded/val/noise_0.01/ | Validation degraded per noise level |
| degraded/test/gt/ | Test ground truth |
| degraded/test/noise_0.1/ | Test degraded per noise level |

Each .pt contains a `torch.Tensor` of shape (3, 256, 256).

## Generation

```bash
python scripts/preprocess.py
```

The script performs: (1) `build_splits()` — scan `data/raw/`, validation, stratified split; (2) GT + degraded generation for each noise level (0.005, 0.01, 0.05, 0.1) with Gaussian blur (σ=2, kernel=9); (3) save visual examples in `data/degraded/examples/`.

> The dataset is available on [Mendeley](https://data.mendeley.com/datasets/zddtpgzv63/2).
> After downloading it to `data/raw/`, run `python scripts/preprocess.py`.
