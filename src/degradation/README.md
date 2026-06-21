# src/degradation/ — Degradation Pipeline

Implements blur + noise degradation, **identical for all methods**.

## Exam Utility

- **Fair comparison**: same `degrade()` function for TV, UNet and DiffPIR — differences in results are only due to the methods
- **Fixed parameters**: Gaussian blur (σ=2, kernel=9) and 4 AWGN levels (0.005, 0.01, 0.05, 0.1)
- **Controlled seed**: seed 42 for reproducibility

## Mathematical Formulation

The degraded image _y_ is obtained from the forward model:

$$ y = (H * x) + n $$

where _H_ is the Gaussian blur kernel, _*_ is 2D convolution, and _n_ ∼ N(0, σ²I) is independent AWGN noise.

The Gaussian kernel is computed as:

$$ G(i,j) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{i^2 + j^2}{2\sigma^2}\right) $$

with _i, j_ ∈ [-K/2, K/2] for kernel size _K=9_, normalized to sum 1.

## Why These Parameters?

| Parameter | Choice | Motivation |
|---|---|---|
| σ_blur = 2 | Moderate blur | Simulates real optical microscope defocus |
| Kernel 9×9 | 99.7% Gaussian energy | Sufficient size for σ=2 without truncation artifacts |
| σ_n ∈ [0.005, 0.1] | 4 noise levels | Covers scenarios from low noise (quality CCD) to high noise (fast acquisition) |

## Usage Example

```python
from src.degradation.degradation import degrade
import torch

x = torch.randn(1, 3, 256, 256)  # clean image [-1, 1]
x_degraded, kernel = degrade(x, noise_level=0.05)
# x_degraded.shape → (1, 3, 256, 256)
# kernel.shape → (1, 1, 9, 9)
```

## Reproducibility

The pipeline sets the deterministic seed **before** each invocation:

```python
import numpy as np
import torch, random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

This guarantees that the identical degraded image is provided to all three methods.

## Pre-computed Dataset

The degraded images are saved in `data/degraded/` to avoid recomputing each time:

```
data/degraded/
├── noise_0.005/      # 962 images
├── noise_0.01/       # 962 images
├── noise_0.05/       # 962 images
└── noise_0.1/        # 962 images
```

Loading is done via `DegradedDataset` in `src/data/dataset.py`, which reads `_degraded.pt` for each original image.
