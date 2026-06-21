# src/methods/ — Restoration Methods

Implementations of three restoration methods: variational (TV), end-to-end (UNet), generative (DiffPIR).

## Exam Utility

- **Methodological coverage**: three different families — variational, deep learning, generative
- **Independence**: each method is a separate module with its own code
- **Critical comparison**: same inputs, same metrics — evaluate strengths and limitations of each

## Methods

| Module | Method | Family | Main file | Key parameters |
|---|---|---|---|---|
| `tv/` | Total Variation | Variational | `tv.py` | λ_reg=0.005, iter=150 |
| `unet/` | UNet | Deep Learning (end-to-end) | `unet.py` | 1.9M params, GroupNorm |
| `diffpir/` | DiffPIR | Generative (diffusion) | `diffpir.py` | ζ=1.0, step=30, DDIM |

## Comparative Results (PSNR / SSIM on test set)

| σ_n | TV | UNet | DiffPIR |
|---|---|---|---|
| 0.005 | 32.09 dB / 0.911 | 29.89 dB / 0.894 | 16.67 dB / 0.235 |
| 0.01  | 32.04 dB / 0.909 | 29.89 dB / 0.894 | 17.32 dB / 0.270 |
| 0.05  | 30.42 dB / 0.837 | 29.63 dB / 0.875 | 22.49 dB / 0.512 |
| 0.1   | 26.54 dB / 0.586 | 28.93 dB / 0.830 | 24.68 dB / 0.664 |

**TV** dominates at low noise (σ_n ≤ 0.01), **UNet** is the most robust across all levels, **DiffPIR** recovers at high noise but struggles at low σ_n.

## Computational Complexity (CPU, 256×256)

| Method | Time per image | Notes |
|---|---|---|
| TV | ~8 s | 150 Adam iterations |
| UNet | ~0.035 s | Single forward pass |
| DiffPIR | ~3 s | 30 DDIM sampling steps |

## When to Use Each Method

- **TV**: when no training data is available and an interpretable baseline is needed
- **UNet**: when data similar to the training distribution is available and inference speed is required
- **DiffPIR**: when more computational cost is acceptable for generative quality and fine details

## Module Structure

```
tv/
├── tv.py          # TV loss + optimizer loop
└── __init__.py

unet/
├── unet.py        # UNet architecture (encoder-decoder, skip connections)
└── __init__.py

diffpir/
├── diffpir.py     # DiffPIR algorithm (FFT data-fidelity, DDIM)
├── model.py       # LightUNet for DDPM (1.26M params)
├── train.py       # DDPM training loop
├── weights/       # Pre-trained weights (not tracked)
└── README.md      # Specific documentation
```

## Example: Running a Method on a Single Image

```python
from src.degradation.degradation import degrade
from src.methods.tv.tv import tv_restore
from src.methods.unet.unet import UNet, unet_restore
from src.methods.diffpir.diffpir import diffpir_restore

x_clean = torch.randn(1,3,256,256)
x_noisy, _ = degrade(x_clean, noise_level=0.05)

x_tv, _ = tv_restore(x_noisy, lmbda=0.005, n_iter=150)
x_unet = unet_restore(x_noisy, model, device)
x_diffpir = diffpir_restore(x_noisy, model, zeta=1.0, steps=30)
```
