# TV — Total Variation (Variational Method)

Implementation of Total Variation regularization for deblur + denoise on cervical LBC images.

## Exam Utility

- **Classic baseline**: simple and interpretable variational method — requires no training or data
- **Regularization term**: penalizes local variations, favoring uniform regions with sharp edges
- **Comparison**: ideal for showing the limitations of classical methods compared to deep learning and generative approaches

## Files

| File | Content |
|---|---|
| tv.py | tv_restore(), tv_loss(), gaussian_kernel_tensor(), apply_blur() |

## Mathematical Formulation

The inverse problem is formalized as minimizing the functional:

```
min_x  J(x) = ||Hx - y||^2_2 + lambda * TV(x)
```

where **H** is the blur operator (Gaussian convolution), **y** is the degraded image, **lambda** is the regularization weight.

### Total Variation (anisotropic)

```
TV(x) = |nabla_h x| + |nabla_v x|
      = sum_{i,j} |x(i+1,j) - x(i,j)| + |x(i,j+1) - x(i,j)|
```

Uses L1 norm on gradient components separately (vs. isotropic sqrt(nabla_h^2 + nabla_v^2)), favoring sharp diagonal edges.

### Loss computation (tv.py:5-8)

```python
def tv_loss(x):
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()
```

diff_h = vertical differences (horizontal gradient), diff_w = horizontal differences (vertical gradient). The .mean() normalizes by size.

### Optimization with Adam

The data-fidelity gradient is 2H^T(Hx - y); the TV gradient is computed via autograd. **Adam** is used instead of SGD for:

- Faster convergence (~150 iter vs. ~500 with SGD)
- Per-parameter adaptive learning rate
- Less sensitivity to lambda tuning

**Clamping**: after each step x.clamp_(-1.0, 1.0) maintains the [-1, 1] range.

## Usage

### Import and single call

```python
from src.methods.tv.tv import tv_restore

restored = tv_restore(
    degraded,                    # tensor [C, H, W] in [-1, 1]
    kernel_size=9, sigma=2.0,   # blur kernel (same degradation)
    lambda_reg=0.005,           # TV weight
    max_iter=150,               # Adam iterations
    lr=0.001                    # learning rate
)
```

### Full execution

```bash
python scripts/run_tv.py
```

Evaluates all 145 test images for the 4 noise levels (0.005, 0.01, 0.05, 0.1).

## Parameters — Motivated Choices

| Parameter | Value | Motivation |
|---|---|---|
| lambda_reg | 0.005 | Balanced: suppresses noise without excessive staircasing |
| max_iter | 150 | Convergence within ~100 iter; 150 for safety margin |
| lr | 0.001 | Default Adam for image optimization |

### Lambda Tuning

- **lambda < 0.001**: residual noise, data-fidelity dominates
- **lambda > 0.05**: excessive smoothing, detail loss, staircasing
- **lambda = 0.005**: empirical optimum for sigma_n in [0.005, 0.1] on LBC images

### Why 150 iterations?

Loss drops rapidly in the first 50 iterations, then stabilizes. Beyond 200 iterations, overfitting to noise is observed (data-fidelity keeps decreasing, SSIM degrades).

## Staircasing Effect

TV-L1 produces **staircasing**: uniform regions with artificial steps (piecewise constant gradients). On LBC images:

- Flat zones with false contours in homogeneous cytoplasmic areas
- Loss of fine texture

For sigma_n = 0.1, PSNR drops to 26.54 dB and staircasing is visible. This is the main limitation of the method vs UNet/DiffPIR.

## Comparison with Other Variational Methods

| Method | Characteristic |
|---|---|
| **TV-L1** (this) | L1 norm on gradient — preserves edges, but staircasing |
| **TV-L2** | L2 norm — less staircasing, but blurred edges |
| **Bilateral TV** | Penalizes gradients by intensity — better on textures |
| **Tikhonov** | L2 regularization — generalized oversmoothing |

## Blur Kernel

Identical to src/degradation/degradation.py for fair comparison:

- **Type**: Gaussian 9x9, sigma = 2.0
- **Channels**: per-channel (groups=3)
- **Padding**: reflection

## Results (145 test images x 4 noise levels)

| sigma_n | PSNR | SSIM |
|---|---|---|
| 0.005 | 32.09 dB | 0.911 |
| 0.01 | 32.04 dB | 0.909 |
| 0.05 | 30.42 dB | 0.837 |
| 0.1 | 26.54 dB | 0.586 |

For low sigma_n (0.005-0.01) TV competes with UNet (~32 vs 29.89 dB). For high sigma_n (0.1) it drops to 26.54 dB vs 28.93 dB for UNet — deep learning generalizes better.
