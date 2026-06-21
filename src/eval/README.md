# src/eval/ — Evaluation Metrics

Implements PSNR and SSIM for quantitative evaluation of restoration methods.

## Exam Utility

- **Standard metrics**: PSNR and SSIM are the metrics required by the exam for comparison
- **Unified function**: `evaluate()` computes both metrics in a single call — used by all methods

## Files

| File | Content |
|---|---|
| `metrics.py` | `compute_psnr()`, `compute_ssim()`, `evaluate()` |

## Full API

### `compute_psnr(pred, gt)` — PSNR in dB

Formula:
```
PSNR = 10 * log10( MAX^2 / MSE )
     = 20 * log10( 1 / sqrt(MSE) )
```
with `MAX = 1.0` (range [0, 1]) and `MSE = mean((pred - gt)^2)`.

### `compute_ssim(pred, gt)` — SSIM

SSIM compares luminance, contrast and structure between two windows:
```
SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)
```
Implemented via `skimage.metrics.structural_similarity` with `channel_axis=-1`.

### `evaluate(pred, gt)` — Combined metric

```python
from src.eval.metrics import compute_psnr, compute_ssim, evaluate

# Single image
psnr_val = compute_psnr(restored, gt)     # e.g. 29.89 dB
ssim_val = compute_ssim(restored, gt)     # e.g. 0.894

# Combined metric
result = evaluate(restored, gt)
print(result)  # {"psnr": 29.89, "ssim": 0.894}
```

### Range conversion

PyTorch tensors arrive in range [-1, 1]. The internal function `to_numpy()` converts:
```python
img = tensor.detach().cpu().permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
img = img * 0.5 + 0.5                                   # [-1,1] -> [0,1]
img = np.clip(img, 0.0, 1.0)
```

### Special cases

| Condition                    | PSNR      | SSIM    |
|------------------------------|-----------|---------|
| pred == gt                   | inf (inf) | 1.0     |
| Black image (all 0)          | ~10-20 dB | ~0.0-0.1|
| White image (all 1)          | ~10-20 dB | ~0.0-0.1|
| Out-of-range values          | auto clip to [0,1] | |

### Usage in scripts

```python
# Excerpt from scripts/run_tv.py / run_unet.py / run_diffpir.py
from src.eval.metrics import evaluate
metrics = evaluate(restored_img, gt_img)
print(f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.3f}")
```
