# src/plots/ — Visualization and Plots

Functions for generating visual comparisons and comparative plots of restoration methods.

## Exam Utility

- **Visual comparison**: degraded vs restored vs GT images side by side — qualitative evaluation
- **Comparison plot**: PSNR/SSIM plot for all methods and noise levels — mandatory deliverable

## Files

| File | Content |
|---|---|
| visualize.py | show_comparison(), plot_metrics() |

## Full API

### show_comparison(images_dict, save_path=None)

Generates a horizontal grid (1xN) with the images in images_dict.

```python
images = {"Ground Truth": gt, "Degraded": deg, "Restored TV": restored}
show_comparison(images, save_path="results/comparison.png")
```

Parameters:
- images_dict: dict[str, torch.Tensor | np.ndarray] — label to image mapping
- save_path: str | Path | None — save path (None = display on screen)

Implementation details:
- Layout: 1 row, N columns, figsize=(4*N, 4)
- PyTorch tensors converted: detach().cpu().permute(1,2,0).clamp(0,1).numpy()
- Saved at 150 DPI, tight_layout
- Output: results/<method>/qualitative/ folder

### plot_metrics(results_dir, save_path=None)

Reads all metrics.csv files from results/<method>/ and generates a 1x2 plot.

```python
plot_metrics("results", save_path="results/comparison.png")
```

Parameters:
- results_dir: str | Path — directory with subfolders per method
- save_path: str | Path | None — save path

Implementation details:
- Reads CSV with pandas (expected columns: method, noise_level, psnr, ssim)
- Plots line with marker o for each method
- Subplots 1x2, figsize=(12, 5), saved at 150 DPI

## Code snippet: generating comparison for TV

```python
from src.data.dataset import DegradedDataset
from src.methods.tv.tv import tv_restore
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison, plot_metrics

ds = DegradedDataset("data/degraded/test/noise_0.05", "data/degraded/test/gt")
for idx in range(5):
    deg, gt = ds[idx]
    restored = tv_restore(deg, lambda_reg=0.005, num_iters=150)
    metrics = evaluate(restored, gt)
    show_comparison({"GT": gt, "Degraded": deg, "TV": restored},
                    save_path=f"results/tv/qualitative/sample_{idx:02d}.png")
```

## Code snippet: generating comparison plot

Execution:
```bash
python scripts/plot_results.py
```

Internally calls plot_metrics("results", "results/comparison.png") and produces
a plot comparing TV, UNet and DiffPIR across the 4 noise levels.

## Organized Outputs

```
results/
├── comparison.png              # Comparison plot (all methods)
├── tv/qualitative/             # 24 TV comparison images
├── unet/qualitative/           # 24 UNet comparison images
└── diffpir/qualitative/        # 24 DiffPIR comparison images
```

Each method produces 6 images for each of the 4 noise levels (24 total).
