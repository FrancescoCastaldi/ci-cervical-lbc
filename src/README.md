# src/ — Source Code

Python modules for the entire pipeline: data loading, degradation, restoration methods, metrics, visualization.

## Exam Utility

- **Modularity**: each method is independent in methods/<method>/ — clean and maintainable code
- **Shared pipeline**: degradation, metrics and plots are common — guaranteed fair comparison
- **Heuristic parameters**: each method has its own hyperparameters, documented in the report

## Submodules

| Module | Content | Role |
|---|---|---|
| data/ | LBCDataset, build_splits, DegradedDataset | Dataset loading and preprocessing |
| degradation/ | degrade(), gaussian_kernel() | Blur + AWGN pipeline (identical for all) |
| methods/tv/ | tv_restore(), tv_loss() | Total Variation (variational) |
| methods/unet/ | UNet, DoubleConv | UNet end-to-end (deep learning) |
| methods/diffpir/ | DiffPIR, LightUNet, training loop | DiffPIR generative (diffusion) |
| eval/ | compute_psnr(), compute_ssim(), evaluate() | Evaluation metrics |
| plots/ | show_comparison(), plot_metrics() | Results visualization |

## Execution Pipeline

```
data/ → degradation/ → methods/<method>/ → eval/ → plots/
```

Three independent methods, a single evaluation pipeline.

## Import Examples

```python
from src.data.dataset import LBCDataset, DegradedDataset, build_splits, load_config
from src.degradation.degradation import degrade, gaussian_kernel
from src.methods.tv.tv import tv_restore
from src.methods.unet.unet import UNet
from src.methods.diffpir.diffpir import DiffPIR
from src.eval.metrics import compute_psnr, compute_ssim, evaluate
from src.plots.visualize import show_comparison, plot_metrics
```

## Full Execution

```python
from src.data.dataset import LBCDataset, DegradedDataset
from src.degradation.degradation import degrade
from src.methods.tv.tv import tv_restore
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison

ds = DegradedDataset("data/degraded/test/noise_0.05", "data/degraded/test/gt")
deg, gt = ds[0]
restored = tv_restore(deg, lambda_reg=0.005, num_iters=150)
metrics = evaluate(restored, gt)
show_comparison({"GT": gt, "Degraded": deg, "Restored": restored},
                save_path="comparison.png")
```

## Testing

```bash
pytest tests/ -v
```

Tests are organized by module: tests/test_degradation.py, tests/test_metrics.py,
tests/test_tv.py, tests/test_unet.py, tests/test_diffpir.py (34 tests total).

## How to Add a Method

1. Create src/methods/<new_method>/ with the main module
2. Implement the restoration function (e.g. my_restore(img, **params))
3. Create scripts/run_<method>.py following the existing pattern
4. Results automatically flow into results/<method>/metrics.csv
5. python scripts/plot_results.py includes the new method in the comparison plot
