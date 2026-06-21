# DiffPIR — Generative Method

Implementation of the DiffPIR method (Denoising Diffusion Models for Plug-and-Play Image Restoration) for restoring degraded images.

## Model

Uses a **LightUNet** (custom DDPM) trained on cervical LBC images:
- Architecture: lightweight UNet with sinusoidal time embedding
- Timesteps: 1000 (training), 15 (sampling)
- Weights: `weights/ddpm_lbc.pt` (~5MB)

### Training

```bash
python -m src.methods.diffpir.train
```

Training on 100 images from the training set, 30 epochs, MSE loss.

## Usage

### Full script
```bash
python scripts/run_diffpir.py
```

### Interactive notebook
```bash
jupyter notebook notebooks/04_diffpir.ipynb
```

## Configuration

Parameters in `configs/experiment.yaml`:
```yaml
diffpir:
  num_steps: 15           # Sampling steps
  noise_level: 0.05       # Reference noise level
  max_test_images: 10     # Number of images to process
  weights: src/methods/diffpir/weights/ddpm_lbc.pt
  lambda: 10.0            # Data-fidelity weight
  zeta: 0.0               # Stochasticity (0 = deterministic)
  t_start: 50             # Starting timestep
```

### DiffPIR Parameters
| Parameter | Default | Role |
|---|---|---|
| `num_steps` | 15 | Sampling steps (sub-sampled from t_start to 0) |
| `lambda_` | 10.0 | Data-fidelity weight |
| `zeta` | 0.0 | Stochasticity (0=deterministic, 1=fully stochastic) |
| `t_start` | 50 | Starting timestep (50 for numerical stability) |

## Results

Metrics on 10 test images:

| σ_n | PSNR | SSIM | Time |
|---|---|---|---|
| 0.005 | 15.78 dB | 0.329 | 3.59 s |
| 0.01 | 16.45 dB | 0.374 | 3.79 s |
| 0.05 | 22.64 dB | 0.677 | 3.73 s |
| 0.1 | 25.46 dB | 0.766 | 3.81 s |

## Output

### Metrics
- `results/diffpir/metrics.csv`: PSNR, SSIM, inference time for each noise level

### Qualitative images
- `results/diffpir/qualitative/noise_{level}_sample{i}.png`: Degraded vs restored vs GT comparison

## Comparison with Other Methods

After running TV and UNet, load the results in the notebook for comparison:
```python
from src.plots.visualize import plot_metrics
plot_metrics("results", save_path="results/comparison.png")
```

## References

- Paper: [Denoising Diffusion Models for Plug-and-Play Image Restoration](https://arxiv.org/pdf/2305.08995.pdf)
- Repository: https://github.com/yuanzhi-zhu/DiffPIR
- Dataset: Mendeley LBC Cervical Cancer
