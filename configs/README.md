# configs/ — Experiment Configuration

Contains experiment.yaml with all experiment parameters centralized for reproducibility.

## Loading in Python

The configuration is loaded by `src/data/dataset.py` and `scripts/run_*.py` via:

```python
import yaml
with open("configs/experiment.yaml") as f:
    cfg = yaml.safe_load(f)
```

Each script imports cfg and accesses sections: cfg["dataset"], cfg["degradation"], etc.

## Complete YAML Structure

```yaml
seed: 42                        # Global seed for numpy, torch, random

dataset:
  root: data/raw/               # Original dataset (unprocessed)
  processed: data/processed/    # Resized + normalized images
  splits: data/splits/          # Train/val/test .txt splits
  degraded: data/degraded/      # Precomputed degraded images
  subset_size: 4000             # Maximum number of images to use
  image_size: 256               # Resize to 256×256 px
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

degradation:
  blur_sigma: 2                 # Gaussian blur σ
  kernel_size: 9                # Blur kernel size (odd)
  noise_levels: [0.005, 0.01, 0.05, 0.1]   # 4 AWGN levels

tv:
  lambda_reg: 0.005             # Regularization weight (λ)
  max_iter: 150                 # Adam iterations

unet:
  lr: 0.0001                    # Adam learning rate
  batch_size: 16                # Training batch size
  epochs: 50                    # Maximum epochs
  in_channels: 4                # RGB + noise level channel
  out_channels: 3               # Reconstructed RGB
  features: [16, 32, 64, 128]  # Channels per level

diffpir:
  num_steps: 15                 # Subsampled DDIM steps
  noise_level: 0.05
  max_test_images: 10           # Test limit for speed
  weights: src/methods/diffpir/weights/ddpm_lbc.pt
  lambda: 10.0                  # Data-fidelity weight (FFT)
  zeta: 0.0                     # zeta>0 → extra noise on steps
  t_start: 50                   # Initial DDIM timestep

eval:
  results_dir: results/
  save_qualitative: 6           # 6 images per noise level
```

## Critical Parameter Explanations

| Parameter | Effect |
|---|---|
| blur_sigma: 2 | Equal for all methods; high σ → strong blur |
| noise_levels | 4 values tested; σₙ=0.1 is the hardest case |
| lambda_reg: 0.005 | Fidelity/regularization trade-off for TV |
| features: [16,32,64,128] | Lightweight architecture (~500K params, 1.9M with GroupNorm) |
| t_start: 50 | Chosen experimentally for our noise levels |
| lambda: 10.0 | Internally adapted per-σₙ for FFT data-fidelity |

## Modifying Parameters for Custom Experiments

1. Edit `configs/experiment.yaml`
2. Create a copy: `configs/experiment_noisy.yaml` and run with:
   ```python
   with open("configs/experiment_noisy.yaml") as f: cfg = yaml.safe_load(f)
   ```
3. For parameter sweeps, modify the parameter and re-run the script

## Seed Management

Fixed seed 42 ensures reproducibility across runs. If the seed changes, splits and initializations change. For fair comparisons, keep the same seed for all methods.

## Validation

The configuration has no automatic validation — verify:
- kernel_size is odd
- train_ratio + val_ratio + test_ratio == 1.0
- Paths in weights must exist
- image_size is a multiple of 16 (UNet/DiffPIR compatibility)
