# CI Cervical LBC

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/github/last-commit/FrancescoCastaldi/ci-cervical-lbc" alt="Last Commit">
  <img src="https://img.shields.io/github/repo-size/FrancescoCastaldi/ci-cervical-lbc" alt="Repo Size">
  <img src="https://img.shields.io/badge/framework-PyTorch-red" alt="PyTorch">
</p>

**Deblur & Denoise** of cervical cytology images using multiple computational imaging methods.

Computational Imaging project — Università di Bologna, LM Informatica.
Prof. Picciolomini & Evangelista.

---

## Task

Inverse problem: recover a high-quality image from a degraded observation (Gaussian blur + AWGN).

| Parameter | Value |
|---|---|
| Blur | Gaussian, σ=2, kernel=9×9 |
| Noise | AWGN, σₙ ∈ {0.005, 0.01, 0.05, 0.1} |
| Dataset | Mendeley LBC Cervical Cancer (962 images → 256×256) |

## Methods

| Method | Family | Status |
|---|---|---|
| Total Variation (TV) | Variational | 🔄 Da eseguire |
| UNet | End-to-end | 🔄 Da eseguire |
| **DiffPIR** | **Generative (Diffusion)** | **✅ Completato** |

### DiffPIR Results

Custom LightUNet (1.26M params) trained on LBC cervical images with FFT-based data-fidelity.

| σₙ | PSNR | SSIM | Time |
|---|---|---|---|
| 0.005 | 16.67 dB | 0.235 | 2.0 s |
| 0.01 | 17.32 dB | 0.270 | 2.0 s |
| 0.05 | **22.49 dB** | **0.512** | 2.0 s |
| 0.1 | **24.68 dB** | **0.664** | 2.0 s |

## Quick Start

```bash
# Setup
pip install -r requirements.txt

# Run DiffPIR (generative method)
python scripts/run_diffpir.py

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
ci-cervical-lbc/
├── configs/                    # experiment.yaml
├── data/
│   ├── raw/                    # Dataset (not tracked)
│   └── splits/                 # Train/val/test split files
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory analysis
│   └── 04_diffpir.ipynb        # DiffPIR demo
├── src/
│   ├── data/dataset.py         # Dataset loading & preprocessing
│   ├── degradation/            # Blur + noise pipeline
│   ├── methods/
│   │   ├── tv/                 # Total Variation
│   │   ├── unet/               # UNet
│   │   └── diffpir/            # DiffPIR (model, algorithm, training)
│   ├── eval/metrics.py         # PSNR, SSIM
│   └── plots/visualize.py      # Comparison plots
├── scripts/                    # Run experiments
├── tests/                      # Unit tests
├── report/                     # Documentation
├── agents.md                   # Project status & task division
└── README.md
```

## Running Experiments

```bash
python scripts/run_tv.py          # Total Variation baseline
python scripts/run_unet.py        # UNet end-to-end
python scripts/run_diffpir.py     # DiffPIR generative
python scripts/plot_results.py    # Comparison plot
```

## Reproducibility

- **Same degraded inputs** for all methods (`src/degradation/degradation.py`)
- **Fixed seed** 42 (`configs/experiment.yaml`)
- **Consistent evaluation**: PSNR/SSIM via `skimage.metrics`

## Contributors

- [Francesco Castaldi](https://github.com/FrancescoCastaldi)
- [Paolo Fusco](https://github.com/PaoloFusco)
