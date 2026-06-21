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
| Total Variation (TV) | Variational | ✅ Complete |
| **UNet** | **End-to-end** | **✅ Complete** |
| **DiffPIR** | **Generative (Diffusion)** | **✅ Complete** |

### Total Variation Results

| σₙ | PSNR | SSIM |
|---|---|---|
| 0.005 | 32.09 dB | 0.911 |
| 0.01 | 32.04 dB | 0.909 |
| 0.05 | 30.42 dB | 0.837 |
| 0.1 | 26.54 dB | 0.586 |

Why do the results worsen with more noise?
It's normal, and there are two reasons:
1. The inverse problem becomes more difficult—with noise=0.1, the signal is much more corrupted, so there's less useful information to start with for reconstruction.
2. Lambda is fixed—lambda_reg=0.005 for all levels, but the optimal parameter changes with noise. For noise=0.1, a higher lambda would be needed for more damping.

### UNet Results

Optimized architecture (1.9M params, GroupNorm, noise conditioning) trained with L1 loss for 50 epochs on CPU.

| σₙ | PSNR | SSIM | Time |
|---|---|---|---|
| 0.005 | **29.58 dB** | **0.896** | 0.013 s |
| 0.01 | **29.57 dB** | **0.895** | 0.013 s |
| 0.05 | **29.37 dB** | **0.874** | 0.012 s |
| 0.1 | **28.72 dB** | **0.821** | 0.012 s |

### DiffPIR Results

Custom LightUNet (1.26M params) trained on LBC cervical images with FFT-based data-fidelity.

| σₙ | PSNR | SSIM | Time |
|---|---|---|---|
| 0.005 | **15.80 dB** | **0.356** | 0.65 s |
| 0.01 | **16.48 dB** | **0.407** | 0.64 s |
| 0.05 | **22.77 dB** | **0.722** | 0.64 s |
| 0.1 | **25.60 dB** | **0.797** | 0.64 s |

## Quick Start

```bash
# Setup
pip install -r requirements.txt

# Generate splits and degraded data (requires raw dataset in data/raw/)
python scripts/preprocess.py

# Run all methods
python scripts/run_tv.py          # Total Variation (variational)
python scripts/run_unet.py        # UNet (end-to-end)
python scripts/run_diffpir.py     # DiffPIR (generative)

# Comparison plot
python scripts/plot_results.py

# Run tests
python -m pytest tests/ -v
```

> **Note on the dataset:** The Mendeley LBC Cervical Cancer dataset contains 962 images total (not 4000 as indicated in the project brief). We used all available images. See `report/report.md` for details.

## Project Structure

```
ci-cervical-lbc/
├── configs/
│   └── experiment.yaml
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_tv.ipynb
│   ├── 03_unet.ipynb
│   ├── 04_diffpir.ipynb
│   └── 05_full_pipeline.ipynb
├── scripts/
│   ├── download_mendeley.py
│   ├── generate_crop_images.py
│   ├── generate_diffpir_2x4.py
│   ├── generate_noise_strip.py
│   ├── generate_qualitative_pngs.py
│   ├── plot_results.py
│   ├── preprocess.py
│   ├── run_diffpir.py
│   ├── run_tv.py
│   └── run_unet.py
├── slides/
│   └── presentation_discursive_en.tex
├── src/                        # Core library
│   ├── data/dataset.py
│   ├── degradation/degradation.py
│   ├── eval/metrics.py
│   ├── methods/
│   │   ├── tv/tv.py
│   │   ├── unet/unet.py
│   │   └── diffpir/  (model.py, train.py, diffpir.py)
│   ├── plots/visualize.py
│   └── utils.py
├── tests/                      # 34 unit tests
│   ├── test_degradation.py
│   ├── test_diffpir.py
│   ├── test_metrics.py
│   ├── test_tv.py
│   └── test_unet.py
├── results/                    # Metrics CSVs + qualitative PNGs
│   ├── tv/metrics.csv
│   ├── unet/metrics.csv
│   ├── diffpir/metrics.csv
│   └── qualitative_slides/     # Slide comparison PNGs
├── report/
│   └── theory.md
├── data/
│   └── degraded/examples/      # Sample degraded images
├── requirements.txt
└── README.md
```

## Reproducibility

- **Same degraded inputs** for all methods (`src/degradation/degradation.py`)
- **Fixed seed** 42 (`configs/experiment.yaml`)
- **Consistent evaluation**: PSNR/SSIM via `skimage.metrics`

## Contributors

- [Francesco Castaldi](https://github.com/FrancescoCastaldi)
- [Paolo Fusco](https://github.com/PaoloFusco19)
