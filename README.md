# CI Cervical LBC

Computational Imaging project – **Deblur & Denoise** on the Mendeley LBC Cervical Cancer dataset.

Università di Bologna – LM Informatica – Computational Imaging
Prof. Picciolomini & Evangelista

## Task

Inverse problem: recover a high-quality image from a degraded observation (Gaussian blur + noise).

**Degradation parameters:**
- Gaussian blur: sigma=2, kernel_size=9
- Noise levels: 0.005, 0.01, 0.05, 0.1

## Methods

| Method | Family |
|---|---|
| Total Variation (TV) | Variational |
| UNet | End-to-end |
| DiffPir | Generative (Diffusion) |

## Dataset

Mendeley LBC Cervical Cancer – [download here](https://data.mendeley.com/datasets/zddtpgzv63/2)
Place the raw images in `data/raw/`.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
ci-cervical-lbc/
├── configs/          # Experiment configuration
├── data/             # Dataset (not tracked by git)
├── notebooks/        # Exploratory analysis
├── src/
│   ├── data/         # Dataset loading and preprocessing
│   ├── degradation/  # Degradation pipeline
│   ├── methods/      # TV, UNet, DiffPir
│   ├── eval/         # Metrics (PSNR, SSIM)
│   └── plots/        # Visualization utilities
├── scripts/          # Run experiments
├── results/          # Output metrics and figures (not tracked)
├── report/
└── slides/
```

## Evaluation

Metrics: **PSNR** and **SSIM** on the test set, for each method and noise level.

## Reproducibility

All experiments use the same degraded inputs and a fixed random seed (see `configs/experiment.yaml`).

## Running Experiments

To run the experiments for each method, use the provided scripts:

- Total Variation: `python scripts/run_tv.py`
- UNet: `python scripts/run_unet.py`
- DiffPir: `python scripts/run_diffpir.py`

Note: DiffPir requires the DiffPIR repository to be cloned and pretrained weights placed in `src/methods/diffpir/weights/`.

## Results

After running the scripts, quantitative results (PSNR, SSIM) will be saved in the `results/` directory, and qualitative examples will be saved in `results/` as well.

## Contributors

- [Francesco Castaldi](https://github.com/FrancescoCastaldi)
- [Paolo Fusco](https://github.com/PaoloFusco)

## License

This project is for educational purposes only.
