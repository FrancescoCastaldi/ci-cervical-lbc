# results/ — Experimental Results

Output of all three methods: quantitative metrics and qualitative images.

## Structure

```
results/
├── comparison.png                     # Comparative plot (all methods)
├── tv/
│   ├── metrics.csv                    # PSNR, SSIM per noise level
│   └── qualitative/                   # 24 PNG (6 per σₙ × 4)
├── unet/
│   ├── metrics.csv                    # PSNR, SSIM, inference time
│   ├── qualitative/                   # 24 PNG
│   └── best_model.pth                 # Model weights (not tracked)
├── diffpir/
│   ├── metrics.csv                    # PSNR, SSIM, inference time
│   └── qualitative/                   # 24 PNG
└── qualitative_slides/                # Grids for presentation
```

## CSV Format

```
method,noise_level,psnr,ssim[,avg_inference_time]
tv,0.005,31.87,0.907
unet,0.01,29.79,0.895,0.028
diffpir,0.1,25.60,0.797,0.64
```

## Image Naming Convention

`noise_{σₙ}_sample{id}.png` — e.g. `noise_0.05_sample2.png` = third test image with σₙ=0.05.

## Summary Metrics

| σₙ | TV (PSNR/SSIM) | UNet (PSNR/SSIM) | DiffPIR (PSNR/SSIM) |
|---|---|---|---|
| 0.005 | **31.87** / **0.907** | 29.79 / 0.896 | 15.80 / 0.356 |
| 0.01  | **31.82** / **0.905** | 29.79 / 0.895 | 16.48 / 0.407 |
| 0.05  | **30.23** / **0.834** | 29.44 / 0.864 | 22.77 / 0.722 |
| 0.1   | 26.42 / 0.583         | **28.46** / **0.795** | 25.60 / 0.797 |

## Results Interpretation

- **Low noise (σₙ ≤ 0.05)**: TV dominates — L2 regularization is optimal for weak AWGN, and blur is handled well by few gradient descent steps. UNet is competitive but slightly inferior due to imperfect generalization over 50 epochs on CPU.
- **High noise (σₙ = 0.1)**: UNet surpasses TV (+2.04 dB PSNR) — the network has learned a strong prior on cervical cytology that helps when the observation is very noisy. DiffPIR approaches (25.60 dB, +0.92 dB over the previous version) but does not reach UNet.
- **DiffPIR**: Lowest PSNR across all regimes but with significant improvements after full-dataset training (50 epochs, 1000 timesteps). The SSIM at σₙ=0.1 (0.797, +0.133) shows that structural quality has clearly improved. Possible residual causes: (1) LightUNet (1.26M params) insufficient to capture the complex distribution; (2) λ=10 not optimal for low σₙ.

## PSNR/SSIM Trade-off

- TV has the best SSIM at low noise (0.907) — preserves sharp edges thanks to L1 on the derivative.
- UNet maintains SSIM ≥ 0.795 across all levels — it is the most robust.
- DiffPIR SSIM at σₙ=0.005 (0.356, +0.121 vs previous version): hallucinations have been reduced thanks to full-dataset training, but the model still struggles at low noise.

## Inference Times (averaged over 10 tests)

| Method | σₙ=0.005 | σₙ=0.01 | σₙ=0.05 | σₙ=0.1 |
|---|---|---|---|---|
| TV | 0.59 s | 0.79 s | 0.58 s | 0.62 s |
| UNet | **0.030 s** | **0.028 s** | **0.027 s** | **0.026 s** |
| DiffPIR | **0.65 s** | **0.64 s** | **0.64 s** | **0.64 s** |

UNet is ~23× faster than TV and ~23× faster than DiffPIR — ideal for real-time applications. DiffPIR has improved efficiency by ~5.5× thanks to the new optimized weights.

## Regenerating Results

```bash
python scripts/run_tv.py        # results/tv/metrics.csv
python scripts/run_unet.py      # results/unet/metrics.csv
python scripts/run_diffpir.py   # results/diffpir/metrics.csv
python scripts/plot_results.py  # results/comparison.png
```

Each script overwrites the qualitative/ folder and metrics.csv file of its own method.

## Why Does DiffPIR Have Lower Scores?

The DDPM model has only 200 epochs of training on 2800 images — insufficient for a good prior estimate. State-of-the-art diffusion models require hundreds of thousands of steps. Additionally, the FFT data-fidelity assumes known blur, but the network was not scheduled with the exact test noise. DiffPIR remains valid as a demonstration of the generative approach on the LBC dataset.
