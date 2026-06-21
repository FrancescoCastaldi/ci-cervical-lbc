# report/ — Documentation and Report

Theoretical documentation, exam report, and study materials.

## Files

| File | Content |
|---|---|
| theory.md | Theoretical foundations: inverse problem, TV regularization, UNet, diffusion models, PSNR/SSIM metrics |
| report.md | Full report: dataset, task, methods, parameters, results, critical comparison, conclusions |
| studio.md | Preparatory study notes, mathematical derivations, paper notes |
| notebook.md | Summary of notebook results, summary tables |

## Structure of theory.md

1. **Inverse Problem** — degradation as y = (k ∗ x)↓ + n; MAP formulation; ill-posedness
2. **Total Variation** — TV-L2 functional, derivative, Adam algorithm, edge-preserving interpretation
3. **UNet** — encoder-decoder architecture with skip connections, GroupNorm, noise conditioning channel
4. **Diffusion Models** — forward SDE, reverse process, DDIM, DiffPIR with FFT data-fidelity
5. **Metrics** — PSNR (dB), SSIM (luminance, contrast, structure), trade-off

## Structure of report.md

1. **Introduction** — context of the LBC Cervical Cancer dataset, deblurring + denoising task
2. **Dataset and Preprocessing** — 962 images, resize 256×256, split 70/15/15, normalization [-1, 1]
3. **Degradation** — Gaussian blur (σ=2, k=9) + 4 AWGN levels, identical pipeline for all
4. **TV** — formulation, parameters (λ=0.005, 150 iter), results with PSNR/SSIM table
5. **UNet** — architecture, training (L1, Adam, 50 epochs CPU), multi-noise augmentation
6. **DiffPIR** — DDPM training (LightUNet 1.26M params), DDIM sampling, FFT data-fidelity
7. **Comparison** — comparative table, analysis by noise regime, inference times
8. **Discussion** — successes and failures, why TV wins at low noise, UNet is more robust
9. **Conclusions** — summary, limitations, future work

## Relationship Between theory.md and report.md

- `theory.md` is timeless — can be used for any computational imaging project
- `report.md` is specific to this project — parameters, results, discussion
- `theory.md` provides the equations; `report.md` applies those equations to the LBC dataset
- For the exam, `report.md` is the main document; `theory.md` is the reference for fundamentals

## Audience

Exam committee for Computational Imaging (Master's degree). Assumes the reader knows: linear algebra, basic probability, neural networks. Does not assume familiarity with diffusion models or variational regularization.

## References

- Rudin, Osher, Fatemi (1992) — TV denoising
- Ronneberger, Fischer, Brox (2015) — U-Net
- Ho, Jain, Abbeel (2020) — Denoising Diffusion Probabilistic Models
- Song, Meng, Ermon (2021) — DDIM
- Zhang et al. (2022) — DiffPIR
- Wang et al. (2004) — SSIM

## Updating the Report

After re-running experiments, update the tables in `report.md` with the new PSNR/SSIM values. The metrics are generated from `results/*/metrics.csv`.
