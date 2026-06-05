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

| Method | Family | Status |
|---|---|---|
| Total Variation (TV) | Variational | 🔄 Da eseguire |
| UNet | End-to-end | 🔄 Da eseguire |
| DiffPIR | Generative (Diffusion) | ✅ Completato |

## DiffPIR Results (10 test images)

| σ_n | PSNR | SSIM | Tempo |
|---|---|---|---|
| 0.005 | 16.67 dB | 0.235 | 2.0 s |
| 0.01 | 17.32 dB | 0.270 | 2.0 s |
| 0.05 | 22.49 dB | 0.512 | 2.0 s |
| 0.1 | 24.68 dB | 0.664 | 2.0 s |

## Dataset

Mendeley LBC Cervical Cancer – [download here](https://data.mendeley.com/datasets/zddtpgzv63/2)
Place the raw images in `data/raw/`.

## Setup

```bash
pip install -r requirements.txt
```

Running DiffPIR requires a trained DDPM model (included: `src/methods/diffpir/weights/ddpm_lbc.pt`).
To retrain:
```bash
python -m src.methods.diffpir.train
```

## Project Structure

```
ci-cervical-lbc/
├── configs/                    # experiment.yaml (seed, paths, parametri)
├── data/
│   ├── raw/                    # Dataset originale (non tracciato)
│   └── splits/                 # Split train/val/test .txt
├── notebooks/
│   ├── 01_eda.ipynb            # EDA completo (classi, statistiche)
│   └── 04_diffpir.ipynb        # DiffPIR demo + valutazione
├── src/
│   ├── data/dataset.py         # LBCDataset, build_splits
│   ├── degradation/degradation.py  # Blur + noise pipeline
│   ├── methods/
│   │   ├── tv/tv.py            # Total Variation
│   │   ├── unet/unet.py        # UNet architecture
│   │   └── diffpir/            # DiffPIR (modulo autonomo)
│   │       ├── diffpir.py      # Algoritmo DiffPIR
│   │       ├── model.py        # LightUNet per DDPM
│   │       ├── train.py        # Training DDPM
│   │       ├── weights/        # Pesi addestrati
│   │       └── README.md       # Documentazione metodo
│   ├── eval/metrics.py         # PSNR, SSIM
│   └── plots/visualize.py      # Comparazione, plot metriche
├── scripts/
│   ├── preprocess.py           # Preprocessing dataset
│   ├── run_tv.py               # Esecuzione TV
│   ├── run_unet.py             # Esecuzione UNet
│   ├── run_diffpir.py          # Esecuzione DiffPIR
│   └── plot_results.py         # Grafico comparativo finale
├── report/
│   ├── teoria.md               # Fondamenti teorici
│   └── notebook.md             # Riassunto risultati notebook
├── slides/                     # Materiale presentazione
├── agents.md                   # Stato progetto e divisione lavoro
└── README.md                   # Questo file
```

## Running Experiments

To run the experiments for each method, use the provided scripts:

- Total Variation: `python scripts/run_tv.py`
- UNet: `python scripts/run_unet.py`
- DiffPIR: `python scripts/run_diffpir.py`

After all methods have results, generate the comparison plot:
```bash
python scripts/plot_results.py
```

## Evaluation

Metrics: **PSNR** and **SSIM** on the test set, for each method and noise level.

Output structure:
```
results/
├── comparison.png              # Grafico comparativo
├── diffpir/
│   ├── metrics.csv             # PSNR, SSIM, tempo
│   └── qualitative/            # Immagini di confronto
├── tv/                         # (dopo esecuzione)
└── unet/                       # (dopo esecuzione)
```

## Reproducibility

All experiments use the same degraded inputs and a fixed random seed (`seed=42` in `configs/experiment.yaml`).
The degradation pipeline (`src/degradation/degradation.py`) applies identical blur + noise to all methods.

## Contributors

- [Francesco Castaldi](https://github.com/FrancescoCastaldi) — DiffPIR
- [Paolo Fusco](https://github.com/PaoloFusco) — TV, UNet

## License

This project is for educational purposes only.
