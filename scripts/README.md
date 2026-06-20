# scripts/ — Script di Esecuzione

Pipeline completa per il preprocessing, esecuzione di tutti i metodi, e generazione dei risultati.

## Utilità per l'esame

- **Riproducibilità**: eseguendo gli script in ordine si rigenerano tutti i risultati
- **Automazione**: ogni metodo ha il suo script dedicato
- **Orchestrazione**: lo script `plot_results.py` raccoglie i risultati di tutti i metodi

## Pipeline completa

```bash
# 0. Prerequisiti: dataset scaricato in data/raw/, pesi UNet/DiffPIR disponibili
# 1. Preprocessing (una volta sola, ~2 min)
python scripts/preprocess.py

# 2. Esecuzione metodi (indipendenti, ordine arbitrario)
python scripts/run_tv.py          # Total Variation (~20 min su 145 immagini)
python scripts/run_unet.py        # UNet training + eval (~2 h CPU)
python scripts/run_diffpir.py     # DiffPIR su test set (~30 min)

# 3. Confronto finale e metriche
python scripts/plot_results.py    # Grafico comparativo PSNR/SSIM

# 4. Immagini per presentazione (opzionale)
python scripts/generate_noise_strip.py       # noise_strip_crops.png
python scripts/generate_crop_images.py       # crop_comparison.png + crop_diff_comparison.png
python scripts/generate_qualitative_pngs.py  # PNG qualitative supplementari
python scripts/generate_diffpir_2x4.py       # DiffPIR 2×4 comparison
```

## Output generati

```
results/
├── comparison.png              # Grafico PSNR/SSIM comparativo
├── tv/metrics.csv              # PSNR, SSIM per ogni immagine
├── tv/qualitative/             # 24 immagini confronto (6 per noise level)
├── unet/metrics.csv
├── unet/qualitative/
├── diffpir/metrics.csv
└── diffpir/qualitative/
```

## Dettaglio script

| Script | Durata | Cosa fa internamente |
|---|---|---|
| `preprocess.py` | ~2 min | Legge raw .jpg, resize 256×256, split 70/15/15, applica degrade(), salva .pt |
| `run_tv.py` | ~20 min | Itera test set (145 img), chiama tv_restore() per 4 noise level, calcola PSNR/SSIM, salva CSV e PNG |
| `run_unet.py` | ~2 h (CPU) | Addestra UNet (multi-noise, 50 epoche, early stopping), salva best model, esegui eval su test set |
| `run_diffpir.py` | ~30 min | Carica LightUNet pesi, chiama diffpir_restore() su test set per 4 noise level |
| `plot_results.py` | ~10 s | Legge tutti i metrics.csv, genera bar chart e line plot comparativi |
| `gen_tv_qual.py` | ~5 min | Solo generazione qualitative TV (utile per rigenerare senza rieseguire metriche) |
| `eval_unet.py` | ~10 min | Carica modello pre-addestrato, esegue solo evaluation (senza training) |

## Prerequisiti e configurazione

- **Dataset**: deve essere scaricato manualmente in `data/raw/` (Mendeley LBC Cervical Cancer)
- **Pesi UNet**: generati da `run_unet.py` in `src/methods/unet/`
- **Pesi DiffPIR**: pre-addestrati in `src/methods/diffpir/weights/`
- **Config**: parametri modificabili in `configs/experiment.yaml`

## Esecuzione su singola immagine

```python
# Esempio: test rapido con TV
from src.methods.tv.tv import tv_restore
from src.data.dataset import LBCDataset

ds = LBCDataset(split="test")  # 145 immagini
x_clean, _ = ds[0]             # prima immagine clean
x_tv, metrics = tv_restore(x_clean.unsqueeze(0), lmbda=0.005, n_iter=150)
```

## Risoluzione problemi

- `FileNotFoundError` in data/: eseguire prima `preprocess.py`
- `ModuleNotFoundError`: verificare `pip install -e .` o PYTHONPATH
- Memoria insufficiente: ridurre batch_size in experiment.yaml
- Tempi lunghi: su GPU i metodi DL sono 10-50× più veloci
