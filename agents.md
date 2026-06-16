# Agents — CI Cervical LBC

## Divisione Metodi

| Metodo | Responsabile | Stato |
|---|---|---|
| **Preprocessing & Degradation** | Condiviso | ✅ Completato |
| **TV** (Variazionale) | Paolo | ✅ Completato |
| **UNet** (End-to-End) | Paolo | ✅ Completato |
| **DiffPIR** (Generativo) | Francesco | ✅ Completato |

Ogni metodo vive in un modulo separato (`src/methods/<metodo>/`), mentre preprocessing, metriche
e visualizzazione sono condivisi (`src/data/`, `src/degradation/`, `src/eval/`, `src/plots/`).

---

## Piano di Lavoro Settimanale

### Giorni 1–2 — Dataset & Preprocessing ✅
- Scaricare e ispezionare il dataset Mendeley LBC Cervical Cancer
- Capire classi (NILM, HSIL, LSIL, SCC), formato, immagini corrotte
- Definire subset (962 immagini, resize 256×256)
- Pipeline: resize, normalizzazione [-1, 1], split fisso 70/15/15
- ✅ `src/data/dataset.py` — `LBCDataset`, `build_splits`
- ✅ `configs/experiment.yaml` — configurazione seed, split, paths

### Giorni 3–4 — Degradazione identica per tutti ✅
- Implementare blur gaussiano (σ=2, kernel=9) + AWGN (4 livelli)
- Stessa pipeline per ogni metodo — confronto equo
- ✅ `src/degradation/degradation.py` — `degrade()`, `gaussian_kernel()`

### Giorni 5–6 — TV (Baseline classica)
- Implementare Total Variation come baseline
- λ_reg = 0.005, 150 iterazioni Adam
- ✅ `src/methods/tv/tv.py` — implementato
- ✅ `scripts/run_tv.py` — script di esecuzione
- ✅ Eseguito con risultati (PSNR: 26.54–32.09 dB)

### Giorni 7–9 — UNet (Deep Learning End-to-End) ✅
- Architettura encoder-decoder con skip connections (64→512 canali)
- Training L1 + Adam, multi-noise augmentation, validation, best model saving
- ✅ `src/methods/unet/unet.py` — implementato (1.9M params, GroupNorm, noise conditioning)
- ✅ `scripts/run_unet.py` — riscritto (multi-noise, validation, CPU-optimized)
- ✅ Eseguito con risultati (PSNR: 28.46–29.79 dB, 50 epoche CPU)

### Giorni 10–11 — DiffPIR (Generativo) ✅
- Modello LightUNet custom (1.26M params) addestrato su LBC
- FFT-based data-fidelity, DDIM sampling
- ✅ `src/methods/diffpir/diffpir.py` — algoritmo completo
- ✅ `src/methods/diffpir/model.py` — architettura DDPM
- ✅ `src/methods/diffpir/train.py` — training loop
- ✅ `scripts/run_diffpir.py` — eseguito con risultati
- ✅ `results/diffpir/metrics.csv` — PSNR/SSIM/tempo su 10 test

### Giorni 12–13 — Valutazione e Confronto ✅
- Calcolare PSNR, SSIM su test set per ogni noise level
- Generare tabelle e plot comparativi
- Salvare immagini qualitative (TV: 24, UNet: 24, DiffPIR: 24)
- ✅ `src/eval/metrics.py` — compute_psnr(), compute_ssim(), evaluate()
- ✅ `src/plots/visualize.py` — show_comparison(), plot_metrics()
- ✅ `scripts/plot_results.py` — genera comparison.png (3 metodi)
- ✅ Qualitative per tutti e 3 i metodi completate

### Giorno 14 — Report, Slide e Consegna ✅
- Scrivere report (teoria + risultati)
- Preparare slide
- Pulire repo e README
- ✅ `report/teoria.md` — teoria completa
- ✅ `report/relazione.md` — relazione completa (673 righe)
- ✅ `report/notebook.md` — riassunto notebook
- ✅ `slides/presentazione.tex` — Presentazione Beamer (LaTeX)
- ✅ `notebooks/01_eda.ipynb` through `04_diffpir.ipynb`
- ✅ 34 test unitari: degradation (10), metrics (9), diffpir (7), unet (8), tv (8)
- ✅ `README.md` — aggiornato con tutti i risultati

---

## Risultati Correnti

### DiffPIR (10 immagini test × 4 noise level)

| σ_n | PSNR | SSIM | Tempo |
|---|---|---|---|
| 0.005 | 15.78 dB | 0.329 | 3.59 s |
| 0.01 | 16.45 dB | 0.374 | 3.79 s |
| 0.05 | 22.64 dB | 0.677 | 3.73 s |
| 0.1 | 25.46 dB | 0.766 | 3.81 s |

### TV (145 immagini test × 4 noise level)

| σ_n | PSNR | SSIM |
|---|---|---|
| 0.005 | 32.09 dB | 0.911 |
| 0.01 | 32.04 dB | 0.909 |
| 0.05 | 30.42 dB | 0.837 |
| 0.1 | 26.54 dB | 0.586 |

### UNet (145 immagini test × 4 noise level, 50 epoche CPU)

| σ_n | PSNR | SSIM | Tempo |
|---|---|---|---|
| 0.005 | **29.79 dB** | **0.896** | **0.030 s** |
| 0.01 | **29.79 dB** | **0.895** | **0.028 s** |
| 0.05 | **29.44 dB** | **0.864** | **0.027 s** |
| 0.1 | **28.46 dB** | **0.795** | **0.026 s** |

### Output organizzati
```
results/
├── comparison.png              # Grafico comparativo (3 metodi)
├── diffpir/
│   ├── metrics.csv             # PSNR, SSIM, tempo
│   └── qualitative/            # 24 immagini (6 per noise level)
├── tv/
│   ├── metrics.csv             # PSNR, SSIM
│   └── qualitative/            # 24 immagini (6 per noise level)
└── unet/
    ├── metrics.csv             # PSNR, SSIM, tempo
    └── qualitative/            # 24 immagini (6 per noise level)
```

## Struttura del Progetto

```
ci-cervical-lbc/
├── configs/                 # experiment.yaml (seed, paths, parametri)
├── data/
│   ├── raw/                 # Dataset originale (non tracciato)
│   ├── splits/              # Split train/val/test .txt
│   └── degraded/            # Degradate pre-calcolate
├── notebooks/
│   ├── 01_eda.ipynb         # EDA completo (classi, statistiche)
│   ├── 02_tv.ipynb          # Total Variation demo + valutazione
│   ├── 03_unet.ipynb        # UNet training + valutazione
│   └── 04_diffpir.ipynb    # DiffPIR demo + valutazione
├── src/
│   ├── data/dataset.py      # LBCDataset, DegradedDataset, build_splits
│   ├── degradation/degradation.py  # blur + noise pipeline
│   ├── methods/
│   │   ├── tv/tv.py         # Total Variation
│   │   ├── unet/unet.py     # UNet architecture
│   │   └── diffpir/         # DiffPIR (modulo autonomo)
│   │       ├── diffpir.py   # Algoritmo DiffPIR
│   │       ├── model.py     # LightUNet per DDPM
│   │       ├── train.py     # Training DDPM
│   │       ├── weights/     # Pesi addestrati (non tracciati)
│   │       └── README.md    # Documentazione metodo
│   ├── eval/metrics.py      # PSNR, SSIM
│   └── plots/visualize.py   # Comparazione, plot metriche
├── scripts/
│   ├── preprocess.py        # Preprocessing dataset
│   ├── run_tv.py            # Esecuzione TV
│   ├── run_unet.py          # Esecuzione UNet
│   ├── run_diffpir.py       # Esecuzione DiffPIR
│   └── plot_results.py      # Grafico comparativo finale
├── report/
│   ├── teoria.md            # Fondamenti teorici (inverso, metodi, metriche)
│   └── notebook.md          # Riassunto risultati notebook
├── README.md                # Panoramica progetto
└── agents.md                # Questo file
```

## Criteri di Valutazione

Il progetto sarà valutato su:

1. **Stessi input degradati** per tutti i metodi ✅ (`src/degradation/degradation.py`)
2. **Scelte motivate dei parametri** (riportato in `report/theory.md`)
3. **Confronto serio** (non solo PSNR più alto)
4. **Esempi di successi e fallimenti**
5. **Organizzazione del codice** — modulare, documentato

## Stato Finale

Il progetto è completato. Tutti i metodi sono implementati, valutati e confrontati.
Vedi `README.md` per i risultati completi e `report/report.md` per la discussione.
