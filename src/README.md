# src/ — Codice Sorgente

Moduli Python per l'intera pipeline: caricamento dati, degradazione, metodi di restauro, metriche, visualizzazione.

## Utilità per l'esame

- **Modularità**: ogni metodo è indipendente in `methods/<metodo>/` — codice pulito e manutenibile
- **Pipeline condivisa**: degradazione, metriche e plot sono comuni — confronto equo garantito
- **Parametri euristicallyi**: ogni metodo ha i propri iperparametri, documentati nel report

## Sottomoduli

| Modulo | Contenuto | Ruolo |
|---|---|---|
| `data/` | `LBCDataset`, `build_splits`, `DegradedDataset` | Caricamento e preprocessing dataset |
| `degradation/` | `degrade()`, `gaussian_kernel()` | Pipeline blur + AWGN (identica per tutti) |
| `methods/tv/` | `tv_restore()`, `tv_loss()` | Total Variation (variazionale) |
| `methods/unet/` | `UNet`, `DoubleConv` | UNet end-to-end (deep learning) |
| `methods/diffpir/` | `DiffPIR`, `LightUNet`, training loop | DiffPIR generativo (diffusion) |
| `eval/` | `compute_psnr()`, `compute_ssim()`, `evaluate()` | Metriche di valutazione |
| `plots/` | `show_comparison()`, `plot_metrics()` | Visualizzazione risultati |

## Pipeline di esecuzione

```
data/ → degradation/ → methods/<metodo>/ → eval/ → plots/
```

Tre metodi indipendenti, un'unica pipeline di valutazione — questo è il punto chiave del progetto.
