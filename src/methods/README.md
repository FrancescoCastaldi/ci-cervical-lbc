# src/methods/ — Metodi di Restauro

Implementazioni dei tre metodi di restauro: variazionale (TV), end-to-end (UNet), generativo (DiffPIR).

## Utilità per l'esame

- **Copertura metodologica**: tre famiglie diverse — variazionale, deep learning, generativo
- **Indipendenza**: ogni metodo è un modulo separato con il proprio codice
- **Confronto critico**: stessi input, stesse metriche — si valutano punti di forza e limiti di ciascuno

## Metodi

| Modulo | Metodo | Famiglia | File principale | Parametri chiave |
|---|---|---|---|---|
| `tv/` | Total Variation | Variazionale | `tv.py` | λ_reg=0.005, iter=150 |
| `unet/` | UNet | Deep Learning (end-to-end) | `unet.py` | 1.9M params, GroupNorm |
| `diffpir/` | DiffPIR | Generativo (diffusion) | `diffpir.py` | ζ=1.0, step=30, DDIM |

## Risultati comparativi (PSNR / SSIM su test set)

| σ_n | TV | UNet | DiffPIR |
|---|---|---|---|
| 0.005 | 32.09 dB / 0.911 | 29.89 dB / 0.894 | 16.67 dB / 0.235 |
| 0.01  | 32.04 dB / 0.909 | 29.89 dB / 0.894 | 17.32 dB / 0.270 |
| 0.05  | 30.42 dB / 0.837 | 29.63 dB / 0.875 | 22.49 dB / 0.512 |
| 0.1   | 26.54 dB / 0.586 | 28.93 dB / 0.830 | 24.68 dB / 0.664 |

**TV** domina a basso rumore (σ_n ≤ 0.01), **UNet** è il più robusto su tutti i livelli, **DiffPIR** recupera a rumore alto ma soffre a σ_n basso.

## Complessità computazionale (CPU, 256×256)

| Metodo | Tempo per immagine | Note |
|---|---|---|
| TV | ~8 s | 150 iterazioni Adam |
| UNet | ~0.035 s | Singolo forward pass |
| DiffPIR | ~3 s | 30 step DDIM sampling |

## Quando usare ciascun metodo

- **TV**: quando non si hanno dati di training e serve una baseline interpretabile
- **UNet**: quando si hanno dati simili alla distribuzione di training, serve velocità in inferenza
- **DiffPIR**: quando si accetta più costo computazionale per qualità generativa e dettagli fini

## Struttura dei moduli

```
tv/
├── tv.py          # TV loss + optimizer loop
└── __init__.py

unet/
├── unet.py        # Architettura UNet (encoder-decoder, skip connections)
└── __init__.py

diffpir/
├── diffpir.py     # Algoritmo DiffPIR (FFT data-fidelity, DDIM)
├── model.py       # LightUNet per DDPM (1.26M params)
├── train.py       # Training loop DDPM
├── weights/       # Pesi pre-addestrati (non tracciati)
└── README.md      # Documentazione specifica
```

## Esempio: eseguire un metodo su una singola immagine

```python
from src.degradation.degradation import degrade
from src.methods.tv.tv import tv_restore
from src.methods.unet.unet import UNet, unet_restore
from src.methods.diffpir.diffpir import diffpir_restore

x_clean = torch.randn(1,3,256,256)
x_noisy, _ = degrade(x_clean, noise_level=0.05)

x_tv, _ = tv_restore(x_noisy, lmbda=0.005, n_iter=150)
x_unet = unet_restore(x_noisy, model, device)
x_diffpir = diffpir_restore(x_noisy, model, zeta=1.0, steps=30)
```
