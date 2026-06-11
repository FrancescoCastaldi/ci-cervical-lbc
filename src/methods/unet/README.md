# UNet — Metodo Deep Learning End-to-End

Architettura UNet con condizionamento del noise level per deblur + denoise.

## Utilità per l'esame

- **Deep learning end-to-end**: impara direttamente la mappatura degraded → restored dai dati
- **Architettura snella**: ~1.9M parametri (vs 31M originale) — addestrabile su CPU
- **Multi-noise augmentation**: training con noise level random per batch — generalizza su tutti i livelli

## File

| File | Contenuto |
|---|---|
| `unet.py` | `UNet`, `DoubleConv` — architettura encoder-decoder con skip connections |

## Architettura

| Componente | Dettaglio |
|---|---|
| Input | 4 canali (RGB + noise map), 256×256 |
| Encoder | 4 livelli: 16 → 32 → 64 → 128 canali |
| Decoder | 4 livelli simmetrici con upsampling |
| Skip connections | Connessioni encoder → decoder |
| Output | 3 canali [-1, 1] |
| Parametri | ~1.9M |

## Training

| Parametro | Valore |
|---|---|
| Loss | L1 (MSE meno blur) |
| Ottimizzatore | Adam, lr=10⁻⁴ |
| Batch size | 16 |
| Epoche | 50 |
| Augmentation | σₙ random per batch |

## Risultati (145 test images × 4 noise level)

| σₙ | PSNR | SSIM | Tempo |
|---|---|---|---|
| 0.005 | 29.89 dB | 0.894 | 0.035 s |
| 0.01 | 29.89 dB | 0.894 | 0.034 s |
| 0.05 | 29.63 dB | 0.875 | 0.034 s |
| 0.1 | 28.93 dB | 0.830 | 0.036 s |
