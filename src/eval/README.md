# src/eval/ — Metriche di Valutazione

Implementa PSNR e SSIM per la valutazione quantitativa dei metodi di restauro.

## Utilità per l'esame

- **Metriche standard**: PSNR e SSIM sono le metriche richieste dall'esame per il confronto
- **Funzione unificata**: `evaluate()` calcola entrambe le metriche in una chiamata - usata da tutti i metodi

## File

| File | Contenuto |
|---|---|
| `metrics.py` | `compute_psnr()`, `compute_ssim()`, `evaluate()` |

## API completa

### `compute_psnr(pred, gt)` - PSNR in dB

Formula:
```
PSNR = 10 * log10( MAX^2 / MSE )
     = 20 * log10( 1 / sqrt(MSE) )
```
con `MAX = 1.0` (range [0, 1]) e `MSE = mean((pred - gt)^2)`.

### `compute_ssim(pred, gt)` - SSIM

SSIM confronta luminanza, contrasto e struttura tra due finestre:
```
SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)
```
Implementata via `skimage.metrics.structural_similarity` con `channel_axis=-1`.

### `evaluate(pred, gt)` - Metrica combinata

```python
from src.eval.metrics import compute_psnr, compute_ssim, evaluate

# Singola immagine
psnr_val = compute_psnr(restored, gt)     # es. 29.89 dB
ssim_val = compute_ssim(restored, gt)     # es. 0.894

# Metrica combinata
result = evaluate(restored, gt)
print(result)  # {"psnr": 29.89, "ssim": 0.894}
```

### Conversione range

I tensori PyTorch arrivano in range [-1, 1]. La funzione interna `to_numpy()` converte:
```python
img = tensor.detach().cpu().permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
img = img * 0.5 + 0.5                                   # [-1,1] -> [0,1]
img = np.clip(img, 0.0, 1.0)
```

### Casistiche particolari

| Condizione                  | PSNR      | SSIM    |
|-----------------------------|-----------|---------|
| pred == gt                  | inf (inf) | 1.0     |
| Immagine nera (tutto 0)     | ~10-20 dB | ~0.0-0.1|
| Immagine bianca (tutto 1)   | ~10-20 dB | ~0.0-0.1|
| Valori fuori range          | clip a [0,1] automatico | |

### Utilizzo negli script

```python
# Estratto da scripts/run_tv.py / run_unet.py / run_diffpir.py
from src.eval.metrics import evaluate
metrics = evaluate(restored_img, gt_img)
print(f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.3f}")
```