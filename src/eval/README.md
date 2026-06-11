# src/eval/ — Metriche di Valutazione

Implementa PSNR e SSIM per la valutazione quantitativa dei metodi di restauro.

## Utilità per l'esame

- **Metriche standard**: PSNR e SSIM sono le metriche richieste dall'esame per il confronto
- **Funzione unificata**: `evaluate()` calcola entrambe le metriche in una chiamata — usata da tutti i metodi

## File

| File | Contenuto |
|---|---|
| `metrics.py` | `compute_psnr()`, `compute_ssim()`, `evaluate()` |

## Funzioni

| Funzione | Ruolo |
|---|---|
| `compute_psnr(restored, gt)` | PSNR in dB su immagini [0, 1] |
| `compute_ssim(restored, gt)` | SSIM (skimage) |
| `evaluate(restored, gt)` | Dizionario con PSNR e SSIM |
