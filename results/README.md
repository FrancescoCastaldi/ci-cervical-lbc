# results/ — Risultati Sperimentali

Output di tutti e tre i metodi: metriche quantitative e immagini qualitative.

## Utilità per l'esame

- **Confronto quantitativo**: PSNR e SSIM per ogni metodo e noise level
- **Confronto qualitativo**: immagini ricostruite affiancate a degraded e GT
- **Grafico riassuntivo**: `comparison.png` visualizza l'andamento PSNR/SSIM per tutti i metodi

## Struttura

```
results/
├── comparison.png          # Grafico comparativo (tutti i metodi insieme)
├── tv/
│   ├── metrics.csv         # PSNR, SSIM per ogni noise level
│   └── qualitative/        # 24 immagini (6 per noise level × 4)
├── unet/
│   ├── metrics.csv         # PSNR, SSIM, tempo inferenza
│   ├── qualitative/        # 24 immagini
│   └── best_model.pth      # Pesi del modello allenato (non tracciato)
└── diffpir/
    ├── metrics.csv         # PSNR, SSIM, tempo inferenza
    └── qualitative/        # 24 immagini
```

## Metriche riassuntive

| σₙ | TV (PSNR/SSIM) | UNet (PSNR/SSIM) | DiffPIR (PSNR/SSIM) |
|---|---|---|---|
| 0.005 | 32.09 / 0.911 | 29.79 / 0.896 | 15.78 / 0.329 |
| 0.01 | 32.04 / 0.909 | 29.79 / 0.895 | 16.45 / 0.374 |
| 0.05 | 30.42 / 0.837 | 29.44 / 0.864 | 22.64 / 0.677 |
| 0.1 | 26.54 / 0.586 | 28.46 / 0.795 | 25.46 / 0.766 |
