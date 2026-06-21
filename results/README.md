# results/ — Risultati Sperimentali

Output di tutti e tre i metodi: metriche quantitative e immagini qualitative.

## Struttura

`
results/
├── comparison.png                     # Grafico comparativo (tutti i metodi)
├── tv/
│   ├── metrics.csv                    # PSNR, SSIM per ogni noise level
│   └── qualitative/                   # 24 PNG (6 per σₙ × 4)
├── unet/
│   ├── metrics.csv                    # PSNR, SSIM, tempo inferenza
│   ├── qualitative/                   # 24 PNG
│   └── best_model.pth                 # Pesi modello (non tracciato)
├── diffpir/
│   ├── metrics.csv                    # PSNR, SSIM, tempo inferenza
│   └── qualitative/                   # 24 PNG
└── qualitative_slides/                # Griglie per presentazione
`

## Formato CSV

`
method,noise_level,psnr,ssim[,avg_inference_time]
tv,0.005,32.09,0.911
unet,0.01,29.89,0.894,0.034
diffpir,0.1,24.68,0.664,2.893
`

## Convenzione nomi immagini


oise_{σₙ}_sample{id}.png — es. 
oise_0.05_sample2.png = terza immagine di test con σₙ=0.05.

## Metriche riassuntive

| σₙ | TV (PSNR/SSIM) | UNet (PSNR/SSIM) | DiffPIR (PSNR/SSIM) |
|---|---|---|---|
| 0.005 | **32.09** / **0.911** | 29.89 / 0.894 | 15.80 / 0.356 |
| 0.01  | **32.04** / **0.909** | 29.89 / 0.894 | 16.48 / 0.407 |
| 0.05  | **30.42** / **0.837** | 29.63 / 0.875 | 22.77 / 0.722 |
| 0.1   | 26.54 / 0.586         | **28.93** / **0.830** | 25.60 / 0.797 |

## Interpretazione dei risultati

- **Basso rumore (σₙ ≤ 0.05)**: TV domina — la regolarizzazione L2 è ottimale per AWGN debole, il blur è gestito bene da pochi step di gradient descent. UNet è competitivo ma leggermente inferiore per via della generalizzazione imperfetta su 50 epoche CPU.
- **Alto rumore (σₙ = 0.1)**: UNet supera TV (−3.61 dB PSNR) — la rete ha appreso una prior forte sulle citologie cervicali che aiuta quando l'osservazione è molto rumorosa. DiffPIR si avvicina (25.60 dB, +0.92 dB rispetto alla versione precedente) ma non raggiunge UNet.
- **DiffPIR**: PSNR più basso in tutti i regimi ma con miglioramenti significativi dopo il training full-dataset (50 epoche, 1000 timestep). Il SSIM a σₙ=0.1 (0.797, +0.133) mostra che la qualità strutturale è nettamente migliorata. Possibili cause residue: (1) LightUNet (1.26M params) insufficiente per catturare la distribuzione complessa; (2) λ=10 non ottimale per σₙ bassi.

## Trade-off PSNR/SSIM

- TV ha il miglior SSIM a basso rumore (0.911) — preserva bordi netti grazie all'L1 sulla derivata.
- UNet mantiene SSIM > 0.83 su tutti i livelli — è il più robusto.
- DiffPIR ha SSIM a σₙ=0.005 (0.356, +0.121 vs versione precedente): le allucinazioni si sono ridotte grazie al training full-dataset, ma il modello fatica ancora a basso rumore.

## Tempi di inferenza (medi su 10 test)

| Metodo | σₙ=0.005 | σₙ=0.01 | σₙ=0.05 | σₙ=0.1 |
|---|---|---|---|---|
| TV | ~5 s | ~5 s | ~5 s | ~5 s |
| UNet | **0.035 s** | **0.034 s** | **0.034 s** | **0.036 s** |
| DiffPIR | **0.65 s** | **0.64 s** | **0.64 s** | **0.64 s** |

UNet è ~15× più veloce di TV e ~20× più veloce di DiffPIR — ideale per applicazioni real-time. DiffPIR ha migliorato l'efficienza di ~5.5× grazie ai nuovi pesi ottimizzati.

## Rigenerare i risultati

`ash
python scripts/run_tv.py        # results/tv/metrics.csv
python scripts/run_unet.py      # results/unet/metrics.csv
python scripts/run_diffpir.py   # results/diffpir/metrics.csv
python scripts/plot_results.py  # results/comparison.png
`

Ogni script sovrascrive la cartella qualitative/ e il file metrics.csv del proprio metodo.

## Perché DiffPIR ha punteggi più bassi?

Il modello DDPM ha solo 200 epoche di training su 2800 immagini — insufficiente per una buona stima della prior. I modelli diffusivi di stato dell'arte richiedono centinaia di migliaia di step. Inoltre, il data-fidelity FFT assume blur noto, ma la rete non è stata schedulata con il rumore esatto del test. DiffPIR rimane valido come dimostrazione dell'approccio generativo sul dataset LBC.

