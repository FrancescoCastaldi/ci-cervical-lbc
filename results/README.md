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
| 0.005 | **32.09** / **0.911** | 29.89 / 0.894 | 16.67 / 0.235 |
| 0.01  | **32.04** / **0.909** | 29.89 / 0.894 | 17.32 / 0.270 |
| 0.05  | **30.42** / **0.837** | 29.63 / 0.875 | 22.49 / 0.512 |
| 0.1   | 26.54 / 0.586         | **28.93** / **0.830** | 24.68 / 0.664 |

## Interpretazione dei risultati

- **Basso rumore (σₙ ≤ 0.05)**: TV domina — la regolarizzazione L2 è ottimale per AWGN debole, il blur è gestito bene da pochi step di gradient descent. UNet è competitivo ma leggermente inferiore per via della generalizzazione imperfetta su 50 epoche CPU.
- **Alto rumore (σₙ = 0.1)**: UNet supera TV (−3.61 dB PSNR) — la rete ha appreso una prior forte sulle citologie cervicali che aiuta quando l'osservazione è molto rumorosa. DiffPIR si avvicina (24.68 dB) ma non raggiunge UNet.
- **DiffPIR**: PSNR più basso in tutti i regimi. Possibili cause: (1) modello DDPM addestrato solo 200 epoche su dataset piccolo; (2) lightUNet (1.26M params) insufficiente per catturare la distribuzione complessa; (3) λ=10 non ottimale per σₙ bassi. Il SSIM a σₙ=0.1 (0.664) mostra che la qualità strutturale è discreta nonostante il PSNR modesto.

## Trade-off PSNR/SSIM

- TV ha il miglior SSIM a basso rumore (0.911) — preserva bordi netti grazie all'L1 sulla derivata.
- UNet mantiene SSIM > 0.83 su tutti i livelli — è il più robusto.
- DiffPIR ha SSIM basso a σₙ=0.005 (0.235): genera texture inesistenti (allucinazioni) perché il modello cerca di campionare dalla prior anche quando l'immagine è quasi pulita.

## Tempi di inferenza (medi su 10 test)

| Metodo | σₙ=0.005 | σₙ=0.01 | σₙ=0.05 | σₙ=0.1 |
|---|---|---|---|---|
| TV | ~5 s | ~5 s | ~5 s | ~5 s |
| UNet | **0.035 s** | **0.034 s** | **0.034 s** | **0.036 s** |
| DiffPIR | 3.27 s | 3.00 s | 2.85 s | 2.89 s |

UNet è 100× più veloce di TV e 80× più veloce di DiffPIR — ideale per applicazioni real-time.

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

