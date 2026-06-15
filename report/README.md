# report/ — Documentazione e Report

Documentazione teorica, relazione di esame e materiali di studio.

## File

| File | Contenuto |
|---|---|
| 	heory.md | Fondamenti teorici: problema inverso, regolarizzazione TV, UNet, modelli diffusivi, metriche PSNR/SSIM |
| eport.md | Relazione completa: dataset, task, metodi, parametri, risultati, confronto critico, conclusioni |
| studio.md | Appunti di studio preparatori, derivazioni matematiche, note sui paper |
| 
otebook.md | Riassunto dei risultati dei notebook, tabelle riassuntive |

## Struttura di theory.md

1. **Problema inverso** — degradazione come y = (k ∗ x)↓ + n; formulazione MAP; ill-posedness
2. **Total Variation** — funzionale TV-L2, derivata, algoritmo Adam, interpretazione edge-preserving
3. **UNet** — architettura encoder-decoder con skip connections, GroupNorm, noise conditioning channel
4. **Modelli diffusivi** — forward SDE, reverse process, DDIM, DiffPIR con data-fidelity FFT
5. **Metriche** — PSNR (dB), SSIM (luminanza, contrasto, struttura), trade-off

## Struttura di report.md

1. **Introduzione** — contesto del dataset LBC Cervical Cancer, task di deblurring + denoising
2. **Dataset e preprocessing** — 962 immagini, resize 256×256, split 70/15/15, normalizzazione [-1, 1]
3. **Degradazione** — blur Gaussiano (σ=2, k=9) + 4 livelli AWGN, pipeline identica per tutti
4. **TV** — formulazione, parametri (λ=0.005, 150 iter), risultati con tabella PSNR/SSIM
5. **UNet** — architettura, training (L1, Adam, 50 epoche CPU), multi-noise augmentation
6. **DiffPIR** — DDPM training (LightUNet 1.26M params), DDIM sampling, FFT data-fidelity
7. **Confronto** — tabella comparativa, analisi per regime di rumore, tempi di inferenza
8. **Discussione** — successi e fallimenti, perché TV vince a basso rumore, UNet è più robusto
9. **Conclusioni** — summary, limiti, lavori futuri

## Relazione tra theory.md e report.md

- 	heory.md è atemporale — può essere usato per qualsiasi progetto di computational imaging
- eport.md è specifico per questo progetto — parametri, risultati, discussione
- 	heory.md fornisce le equazioni; eport.md applica quelle equazioni al dataset LBC
- Per l'esame, eport.md è il documento principale; 	heory.md è il riferimento per i fondamenti

## Destinatari

Commissione d'esame di Computational Imaging (Laurea Magistrale). Si assume che il lettore conosca: algebra lineare, basi di probabilità, reti neurali. Non si assume familiarità con modelli diffusivi o regolarizzazione variazione.

## Riferimenti

- Rudin, Osher, Fatemi (1992) — TV denoising
- Ronneberger, Fischer, Brox (2015) — U-Net
- Ho, Jain, Abbeel (2020) — Denoising Diffusion Probabilistic Models
- Song, Meng, Ermon (2021) — DDIM
- Zhang et al. (2022) — DiffPIR
- Wang et al. (2004) — SSIM

## Aggiornare il report

Dopo aver rieseguito gli esperimenti, aggiornare le tabelle in eport.md con i nuovi valori PSNR/SSIM. Le metriche sono generate da esults/*/metrics.csv.

