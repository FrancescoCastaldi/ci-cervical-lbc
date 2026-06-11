# Call con Paolo — Meeting Notes

## Riunione 1 — Setup e Divisione Lavoro

- Deciso di escludere Weighted TV (siamo in 2)
- Assegnazione: Paolo → TV + UNet, Francesco → DiffPIR
- Struttura repository e pipeline condivisa

## Riunione 2 — Review TV e UNet

- TV completato (λ=0.005, 150 iter, PSNR 26.54–32.09 dB)
- UNet prima versione: 31M params, 1 epoca CPU, PSNR 21.87–24.07 dB
- Deciso di ottimizzare UNet per CPU

## Riunione 3 — UNet Ottimizzato

- Architettura snella: 16→32→64→128 canali (~500K params poi 1.9M)
- GroupNorm, L1 loss, noise conditioning
- 50 epoche CPU → **29.89 dB** (+5.8 dB)
- Supera TV ad alto rumore (28.93 vs 26.54 dB a σ=0.1)

## Riunione 4 — DiffPIR Completato, Confronto Finale

- DiffPIR: LightUNet 1.26M params, FFT data-fidelity, DDIM sampling
- Parametri euristici: t_start=50, λ=10, num_steps=15, ζ=0
- Risultati: PSNR cresce col rumore (16.67→24.68 dB)
- Generato comparison.png, qualitative per tutti i metodi

## Riunione 5 — Report, Slide e Consegna

- Report completato: teoria + relazione (673 righe)
- Slide PowerPoint (10 slide)
- Notebook: 01_eda, 02_tv, 03_unet, 04_diffpir
- 34 test unitari (degradation, metrics, diffpir, unet, tv)
- README.md aggiornato

## Stato Finale (Giugno 2026)

| Cosa | Stato |
|---|---|
| Dataset preprocessing | ✅ Completato |
| TV (variazionale) | ✅ Completato — PSNR 26.54–32.09 dB |
| UNet (end-to-end) | ✅ Completato — PSNR 28.93–29.89 dB |
| DiffPIR (generativo) | ✅ Completato — PSNR 16.67–24.68 dB |
| Weighted TV | ❌ Escluso (2 studenti) |
| Confronto quantitativo | ✅ PSNR/SSIM per tutti |
| Confronto qualitativo | ✅ 24 immagini per metodo |
| Grafico comparativo | ✅ comparison.png |
| Report | ✅ teoria.md + relazione.md |
| Slide | ✅ presentazione.pptx (10 slide) |
| Test unitari | ✅ 34 test |
| GitHub | ✅ https://github.com/FrancescoCastaldi/ci-cervical-lbc |
