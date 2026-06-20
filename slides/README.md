# slides/ — Presentazione Orale

Contiene la presentazione PowerPoint e gli script per rigenerarla.

## File

| File | Descrizione |
|---|---|
| presentazione.pptx | Presentazione PowerPoint (~10 slide, 13.333×7.5") |
| generate_pptx.py | Script Python per rigenerare le slide da zero |
| _extract_pptx.py | Utility per estrarre testo dalla presentazione esistente |
| Exam_Assignment/Group R.pdf | Testo originale dell'esame (riferimento) |

## Struttura delle slide

1. **Copertina** — titolo, gruppo R, nomi
2. **Obiettivo** — deblurring + denoising di immagini citologiche cervicali; formulazione del problema inverso
3. **Dataset** — LBC Cervical Cancer: classi (NILM, HSIL, LSIL, SCC), 962 immagini, split 70/15/15
4. **Degradazione** — blur Gaussiano (σ=2) + AWGN (4 livelli); pipeline identica per confronto equo
5. **Metodo 1: TV** — regolarizzazione L1 sulle derivate; λ=0.005, 150 iter Adam; risultati
6. **Metodo 2: UNet** — architettura encoder-decoder (1.9M params); training L1 multi-noise; noise conditioning
7. **Metodo 3: DiffPIR** — DDPM (LightUNet 1.26M params) + DDIM sampling + FFT data-fidelity
8. **Risultati quantitativi** — tabella PSNR/SSIM per tutti i metodi e noise level; grafico comparativo
9. **Risultati qualitativi** — griglie di ricostruzioni affiancate; casi di successo e fallimento
10. **Conclusioni** — summary, trade-off, limiti, lavori futuri

## Rigenerare la presentazione

`ash
python slides/generate_pptx.py
`

Lo script:
1. Genera presentazione.pptx nella stessa cartella
2. Usa python-pptx (installare con pip install python-pptx)
3. Include immagini da esults/qualitative_slides/ (generate da scripts/plot_results.py)
4. Tema scuro con accenti blu (#00B4D8) e rosa (#E01E79)

Se esults/qualitative_slides/ è vuoto, eseguire prima:
`ash
python scripts/plot_results.py
`

## Suggerimenti per l'esposizione

| Slide | Durata consigliata | Punti chiave |
|---|---|---|
| 1-3 | 2 min | Contestualizzare il problema medico |
| 4-5 | 3 min | Degradazione + TV (baseline classica) |
| 6-7 | 4 min | UNet e DiffPIR (deep learning) |
| 8-9 | 4 min | Risultati: enfatizzare il confronto equo |
| 10 | 2 min | Conclusioni chiare, nessuna sorpresa |

Target totale: **15 minuti** di presentazione + 5 di domande.

## Risultati da enfatizzare

- TV è la migliore a basso rumore per la regolarizzazione ottimale L2+TV
- UNet è il più robusto su tutti i noise level, con SSIM sempre > 0.83
- UNet è 100× più veloce di TV e 80× più veloce di DiffPIR
- DiffPIR dimostra il potenziale dell'approccio generativo sul dataset LBC

## Personalizzazione

Modificare generate_pptx.py per:
- Cambiare colori (variabili DARK, ACCENT, ACCENT2)
- Aggiungere/rimuovere slide (modificare la sequenza di chiamate)
- Aggiornare tabelle risultati
- Sostituire immagini qualitative

