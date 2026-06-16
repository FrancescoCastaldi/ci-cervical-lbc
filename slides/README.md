# slides/ — Presentazione Orale (LaTeX)

Contiene la presentazione in formato LaTeX (Beamer) e i materiali per l'esposizione orale.

## Utilità per l'esame

- **Presentazione orale**: slide Beamer per l'esposizione del progetto (deliverable obbligatorio)
- **Immagini generate**: tutte le figure necessarie sono generate dagli script in `scripts/`
- **Compilazione**: richiede `pdflatex` — compilare localmente o su Overleaf

## File

| File | Descrizione |
|---|---|
| `presentazione.tex` | Slide per esposizione orale (Beamer, compatta) |
| `presentazione_discorsiva.tex` | Slide con note discorsive estese |
| `figures/` | Cartella con immagini per le slide (logo, figure qualitative) |

## Note

- Le immagini nelle slide sono generate da `scripts/generate_noise_strip.py` e `scripts/generate_crop_images.py`
- Il logo UniBO è in `figures/unibo_logo.png`
- I PDF compilati non sono tracciati in git (`.gitignore`)
