# data/ — Dataset e Dati Processati

Contiene il dataset originale, gli split, e i dati degradati pre-calcolati.

## Utilità per l'esame

- **Dataset reale**: immagini LBC di citologia cervicale (962 immagini, 4 classi diagnostiche)
- **Split fissi**: train/val/test 70/15/15 — stesso split per tutti i metodi (confronto equo)
- **Degradazione identica**: tutti i metodi ricevono la stessa immagine degradata (confronto equo)

## Struttura

| Cartella | Contenuto | Tracciato? |
|---|---|---|
| `raw/` | Immagini originali del dataset Mendeley | ❌ (`.gitignore`) |
| `splits/` | `train.txt`, `val.txt`, `test.txt` con percorsi immagini | ❌ (`.gitignore`) |
| `degraded/` | `.pt` files: GT e immagini degradate per ogni noise level | ❌ (`.gitignore`) |

> Il dataset è disponibile su [Mendeley](https://data.mendeley.com/datasets/zddtpgzv63/2).
> Dopo averlo scaricato in `data/raw/`, esegui `python scripts/preprocess.py` per generare splits e degraded.
