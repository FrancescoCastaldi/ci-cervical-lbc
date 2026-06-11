# src/data/ — Caricamento e Preprocessing Dataset

Modulo per il caricamento del dataset, creazione degli split, e trasformazioni.

## Utilità per l'esame

- **Pipeline condivisa**: tutti i metodi usano lo stesso `LBCDataset` — stesse immagini, stessa normalizzazione
- **Split stratificato**: ripartizione 70/15/15 con seed fisso — riproducibilità
- **Formato coerente**: resize 256×256, normalizzazione [-1, 1]

## File

| File | Contenuto |
|---|---|
| `dataset.py` | `LBCDataset`, `DegradedDataset`, `build_splits()`, `load_config()` |

## Classi principali

| Classe/Funzione | Ruolo |
|---|---|
| `LBCDataset` | Carica immagini da file split, applica resize + ToTensor + normalizzazione [-1, 1] |
| `DegradedDataset` | Carica coppie (degradata, GT) da file `.pt` pre-calcolati |
| `build_splits()` | Scansiona `data/raw/`, crea split stratificati, salva file `.txt` |
| `load_config()` | Carica `configs/experiment.yaml` |
