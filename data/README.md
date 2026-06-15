# data/ — Dataset e Dati Processati

Contiene il dataset originale Mendeley, gli split stratificati e i dati degradati pre-calcolati per tutti i metodi.

## Utilità per l'esame

- **Dataset reale**: immagini LBC di citologia cervicale (962 immagini, 4 classi diagnostiche)
- **Split fissi**: train/val/test 70/15/15 — stesso split per tutti i metodi (confronto equo)
- **Degradazione identica**: tutti i metodi ricevono la stessa immagine degradata (confronto equo)

## Struttura del dataset Mendeley

`
data/raw/
├── NILM/       # Negative for Intraepithelial Lesion or Malignancy
├── HSIL/       # High-grade Squamous Intraepithelial Lesion
├── LSIL/       # Low-grade Squamous Intraepithelial Lesion
└── SCC/        # Squamous Cell Carcinoma
`

Ogni sottocartella contiene immagini PNG a 3 canali RGB a risoluzione variabile (~512×512).

## Statistiche classi

| Classe   | Conteggio | %     |
|----------|-----------|-------|
| NILM     | ~560      | 58.2% |
| HSIL     | ~160      | 16.6% |
| LSIL     | ~142      | 14.8% |
| SCC      | ~100      | 10.4% |
| Totale   | 962       | 100%  |

## Preprocessing pipeline

1. **Validazione**: PIL.Image.verify() scarta immagini corrotte
2. **Resize**: tutte le immagini portate a 256×256 pixel (bilinear)
3. **Normalizzazione**: ToTensor() [0,1] → Normalize(0.5, 0.5) → [-1, 1]
4. **Split stratificato**: 70/15/15 con seed=42, shuffle prima del subset

`python
from src.data.dataset import LBCDataset, DegradedDataset

# Carica immagini GT dal test set
ds = LBCDataset("data/splits/test.txt", image_size=256)
img_tensor = ds[0]  # shape: (3, 256, 256), range: [-1, 1]

# Carica coppie (degradata, GT) per noise_level=0.05
ds = DegradedDataset("data/degraded/test/noise_0.05", "data/degraded/test/gt")
deg, gt = ds[0]  # due tensori (3, 256, 256)
`

## Organizzazione file .pt

| Cartella | Contenuto |
|---|---|
| degraded/train/gt/ | Ground truth training (00000.pt ... 00673.pt) |
| degraded/train/noise_0.05/ | Degradate training (stessi indici) |
| degraded/val/gt/ | Ground truth validation |
| degraded/val/noise_0.01/ | Degradate validation per noise level |
| degraded/test/gt/ | Ground truth test |
| degraded/test/noise_0.1/ | Degradate test per noise level |

Ogni .pt contiene un tensore PyTorch 	orch.Tensor di shape (3, 256, 256).

## Generazione

`ash
python scripts/preprocess.py
`

Lo script esegue: (1) uild_splits() — scansione data/raw/, validazione, split stratificato; (2) generazione GT + degradate per ogni noise level (0.005, 0.01, 0.05, 0.1) con blur gaussiano (σ=2, kernel=9); (3) salvataggio esempi visivi in data/degraded/examples/.

> Il dataset è disponibile su [Mendeley](https://data.mendeley.com/datasets/zddtpgzv63/2).
> Dopo averlo scaricato in data/raw/, esegui python scripts/preprocess.py.
