# src/data/ — Caricamento e Preprocessing Dataset

Modulo per il caricamento del dataset, creazione degli split stratificati e trasformazioni.

## Utilità per l'esame

- **Pipeline condivisa**: tutti i metodi usano lo stesso LBCDataset — stesse immagini, stessa normalizzazione
- **Split stratificato**: ripartizione 70/15/15 con seed fisso 42 — riproducibilità
- **Formato coerente**: resize 256×256, normalizzazione [-1, 1]

## File

| File | Contenuto |
|---|---|
| dataset.py | LBCDataset, DegradedDataset, uild_splits(), load_config() |

## API completa

### LBCDataset(split_file, image_size=256, return_path=False)

`python
ds = LBCDataset("data/splits/train.txt", image_size=256)
img = ds[0]                   # torch.Tensor, shape (3, 256, 256), range [-1, 1]
img, path = ds[0]             # con return_path=True
len(ds)                       # numero immagini nel subset
`

La transform applicata è:
`python
T.Compose([
    T.Resize((image_size, image_size)),   # bilinear interpolation
    T.ToTensor(),                         # uint8 → float [0, 1]
    T.Normalize(mean=[0.5], std=[0.5]),   # [0,1] → [-1,1]: x' = (x - 0.5) / 0.5
])
`

### DegradedDataset(degraded_dir, gt_dir)

`python
ds = DegradedDataset(
    "data/degraded/test/noise_0.05",
    "data/degraded/test/gt"
)
deg, gt = ds[0]   # due tensori (3, 256, 256), range [-1, 1]
`

I file .pt sono allineati per nome: stesso_indice.pt = stessa immagine.

### uild_splits(config)

1. Scansiona data/raw/ ricorsivamente per file .png e .jpg
2. Valida ogni immagine con PIL.Image.verify() (scarta corrotte)
3. Shuffle con andom.seed(42), prende subset di 962 immagini
4. Ripartizione: 70% train (673), 15% val (144), 15% test (145)
5. Salva 	rain.txt, al.txt, 	est.txt in data/splits/

`python
from src.data.dataset import load_config, build_splits
config = load_config()
splits = build_splits(config)
# splitts: {"train": [...], "val": [...], "test": [...]}
`

### load_config(path=None)

Carica configs/experiment.yaml come dict Python. Definisce seed, paths, parametri
di degradazione e iperparametri dei metodi.

## Normalizzazione [-1, 1]

`
input_uint8 [0, 255] → ToTensor → [0, 1] → Normalize → output [-1, 1]
inverso: restored = (tensor * 0.5 + 0.5).clamp(0, 1)
`

Tutti i metodi operano in range [-1, 1]. Le metriche riconvertono a [0, 1] internamente.
