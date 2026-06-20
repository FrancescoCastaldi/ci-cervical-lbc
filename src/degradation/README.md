# src/degradation/ — Pipeline di Degradazione

Implementa la degradazione blur + rumore, **identica per tutti i metodi**.

## Utilità per l'esame

- **Confronto equo**: stessa funzione `degrade()` per TV, UNet e DiffPIR — le differenze nei risultati sono solo dovute ai metodi
- **Parametri fissi**: blur gaussiano (σ=2, kernel=9) e 4 livelli di AWGN (0.005, 0.01, 0.05, 0.1)
- **Seed controllato**: seed 42 per riproducibilità

## Formulazione matematica

L'immagine degradata _y_ è ottenuta dal modello diretto:

$$ y = (H * x) + n $$

dove _H_ è il kernel di blur gaussiano, _*_ è la convoluzione 2D, e _n_ ∼ N(0, σ²I) è rumore AWGN indipendente.

Il kernel gaussiano si calcola come:

$$ G(i,j) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{i^2 + j^2}{2\sigma^2}\right) $$

con _i, j_ ∈ [-K/2, K/2] per kernel di dimensione _K=9_, normalizzato a somma 1.

## Perché questi parametri?

| Parametro | Scelta | Motivazione |
|---|---|---|
| σ_blur = 2 | Blur moderato | Simula sfocatura da microscopio ottico reale |
| Kernel 9×9 | 99.7% energia gaussiana | Dimensione sufficiente per σ=2 senza artefatti di troncamento |
| σ_n ∈ [0.005, 0.1] | 4 livelli di rumore | Copre scenari da rumore basso (CCD di qualità) ad alto (acquisizione rapida) |

## Esempio d'uso

```python
from src.degradation.degradation import degrade
import torch

x = torch.randn(1, 3, 256, 256)  # immagine clean [-1, 1]
x_degraded, kernel = degrade(x, noise_level=0.05)
# x_degraded.shape → (1, 3, 256, 256)
# kernel.shape → (1, 1, 9, 9)
```

## Riproducibilità

La pipeline fissa il seed deterministico **prima** di ogni invocazione:

```python
import numpy as np
import torch, random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

Questo garantisce che l'identica immagine degradata venga fornita ai tre metodi.

## Dataset pre-calcolato

Le immagini degradate sono salvate in `data/degraded/` per evitare di ricalcolare ogni volta:

```
data/degraded/
├── noise_0.005/      # 962 immagini
├── noise_0.01/       # 962 immagini
├── noise_0.05/       # 962 immagini
└── noise_0.1/        # 962 immagini
```

Il caricamento avviene tramite `DegradedDataset` in `src/data/dataset.py`, che legge `_degraded.pt` per ogni immagine originale.
