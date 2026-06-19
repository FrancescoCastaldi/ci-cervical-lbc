# src/ — Codice Sorgente

Moduli Python per l'intera pipeline: caricamento dati, degradazione, metodi di restauro, metriche, visualizzazione.

## Utilità per l'esame

- **Modularità**: ogni metodo è indipendente in methods/<metodo>/ — codice pulito e manutenibile
- **Pipeline condivisa**: degradazione, metriche e plot sono comuni — confronto equo garantito
- **Parametri euristici**: ogni metodo ha i propri iperparametri, documentati nel report

## Sottomoduli

| Modulo | Contenuto | Ruolo |
|---|---|---|
| data/ | LBCDataset, uild_splits, DegradedDataset | Caricamento e preprocessing dataset |
| degradation/ | degrade(), gaussian_kernel() | Pipeline blur + AWGN (identica per tutti) |
| methods/tv/ | 	v_restore(), 	v_loss() | Total Variation (variazionale) |
| methods/unet/ | UNet, DoubleConv | UNet end-to-end (deep learning) |
| methods/diffpir/ | DiffPIR, LightUNet, training loop | DiffPIR generativo (diffusion) |
| eval/ | compute_psnr(), compute_ssim(), evaluate() | Metriche di valutazione |
| plots/ | show_comparison(), plot_metrics() | Visualizzazione risultati |

## Pipeline di esecuzione

`
data/ → degradation/ → methods/<metodo>/ → eval/ → plots/
`

Tre metodi indipendenti, un'unica pipeline di valutazione.

## Esempi di import

`python
from src.data.dataset import LBCDataset, DegradedDataset, build_splits, load_config
from src.degradation.degradation import degrade, gaussian_kernel
from src.methods.tv.tv import tv_restore
from src.methods.unet.unet import UNet
from src.methods.diffpir.diffpir import DiffPIR
from src.eval.metrics import compute_psnr, compute_ssim, evaluate
from src.plots.visualize import show_comparison, plot_metrics
`

## Esecuzione completa

`python
from src.data.dataset import LBCDataset, DegradedDataset
from src.degradation.degradation import degrade
from src.methods.tv.tv import tv_restore
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison

ds = DegradedDataset("data/degraded/test/noise_0.05", "data/degraded/test/gt")
deg, gt = ds[0]
restored = tv_restore(deg, lambda_reg=0.005, num_iters=150)
metrics = evaluate(restored, gt)
show_comparison({"GT": gt, "Degraded": deg, "Restored": restored},
                save_path="comparison.png")
`

## Testing

`ash
pytest tests/ -v
`

I test sono organizzati per modulo: 	ests/test_degradation.py, 	ests/test_metrics.py,
	ests/test_tv.py, 	ests/test_unet.py, 	ests/test_diffpir.py (34 test totali).

## Come aggiungere un metodo

1. Creare src/methods/<nuovo_metodo>/ con il modulo principale
2. Implementare la funzione di restauro (es. my_restore(img, **params))
3. Creare scripts/run_<metodo>.py seguendo il pattern esistente
4. I risultati confluiscono automaticamente in esults/<metodo>/metrics.csv
5. python scripts/plot_results.py include il nuovo metodo nel grafico comparativo
