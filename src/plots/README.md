# src/plots/ - Visualizzazione e Grafici

Funzioni per generare confronti visivi e grafici comparativi dei metodi di restauro.

## Utilità per l'esame

- **Confronto visivo**: immagini degraded vs restored vs GT affiancate - valutazione qualitativa
- **Grafico comparativo**: plot PSNR/SSIM per tutti i metodi e noise level - deliverable obbligatorio

## File

| File | Contenuto |
|---|---|
| isualize.py | show_comparison(), plot_metrics() |

## API completa

### show_comparison(images_dict, save_path=None)

Genera una griglia orizzontale (1xN) con le immagini in images_dict.

`python
images = {"Ground Truth": gt, "Degradata": deg, "Restored TV": restored}
show_comparison(images, save_path="risultati/confronto.png")
`

Parametri:
- images_dict: dict[str, torch.Tensor | np.ndarray] - etichetta verso immagine
- save_path: str | Path | None - percorso salvataggio (None = mostra a schermo)

Dettagli implementativi:
- Layout: 1 riga, N colonne, figsize=(4*N, 4)
- Tensori PyTorch convertiti: detach().cpu().permute(1,2,0).clamp(0,1).numpy()
- Salvataggio a 150 DPI, tight_layout
- Output: cartella esults/<metodo>/qualitative/

### plot_metrics(results_dir, save_path=None)

Legge tutti i metrics.csv da esults/<metodo>/ e genera un grafico 1x2.

`python
plot_metrics("results", save_path="results/comparison.png")
`

Parametri:
- esults_dir: str | Path - directory con sottocartelle per metodo
- save_path: str | Path | None - percorso salvataggio

Dettagli implementativi:
- Legge CSV con pandas (colonne attese: method, 
oise_level, psnr, ssim)
- Traccia linea con marker o per ogni metodo
- Subplots 1x2, figsize=(12, 5), salvataggio 150 DPI

## Code snippet: generazione confronto per TV

`python
from src.data.dataset import DegradedDataset
from src.methods.tv.tv import tv_restore
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison, plot_metrics

ds = DegradedDataset("data/degraded/test/noise_0.05", "data/degraded/test/gt")
for idx in range(5):
    deg, gt = ds[idx]
    restored = tv_restore(deg, lambda_reg=0.005, num_iters=150)
    metrics = evaluate(restored, gt)
    show_comparison({"GT": gt, "Degraded": deg, "TV": restored},
                    save_path=f"results/tv/qualitative/sample_{idx:02d}.png")
`

## Code snippet: generazione grafico comparativo

Esecuzione:
`ash
python scripts/plot_results.py
`

Internamente chiama plot_metrics("results", "results/comparison.png") e produce
un grafico che confronta TV, UNet e DiffPIR sui 4 noise level.

## Output organizzati

`
results/
├── comparison.png              # Grafico comparativo (tutti i metodi)
├── tv/qualitative/             # 24 immagini di confronto TV
├── unet/qualitative/           # 24 immagini di confronto UNet
└── diffpir/qualitative/        # 24 immagini di confronto DiffPIR
`

Ogni metodo produce 6 immagini per ognuno dei 4 noise level (24 totali).