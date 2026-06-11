# src/plots/ — Visualizzazione e Grafici

Funzioni per generare i confronti visivi e i grafici comparativi.

## Utilità per l'esame

- **Confronto visivo**: immagini degraded vs restored vs GT affiancate — valutazione qualitativa
- **Grafico comparativo**: plot PSNR/SSIM per tutti i metodi e noise level — deliverable obbligatorio

## File

| File | Contenuto |
|---|---|
| `visualize.py` | `show_comparison()`, `plot_metrics()` |

## Funzioni

| Funzione | Ruolo |
|---|---|
| `show_comparison(images, save_path)` | Griglia con GT, degraded, restored per confronto visivo |
| `plot_metrics(results_dir, save_path)` | Grafico PSNR/SSIM per metodo e noise level |

## Output

- `show_comparison()` → immagini in `results/<metodo>/qualitative/`
- `plot_metrics()` → `results/comparison.png`
