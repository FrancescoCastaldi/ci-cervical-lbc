# configs/ — Configurazione Esperimento

Contiene `experiment.yaml` con tutti i parametri dell'esperimento.

## Utilità per l'esame

- **Riproducibilità**: seed fisso (42), parametri di degradazione e iperparametri di ogni metodo sono centralizzati in un unico file
- **Documentazione delle scelte**: ogni parametro (λ_reg, learning rate, noise levels, ecc.) è chiaramente visibile e giustificato nel report

## File

| File | Contenuto |
|---|---|
| `experiment.yaml` | Parametri globali: dataset, degradazione, TV, UNet, DiffPIR, eval |

## Parametri principali

| Sezione | Parametri chiave |
|---|---|
| `dataset` | `subset_size: 4000`, `image_size: 256`, split ratios |
| `degradation` | `blur_sigma: 2`, `kernel_size: 9`, `noise_levels: [0.005, 0.01, 0.05, 0.1]` |
| `tv` | `lambda_reg: 0.005`, `max_iter: 150` |
| `unet` | `lr: 0.0001`, `epochs: 50`, `features: [16, 32, 64, 128]` |
| `diffpir` | `num_steps: 15`, `lambda: 10.0`, `t_start: 50` |
