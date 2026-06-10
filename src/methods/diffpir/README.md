# DiffPIR — Metodo Generativo

Implementazione del metodo DiffPIR (Denoising Diffusion Models for Plug-and-Play Image Restoration) per il restauro di immagini degradate.

## Modello

Usa una **LightUNet** (custom DDPM) addestrata su immagini LBC cervicali:
- Architettura: UNet leggero con embedding temporale sinusoidale
- Timesteps: 1000 (training), 15 (sampling)
- Pesi: `weights/ddpm_lbc.pt` (~5MB)

### Addestramento

```bash
python -m src.methods.diffpir.train
```

Training su 100 immagini del training set, 30 epoche, MSE loss.

## Utilizzo

### Script completo
```bash
python scripts/run_diffpir.py
```

### Notebook interattivo
```bash
jupyter notebook notebooks/04_diffpir.ipynb
```

## Configurazione

Parametri in `configs/experiment.yaml`:
```yaml
diffpir:
  num_steps: 15           # Step di sampling
  noise_level: 0.05       # Livello rumore di riferimento
  max_test_images: 10     # Numero immagini da processare
  weights: src/methods/diffpir/weights/ddpm_lbc.pt
  lambda: 10.0            # Peso data-fidelity
  zeta: 0.0               # Stocasticità (0 = deterministico)
  t_start: 50             # Timestep di partenza
```

### Parametri DiffPIR
| Parametro | Default | Ruolo |
|---|---|---|
| `num_steps` | 15 | Step di sampling (sub-campionati da t_start a 0) |
| `lambda_` | 10.0 | Peso data-fidelity |
| `zeta` | 0.0 | Stocasticità (0=deterministico, 1=fully stochastic) |
| `t_start` | 50 | Timestep di partenza (50 per stabilità numerica) |

## Risultati

Metriche su 10 immagini di test:

| σ_n | PSNR | SSIM | Tempo |
|---|---|---|---|
| 0.005 | 16.67 dB | 0.235 | 2.0 s |
| 0.01 | 17.32 dB | 0.270 | 2.0 s |
| 0.05 | 22.49 dB | 0.512 | 2.0 s |
| 0.1 | 24.68 dB | 0.664 | 2.0 s |

## Output

### Metriche
- `results/diffpir/metrics.csv`: PSNR, SSIM, tempo inferenza per ogni noise level

### Immagini qualitative
- `results/diffpir/qualitative/noise_{level}_sample{i}.png`: Confronto degraded vs restored vs GT

## Confronto con altri metodi

Dopo aver eseguito TV e UNet, carica i risultati nel notebook per il confronto:
```python
from src.plots.visualize import plot_metrics
plot_metrics("results", save_path="results/comparison.png")
```

## Riferimenti

- Paper: [Denoising Diffusion Models for Plug-and-Play Image Restoration](https://arxiv.org/pdf/2305.08995.pdf)
- Repository: https://github.com/yuanzhi-zhu/DiffPIR
- Dataset: Mendeley LBC Cervical Cancer
