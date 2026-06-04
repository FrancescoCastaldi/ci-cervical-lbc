# DiffPIR — Metodo Generativo

Implementazione del metodo DiffPIR (Denoising Diffusion Models for Plug-and-Play Image Restoration) per il restauro di immagini degradate.

## Setup

### 1. Clona la repository DiffPIR
```bash
git clone https://github.com/yuanzhi-zhu/DiffPIR.git external/diffpir
```

### 2. Scarica i pesi pretrained
```bash
mkdir -p src/methods/diffpir/weights
curl -L "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt" -o src/methods/diffpir/weights/256x256_diffusion_uncond.pt
```

**Nota:** Il file pesa ~2GB e potrebbe richiedere 10-20 minuti per il download.

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
  num_steps: 20          # Step di sampling (20 per CPU, 100 per GPU)
  noise_level: 0.05      # Livello rumore di riferimento
  max_test_images: 10    # Numero immagini da processare
```

### Parametri DiffPIR
- **num_steps**: Numero di step di sampling diffusion (default: 20)
  - CPU: 20-50 step (5-15 sec/immagine)
  - GPU: 100 step (2-3 sec/immagine)
- **lambda_**: Peso data-fidelity (default: 1.0)
- **zeta**: Parametro stocasticità (default: 0.1)

## Output

### Metriche
- `results/diffpir_results.csv`: PSNR, SSIM, tempo inferenza per ogni noise level

### Immagini qualitative
- `results/diffpir_noise{level}_sample{i}.png`: Confronto degraded vs restored vs GT

## Performance

### Tempo di inferenza (CPU)
- **num_steps=20**: ~10-15 sec/immagine
- **num_steps=50**: ~25-35 sec/immagine
- **num_steps=100**: ~50-70 sec/immagine

### Tempo di inferenza (GPU CUDA)
- **num_steps=100**: ~2-3 sec/immagine

## Confronto con altri metodi

Dopo aver eseguito TV e UNet, carica i risultati nel notebook per il confronto:
```python
df_tv = pd.read_csv('results/tv_results.csv')
df_unet = pd.read_csv('results/unet_results.csv')
df_diffpir = pd.read_csv('results/diffpir_results.csv')
```

## Riferimenti

- Paper: [Denoising Diffusion Models for Plug-and-Play Image Restoration](https://arxiv.org/pdf/2305.08995.pdf)
- Repository: https://github.com/yuanzhi-zhu/DiffPIR
- Dataset: Mendeley LBC Cervical Cancer
