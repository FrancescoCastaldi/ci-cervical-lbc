# configs/ — Configurazione Esperimento

Contiene experiment.yaml con tutti i parametri dell'esperimento centralizzati per riproducibilità.

## Caricamento in Python

La configurazione viene caricata da src/data/dataset.py e scripts/run_*.py tramite:

`python
import yaml
with open("configs/experiment.yaml") as f:
    cfg = yaml.safe_load(f)
`

Ogni script importa cfg e accede alle sezioni: cfg["dataset"], cfg["degradation"], ecc.

## Struttura completa del YAML

`yaml
seed: 42                        # Seed globale per numpy, torch, random

dataset:
  root: data/raw/               # Dataset originale (non processato)
  processed: data/processed/    # Immagini resize + normalizzate
  splits: data/splits/          # Split train/val/test .txt
  degraded: data/degraded/      # Degradate pre-calcolate
  subset_size: 4000             # Numero massimo immagini da usare
  image_size: 256               # Resize a 256×256 px
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

degradation:
  blur_sigma: 2                 # σ del blur gaussiano
  kernel_size: 9                # Dimensione kernel blur (dispari)
  noise_levels: [0.005, 0.01, 0.05, 0.1]   # 4 livelli AWGN

tv:
  lambda_reg: 0.005             # Peso regolarizzazione (λ)
  max_iter: 150                 # Iterazioni Adam

unet:
  lr: 0.0001                    # Learning rate Adam
  batch_size: 16                # Batch size training
  epochs: 50                    # Epoche massime
  in_channels: 4                # RGB + canale noise level
  out_channels: 3               # RGB ricostruito
  features: [16, 32, 64, 128]  # Canali per ogni livello

diffpir:
  num_steps: 15                 # Step DDIM sub-campionati
  noise_level: 0.05
  max_test_images: 10           # Limite test per velocità
  weights: src/methods/diffpir/weights/ddpm_lbc.pt
  lambda: 10.0                  # Peso data-fidelity (FFT)
  zeta: 0.0                     # zeta>0 → rumore extra su step
  t_start: 50                   # Timestep iniziale DDIM

eval:
  results_dir: results/
  save_qualitative: 6           # 6 immagini per noise level
`

## Spiegazione parametri critici

| Parametro | Effetto |
|---|---|
| lur_sigma: 2 | Equalizza per tutti i metodi; σ alto → blur forte |
| 
oise_levels | 4 valori testati; σₙ=0.1 è il caso più difficile |
| lambda_reg: 0.005 | Bilanciamento fedeltà/regolarizzazione TV |
| eatures: [16,32,64,128] | Architettura snella (~500K params, 1.9M con GroupNorm) |
| 	_start: 50 | Scelto sperimentalmente per i nostri noise level |
| lambda: 10.0 | Adattato internamente per-σₙ per il data-fidelity FFT |

## Modificare parametri per esperimenti custom

1. Editare configs/experiment.yaml
2. Creare una copia: configs/experiment_noisy.yaml e lanciare con:
   `python
   with open("configs/experiment_noisy.yaml") as f: cfg = yaml.safe_load(f)
   `
3. Per sweep parametrico, modificare il parametro e rieseguire lo script

## Seed management

Seed fisso 42 garantisce riproducibilità tra esecuzioni. Se si cambia seed, split e inizializzazioni cambiano. Per confronti equi, mantenere lo stesso seed per tutti i metodi.

## Validazione

La configurazione non ha validazione automatica — verificare:
- kernel_size dispari
- 	rain_ratio + val_ratio + test_ratio == 1.0
- Path nei weights devono esistere
- image_size multiplo di 16 (compatibilità UNet/DiffPIR)

