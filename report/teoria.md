# Fondamenti Teorici — Computational Imaging: Deblur & Denoise

## 1. Il Problema Inverso

Il restauro di immagini degradate è un **problema inverso**: data un'osservazione degradata $y$, si vuole stimare l'immagine originale $x$ (ground truth).

Il modello di degradazione è:

$$y = \mathcal{H}(x) + n$$

dove:
- $\mathcal{H}$ è l'operatore di degradazione (blur)
- $n$ è il rumore additivo (AWGN)

Nel nostro caso:
- **Blur gaussiano**: $\sigma = 2$, kernel size $= 9$
- **Rumore AWGN**: $\sigma_n \in \{0.005, 0.01, 0.05, 0.1\}$

Il problema è **mal posto** (ill-posed): esistono infinite soluzioni $x$ compatibili con $y$. Per questo servono regolarizzazioni o prior che vincolino la soluzione.

## 2. Dataset

**Mendeley LBC Cervical Cancer** — immagini di citologia cervicale (Pap test) classificate in 4 categorie:

| Classe | Descrizione | Immagini |
|---|---|---|
| NILM | Negative for Intraepithelial Malignancy | 612 (63.6%) |
| HSIL | High Squamous Intra-epithelial Lesion | 163 (16.9%) |
| LSIL | Low Squamous Intra-epithelial Lesion | 113 (11.7%) |
| SCC | Squamous Cell Carcinoma | 74 (7.7%) |

**Totale**: 962 immagini, dimensioni originali 2048×1536, resize a 256×256.

Il dataset è eterogeneo per contenuto ma uniforme per dimensioni. La distribuzione sbilanciata non influisce sul task di restauro (non facciamo classificazione), ma è rilevante per la varietà dei contenuti.

## 3. Metodi

### 3.1 Total Variation (TV) — Metodo Variazionale

La Total Variation è un regolarizzatore classico che penalizza le variazioni locali dell'immagine, preservando i bordi.

**Funzione obiettivo:**

$$\hat{x} = \arg\min_x \frac{1}{2}\|y - x\|_2^2 + \lambda \cdot TV(x)$$

dove:

$$TV(x) = \sum_{i,j} |x_{i+1,j} - x_{i,j}| + |x_{i,j+1} - x_{i,j}|$$

**Parametri:**
- $\lambda_{reg} = 0.1$ (peso regolarizzazione)
- 300 iterazioni con ottimizzatore Adam ($lr = 0.01$)

**Pro:** Interpretabile, robusto, non richiede training.
**Contro:** Tende a produrre immagini "a blocchi" (staircasing), dettaglio limitato.

### 3.2 UNet — Metodo Deep Learning End-to-End

La UNet è un'architettura encoder-decoder con skip connections, progettata per tasks di image-to-image.

**Architettura:**
- Encoder: 4 livelli (64, 128, 256, 512 canali) + bottleneck (1024)
- Decoder: simmetrico con skip connections
- Output: convoluzione 1×1 + sigmoid

**Training:**
- Loss: MSE (Mean Squared Error)
- Ottimizzatore: Adam ($lr = 0.0001$)
- Batch size: 16, 50 epoche
- Training su coppie (degraded, ground truth)

**Pro:** Inferenza veloce dopo il training, buona qualità.
**Contro:** Richiede training, generalizzazione limitata ai dati visti.

### 3.3 DiffPIR — Metodo Generativo (Diffusion)

DiffPIR integra un modello di diffusione pretrained in un framework plug-and-play per il restauro di immagini.

**Idea chiave:** Alternare due step ad ogni timestep $t$:

1. **Denoising step** (prior): il modello di diffusione stima $x_0^{(t)}$ da $x_t$
2. **Data-fidelity step**: proiezione vincolata sull'osservazione $y$ via FFT

$$x_0^{(t)} = \text{Denoiser}(x_t)$$
$$\hat{x}_0^{(t)} = \arg\min_x \|y - \mathcal{H}(x)\|^2 + \rho_t \|x - x_0^{(t)}\|^2$$

Il data-fidelity step si risolve analiticamente nel dominio della frequenza:

$$\hat{x}_0 = \mathcal{F}^{-1}\left(\frac{\mathcal{F}(y) \cdot \overline{\mathcal{F}(k)} + \rho \cdot \mathcal{F}(x_0)}{|\mathcal{F}(k)|^2 + \rho}\right)$$

**Modello diffusion:** 256×256 unconditional (ImageNet), 1000 timestep di training, 20-100 step di sampling.

**Parametri chiave:**
- $\lambda$ (peso data-fidelity): 1.0
- $\zeta$ (stocasticità): 0.1
- `num_steps`: 20 (CPU) / 100 (GPU)

**Pro:** Qualità generativa superiore, preserva dettagli fini.
**Contro:** Molto lento in inferenza, complesso da configurare.

## 4. Metriche di Valutazione

### PSNR (Peak Signal-to-Noise Ratio)

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)$$

dove $MAX = 1.0$ (immagini normalizzate in [0,1]) e $MSE = \frac{1}{N}\sum(x_i - \hat{x}_i)^2$.

Valori più alti = migliore qualità. Tipicamente 20-35 dB per restauro.

### SSIM (Structural Similarity Index)

$$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Misura la similarità strutturale (luminanza, contrasto, struttura). Range: [-1, 1], dove 1 = identico.

### Tempo di Inferenza

Misura il costo computazionale per immagine. Rilevante per applicazioni real-time.

## 5. Protocollo di Confronto

Il punto fondamentale del progetto è la **correttezza del confronto**:

1. **Stessi dati**: tutte le immagini del test set (145 immagini, split 15%)
2. **Stessa degradazione**: pipeline unica (`src/degradation/degradation.py`) applicata a tutti i metodi
3. **Stesse metriche**: PSNR e SSIM calcolati con `skimage.metrics`
4. **Stesso seed**: 42, per riproducibilità

Ogni metodo riceve in input la stessa immagine degradata e produce un'immagine restaurata della stessa dimensione. Le metriche sono calcolate rispetto alla stessa ground truth.

## 6. Preprocessing

1. **Resize**: 2048×1536 → 256×256 (bilineare)
2. **ToTensor**: [0, 255] → [0, 1]
3. **Normalizzazione**: $(x - 0.5) / 0.5$ → [-1, 1]
4. **Split**: 70% train / 15% val / 15% test (seed=42)

La normalizzazione in [-1, 1] è coerente con il range atteso dai modelli diffusion e dalla loss MSE.
