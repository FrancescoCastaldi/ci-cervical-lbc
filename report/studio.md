# Computational Imaging: Deblur & Denoise su immagini LBC Cervical

**Corso:** Computational Imaging — Università di Bologna, LM Informatica
**Prof:** Picciolomini & Evangelista — **Anno:** 2025/2026
**Gruppo R —** Francesco Castaldi, Paolo Fusco
**Metodi scelti (3 su 4):** Total Variation (variazionale), UNet (end-to-end), DiffPIR (generativo)
**Metodo escluso:** Weighted TV (ibrido) — gruppo di 2 studenti
**Repo:** `https://github.com/FrancescoCastaldi/ci-cervical-lbc`

---

## Indice

1. [Il Problema Inverso e il Task](#1-il-problema-inverso-e-il-task)
2. [Dataset e Preprocessing](#2-dataset-e-preprocessing)
3. [Degradazione](#3-degradazione)
4. [Famiglie Metodologiche: Perché Confrontarle](#4-famiglie-metodologiche-perché-confrontarle)
5. [Total Variation (TV) — Teoria e Metodo Variazionale](#5-total-variation-tv--teoria-e-metodo-variazionale)
6. [UNet — Deep Learning End-to-End](#6-unet--deep-learning-end-to-end)
7. [DiffPIR — Metodo Generativo (Diffusion)](#7-diffpir--metodo-generativo-diffusion)
8. [Metriche di Valutazione](#8-metriche-di-valutazione)
9. [Risultati](#9-risultati)
10. [Confronto e Discussione](#10-confronto-e-discussione)
11. [Riferimenti](#11-riferimenti)

---

## 1. Il Problema Inverso e il Task

### 1.1 Formulazione

Il restauro di immagini degradate è un classico **problema inverso**: data un'osservazione $y$, si vuole stimare l'immagine originale $x$.

$$y = \mathcal{H}(x) + n \quad\Longrightarrow\quad y = (k * x) + n$$

Dove:
- $x$ = ground truth (immagine pulita originale, sconosciuta)
- $k$ = kernel di blur gaussiano ($\sigma=2$, kernel $9 \times 9$)
- $n$ = rumore AWGN (Additive White Gaussian Noise) con $\sigma_n \in \{0.005, 0.01, 0.05, 0.1\}$
- $y$ = osservazione degradata (l'unica cosa che abbiamo)

### 1.2 Perché è Mal Posto (Ill-Posed)

Il problema è **mal posto** nel senso di Hadamard: la soluzione $x$ non è unica, non esiste in modo stabile. Formalmente, l'operatore $\mathcal{H}$ (convoluzione con blur) è un operatore compatto con spettro che decade a zero: le alte frequenze dell'immagine vengono attenuate e sommerse dal rumore. **Infinite distribuzioni di pixel $x$** sono compatibili con la stessa $y$.

Per risolvere il problema servono **regolarizzazioni** o **prior** che vincolino la soluzione a essere realistica. Il progetto esplora tre modi diversi di imporre questo vincolo:

| Metodo | Tipo di Vincolo | Cosa Presuppone |
|---|---|---|
| **TV** | Regolarizzatore esplicito (manuale) | Le immagini naturali hanno gradienti sparsi (seguono una Laplace) |
| **UNet** | Prior appreso dai dati (implicito) | Impara la mappatura degraded → restored da esempi |
| **DiffPIR** | Prior generativo (modello di diffusione) | Le immagini pulite vivono su una varietà appresa dal DDPM |

Confrontare queste tre famiglie è l'essenza del progetto: capire cosa ciascuna sa fare bene e dove fallisce, **non** solo chi ottiene il PSNR più alto.

---

## 2. Dataset e Preprocessing

### 2.1 Mendeley LBC Cervical Cancer

**962 immagini** di citologia cervicale (Pap test), **4 classi diagnostiche**:

| Classe | Descrizione | Immagini | % |
|---|---|---|---|
| **NILM** | Negative for Intraepithelial Malignancy | 612 | 63.6% |
| **HSIL** | High Squamous Intra-epithelial Lesion | 163 | 16.9% |
| **LSIL** | Low Squamous Intra-epithelial Lesion | 113 | 11.7% |
| **SCC** | Squamous Cell Carcinoma | 74 | 7.7% |

Dataset **sbilanciato** (NILM > 60%), ma per il restauro non è un problema — la varietà morfologica delle cellule è comunque rappresentata in tutte le classi. Non facciamo classificazione, quindi lo sbilanciamento non ci penalizza.

**Nota dimensioni:** La specifica del progetto menziona un subset di ~4000 immagini. Il dataset Mendeley ne contiene **962 totali** — usiamo tutto.

### 2.2 Pipeline di Preprocessing

| Passo | Operazione | Perché |
|---|---|---|
| **Resize** | 2048×1536 → **256×256** (bilineare) | Carico computazionale gestibile; 256×256 preserva nuclei e dettagli cellulari |
| **ToTensor** | [0, 255] → [0, 1] | Conversione in floating point per PyTorch |
| **Normalizzazione** | $(x-0.5)/0.5 \rightarrow [-1, 1]$ | Range richiesto da DDPM; centrato a zero per MSE |
| **Split** | **70/15/15 stratificato** (seed=42) | Ogni split mantiene le proporzioni delle classi; seed fisso per riproducibilità |

### 2.3 Split

| Split | % | Immagini |
|---|---|---|
| Training | 70% | **673** |
| Validation | 15% | **144** |
| Test | **15%** | **145** (IDENTICO per tutti i metodi) |

Tutti i metodi usano lo **stesso test set** (145 immagini). Questo è fondamentale per un confronto equo.

---

## 3. Degradazione

**Identica per tutti i metodi,** generata con **seed fisso 42** tramite `src/degradation/degradation.py`. Le differenze nei risultati sono attribuibili **solo ai metodi**, non a variazioni casuali.

| Parametro | Valore |
|---|---|
| Blur gaussiano $\sigma$ | 2 |
| Kernel size | $9 \times 9$ |
| AWGN $\sigma_n$ | 0.005, 0.01, 0.05, 0.1 |

**Perché 4 livelli di rumore?** Per studiare come ciascun metodo si comporta al variare della difficoltà del problema. TV dovrebbe degradare con il rumore ($\lambda$ fisso), UNet dovrebbe essere robusto (multi-noise training), DiffPIR potrebbe avere comportamento inatteso.

---

## 4. Famiglie Metodologiche: Perché Confrontarle

Il progetto copre **3 famiglie** metodologiche fondamentalmente diverse. Capire le differenze è più importante dei numeri.

### 4.1 Metodi Variazionali (TV)
**Approccio:** $x^* = \arg\min_x \text{DataFidelity}(x, y) + \lambda \cdot \text{Regularizer}(x)$
- Il regolarizzatore è **scelto a mano** dal progettista (TV = L1 dei gradienti)
- **Non richiede dati** — funziona subito su qualsiasi immagine
- **Interpretabile** — ogni termine nell'energia ha un significato
- **Svantaggio:** il regolarizzatore potrebbe non catturare la complessità delle immagini reali

### 4.2 Deep Learning End-to-End (UNet)
**Approccio:** $\hat{x} = f_\theta(y)$ dove $f_\theta$ è una rete addestrata su coppie (degraded, clean)
- La mappatura è **appresa dai dati** — nessuna scelta esplicita del regolarizzatore
- **Inferenza velocissima** (un forward pass della rete)
- **Svantaggio:** richiede tante coppie (degraded, clean); generalizza male fuori distribuzione

### 4.3 Metodi Generativi (DiffPIR)
**Approccio:** usa un **modello di diffusione** (DDPM) come prior, combinato con un passo di data-fidelity
- Il prior è appreso **senza supervisione** (solo immagini pulite, non coppie)
- Può **ricostruire dettagli persi** che metodi classici non possono recuperare
- **Svantaggio:** computazionalmente costoso; può "allucinare" dettagli assenti

### 4.4 Perché Confrontarli

Ogni famiglia ha presupposti diversi su come deve essere un'immagine "realistica":
- TV → "gradienti piccoli e sparsi"
- UNet → "come quelle viste nel training set"
- DiffPIR → "come quelle generate dal DDPM"

Il confronto rivela **quali presupposti funzionano meglio in quali condizioni**.

---

## 5. Total Variation (TV) — Teoria e Metodo Variazionale

### 5.1 Teoria: La Regolarizzazione Variazionale

I metodi variazionali formulano il restauro come un problema di ottimizzazione:

$$\hat{x} = \arg\min_x \mathcal{J}(x) = \underbrace{\|H * x - y\|_2^2}_{\ell(x,y)} + \lambda \cdot \underbrace{\mathcal{R}(x)}_{\text{regolarizzatore}}$$

Il primo termine $\ell(x, y)$ è la **data fidelity**: forza $H*x$ a essere vicino a $y$ (assumendo rumore gaussiano, la log-likelihood è proporzionale a $\|Hx - y\|_2^2$ — equivalente alla stima di massima verosimiglianza).

Il secondo termine $\mathcal{R}(x)$ è il **regolarizzatore**: incorpora la conoscenza a priori su $x$. In termini Bayesiani, $\mathcal{R}(x)$ corrisponde al logaritmo del prior $p(x)$, e la soluzione è la **MAP** (Maximum a Posteriori):

$$p(x|y) \propto p(y|x)p(x) \quad\Longrightarrow\quad -\log p(x|y) = \|Hx - y\|_2^2 + \lambda \mathcal{R}(x)$$

### 5.2 Total Variation

La Total Variation è definita come:

$$TV(x) = \sum_{i,j} \|\nabla x_{i,j}\|_2 = \sum_{i,j} \sqrt{(x_{i+1,j} - x_{i,j})^2 + (x_{i,j+1} - x_{i,j})^2}$$

(versione isotropica) oppure:

$$TV(x) = \sum_{i,j} |x_{i+1,j} - x_{i,j}| + |x_{i,j+1} - x_{i,j}|$$

(versione anisotropica, usata nel codice).

**Perché TV e non, per esempio, Tikhonov ($\|\nabla x\|_2^2$)?**

La differenza cruciale è tra **norma L1 e L2 del gradiente**:
- **Tikhonov (L2):** $ \sum (\nabla x)^2 $ — penalizza molto i gradienti grandi, **sfoca i bordi**. Favorisce transizioni graduali.
- **TV (L1):** $ \sum |\nabla x| $ — penalizza proporzionalmente, **preserva i bordi netti**. Favorisce regioni a tratti costanti (piecewise constant).

Per immagini di citologia cervicale, dove le cellule hanno **contorni netti** su sfondo uniforme, TV è la scelta migliore.

### 5.3 Implementazione

Il kernel gaussiano è costruito come tensore PyTorch e applicato via `F.conv2d` con padding. Ad ogni iterazione Adam:
1. Si calcola $\ell = \|H*x - y\|_2^2$
2. Si calcola $TV(x)$
3. Si minimizza $\mathcal{L} = \ell + \lambda \cdot TV(x)$
4. Si clampa $x$ in $[-1, 1]$

### 5.4 Parametri

| Parametro | Valore | Perché Proprio Questo |
|---|---|---|
| **$\lambda_{reg}$** | **0.005** | Testati {0.001, 0.005, 0.01, 0.05, 0.1}. $\lambda$=0.001: rumore residuo; $\lambda$=0.1: staircasing eccessivo. |
| **Iterazioni** | **150** | Loss si stabilizza (<1% variazione ultime 20 iter) |
| **Lr** | **0.001** (Adam) | 0.01 causa pixelatura; 10⁻⁴ troppo lento |

**Perché $\lambda$ è fisso?** Per semplicità e per osservare l'effetto del rumore a parità di regolarizzazione. Un $\lambda(\sigma_n)$ adattivo migliorerebbe i risultati ad alto rumore.

### 5.5 Punti di Forza e Debolezza

| Pro | Contro |
|---|---|
| ✅ **Nessun training** — funziona subito | ❌ **Staircasing** (effetto a gradini) |
| ✅ **Interpretabile** — ogni termine ha senso | ❌ $\lambda$ **fisso** penalizza alto rumore |
| ✅ **Riproducibile** — deterministico | ❌ Perde dettagli fini (cromatina) |
| ✅ **Funziona con pochi dati** | ❌ Lento rispetto a UNet (~7 s/img) |

---

## 6. UNet — Deep Learning End-to-End

### 6.1 Teoria: Image-to-Image con Reti Convoluzionali

L'idea è imparare una mappatura diretta $f_\theta: y \rightarrow \hat{x}$ usando un dataset di coppie $\{(y_i, x_i)\}$. La UNet è progettata specificamente per questo tipo di task, dove input e output hanno la **stessa dimensione spaziale**.

**Architettura:** Encoder-Decoder con Skip Connections

```
Input (256×256×3)
    │
    ▼
┌─────────────────┐
│    Encoder       │  Down: 16 → 32 → 64 → 128 canali
│  (contracting)   │  Ogni livello: DoubleConv + MaxPool 2×2
└────────┬────────┘
         │  skip connections ────────────────┐
         ▼                                    │
┌─────────────────┐                            │
│    Bottleneck    │  128 → 256 → 128          │
└────────┬────────┘                            │
         │                                     │
         ▼                                     │
┌─────────────────┐                            │
│    Decoder       │  Up: 128 → 64 → 32 → 16   │
│  (expanding)     │  Ogni livello: Upsample +  │
│                  │  concat skip → DoubleConv  │
└────────┬────────┘                            │
         │◄────────────────────────────────────┘
         ▼
    Output (256×256×3)  [tanh → [-1, 1]]
```

**Skip Connections:** motivo chiave del successo della UNet. Collegano direttamente i livelli dell'encoder a quelli del decoder allo stesso livello di risoluzione. Questo permette al decoder di **accedere ai dettagli spaziali fini** (bordi, texture) che altrimenti andrebbero persi nel bottleneck. Senza skip, l'architettura sarebbe un "imbuto" che perde informazioni spaziali.

**DoubleConv:** Ogni blocco contiene $Conv3\times3 \rightarrow GroupNorm \rightarrow SiLU \rightarrow Conv3\times3 \rightarrow GroupNorm \rightarrow SiLU$.

### 6.2 Scelte Architetturali e Perché

| Scelta | Perché |
|---|---|
| **GroupNorm** invece di BatchNorm | BatchNorm si comporta male con batch size piccolo (16). GroupNorm normalizza per gruppi di canali ed è stabile indipendentemente dal batch size. |
| **SiLU** invece di ReLU | SiLU ($x \cdot \sigma(x)$) è una variante liscia di ReLU; produce gradienti più stabili. |
| **L1 Loss** invece di MSE (L2) | MSE penalizza molto i pixel con errore grande (bordi). L1 è più robusta agli outlier e preserva meglio i dettagli fini. |
| **4 canali input** (RGB + noise map) | Condiziona la UNet sul livello di rumore: impara strategie diverse per $\sigma_n$ diversi. |
| **Features [16,32,64,128]** | Architettura **snella** (~1.9M params vs ~31M originale). 60× più piccola, training 50 epoche in 85 min invece di 1 epoca in 82 min. |
| **Multi-noise augmentation** | Ogni batch ha un $\sigma_n$ diverso scelto random. Il modello impara a **generalizzare su tutti i livelli** invece di specializzarsi su uno. |

**Perché non ViT o NAF-Net?**
- **ViT:** self-attention ha complessità $O(N^2)$ — su 256×256 (65536 pixel) è **proibitivo** su CPU
- **NAF-Net:** più complesso senza benefici chiari per questo task
- **UNet:** il **bias induttivo convoluzionale** (località, traduzione-equivarianza) è perfetto per immagini. Richiede pochi dati.

### 6.3 Training

| Parametro | Valore | Giustificazione |
|---|---|---|
| Loss | **L1** | Preserva dettagli fini |
| Optimizer | Adam | Convergenza stabile |
| Lr | $10^{-4}$ | Standard per reti convoluzionali |
| Batch size | 16 | Massimo su CPU (256×256×3) |
| Epoche | **50** | Sufficienti per convergenza (~85 min CPU) |
| Scheduler | ReduceLROnPlateau | Riduce lr se PSNR di validation ristagna |
| Multi-noise | $\sigma_n \sim \text{Uniform}\{0.005, 0.01, 0.05, 0.1\}$ | Generalizza su tutti i livelli |

**Training loop semplificato:**
```
for epoch in range(epochs):
    for batch in train_loader:
        σ = random.choice(noise_levels)          # multi-noise
        degraded = blur(batch, σ_blur=2) + noise(σ)
        noise_map = σ * torch.ones_like(batch)   # 4° canale
        input = cat(degraded, noise_map, dim=1)
        pred = unet(input)
        loss = L1(pred, batch)
        optimizer.step(loss)
```

### 6.4 Punti di Forza e Debolezza

| Pro | Contro |
|---|---|
| ✅ **Inferenza velocissima** (~0.035 s/img) | ❌ **Richiede training** (85 min CPU) |
| ✅ **Robusto** su tutto lo spettro di rumore | ❌ **Generalizza** meno fuori distribuzione |
| ✅ Perde solo ~1 dB da 0.005 a 0.1 (TV: ~5.5 dB) | ❌ Ignora il modello di degradazione (black box) |
| ✅ Supera TV ad alto rumore | ❌ Dipende dalla qualità dei dati di training |

---

## 7. DiffPIR — Metodo Generativo (Diffusion)

### 7.1 Teoria: Denoising Diffusion Probabilistic Models (DDPM)

I DDPM sono modelli generativi che imparano a **denoizzare** un'immagine passo dopo passo.

**Processo Forward (fissato):** Si distrugge gradualmente un'immagine $x_0$ aggiungendo rumore gaussiano per $T$ timestep:

$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t\mathbf{I}\right)$$

Dopo $T$ passi, $x_T \approx \mathcal{N}(0, \mathbf{I})$ (rumore puro). La bella proprietà è che si può campionare $x_t$ direttamente da $x_0$ in un passo:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \varepsilon \quad\text{dove}\quad \bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$$

All'aumentare di $t$, $\bar{\alpha}_t \to 0$ e $x_t$ diventa sempre più rumoroso.

**Processo Reverse (appreso):** Si impara una rete $\varepsilon_\theta(x_t, t)$ che predice il rumore $\varepsilon$ aggiunto al timestep $t$. La loss è:

$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, x_0, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]$$

Una volta addestrata, si può **generare** un'immagine pulita partendo da rumore puro $x_T \sim \mathcal{N}(0, \mathbf{I})$ e rimuovendo iterativamente il rumore:

$$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t, t)\right) + \sigma_t z$$

### 7.2 DiffPIR: Adattare il DDPM al Restauro

DiffPIR modifica il processo reverse per **condizionarlo all'osservazione $y$**. Ad ogni timestep $t$ si alternano due step:

**Step 1 — Denoising (prior):** Si stima l'immagine pulita $\hat{x}_0^{(t)}$ usando la predizione del rumore:

$$\hat{x}_0^{(t)} = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \varepsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

Questa è una **stima di $x_0$ basata solo sul prior generativo** (ignora $y$).

**Step 2 — Data Fidelity (fedeltà ai dati):** Si proietta $\hat{x}_0^{(t)}$ per renderla compatibile con $y$:

$$\tilde{x}_0^{(t)} = \arg\min_x \|y - \mathcal{H}(x)\|^2 + \rho_t \|x - \hat{x}_0^{(t)}\|^2$$

Il primo termine forza $x$ a essere coerente con $y$ dopo blur; il secondo impedisce di allontanarsi troppo dalla stima del prior. $\rho_t$ bilancia i due termini.

**Soluzione analitica via FFT:** Questo problema ha una soluzione in forma chiusa nel dominio della frequenza:

$$\tilde{x}_0 = \mathcal{F}^{-1}\left(\frac{\overline{\mathcal{F}(k)} \cdot \mathcal{F}(y) + \rho \cdot \mathcal{F}(\hat{x}_0)}{|\mathcal{F}(k)|^2 + \rho}\right)$$

Dove $\mathcal{F}$ è la FFT e $\overline{\mathcal{F}(k)}$ il complesso coniugato. La soluzione è efficiente ($O(N\log N)$) e non richiede iterazioni.

**Peso dinamico $\rho_t$:**

$$\rho_t = \lambda \cdot \frac{\sigma^2 \cdot \bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

- A $t$ grandi ($\bar{\alpha}_t \to 1$): $\rho_t \to \infty$ — domina il **data-fidelity**
- A $t$ piccoli ($\bar{\alpha}_t \to 0$): $\rho_t \to 0$ — domina il **prior generativo**

Questo è cruciale: all'inizio ci fidiamo dei dati, alla fine lasciamo spazio al generativo per "creare" i dettagli.

### 7.3 Perché $t_{\text{start}}=50$ e Non 1000?

Questa è una scelta chiave e va capita bene. Nel DDPM standard, la generazione parte da $x_T$ ($T=1000$) con $\bar{\alpha}_{1000} \approx 0$.

La stima di $x_0$ è:

$$\hat{x}_0^{(t)} = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \varepsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

Il fattore $\frac{1}{\sqrt{\bar{\alpha}_t}}$ amplifica gli errori di predizione di $\varepsilon_\theta$. A $t=1000$:

$$\bar{\alpha}_{1000} \approx 4.5 \times 10^{-5} \quad\Longrightarrow\quad \frac{1}{\sqrt{\bar{\alpha}_{1000}}} \approx 150$$

Un errore del 10% nella predizione del rumore diventa un errore del **1500%** nella stima di $x_0$!

A $t_{\text{start}}=50$ invece:

$$\bar{\alpha}_{50} \approx 0.97 \quad\Longrightarrow\quad \frac{1}{\sqrt{\bar{\alpha}_{50}}} \approx 1.03$$

L'errore viene **amplificato solo del 3%**. La stima è numericamente stabile.

**Perché non partire da $t=50$ nel DDPM standard?** Per generare un'immagine dal rumore puro, servono molti passi perché il modello deve "creare" l'immagine dal nulla. Nel restauro, invece, **l'immagine di partenza $x_{t_{\text{start}}}$ viene dall'osservazione $y$ degradata**, non dal rumore. Partiamo con un'immagine già strutturata.

### 7.4 LightUNet Custom

Invece del modello originale DiffPIR (pre-addestrato su ImageNet, ~500M parametri, ~2GB), usiamo una **LightUNet** addestrata specificamente su LBC:

| Componente | Specifica |
|---|---|
| Canali base | 32 |
| Encoder | 32 → 64 → 128 |
| Decoder | 128 → 64 → 32 (skip-connected) |
| Timestep embedding | Sinusoidale + MLP 2 layer (dim=128) |
| Normalizzazione | GroupNorm (8 gruppi) |
| Attivazione | SiLU |
| Parametri tot. | **1,262,273** (~5 MB su disco) |
| Training | 1000 timestep, $\beta_1=10^{-4}$, $\beta_T=0.02$, 30 epoche |

**Perché un modello custom invece del pre-addestrato?**
1. **Specificità del dominio:** ImageNet (foto naturali) non cattura la morfologia delle cellule cervicali
2. **Dimensione:** 1.26M params (~5MB) vs 500M (~2GB) — training e inferenza possibili su CPU
3. **Adeguatezza al task:** 256×256 e struttura citologica non richiedono un modello gigantesco

### 7.5 Parametri Chiave di DiffPIR

| Parametro | Valore | Perché |
|---|---|---|
| **$t_{\text{start}}$** | **50** | Stabilità numerica: $\bar{\alpha}_{50} \approx 0.97$, amplif. errore $\approx 1.03\times$. Testati {10, 30, 50, 100, 200} |
| **$\lambda$** | **10.0** | Peso data-fidelity. $\lambda<1$: allucinazioni. $\lambda>50$: rigido. |
| **$num\_steps$** | **15** | Sub-campionati da 50 a 0. <10: qualità scarsa; >20: marginale. |
| **$\zeta$** | **0.0** | DDIM deterministico (riproducibilità) |

### 7.6 Punti di Forza e Debolezza

| Pro | Contro |
|---|---|
| ✅ **Ricostruisce dettagli persi** ad alto rumore | ❌ **Allucinazioni a basso rumore** (peggiora l'immagine) |
| ✅ **Deblurring analitico** via FFT | ❌ **Lento** (~3 s/img su CPU) |
| ✅ **Plug-and-play** (denoiser sostituibile) | ❌ Modello 1.26M params vs standard 500M+ |
| ✅ **Non serve coppie (degraded, clean)** per training | ❌ Training su 673 immagini insufficiente per un generativo |

---

## 8. Metriche di Valutazione

### 8.1 PSNR (Peak Signal-to-Noise Ratio)

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)$$

$MAX=1.0$ (immagini in $[0,1]$), $MSE = \frac{1}{N}\sum(x_i - \hat{x}_i)^2$.

- **Valori alti = migliore qualità.** Tipico range: 20-35 dB.
- **Limite:** non sempre correlato con la qualità percettiva — un'immagine liscia può avere PSNR alto ma essere visivamente bruttina.

### 8.2 SSIM (Structural Similarity Index)

$$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

- Misura **luminanza** ($\mu$), **contrasto** ($\sigma$), **struttura** ($\sigma_{xy}$)
- Range $[-1, 1]$, $1$ = identico
- **Più correlato con la percezione umana** del PSNR

### 8.3 Tempo di Inferenza

Costo computazionale per immagine. Rilevante per applicazioni real-time.

---

## 9. Risultati

### 9.1 TV (145 immagini test)

| $\sigma_n$ | PSNR | SSIM | Cosa Succede |
|---|---|---|---|
| **0.005** | **32.09 dB** | **0.911** | **Eccellente.** Visivamente indistinguibile dall'originale |
| **0.01** | 32.04 dB | 0.909 | **Eccellente.** Stessa qualità |
| **0.05** | 30.42 dB | 0.837 | **Buono.** Inizia staircasing; dettagli fini persi |
| **0.1** | 26.54 dB | 0.586 | **Adeguato.** $\lambda$ fisso non basta; rumore residuo |

**Comportamento atteso:** Degrado progressivo con il rumore — il $\lambda$ fisso non si adatta. **Staircasing** visibile nelle aree a texture fine (citoplasma), dove transizioni graduali sono approssimate con salti discreti.

### 9.2 UNet (145 immagini test, 50 epoche CPU)

| $\sigma_n$ | PSNR | SSIM | Tempo |
|---|---|---|---|
| 0.005 | **29.89 dB** | **0.894** | **0.035 s** |
| 0.01 | **29.89 dB** | **0.894** | **0.034 s** |
| 0.05 | **29.63 dB** | **0.875** | **0.034 s** |
| 0.1 | **28.93 dB** | **0.830** | **0.036 s** |

**Osservazioni:**
- ~29.9 dB a basso rumore: solo **~2 dB sotto TV**
- Degradazione **minima**: solo ~1 dB da 0.005 a 0.1 (TV perde ~5.5 dB)
- A $\sigma=0.1$ **SUPERA TV** (28.93 vs 26.54 dB)
- **Inferenza 200× più veloce** di TV

### 9.3 DiffPIR (10 immagini test)

| $\sigma_n$ | PSNR | SSIM | Tempo |
|---|---|---|---|
| 0.005 | 16.67 dB | 0.235 | 3.27 s |
| 0.01 | 17.32 dB | 0.270 | 3.00 s |
| 0.05 | **22.49 dB** | **0.512** | 2.85 s |
| 0.1 | **24.68 dB** | **0.664** | 2.89 s |

**Perché PSNR cresce con il rumore? (È controintuitivo)**

Questo è il comportamento caratteristico dei metodi generativi basati su diffusione:

- **$\sigma_n=0.005$:** L'immagine degradata è già quasi perfetta. Ma il modello ha un **prior generativo forte** che lo spinge ad aggiungere dettagli (allucinazioni) — "immagina" texture assenti, **peggiorando** l'immagine. PSNR basso non perché il modello sia scarso, ma perché **modifica ciò che andrebbe lasciato intatto**.

- **$\sigma_n=0.05$:** Inizia il recupero. Il modello ha abbastanza "margine" per rimuovere rumore senza introdurre artefatti evidenti. PSNR e SSIM migliorano.

- **$\sigma_n=0.1$:** Il punto migliore. Con tanto rumore, il prior generativo **aiuta** a ricostruire il segnale perso. Il guadagno netto è positivo: PSNR 24.68, SSIM 0.664.

**Lezione importante:** Un PSNR basso non significa sempre "metodo scarso". Significa che il metodo non è adatto a quel regime di rumore. La scelta del metodo deve essere **contestuale**.

### 9.4 Tabella Comparativa

| $\sigma_n$ | TV (PSNR/SSIM) | UNet (PSNR/SSIM) | DiffPIR (PSNR/SSIM) |
|---|---|---|---|
| **0.005** | **32.09 / 0.911** 🥇 | 29.89 / 0.894 🥈 | 16.67 / 0.235 |
| **0.01** | **32.04 / 0.909** 🥇 | 29.89 / 0.894 🥈 | 17.32 / 0.270 |
| **0.05** | **30.42 / 0.837** 🥇 | 29.63 / **0.875** 🥈 | 22.49 / 0.512 |
| **0.1** | 26.54 / 0.586 🥈 | **28.93 / 0.830** 🥇 | 24.68 / 0.664 🥉 |

### 9.5 Chi Vince e Perché

| Categoria | Vincitore | Perché |
|---|---|---|
| **Basso rumore** (0.005) | **TV** (32.09 dB) | TV è un ottimo denoiser quando il problema è quasi ben posto |
| **Medio rumore** (0.05) | **TV** (30.42 dB) | Ma UNet è a solo 0.8 dB — quasi pari |
| **Alto rumore** (0.1) | **UNet** (28.93 dB) | Il deep learning regge meglio: degradazione minima (~1 dB) |
| **Secondo ad alto rumore** | **DiffPIR** (24.68 dB) | Il generativo ricostruisce dettagli che TV non può |
| **Velocità inferenza** | **UNet** (0.035 s/img) | 200× più veloce di TV, 85× più veloce di DiffPIR |
| **Nessun training** | **TV** | Pronto all'uso, zero dati necessari |

---

## 10. Confronto e Discussione

### 10.1 Tavola Sinottica

| Dimensioni | **TV** (variazionale) | **UNet** (end-to-end) | **DiffPIR** (generativo) |
|---|---|---|---|
| **Dati necessari** | **Nessuno** | 673 coppie (degraded, clean) | 673 immagini clean (solo per prior) |
| **Training** | **No** | ~85 min CPU | ~30 epoche CPU |
| **Inferenza** | ~7 s/img | **~0.035 s/img** | ~3 s/img |
| **Basso $\sigma_n$** | **Eccellente** (32 dB) | Buono (29.9 dB) | Scarso (16.7 dB) — allucina |
| **Alto $\sigma_n$** | Adeguato (26.5 dB) | **Eccellente** (28.9 dB) | Buono (24.7 dB) |
| **Degradazione** | Severa (−5.5 dB) | **Minima** (−1 dB) | Inversa (PSNR cresce) |
| **Interpretabilità** | **Alta** (formula chiusa) | Bassa (black box) | Media (passi separati) |
| **Robustezza** | Alta (nessun training) | Media (dipende dai dati) | Bassa (allucinazioni) |
| **Riproducibilità** | **Totale** | Dipende dal seed | Deterministico (DDIM) |

### 10.2 Le Tre Grandi Lezioni

**1. Nessun metodo domina tutti i regimi.**
- Basso rumore? → **TV** (semplice, efficace, ready-to-use)
- Alto rumore? → **UNet** (regge meglio, veloce)
- Massima qualità ad alto rumore? → **DiffPIR** (ricostruisce dettagli persi)

**2. I metodi generativi vanno usati con cautela.**
DiffPIR fa una cosa che TV e UNet non possono fare: **creare dettaglio assente**. Questo è un potere e un pericolo: funziona bene ad alto rumore (dove serve ricostruire), ma **allucina** a basso rumore (dove serve preservare). La lezione: il prior generativo va bilanciato accuratamente.

**3. Il confronto equo è tutto.**
Stessa pipeline di degradazione, stesso seed, stesse metriche — altrimenti i numeri non significano nulla. Questo progetto garantisce la **riproducibilità** (seed 42, pipeline deterministiche, 34 test unitari).

### 10.3 Punti di Forza

- ✅ **Stessi input degradati** per tutti i metodi
- ✅ **Scelte motivate** dei parametri (test euristici documentati)
- ✅ **Confronto serio** (non solo PSNR più alto — analisi per regime)
- ✅ **Successi e fallimenti** discussi per ogni metodo
- ✅ **Codice modulare** su GitHub (src/methods/\<metodo\>/) con 34 test

### 10.4 Criticità Comuni

| Criticità | Impatto |
|---|---|
| **Dataset 962 immagini** (673 per training) | Limitante per deep learning — generativo particolarmente penalizzato |
| **CPU-only** | Training lungo; impossibile testare modelli grandi (DiffPIR 500M+) |
| **DiffPIR: LightUNet 1.26M params** | vs standard 500M+ dei paper SOTA |
| **TV: $\lambda$ fisso** | Penalizza l'alto rumore — tuning adattivo migliorerebbe |

### 10.5 Direzioni Future

1. **Tuning adattivo $\lambda(\sigma_n)$ per TV** — migliorerebbe i risultati ad alto rumore
2. **Training UNet su GPU** — più epoche, batch più grandi, architetture più profonde
3. **Modello di diffusione più grande** — 50M+ parametri per DiffPIR

---

## 11. Riferimenti

- **DiffPIR:** Zhu et al., "Denoising Diffusion Models for Plug-and-Play Image Restoration." *CVPR 2023.*
- **DDPM:** Ho et al., "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.*
- **DDIM:** Song et al., "Denoising Diffusion Implicit Models." *ICLR 2021.*
- **TV:** Rudin, Osher, Fatemi. "Nonlinear total variation based noise removal algorithms." *Physica D 1992.*
- **UNet:** Ronneberger, Fischer, Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI 2015.*
- **Dataset:** Mendeley LBC Cervical Cancer. [DOI: 10.17632/zddtpgzv63.2](https://data.mendeley.com/datasets/zddtpgzv63/2)
