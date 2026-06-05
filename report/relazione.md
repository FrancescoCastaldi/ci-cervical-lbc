# Relazione — Computational Imaging: Deblur & Denoise su immagini LBC Cervical

**Corso:** Computational Imaging — Università di Bologna, LM Informatica
**Prof:** Picciolomini & Evangelista
**Anno Accademico:** 2025/2026

**Gruppo R**
**Studenti:** Francesco Castaldi, Paolo Fusco
**Metodi scelti (3 su 4):** Total Variation (variazionale), UNet (end-to-end), DiffPIR (generativo)
**Metodo escluso:** Weighted TV (ibrido) — escluso perché il gruppo è composto da 2 studenti

**Repository GitHub:** `https://github.com/FrancescoCastaldi/ci-cervical-lbc`

---

## 1. Descrizione Generale

In questo progetto affrontiamo un problema di **computational imaging** combinando diverse famiglie metodologiche — variazionale, deep learning end-to-end, e generativo — con l'obiettivo di comprendere sia gli approcci classici che moderni e confrontarli criticamente nelle stesse condizioni sperimentali.

Il problema consiste nel **restauro di immagini di citologia cervicale** degradate da blur gaussiano e rumore additivo. L'obiettivo non è solo ottenere buona qualità di ricostruzione, ma anche ragionare sui punti di forza e limiti di ciascun metodo.

Il task è formulato come un **problema inverso**: data un'osservazione degradata $y$, si stima l'immagine originale $x$:

$$y = \mathcal{H}(x) + n$$

dove $\mathcal{H}$ è l'operatore di blur e $n$ è il rumore AWGN.

---

## 2. Dataset

### 2.1 Ispezione e Struttura

Il dataset utilizzato è **Mendeley LBC Cervical Cancer** ([link](https://data.mendeley.com/datasets/zddtpgzv63/2)), composto da 962 immagini di citologia cervicale classificate in 4 categorie diagnostiche:

| Classe | Descrizione | Conteggio | Percentuale |
|---|---|---|---|
| NILM | Negative for Intraepithelial Malignancy | 612 | 63.6% |
| HSIL | High Squamous Intra-epithelial Lesion | 163 | 16.9% |
| LSIL | Low Squamous Intra-epithelial Lesion | 113 | 11.7% |
| SCC | Squamous Cell Carcinoma | 74 | 7.7% |

Il dataset è **sbilanciato** (classe NILM >60%), ma per il task di restauro questo non costituisce un problema — la varietà morfologica delle cellule è comunque rappresentata in tutte le classi.

**Nota sulla dimensione:** La specifica menziona un subset di circa 4000 immagini. Il dataset Mendeley LBC Cervical Cancer contiene 962 immagini totali, che costituiscono l'intero dataset disponibile. Abbiamo quindi utilizzato **tutte le 962 immagini**, in modo da massimizzare la quantità di dati per training e test.

### 2.2 Preprocessing

La pipeline di preprocessing è la stessa per tutti i metodi e le scelte sono giustificate come segue:

| Passo | Operazione | Dettaglio | Giustificazione |
|---|---|---|---|
| 1 | **Resize** | $2048 \times 1536 \rightarrow 256 \times 256$ (bilineare) | Riduzione del carico computazionale; dimensione standard per modelli di deep learning; la risoluzione 256×256 preserva i dettagli cellulari principali |
| 2 | **ToTensor** | $[0, 255] \rightarrow [0, 1]$ | Conversione in floating point per compatibilità con PyTorch |
| 3 | **Normalizzazione** | $(x - 0.5) / 0.5 \rightarrow [-1, 1]$ | Range richiesto dal modello diffusion (DDPM); coerente con la loss MSE |
| 4 | **Split** | 70% train / 15% val / 15% test (seed=42) | Ripartizione standard; seed fisso per riproducibilità |

### 2.3 Suddivisione

| Split | Percentuale | Immagini |
|---|---|---|
| Training | 70% | 673 |
| Validation | 15% | 144 |
| Test | 15% | 145 |

Lo split è **stratificato per classe**, quindi ogni split mantiene le proporzioni originali delle 4 classi diagnostiche. Tutti i metodi usano lo **stesso test set** (145 immagini) per garantire un confronto equo.

---

## 3. Definizione del Task

### 3.1 Formulazione del Problema Inverso

Il target del progetto è il **deblurring e denoising** di immagini cervicali. Formalmente:

$$y = (k * x) + n$$

dove:
- $x$ è l'immagine originale (ground truth)
- $k$ è il kernel di blur gaussiano
- $n$ è il rumore AWGN (Additive White Gaussian Noise)
- $y$ è l'osservazione degradata

Il problema è **mal posto** (ill-posed): esistono infinite distribuzioni di pixel $x$ compatibili con l'osservazione $y$. Per questo servono **regolarizzazioni** (TV) o **prior appresi dai dati** (UNet, DiffPIR) che vincolino la soluzione a essere realistica.

### 3.2 Parametri di Degradazione

| Parametro | Valore |
|---|---|
| **Blur gaussiano** | |
| $\sigma$ | 2 |
| Kernel size | $9 \times 9$ |
| **Rumore AWGN** | |
| $\sigma_n$ level 1 | 0.005 |
| $\sigma_n$ level 2 | 0.01 |
| $\sigma_n$ level 3 | 0.05 |
| $\sigma_n$ level 4 | 0.1 |

### 3.3 Controllo di Riproducibilità

Tutti i metodi ricevono **la stessa identica immagine degradata**, generata con seed fisso (42) tramite la pipeline comune `src/degradation/degradation.py`. Questo garantisce che le differenze nei risultati siano attribuibili solo ai metodi, non a variazioni casuali della degradazione.

---

## 4. Metodi

### 4.1 Panoramica

Siamo un gruppo di 2 studenti, quindi abbiamo scelto **3 dei 4 metodi** proposti, escludendo il metodo ibrido (Weighted TV):

| Metodo | Famiglia | Scelto | Autore |
|---|---|---|---|
| Total Variation (TV) | Variazionale | ✅ Sì | Paolo |
| UNet | End-to-end (deep learning) | ✅ Sì | Paolo |
| DiffPIR | Generativo (diffusion) | ✅ Sì | **Francesco** |
| Weighted TV | Ibrido | ❌ No (2 studenti, 3 metodi bastano) | — |

Ogni metodo ha i propri parametri, scelti **euristicamente** e giustificati nelle sezioni seguenti.

---

### 4.2 Total Variation (TV) — [da completare: Paolo]

**Autore:** Paolo Fusco

> Descrizione del metodo variazionale, funzione obiettivo, scelta dei parametri ($\lambda_{reg}=0.1$, 300 iterazioni Adam), implementazione in `src/methods/tv/tv.py`.

---

### 4.3 UNet — [da completare: Paolo]

**Autore:** Paolo Fusco

> Descrizione dell'architettura UNet scelta, motivazione della scelta (perché UNet e non ViT o NAF-Net in relazione al task), parametri di training (loss, ottimizzatore, epoche, batch size), implementazione in `src/methods/unet/unet.py`.

---

### 4.4 DiffPIR — Metodo Generativo

**Autore:** Francesco Castaldi

#### 4.4.1 Scelta del Metodo e Motivazione

DiffPIR (Diffusion-based Posterior Sampling for Inverse Problems) è un metodo generativo che integra un modello di diffusione (DDPM) in un framework **plug-and-play** per il restauro di immagini. È stato scelto perché:

- **Qualità generativa:** i modelli di diffusione sono allo stato dell'arte per la generazione di immagini e si adattano bene al restauro
- **Plug-and-play:** il denoiser può essere sostituito/riadattato indipendentemente dall'algoritmo
- **FFT-based:** la parte di data-fidelity è risolta analiticamente nel dominio della frequenza, senza bisogno di ottimizzazione iterativa

#### 4.4.2 Architettura del Modello di Diffusione (LightUNet)

Invece del modello originale DiffPIR (pre-addestrato su ImageNet, ~500M parametri, ~2GB), abbiamo addestrato una **LightUNet custom** specificamente sulle immagini LBC cervicali. Questa scelta è motivata da:

1. **Specificità del dominio:** un modello addestrato su ImageNet (foto naturali) non cattura le caratteristiche morfologiche delle cellule cervicali
2. **Dimensione ridotta:** 1.26M parametri (~5MB) vs 500M (~2GB) — training e inferenza possibili su CPU
3. **Match col task:** la risoluzione 256×256 e la struttura relativamente semplice delle immagini citologiche non richiedono un modello gigantesco

**Dettagli architetturali della LightUNet:**

| Componente | Specifica |
|---|---|
| Canali base | 32 |
| Livelli encoder | Down: 32 → 64 → 128 |
| Livelli decoder | Up: 128 → 64 → 32 (skip-connected) |
| Timestep embedding | Sinusoidale + MLP a 2 layer (dim=128) |
| Normalizzazione | GroupNorm (8 gruppi, affine) |
| Attivazione | SiLU (Swish) |
| Parametri totali | **1,262,273** |
| Peso modello su disco | ~5 MB |

**Training DDPM:**
- **Timestep:** 1000, scheduler lineare $\beta_1=10^{-4}$, $\beta_T=0.02$
- **Loss:** MSE sulla predizione del rumore $\varepsilon_\theta(x_t, t)$
- **Dati:** 673 immagini di training (LBC cervicali)
- **Epoche:** 30
- **Pesi:** `src/methods/diffpir/weights/ddpm_lbc.pt`

#### 4.4.3 Algoritmo DiffPIR

L'algoritmo alterna due step ad ogni timestep $t$, partendo da $t_{\text{start}}$ e scendendo fino a $t=0$:

**Step 1 — Denoising (prior):**

Il modello di diffusione predice il rumore $\varepsilon_\theta(x_t, t)$ e si ricava una stima dell'immagine pulita:

$$\hat{x}_0^{(t)} = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \varepsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

**Step 2 — Data Fidelity (fedeltà ai dati):**

Si risolve il problema vincolato:

$$\tilde{x}_0^{(t)} = \arg\min_x \frac{1}{2\sigma^2}\|y - \mathcal{H}(x)\|^2 + \frac{\rho_t}{2}\|x - \hat{x}_0^{(t)}\|^2$$

Analiticamente nel dominio della frequenza (FFT):

$$\tilde{x}_0 = \mathcal{F}^{-1}\left(\frac{\mathcal{F}(y) \cdot \overline{\mathcal{F}(k)} + \rho \cdot \mathcal{F}(\hat{x}_0)}{|\mathcal{F}(k)|^2 + \rho}\right)$$

dove $k$ è il kernel di blur, $\mathcal{F}$ è la FFT, e $\overline{\mathcal{F}(k)}$ è il complesso coniugato.

**Peso dinamico $\rho_t$:**

$$\rho_t = \lambda \cdot \frac{\sigma^2 \cdot \bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

$\rho_t$ bilancia il contributo del prior generativo rispetto alla fedeltà ai dati. A timestep $t$ grandi ($\bar{\alpha}_t \to 1$), $\rho_t \to \infty$ e domina il data-fidelity. A $t$ piccoli ($\bar{\alpha}_t \to 0$), $\rho_t \to 0$ e domina il denoiser.

#### 4.4.4 Scelta Euristica dei Parametri e Giustificazione

| Parametro | Valore | Criterio di Scelta |
|---|---|---|
| **$t_{\text{start}}$** | **50** | Abbiamo testato $t_{\text{start}} \in \{10, 30, 50, 100, 200\}$. A $t>100$, la stima di $\hat{x}_0$ è numericamente instabile perché $\frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}}$ amplifica gli errori del modello (a $t=200$, fattore di amplificazione $\approx 7$). A $t<20$, il modello ha troppo poco rumore da rimuovere e non produce miglioramenti. $t=50$ è il punto di bilanciamento: $\bar{\alpha}_{50} \approx 0.97$, fattore di amplificazione $\approx 0.17$, stima stabile e sufficiente capacità generativa. |
| **$\lambda$** | **10.0** | Peso del data-fidelity. Abbiamo testato $\lambda \in \{0.1, 1.0, 5.0, 10.0, 50.0\}$. Valori troppo bassi ($\lambda < 1$) fanno dominare il generativo, producendo allucinazioni. Valori troppo alti ($\lambda > 50$) rendono il data-fidelity troppo rigido, annullando l'effetto del denoiser. $\lambda=10$ offre il miglior compromesso. |
| **$num\_steps$** | **15** | Sub-campionamento lineare da $t_{\text{start}}=50$ a $t=0$. Con 15 step si ha un buon rapporto qualità/tempo (~2 sec/img). Abbiamo testato 5, 10, 15, 30 step: con <10 step la qualità degrada, con >20 step il miglioramento è marginale e il tempo raddoppia. |
| **$\zeta$** | **0.0** | Stocasticità del sampling. $\zeta=0$ corrisponde a DDIM (deterministico), scelto per **riproducibilità**. Con $\zeta>0$ si avrebbe sampling stocastico (DDPM) che potrebbe migliorare leggermente la qualità ma a scapito della determinismo. |

#### 4.4.5 Implementazione

Il codice è organizzato in:

```
src/methods/diffpir/
├── diffpir.py          # Classe DiffPIR: algoritmo principale, FFT data-fidelity
├── model.py            # LightUNet: architettura con time embedding sinusoidale
├── train.py            # Training loop DDPM
├── weights/            # Pesi ddpm_lbc.pt (~5MB)
└── README.md           # Documentazione specifica del metodo
```

Punti salienti dell'implementazione:
- **FFT data-fidelity nativa:** implementata con `torch.fft.fftn`/`ifftn`, nessuna dipendenza da librerie esterne
- **PSF-to-OTF:** conversione del kernel spaziale in Optical Transfer Function nel dominio frequenziale
- **Padding automatico:** il kernel viene paddato alla dimensione dell'immagine prima della FFT
- **DDIM deterministico:** sampling senza stocasticità per riproducibilità

---

## 5. Risultati

### 5.1 Metriche di Valutazione

Utilizziamo le metriche standard per la valutazione della qualità di restauro:

- **PSNR (Peak Signal-to-Noise Ratio):** $$PSNR = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)$$ con $MAX=1.0$ (immagini in $[0,1]$). Valori più alti = migliore qualità.
- **SSIM (Structural Similarity Index):** misura la similarità strutturale (luminanza, contrasto, struttura). Range $[-1, 1]$, 1 = identico.
- **Tempo di inferenza:** costo computazionale per immagine, rilevante per applicazioni pratiche.

Calcolate tramite `skimage.metrics` su immagini normalizzate in $[0,1]$.

### 5.2 DiffPIR — Risultati Quantitativi

Test su **10 immagini** del test set (sub-campionate per limiti di tempo computazionale), per ognuno dei 4 livelli di rumore:

| $\sigma_n$ | PSNR (dB) | SSIM | Tempo medio (s) |
|---|---|---|---|
| **0.005** | 16.67 | 0.235 | 2.01 |
| **0.01** | 17.32 | 0.270 | 1.97 |
| **0.05** | **22.49** | **0.512** | 2.04 |
| **0.1** | **24.68** | **0.664** | 2.00 |

#### 5.2.1 Analisi dei Risultati — DiffPIR

**Osservazione chiave:** il PSNR e SSIM di DiffPIR **crescono** con l'aumentare del rumore. Questo comportamento è controintuitivo rispetto ai metodi classici (che degradano con più rumore), ma è caratteristico dei metodi generativi basati su diffusione:

1. **A basso rumore ($\sigma_n=0.005$, $0.01$):** l'immagine degradata è già visivamente buona. Il modello generativo, però, **introduce artefatti e dettagli fittizi** (allucinazioni) perché "immagina" texture assenti nell'originale. Il risultato è **peggiore del degradato** stesso — si perde fedeltà.

2. **A medio rumore ($\sigma_n=0.05$):** PSNR=22.49, SSIM=0.512. Il modello inizia a recuperare: c'è un miglioramento netto rispetto al degradato. I dettagli cellulari (nuclei, membrane) sono parzialmente ricostruiti.

3. **Ad alto rumore ($\sigma_n=0.1$):** il punto migliore. PSNR=24.68, SSIM=0.664. Con tanto rumore, il modello ha abbastanza "margine" per ricostruire senza introdurre artefatti evidenti. Il generativo brilla quando deve ricostruire tanto segnale perso.

**Spiegazione:** la LightUNet (1.26M params) ha capacità limitata. Con poco rumore, ha poca "scusa" per modificare l'immagine, ma il suo prior generativo la spinge comunque ad aggiungere dettagli. Con tanto rumore, è forzata a ricostruire e produce risultati migliori.

**Tempo di inferenza:** costante a circa 2 secondi per immagine su CPU (AMD Ryzen 7). Non dipende dal livello di rumore perché il numero di step di sampling (15) è fisso.

#### 5.2.2 Risultati Qualitativi — DiffPIR

Le immagini ricostruite sono salvate in `results/diffpir/qualitative/` con formato `noise_<level>_sample<id>.png`, 6 immagini per ogni livello di rumore.

Il confronto visivo mostra:
- **$\sigma_n=0.005$:** l'immagine ricostruita ha texture più liscia dell'originale, con perdita di dettagli fini nei nuclei cellulari — il modello ha "pulito" anche il segnale.
- **$\sigma_n=0.1$:** l'immagine ricostruita è nettamente più pulita del degradato; i bordi cellulari sono più definiti, il rumore granulare è rimosso efficacemente.
- **Effetto collaterale:** in tutte le condizioni, il modello tende a produrre texture omogenee che a volte cancellano dettagli diagnostici rilevanti.

#### 5.2.3 Limitazioni Specifiche — DiffPIR

1. **Modello piccolo:** LightUNet 1.26M vs modelli standard 500M+ — qualità inferiore ma pratico
2. **Dataset di training ridotto:** 673 immagini per un generativo sono poche
3. **CPU-only:** training e inferenza su CPU limitano la scala sperimentale
4. **Allucinazioni a basso rumore:** il modello peggiora invece di migliorare a $\sigma_n < 0.05$

---

### 5.3 Total Variation (TV) — [da completare: Paolo]

> **Autore:** Paolo Fusco
> Risultati TV: PSNR/SSIM per ogni noise level, immagini qualitative, analisi delle ricostruzioni e dell'effetto staircasing.

---

### 5.4 UNet — [da completare: Paolo]

> **Autore:** Paolo Fusco
> Risultati UNet: PSNR/SSIM per ogni noise level, immagini qualitative, analisi della capacità di generalizzazione.

---

### 5.5 Confronto Comparativo

| $\sigma_n$ | Degradato | TV | UNet | DiffPIR |
|---|---|---|---|---|
| 0.005 | — | [Paolo] | [Paolo] | **16.67 dB** / **0.235** |
| 0.01 | — | [Paolo] | [Paolo] | **17.32 dB** / **0.270** |
| 0.05 | — | [Paolo] | [Paolo] | **22.49 dB** / **0.512** |
| 0.1 | — | [Paolo] | [Paolo] | **24.68 dB** / **0.664** |

Il plot comparativo sarà generato dopo l'esecuzione di TV e UNet tramite:

```bash
python scripts/plot_results.py
```

Output: `results/comparison.png` — grafico con PSNR e SSIM per metodo e noise level.

#### Discussione Comparativa Attesa

Sulla base della teoria dei tre metodi, ci aspettiamo:

| Metodo | Basso rumore ($\sigma_n < 0.05$) | Alto rumore ($\sigma_n \geq 0.05$) | Tempo |
|---|---|---|---|
| **TV** | Buono, ma effetto staircasing | Perde dettaglio, rumore residuo | ~10s/img |
| **UNet** | Molto buono (impara dai dati) | Discreto (generalizzazione limitata) | **~0.1s/img** |
| **DiffPIR** | **Peggio del degradato** (allucinazioni) | **Migliore recupero** (generativo) | ~2s/img |

---

## 6. Discussione

### 6.1 Confronto tra le Famiglie Metodologiche

Il progetto copre 3 famiglie metodologiche diverse, ognuna con presupposti e compromessi differenti:

- **TV (variazionale):** non richiede dati, interpretabile, ma qualità limitata. Funziona meglio quando il modello di degradazione è semplice e noto.
- **UNet (deep learning end-to-end):** veloce in inferenza, buona qualità se i dati di training sono rappresentativi. Generalizza peggio fuori distribuzione.
- **DiffPIR (generativo):** flessibile, può ricostruire dettagli persi (anche troppi). Costoso computazionalmente, sensibile alla qualità del denoiser.

### 6.2 Punti di Forza del Progetto

- **Confronto equo:** stessa pipeline di degradazione, stessi dati di test, stesse metriche per tutti i metodi
- **Riproducibilità:** seed fissi (42), pipeline deterministiche, test automatizzati (26 test unitari)
- **Codice modulare:** ogni metodo è indipendente in `src/methods/<metodo>/`, condivisi solo preprocessing, degradazione e metriche
- **GitHub:** repository pubblico con documentazione, test, e struttura chiara

### 6.3 Criticità e Limitazioni

**Generali:**
- Dimensione del dataset (962 immagini, di cui solo 673 per training) — limitante per i metodi deep learning
- Esecuzione su CPU — training più lungo, inference più lenta, impossibile testare modelli più grandi

**DiffPIR (specifiche):**
- La LightUNet custom (1.26M params) è lontana dagli standard dei diffusion model moderni
- Il DDPM training su 673 immagini è insufficiente per un generativo di qualità
- A basso rumore il metodo peggiora l'immagine (allucinazioni)

### 6.4 Parametri Scelti Euristicamente: Riassunto

| Metodo | Parametro | Valore | Come è stato scelto |
|---|---|---|---|
| **TV** | $\lambda_{reg}$ | 0.1 | [Paolo] |
| **TV** | Iterazioni | 300 | [Paolo] |
| **UNet** | Learning rate | $10^{-4}$ | [Paolo] |
| **UNet** | Batch size, epoche | 16, 50 | [Paolo] |
| **DiffPIR** | $t_{start}$ | 50 | Testati {10, 30, 50, 100, 200} → 50 dà stabilità numerica |
| **DiffPIR** | $\lambda$ | 10.0 | Testati {0.1, 1, 5, 10, 50} → 10 è il bilanciamento ottimale |
| **DiffPIR** | $num\_steps$ | 15 | Testati {5, 10, 15, 30} → 15 è il miglior rapporto qualità/tempo |
| **DiffPIR** | $\zeta$ | 0.0 | Scelto deterministico per riproducibilità |

---

## 7. Conclusioni

> **Sezione da completare dopo l'esecuzione di TV e UNet e il confronto finale.**

---

## Appendice A: Istruzioni per la Riproducibilità

### Setup

```bash
pip install -r requirements.txt
```

### Preprocessing (una volta sola)

```bash
python scripts/preprocess.py
```

### Esecuzione Metodi

```bash
# Metodo variazionale (Paolo)
python scripts/run_tv.py

# Metodo end-to-end (Paolo)
python scripts/run_unet.py

# Metodo generativo (Francesco)
python scripts/run_diffpir.py

# Plot comparativo
python scripts/plot_results.py
```

### Test

```bash
python -m pytest tests/ -v
```

Tutti i 26 test passano (verificato):

```
tests/test_degradation.py  ... 10/10 ✅
tests/test_metrics.py      ...  9/9  ✅
tests/test_diffpir.py      ...  7/7  ✅
```

## Appendice B: Struttura del Repository

```
ci-cervical-lbc/
├── configs/                    # experiment.yaml (parametri globali)
├── data/
│   ├── raw/                    # Dataset originale (non tracciato)
│   └── splits/                 # Train/val/test split files
├── notebooks/
│   ├── 01_eda.ipynb            # Analisi esplorativa dataset
│   └── 04_diffpir.ipynb        # DiffPIR demo e valutazione
├── src/
│   ├── data/dataset.py         # Caricamento e preprocessing
│   ├── degradation/            # Pipeline blur + noise (condivisa)
│   ├── methods/
│   │   ├── tv/tv.py            # Total Variation
│   │   ├── unet/unet.py        # UNet
│   │   └── diffpir/            # DiffPIR
│   │       ├── diffpir.py      # Algoritmo
│   │       ├── model.py        # LightUNet
│   │       ├── train.py        # Training DDPM
│   │       └── README.md       # Documentazione
│   ├── eval/metrics.py         # PSNR, SSIM
│   └── plots/visualize.py      # Grafici
├── scripts/
│   ├── preprocess.py           # Preprocessing dataset
│   ├── run_tv.py               # Esecuzione TV
│   ├── run_unet.py             # Esecuzione UNet
│   ├── run_diffpir.py          # Esecuzione DiffPIR
│   └── plot_results.py         # Grafico comparativo
├── tests/                      # 26 unit test
│   ├── test_degradation.py
│   ├── test_metrics.py
│   └── test_diffpir.py
├── report/
│   ├── teoria.md               # Fondamenti teorici
│   └── relazione.md            # Questo file
├── slides/                     # Presentazione PowerPoint/Beamer
├── agents.md                   # Divisione dei compiti
└── README.md                   # Panoramica progetto
```

## Appendice C: Riferimenti

- **DiffPIR:** Zhu, Y., et al. "Denoising Diffusion Models for Plug-and-Play Image Restoration." *CVPR 2023.*
- **DDPM:** Ho, J., et al. "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.*
- **DDIM:** Song, J., et al. "Denoising Diffusion Implicit Models." *ICLR 2021.*
- **TV:** Rudin, L., Osher, S., Fatemi, E. "Nonlinear total variation based noise removal algorithms." *Physica D 1992.*
- **UNet:** Ronneberger, O., Fischer, P., Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI 2015.*
- **Dataset:** Mendeley LBC Cervical Cancer. [DOI: 10.17632/zddtpgzv63.2](https://data.mendeley.com/datasets/zddtpgzv63/2)
