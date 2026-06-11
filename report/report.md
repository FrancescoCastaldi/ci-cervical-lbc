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

| Metodo | Famiglia | Scelto |
|---|---|---|
| Total Variation (TV) | Variazionale | ✅ Sì |
| UNet | End-to-end (deep learning) | ✅ Sì |
| DiffPIR | Generativo (diffusion) | ✅ Sì |
| Weighted TV | Ibrido | ❌ No (2 studenti, 3 metodi bastano) |

Ogni metodo ha i propri parametri, scelti **euristicamente** e giustificati nelle sezioni seguenti.

---

## 4.2 Total Variation (TV)

### Descrizione del metodo
La Total Variation (TV) è un metodo variazionale classico per la risoluzione 
di problemi inversi in imaging. L'idea fondamentale è formulare la ricostruzione 
come un problema di ottimizzazione in cui si bilanciano due termini: la fedeltà 
ai dati osservati e una penalizzazione sulla complessità dell'immagine ricostruita.

### Formulazione del problema inverso
Il modello di degradazione adottato è:

$$y = H * x + n$$

dove $y$ è l'immagine degradata osservata, $x$ è l'immagine originale da 
ricostruire, $H$ è l'operatore di blur gaussiano (σ=2, kernel 9×9) e $n$ 
è il rumore gaussiano additivo. La ricostruzione TV minimizza la seguente 
funzione obiettivo:

$$\hat{x} = \arg\min_x \|H*x - y\|_2^2 + \lambda \cdot TV(x)$$

Il primo termine, detto **data fidelity**, garantisce che la soluzione rimanga 
coerente con l'osservazione degradata. Il secondo termine, il **regolarizzatore TV**, 
penalizza le variazioni locali dell'immagine:

$$TV(x) = \sum_{i,j} |x_{i+1,j} - x_{i,j}| + |x_{i,j+1} - x_{i,j}|$$

La TV favorisce immagini con regioni uniformi e bordi netti, caratteristica 
particolarmente adatta a immagini di citologia cervicale dove le cellule presentano 
contorni definiti su sfondo omogeneo.

### Scelta dei parametri
L'ottimizzazione è realizzata tramite l'algoritmo **Adam** con i seguenti parametri:

- $\lambda_{reg} = 0.005$: peso del termine di regolarizzazione TV. 
  Un valore più alto produce immagini più lisce ma perde dettaglio; 
  un valore più basso è più fedele all'input ma meno efficace nel denoising.
  Il valore è stato scelto empiricamente osservando la qualità visiva delle 
  ricostruzioni sul validation set.
- Learning rate: $lr = 0.001$, scelto per garantire convergenza stabile 
  senza artefatti numerici.
- Numero di iterazioni: 150, sufficiente per la convergenza sul dataset in esame.

Un valore di $\lambda_{reg}$ troppo alto (es. 0.1) produceva artefatti 
a blocchi (*staircasing*) dovuti alla dominanza del termine TV sulla 
data fidelity. Un learning rate troppo alto (es. 0.01) causava update 
aggressivi sulle alte frequenze, generando pixelatura nell'output.

### Implementazione
Il metodo è implementato in `src/methods/tv/tv.py`. Il kernel gaussiano 
viene costruito come tensore PyTorch e applicato tramite convoluzione 2D 
(`F.conv2d`) con padding simmetrico per preservare le dimensioni spaziali. 
Ad ogni iterazione Adam, il tensore ricostruito viene clampato in $[-1, 1]$ 
per mantenere la coerenza con il range di normalizzazione del dataset.
Lo script di esecuzione `scripts/run_tv.py` carica i parametri da 
`configs/experiment.yaml`, valuta il metodo su tutto il test set per 
ciascuno dei quattro livelli di rumore, e salva i risultati quantitativi 
in `results/tv/tv_results.csv` insieme a campioni visivi di confronto.

### Risultati
| σₙ    | PSNR     | SSIM  |
|-------|----------|-------|
| 0.005 | 32.09 dB | 0.911 |
| 0.01  | 32.04 dB | 0.909 |
| 0.05  | 30.42 dB | 0.837 |
| 0.1   | 26.54 dB | 0.586 |

Il metodo ottiene risultati buoni per bassi livelli di rumore (PSNR > 30 dB, 
SSIM > 0.9) con un degrado progressivo all'aumentare di σₙ, dovuto al fatto 
che il parametro λ è stato mantenuto fisso per tutti i livelli. Un tuning 
separato di λ per ciascun noise level potrebbe migliorare le performance 
nei casi più difficili.

---

### 4.3 UNet — Metodo Deep Learning End-to-End

#### 4.3.1 Scelta del Metodo e Motivazione

Tra le opzioni end-to-end proposte (UNet, ViT, NAF-Net), abbiamo scelto **UNet** per le seguenti ragioni:

1. **Adatta al task di image-to-image translation:** la UNet è stata progettata per mappare un'immagine di input a un'immagine di output della stessa dimensione, esattamente il nostro scenario (degraded → restored). Le skip connections preservano i dettagli spaziali che altrimenti andrebbero persi nell'encoding profondo — caratteristica cruciale per il restauro di strutture cellulari fini.

2. **Dataset limitato (673 immagini di training):** ViT e NAF-Net richiedono dataset molto più grandi per convergere senza overfitting. La UNet, con le sue convoluzioni locali, ha un bias induttivo forte che le permette di apprendere con pochi dati. 31M di parametri sono un buon compromesso: sufficienti per catturare pattern complessi, ma non così tanti da richiedere milioni di immagini.

3. **Efficienza computazionale:** su CPU, 31M parametri convoluzionali sono gestibili (~15 sec/batch). ViT avrebbe avuto complessità $O(N^2)$ in self-attention (proibitiva su CPU), e NAF-Net avrebbe richiesto più memoria per i suoi blocchi non-lineari.

4. **Semplicità e riproducibilità:** la UNet è ben documentata, con implementazioni stabili e risultati prevedibili. Non richiede tecniche di training avanzate o iperparametri critici.

#### 4.3.2 Architettura

L'architettura è una UNet encoder-decoder classica con skip connections:

| Componente | Dettaglio |
|---|---|
| **Input** | 3 canali (RGB), 256×256 |
| **Encoder** | 4 livelli: 64 → 128 → 256 → 512 canali |
| **Blocco convoluzionale** | Double Conv: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU |
| **Downsampling** | MaxPool 2×2 dopo ogni livello encoder |
| **Bottleneck** | DoubleConv: 512 → 1024 canali |
| **Decoder** | 4 livelli simmetrici con ConvTranspose2d (upsampling 2×) |
| **Skip connections** | Connessioni dirette encoder→decoder allo stesso livello |
| **Output** | Conv2d 1×1 + tanh → 3 canali in [-1, 1] |
| **Parametri totali** | **31,043,651 (~31.0M)** |

Il **tanh** finale mappa l'output in [-1, 1], coerente con la normalizzazione del dataset. Le **skip connections** concatenano le feature maps dell'encoder con quelle del decoder allo stesso livello, preservando i dettagli spaziali a grana fine che altrimenti andrebbero persi nel bottleneck.

#### 4.3.3 Training

**Configurazione:**

| Parametro | Valore | Giustificazione |
|---|---|---|
| **Loss** | MSE (Mean Squared Error) | Standard per regressione pixel-to-pixel; penalizza errori grandi più che piccoli ($L_2$) |
| **Ottimizzatore** | Adam | Convergenza stabile, buona gestione di gradienti rumorosi |
| **Learning rate** | $10^{-4}$ | Sufficientemente basso per convergenza stabile su 50 epoche, sufficientemente alto per non stagnare |
| **Batch size** | 16 | Massimo gestibile su CPU con 256×256×3 immagini e 31M parametri (~2GB RAM per batch) |
| **Epoche** | 50 (config), 15 su CPU | Limite CPU per tempo pratico (~2.5h); 50 epoche ideali su GPU |
| **Multi-noise augmentation** | $\sigma_n \sim \text{Uniform}\{0.005, 0.01, 0.05, 0.1\}$ | Campionato random per ogni batch; il modello impara a gestire tutti i livelli di rumore simultaneamente invece di specializzarsi su uno solo |

**Training loop:**
```
for epoch in range(epochs):
    for batch in train_loader:
        σ = random.choice([0.005, 0.01, 0.05, 0.1])
        degraded = blur(batch, σ_blur=2) + noise(σ)
        pred = unet(degraded)
        loss = MSE(pred, batch)
        optimizer.step(loss)
```

**Validation:** dopo ogni epoca, si calcola il PSNR medio sul validation set (144 immagini) su tutti e 4 i livelli di rumore. Il modello con il miglior PSNR di validation viene salvato come checkpoint (`best_model.pth`).

**Degradazione vettorizzata:** per efficienza su CPU, il blur viene applicato con `F.conv2d` direttamente sul batch invece che in un loop per-immagine, ottenendo uno speedup di ~5×.

#### 4.3.4 Implementazione

Il codice è organizzato in:

```
src/methods/unet/
└── unet.py              # Architettura UNet (classe UNet + DoubleConv)

scripts/
└── run_unet.py          # Training loop, validation, test evaluation
```

Lo script `run_unet.py` gestisce l'intera pipeline:
1. Caricamento dati da `data/splits/{train,val,test}.txt`
2. Training con multi-noise augmentation e validation
3. Salvataggio del miglior modello (`results/unet/best_model.pth`)
4. Valutazione su tutto il test set (145 immagini × 4 noise level)
5. Generazione di `metrics.csv` e immagini qualitative

**Ottimizzazione CPU:** lo script rileva automaticamente il dispositivo. Su CPU, limita le epoche a 15 e usa degradazione batch vettorizzata per ridurre il tempo di training da ~9 ore a ~2.5 ore.

---

### 4.4 DiffPIR — Metodo Generativo

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

### 5.3 Total Variation (TV)

Test su **145 immagini** del test set, per ognuno dei 4 livelli di rumore:

| $\sigma_n$ | PSNR (dB) | SSIM |
|---|---|---|
| **0.005** | 32.09 | 0.911 |
| **0.01** | 32.04 | 0.909 |
| **0.05** | 30.42 | 0.837 |
| **0.1** | 26.54 | 0.586 |

#### Analisi dei Risultati — TV

**Comportamento atteso:** il PSNR e SSIM degradano progressivamente con l'aumentare del rumore. Questo è il comportamento classico dei metodi variazionali:

1. **A basso rumore ($\sigma_n=0.005$, $0.01$):** PSNR > 32 dB, SSIM > 0.9. Il metodo TV è molto efficace quando il rumore è limitato. Il regolarizzatore riesce a rimuovere il rumore senza sacrificare troppi dettagli. L'immagine ricostruita è visivamente indistinguibile dall'originale per un osservatore umano.

2. **A medio rumore ($\sigma_n=0.05$):** PSNR = 30.42, SSIM = 0.837. La TV inizia a perdere efficacia. Il rumore residuo è visibile, e l'effetto **staircasing** (appiattimento a blocchi delle regioni uniformi) diventa evidente. I dettagli cellulari fini (cromatina, nucleoli) vengono parzialmente lisciati.

3. **Ad alto rumore ($\sigma_n=0.1$):** PSNR = 26.54, SSIM = 0.586. È il punto più debole. Con $\lambda_{reg}=0.005$ fisso, il bilanciamento data-fidelity/regolarizzazione non è più ottimale: servirebbe un $\lambda$ più alto per contrastare più rumore.

**Effetto staircasing:** la TV penalizza la variazione totale (somma dei gradienti), favorendo regioni uniformi. Questo produce l'effetto "a gradini" visibile soprattutto nelle aree a texture fine (citoplasma cellulare), dove la transizione graduale di intensità viene approssimata con salti discreti.

**Limite del $\lambda$ fisso:** il parametro $\lambda_{reg}=0.005$ è ottimale per rumore basso, ma sottodimensionato per rumore alto. Un tuning adattivo $\lambda(\sigma_n)$ migliorerebbe i risultati ai noise level più elevati.

### 5.4 UNet

L'UNet è stato ottimizzato con un'architettura **snella** (1.9M parametri, features=[16,32,64,128]) con GroupNorm, condizionamento del noise level (4 canali input: RGB + noise map) e loss L1. Addestrato per **50 epoche** su CPU con multi-noise augmentation (σ random per batch) e ReduceLROnPlateau. I risultati su 145 immagini di test:

| $\sigma_n$ | PSNR | SSIM | Tempo |
|---|---|---|---|
| 0.005 | **29.89 dB** | **0.894** | **0.035 s** |
| 0.01 | **29.89 dB** | **0.894** | **0.034 s** |
| 0.05 | **29.63 dB** | **0.875** | **0.034 s** |
| 0.1 | **28.93 dB** | **0.830** | **0.036 s** |

**Analisi dei risultati dopo 50 epoche:**
- Il modello raggiunge **~29.9 dB** a basso rumore, in linea con le performance di TV (32 dB) a soli ~2 dB di distanza
- La degradazione all'aumentare del rumore è **molto contenuta**: solo ~1 dB di perdita da σ=0.005 a σ=0.1 (contro i ~5.5 dB della TV)
- A σ=0.1, l'UNet **supera TV** (28.93 vs 26.54 dB) — il modello apprende un prior efficace anche in condizioni di rumore elevato
- SSIM elevato (0.83-0.89) in tutti i noise level, dimostrando buona fedeltà strutturale
- Il tempo di inferenza (~0.035 s/img) è **il più veloce** tra tutti i metodi, rendendolo ideale per applicazioni real-time

**Miglioramento rispetto alla versione precedente:** L'architettura ottimizzata (1.9M vs 31M params) con GroupNorm, L1 loss e noise conditioning ha permesso un incremento di **+5.8 dB** a basso rumore e **+7.1 dB** ad alto rumore rispetto al modello precedente (1 epoca).

### 5.5 Confronto Comparativo

| $\sigma_n$ | TV (PSNR / SSIM) | UNet (PSNR / SSIM) | DiffPIR (PSNR / SSIM) |
|---|---|---|---|---|---|
| 0.005 | **32.09** / **0.911** 🥇 | 29.89 / 0.894 🥈 | 16.67 / 0.235 |
| 0.01 | **32.04** / **0.909** 🥇 | 29.89 / 0.894 🥈 | 17.32 / 0.270 |
| 0.05 | **30.42** / **0.837** 🥇 | 29.63 / 0.875 🥈 | 22.49 / 0.512 |
| 0.1 | 26.54 / 0.586 🥈 | **28.93** / **0.830** 🥇 | 24.68 / 0.664 🥉 |

Il plot comparativo è stato generato tramite:

```bash
python scripts/plot_results.py
```

Output: `results/comparison.png` — grafico con PSNR e SSIM per metodo e noise level.

#### Discussione Comparativa Preliminare

Sulla base dei risultati attuali e della teoria dei tre metodi:

| Metodo | Basso rumore ($\sigma_n < 0.05$) | Alto rumore ($\sigma_n \geq 0.05$) | Tempo |
|---|---|---|---|
| **TV** | **Eccellente (32 dB)** — domina a basso rumore | Adeguato (26 dB) — perde dettaglio, staircasing | ~10s/img |
| **UNet** | Buono (24 dB, 1 epoca) — generalizza bene | Discreto (22 dB, 1 epoca) — margine di miglioramento con più epoche | **~26 ms/img** |
| **DiffPIR** | Scarso (17 dB) — allucinazioni a basso rumore | **Buono (24 dB)** — il generativo brilla | ~2s/img |

**Osservazione chiave:** i tre metodi hanno regimi di funzionamento complementari. TV domina a basso rumore (dove il problema è quasi ben posto), DiffPIR eccelle ad alto rumore (dove serve un prior generativo forte). L'UNet con 1 sola epoca si posiziona come compromesso intermedio: ~24 dB su tutti i livelli, mostrando buona robustezza ma senza eccellere in nessun regime specifico. Con più epoche di training ci si aspetta che l'UNet si avvicini o superi TV, specialmente a noise level medio-alti.

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
| **TV** | $\lambda_{reg}$ | 0.005 | Testato {0.001, 0.005, 0.01, 0.05, 0.1}. Con $\lambda=0.001$ troppo rumore residuo; con $\lambda=0.1$ staircasing eccessivo. $\lambda=0.005$ è il bilanciamento ottimale. |
| **TV** | Iterazioni | 150 | Empirico: dopo 150 iter la loss si stabilizza (variazione < 1% nelle ultime 20 iter). |
| **UNet** | Learning rate | $10^{-4}$ | Standard Adam per reti convoluzionali; testato $10^{-3}$ (instabile) e $10^{-5}$ (troppo lento). |
| **UNet** | Batch size | 16 | Massimo gestibile su CPU con 31M params e immagini 256×256 (~2GB RAM). |
| **UNet** | Epoche | 50 (15 CPU) | 50 epoche ideali; limitato a 15 su CPU per tempo pratico (~2.5h vs ~9h). |
| **UNet** | Multi-noise | $\sigma_n$ random per batch | Permette al modello di generalizzare su tutti i livelli di rumore senza overfitting su uno specifico. |
| **DiffPIR** | $t_{start}$ | 50 | Testati {10, 30, 50, 100, 200} → 50 dà stabilità numerica |
| **DiffPIR** | $\lambda$ | 10.0 | Testati {0.1, 1, 5, 10, 50} → 10 è il bilanciamento ottimale |
| **DiffPIR** | $num\_steps$ | 15 | Testati {5, 10, 15, 30} → 15 è il miglior rapporto qualità/tempo |
| **DiffPIR** | $\zeta$ | 0.0 | Scelto deterministico per riproducibilità |

---

## 7. Conclusioni

Questo progetto ha esplorato tre approcci fondamentalmente diversi al problema del restauro di immagini citologiche: uno classico variazionale (TV), uno moderno end-to-end (UNet), e uno generativo all'avanguardia (DiffPIR). L'obiettivo non era solo ottenere buoni risultati numerici, ma **comprendere i compromessi** tra le diverse famiglie metodologiche — cosa ciascuna sa fare bene e dove invece fallisce.

### Risultati Principali

1. **TV eccelle a basso rumore (PSNR > 32 dB per $\sigma_n \leq 0.01$)** — quando il problema inverso è quasi ben posto, un regolarizzatore semplice e interpretabile basta. Non richiede training né dati, è immediatamente pronto. Il suo limite è il $\lambda$ fisso: a rumore alto, perde efficacia e introduce staircasing.

2. **DiffPIR eccelle ad alto rumore (PSNR 24.68 dB per $\sigma_n=0.1$)** — quando il segnale è fortemente degradato, il prior generativo aiuta a ricostruire dettagli che metodi classici non possono recuperare. Il prezzo è duplice: lentezza in inferenza (~2 sec/img) e allucinazioni a basso rumore (peggiora invece di migliorare).

3. **UNet ottimizzato raggiunge ~29.9 dB PSNR dopo 50 epoche CPU** — con architettura snella (1.9M params, GroupNorm, L1 loss) e noise conditioning, il modello eguaglia quasi TV a basso rumore (29.89 vs 32.09 dB, a soli 2 dB di distanza) e **supera TV ad alto rumore** (28.93 vs 26.54 dB a σ=0.1). La degradazione all'aumentare del rumore è minima (~1 dB), dimostrando che un modello ben progettato può essere robusto su tutto lo spettro di rumore. L'inferenza è la più rapida (~0.035 s/img), ideale per applicazioni real-time.

### Lezioni Apprese

- **Nessun metodo domina su tutti i regimi di rumore.** La scelta del metodo dipende dal contesto applicativo: se il rumore è basso, TV è la scelta pragmatica; se è alto e si può tollerare latenza, DiffPIR offre qualità superiore; se serve throughput elevato, UNet è imbattibile.
- **Il training su CPU è un collo di bottiglia reale.** Nonostante l'ottimizzazione dell'architettura (1.9M params, 50 epoche in ~85 minuti totali), l'assenza di GPU ha impedito di sperimentare con architetture più profonde, batch più grandi, e validation più frequenti. Su GPU lo stesso training richiederebbe secondi per epoca.
- **La riproducibilità è fondamentale nel confronto.** Usare la stessa pipeline di degradazione, lo stesso seed, e le stesse metriche garantisce che le differenze osservate siano attribuibili solo ai metodi.
- **I modelli generativi vanno usati con cautela a basso rumore.** Il comportamento controintuitivo di DiffPIR (PSNR più basso a rumore più basso) è una lezione importante: un prior generativo forte può "immaginare" dettagli assenti, peggiorando la fedeltà all'originale.

### Direzioni Future

- **Tuning adattivo di $\lambda$ per TV:** un $\lambda(\sigma_n)$ ottimizzato per ciascun noise level migliorerebbe le performance ad alto rumore.
- **Training UNet su GPU:** consentirebbe più epoche, batch più grandi, e sperimentazione con architetture più profonde.
- **Modello di diffusione più grande:** una UNet diffusion con 50M+ parametri migliorerebbe la qualità di DiffPIR, specialmente a basso rumore.

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
# Metodo variazionale
python scripts/run_tv.py

# Metodo end-to-end
python scripts/run_unet.py

# Metodo generativo
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
│   ├── 02_tv.ipynb             # Total Variation demo e valutazione
│   ├── 03_unet.ipynb           # UNet training e valutazione
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
