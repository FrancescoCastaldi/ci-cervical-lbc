# Computational Imaging - Manuale Completo per l'Esame Orale

> **Unibo - Laurea Magistrale in Informatica**
> Modulo 1: Prof.ssa Elena Loli Piccolomini | Modulo 2: Davide Evangelista
> Link: [Modulo 1](https://elenaloli.github.io/computational-imaging/intro.html) | [Modulo 2](https://devangelista2.github.io/computational-imaging/2025-26/intro/environment-setup.html)
> GitHub Modulo 1: [elenaloli/computational-imaging](https://github.com/elenaloli/computational-imaging) | GitHub Modulo 2: [devangelista2/computational-imaging](https://github.com/devangelista2/computational-imaging)

---

## Indice

- **PARTE I - Fondamenti** (Modulo 1, sez. 1-6) `[p.1-11]`
  - 1. Introduzione `[p.1-2]`
  - 2. Pixel Processing `[p.2-3]`
  - 3. Filtri e Convoluzione `[p.4-5]`
  - 4. Trasformata di Fourier `[p.6-8]`
  - 5. Problemi Inversi `[p.8-10]`
  - 6. Regolarizzazione `[p.10-11]`
- **PARTE II - Deep Learning** (Modulo 2, sez. 7-16) `[p.12-31]`
  - 7. Processing per Reti Neurali `[p.12-13]`
  - 8. PyTorch Essentials `[p.14-15]`
  - 9. ML -> Neural Networks `[p.15-16]`
  - 10. CNN `[p.17-18]`
  - 11. UNet `[p.18-20]`
  - 12. ViT & Loss Design `[p.20-22]`
  - 13. Cross-Domain `[p.22-23]`
  - 14. VAE & GAN `[p.23-26]`
  - 15. Diffusion Models `[p.26-28]`
  - 16. Diffusion per Problemi Inversi `[p.29-31]`
- **Schede Riassuntive** `[p.32-34]`
- **Domande d'Esame** `[p.35-42]`

---

## PARTE I - FONDAMENTI DI IMAGING (Modulo 1)

---

### 1. Introduzione al Computational Imaging `[p.1-2]`

> **Termini**: discretizzazione = convertire segnale continuo in discreto · Nyquist = $f_s \ge 2f_{max}$ · aliasing = artefatto da sottocampionamento (ruote carro) · DICOM = formato medico con metadati · DCT = base JPEG separa frequenze · pixel = unità minima
> **A cosa serve**: capire come campionare senza perdere informazione e comprimere immagini. Dalla fotografia alla risonanza magnetica, senza Nyquist avremmo solo artefatti.

> **Spiegazioni**: CI inverte il **forward model** $y=Ax+e$: misuri dati indiretti ($y$) e ricostruisci l'immagine ($x$) — come capire cosa c'è in una scatola chiusa dal rumore che fa scuotendola. **Nyquist**: $f_s \ge 2f_{max}$ — se campioni sotto questa frequenza, le alte frequenze si ripiegano su quelle basse (**aliasing**: ruote carro nei film). **JPEG**: DCT + quantizzazione — butta via le alte frequenze che l'occhio non percepisce (come descrivere «sabbia colorata» invece di ogni granello). **DICOM**: standard per imaging medico con metadati.

> **All'orale**: «Cos'è il CI?» = invertire $y=Ax+e$ per ricostruire immagini da misure indirette (CT, risonanza). Nyquist: se campioni a meno di $2f_{max}$ le frequenze si ripiegano (aliasing). Ruote carro girano all'indietro. Domanda tipica: «Perché serve Nyquist?» = per evitare aliasing.


**Computational Imaging (CI)** = campo interdisciplinare che unisce acquisizione, elaborazione e analisi delle immagini attraverso modelli computazionali.

Tre pilastri:
- **Image Processing**: elaborare un'immagine per migliorarla o estrarre informazione (filtraggio, enhancement, restauro)
- **Computer Vision**: estrarre informazione semantica dalle immagini (riconoscimento, segmentazione, ricostruzione 3D)
- **Computer Graphics**: sintetizzare immagini a partire da modelli (rendering, simulazione)

#### Immagine Digitale

Un'immagine digitale e una matrice discreta $I \in \mathbb{R}^{M \times N}$ (o $\mathbb{R}^{M \times N \times C}$ per immagini a colori).

- **Risoluzione spaziale**: $M \times N$ pixel
- **Profondita di bit**: numero di valori per pixel (8-bit = 256 livelli)
- **Canali**: 1 (grigio), 3 (RGB), 4 (RGBA)

**Formati file**:
- **DICOM** (Digital Imaging and Communications in Medicine): standard per imaging medico, include metadati clinici
- **PNG**: compressione lossless
- **JPEG**: compressione lossy (DCT + quantizzazione)

**Lettura immagini in Python**:
```python
import cv2          # OpenCV: BGR, uint8, shape (H,W,C)
import matplotlib   # RGB, float [0,1]
from PIL import Image  # Pillow: RGB, vari formati
import imageio      # versatile, supporta DICOM
```

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/intro1.png" alt="Immagine digitale" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/intro2.png" alt="Esempio immagini" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/intro10.png" alt="Tipi immagini" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/intro15.png" alt="CT" width="400">
---

### 2. Pixel Processing `[p.2-3]`

> **Termini**: istogramma = conteggio frequenza livelli grigio · CDF = somma cumulativa · PDF = istogramma normalizzato · equalizzazione = CDF come lookup table · CLAHE = equalizzazione locale a blocchi con clipping · gaussiano = rumore normale · salt&pepper = pixel bianchi/neri · Poisson = rumore dipendente dal segnale
> **A cosa serve**: migliorare la qualita di un'immagine grezza. Aumentare contrasto, regolare luminosita, correggere colori — operazioni base prima di qualsiasi analisi.

> **Spiegazioni**: **Istogramma** $h(k)$ = conteggio pixel per ogni intensità $k$. **Equalizzazione**: usa la **CDF** come lookup table per distribuire i livelli uniformemente su 0-255 — aumenta il contrasto globale (come alzare le ombre in Photoshop). **CLAHE**: versione locale a blocchi con **contrast clipping** (taglia i picchi dell'istogramma locale per non amplificare rumore nelle zone uniformi). **Rumore gaussiano** $\mathcal{N}(0,\sigma^2)$: grana fine, additivo (ISO alto). **Sale&pepe**: pixel impulsivi a 0 o 255. **Filtro mediano** (non lineare): ordina i pixel nella finestra, sceglie la mediana — rimuove sale&pepe SENZA sfocare perché non media ma seleziona il valore centrale.

> **All'orale**: Equalizzazione = CDF come lookup table. CLAHE divide in blocchi + taglia picchi di contrasto (non amplifica rumore). Se mostrano istogramma stretto: «poco contrasto, equalizzazione allarga». Domanda: «equalizzazione vs CLAHE?» = CLAHE locale + evita amplificazione rumore.


Operazioni che agiscono su ogni pixel **indipendentemente** dagli altri (punto-a-punto).

#### 2.1 Istogramma

L'istogramma $h(k)$ conta il numero di pixel con intensita $k$:

$$h(k) = |\{(i,j) : I(i,j) = k\}|$$

L'istogramma normalizzato e una stima della PDF dell'intensita: $p(k) = h(k) / (M \cdot N)$.

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/pixproc1.png" alt="Istogramma" width="400">
#### 2.2 Equalizzazione dell'Istogramma

Trasformazione che rende l'istogramma il piu possibile uniforme. La funzione di trasformazione e la **CDF**:

$$T(k) = \sum_{i=0}^{k} p(i)$$

- **Effetto**: aumenta il contrasto globale dell'immagine
- **Limite**: non preserva il contrasto locale; puo amplificare il rumore

#### 2.3 Rumore

Il rumore e una degradazione stocastica dell'immagine.

Modelli principali:

| Tipo | Modello | Caratteristica |
|------|---------|----------------|
| **Gaussiano additivo** | $I_{noisy} = I + n$, $n \sim \mathcal{N}(0, \sigma^2)$ | Indipendente dal segnale, sempre presente |
| **Salt & Pepper** | Pixel casuali a 0 o 255 | Impulsivo, colpisce una frazione di pixel |
| **Poisson** | $I_{noisy} \sim \text{Poisson}(I)$ | Correlato all'intensita (shot noise) |
| **Speckle** | $I_{noisy} = I \cdot n$, $n \sim \mathcal{N}(1, \sigma^2)$ | Moltiplicativo, tipico di radar/ultrasuoni |

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/pixproc3.png" alt="Rumore" width="500">
---

### 3. Filtri e Convoluzione `[p.4-5]`

> **Termini**: LSIS = sistema lineare shift-invariante (stessa risposta ovunque) · PSF = risposta impulsiva · convoluzione = $y=x*h$ · padding = estensione bordi · separable filter = 2D scomposto in 1D (più veloce) · gaussiano = sfocatura a campana · mediano = valore centrale dopo ordinamento · bilaterale = gaussiano + differenza intensità
> **A cosa serve**: ripulire le immagini dal rumore. E il gesto base di qualsiasi elaborazione: come aggiustare una foto sfocata o piena di puntini bianchi prima di usarla.

> **Spiegazioni**: **Convoluzione** $y = x*h$: kernel $k\times k$ scorre sull'immagine (LSIS, shift-invariante) — moltiplica pesi × pixel e somma, come un timbro che stampa lo stesso kernel su tutto il foglio. **Gaussiano**: pesi a campana, **separabile** $G(x,y)=G(x)G(y)$ — costo $O(k)$ invece di $O(k^2)$. **Mediano** (non lineare): ordina i pixel nella finestra, sceglie la mediana — rimuove sale&pepe preservando i bordi (perché non media). **Bilaterale**: $w = w_s\cdot w_r$ — peso **spaziale** (gaussiano sulla distanza) × peso di **range** (gaussiano sulla differenza d'intensità) — preserva i bordi perché non media pixel di colore diverso (come colorare dentro le linee).

> **All'orale**: LSIS = $y=x*h$, stesso filtro ovunque. Mediano = NON sfoca (a differenza del gaussiano). Bilaterale = gaussiano + differenza intensità: preserva bordi. Separabile: 2D = 1D×1D (costo $O(k^2) \to O(k)$). Domanda: «filtro che toglie rumore senza sfocare?» = mediano o bilaterale.


#### 3.1 Sistemi LSIS (Lineari Shift-Invarianti)

Un sistema LSIS e completamente caratterizzato dalla sua **risposta impulsiva** $h$. L'uscita e la **convoluzione** dell'ingresso con $h$:

$$y = x * h$$

Proprieta: commutativa, associativa, distributiva.

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/LSIS1.png" alt="LSIS1" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/LSIS2.png" alt="LSIS2" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/LSIS3.png" alt="LSIS3" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/LSIS4.png" alt="LSIS4" width="400">
#### 3.2 Convoluzione 1D e 2D

- Il kernel $H$ viene "ribaltato" e fatto scorrere sull'immagine
- Ai bordi si usa **padding** (zero, replicate, reflect)

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv1.png" alt="Conv1" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv2.png" alt="Conv2" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv3.png" alt="Conv3" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv4.png" alt="Conv4" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv5.png" alt="Conv5" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv6.png" alt="Conv6" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv7.png" alt="Conv7" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv8.png" alt="Conv8" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv9.png" alt="Conv9" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/conv10.png" alt="Conv10" width="400">
#### 3.3 Filtri Lineari

**Box Filter** (media uniforme):
$$H = \frac{1}{k^2} \begin{bmatrix} 1 & \cdots & 1 \\ \vdots & \ddots & \vdots \\ 1 & \cdots & 1 \end{bmatrix}$$
- Effetto: sfocatura (low-pass), rimuove alte frequenze
- Costo: $O(k^2)$ per pixel
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt1.png" alt="Box" width="400">
**Filtro Gaussiano**:
$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$
- Sfocatura piu naturale del box filter (pesi maggiori al centro, minori ai bordi)
- **Separabile**: $G(x,y) = G(x) \cdot G(y)$ -> costo $O(k)$ per asse invece di $O(k^2)$
- $\sigma$ controlla l'ampiezza: $\sigma$ grande = piu sfocatura

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt5.png" alt="Gaussiano" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt2.png" alt="Passa-basso" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt3.png" alt="Passa-alto" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt4.png" alt="Confronto" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt7.png" alt="Laplaciano" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt8.png" alt="Edge detection" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt9.png" alt="Sobel" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt10.png" alt="Prewitt" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt11.png" alt="Direzionali" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt12.png" alt="Bordi confronto" width="400">
#### 3.4 Filtri Non Lineari

**Filtro Mediana**:
- Sostituisce ogni pixel con la mediana del vicinato
- **Non lineare** (non esprimibile come convoluzione)
- Eccellente per rimuovere **salt & pepper** preservando i bordi

**Filtro Bilaterale**:
$$I_{filtered}[i,j] = \frac{\sum_{(m,n) \in \Omega} w_s(m,n) \cdot w_r(I[m,n], I[i,j]) \cdot I[m,n]}{\sum_{(m,n) \in \Omega} w_s(m,n) \cdot w_r(I[m,n], I[i,j])}$$

Due pesi:
- $w_s$: peso spaziale (vicinanza, gaussiano)  pixel vicini pesano di piu
- $w_r$: peso di range (similarita di intensita, gaussiano)  pixel con intensita simile pesano di piu

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt20.png" alt="Bilaterale" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt13.png" alt="Mediana" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt14.png" alt="Mediana vs media" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt15.png" alt="Non lineare" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt16.png" alt="Bilaterale dettaglio" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt17.png" alt="Bilaterale esempio" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt18.png" alt="Confronto filtri" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt19.png" alt="Scelta filtro" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt21.png" alt="Freq domain" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt22.png" alt="Filtri freq" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt23.png" alt="Band-pass" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt24.png" alt="Bordi freq" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt25.png" alt="Rumore freq" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/filt26.png" alt="Deblurring" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fillt6.png" alt="Separabile" width="400">

---

### 4. Trasformata di Fourier `[p.6-8]`

> **Termini**: DFT = da spazio a frequenze · FFT = algoritmo veloce $O(N \log N)$ · magnitudine = ampiezza frequenze · fase = posizione/struttura (più importante) · teo. convoluzione = spazio $*$ freq $\cdot$ · aliasing = frequenze oltre Nyquist si ripiegano · ringing = artefatto da troncamento brusco
> **A cosa serve**: vedere l'immagine "dall'altro lato" — non pixel ma frequenze. Serve a comprimere (JPEG), rimuovere rumori periodici o capire perche un'immagine ha artefatti.

> **Spiegazioni**: **DFT**: scompone l'immagine in basi di Fourier — come un accordo in note singole. Basse frequenze = zone uniformi (cielo). Alte frequenze = dettagli fini (bordi, erba). **Teorema della convoluzione**: $x*h \xrightarrow{\mathcal{F}} X\cdot H$ — convolvere nello spazio = MOLTIPLICARE in frequenza ($O(N\log N)$ con FFT). **Fase** $\phi$: codifica la POSIZIONE dei bordi — è l'informazione PIÙ importante (esperimento: fase di A + magnitudine di B = si riconosce A). **Magnitudine** $|X|$: energia/contrasto di ogni frequenza. **Aliasing**: frequenze $> f_{Nyquist}$ si ripiegano su frequenze più basse (ruote carro). **Ringing**: artefatto da troncamento brusco in frequenza.

> **All'orale**: Teorema convoluzione: filtrare nello spazio = moltiplicare in frequenza (velocissimo). FASE più importante della magnitudine (struttura dell'immagine). Aliasing: frequenze > Nyquist si ripiegano. DFT per analisi, FFT per calcolo. Domanda: «fase o magnitudine?» = fase, contiene la struttura.


#### 4.1 Numeri Complessi

Un numero complesso $z = a + ib = r e^{i\theta}$:
- **Magnitudine**: $|z| = \sqrt{a^2 + b^2}$ ? "quanto"
- **Fase**: $\theta = \arctan(b/a)$ ? "dove"

#### 4.2 Serie di Fourier

Ogni segnale periodico $f(t)$ con periodo $T$ si puo scrivere come somma di sinusoidi:

$$f(t) = \sum_{n=-\infty}^{\infty} c_n e^{i 2\pi n t / T}$$

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/square.png" alt="Serie Fourier" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier.png" alt="Fourier" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier1.png" alt="Fourier1" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier2.png" alt="Armoniche" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier3.png" alt="Sintesi" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier4.png" alt="Esempio" width="400">
#### 4.3 DFT (Discrete Fourier Transform)

**1D**: $X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi kn/N}$

**2D**: $X[u,v] = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} I[m,n] e^{-i 2\pi (um/M + vn/N)}$

- **FFT**: calcola la DFT in $O(N \log N)$ invece di $O(N^2)$
- Simmetria coniugata: $X[u,v] = X^*[-u,-v]$

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/dft.png" alt="DFT" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier5.png" alt="DFT2D" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier6.png" alt="DFTes" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier7.png" alt="FFT" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier8.png" alt="fftshift" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/Nuova%20cartella%20con%20elementi/fourier9.png" alt="DFTprop" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/Nuova%20cartella%20con%20elementi/fourier10.png" alt="DFTsim" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier11.png" alt="Freq" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier12.png" alt="Filtraggio" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier13.png" alt="Rumore" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier14.png" alt="Filtri" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier15.png" alt="Freq filt" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier16.png" alt="Low-pass" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier17.png" alt="High-pass" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier18.png" alt="Band-pass" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier19.png" alt="Ideale" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/fourier20.png" alt="Gaussiano" width="400">
#### 4.4 Teorema della Convoluzione

$$x * h \xleftrightarrow{\mathcal{F}} X \cdot H$$

La convoluzione nel dominio spaziale diventa **prodotto** nel dominio della frequenza (e viceversa).

**Implicazione pratica**: filtrare in frequenza e $O(N \log N)$ contro $O(N \cdot k^2)$ nello spazio.

#### 4.5 Filtraggio in Frequenza

- **Low-pass** (taglia alte frequenze): sfocatura, denoising
- **High-pass** (taglia basse frequenze): edge detection, sharpening
- **Band-pass**: seleziona una banda di frequenze

#### 4.6 Importanza di Fase e Magnitudine

- **Fase** ? contiene la **struttura** (bordi, forme, posizione)
- **Magnitudine** ? contiene il **"quanto"** (contrasto globale, energia)

#### 4.7 Compressione DFT

Si possono scartare le componenti di Fourier con magnitudine piccola (sotto una soglia). Poche componenti a bassa frequenza catturano l'essenza dell'immagine.

#### 4.8 Hybrid Images

Combinare basse frequenze di un'immagine con alte frequenze di un'altra:
- Da vicino si vede l'immagine ad alta frequenza
- Da lontano si vede l'immagine a bassa frequenza

---

### 5. Computational Imaging e Problemi Inversi `[p.8-10]`

> **Termini**: forward model = $y=Ax+e$ · ill-posed = viola esistenza/unicità/stabilità · condizionamento = $\sigma_{max}/\sigma_{min}$ · SVD = $A=U\Sigma V^T$ · $\sigma_i$ = valori singolari · rumore amplificato = $\sigma_i$ piccoli amplificano rumore
> **A cosa serve**: ricostruire un'immagine da misure indirette. TAC, risonanza magnetica, deblurring: senza teoria dei problemi inversi queste applicazioni non esisterebbero.

> **Spiegazioni**: **Forward model**: $y = Ax + e$. **Ill-posed** (Hadamard): viola esistenza/unicità/stabilità. Esempio: TAC con poche angolazioni — sistema sotto-determinato ($m < n$), infinite ossa possono dare le stesse ombre. **SVD**: $A = U\Sigma V^T$. **Soluzione naive**: $x = \sum (u_i^T y / \sigma_i) v_i$ — quando $\sigma_i$ (valore singolare) è piccolo, $1/\sigma_i$ AMPLIFICA il rumore (come microfono al massimo: senti anche il fruscio). **Numero di condizionamento**: $\sigma_{max}/\sigma_{min}$ — più alto = peggio condizionato. Servono **prior** (conoscenza a priori: «ossa bianche, aria nera») e regolarizzazione per scegliere la soluzione giusta.

> **All'orale**: Hadamard: servono esistenza + unicità + stabilità. Ill-posed se manca una. $\sigma_i$ piccoli dividono per numeri piccoli = amplificano rumore. Domanda tipica: «perché ill-posed?» = $\sigma_i$ piccoli fanno esplodere il rumore.


#### 5.1 Il Framework dei Problemi Inversi

$$\boldsymbol{y} = A\boldsymbol{x} + \boldsymbol{e}$$

- $\boldsymbol{x} \in \mathbb{R}^n$: immagine incognita (ground truth)
- $A \in \mathbb{R}^{m \times n}$: operatore di acquisizione (forward model)
- $\boldsymbol{y} \in \mathbb{R}^m$: dati misurati
- $\boldsymbol{e}$: rumore di misura

**Problema diretto**: dato $x$, calcolare $y = Ax$ ? facile, ben posto.
**Problema inverso**: dato $y$, trovare $x$ ? difficile, spesso mal posto.

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim1.jpg" alt="Forward model" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim2.jpg" alt="Inverso" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim3.jpg" alt="CT" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim4.jpg" alt="MRI" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim5.png" alt="Framework" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim6.png" alt="Diretto vs inverso" width="400">
<img src="https://raw.githubusercontent.com/devangelista2/computational-imaging/main/years/2025-26/imgs/GoPro.jpg" alt="Deblurring esempio" width="400">
#### 5.2 Problemi Ill-Posed (Mal Posti)

Un problema e **ben posto** (Hadamard) se soddisfa:
1. **Esistenza**: la soluzione esiste
2. **Unicita**: la soluzione e unica
3. **Stabilita**: la soluzione dipende con continuita dai dati

I problemi inversi in imaging sono tipicamente **ill-posed**.

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim7.png" alt="Ill-posed" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim8.png" alt="Hadamard" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim9.png" alt="Esempio ill-posed" width="400">
#### 5.3 SVD (Singular Value Decomposition)

$$A = U \Sigma V^T$$

- $U, V$: matrici ortogonali (basi ortonormali)
- $\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r)$: valori singolari, $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim10.png" alt="SVD" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim11.png" alt="SVD valori" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim12.png" alt="SVD basi" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim13.png" alt="Condizionamento" width="400">
#### 5.4 Soluzione Naive e Amplificazione del Rumore

La soluzione naive (inversa diretta):

$$\boldsymbol{x}_{naive} = A^{-1}\boldsymbol{y} = A^{-1}(A\boldsymbol{x} + \boldsymbol{e}) = \boldsymbol{x} + A^{-1}\boldsymbol{e}$$

In termini SVD:

$$\boldsymbol{x}_{naive} = \sum_{i=1}^{n} \frac{\boldsymbol{u}_i^T \boldsymbol{y}}{\sigma_i} \boldsymbol{v}_i$$

**Il problema**: quando $\sigma_i$ e piccolo, $1/\sigma_i$ amplifica enormemente il rumore.

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim14.png" alt="Naive problem" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim15.png" alt="Naive solution" width="500">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim16.png" alt="Confronto" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim17.png" alt="Amplificazione" width="400">
---

### 6. Regolarizzazione `[p.10-11]`

> **Termini**: Tikhonov/L2 = $\|x\|^2$ soluzione liscia · TV = $\|\nabla x\|_1$ preserva bordi · L1 = coefficienti nulli (sparsità) · L-curve = curva per scegliere $\lambda$ · $\lambda$ = trade-off fedeltà vs regolarizzazione
> **A cosa serve**: rendere stabile la ricostruzione di un problema inverso. Senza regolarizzazione le immagini sarebbero inutilizzabili. Scegliere L2 o TV cambia il risultato clinico.

> **Spiegazioni**: **Regolarizzazione**: $\hat{x} = \arg\min \|Ax-y\|^2 + \lambda R(x)$. **Data fidelity** ($\|Ax-y\|^2$): spiega i dati misurati. **Regolarizzatore** $R(x)$: premi la semplicità. **L2 (Tikhonov)** $R(x)=\|x\|^2$: tira tutto verso zero — soluzione liscia, nessun bordo netto (palla da bowling). **TV** $R(x)=\|\nabla x\|_1$: somma del gradiente in L1 — permette salti netti (bordi) ma penalizza oscillazioni (ORIGAMI: pieghe nette, facce lisce). **L1** $R(x)=\|x\|_1$: preferisce zeri — soluzione **sparsa** (compressed sensing). **$\lambda$**: trade-off. $\lambda \to 0$ = overfitting (rumoroso), $\lambda \to \infty$ = underfitting (troppo liscio). **L-curve**: metodo empirico per scegliere $\lambda$ ottimale.

> **All'orale**: $\hat{x} = \arg\min \|Ax-y\|^2 + \lambda R(x)$. L2 = liscio. TV = preserva bordi ($\|\nabla x\|_1$). $\lambda$ grande = troppo liscio, piccolo = rumoroso. L-curve per scegliere $\lambda$. Domanda: «L2 o TV?» = TV se vuoi bordi netti.


#### 6.1 Il Framework Model-Based

$$\hat{\boldsymbol{x}} = \arg\min_{\boldsymbol{x}} \underbrace{\|A\boldsymbol{x} - \boldsymbol{y}\|_2^2}_{\text{data fidelity}} + \lambda \underbrace{R(\boldsymbol{x})}_{\text{regolarizzazione}}$$

#### 6.2 Scelta di $\lambda$

- $\lambda \to 0$: overfitting (soluzione rumorosa)
- $\lambda \to \infty$: underfitting (soluzione troppo liscia)
- $\lambda$ ottimale: bilancia fedelta e plausibilita

#### 6.3 Tipi di Regolarizzazione

| Regolarizzatore | Formula | Effetto |
|-----------------|---------|---------|
| **Tikhonov** ($L_2$) | $R(x) = \|x\|_2^2$ | Soluzione liscia, penalizza grandi valori |
| **TV** (Total Variation) | $R(x) = \|\nabla x\|_1$ | Preserva bordi, piecewise constant |
| **Sparsita** ($L_1$) | $R(x) = \|x\|_1$ | Soluzione sparsa (compressed sensing) |

<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim18.png" alt="Reg confronto" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim19.png" alt="Tikhonov vs TV" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/compim20.png" alt="Lambda" width="400">
<img src="https://raw.githubusercontent.com/elenaloli/computational-imaging/main/book/immagini_sorgente/reg1.png" alt="Regolarizzazione" width="400">
#### 6.4 Collegamento con il Deep Learning

Il framework model-based classico usa regolarizzatori **hand-crafted**. Il deep learning sostituisce $R(\boldsymbol{x})$ con un **prior appreso** dai dati.

---

## PARTE II - DEEP LEARNING PER IMAGING (Modulo 2)

---

### 7. Processing Images per Reti Neurali `[p.12-13]`

> **Termini**: tensore = $(B,C,H,W)$ PyTorch · normalizzazione = mean/std del dataset (NON 0-1) · Dataset = classe che carica dati · DataLoader = iteratore batch+shuffle+workers · augmentation = trasformazioni casuali
> **A cosa serve**: preparare le immagini per la rete neurale. Dimensioni giuste, valori normalizzati, batch organizzati. Senza preprocessing la rete non impara.

> **Spiegazioni**: **Tensori**: $(B, C, H, W)$ = (batch, canali, altezza, larghezza). **Normalizzazione** $x = (x - \mu)/\sigma$: sottrai la MEDIA del dataset e dividi per la DEVIAZIONE STANDARD (NON portare a [0,1]!) — serve a scalare tutte le feature uniformemente per stabilizzare il training e prevenire gradienti esplosivi. **Dataset + DataLoader**: il DataLoader carica in batch (tante immagini alla volta), mescola (shuffle per evitare correlazione nell'ordine), usa **workers** multipli per parallelizzare il caricamento da disco. **Augmentation**: flip, rotazione, crop, elastic deform — aumenta la varietà dei dati senza raccoglierne di nuovi (data augmentation). Pipeline: end-to-end $y^\delta \xrightarrow{f_\Theta} x_{pred}$ o ibrida FBP + UNet.

> **All'orale**: Tensore = (B, C, H, W). NON normalizzare a [0,1]: usa mean/std del dataset (es. ImageNet: mean=[0.485,0.456,0.406]). DataLoader = batch + shuffle + workers paralleli. Augmentation = più dati senza raccoglierli. Domanda: «normalizzazione?» = sottrarre media, dividere std, NON 0-1.


#### 7.1 Tensori

Le immagini per le NN sono **tensori 4D**: $(B, C, H, W)$

- $B$: batch size
- $C$: canali (1=grigio, 3=RGB)
- $H, W$: altezza e larghezza

```python
# Esempio: batch di 32 immagini RGB 256x256
x.shape  # torch.Size([32, 3, 256, 256])
```

#### 7.2 Normalizzazione

- **Range [0, 1]**: `x = x.float() / 255.0` o `transforms.ToTensor()`
- **Range [-1, 1]**: `x = (x - 0.5) / 0.5` (usato con GAN, diffusion)
- **Standardizzazione**: `x = (x - mean) / std` (per canale)

**dtype**: `float32` per il training (bilancio precisione/memoria).

<img src="https://raw.githubusercontent.com/devangelista2/computational-imaging/main/years/2025-26/imgs/dtype.png" alt="Dtype" width="400">
<img src="https://raw.githubusercontent.com/devangelista2/computational-imaging/main/years/2025-26/imgs/preprocessing.png" alt="Preprocessing" width="400">
#### 7.3 Dataset e DataLoader

```python
class MayoDataset(Dataset):
    def __init__(self, data_path):
        self.files = sorted(glob(f'{data_path}/*/*.png'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx]).convert('L'))

loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Mayo Dataset**: 3305 immagini train, 327 test - slice CT addominali.

<img src="https://raw.githubusercontent.com/devangelista2/computational-imaging/main/years/2025-26/imgs/Mayo.png" alt="Mayo" width="400">
#### 7.4 Pipeline di Ricostruzione

**End-to-end**: $y^\delta \xrightarrow{f_\Theta} x_{pred}$

<img src="https://devangelista2.github.io/computational-imaging/2025-26/_images/end-to-end.png" alt="End-to-end" width="400">
**Hybrid**: $y^\delta \xrightarrow{\text{FBP}} \tilde{x} \xrightarrow{f_\Theta} x_{pred}$

<img src="https://devangelista2.github.io/computational-imaging/2025-26/_images/hybrid-approach.png" alt="Hybrid" width="400">
---

### 8. PyTorch Essentials `[p.14-15]`

> **Termini**: autograd = calcolo automatico gradienti · computational graph = grafo forward · backward() = chain rule all'indietro · SGD = $w = w - lr \cdot \nabla w$ · Adam = lr adattivo per parametro · scheduler = riduce lr durante training
> **A cosa serve**: implementare reti neurali con gradienti automatici, GPU e ottimizzatori. E il toolkit essenziale per qualsiasi progetto di deep learning in imaging.

> **Spiegazioni**: **Autograd**: costruisce un **computational graph** delle operazioni. `loss.backward()` calcola i gradienti di TUTTI i parametri con **chain rule** in un colpo solo (come domino: la loss è l'ultima pedina). Durante **valutazione** usi `torch.no_grad()` per risparmiare memoria (non serve il grafo). **SGD**: $w = w - lr \cdot \nabla w$ — semplice, può oscillare. **SGD + momentum**: media mobile del gradiente per ridurre oscillazioni. **Adam**: momentum + adaptive lr (media mobile del gradiente quadro) — learning rate DIVERSO per ogni parametro. **Scheduler**: riduce il learning rate quando la loss smette di migliorare (CosineAnnealingLR, StepLR).

> **All'orale**: Autograd = grafo delle operazioni. backward() = gradienti con chain rule. `torch.no_grad()` quando valuti (non serve grafo). Adam = lr adattivo per parametro (media mobile gradiente + gradiente quadro). Domanda: «backprop?» = chain rule sul grafo computazionale.


#### 8.1 Tensori

```python
x = torch.randn(3, 4)
x.shape, x.dtype, x.device
x.requires_grad = True
```

#### 8.2 Moduli Parametrizzati

```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)
    def forward(self, x):
        return self.linear(x)
```

#### 8.3 Autograd

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad)  # tensor([7.0]) = 2*2 + 3
```

<img src="https://devangelista2.github.io/computational-imaging/2025-26/_images/backpropagation.png" alt="Backprop" width="400">
#### 8.4 Ottimizzatori

| Ottimizzatore | Formula | Note |
|---------------|---------|------|
| **SGD** | $\theta \leftarrow \theta - \eta \nabla L$ | Semplice, puo oscillare |
| **SGD + momentum** | $v \leftarrow \mu v + \nabla L$; $\theta \leftarrow \theta - \eta v$ | Riduce oscillazioni |
| **Adam** | Combina momentum + RMSProp | Default nella pratica |

**Learning Rate Scheduler**: `CosineAnnealingLR`

<img src="https://devangelista2.github.io/computational-imaging/2025-26/_images/lr_scheduler.png" alt="LR Scheduler" width="300">
#### 8.5 IPPy (libreria del corso)

```
IPPy/
+-- operators/   # Forward operators (Blurring, Radon, etc.)
+-- solvers/     # Solvers classici (Tikhonov, ISTA, etc.)
+-- nn/          # Architetture neurali (UNet, DiffusionUNet, etc.)
+-- utilities/   # Device detection, noise generation, etc.
```

<img src="https://devangelista2.github.io/computational-imaging/2025-26/_images/IPPy-structure.png" alt="IPPy" width="300">
---

### 9. Da Machine Learning a Neural Networks `[p.15-16]`

> **Termini**: ReLU = $\max(0,x)$ evita vanishing gradient · sigmoid = $1/(1+e^{-x})$ satura · softmax = output in probabilità · MLP = fully-connected · universal approx = 1 hidden layer approssima qualsiasi funzione · feature = rappresentazioni intermedie · overfitting = memorizza training non generalizza
> **A cosa serve**: capire come un neurone artificiale impara: combinazione lineare + attivazione + ottimizzazione. Serve a risolvere problemi complessi come classificazione e segmentation.

> **Spiegazioni**: **ReLU** $\max(0,x)$: non satura — evita **vanishing gradient** (problema della sigmoide/tanh che per grandi valori si appiattiscono e bloccano l'apprendimento). MA **Dying ReLU**: neuroni con input sempre negativi muoiono per sempre. Soluzione: **LeakyReLU** ($\max(\alpha x, x)$) lascia passare un piccolo flusso. **Softmax**: output in probabilità (classificazione). **MLP** (fully-connected): ogni neurone collegato a TUTTI gli altri. **Universal Approximation Theorem**: 1 hidden layer con sufficienti neuroni approssima qualsiasi funzione continua. MA per img 100×100: 10.000 input × 1000 neuroni = 10M parametri — impraticabile. **Feature gerarchiche**: layer bassi (bordi) → intermedi (texture) → alti (oggetti). CNN risolve con **weight sharing** (kernel condiviso = timbro).

> **All'orale**: ReLU evita vanishing gradient (sigmoide satura), MA muore se input sempre <0 (LeakyReLU risolve). MLP = universal approximator. Domanda: «perché CNN meglio di MLP per immagini?» = weight sharing, meno parametri, sfrutta struttura spaziale.


#### 9.1 Limiti dei Modelli Lineari

Un modello lineare $f(x) = Wx + b$ puo rappresentare solo relazioni lineari.

#### 9.2 Funzioni di Attivazione

| Attivazione | Formula | Uso |
|-------------|---------|-----|
| **ReLU** | $\max(0, x)$ | Default per hidden layers |
| **LeakyReLU** | $\max(\alpha x, x)$ | Evita "dying ReLU" |
| **Sigmoid** | $1/(1+e^{-x})$ | Output binario [0,1] |
| **Tanh** | $(e^x - e^{-x})/(e^x + e^{-x})$ | Output [-1,1] |
| **GELU** | $x \cdot \Phi(x)$ | Transformers |
| **SiLU/Swish** | $x \cdot \sigma(x)$ | Modelli generativi |
| **Softmax** | $e^{x_i}/\sum e^{x_j}$ | Classificazione |

<img src="https://raw.githubusercontent.com/devangelista2/computational-imaging/main/years/2025-26/imgs/NN.png" alt="NN" width="400">
#### 9.3 MLP (Multi-Layer Perceptron)

**Teorema di approssimazione universale**: un MLP con un solo hidden layer (sufficientemente largo) puo approssimare qualsiasi funzione continua su un compatto.

<img src="https://devangelista2.github.io/computational-imaging/2025-26/_images/MLP.png" alt="MLP" width="400">
#### 9.4 Feature Gerarchiche

- Layer bassi: bordi, texture, gradienti
- Layer intermedi: forme, pattern locali
- Layer alti: oggetti, strutture semantiche

#### 9.5 Training

- **SGD/Minibatch**: aggiornamento su sottoinsiemi
- **Backpropagation**: calcolo efficiente dei gradienti
- **Generalizzazione**: performance su dati non visti

---

### 10. CNN (Convolutional Neural Networks) `[p.17-18]`

> **Termini**: convoluzione 2D = kernel $k\times k$ scorre sull'immagine · stride = passo (2 = output dimezzato) · padding = bordi extra per mantenere dimensione · pooling = max pooling riduce risoluzione · channel progression = $64 \to 128 \to 256$ · translation equivariance = shift input = shift output
> **A cosa serve**: elaborare immagini in modo efficiente. Invece di connettere ogni pixel a ogni neurone (MLP), le CNN riusano lo stesso pattern su tutta l'immagine. Pochi parametri, grande potenza.

> **Spiegazioni**: **Convoluzione 2D** (cross-correlation, senza ribaltamento del kernel): kernel scorre su input, moltiplica pesi × pixel e somma. **Weight sharing**: STESSI 9 pesi (kernel 3×3) su tutta l'immagine — ~3500× meno parametri di un MLP. **Stride**: passo del kernel (stride 2 = downsampling 2×). **Padding**: zeri/replicate/reflect ai bordi per mantenere dimensione ($p = (k-1)/2$ per stessa dimensione). **Pooling** (max/average): dimezza risoluzione, seleziona feature più forti (max) o media (avg). **Channel progression**: 64 → 128 → 256 — più canali compensano la riduzione spaziale. **Translation equivariance**: shift dell'input = stesso shift della feature map (grazie a weight sharing). **Stack 3×3 > 7×7**: meno parametri ($27$ vs $49$) + più non-linearità (più ReLU).

> **All'orale**: Weight sharing = ~3500× meno parametri di MLP. Stack 3×3 > 7×7: meno parametri + più ReLU. Channel progression: 64 → 128 → 256. Domanda: «perché più canali in profondità?» = compensa la riduzione spaziale.


#### 10.1 Convoluzione 2D nelle CNN

$$y[i,j] = \sum_{m}\sum_{n} x[i+m, j+n] \cdot w[m,n] + b$$

Nota: nelle CNN si usa la **cross-correlazione** (senza ribaltamento del kernel), non la convoluzione matematica.

**Padding**: aggiunta di pixel ai bordi per controllare la dimensione dell'output.
- `padding = (k-1)/2` (con stride=1) -> output stessa dimensione dell'input

<img src="immagini_locali/padded-convolution.png" alt="Convoluzione con padding" width="400">
#### 10.2 Architettura CNN

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 1?32 canali
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), # 32?64 canali
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),  # 64?1 canale (output)
        )
    def forward(self, x):
        return self.net(x)
```

**Pooling (opzionale, non nella SimpleCNN)**:
- **Max Pooling**: seleziona il valore massimo in una finestra $2 \times 2$ ? dimezza risoluzione, mantiene le feature piu attive
- **Average Pooling**: media i valori nella finestra -> dimezza risoluzione, smoothing
- **Perche si usa il pooling?**: Riduce la risoluzione spaziale (e il costo computazionale) e aumenta il receptive field dei layer successivi. Nelle CNN per ricostruzione (senza downsampling), il pooling non si usa perche si vuole mantenere la risoluzione piena.

<img src="immagini_locali/CNN.png" alt="CNN" width="600">
#### 10.3 Translation Equivariance

Le CNN sono **equivarianti** alle traslazioni: se l'input si sposta, l'output si sposta nello stesso modo. Questo perche i pesi sono **condivisi** su tutta l'immagine (weight sharing).

#### 10.4 Training End-to-End per Ricostruzione

Nel corso si usa un approccio **sintetico**: i dati di training sono generati online:

$$\boldsymbol{y} = K\boldsymbol{x} + \boldsymbol{e}$$

dove $K$ e un operatore di blurring e $\boldsymbol{e}$ e rumore gaussiano. Non serve un dataset paired.

**Attivazione finale**:
- CNN standard: ReLU o nessuna (output $\geq 0$)
- ResCNN: Tanh (output in $[-1, 1]$ per il residuo)

---

### 11. Residual Learning e UNet `[p.18-20]`

> **Termini**: residual learning = impara $f(x)-x$ (più facile di $f(x)$) · receptive field = regione input che influenza un neurone · skip connection = salta layer (autostrada per gradiente) · encoder-decoder = imbuto comprime + ricostruisce · bottleneck = punto di massima compressione
> **A cosa serve**: segmentare e ricostruire immagini mediche. La UNet e lo standard per CT e MRI. Le skip connections hanno reso possibile addestrare reti molto profonde.

> **Spiegazioni**: **ResNet**: impara il **residuo** $f_\Theta(y^\delta) \approx x - y^\delta$, poi $x_{pred} = y^\delta + f_\Theta(y^\delta)$. Più facile quando input e output sono simili (ritoccare un disegno vs rifarlo da capo). **Skip connection**: salta layer — **somma** (ResNet) o **concatenazione** (UNet) — autostrada per il gradiente, risolve **vanishing gradient** (reti con centinaia di layer possibili). **Receptive field**: regione di input che influenza un neurone — $r_L = r_{L-1} + (k_L-1)\prod s_i$. Più deep = RF più grande. **UNet encoder-decoder**: encoder comprime (cosa?), decoder ricostruisce (dove?), **skip connections concatenano** dettagli spaziali dall'encoder al decoder (la compressione perde posizione). **Bottleneck**: punto di massima compressione = contesto globale. Varianti: Residual UNet, Attention UNet, UNet++.

> **All'orale**: ResNet: $f(x)-x$ = più facile se input ≈ output. UNet: encoder-decoder + skip (dettagli). Bottleneck = compressione. Domanda: «UNet?» = imbuto (contesto) + skip (dettagli).


#### 11.1 ResCNN (Residual CNN)

Invece di imparare la mappa diretta $y^\delta \to x$, la ResCNN impara il **residuo**:

$$f_\Theta(y^\delta) \approx x - y^\delta$$

$$x_{pred} = y^\delta + f_\Theta(y^\delta)$$

**Vantaggio**: il residuo e tipicamente piccolo e strutturato -> piu facile da imparare.

**Skip connection**: $x_{pred} = y^\delta + f_\Theta(y^\delta)$ e essa stessa una skip connection globale.

<img src="immagini_locali/ResCNN.png" alt="ResCNN" width="400">
#### 11.2 Receptive Field

Il **receptive field** di un neurone e la regione dell'input che influenza la sua attivazione.

Per una CNN con $L$ layer convoluzionali:

$$r_L = r_{L-1} + (k_L - 1) \prod_{i=1}^{L-1} s_i$$

dove $k_L$ e la dimensione del kernel e $s_i$ lo stride al layer $i$.

<img src="immagini_locali/receptive_field.png" alt="Receptive Field" width="400">
**Problema**: CNN poco profonde hanno receptive field piccolo -> non catturano contesto globale. Soluzione nell'UNet: downsampling + bottleneck permette ai layer profondi di avere receptive field molto grande (tutta l'immagine dopo vari stride 2).

#### 11.3 UNet

Architettura **encoder-decoder** multi-scala con **skip connections**:

```
Encoder (downsampling):       Decoder (upsampling):
  x -> Conv -> 64               512 -> Conv -> 256 -- concat(skip) -> Conv
 64 -> Conv -> 128              256 -> Conv -> 128 -- concat(skip) -> Conv
128 -> Conv -> 256              128 -> Conv ->  64 -- concat(skip) -> Conv
256 -> Conv -> 512 (bottleneck)  64 -> Conv -> output
```

- **Encoder**: riduce risoluzione (downsampling con stride 2 o pooling), aumenta canali -> cattura contesto globale a varie scale
- **Decoder**: aumenta risoluzione (upsampling con trasposta o interpolazione), riduce canali -> ricostruisce dettagli spaziali
- **Skip connections (concat)**: collegano encoder e decoder alla stessa risoluzione tramite **concatenazione** (non somma come ResNet) -> preservano dettagli spaziali che altrimenti andrebbero persi nel bottleneck
- **Bottleneck**: rappresentazione compatta al centro con il maggior numero di canali e la minima risoluzione

<img src="immagini_locali/UNet.png" alt="UNet" width="600">
**Varianti**:
- **Residual UNet**: blocchi residui al posto delle conv semplici
- **Attention UNet**: attention gates sulle skip connections
- **UNet++**: skip connections nidificate e dense

---

### 12. Vision Transformers e Loss Design `[p.20-22]`

> **Termini**: patch embedding = dividi img in patch e proietta in vettori · self-attention = $\text{softmax}(QK^T/\sqrt{d})V$ · positional encoding = vettore posizione (attention è invariante) · PSNR = pixel a pixel >30 dB buono · SSIM = struttura locale · LPIPS = feature VGG percettiva
> **A cosa serve**: scegliere l'architettura (Transformer vs CNN) e la loss function giusta. MSE da risultati sfocati, LPIPS da risultati realistici. La differenza tra "pixel uguali" e "sembra vero".

> **Spiegazioni**: **ViT**: divide l'immagine in **patch** $P\times P$, le proietta in token embeddings. **Self-attention**: $\text{softmax}(QK^T/\sqrt{d})V$ — ogni patch guarda TUTTE le altre (contesto globale, costo $O(N^2)$). $Q$=domanda, $K$=tipo, $V$=contenuto. **MHSA**: $h$ teste in parallelo, ognuna impara relazioni diverse. **Positional encoding**: necessario perché self-attention è **permutation-invariant** («gatto a sinistra» = «gatto a destra» senza posizione). **MSE/L2**: $\frac{1}{n}\|x-\hat{x}\|^2$ — pixel a pixel, produce immagini SFOCATE (media delle soluzioni possibili). **LPIPS**: distanza in **feature space** di VGG pre-addestrata — controlla «sembra un osso?» non «i pixel sono uguali?». **SSIM**: composito (luminanza + contrasto + struttura), range [-1,1]. **PSNR**: $10\log_{10}(MAX^2/\text{MSE})$ dB — tipico 25-40dB in imaging medico.

> **All'orale**: Attention = softmax(QK^T/√d)V. Self-attention = contesto globale ma $O(N^2)$. MSE = media = sfoca. LPIPS = percettivo (feature VGG). SSIM = struttura locale. Domanda: «loss migliore?» = LPIPS per qualità visiva, PSNR per fedeltà pixel.


#### 12.1 Vision Transformer (ViT)

I Transformer, nati per NLP, sono stati adattati alle immagini.

**Patch Embedding**: l'immagine $H \times W$ viene divisa in patch $P \times P$:
- Numero di patch: $N = (H/P) \times (W/P)$
- Ogni patch -> vettore tramite proiezione lineare
- Sequenza di $N$ token + token [CLS] opzionale

<img src="immagini_locali/Patching.png" alt="Patching" width="300">
**Self-Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- $Q$ (Query), $K$ (Key), $V$ (Value): proiezioni lineari dell'input
- $\sqrt{d_k}$: scaling per evitare gradienti troppo piccoli
- Ogni token "guarda" tutti gli altri token ? **contesto globale**

<img src="immagini_locali/SA.png" alt="Self-Attention" width="400">
**Multi-Head Self-Attention (MHSA)**: $h$ attention heads in parallelo, ognuno impara relazioni diverse.

<img src="immagini_locali/MHSA.png" alt="MHSA" width="400">
**Positional Encoding**: aggiunto ai patch embeddings per dare informazione di posizione (senza di esso, il transformer e permutation-invariant).

<img src="immagini_locali/ViT.png" alt="ViT" width="600">
#### 12.2 Encoder-Decoder per Image-to-Image

Per tasks di ricostruzione, si usa un ViT encoder + decoder (spesso convoluzionale):
- Encoder ViT: cattura dipendenze globali
- Decoder: ricostruisce l'immagine a risoluzione piena

**CNN vs ViT**:
- CNN: efficiente per feature locali, induttive bias forte (localita, equivarianza)
- ViT: contesto globale, meno bias induttivo, richiede piu dati

<img src="immagini_locali/CNN-issue.png" alt="CNN issue" width="400">
#### 12.3 Metriche di Qualita

**PSNR** (Peak Signal-to-Noise Ratio):
$$\text{PSNR} = 10 \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right) \quad [\text{dB}]$$
- Piu alto = migliore. Tipico: 25-40 dB per imaging medico.

**SSIM** (Structural Similarity Index):
$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
- Range $[-1, 1]$, 1 = identico. Considera luminanza, contrasto e struttura.

**LPIPS** (Learned Perceptual Image Patch Similarity):
- Distanza nello spazio feature di una rete pre-addestrata (VGG/AlexNet)
- Piu basso = piu simile percettivamente

#### 12.4 Loss Functions

| Loss | Formula | Caratteristica |
|------|---------|----------------|
| **MSE** ($L_2$) | $\frac{1}{n}\|x - \hat{x}\|_2^2$ | Media, produce immagini sfocate |
| **L1** | $\frac{1}{n}\|x - \hat{x}\|_1$ | Mediana, piu sharp di MSE |
| **Perceptual** | $\|\phi(x) - \phi(\hat{x})\|_2^2$ | Feature space (VGG), dettagli realistici |

---

### 13. Problemi Cross-Domain `[p.22-23]`

> **Termini**: sinogramma = proiezioni a vari angoli · Radon $\mathcal{R}$ = integrale densità lungo linea · FBP = filtro Ramp + backprojection · ramp filter = $|\omega|$ amplifica alte freq · apodization = finestra per ridurre rumore · streaking = strisce con pochi angoli
> **A cosa serve**: fare CT dal mondo reale. Dalla macchina a raggi X all'immagine finale: sinogramma, FBP, e pipeline ibrida con deep learning. Esempio completo di problema inverso risolto.

> **Spiegazioni**: **Problema cross-domain**: dominio dei dati (sinogramma) ≠ dominio dell'immagine. **CT**: $y = \mathcal{R}x + e$ con **Trasformata di Radon** $\mathcal{R}$ = integrale di densità lungo linee — i raggi X ATTRAVERSANO il corpo a vari angoli e vengono attenuati in base alla densità dei tessuti, producendo un **SINOGRAMMA** (ombre a ogni angolo). **FBP** (Filtered Backprojection): 1) **Ramp filter** $|\omega|$ in frequenza (amplifica alte freq per nitidezza) 2) **Backprojection** = riporta le proiezioni al loro posto spaziale. **Streaking**: artefatti a strisce con pochi angoli. **Apodization**: finestra sul Ramp filter (es. Hann, cosine) — riduce rumore ma perde risoluzione (trade-off). **Pipeline ibrida**: $y \xrightarrow{\text{FBP}} \tilde{x} \xrightarrow{f_\Theta} x_{pred}$ — FBP gestisce la fisica, UNet rimuove artefatti.

> **All'orale**: CT = dalle proiezioni (sinogramma) ricostruisci l'immagine = problema inverso. Radon = integrale densità. FBP = Ramp + backprojection. Apodization = toglie rumore ma perde risoluzione. Domanda: «perché CT è problema inverso?» = misuri proiezioni, devi ricostruire l'immagine.


#### 13.1 Il Problema

In CT (Tomografia Computerizzata), i dati misurati sono **sinogrammi** (proiezioni a vari angoli), non immagini:

$$\boldsymbol{y} = \mathcal{R}\boldsymbol{x} + \boldsymbol{e}$$

dove $\mathcal{R}$ e la **Trasformata di Radon**.

Il dominio dei dati (sinogramma) e diverso dal dominio dell'immagine -> problema **cross-domain**.

<img src="immagini_locali/CT_acquisition.png" alt="CT Acquisition" width="400">
#### 13.2 FBP (Filtered Back-Projection)

Metodo classico per ricostruire da sinogrammi:
1. **Filtraggio**: le proiezioni sono filtrate con un **ramp filter** (filtro passa-alto $|f|$ in frequenza) per compensare lo "smoothness" della back-projection
2. **Back-projection**: ogni proiezione filtrata viene "spalmata" all'indietro lungo la stessa direzione di acquisizione, e tutte le direzioni vengono sommate

**FBP e veloce ma produce artefatti a basse dosi** (rumore, streaking).

#### 13.3 Pipeline Ibrida

$$\boldsymbol{y} \xrightarrow{\text{FBP}} \tilde{\boldsymbol{x}} \xrightarrow{f_\Theta} \boldsymbol{x}_{pred}$$

1. **Preprocessing**: FBP converte sinogramma -> immagine (rumorosa)
2. **Post-processing**: UNet rimuove artefatti e rumore

**Vantaggio**: la rete lavora nel dominio immagine (piu semplice), FBP gestisce la fisica.

---

### 14. Deep Generative Models: VAE e GAN `[p.23-26]`

> **Termini**: ELBO = ricostruzione - KL (obiettivo VAE) · KL = divergenza da $\mathcal{N}(0,I)$ · reparameterization = $z = \mu + \sigma\varepsilon$ (per backprop) · discriminator = distingue reali da generate · minimax = G minimizza D massimizza · mode collapse = G produce poca varietà · WGAN = Wasserstein distance
> **A cosa serve**: generare immagini sintetiche realistiche. Data augmentation, prior per problemi inversi, compressione. VAE e la base stabile, GAN da risultati piu nitidi.

> **Spiegazioni**: **VAE**: $x \xrightarrow{E} (\mu, \sigma) \to z = \mu + \sigma\varepsilon \xrightarrow{D} \hat{x}$. **Reparameterization trick**: separa $\mu,\sigma$ dal rumore $\varepsilon$ — rende il campionamento **differenziabile** (altrimenti backprop non funziona sul sampling stocastico). **ELBO** $= \mathbb{E}[\log p(x|z)] - \text{KL}(q(z|x)\|p(z))$: ricostruzione (MSE/L1 tra $x$ e $\hat{x}$) + **KL divergence** (forza $q(z|x)$ verso $\mathcal{N}(0,I)$). VAE: **stabile** ma **sfocato** (minimizza MSE = media delle soluzioni possibili). **GAN**: $\min_G\max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ — G genera, D classifica reale/finto. **Non-saturating loss** per G (più stabile del minimax originale). GAN: **nitido** ma **mode collapse** (G impara a produrre sempre la stessa immagine). **WGAN**: Wasserstein distance + gradient penalty — più stabile, meno mode collapse. **DGP**: $z$ ottimizzato con G congelato per inversion.

> **All'orale**: VAE: ELBO = ricostruzione - KL. Stabile ma sfocato. GAN: minimax, nitido ma mode collapse. WGAN = più stabile. Domanda: «VAE o GAN?» = VAE stabile ma sfocato, GAN nitido ma instabile.


#### 14.1 Modelli Generativi vs Discriminativi

- **Discriminativo**: impara $p(y|x)$ o $f: x \to y$ (mappatura diretta)
- **Generativo**: impara $p_{data}(x)$ (distribuzione delle immagini) -> puo generare nuovi campioni

**Deep Latent Variable Model (DLVM)**:

$$\boldsymbol{z} \sim p(\boldsymbol{z}), \quad \boldsymbol{x} \sim p_\Theta(\boldsymbol{x}|\boldsymbol{z})$$

- $\boldsymbol{z} \in \mathbb{R}^d$ ($d \ll n$): variabile latente (rappresentazione compressa)
- $p(\boldsymbol{z}) = \mathcal{N}(0, I)$: prior semplice
- $p_\Theta(\boldsymbol{x}|\boldsymbol{z})$: decoder (rete neurale)

<img src="immagini_locali/DLVM.png" alt="DLVM" width="300">
**Latent vector**: descrizione compressa dell'immagine. Non memorizza ogni pixel, ma codifica i fattori principali (anatomia, struttura, texture).

#### 14.2 VAE (Variational Autoencoder)

Un VAE e un AutoEncoder probabilistico:

$$\boldsymbol{x} \xrightarrow{E_\phi} (\boldsymbol{\mu}_\phi(\boldsymbol{x}), \boldsymbol{\sigma}_\phi(\boldsymbol{x})) \xrightarrow{\text{sampling}} \boldsymbol{z} \xrightarrow{D_\Theta} \hat{\boldsymbol{x}}$$

**ELBO** (Evidence Lower Bound):

$$\log p_\Theta(\boldsymbol{x}) \geq \underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})}[\log p_\Theta(\boldsymbol{x}|\boldsymbol{z})]}_{\text{ricostruzione}} - \underbrace{\text{KL}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}))}_{\text{regolarizzazione latente}}$$

- **Termine di ricostruzione**: il decoder deve ricostruire bene $\boldsymbol{x}$ a partire da $\boldsymbol{z}$
- **Termine KL**: il latente $\boldsymbol{z}$ deve essere vicino a $\mathcal{N}(0,I)$

**Reparameterization trick**:
$$\boldsymbol{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$$
Rende il sampling differenziabile -> backprop possibile.

<img src="immagini_locali/VAE.png" alt="VAE" width="400">
<img src="immagini_locali/VAE_architecture.png" alt="VAE Architecture" width="600">
**Training in 2 stadi** (nel corso):
1. **Pretraining AE**: training deterministico (senza KL) -> buona ricostruzione
2. **Fine-tuning VAE**: attivazione KL con capacity annealing -> spazio latente regolare

#### 14.3 GAN (Generative Adversarial Network)

Due reti in competizione:

- **Generator** $G_\Theta$: $\boldsymbol{z} \to \boldsymbol{x}_{fake}$ (decoder-like)
- **Discriminator** $D_\Psi$: $\boldsymbol{x} \to$ score reale/falso (encoder-like)

**Obiettivo minimax**:
$$\min_\Theta \max_\Psi \; \mathbb{E}_{\boldsymbol{x}}[\log D_\Psi(\boldsymbol{x})] + \mathbb{E}_{\boldsymbol{z}}[\log(1 - D_\Psi(G_\Theta(\boldsymbol{z})))]$$

**Non-saturating loss** (piu stabile per il generator):
$$\min_\Theta -\mathbb{E}_{\boldsymbol{z}}[\log D_\Psi(G_\Theta(\boldsymbol{z}))]$$

<img src="immagini_locali/GAN.png" alt="GAN" width="400">
<img src="immagini_locali/GAN_architecture.png" alt="GAN Architecture" width="600">
**Problemi delle GAN**:
- **Mode collapse**: il generator produce poca varieta
- **Training instabile**: equilibrio delicato tra G e D
- **WGAN/WGAN-GP**: usano la distanza di Wasserstein + gradient penalty per stabilita
- **Spectral normalization**: controlla la norma dei layer del critic

**Varianti importanti**: DCGAN, Conditional GAN, Pix2Pix, CycleGAN, WGAN, WGAN-GP, LSGAN.

#### 14.4 Deep Generative Prior (DGP) per Problemi Inversi

Un generatore pre-addestrato $G$ definisce un **prior** sulle immagini:

$$\hat{\boldsymbol{z}} = \arg\min_{\boldsymbol{z}} \|K G(\boldsymbol{z}) - \boldsymbol{y}^\delta\|_2^2 + \lambda \|\boldsymbol{z}\|_2^2$$

$$\hat{\boldsymbol{x}} = G(\hat{\boldsymbol{z}})$$

- Il generatore e **congelato**, si ottimizza solo $\boldsymbol{z}$
- La soluzione e vincolata al **range** del generatore: $\mathcal{M} = \{G(\boldsymbol{z}) : \boldsymbol{z} \in \mathbb{R}^d\}$
- Non servono dati paired per la ricostruzione

**Limiti**:
- **Representation error**: se $x_{true} \notin \mathcal{M}$, non puo essere recuperato
- Ottimizzazione non convessa in $\boldsymbol{z}$
- GAN prior: latente irregolare, ottimizzazione fragile
- VAE prior: spazio piu regolare, ma immagini piu sfocate

<img src="immagini_locali/DGP.png" alt="DGP" width="400">
---

### 15. Diffusion Models `[p.26-28]`

> **Termini**: DDPM = aggiunge rumore poi denoizza · forward = $\boldsymbol{x}_t = \sqrt{\alpha_t}\boldsymbol{x}_0 + \sqrt{1-\alpha_t}\boldsymbol{\varepsilon}$ · noise schedule = $\beta_t$ quanto rumore · score matching = modello impara $\nabla \log p(\boldsymbol{x})$ · DDIM = deterministico salta timestep (10× più veloce)
> **A cosa serve**: generare immagini della massima qualita. DALL-E, Stable Diffusion, Midjourney — tutti diffusion models. Imparano a trasformare rumore puro in immagini perfette.

> **Spiegazioni**: **Forward process** (DDPM): $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ — **noise schedule** $\beta_t$ (tipicamente cosine) controlla quanto rumore aggiungere a ogni passo. **Formula chiusa** FONDAMENTALE: $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\varepsilon$, $\alpha_t = \prod_{s=1}^t(1-\beta_s)$ — da $x_0$ salti DIRETTAMENTE a qualsiasi $x_t$ senza iterare. Per $t=T$: $x_T \sim \mathcal{N}(0,I)$. **Training**: $\min \|\varepsilon - \varepsilon_\Theta(x_t, t)\|^2$ — MSE sul rumore (denoising autoencoder). **Score matching**: $\varepsilon_\Theta \approx -\nabla_{x_t}\log p(x_t)$ (score function). **Sampling DDPM**: stocastico, rimuove rumore con rumore. **DDIM**: deterministico, salta timestep ($S \ll T$, es. 50 invece di 1000 = 20× più veloce). Architettura: **DiffusionUNet** con time embedding sinusoidale + self-attention + GroupNorm + SiLU + EMA.

> **All'orale**: Forward = formula chiusa $x_t$ da $x_0$. Training = MSE sul rumore. Sampling = da rumore a immagine. DDPM = stocastico, DDIM = deterministico + veloce. Domanda: «Diffusion in breve?» = aggiungi rumore forward, togli rumore backward.


#### 15.1 Idea Centrale

I diffusion models generano dati imparando a **invertire un processo di aggiunta graduale di rumore**.

<img src="immagini_locali/DM.png" alt="Diffusion Model" width="600">
#### 15.2 Forward Process (DDPM)

Sequenza di variabili latenti $\boldsymbol{x}_0, \boldsymbol{x}_1, \ldots, \boldsymbol{x}_T$:

$$q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,\boldsymbol{x}_{t-1},\; \beta_t I)$$

dove $\beta_t \in (0,1)$ e il **noise schedule**.

**Formula chiusa** (campionamento diretto a qualsiasi $t$):

$$\boldsymbol{x}_t = \sqrt{\alpha_t}\,\boldsymbol{x}_0 + \sqrt{1-\alpha_t}\,\boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(0,I)$$

dove $\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$.

- $t$ piccolo: $\alpha_t \approx 1$, $\sqrt{1-\alpha_t} \approx 0$ ? immagine quasi pulita
- $t$ grande: $\alpha_t \approx 0$, $\sqrt{1-\alpha_t} \approx 1$ ? quasi puro rumore gaussiano
- Per $t = T$ (tipicamente $T=1000$): $\alpha_T \approx 0$, $\boldsymbol{x}_T \sim \mathcal{N}(0,I)$

#### 15.3 Training (Noise Prediction)

Una singola rete $\boldsymbol{\varepsilon}_\Theta(\boldsymbol{x}_t, t)$ impara a predire il rumore:

$$\min_\Theta \; \mathbb{E}_{\boldsymbol{x}_0, \boldsymbol{\varepsilon}_t, t}\left[\|\boldsymbol{\varepsilon}_t - \boldsymbol{\varepsilon}_\Theta(\boldsymbol{x}_t, t)\|_2^2\right]$$

**Stima dell'immagine pulita**:
$$\hat{\boldsymbol{x}}_0(\boldsymbol{x}_t, t) = \frac{\boldsymbol{x}_t - \sqrt{1-\alpha_t}\,\boldsymbol{\varepsilon}_\Theta(\boldsymbol{x}_t, t)}{\sqrt{\alpha_t}}$$

#### 15.4 Architettura del Denoiser

**DiffusionUNet**: UNet time-conditioned con:
- Residual blocks + Group Normalization + SiLU
- Self-attention a risoluzioni intermedie
- **Sinusoidal embedding** per il timestep $t$ (come positional encoding nei transformer)

<img src="immagini_locali/sinusoidal_embedding.png" alt="Sinusoidal Embedding" width="300">
<img src="immagini_locali/DiffusionUNet.png" alt="DiffusionUNet" width="600">
Il time embedding viene proiettato e **aggiunto** dentro i residual blocks -> ogni blocco sa a quale livello di rumore sta operando.

**EMA** (Exponential Moving Average): si mantiene una copia "mediata" dei pesi per sampling piu stabile.

#### 15.5 DDPM Sampling (Reverse Process)

Partendo da rumore puro $\boldsymbol{x}_T \sim \mathcal{N}(0,I)$, si applicano $T$ step di denoising:

$$\boldsymbol{x}_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\alpha_t}}\boldsymbol{\varepsilon}_\Theta(\boldsymbol{x}_t, t)\right) + \sigma_t \boldsymbol{z}$$

dove $\boldsymbol{z} \sim \mathcal{N}(0,I)$ e $\sigma_t$ e la varianza del passo.

**Lento**: richiede $T$ valutazioni della rete (tipicamente $T = 400-1000$). Ogni valutazione e una forward pass della DiffusionUNet.

#### 15.6 DDIM Sampling (Deterministico e Veloce)

**DDIM** (Denoising Diffusion Implicit Models) modifica il sampling:

$$\boldsymbol{x}_{s} = \sqrt{\alpha_s}\,\hat{\boldsymbol{x}}_0 + \sqrt{1-\alpha_s}\,\boldsymbol{\varepsilon}_\Theta(\boldsymbol{x}_t, t)$$

- **Deterministico**: stesso $\boldsymbol{x}_T$ ? stessa immagine generata
- **Accelerato**: si possono saltare timestep (es. 40 step invece di 400)
- **Stesso modello**: non serve ri-addestrare

---

### 16. Diffusion Models per Problemi Inversi `[p.29-31]`

> **Termini**: posterior sampling = campiona da $p(\boldsymbol{x}|\boldsymbol{y})$ · likelihood = $\|K\boldsymbol{x} - \boldsymbol{y}\|^2$ · DPS = gradiente likelihood corregge sampling ogni step · DiffPIR = alterna denoising + data consistency · score function = $\nabla \log p(\boldsymbol{x})$ direzione per aumentare probabilità
> **A cosa serve**: unire la qualita dei diffusion models con i dati di misura reali. Ottenere ricostruzioni CT/MRI realistiche che rispettano le misure acquisite. E il fronte della ricerca.

> **Spiegazioni**: **Bayes**: $p(x|y) \propto p(y|x)p(x)$. **Prior** $p(x)$: distribuzione delle immagini imparata dal diffusion model (score function $\nabla \log p(x)$ a tutti i livelli di rumore). **Likelihood** $p(y|x) \propto \exp(-\|Kx-y\|^2/2\sigma_y^2)$: accordo con le misure. **DPS** (Diffusion Posterior Sampling): a ogni step di denoising, correggi con $\nabla_{x_t}\|K\hat{x}_0(x_t,t) - y\|^2$ (gradiente approssimato usando $\hat{x}_0$ = stima dell'immagine pulita) — guida il sampling verso immagini coerenti coi dati. **DiffPIR**: alterna 1) denoising (prior, step DDIM) e 2) **data consistency** $x - \tau K^T(Kx - y)$ (proiezione verso le misure) — più interpretabile, connesso a metodi **proximal** classici. **Prior diffusion > prior GAN**: copre TUTTO lo spazio immagini (non una varietà a bassa dimensione $\mathcal{M}=\{G(z):z\in\mathbb{R}^d\}$) — nessun representation error, riusabile per diversi operatori $K$.

> **All'orale**: DPS = denoising + gradiente $\nabla_x\|Ax-y\|^2$. DiffPIR = denoising alternato a data consistency. Prior diffusion > GAN: copre tutto lo spazio immagini. Domanda: «Diffusione per problemi inversi?» = prior diffusivo + guida della misura.


#### 16.1 Punto di Vista Bayesiano

Dato il modello di misura $\boldsymbol{y}^\delta = K\boldsymbol{x}^\dagger + \boldsymbol{e}$ con rumore gaussiano:

$$p(\boldsymbol{x}|\boldsymbol{y}^\delta) \propto \underbrace{p(\boldsymbol{y}^\delta|\boldsymbol{x})}_{\text{likelihood}} \cdot \underbrace{p(\boldsymbol{x})}_{\text{prior}}$$

- **Likelihood**: $p(\boldsymbol{y}^\delta|\boldsymbol{x}) \propto \exp\left(-\frac{1}{2\sigma_y^2}\|K\boldsymbol{x} - \boldsymbol{y}^\delta\|_2^2\right)$  descrive l'accordo con le misure
- **Prior**: il diffusion model fornisce informazione di denoising/score a molti livelli di rumore  descrive la plausibilita dell'immagine

Il diffusion model non da una densita $p(\boldsymbol{x})$ in forma chiusa, ma fornisce **informazione di score** $\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t)$ che guida la ricostruzione iterativa.

#### 16.2 DPS (Diffusion Posterior Sampling)

**Idea**: modificare la traiettoria del reverse diffusion con un gradiente di likelihood.

Score della posterior:
$$\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t|\boldsymbol{y}^\delta) = \underbrace{\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t)}_{\text{dal diffusion model}} + \underbrace{\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}^\delta|\boldsymbol{x}_t)}_{\text{dalla likelihood}}$$

Il gradiente di likelihood e approssimato usando $\hat{\boldsymbol{x}}_0$:

$$\boldsymbol{g}_t \approx \nabla_{\boldsymbol{x}_t} \frac{1}{2\sigma_y^2}\|K\hat{\boldsymbol{x}}_0(\boldsymbol{x}_t, t) - \boldsymbol{y}^\delta\|_2^2$$

**Update**:
$$\boldsymbol{x}_{s} \approx R_\Theta(\boldsymbol{x}_t, t, s) - \eta\,\boldsymbol{g}_t$$

dove $R_\Theta$ e lo step DDIM e $\eta$ e la guidance strength.

<img src="immagini_locali/DPS.png" alt="DPS" width="400">
**Pro**: prior diffusionale sempre nel loop, riusabile per diversi operatori $K$ senza riaddestrare.
**Contro**: computazionalmente pesante (gradiente ad ogni step), $\eta$ da tuningare, gradiente approssimato.

#### 16.3 DiffPIR (Diffusion Plug-and-Play Image Restoration)

**Filosofia plug-and-play**: alternare denoising (prior) e data consistency (misure).

Ad ogni timestep:
1. **Prior step**: step DDIM -> $\boldsymbol{x}_{prior}$ (spinto verso il prior)
2. **Data consistency**: correzione verso le misure:

$$\boldsymbol{x}_{next} = \boldsymbol{x}_{prior} - \tau K^T(K\boldsymbol{x}_{prior} - \boldsymbol{y}^\delta)$$

dove $\tau > 0$ e uno step size.

<img src="immagini_locali/DiffPIR.png" alt="DiffPIR" width="400">
**Pro**: modulare, interpretabile, connesso a metodi di ottimizzazione classici (proximal algorithms).
**Contro**: non e un campionatore posterio esatto, bilanciamento denoising/correction delicato.

#### 16.4 Confronto DPS vs DiffPIR

| | DPS | DiffPIR |
|---|-----|---------|
| **Filosofia** | Steering della traiettoria posterio | Operator splitting (alternanza) |
| **Data consistency** | Gradiente di likelihood | Correzione esplicita $K^T(Kx - y)$ |
| **Interpretabilita** | Meno trasparente | Piu trasparente (proximal) |
| **Stabilita** | Sensibile a $\eta$ | Piu stabile con $\tau$ |
| **Costo** | Gradiente via autograd | Solo forward di $K$ e $K^T$ |

#### 16.5 Limitazioni Generali

- **Computazionalmente costosi**: molte valutazioni della rete per ricostruzione
- **Sensibili al forward model**: se $K$ e sbagliato, la ricostruzione peggiora
- **Plausibilita $\neq$ correttezza**: immagini realistiche ma potenzialmente inventate
- **Distribution shift**: prior addestrato su una popolazione puo fallire su anatomie diverse
- **Iperparametri critici**: guidance strength, noise schedule, numero di step

---

## SCHEDE RIASSUNTIVE `[p.32-34]`

### Mappa Concettuale del Corso

```
COMPUTATIONAL IMAGING
+-- PARTE I: Fondamenti
   +-- Immagine digitale (matrice, formati, DICOM)
   +-- Pixel processing (istogramma, equalizzazione, rumore)
   +-- Filtri (box, gaussiano, mediana, bilaterale)
   +-- Fourier (DFT, FFT, conv?prodotto, fase vs magnitudine)
   +-- Problemi inversi (y=Ax+e, ill-posed, SVD, naive solution)
   +-- Regolarizzazione (min||Ax-y|| + ?R(x))

+-- PARTE II: Deep Learning
    +-- Tensori e preprocessing (B,C,H,W, normalizzazione)
    +-- PyTorch (autograd, ottimizzatori, scheduler)
    +-- NN basics (MLP, attivazioni, universal approximation)
    +-- CNN (conv2d, padding, translation equivariance)
    +-- ResCNN + UNet (residuo, receptive field, encoder-decoder)
    +-- ViT (patch, self-attention, multi-head)
    +-- Metriche (PSNR, SSIM, LPIPS) e Loss (MSE, L1, perceptual)
    +-- Cross-domain (CT, FBP, pipeline ibrida)
    +-- Generativi: VAE (ELBO, KL, reparam trick)
    +-- Generativi: GAN (minimax, mode collapse, WGAN)
    +-- DGP (latent optimization per inverse problems)
    +-- Diffusion (DDPM, DDIM, noise prediction)
    +-- Diffusion Inverse: DPS (gradient guidance) + DiffPIR (plug-and-play)
```

### Formule Chiave da Ricordare

| Concetto | Formula |
|----------|---------|
| Convoluzione 2D | $(I*H)[i,j] = \sum_m\sum_n I[i-m,j-n] H[m,n]$ |
| Teorema convoluzione | $\mathcal{F}(x*h) = \mathcal{F}(x) \cdot \mathcal{F}(h)$ |
| Forward model | $\boldsymbol{y} = A\boldsymbol{x} + \boldsymbol{e}$ |
| SVD | $A = U\Sigma V^T$ |
| Regolarizzazione | $\hat{x} = \arg\min \|Ax-y\|^2 + \lambda R(x)$ |
| ELBO (VAE) | $\mathcal{L} = \mathbb{E}[\log p(x|z)] - \text{KL}(q(z|x)\|p(z))$ |
| GAN minimax | $\min_G\max_D \;\mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ |
| Diffusion forward | $x_t = \sqrt{\alpha_t}\,x_0 + \sqrt{1-\alpha_t}\,\varepsilon$ |
| Diffusion x0 estimate | $\hat{x}_0 = (x_t - \sqrt{1-\alpha_t}\,\varepsilon_\Theta)/\sqrt{\alpha_t}$ |
| DDIM step | $x_s = \sqrt{\alpha_s}\,\hat{x}_0 + \sqrt{1-\alpha_s}\,\varepsilon_\Theta$ |
| DPS guidance | $x_s = R_\Theta(x_t) - \eta\nabla_{x_t}\|K\hat{x}_0 - y^\delta\|^2$ |
| DiffPIR step | $x_{next} = x_{prior} - \tau K^T(Kx_{prior} - y^\delta)$ |
| PSNR | $10\log_{10}(\text{MAX}^2/\text{MSE})$ |
| Self-attention | $\text{softmax}(QK^T/\sqrt{d_k})V$ |

### Domande Tipiche d'Esame `[p.35-42]`

---

#### MODULO 1 - Risposte

**1. Cos'e un problema inverso? Perche e ill-posed?**

Un problema inverso consiste nel risalire alla causa (immagine incognita $\boldsymbol{x}$) a partire dall'effetto osservato (dati misurati $\boldsymbol{y}$), invertendo il modello di acquisizione $\boldsymbol{y} = A\boldsymbol{x} + \boldsymbol{e}$.

E ill-posed (mal posto secondo Hadamard) perche tipicamente viola una o piu delle tre condizioni:
- **Esistenza**: $A$ potrebbe non essere suriettiva, quindi non per ogni $\boldsymbol{y}$ esiste un $\boldsymbol{x}$ tale che $A\boldsymbol{x} = \boldsymbol{y}$ (specialmente con rumore)
- **Unicita**: se $A$ ha null space non banale ($\ker(A) \neq \{0\}$), esistono infinite soluzioni (es. se $m < n$, il sistema e sotto-determinato)
- **Stabilita**: anche quando la soluzione esiste ed e unica, piccoli errori nei dati $\boldsymbol{y}$ (rumore) possono causare grandi variazioni nella soluzione $\boldsymbol{x}$, perche $A^{-1}$ amplifica le componenti associate a valori singolari piccoli

In imaging, quasi tutti i problemi sono ill-posed: deblurring (A mal condizionato), CT con poche proiezioni (sotto-determinato), super-risoluzione ($m \ll n$).

---

**2. Spiega la SVD e il suo ruolo nei problemi inversi.**

La SVD (Singular Value Decomposition) fattorizza $A \in \mathbb{R}^{m \times n}$ come:
$$A = U \Sigma V^T$$
dove $U \in \mathbb{R}^{m \times m}$ e $V \in \mathbb{R}^{n \times n}$ sono matrici ortogonali e $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)$ con $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$.

**Ruolo nei problemi inversi:**
- I valori singolari $\sigma_i$ rivelano il **condizionamento** del problema: il rapporto $\sigma_1/\sigma_r$ (numero di condizionamento) indica quanto il problema e mal posto
- Le colonne di $V$ (right singular vectors) associate a $\sigma_i$ piccoli rappresentano componenti ad **alta frequenza** dell'immagine
- Le colonne di $U$ associate a $\sigma_i$ piccoli rappresentano direzioni nei dati dove il segnale e debole
- La soluzione naive $\boldsymbol{x}_{naive} = \sum_i \frac{\boldsymbol{u}_i^T \boldsymbol{y}}{\sigma_i} \boldsymbol{v}_i$ mostra esplicitamente come i piccoli $\sigma_i$ amplifichino il rumore
- La SVD e la base per metodi di regolarizzazione come la **Truncated SVD** (TSVD), dove si tagliano le componenti con $\sigma_i$ sotto una soglia

---

**3. Perche la soluzione naive amplifica il rumore?**

La soluzione naive e $\boldsymbol{x}_{naive} = A^{-1}\boldsymbol{y} = A^{-1}(A\boldsymbol{x} + \boldsymbol{e}) = \boldsymbol{x} + A^{-1}\boldsymbol{e}$.

In termini SVD:
$$\boldsymbol{x}_{naive} = \sum_{i=1}^{n} \frac{\boldsymbol{u}_i^T \boldsymbol{y}}{\sigma_i} \boldsymbol{v}_i = \sum_{i=1}^{n} \frac{\boldsymbol{u}_i^T(A\boldsymbol{x} + \boldsymbol{e})}{\sigma_i} \boldsymbol{v}_i = \boldsymbol{x} + \sum_{i=1}^{n} \frac{\boldsymbol{u}_i^T \boldsymbol{e}}{\sigma_i} \boldsymbol{v}_i$$

Il termine di errore e $\sum_i \frac{\boldsymbol{u}_i^T \boldsymbol{e}}{\sigma_i} \boldsymbol{v}_i$. Quando $\sigma_i$ e piccolo, il fattore $1/\sigma_i$ e molto grande e amplifica la componente del rumore lungo la direzione $\boldsymbol{v}_i$. Poiche i valori singolari piccoli corrispondono tipicamente a componenti ad alta frequenza dell'immagine, il rumore amplificato si manifesta come oscillazioni rapide e artefatti che dominano la ricostruzione, rendendola inutilizzabile.

---

**4. Cos'e la regolarizzazione? Che ruolo ha $\lambda$?**

La regolarizzazione e una tecnica per stabilizzare la soluzione di problemi ill-posed, aggiungendo informazione a priori al problema:
$$\hat{\boldsymbol{x}} = \arg\min_{\boldsymbol{x}} \|A\boldsymbol{x} - \boldsymbol{y}\|_2^2 + \lambda R(\boldsymbol{x})$$

- $\|A\boldsymbol{x} - \boldsymbol{y}\|_2^2$ (**data fidelity**): forza la soluzione a essere coerente con i dati misurati
- $R(\boldsymbol{x})$ (**regolarizzatore**): codifica la conoscenza a priori sulla soluzione (es. smoothness, sparsita, TV)
- $\lambda > 0$ (**parametro di regolarizzazione**): controlla il trade-off tra i due termini

**Ruolo di $\lambda$:**
- $\lambda \to 0$: prevale la data fidelity -> la soluzione si avvicina a quella naive, con rumore amplificato (overfitting)
- $\lambda \to \infty$: prevale il regolarizzatore -> la soluzione e troppo vincolata dal prior, perde dettagli e diventa troppo liscia (underfitting)
- $\lambda$ ottimale: bilancia fedelta ai dati e plausibilita della soluzione. Si sceglie con metodi come la curva di L, cross-validation, o il principio di discrepanza di Morozov

---

**5. Differenza tra filtro gaussiano e mediana.**

| | Filtro Gaussiano | Filtro Mediana |
|---|---|---|
| **Tipo** | Lineare (convoluzione) | Non lineare (statistica d'ordine) |
| **Operazione** | Media pesata con pesi gaussiani | Mediana dei pixel nel vicinato |
| **Bordi** | Li sfoca | Li preserva bene |
| **Rumore gaussiano** | Lo riduce (low-pass) | Lo riduce moderatamente |
| **Rumore salt&pepper** | Inefficace (spalma gli impulsi) | Eccellente (elimina gli impulsi) |
| **Costo** | $O(k)$ per asse (separabile) | $O(k^2 \log k)$ (ordinamento) |
| **Parametri** | $\sigma$ (larghezza) | $k$ (dimensione finestra) |

Il filtro gaussiano e un low-pass lineare: attenua le alte frequenze uniformemente, il che significa sfocare sia il rumore che i bordi. La mediana e un filtro non lineare che sostituisce ogni pixel con il valore centrale del vicinato ordinato: questo elimina i valori estremi (impulsi) senza mediare, preservando le discontinuita (bordi).

---

**6. Teorema della convoluzione e implicazioni pratiche.**

Il teorema afferma:
$$\mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g)$$

Ovvero: la convoluzione nel dominio spaziale corrisponde al prodotto punto-a-punto nel dominio della frequenza (e viceversa).

**Implicazioni pratiche:**
- **Filtraggio efficiente**: per immagini grandi ($N \times N$) e kernel grandi ($k \times k$), filtrare in frequenza costa $O(N^2 \log N)$ (due FFT + prodotto) invece di $O(N^2 k^2)$ (convoluzione spaziale). Conviene quando $k \gg \log N$
- **Analisi dei filtri**: nel dominio della frequenza si vede direttamente quali frequenze un filtro attenua o amplifica (risposta in frequenza)
- **Design dei filtri**: si puo progettare un filtro specificando la risposta in frequenza desiderata e poi antitrasformare
- **Compressione**: si possono eliminare le componenti frequenziali poco importanti (es. alte frequenze di magnitudine piccola)

---

**7. Perche la fase e piu importante della magnitudine?**

Nella DFT di un'immagine, $X[u,v] = |X[u,v]| \cdot e^{i\phi[u,v]}$:
- **Magnitudine** $|X|$: contiene l'energia/contrasto di ciascuna componente frequenziale ("quanto" di ogni frequenza)
- **Fase** $\phi$: contiene l'informazione di posizione e struttura ("dove" sono le feature)

L'esperimento chiave: scambiando fase e magnitudine tra due immagini A e B:
- Fase di A + Magnitudine di B -> si riconosce l'immagine A
- Fase di B + Magnitudine di A -> si riconosce l'immagine B

Questo perche la fase codifica la **posizione** dei bordi, delle forme e delle strutture dell'immagine. La magnitudine dice solo quanta energia c'e a ciascuna frequenza, ma non dove si trova spazialmente. Senza la fase corretta, anche con la magnitudine giusta, l'immagine ricostruita e irriconoscibile (appare come rumore strutturato).

---

#### MODULO 2 - Risposte

**1. Differenza tra approccio end-to-end e ibrido.**

**End-to-end**: una rete neurale impara la mappa diretta dai dati misurati all'immagine:
$$\boldsymbol{y}^\delta \xrightarrow{f_\Theta} \boldsymbol{x}_{pred}$$
- Pro: semplice, un solo modello, potenzialmente ottimale se i dati di training sono sufficienti
- Contro: la rete deve imparare sia la fisica dell'acquisizione che la prior sulle immagini; richiede grandi dataset paired $(\boldsymbol{y}^\delta, \boldsymbol{x})$; non sfrutta la conoscenza del forward model $A$

**Ibrido**: si usa un metodo classico (es. FBP) per una ricostruzione iniziale, poi una rete neurale la migliora:
$$\boldsymbol{y}^\delta \xrightarrow{\text{FBP}} \tilde{\boldsymbol{x}} \xrightarrow{f_\Theta} \boldsymbol{x}_{pred}$$
- Pro: la fisica e gestita dal metodo classico (FBP); la rete deve solo imparare a rimuovere artefatti (compito piu semplice); funziona con meno dati di training
- Contro: la qualita dipende dal preprocessing; artefatti del FBP potrebbero essere difficili da rimuovere

Nel corso, l'approccio ibrido e preferito per problemi cross-domain come la CT, dove il sinogramma e nel dominio diverso dall'immagine.

---

**2. Cos'e il receptive field e perche e importante?**

Il receptive field di un neurone in una CNN e la regione dell'immagine di input che influenza la sua attivazione. Per una CNN con $L$ layer convoluzionali:
$$r_L = r_{L-1} + (k_L - 1) \prod_{i=1}^{L-1} s_i$$

**Perche e importante:**
- Un neurone puo elaborare solo l'informazione nel suo receptive field
- Per tasks che richiedono **contesto globale** (es. ricostruzione di strutture anatomiche grandi), serve un receptive field grande
- CNN poco profonde hanno receptive field piccolo -> catturano solo dettagli locali
- Per aumentare il receptive field: piu layer, kernel piu grandi, stride > 1, o dilated convolutions
- La UNet risolve questo problema con il bottleneck: il downsampling riduce la risoluzione spaziale, permettendo ai layer profondi di avere un receptive field molto grande rispetto all'immagine originale

---

**3. Perche la UNet ha skip connections?**

Le skip connections nella UNet collegano i feature maps dell'encoder (ad una certa risoluzione) con quelli del decoder (alla stessa risoluzione), tramite concatenazione.

**Motivazioni:**
1. **Preservare dettagli spaziali**: l'encoder, attraverso il downsampling, perde informazione spaziale fine (bordi, texture). Le skip connections re-iniettano questa informazione direttamente nel decoder, permettendo di ricostruire dettagli ad alta risoluzione
2. **Facilitare l'ottimizzazione**: le skip connections forniscono un percorso "breve" per il gradiente durante la backpropagation, mitigando il vanishing gradient e rendendo la rete piu facile da addestrare
3. **Combinare contesto e localizzazione**: l'encoder cattura il contesto semantico (cosa c'e), il decoder con le skip connections recupera la localizzazione precisa (dove c'e). Insieme producono una ricostruzione accurata sia globalmente che localmente

Senza skip connections, la UNet sarebbe un semplice autoencoder bottleneck: la ricostruzione sarebbe molto piu sfocata perche tutta l'informazione dovrebbe passare attraverso il bottleneck.

---

**4. Cos'e la self-attention e come differisce dalla convoluzione?**

**Self-Attention**: ogni token (patch) calcola la sua relazione con TUTTI gli altri token:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

| | Convoluzione | Self-Attention |
|---|---|---|
| **Contesto** | Locale (finestra $k \times k$) | Globale (tutti i token) |
| **Peso** | Fisso (pesi condivisi) | Dinamico (dipende dall'input) |
| **Complessita** | $O(N \cdot k^2)$ | $O(N^2 \cdot d)$ |
| **Inductive bias** | Forte (localita, equivarianza) | Debole (serve positional encoding) |
| **Dati richiesti** | Funziona con dataset piccoli | Richiede dataset grandi |

La convoluzione ha un receptive field fisso e locale: ogni neurone guarda solo i $k \times k$ pixel vicini con pesi fissi. La self-attention ha un receptive field globale e dinamico: ogni patch "decide" a quali altre patch prestare attenzione, con pesi che dipendono dal contenuto (query-key similarity). Questo permette di catturare dipendenze a lungo raggio, ma al costo di complessita quadratica e minore inductive bias.

---

**5. Differenza tra VAE e GAN (obiettivi, training, output).**

| | VAE | GAN |
|---|---|---|
| **Obiettivo** | Massimizzare la likelihood (ELBO) | Vincere il gioco minimax G vs D |
| **Modello** | Esplicito (probabilistico) | Implicito (generatore deterministico) |
| **Training** | Singola rete (encoder+decoder), loss = ELBO | Due reti (G e D) in competizione |
| **Spazio latente** | Regolare (KL verso $\mathcal{N}(0,I)$) | Irregolare (non vincolato) |
| **Output** | Tendenzialmente sfocato (MSE-like) | Nitido e realistico |
| **Stabilita** | Stabile (ottimizzazione standard) | Instabile (mode collapse, equilibrio G/D) |
| **Sampling** | $\boldsymbol{z} \sim \mathcal{N}(0,I) \to D(\boldsymbol{z})$ | $\boldsymbol{z} \sim p(\boldsymbol{z}) \to G(\boldsymbol{z})$ |

Il VAE ottimizza un lower bound della log-likelihood: il termine di ricostruzione tende a produrre immagini sfocate (media delle possibili soluzioni), mentre il termine KL regolarizza lo spazio latente. Il GAN non ha una likelihood esplicita: il generatore impara a produrre immagini che il discriminatore non riesce a distinguere da quelle reali, producendo risultati piu nitidi ma con training piu fragile.

---

**6. Cos'e l'ELBO e quali sono i suoi due termini?**

L'ELBO (Evidence Lower Bound) e il lower bound della log-likelihood che il VAE massimizza:

$$\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})}[\log p_\Theta(\boldsymbol{x}|\boldsymbol{z})]}_{\text{Termine di ricostruzione}} - \underbrace{\text{KL}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}))}_{\text{Termine di regolarizzazione}}$$

**Termine di ricostruzione** ($\mathbb{E}[\log p(\boldsymbol{x}|\boldsymbol{z})]$): misura quanto bene il decoder ricostruisce l'immagine a partire dal latente campionato. In pratica si implementa come MSE o L1 loss tra $\boldsymbol{x}$ e $\hat{\boldsymbol{x}}$. Spinge il modello a preservare l'informazione.

**Termine KL** ($\text{KL}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}))$): misura quanto la distribuzione approssimata del latente (prodotta dall'encoder) si discosta dal prior $\mathcal{N}(0,I)$. Spinge lo spazio latente ad essere regolare e continuo, permettendo il sampling da $\mathcal{N}(0,I)$.

Il trade-off e cruciale: KL troppo debole -> buona ricostruzione ma sampling scarso (lo spazio latente non e regolare). KL troppo forte -> spazio latente regolare ma ricostruzioni sfocate (il bottleneck perde troppa informazione).

---

**7. Come funziona il forward process nei diffusion models?**

Il forward process e un processo Markoviano che aggiunge gradualmente rumore gaussiano a un'immagine pulita $\boldsymbol{x}_0$:

$$q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,\boldsymbol{x}_{t-1},\; \beta_t I)$$

dove $\beta_t \in (0,1)$ e il noise schedule (tipicamente cosine schedule).

**Proprieta chiave - formula chiusa**: si puo campionare $\boldsymbol{x}_t$ direttamente da $\boldsymbol{x}_0$ senza simulare tutti gli step intermedi:

$$\boldsymbol{x}_t = \sqrt{\alpha_t}\,\boldsymbol{x}_0 + \sqrt{1-\alpha_t}\,\boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0,I)$$

dove $\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$.

- Per $t$ piccolo: $\alpha_t \approx 1$, $\boldsymbol{x}_t \approx \boldsymbol{x}_0$ (poco rumore)
- Per $t$ grande: $\alpha_t \approx 0$, $\boldsymbol{x}_t \approx \boldsymbol{\varepsilon}$ (quasi puro rumore)
- Per $t = T$: $\boldsymbol{x}_T \sim \mathcal{N}(0,I)$ (rumore gaussiano puro)

Il forward process e **fissato** (non ha parametri apprendibili). Il modello impara solo il reverse process.

---

**8. Differenza tra DDPM e DDIM sampling.**

| | DDPM | DDIM |
|---|---|---|
| **Tipo** | Stocastico (Markoviano) | Deterministico (non-Markoviano) |
| **Update** | $\boldsymbol{x}_{t-1} = \mu_\Theta(\boldsymbol{x}_t,t) + \sigma_t \boldsymbol{z}$ | $\boldsymbol{x}_s = \sqrt{\alpha_s}\hat{\boldsymbol{x}}_0 + \sqrt{1-\alpha_s}\boldsymbol{\varepsilon}_\Theta$ |
| **Rumore** | Aggiunge rumore ad ogni step ($\boldsymbol{z} \sim \mathcal{N}(0,I)$) | Nessun rumore aggiuntivo |
| **Velocita** | Lento ($T$ step, tipicamente 400-1000) | Veloce (si possono saltare step, es. 40) |
| **Riproducibilita** | Diverso ogni run (stocastico) | Stesso $\boldsymbol{x}_T$ ? stessa immagine |
| **Modello** | Stesso modello addestrato | Stesso modello addestrato (no riaddestramento) |
| **Qualita** | Alta | Simile, leggermente inferiore con pochi step |

DDPM segue il processo inverso Markoviano esatto, aggiungendo rumore stocastico ad ogni step. DDIM deriva un processo inverso deterministico che permette di "saltare" timestep: invece di fare $T$ step, se ne fanno $S \ll T$ (es. 40 invece di 400), ottenendo un'accelerazione di 10x con qualita simile. La deterministicita di DDIM e anche utile per problemi inversi, perche permette di definire una traiettoria riproducibile.

---

**9. DPS vs DiffPIR: filosofia e differenze.**

**DPS (Diffusion Posterior Sampling)**:
- Filosofia: **steering della traiettoria** del reverse diffusion usando il gradiente della likelihood
- Ad ogni step: calcola $\hat{\boldsymbol{x}}_0$, poi il gradiente $\nabla_{\boldsymbol{x}_t}\|K\hat{\boldsymbol{x}}_0 - \boldsymbol{y}^\delta\|^2$ via autograd, e sottrae questo gradiente (pesato per $\eta$) dallo step DDIM
- Il prior e la data consistency sono **accoppiati** nello stesso step
- Pro: flessibile (basta poter calcolare un gradiente), riutilizzabile per diversi $K$
- Contro: costoso (autograd ad ogni step), gradiente approssimato, $\eta$ delicato

**DiffPIR (Diffusion Plug-and-Play Image Restoration)**:
- Filosofia: **operator splitting** (alternanza esplicita tra prior e data consistency)
- Ad ogni step: (1) step DDIM -> $\boldsymbol{x}_{prior}$ (prior step), (2) correzione $\boldsymbol{x}_{next} = \boldsymbol{x}_{prior} - \tau K^T(K\boldsymbol{x}_{prior} - \boldsymbol{y}^\delta)$ (data consistency step)
- Il prior e la data consistency sono **separati** in due sottostep distinti
- Pro: modulare, interpretabile (connesso a metodi proximal), solo forward di $K$ e $K^T$ (no autograd)
- Contro: non e un campionatore posterio esatto, $\tau$ da tuningare

**In sintesi**: DPS "piega" la traiettoria del diffusion verso le misure (guidance implicita). DiffPIR alterna esplicitamente denoising (prior) e proiezione verso i dati (consistency), come nei metodi plug-and-play classici.

---

**10. Perche un diffusion model e un prior migliore di un GAN per problemi inversi?**

Un diffusion model e un prior superiore a un GAN per diversi motivi:

1. **Ricchezza del prior**: un GAN vincola la soluzione al range di un generatore a bassa dimensionalita ($\mathcal{M} = \{G(\boldsymbol{z}) : \boldsymbol{z} \in \mathbb{R}^d\}$ con $d \ll n$). Un diffusion model impara a denoisare a molti livelli di rumore, coprendo effectively l'intero spazio delle immagini realistiche, non solo una varieta a bassa dimensione

2. **No representation error**: con un GAN, se l'immagine vera non e nel range del generatore ($\boldsymbol{x}^\dagger \notin \mathcal{M}$), la ricostruzione non puo mai essere corretta, anche con ottimizzazione perfetta. Il diffusion model non ha questo vincolo

3. **No latent optimization**: il GAN richiede di ottimizzare $\boldsymbol{z}$ per ogni nuova misura (problema non convesso, sensibile all'inizializzazione). Il diffusion model usa un processo iterativo diretto nello spazio immagine

4. **Score information a tutti i livelli**: il diffusion model fornisce informazione di score/denoising a ogni livello di rumore, permettendo di combinare prior e data consistency in modo graduale e controllato

5. **Meno mode collapse**: i GAN soffrono di mode collapse (generano poca varieta). I diffusion models, avendo un training piu stabile (semplice MSE sul rumore), coprono meglio la distribuzione dei dati

6. **Flessibilita**: lo stesso diffusion prior puo essere riusato per diversi operatori $K$ (deblurring, inpainting, super-risoluzione, CT) senza riaddestramento, semplicemente cambiando il termine di data consistency

Il costo principale e la **lentezza**: ogni ricostruzione richiede decine/centinaia di valutazioni della rete, contro una singola forward pass del generatore GAN.
