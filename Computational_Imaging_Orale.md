# Computational Imaging - Versione Discorsiva per l'Esame Orale

> **Unibo - Laurea Magistrale in Informatica**
> Modulo 1: Prof.ssa Elena Loli Piccolomini | Modulo 2: Davide Evangelista
> Link: [Modulo 1](https://elenaloli.github.io/computational-imaging/intro.html) | [Modulo 2](https://devangelista2.github.io/computational-imaging/2025-26/intro/environment-setup.html)

---

**Come usare questo documento**: a differenza della versione schematica, qui trovi gli stessi contenuti (stesse formule, stessi concetti, stesse domande d'esame) ma scritti in forma discorsiva, a paragrafi. Ogni sezione è pensata per essere letta come una spiegazione ad alta voce. Le formule sono integrate nel discorso, non isolate. Le domande d'esame in fondo hanno risposte complete e argomentate, da ripetere all'orale.

---

## Indice

- **PARTE I - Fondamenti** (Modulo 1)
  - 1. Introduzione al Computational Imaging
  - 2. Pixel Processing
  - 3. Filtri e Convoluzione
  - 4. Trasformata di Fourier
  - 5. Problemi Inversi
  - 6. Regolarizzazione
- **PARTE II - Deep Learning per Imaging** (Modulo 2)
  - 7. Processing Images per Reti Neurali
  - 8. PyTorch Essentials
  - 9. Da Machine Learning a Neural Networks
  - 10. CNN (Convolutional Neural Networks)
  - 11. Residual Learning e UNet
  - 12. Vision Transformers e Loss Design
  - 13. Problemi Cross-Domain
  - 14. Deep Generative Models: VAE e GAN
  - 15. Diffusion Models
  - 16. Diffusion Models per Problemi Inversi
- **Domande d'Esame e Risposte Discorsive**

---

## PARTE I - FONDAMENTI DI IMAGING (Modulo 1)

---

### 1. Introduzione al Computational Imaging

Il **Computational Imaging** (CI) è un campo interdisciplinare che unisce acquisizione, elaborazione e analisi delle immagini attraverso modelli computazionali. Si fonda su tre pilastri: l'Image Processing, che si occupa di elaborare un'immagine per migliorarla o estrarne informazione (filtraggio, enhancement, restauro); la Computer Vision, che estrae informazione semantica dalle immagini (riconoscimento, segmentazione, ricostruzione 3D); e la Computer Graphics, che sintetizza immagini a partire da modelli (rendering, simulazione).

Un'immagine digitale è una matrice discreta $I \in \mathbb{R}^{M \times N}$ (o $\mathbb{R}^{M \times N \times C}$ per immagini a colori). Ogni elemento della matrice è un pixel, e tre parametri la caratterizzano: la **risoluzione spaziale** $M \times N$, la **profondità di bit** (numero di valori per pixel: 8-bit = 256 livelli), e il numero di **canali** (1 per il grigio, 3 per RGB, 4 per RGBA).

Per passare dal mondo continuo a quello digitale serve la **discretizzazione**. Qui entra in gioco il teorema di **Nyquist**: la frequenza di campionamento $f_s$ deve essere almeno il doppio della massima frequenza presente nel segnale, $f_s \ge 2f_{max}$. Se si campiona sotto questa soglia, le alte frequenze si ripiegano su quelle basse creando l'**aliasing** — l'esempio classico sono le ruote del carro nei film che sembrano girare all'indietro. Senza Nyquist, in imaging medico avremmo solo artefatti invece di diagnosi attendibili.

Per memorizzare e trasmettere immagini si usano vari formati. Il **DICOM** (Digital Imaging and Communications in Medicine) è lo standard per l'imaging medico: non salva solo i pixel ma anche metadati clinici (paziente, modalità di acquisizione, parametri). **JPEG** usa una compressione lossy basata su DCT (Discrete Cosine Transform) seguita da quantizzazione: butta via le alte frequenze che l'occhio umano non percepisce — un po' come descrivere "sabbia colorata" invece di ogni singolo granello. **PNG** offre compressione lossless, ideale quando ogni dettaglio conta.

In Python si leggono immagini con librerie diverse: OpenCV (formato BGR, uint8), Matplotlib (RGB, float [0,1]), Pillow (RGB, vari formati), o imageio (versatile, supporta DICOM).

**All'esame**: la domanda classica è "Cos'è il CI?" — bisogna rispondere che inverte il forward model $y=Ax+e$ per ricostruire immagini da misure indirette (CT, risonanza magnetica). E poi spiegare Nyquist: se campioni a meno di $2f_{max}$ le frequenze si ripiegano (aliasing). Esempio delle ruote del carro.

---

### 2. Pixel Processing

Le operazioni di pixel processing agiscono su ogni pixel indipendentemente dagli altri, punto-a-punto. Sono le operazioni base per migliorare la qualità di un'immagine grezza prima di qualsiasi analisi più complessa.

#### Istogramma e Equalizzazione

L'**istogramma** $h(k)$ conta il numero di pixel con intensità $k$: $h(k) = |\{(i,j) : I(i,j) = k\}|$. L'istogramma normalizzato $p(k) = h(k)/(M \cdot N)$ è una stima della PDF dell'intensità.

L'**equalizzazione** dell'istogramma è una trasformazione che rende l'istogramma il più possibile uniforme. La funzione di trasformazione è la **CDF** (Cumulative Distribution Function): $T(k) = \sum_{i=0}^{k} p(i)$. In pratica, si usa la CDF come lookup table per distribuire i livelli uniformemente su 0–255, aumentando il contrasto globale — come alzare le ombre in Photoshop. Il limite è che non preserva il contrasto locale e può amplificare il rumore.

Per ovviare a questo si usa **CLAHE** (Contrast Limited Adaptive Histogram Equalization): una versione locale che divide l'immagine in blocchi, equalizza ciascuno, e applica un **contrast clipping** (taglia i picchi dell'istogramma locale) per non amplificare il rumore nelle zone uniformi.

#### Modelli di Rumore

Il rumore è una degradazione stocastica dell'immagine. I modelli principali sono quattro:

Il **rumore gaussiano additivo** segue $I_{noisy} = I + n$ con $n \sim \mathcal{N}(0, \sigma^2)$: è indipendente dal segnale, dà una grana fine sempre presente (come ISO alto in fotografia). Il **rumore Sale & Pepe** colpisce pixel casuali portandoli a 0 o 255: è impulsivo, e si rimuove efficacemente con un filtro mediano (non lineare, ordina i pixel nella finestra e sceglie la mediana — preserva i bordi perché non media). Il **rumore di Poisson** (shot noise) è correlato all'intensità del segnale $I_{noisy} \sim \text{Poisson}(I)$, tipico in condizioni di poca luce. Il **rumore Speckle** è moltiplicativo $I_{noisy} = I \cdot n$ con $n \sim \mathcal{N}(1, \sigma^2)$, tipico di radar e ultrasuoni.

**All'esame**: "Equalizzazione" = CDF come lookup table. "CLAHE" = divide in blocchi + taglia picchi di contrasto (non amplifica rumore). Se mostrano un istogramma stretto, rispondere "poco contrasto, equalizzazione allarga". Differenza equalizzazione vs CLAHE: CLAHE è locale e non amplifica rumore.

---

### 3. Filtri e Convoluzione

I filtri sono il gesto base dell'elaborazione di immagini: permettono di ripulire il rumore, enfatizzare bordi, sfocare — come aggiustare una foto mossa o piena di puntini bianchi.

#### Sistemi LSIS

Un sistema **LSIS** (Lineare Shift-Invariante) è caratterizzato dalla sua **risposta impulsiva** $h$. È "lineare" perché l'output a una combinazione di input è la combinazione degli output; è "shift-invariante" perché se sposti l'input, l'output si sposta uguale — come un timbro che stampa la stessa forma dappertutto. L'uscita è la **convoluzione** dell'ingresso con $h$: $y = x * h$. Le proprietà sono commutativa, associativa e distributiva (come la moltiplicazione tra numeri).

Un filtro mediano **non** è LSIS perché non è lineare — non può essere espresso come convoluzione. Questa è una domanda subdola che capita spesso all'esame.

#### Convoluzione 1D e 2D

La convoluzione è l'operazione fondamentale: il kernel $k \times k$ scorre sull'immagine, moltiplica i pesi per i pixel e somma. È come un pennello che dipinge mescolando ogni pixel con i suoi vicini, e il kernel è la forma del pennello. Il ribaltamento del kernel serve matematicamente per rendere l'operazione commutativa ($x*h = h*x$).

Nelle CNN si usa in realtà la **cross-correlazione** (senza ribaltamento del kernel), ma per abitudine viene chiamata convoluzione — all'esame può capitare la domanda "convoluzione vs cross-correlazione?".

Il **padding** è necessario ai bordi per mantenere la stessa dimensione dopo la convoluzione: si possono aggiungere zeri, replicare il bordo, o riflettere. Con $p = (k-1)/2$ e stride 1, l'output ha la stessa dimensione dell'input.

#### Filtri Lineari

Il **Box Filter** è la media uniforme in una finestra $k \times k$: costo $O(k^2)$ per pixel, effetto low-pass (sfocatura). Il **Filtro Gaussiano** $G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$ produce una sfocatura più naturale perché dà più peso ai pixel centrali. È **separabile**: $G(x,y) = G(x) \cdot G(y)$, quindi il costo scende da $O(k^2)$ a $O(k)$ per asse — come dipingere prima tutte le righe orizzontali e poi le verticali. Il parametro $\sigma$ controlla l'ampiezza: $\sigma$ grande = più sfocatura.

Una domanda subdola che capita: "Convolvere due/tre volte un gaussiano piccolo o usare un gaussiano grande?" — è equivalente, perché la convoluzione di due gaussiani è ancora un gaussiano.

#### Filtri Non Lineari

Il **Filtro Mediano** sostituisce ogni pixel con la mediana del vicinato ordinato. È eccellente per rimuovere rumore **salt & pepper** perché elimina i valori estremi senza mediare, e quindi **preserva i bordi** — a differenza del gaussiano che sfuma tutto. Non è lineare e non è separabile.

Il **Filtro Bilaterale** combina due pesi: un peso **spaziale** $w_s$ (gaussiano sulla distanza, come il filtro gaussiano) e un peso **di range** $w_r$ (gaussiano sulla differenza di intensità). Il risultato è $I_{filtered}[i,j] = \frac{\sum_{(m,n)} w_s(m,n) \cdot w_r(I[m,n], I[i,j]) \cdot I[m,n]}{\sum_{(m,n)} w_s(m,n) \cdot w_r(I[m,n], I[i,j])}$. Questo permette di sfumare solo pixel di colore simile, fermandosi ai bordi — come colorare dentro le linee. Non è separabile perché $w_r$ dipende dai pixel dell'immagine, non è una funzione fissa.

**All'esame**: "LSIS" = stesso filtro ovunque, lineare + shift-invariante. "Mediano NON sfoca" (a differenza del gaussiano). "Bilaterale" = gaussiano + differenza intensità: preserva bordi. "Separabile" = costo $O(k^2) \to O(k)$. Domanda tipica: "filtro che toglie rumore senza sfocare?" = mediano o bilaterale.

---

### 4. Trasformata di Fourier

La trasformata di Fourier permette di vedere l'immagine "dall'altro lato": non più pixel, ma **frequenze**. Serve a comprimere (JPEG), rimuovere rumori periodici, e capire perché un'immagine ha artefatti.

#### Richiami sui Numeri Complessi

Un numero complesso $z = a + ib = r e^{i\theta}$ ha due componenti: la **magnitudine** $|z| = \sqrt{a^2 + b^2}$ che dice "quanto", e la **fase** $\theta = \arctan(b/a)$ che dice "dove".

#### Serie e Trasformata di Fourier

Ogni segnale periodico $f(t)$ con periodo $T$ si scrive come somma di sinusoidi: $f(t) = \sum_{n=-\infty}^{\infty} c_n e^{i 2\pi n t / T}$. È come scomporre un accordo musicale nelle sue note singole.

La **DFT** (Discrete Fourier Transform) estende questo alle immagini. In 2D: $X[u,v] = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} I[m,n] e^{-i 2\pi (um/M + vn/N)}$. La **FFT** (Fast Fourier Transform) la calcola in $O(N \log N)$ invece di $O(N^2)$ — un'enorme differenza pratica.

Le **basse frequenze** corrispondono a zone uniformi (cielo, pareti), mentre le **alte frequenze** corrispondono a dettagli fini (bordi, texture, erba).

#### Teorema della Convoluzione

Il teorema più importante: $x * h \xleftrightarrow{\mathcal{F}} X \cdot H$. La convoluzione nel dominio spaziale diventa un semplice **prodotto** nel dominio della frequenza (e viceversa). Implicazione pratica: filtrare in frequenza costa $O(N \log N)$ contro $O(N \cdot k^2)$ nello spazio — molto più veloce per kernel grandi, perché due FFT + un prodotto punto-a-punto bastano.

Si possono così realizzare filtri: low-pass (taglia alte frequenze = sfocatura, denoising), high-pass (taglia basse frequenze = edge detection, sharpening), band-pass (seleziona una banda).

#### Fase vs Magnitudine

La **fase** $\phi$ contiene la **struttura** dell'immagine — bordi, forme, posizione. La **magnitudine** $|X|$ contiene il "quanto" — il contrasto globale, l'energia. L'esperimento chiave: se prendi la fase dell'immagine A e la magnitudine dell'immagine B, nell'immagine ricostruita si riconosce A, non B. Questo dimostra che la fase è molto più importante della magnitudine. Senza la fase corretta, anche con la magnitudine giusta, l'immagine è irriconoscibile.

#### Aliasing e Ringing

L'**aliasing** si verifica quando frequenze oltre $f_{Nyquist}$ si ripiegano su frequenze più basse: l'esempio delle ruote del carro nei film. Il **ringing** è un artefatto da troncamento brusco in frequenza (come le ondulazioni attorno ai bordi netti in una foto JPEG compressa troppo).

#### Compressione e Hybrid Images

La compressione DFT scarta le componenti di Fourier con magnitudine piccola — poche componenti a bassa frequenza catturano già l'essenza dell'immagine (come JPEG: tieni le basse frequenze, butta il rumore fine). Le **Hybrid Images** combinano basse frequenze di un'immagine con alte frequenze di un'altra: da vicino si vede l'immagine ad alta frequenza, da lontano quella a bassa frequenza — un effetto ottico affascinante.

**All'esame**: "Teorema convoluzione" = filtrare nello spazio = moltiplicare in frequenza (velocissimo). "Fase più importante della magnitudine" perché contiene la struttura dell'immagine. "Aliasing" = frequenze oltre Nyquist si ripiegano. Domanda classica: "Fase o magnitudine?" = Fase.

---

### 5. Problemi Inversi e Computational Imaging

Il cuore del Computational Imaging è il **problema inverso**. Il **forward model** è $\boldsymbol{y} = A\boldsymbol{x} + \boldsymbol{e}$ dove $\boldsymbol{x} \in \mathbb{R}^n$ è l'immagine incognita (ground truth), $A \in \mathbb{R}^{m \times n}$ è l'operatore di acquisizione, $\boldsymbol{y} \in \mathbb{R}^m$ sono i dati misurati, e $\boldsymbol{e}$ è il rumore di misura. Il problema diretto (dato $x$, calcolare $y$) è facile e ben posto. Il problema inverso (dato $y$, trovare $x$) è difficile e spesso **mal posto**.

#### Problemi Ill-Posed (Hadamard)

Secondo Hadamard, un problema è ben posto se soddisfa tre condizioni: **esistenza** (per ogni $y$ esiste una soluzione $x$), **unicità** (la soluzione è unica), **stabilità** (piccole variazioni in $y$ causano piccole variazioni in $x$). Se anche una sola manca, il problema è ill-posed.

Esempio concreto: la TAC con poche angolazioni. Il sistema è sottodeterminato ($m < n$), quindi infinite ossa possono dare le stesse ombre (non unicità). È come cercare di ricostruire un pupazzo di neve dalla pozzanghera che lascia sciogliendosi — troppe configurazioni possibili.

#### SVD (Singular Value Decomposition)

La SVD fattorizza $A = U \Sigma V^T$ dove $U$ e $V$ sono matrici ortogonali e $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)$ con $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$. I valori singolari $\sigma_i$ rivelano il **condizionamento** del problema: il numero di condizionamento $\sigma_1/\sigma_r$ indica quanto il problema è mal posto. Più è alto, più il problema è mal condizionato.

La soluzione in termini SVD è $x = \sum (u_i^T y / \sigma_i) v_i$. Qui si vede il problema: quando $\sigma_i$ è piccolo, $1/\sigma_i$ è enorme e **amplifica il rumore**. I valori singolari piccoli corrispondono alle alte frequenze, quindi il rumore amplificato si manifesta come oscillazioni rapide che rovinano la ricostruzione. È come alzare il volume al massimo per sentire un sussurro — senti il sussurro, ma anche ogni minimo fruscio amplificato.

**All'esame**: "Hadamard" = esistenza + unicità + stabilità. "Ill-posed se manca una delle tre". "$\sigma_i$ piccoli" = dividono per numeri piccoli = amplificano rumore. "Perché ill-posed?" = $\sigma_i$ piccoli fanno esplodere il rumore.

---

### 6. Regolarizzazione

La **regolarizzazione** è la tecnica che rende stabile la soluzione di problemi ill-posed. Senza, le immagini ricostruite sarebbero inutilizzabili. Il framework è:

$$\hat{\boldsymbol{x}} = \arg\min_{\boldsymbol{x}} \underbrace{\|A\boldsymbol{x} - \boldsymbol{y}\|_2^2}_{\text{data fidelity}} + \lambda \underbrace{R(\boldsymbol{x})}_{\text{regolarizzatore}}$$

Il primo termine (data fidelity) spinge la soluzione a essere coerente con i dati misurati. Il secondo termine (regolarizzatore) codifica la conoscenza a priori sulla soluzione. $\lambda$ è il parametro che controlla il trade-off tra i due.

#### Tipi di Regolarizzazione

**Tikhonov (L2)**: $R(x) = \|x\|_2^2$. Tira tutto verso zero, produce una soluzione liscia — penalizza i valori grandi comprimendoli. È come tirare tutto verso zero con elastici: le punte si appiattiscono. Funziona bene quando la soluzione è effettivamente liscia, ma non preserva bordi netti.

**Total Variation (TV)**: $R(x) = \|\nabla x\|_1$. Somma il gradiente in norma L1 — permette salti netti (bordi) ma penalizza oscillazioni. È come piegare un foglio di carta (ORIGAMI): pieghe nette, ma ogni faccia è liscia. È la scelta giusta quando vuoi preservare i bordi nell'immagine ricostruita.

**Sparsità (L1)**: $R(x) = \|x\|_1$. Preferisce soluzioni con molti coefficienti nulli — compressed sensing. Come volere il minor numero possibile di ingredienti.

#### Scelta di $\lambda$

$\lambda$ controlla il bilanciamento. Se $\lambda \to 0$, prevale la data fidelity: la soluzione è rumorosa (overfitting, insegue il rumore). Se $\lambda \to \infty$, prevale la regolarizzazione: la soluzione è troppo liscia (underfitting, perde dettagli). Il $\lambda$ ottimale è dove smetti di sentire il rumore ma i bordi sono ancora nitidi.

Per scegliere $\lambda$ si usa la **L-curve**: un grafico log-log di $\|Ax-y\|$ vs $\|R(x)\|$ — il gomito della curva è il punto ottimale. In pratica si usa anche cross-validazione.

#### Collegamento con il Deep Learning

Il framework model-based classico usa regolarizzatori **hand-crafted** (progettati a mano). Il deep learning sostituisce $R(x)$ con un **prior appreso dai dati**: una rete neurale impara direttamente che aspetto hanno le immagini realistiche. Questo è il ponte verso la Parte II.

**All'esame**: "$\hat{x} = \arg\min \|Ax-y\|^2 + \lambda R(x)$". "L2 = liscio". "TV = preserva bordi ($\|\nabla x\|_1$)". "$\lambda$ grande = troppo liscio, piccolo = rumoroso". "L-curve per scegliere $\lambda$". Domanda: "L2 o TV?" = TV se vuoi bordi netti.

---

## PARTE II - DEEP LEARNING PER IMAGING (Modulo 2)

---

### 7. Processing Images per Reti Neurali

Prima di dare un'immagine a una rete neurale, bisogna prepararla nel formato giusto. Le immagini per le reti neurali sono **tensori 4D**: $(B, C, H, W)$ dove $B$ è il batch size (quante immagini alla volta), $C$ è il numero di canali (1 per grigio, 3 per RGB), $H$ e $W$ sono altezza e larghezza. In PyTorch: `x.shape` restituisce `torch.Size([32, 3, 256, 256])` per un batch di 32 immagini RGB 256×256.

#### Normalizzazione

La normalizzazione è cruciale per il training. Non basta portare i pixel in [0,1]: si deve **sottrarre la media del dataset e dividere per la deviazione standard**. Questo serve a scalare tutte le feature uniformemente per stabilizzare il training e prevenire gradienti esplosivi. Per ImageNet, ad esempio, si usano mean=[0.485, 0.456, 0.406] e std=[0.229, 0.224, 0.225].

Se l'output deve essere in [-1,1] (tipico per GAN e diffusion models), si usa $x = (x - 0.5)/0.5$. Il dtype standard per il training è `float32`.

#### Dataset e DataLoader

In PyTorch, un **Dataset** è una classe che carica i dati. Un **DataLoader** ci gira sopra e gestisce: batch multipli (tante immagini alla volta), shuffle (mescola per evitare correlazione nell'ordine), e workers multipli (parallelizza il caricamento da disco).

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

Il **Mayo Dataset** usato nel corso ha 3305 immagini train e 327 test di slice CT addominali.

La **data augmentation** (flip, rotazione, crop, elastic deform) aumenta la varietà dei dati senza raccoglierne di nuovi — fondamentale quando il dataset è piccolo.

#### Pipeline di Ricostruzione

Ci sono due approcci. **End-to-end**: $y^\delta \xrightarrow{f_\Theta} x_{pred}$ — la rete impara tutto, dalla fisica dell'acquisizione al denoising. **Ibrido**: $y^\delta \xrightarrow{\text{FBP}} \tilde{x} \xrightarrow{f_\Theta} x_{pred}$ — un metodo classico (FBP) fa una ricostruzione iniziale, la rete rimuove artefatti. L'approccio ibrido è preferito per problemi cross-domain come la CT.

**All'esame**: "Tensore = (B, C, H, W)". "Non normalizzare a [0,1]: usa mean/std del dataset". "DataLoader = batch + shuffle + workers". "Augmentation = più dati virtuali". "End-to-end vs Ibrido: l'ibrido separa fisica da deep learning".

---

### 8. PyTorch Essentials

PyTorch è il toolkit essenziale per il deep learning in imaging. Il cuore è **autograd**: costruisce automaticamente un **computational graph** delle operazioni. Quando chiami `loss.backward()`, calcola i gradienti di **tutti** i parametri con la **chain rule** in un colpo solo — come un domino dove la loss è l'ultima pedina.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad)  # tensor([7.0]) = 2*2 + 3
```

Durante la **valutazione** si usa `torch.no_grad()` per risparmiare memoria (non serve il grafo).

#### Ottimizzatori

- **SGD**: $\theta \leftarrow \theta - \eta \nabla L$. Semplice, può oscillare in loss landscape complessi.
- **SGD + momentum**: $v \leftarrow \mu v + \nabla L$; $\theta \leftarrow \theta - \eta v$. Media mobile del gradiente per ridurre oscillazioni.
- **Adam**: combina momentum + RMSProp. Ha un learning rate **diverso per ogni parametro** — adattivo, è il default nella pratica.

Il **Learning Rate Scheduler** (es. CosineAnnealingLR) riduce il learning rate quando la loss smette di migliorare.

#### IPPy

La libreria del corso si chiama IPPy e contiene: `operators/` (Blurring, Radon, etc.), `solvers/` (Tikhonov, ISTA, metodi classici), `nn/` (UNet, DiffusionUNet), `utilities/` (device detection, noise generation).

**All'esame**: "Autograd = grafo delle operazioni, backward() = gradienti con chain rule". "`torch.no_grad()` quando valuti (non serve grafo)". "Adam = lr adattivo per parametro". "Domanda: backprop? = chain rule sul grafo computazionale".

---

### 9. Da Machine Learning a Neural Networks

Un modello lineare $f(x) = Wx + b$ può rappresentare solo relazioni lineari. Per problemi complessi servono **non-linearità**. Le **funzioni di attivazione** introducono questa non-linearità.

La più usata è **ReLU** ($\max(0, x)$): non satura (a differenza di sigmoid/tanh che per grandi valori si appiattiscono bloccando l'apprendimento — **vanishing gradient**). Ma ha un problema: se un neurone riceve sempre input negativi, muore per sempre (non impara più). La soluzione è **LeakyReLU** ($\max(\alpha x, x)$) che lascia passare un piccolo flusso anche per input negativi.

La **Softmax** trasforma un vettore in probabilità (per classificazione). **GELU** e **SiLU/Swish** sono usate in Transformers e modelli generativi.

Un **MLP** (Multi-Layer Perceptron) è una rete fully-connected: ogni neurone è connesso a tutti quelli dello strato successivo. Il **Teorema di Approssimazione Universale** dice che un MLP con un solo hidden layer (sufficientemente largo) può approssimare qualsiasi funzione continua su un compatto. Sembra potente, ma per un'immagine 100×100: 10.000 input × 1000 neuroni = 10 milioni di parametri — impraticabile.

Ecco perché servono le **CNN**: risolvono il problema con **weight sharing** (lo stesso kernel viene usato su tutta l'immagine, come un timbro), riducendo drasticamente il numero di parametri.

Le **feature** nelle reti profonde sono **gerarchiche**: layer bassi catturano bordi e texture, layer intermedi catturano forme e pattern locali, layer alti catturano oggetti e strutture semantiche.

**All'esame**: "ReLU evita vanishing gradient (sigmoide satura)". "LeakyReLU risolve dying ReLU". "MLP = universal approximator". "Perché CNN meglio di MLP per immagini? = weight sharing, meno parametri, sfrutta struttura spaziale".

---

### 10. CNN (Convolutional Neural Networks)

Le CNN sono il cavallo di battaglia per l'elaborazione di immagini. Invece di connettere ogni pixel a ogni neurone (MLP), le CNN riusano lo stesso pattern su tutta l'immagine.

#### Convoluzione 2D nelle CNN

La convoluzione 2D nelle CNN è in realtà **cross-correlazione** (senza ribaltamento del kernel) — ma per abitudine si chiama convoluzione. Un kernel $k \times k$ scorre sull'immagine con un certo **stride** (passo). Con stride 2, l'output è dimezzato in ogni dimensione.

Il **weight sharing** è la chiave: gli stessi pesi del kernel vengono riutilizzati su tutta l'immagine. Questo significa circa 3500× meno parametri di un MLP equivalente. Inoltre, dà **translation equivariance**: se l'input si sposta, l'output si sposta nello stesso modo.

Il **padding** serve per non perdere i bordi dell'immagine: senza padding, l'immagine si restringe a ogni convoluzione. Con $p = (k-1)/2$, l'output mantiene la stessa dimensione.

#### Architettura CNN

Tipicamente i canali **progrediscono** in profondità: 64 → 128 → 256. Questo perché la risoluzione spaziale si riduce (con stride o pooling), e più canali compensano la riduzione spaziale mantenendo la capacità rappresentativa.

Un trucco importante: **stack di più kernel 3×3** è meglio di un singolo kernel grande (es. 7×7). Perché? Due 3×3 in sequenza hanno 27 parametri contro 49 di un 7×7, e in più si hanno due ReLU invece di una — più non-linearità e stessa capacità di coprire l'area.

Il **pooling** (max pooling o average pooling) riduce la risoluzione spaziale e aumenta il receptive field, ma nelle CNN per ricostruzione (dove serve la risoluzione piena in output) si evita.

#### Training per Ricostruzione

Nel corso si usa un approccio **sintetico**: i dati di training sono generati online con $\boldsymbol{y} = K\boldsymbol{x} + \boldsymbol{e}$, dove $K$ è un operatore di blurring e $\boldsymbol{e}$ è rumore gaussiano. Non serve un dataset paired reale.

**All'esame**: "Weight sharing = ~3500× meno parametri di MLP". "Stack 3×3 > 7×7: meno parametri + più ReLU". "Channel progression: 64 → 128 → 256 compensa riduzione spaziale". "Perché più canali in profondità? = compensa la riduzione spaziale".

---

### 11. Residual Learning e UNet

#### ResCNN (Residual CNN)

Invece di imparare la mappa diretta $y^\delta \to x$, la ResCNN impara il **residuo**: $f_\Theta(y^\delta) \approx x - y^\delta$. Poi $x_{pred} = y^\delta + f_\Theta(y^\delta)$. Perché funziona? Perché quando input e output sono simili (immagine degradata e ricostruzione), è più facile imparare la differenza che l'intera mappa. È come correggere un compito già scritto invece di riscriverlo da zero.

La **skip connection** $x_{pred} = y^\delta + f_\Theta(y^\delta)$ è essa stessa una skip connection globale. I vantaggi sono due: il residuo è più facile da imparare (bias induttivo: l'identity mapping è già una buona soluzione), e le skip connections permettono ai gradienti di fluire direttamente ai layer iniziali (risolvendo il vanishing gradient). Lo svantaggio è che se il rumore è molto forte ($\sigma_n$ alto), $y^\delta$ non è una buona approssimazione e il residuo diventa grande e difficile da imparare.

#### Receptive Field

Il **receptive field** di un neurone è la regione dell'input che influenza la sua attivazione. Per una CNN con $L$ layer convoluzionali: $r_L = r_{L-1} + (k_L - 1)\prod_{i=1}^{L-1}s_i$. Il problema delle CNN poco profonde è che hanno un receptive field piccolo — non catturano contesto globale.

L'UNet risolve questo con il **downsampling**: layer profondi dopo vari stride 2 hanno un receptive field enorme (tutta l'immagine). È come guardare un quadro: da vicino vedi dettagli ma non l'insieme; allontanandoti vedi tutto ma perdi dettagli. L'UNet fa entrambe le cose.

#### UNet

L'UNet ha un'architettura **encoder-decoder** multi-scala:

- **Encoder**: riduce la risoluzione (downsampling con stride 2), aumenta i canali — cattura contesto globale a varie scale
- **Bottleneck**: rappresentazione compatta al centro, massima compressione, massimo numero di canali
- **Decoder**: aumenta la risoluzione (upsampling con trasposta o interpolazione), riduce i canali — ricostruisce dettagli spaziali
- **Skip connections**: collegano encoder e decoder tramite **concatenazione** (non somma come ResNet) — preservano dettagli spaziali che altrimenti andrebbero persi nel bottleneck

Le skip connections sono fondamentali: l'encoder, attraverso il downsampling, perde informazione spaziale fine. Le skip connections re-iniettano questa informazione nel decoder. È come un artista che prima fa uno schizzo a matita dell'intera scena (encoder) e poi aggiunge i dettagli guardando il modello originale (skip connection).

L'UNet funziona bene per il deblur per tre motivi: (1) la struttura multi-scala cattura blur a diverse scale, (2) le skip connections preservano bordi e dettagli fini, (3) il bottleneck forza una rappresentazione compatta che aiuta a rimuovere rumore incoerente.

**All'esame**: "ResNet: $f(x)-x$ = più facile se input ≈ output". "UNet: encoder-decoder + skip (dettagli)". "Bottleneck = compressione". "Skip connection ResNet (somma) vs UNet (concatenazione): ResNet riduce carico, UNet preserva informazione spaziale". "Perché UNet funziona per deblur? = multi-scala + skip + bottleneck".

---

### 12. Vision Transformers e Loss Design

#### Vision Transformer (ViT)

I Transformer, nati per NLP, sono stati adattati alle immagini. Il ViT divide l'immagine in **patch** $P \times P$, le proietta linearmente in token embeddings. Il numero di patch è $N = (H/P) \times (W/P)$.

La **Self-Attention** è il meccanismo chiave: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V$. Ogni patch "guarda" tutte le altre — contesto globale, ma costo $O(N^2)$. $Q$ (Query) chiede "cosa cerco?", $K$ (Key) dice "cosa ho?", $V$ (Value) dice "cosa offro?". La **Multi-Head Self-Attention (MHSA)** usa $h$ teste in parallelo, ognuna impara relazioni diverse.

Il **Positional Encoding** è necessario perché la self-attention è **permutation-invariant** — senza posizione, "gatto a sinistra" e "gatto a destra" sono la stessa cosa.

**CNN vs ViT**: la CNN ha un forte inductive bias (località, equivarianza) e funziona con dataset piccoli. Il ViT ha contesto globale, meno bias induttivo, ma richiede molti più dati.

#### Metriche di Qualità

- **PSNR** (Peak Signal-to-Noise Ratio): $\text{PSNR} = 10 \log_{10}(\text{MAX}^2/\text{MSE})$ in dB. Più alto = migliore. Tipico 25–40 dB per imaging medico.
- **SSIM** (Structural Similarity Index): considera luminanza, contrasto e struttura locale. Range [-1, 1], 1 = identico.
- **LPIPS** (Learned Perceptual Image Patch Similarity): distanza nello spazio feature di una rete pre-addestrata (VGG/AlexNet). Più basso = più simile percettivamente. È la metrica che meglio cattura la qualità visiva percepita dall'occhio umano.

#### Loss Functions

- **MSE (L2)**: $\frac{1}{n}\|x - \hat{x}\|_2^2$. Produce immagini **sfocate** perché minimizzare MSE porta alla media delle possibili interpretazioni — e la media di tante immagini è sfocata (come una foto lunga esposizione di una folla).
- **L1**: $\frac{1}{n}\|x - \hat{x}\|_1$. Più sharp di MSE, perché la mediana è meno sensibile ai valori estremi.
- **Perceptual**: $\|\phi(x) - \phi(\hat{x})\|_2^2$ nello spazio feature di VGG. Produce dettagli realistici perché controlla "sembra un osso?" non "i pixel sono uguali?".

**All'esame**: "Attention = softmax(QK^T/√d)V". "Self-attention = contesto globale ma O(N²)". "MSE = media = sfoca". "LPIPS = percettivo (feature VGG)". "SSIM = struttura locale". "Miglior loss? = LPIPS per qualità visiva, PSNR per fedeltà pixel".

---

### 13. Problemi Cross-Domain

In **CT** (Tomografia Computerizzata), i dati misurati sono **sinogrammi** (proiezioni a vari angoli), non immagini. Il forward model è $\boldsymbol{y} = \mathcal{R}\boldsymbol{x} + \boldsymbol{e}$ dove $\mathcal{R}$ è la **Trasformata di Radon**: l'integrale della densità lungo linee che attraversano il corpo. Il dominio dei dati (sinogramma) è diverso dal dominio dell'immagine — è un problema **cross-domain**.

Il metodo classico per ricostruire è la **FBP** (Filtered Back-Projection): (1) si filtrano le proiezioni con un **ramp filter** $|\omega|$ in frequenza (passa-alto, compensa lo smoothness della back-projection amplificando le alte frequenze per nitidezza); (2) si fa la back-projection: ogni proiezione filtrata viene "spalmata" all'indietro lungo la stessa direzione di acquisizione, e tutte le direzioni vengono sommate. L'**apodization** (finestre come Hann, cosine) sul ramp filter riduce il rumore ma perde risoluzione — è un trade-off.

Con pochi angoli, la FBP produce **streaking** (artefatti a strisce). Qui entra la **pipeline ibrida**: $y \xrightarrow{\text{FBP}} \tilde{x} \xrightarrow{f_\Theta} x_{pred}$. La FBP gestisce la fisica (converte sinogramma in immagine), l'UNet rimuove artefatti e rumore. Vantaggio: la rete lavora nel dominio immagine (più semplice), la FBP gestisce la trasformata di Radon.

**All'esame**: "CT = dalle proiezioni (sinogramma) ricostruisci l'immagine = problema inverso". "Radon = integrale densità lungo linee". "FBP = Ramp filter + backprojection". "Apodization = toglie rumore ma perde risoluzione". "Pipeline ibrida: FBP + UNet".

---

### 14. Deep Generative Models: VAE e GAN

I modelli generativi imparano la distribuzione dei dati $p_{data}(x)$ (a differenza dei modelli discriminativi che imparano $p(y|x)$ o una mappatura diretta). Un generativo può campionare nuovi esempi — utile per data augmentation, priors per problemi inversi, e compressione.

#### VAE (Variational Autoencoder)

Un VAE è un autoencoder probabilistico: $x \xrightarrow{E} (\mu, \sigma) \to z = \mu + \sigma\varepsilon \xrightarrow{D} \hat{x}$. L'encoder produce media e varianza per il latente $z$, non un codice deterministico.

L'obiettivo è l'**ELBO** (Evidence Lower Bound): $\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$. Il primo termine (ricostruzione) spinge a ricostruire bene $x$ da $z$. Il secondo termine (KL divergence) forza la distribuzione del latente $q(z|x)$ a essere vicina a $\mathcal{N}(0, I)$, regolarizzando lo spazio latente.

Il **Reparameterization trick** separa $\mu, \sigma$ dal rumore $\varepsilon$: $z = \mu + \sigma \odot \varepsilon$ con $\varepsilon \sim \mathcal{N}(0, I)$. Questo rende il sampling **differenziabile**, permettendo la backpropagation.

Il VAE è **stabile** (singola loss, allenamento standard) ma produce immagini **sfocate** — perché minimizzare MSE porta alla media delle possibili interpretazioni. Come una foto lunga esposizione di una folla: tutto è sfuocato.

#### GAN (Generative Adversarial Network)

Due reti in competizione: il **Generator** $G$ produce immagini fake da rumore $z$, il **Discriminator** $D$ classifica reale/finto. L'obiettivo è minimax: $\min_G\max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$. In pratica si usa la **non-saturating loss** per G ($-\mathbb{E}[\log D(G(z))]$) che è più stabile.

Il GAN è come un falsario (G) e un esperto d'arte (D): il falsario migliora a creare falsi, l'esperto migliora a smascherarli. All'equilibrio, i falsi sono indistinguibili.

Problemi: **mode collapse** — il Generator impara a produrre sempre la stessa immagine (o poche varianti) perché scopre un punto debole del Discriminator e lo sfrutta. **Training instabile** — il bilanciamento G/D è delicato. Soluzioni: **WGAN** (Wasserstein distance, più stabile), **spectral normalization** (limita la norma dei layer).

Il GAN produce immagini **nitide** ma soffre di mode collapse. La scelta VAE vs GAN è un trade-off: VAE stabile ma sfocato, GAN nitido ma instabile.

#### Deep Generative Prior (DGP)

Un generatore pre-addestrato $G$ definisce un **prior** sulle immagini: si ottimizza solo il latente $z$ mantenendo $G$ congelato: $\hat{z} = \arg\min_z \|K G(z) - y^\delta\|^2 + \lambda \|z\|^2$, poi $\hat{x} = G(\hat{z})$.

Il problema è il **representation error**: se l'immagine vera non è rappresentabile dal generatore ($x_{true} \notin \mathcal{M}$ dove $\mathcal{M} = \{G(z) : z \in \mathbb{R}^d\}$), non la recuperi per quanto ottimizzi $z$. È come cercare una persona in un quartiere sbagliato.

**All'esame**: "VAE: ELBO = ricostruzione - KL. Stabile ma sfocato". "GAN: minimax, nitido ma mode collapse". "WGAN = più stabile". "Reparameterization trick: rende il sampling differenziabile". "DGP: limite principale = representation error". "VAE o GAN? = dipende dal trade-off stabilità/qualità".

---

### 15. Diffusion Models

I diffusion models sono lo stato dell'arte per la generazione di immagini (DALL-E, Stable Diffusion, Midjourney). Imparano a **invertire un processo di aggiunta graduale di rumore**.

#### Forward Process (DDPM)

Il forward process aggiunge rumore gaussiano a un'immagine pulita $x_0$ in $T$ passi (tipicamente 1000): $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I)$. Il **noise schedule** $\beta_t$ controlla quanto rumore aggiungere a ogni passo.

La **formula chiusa** è fondamentale: $x_t = \sqrt{\alpha_t}\, x_0 + \sqrt{1-\alpha_t}\, \varepsilon$ con $\varepsilon \sim \mathcal{N}(0,I)$ e $\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$. Questa formula permette di saltare direttamente a qualsiasi $t$ senza iterare. Per $t$ piccolo: $\alpha \approx 1$, quasi immagine pulita. Per $t$ grande: $\alpha \approx 0$, quasi puro rumore. Per $t=T$: $x_T \sim \mathcal{N}(0,I)$.

Il forward process è **fissato** (nessun parametro apprendibile). Il modello impara solo il reverse process.

#### Training (Noise Prediction)

Una singola rete $\varepsilon_\Theta(x_t, t)$ impara a **predire il rumore**: $\min_\Theta \mathbb{E}[\|\varepsilon - \varepsilon_\Theta(x_t, t)\|^2]$. È più facile predire il rumore (distribuzione fissa $\mathcal{N}(0,I)$) che l'immagine (distribuzione complessa). Da questa predizione si può stimare l'immagine pulita: $\hat{x}_0 = (x_t - \sqrt{1-\alpha_t}\,\varepsilon_\Theta(x_t, t)) / \sqrt{\alpha_t}$.

L'architettura è una **DiffusionUNet**: UNet time-conditioned con residual blocks, GroupNorm, SiLU, self-attention a risoluzioni intermedie, e **sinusoidal embedding** per il timestep $t$ (come il positional encoding dei Transformer). Il time embedding viene iniettato in ogni blocco — così la stessa rete sa a che livello di rumore sta operando. Si usa anche **EMA** (Exponential Moving Average) per sampling più stabile.

#### Sampling DDPM

Partendo da rumore puro $x_T \sim \mathcal{N}(0,I)$, si applicano $T$ step di denoising. L'update è: $x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}(x_t - \frac{\beta_t}{\sqrt{1-\alpha_t}}\varepsilon_\Theta(x_t, t)) + \sigma_t z$. È **lento**: servono $T$ valutazioni della rete (tipicamente 400-1000), ognuna una forward pass della DiffusionUNet. Confronto: una GAN genera in un colpo solo.

#### DDIM Sampling

**DDIM** (Denoising Diffusion Implicit Models) modifica il sampling: $x_s = \sqrt{\alpha_s}\,\hat{x}_0 + \sqrt{1-\alpha_s}\,\varepsilon_\Theta(x_t, t)$. È **deterministico** (stesso $x_T$ = stessa immagine) e **accelerato** (salta timestep, tipicamente 20-50 passi invece di 1000). Non serve riaddestrare il modello.

DDIM è utile per problemi inversi perché è deterministico e permette di guidare il sampling.

**All'esame**: "Forward = formula chiusa $x_t$ da $x_0$". "Training = MSE sul rumore". "DDPM: stocastico, lento". "DDIM: deterministico, veloce (salta step)". "Perché predice rumore e non immagine? = il rumore ha distribuzione fissa $\mathcal{N}(0,I)$, è più facile". "Formula chiusa: $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\varepsilon$".

---

### 16. Diffusion Models per Problemi Inversi

I diffusion models possono essere usati come **priors** per problemi inversi, unendo la qualità generativa con i dati di misura reali.

#### Punto di Vista Bayesiano

$p(x|y^\delta) \propto p(y^\delta|x) \cdot p(x)$. La **likelihood** $p(y|x) \propto \exp(-\|Kx - y\|^2 / 2\sigma_y^2)$ misura l'accordo con le misure. Il **prior** $p(x)$ è la distribuzione delle immagini imparata dal diffusion model. Il diffusion model non dà $p(x)$ in forma chiusa, ma fornisce informazione di **score** $\nabla \log p(x_t)$ a tutti i livelli di rumore.

Il prior diffusion è superiore al prior GAN: copre **tutto** lo spazio delle immagini (non una varietà a bassa dimensione $\mathcal{M} = \{G(z): z \in \mathbb{R}^d\}$), quindi nessun representation error. Ed è riutilizzabile per diversi operatori $K$ senza riaddestramento.

#### DPS (Diffusion Posterior Sampling)

DPS modifica la traiettoria del reverse diffusion con un gradiente di likelihood. A ogni step: calcola $\hat{x}_0$, poi il gradiente $\nabla_{x_t}\|K\hat{x}_0 - y^\delta\|^2$ via autograd, e sottrae questo gradiente (pesato per $\eta$) dallo step DDIM. È come correggere la rotta di una nave in mezzo alla nebbia: il prior dice la direzione generale, il GPS ($K\hat{x}_0 - y$) corregge a ogni passo.

Pro: flessibile, riutilizzabile per diversi $K$. Contro: computazionalmente pesante (autograd a ogni step), gradiente approssimato (usa $\hat{x}_0$, non $x_t$), $\eta$ da tuningare.

#### DiffPIR (Diffusion Plug-and-Play Image Restoration)

DiffPIR alterna due fasi a ogni timestep: (1) **prior step**: step DDIM verso $x_{prior}$, (2) **data consistency**: $x_{next} = x_{prior} - \tau K^T(K x_{prior} - y^\delta)$. È come un pittore che alterna: dipinge liberamente seguendo l'ispirazione (prior), poi controlla la foto reale e corregge (data consistency).

Pro: modulare, interpretabile (connesso a metodi proximal classici come HQS, ADMM), solo forward di $K$ e $K^T$ (no autograd). Contro: non è un vero campionatore bayesiano.

**DPS vs DiffPIR**: DPS "steering" della traiettoria (gradiente ad ogni step), DiffPIR "operator splitting" (alternanza). DPS più costoso ma più fedele alla teoria bayesiana; DiffPIR più veloce e interpretabile.

#### Limitazioni Generali

Computazionalmente costosi (molte valutazioni della rete). Sensibili al forward model (se $K$ è sbagliato, la ricostruzione peggiora). **Plausibilità $\neq$ correttezza**: immagini realistiche ma potenzialmente inventate (allucinazioni). **Distribution shift**: prior addestrato su una popolazione può fallire su anatomie diverse.

**All'esame**: "DPS = denoising + gradiente $\nabla_x\|Ax-y\|^2$". "DiffPIR = denoising alternato a data consistency". "Prior diffusion > GAN: copre tutto lo spazio immagini, no representation error". "DPS vs DiffPIR: steering vs splitting".

---

## DOMANDE D'ESAME E RISPOSTE DISCORSIVE

Di seguito trovi le domande tipiche dell'esame, con risposte pensate per essere ripetute ad alta voce all'orale. Ogni risposta è un discorso completo, non appunti.

---

### MODULO 1

#### 1. Cos'è un problema inverso? Perché è ill-posed?

Un problema inverso consiste nel risalire alla causa — l'immagine incognita $x$ — a partire dall'effetto osservato — i dati misurati $y$ — invertendo il modello di acquisizione $y = Ax + e$. È un po' come cercare di capire cosa c'è in una scatola chiusa dal rumore che fa scuotendola: hai le misure indirette (il rumore) e devi ricostruire l'oggetto interno.

Il problema è **ill-posed** (mal posto secondo Hadamard) quando viola una o più delle tre condizioni di Hadamard: esistenza, unicità e stabilità. Tipicamente nei problemi di imaging ne viola almeno una. La **non esistenza** si ha quando $A$ non è suriettiva: non per ogni $y$ esiste un $x$ che soddisfa $Ax = y$, specialmente in presenza di rumore. La **non unicità** si ha quando $A$ ha un null space non banale: se $\ker(A) \neq \{0\}$, esistono infinite soluzioni — per esempio quando $m < n$ (sistema sottodeterminato). La **non stabilità** è la più subdola: anche quando soluzione esiste ed è unica, piccoli errori nei dati (rumore) causano grandi variazioni nella soluzione perché $A^{-1}$ amplifica le componenti associate a valori singolari piccoli.

In imaging, quasi tutti i problemi sono ill-posed: deblurring ($A$ mal condizionato), CT con poche proiezioni (sotto-determinato), super-risoluzione ($m \ll n$).

#### 2. Spiega la SVD e il suo ruolo nei problemi inversi.

La SVD (Singular Value Decomposition) fattorizza una matrice $A \in \mathbb{R}^{m \times n}$ come $A = U \Sigma V^T$, dove $U$ e $V$ sono matrici ortogonali e $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)$ con $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$.

Nei problemi inversi, la SVD è fondamentale per diversi motivi. Primo, i valori singolari $\sigma_i$ rivelano il **condizionamento** del problema: il rapporto $\sigma_1/\sigma_r$, detto numero di condizionamento, indica quanto il problema è mal posto. Più alto è, più il problema è mal condizionato.

Secondo, la SVD mostra esplicitamente il problema dell'amplificazione del rumore. La soluzione naive in termini SVD è $x_{naive} = \sum_i (u_i^T y / \sigma_i) v_i = x + \sum_i (u_i^T e / \sigma_i) v_i$. Quando $\sigma_i$ è piccolo, $1/\sigma_i$ amplifica enormemente la componente di rumore. Poiché i valori singolari piccoli corrispondono tipicamente alle alte frequenze, il rumore amplificato si manifesta come oscillazioni rapide nell'immagine ricostruita.

Terzo, la SVD è la base per metodi di regolarizzazione come la **Truncated SVD** (TSVD), dove si tagliano le componenti con $\sigma_i$ sotto una soglia, eliminando le direzioni più rumorose.

#### 3. Perché la soluzione naive amplifica il rumore?

La soluzione naive è $x_{naive} = A^{-1}y = A^{-1}(Ax + e) = x + A^{-1}e$. In termini SVD, $x_{naive} = x + \sum_i (u_i^T e / \sigma_i) v_i$. Il problema è il termine $1/\sigma_i$: quando $\sigma_i$ è molto piccolo, questo fattore diventa enorme e amplifica la componente di rumore lungo la direzione $v_i$.

Immagina di avere un microfono al massimo volume per sentire un sussurro: sentirai il sussurro, ma anche ogni minimo fruscio amplificato a dismisura. I valori singolari piccoli corrispondono alle alte frequenze spaziali — dettagli fini e bordi — e il rumore amplificato si manifesta come oscillazioni rapide che rendono l'immagine inutilizzabile. Ecco perché serve la regolarizzazione.

#### 4. Cos'è la regolarizzazione? Che ruolo ha $\lambda$?

La regolarizzazione è una tecnica per stabilizzare la soluzione di problemi ill-posed, aggiungendo informazione a priori. Il framework è: $\hat{x} = \arg\min \|Ax - y\|^2 + \lambda R(x)$. Il primo termine, la **data fidelity**, forza la soluzione a essere coerente con i dati misurati. Il secondo termine, il **regolarizzatore** $R(x)$, codifica la conoscenza a priori (smoothness, sparsità, bordi). $\lambda$ controlla il trade-off tra i due.

Se $\lambda$ tende a zero, prevale la data fidelity: la soluzione si avvicina a quella naive e diventa rumorosa — overfitting. Se $\lambda$ tende a infinito, prevale il regolarizzatore: la soluzione è troppo vincolata e perde dettagli — underfitting, tutto troppo liscio. Il $\lambda$ ottimale è dove smetti di sentire il rumore ma i bordi sono ancora nitidi. Si sceglie con metodi come la L-curve (il gomito del grafico log-log), cross-validazione, o il principio di discrepanza di Morozov.

I tipi principali di regolarizzazione sono tre. **Tikhonov (L2)**: $R(x) = \|x\|^2$, produce soluzioni lisce, penalizza valori grandi. **TV (Total Variation)**: $R(x) = \|\nabla x\|_1$, preserva bordi perché usa L1 sul gradiente — permette salti netti. **L1 (sparsità)**: $R(x) = \|x\|_1$, preferisce coefficienti nulli — compressed sensing.

#### 5. Differenza tra filtro gaussiano e mediana.

Il filtro gaussiano è un **filtro lineare** (una convoluzione) con pesi a campana: ogni pixel diventa la media pesata dei suoi vicini. È un low-pass: attenua le alte frequenze, quindi riduce il rumore gaussiano ma **sfoca anche i bordi**. È separabile in due passaggi 1D, il che lo rende efficiente ($O(k)$ per asse).

Il filtro mediano è un **filtro non lineare**: ordina i pixel nella finestra e sceglie il valore centrale. Non media, ma seleziona — e questo fa la differenza. È eccellente per rimuovere rumore **salt & pepper** (valori impulsivi) perché quei pixel estremi finiscono in coda o in testa all'ordinamento e vengono scartati. Soprattutto, **preserva i bordi**: non sfuma la transizione perché non mescola pixel di valori diversi.

La differenza cruciale: il gaussiano media (sfoca tutto, rumore e bordi), il mediano seleziona (toglie gli impulsi, lascia i bordi). Non è separabile e costa $O(k^2 \log k)$.

#### 6. Teorema della convoluzione e implicazioni pratiche.

Il teorema della convoluzione afferma che $\mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g)$: la convoluzione nel dominio spaziale corrisponde al **prodotto punto-a-punto** nel dominio della frequenza.

Le implicazioni pratiche sono enormi. Primo, **filtraggio efficiente**: per immagini grandi e kernel grandi, filtrare in frequenza costa $O(N^2 \log N)$ (due FFT + prodotto) invece di $O(N^2 k^2)$ — è conveniente quando $k \gg \log N$. Secondo, **analisi dei filtri**: nel dominio della frequenza si vede direttamente quali frequenze un filtro attenua o amplifica (risposta in frequenza). Terzo, **design dei filtri**: si può progettare un filtro specificando la risposta in frequenza desiderata. Quarto, **compressione**: si eliminano componenti frequenziali poco importanti — esattamente ciò che fa JPEG.

#### 7. Perché la fase è più importante della magnitudine?

Nella DFT, $X[u,v] = |X[u,v]| \cdot e^{i\phi[u,v]}$. La **magnitudine** $|X|$ contiene l'energia/contrasto di ciascuna frequenza — dice "quanto" di ogni frequenza è presente. La **fase** $\phi$ contiene l'informazione di **posizione e struttura** — dice "dove" sono le feature.

L'esperimento chiave che dimostra la superiorità della fase: prendi la fase dell'immagine A e la magnitudine dell'immagine B, ricostruisci — si riconosce l'immagine A, non B. Scambiando, si riconosce B. Perché la fase codifica la **posizione dei bordi**, delle forme, delle strutture. Senza la fase corretta, anche con la magnitudine giusta, l'immagine è irriconoscibile — appare come rumore strutturato.

---

### MODULO 2

#### 1. Differenza tra approccio end-to-end e ibrido.

L'approccio **end-to-end** usa una rete neurale che impara la mappa diretta dai dati misurati all'immagine: $y^\delta \to x_{pred}$. È semplice: un solo modello, potenzialmente ottimale con dati sufficienti. Ma la rete deve imparare sia la fisica dell'acquisizione che il prior sulle immagini — richiede grandi dataset paired e non sfrutta la conoscenza del forward model $A$.

L'approccio **ibrido** usa un metodo classico (es. FBP) per una ricostruzione iniziale, poi una rete neurale la migliora: $y^\delta \xrightarrow{\text{FBP}} \tilde{x} \xrightarrow{f_\Theta} x_{pred}$. La fisica è gestita dal metodo classico, la rete impara solo a rimuovere artefatti — un compito più semplice che richiede meno dati. L'ibrido è preferito per problemi cross-domain come la CT.

#### 2. Cos'è il receptive field e perché è importante?

Il **receptive field** di un neurone in una CNN è la regione dell'immagine di input che influenza la sua attivazione. Per una CNN con $L$ layer: $r_L = r_{L-1} + (k_L - 1)\prod_{i=1}^{L-1} s_i$.

È importante perché un neurone può elaborare solo l'informazione nel suo receptive field. Per task che richiedono **contesto globale** (ricostruzione di strutture anatomiche grandi), serve un receptive field grande. CNN poco profonde hanno receptive field piccolo — catturano solo dettagli locali. Per aumentarlo: più layer, kernel più grandi, stride > 1, o dilated convolutions.

L'UNet risolve questo problema con il bottleneck: il downsampling riduce la risoluzione spaziale, permettendo ai layer profondi di avere un receptive field molto grande rispetto all'immagine originale. È come allontanarsi da un quadro per vederlo tutto.

#### 3. Perché la UNet ha skip connections?

Le skip connections nella UNet collegano encoder e decoder alla stessa risoluzione tramite **concatenazione**. Sono fondamentali per tre motivi.

Primo, **preservano dettagli spaziali**: l'encoder, attraverso il downsampling, perde informazione spaziale fine (bordi, texture). Le skip connections re-iniettano questa informazione direttamente nel decoder.

Secondo, **facilitano l'ottimizzazione**: forniscono un percorso breve per il gradiente durante la backpropagation, mitigando il vanishing gradient.

Terzo, **combinano contesto e localizzazione**: l'encoder cattura il contesto semantico (cosa c'è), le skip connections recuperano la localizzazione precisa (dove c'è). Insieme producono una ricostruzione accurata sia globalmente che localmente.

Senza skip connections, la UNet sarebbe un semplice autoencoder: tutta l'informazione dovrebbe passare attraverso il bottleneck, e la ricostruzione sarebbe molto più sfocata.

Nota importante: la UNet usa **concatenazione** (concatena feature maps lungo il canale), mentre ResNet usa **somma** ($x + f(x)$). La differenza è che ResNet alleggerisce il carico (impara il residuo), mentre UNet preserva l'informazione spaziale.

#### 4. Cos'è la self-attention e come differisce dalla convoluzione?

La **self-attention** è il meccanismo alla base dei Transformer. Ogni token (patch) calcola la sua relazione con **tutti** gli altri token: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V$. $Q$ (Query) chiede "cosa cerco?", $K$ (Key) dice "cosa ho?", $V$ (Value) dice "cosa offro?". Il prodotto $QK^T$ calcola la similarità tra ogni coppia di token, e il softmax la normalizza in pesi di attenzione.

La differenza fondamentale con la convoluzione è quadruplice. La convoluzione ha contesto **locale** (finestra $k \times k$), la self-attention ha contesto **globale** (tutti i token). La convoluzione ha pesi **fissi** (appresi e condivisi spazialmente), la self-attention ha pesi **dinamici** (dipendono dal contenuto dell'input). La convoluzione costa $O(N \cdot k^2)$, la self-attention costa $O(N^2 \cdot d)$ — quadratica nel numero di token. La convoluzione ha un forte **inductive bias** (località, equivarianza), mentre la self-attention ne ha poco e richiede positional encoding e molti dati.

#### 5. Differenza tra VAE e GAN.

Il **VAE** ottimizza un lower bound della log-likelihood (ELBO). Il training è stabile (singola rete, ottimizzazione standard). Produce immagini tendenzialmente **sfocate** perché il termine di ricostruzione (MSE) tende alla media delle possibili interpretazioni. Lo spazio latente è **regolare** grazie al termine KL che lo forza verso $\mathcal{N}(0,I)$.

La **GAN** è un gioco minimax tra due reti. Il training è **instabile** (bilanciamento G/D delicato, mode collapse). Produce immagini **nitide e realistiche**, ma soffre di mode collapse (il generatore impara a produrre poca varietà). Lo spazio latente è **irregolare** (non vincolato).

La scelta è un trade-off: VAE se vuoi stabilità e uno spazio latente regolare (utile per inversion), GAN se vuoi la massima qualità visiva e puoi gestire l'instabilità.

#### 6. Cos'è l'ELBO e quali sono i suoi due termini?

L'ELBO (Evidence Lower Bound) è il lower bound della log-likelihood che il VAE massimizza: $\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$.

Il **termine di ricostruzione** $\mathbb{E}[\log p(x|z)]$ misura quanto bene il decoder ricostruisce l'immagine dal latente campionato. In pratica si implementa come MSE o L1. Spinge il modello a preservare informazione.

Il **termine KL** $\text{KL}(q(z|x) \| p(z))$ misura quanto la distribuzione approssimata del latente si discosta dal prior $\mathcal{N}(0,I)$. Spinge lo spazio latente a essere regolare e continuo, permettendo il sampling da $\mathcal{N}(0,I)$.

Il trade-off è cruciale. KL troppo debole: buona ricostruzione ma sampling scarso (spazio latente irregolare). KL troppo forte: spazio latente regolare ma ricostruzioni sfocate (troppa compressione).

#### 7. Come funziona il forward process nei diffusion models?

Il forward process è un processo Markoviano che aggiunge gradualmente rumore gaussiano a un'immagine pulita $x_0$: $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)$, dove $\beta_t$ è il noise schedule (tipicamente cosine schedule).

La **proprietà chiave** è che si può campionare $x_t$ direttamente da $x_0$ senza simulare tutti gli step: $x_t = \sqrt{\alpha_t}\, x_0 + \sqrt{1-\alpha_t}\, \varepsilon$ con $\varepsilon \sim \mathcal{N}(0,I)$ e $\alpha_t = \prod_{s=1}^t (1-\beta_s)$. Per $t$ piccolo, $\alpha_t \approx 1$ e $x_t$ è quasi uguale a $x_0$; per $t$ grande, $\alpha_t \approx 0$ e $x_t$ è quasi puro rumore. A $t=T$, $x_T \sim \mathcal{N}(0,I)$.

Attenzione: il forward process è **fissato** (non ha parametri apprendibili). Il modello impara solo il reverse process.

#### 8. Differenza tra DDPM e DDIM sampling.

**DDPM** è stocastico e Markoviano: a ogni step aggiunge rumore $z \sim \mathcal{N}(0,I)$. È lento (400-1000 step) e genera variabilità — stesso punto di partenza può dare immagini diverse. L'update è $x_{t-1} = \mu_\Theta(x_t,t) + \sigma_t z$.

**DDIM** è deterministico e non-Markoviano: nessun rumore aggiuntivo a ogni step. È veloce (20-50 step, salta timestep) e riproducibile — stesso $x_T$ dà sempre la stessa immagine. L'update è $x_s = \sqrt{\alpha_s}\,\hat{x}_0 + \sqrt{1-\alpha_s}\,\varepsilon_\Theta$.

Vantaggio enorme: DDIM usa lo **stesso modello addestrato** di DDPM, non serve riaddestrare. Si ottiene un'accelerazione di 10-20× con qualità simile. La deterministicità è utile per problemi inversi (inversion, guidance).

#### 9. DPS vs DiffPIR: filosofia e differenze.

**DPS** (Diffusion Posterior Sampling) ha una filosofia di **steering**: modifica la traiettoria del reverse diffusion con il gradiente della likelihood. A ogni step calcola $\hat{x}_0$, poi il gradiente $\nabla_{x_t}\|K\hat{x}_0 - y\|^2$ via autograd, e corregge lo step DDIM. Il prior e la data consistency sono **accoppiati** nello stesso step. È flessibile ma costoso (autograd), e il gradiente è approssimato.

**DiffPIR** (Diffusion Plug-and-Play Image Restoration) ha una filosofia di **operator splitting**: alterna esplicitamente (1) prior step (DDIM) e (2) data consistency ($x - \tau K^T(Kx - y)$). Il prior e la data consistency sono **separati** in due sottostep. È modulare, interpretabile (connesso a metodi proximal classici), e non serve autograd — solo forward di $K$ e $K^T$.

In sintesi: DPS "piega" la traiettoria del diffusion verso le misure. DiffPIR alterna denoising e proiezione verso i dati.

#### 10. Perché un diffusion model è un prior migliore di un GAN per problemi inversi?

Per sei motivi. Primo, **ricchezza del prior**: un GAN vincola la soluzione al range di un generatore a bassa dimensionalità ($\mathcal{M} = \{G(z) : z \in \mathbb{R}^d\}$ con $d \ll n$). Un diffusion model impara a denoisare a molti livelli di rumore, coprendo tutto lo spazio delle immagini realistiche.

Secondo, **no representation error**: con un GAN, se $x_{true} \notin \mathcal{M}$, la ricostruzione non può mai essere corretta. Il diffusion model non ha questo vincolo.

Terzo, **no latent optimization**: il GAN richiede di ottimizzare $z$ per ogni nuova misura (problema non convesso, sensibile all'inizializzazione). Il diffusion model usa un processo iterativo diretto nello spazio immagine.

Quarto, **score information a tutti i livelli**: il diffusion model fornisce informazione di score/denoising a ogni livello di rumore.

Quinto, **meno mode collapse**: i GAN soffrono di mode collapse, i diffusion models hanno training più stabile (semplice MSE sul rumore).

Sesto, **flessibilità**: lo stesso diffusion prior può essere riusato per diversi operatori $K$ (deblur, inpainting, super-res, CT) senza riaddestramento.

Il costo principale è la lentezza: ogni ricostruzione richiede decine/centinaia di valutazioni della rete, contro una singola forward pass del generatore GAN.
