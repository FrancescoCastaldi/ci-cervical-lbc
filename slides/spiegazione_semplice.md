# Spiegazione del progetto

> Da leggere alla prof seguendo le slide.

---

## Slide 1 — Copertina

**Sulla slide:** Titolo, nomi, logo UNIBO, link GitHub.

**Concetti chiave:**

**Restauro immagini**
- *Cos'è:* Processo che recupera una foto danneggiata per riportarla alla versione originale
- *Cosa fa:* Prende un'immagine rovinata (sfocata, rumorosa) e produce un'immagine pulita e nitida

**Metodo**
- *Cos'è:* Una procedura o algoritmo specifico per risolvere un problema
- *Cosa fa:* Definisce i passaggi precisi per passare dall'immagine degradata a quella restaurata

**Esempio:** È come portare tre foto rovinate in tre diversi laboratori di restauro: uno usa prodotti chimici (TV), uno usa un computer che ha visto migliaia di foto (UNet), uno usa un artista che sa ridipingere i dettagli (DiffPIR). Alla fine confrontiamo i risultati.

---

## Slide 2 — Il problema inverso

**Sulla slide:** Formula y = H(x) + n. Riquadro arancione "Why it is difficult".

**Concetti chiave:**

**Problema inverso**
- *Cos'è:* Un problema in cui si conosce il risultato e si vuole risalire alla causa. Invece di "ho una foto, la rovino" (diretto), si fa "ho una foto rovinata, recupero l'originale" (inverso)
- *Cosa fa:* Cerca di invertire il processo di degradazione per ricostruire ciò che è andato perso

**Mal posto (ill-posed)**
- *Cos'è:* Un problema che non soddisfa le tre condizioni di Hadamard (esistenza, unicità, stabilità della soluzione)
- *Cosa fa:* Rende impossibile risolvere il problema semplicemente invertendo la formula — il rumore viene amplificato a dismisura, distruggendo il risultato

**Prior**
- *Cos'è:* Una conoscenza preliminare che dice "le immagini pulite di solito sono fatte così"
- *Cosa fa:* Guida la scelta tra tutte le possibili soluzioni, selezionando quella più "plausibile"

**AWGN (Additive White Gaussian Noise)**
- *Cos'è:* Rumore bianco gaussiano additivo. È un disturbo casuale, a forma di campana (gaussiano), che si somma (additivo) a ogni pixel in modo indipendente
- *Cosa fa:* Aggiunge a ogni pixel un valore casuale tratto da una distribuzione normale. Più è alto sigma, più il disturbo è forte

**Gaussian blur (sfocatura gaussiana)**
- *Cos'è:* Un filtro che sfoca l'immagine usando una funzione a campana (gaussiana). Ogni pixel viene mescolato con i suoi vicini
- *Cosa fa:* Rimuove i dettagli fini e le high frequency (bordi netti), lasciando solo le strutture più grandi

**SVD (Singular Value Decomposition)**
- *Cos'è:* Decomposizione in valori singolari. Un modo di scomporre una trasformazione lineare in componenti base
- *Cosa fa:* Rivela perché il problema inverso amplifica il rumore: quando sigma (valore singolare) è piccolo, il fattore 1/sigma esplode moltiplicando il rumore

**Esempio:** Avete scritto una lettera a matita. Passate la gomma (sfocatura) e versate del tè sopra (rumore). Ora rileggete. Se provate a "invertire" la gomma, le macchie di tè diventano ancora più grandi. Dovete usare un metodo intelligente che indovini cosa c'era scritto basandosi su come sono fatte di solito le lettere. Quel "come sono fatte di solito" è il prior.

**Riferimento teorico:** Hadamard (1902) ha definito le tre condizioni per un problema ben posto. Il nostro problema non le soddisfa. La SVD mostra che per valori singolari piccoli, il fattore 1/sigma amplifica il rumore in modo catastrofico.

---

## Slide 3 — I tre metodi scelti

**Sulla slide:** Tre colonne con immagini: TV (Variazionale), UNet (Deep Learning), DiffPIR (Generativo).

**Concetti chiave:**

**Metodo variazionale**
- *Cos'è:* Un approccio che scrive una formula matematica esplicita (funzionale) e la minimizza per trovare la soluzione
- *Cosa fa:* Definisce a mano una funzione di costo che bilancia fedeltà ai dati e regolarità. La soluzione è il punto di minimo di questa funzione

**Deep Learning**
- *Cos'è:* Un ramo dell'intelligenza artificiale che usa reti neurali con molti strati per imparare dai dati
- *Cosa fa:* Mostra al computer migliaia di esempi (coppie sporco/pulito) e lui impara autonomamente la relazione tra input e output

**Metodo generativo**
- *Cos'è:* Un approccio che impara la distribuzione statistica dei dati puliti e poi genera nuove immagini simili
- *Cosa fa:* Sa come sono fatte le immagini pulite in generale e le adatta al dato specifico, ricostruendo anche dettagli completamente persi

**Total Variation (TV)**
- *Cos'è:* Una misura della "quantità di variazione" di un'immagine, definita come la somma delle differenze tra pixel adiacenti
- *Cosa fa:* Penalizza le oscillazioni (rumore) ma permette i salti netti (bordi). Converte il restauro in un problema di minimizzazione

**UNet**
- *Cos'è:* Una rete neurale a forma di "U" che comprime e poi espande l'immagine, con connessioni che preservano i dettagli
- *Cosa fa:* Prende un'immagine degradata in input e produce l'immagine restaurata in output, imparando dai dati di training

**DiffPIR (Diffusion Plug-and-Play Image Restoration)**
- *Cos'è:* Un metodo del 2023 che integra un modello di diffusione (DDPM) dentro un framework Plug-and-Play
- *Cosa fa:* Alterna passi di denoising generativo (DDPM) e passi di fedeltà ai dati (risolti con FFT), combinando creatività e precisione

**Lambda (λ)**
- *Cos'è:* Un parametro numerico che controlla quanto è forte la regolarizzazione rispetto alla fedeltà ai dati
- *Cosa fa:* Lambda=0.005 è il valore ottimale trovato per il nostro problema. Più alto = immagini più lisce. Più basso = più dettagli ma anche più rumore

**Esempio:** Tre cuochi con lo stesso piatto bruciato: TV ha una ricetta scritta (formula matematica) e segue le istruzioni alla lettera. UNet ha cucinato questo piatto mille volte e sa già come aggiustarlo a occhio. DiffPIR sa cucinare qualsiasi piatto da zero e ricostruisce gli ingredienti mancanti come meglio crede.

---

## Slide 4 — Il dataset: Mendeley LBC Cervical Cancer

**Sulla slide:** Esempio immagine, tabella classi.

**Concetti chiave:**

**Mendeley**
- *Cos'è:* Una piattaforma online dove i ricercatori condividono dataset scientifici
- *Cosa fa:* Rende disponibili le 962 immagini di cellule cervicali che abbiamo usato nel progetto

**LBC (Liquid-Based Cytology)**
- *Cos'è:* Una tecnica di preparazione dei vetrini per l'esame delle cellule cervicali, più pulita e uniforme del metodo tradizionale
- *Cosa fa:* Produce immagini più chiare e standardizzate, riducendo artefatti e sovrapposizioni cellulari

**Classi diagnostiche (NILM, LSIL, HSIL, SCC)**
- *Cos'è:* Categorie che vanno da normale (NILM) a lesioni sempre più gravi fino a tumore (SCC)
- *Cosa fa:* Descrivono il tipo di cellula. Noi non classifichiamo — ripariamo — quindi la classe non influenza il restauro

**256×256 pixel**
- *Cos'è:* La risoluzione a cui abbiamo ridotto le immagini originali (che erano 2048×1536)
- *Cosa fa:* Bilanciamento tra dettaglio visibile (i nuclei cellulari sono ancora chiari) e velocità di elaborazione su CPU

**Normalizzazione [-1, 1]**
- *Cos'è:* Trasformazione che porta i valori dei pixel da [0, 255] a [-1, 1]
- *Cosa fa:* Centra i dati attorno allo zero, condizione necessaria per il DDPM (che assume rumore a media zero) e stabilizza l'addestramento delle reti neurali

**Esempio:** Immaginate di avere 962 fototessere. Alcune sono nitide, altre mosse. Volete insegnare a un programma a sistemarle tutte. Le fototessere sono grandi 20×15 cm — troppo grandi. Le riducete a 3×3 cm (256×256). I dettagli importanti (occhi, naso) si vedono ancora, ma ora ci potete lavorare velocemente.

---

## Slide 5 — Preprocessing e split

**Sulla slide:** Tabella pipeline, tabella split.

**Concetti chiave:**

**Preprocessing**
- *Cos'è:* L'insieme delle operazioni preliminari fatte sui dati prima di applicare i metodi di restauro
- *Cosa fa:* Prepara le immagini crude per essere processate: ridimensionamento, conversione in tensori, normalizzazione

**Resize bilineare**
- *Cos'è:* Un algoritmo di ridimensionamento delle immagini che calcola i nuovi pixel come media pesata dei 4 pixel originali più vicini
- *Cosa fa:* Riduce l'immagine da 2048×1536 a 256×256 preservando la struttura, senza creare artefatti a scalini

**ToTensor**
- *Cos'è:* Conversione dei dati in tensori, il formato numerico che le reti neurali e PyTorch elaborano
- *Cosa fa:* Trasforma una matrice di pixel interi (0-255) in un tensore di numeri decimali (0.0-1.0) a 32 bit

**Split stratificato**
- *Cos'è:* Divisione dei dati in training/validazione/test che preserva le proporzioni delle classi in ogni gruppo
- *Cosa fa:* Garantisce che ogni gruppo abbia la stessa percentuale di NILM, LSIL, HSIL, SCC. Senza stratificazione, potrebbe capitare che tutte le SCC vadano nel test

**Training/Validazione/Test (70/15/15)**
- *Cos'è:* Divisione standard dei dati: 70% per imparare, 15% per regolare i parametri, 15% per la valutazione finale
- *Cosa fa:* 673 immagini per il training, 144 per la validazione, 145 per il test. Il test viene usato UNA SOLA volta, alla fine

**Seed 42**
- *Cos'è:* Un numero fisso che inizializza il generatore di numeri casuali
- *Cosa fa:* Garantisce che tutti i processi casuali (split, rumore) siano identici ogni volta. Riproducibilità: chiunque esegua il codice ottiene gli stessi risultati

**Esempio:** Avete 962 foto di cani e gatti. Per insegnare a un programma a riconoscerli, ne usate 674 per studiare (compiti a casa), 144 per le verifiche intermedie, e le ultime 145 per l'esame finale. Se lo split fosse casuale, potrebbero capitare tutti i gatti nell'esame finale. Per questo lo split è stratificato: manteniamo la stessa percentuale di cani e gatti in ogni gruppo.

---

## Slide 6 — La degradazione

**Sulla slide:** Due colonne blur/rumore, pipeline, strip immagini degradate.

**Concetti chiave:**

**Degradazione**
- *Cos'è:* Il processo artificiale che rovina un'immagine pulita per simulare un'acquisizione reale imperfetta
- *Cosa fa:* Applica sfocatura gaussiana + rumore gaussiano additivo (AWGN) in sequenza. Usiamo 4 livelli di rumore per vedere come i metodi si comportano in condizioni diverse

**Kernel 9×9**
- *Cos'è:* Una matrice quadrata 9×9 di numeri che definisce come ogni pixel si mescola con i vicini durante la sfocatura
- *Cosa fa:* Più grande è il kernel, più lontani arrivano gli effetti della sfocatura. 9×9 copre un'area di 9 pixel attorno a ciascun pixel

**Blur gaussiano con sigma=2**
- *Cos'è:* Una sfocatura a forma di campana, dove sigma controlla la larghezza della campana
- *Cosa fa:* Sigma=2 significa che la sfocatura si estende per circa 4-5 pixel in ogni direzione, simulando una messa a fuoco imperfetta

**4 livelli di sigma (0.005, 0.01, 0.05, 0.1)**
- *Cos'è:* Quattro intensità di rumore, da quasi impercettibile a molto forte
- *Cosa fa:* Permettono di studiare come ogni metodo si comporta al variare del rumore. 0.005 ≈ deblur puro (rumore trascurabile). 0.1 ≈ il rumore domina sulla sfocatura

**Pipeline**
- *Cos'è:* Una sequenza fissa di operazioni applicate a ogni immagine
- *Cosa fa:* Garantisce che TUTTI i metodi vedano le stesse identiche immagini degradate. Confronto equo

**Esempio:** Prendete 4 copie della stessa foto. Sulla prima versate una goccia d'acqua (σ=0.005), sulla seconda mezzo bicchiere (σ=0.01), sulla terza un bicchiere pieno (σ=0.05), sulla quarta immergetela nella vasca da bagno (σ=0.1). Poi chiamate tre restauratori diversi e vedete chi la recupera meglio a ogni livello di bagnato.

**Riferimento teorico:** Il kernel gaussiano è separabile: la sfocatura 2D si scompone in due sfocature 1D (righe poi colonne). Invece di 9×9=81 operazioni per pixel, ne servono 9+9=18. Fattore 4.5× più veloce.

---

## Slide 7 — TV: Teoria

**Sulla slide:** Formula TV, formula TV(x), spiegazione L1 vs Tikhonov, parametri.

**Concetti chiave:**

**TV (Total Variation)**
- *Cos'è:* La somma delle differenze assolute tra pixel adiacenti in un'immagine
- *Cosa fa:* Misura quanto un'immagine "varia". Immagini uniformi hanno TV bassa. Immagini con molti bordi hanno TV alta. Minimizzare la TV significa rendere l'immagine "a tratti costanti"

**Data fidelity (fedeltà ai dati)**
- *Cos'è:* Il termine ||Hx - y||² che misura quanto l'immagine ricostruita, dopo essere stata sfocata, assomiglia all'osservazione
- *Cosa fa:* Costringe la soluzione a spiegare i dati. Se l'immagine ricostruita è troppo diversa, questo termine aumenta

**Regularization (regolarizzazione)**
- *Cos'è:* Il termine λ·TV(x) che aggiunge informazione extra per rendere il problema ben posto
- *Cosa fa:* Dice "l'immagine deve essere regolare". Bilanciata con la fedeltà ai dati, impedisce soluzioni troppo rumorose o instabili

**L1 norm vs L2 norm (Tikhonov)**
- *Cos'è:* Due modi di misurare l'errore. L1 usa il valore assoluto, L2 usa il quadrato
- *Cosa fa:* L1 (TV) preserva i bordi perché tratta un bordo netto e una rampa graduale allo stesso modo. L2 (Tikhonov) preferisce le rampe (bordi sfumati) perché penalizza meno i gradienti piccoli. L1 rende i gradienti "sparsi" — pochi valori grandi (bordi), molti zero (sfondo)

**Staircasing (effetto a scalini)**
- *Cos'è:* Un artefatto della TV che crea "gradini" invece di transizioni graduali
- *Cosa fa:* Quando λ è troppo alto, l'immagine diventa "a blocchi" o "a mosaico", come un disegno fatto su un foglio a quadretti

**Convesso**
- *Cos'è:* Proprietà matematica di una funzione per cui ha un solo minimo globale
- *Cosa fa:* Garantisce che qualunque sia il punto di partenza, l'algoritmo di ottimizzazione troverà sempre la stessa soluzione ottima. Le reti neurali NON hanno questa proprietà

**Esempio:** Prendete una foto di un gatto. Il bordo tra gatto e sfondo è netto. Con L2 (Tikhonov) il bordo diventa sfumato — il gatto si scioglie nello sfondo. Perché L2 penalizza tanto i gradienti grandi, quindi preferisce "spalmare" il bordo. Con L1 (TV) il bordo resta netto — L1 non ha motivo di preferire la rampa. È come la differenza tra tagliare con forbici affilate (L1) o con forbici smussate (L2).

**Riferimento teorico:** Interpretazione Bayesiana di Rudin-Osher-Fatemi (1992): i gradienti dell'immagine seguono una distribuzione di Laplace, che favorisce valori piccoli ma permette picchi grandi (bordi). La funzione è convessa — minimo globale unico garantito — proprietà che le reti neurali non hanno.

---

## Slide 8 — TV: Algoritmo

**Sulla slide:** Pseudo-codice, dettagli ottimizzazione, tabella scelta lambda.

**Concetti chiave:**

**Inizializzazione x = y**
- *Cos'è:* Il punto di partenza dell'ottimizzazione: l'immagine degradata stessa
- *Cosa fa:* Parte dall'immagine rovinata e la modifica gradualmente, passo dopo passo, per migliorarla

**Iterazione**
- *Cos'è:* Una singola ripetizione del ciclo di ottimizzazione
- *Cosa fa:* Ogni iterazione calcola il gradiente, aggiorna l'immagine e la clamp. Dopo 150 iterazioni il risultato converge (non migliora più significativamente)

**Adam (Adaptive Moment Estimation)**
- *Cos'è:* Un ottimizzatore che adatta il passo di apprendimento per ogni singolo parametro (pixel)
- *Cosa fa:* Accelera nelle zone uniformi (dove il gradiente è piccolo), rallenta ai bordi (dove il gradiente è grande). È come un'auto che accelera in autostrada e frena in curva

**Backpropagation**
- *Cos'è:* Algoritmo che calcola quanto ogni pixel contribuisce all'errore totale
- *Cosa fa:* Propaga l'errore all'indietro attraverso il modello, calcolando per ogni pixel la direzione e l'intensità della modifica necessaria

**Clamp [-1, 1]**
- *Cos'è:* Operazione che taglia i valori dei pixel all'intervallo consentito
- *Cosa fa:* Dopo ogni aggiornamento, riporta i pixel che escono da [-1,1] al limite più vicino. Impedisce valori fuori range che romperebbero il modello

**Esempio:** È come scolpire una statua partendo da un blocco di marmo già un po' rovinato. A ogni passaggio (iterazione) togliete un po' di marmo. Adam è come uno scalpello intelligente: toglie tanto dove il marmo è uniforme (zone piatte), toglie poco dove ci sono dettagli delicati (bordi). Dopo 150 passaggi la statua è pronta. Se togliete troppo (lambda alto) la statua diventa un cubo (staircasing). Se togliete troppo poco (lambda basso) restano le schegge (rumore residuo).

**Riferimento teorico:** Il gradiente di TV ha un comportamento particolare: vale 0 nelle zone uniformi e ha salti discreti (±1) ai bordi. Adam adatta il passo di apprendimento per ogni parametro usando la media mobile del gradiente e del suo quadrato. SGD (discesa del gradiente stocastica) con passo fisso sarebbe subottimale: convergerebbe lentamente nelle zone piatte o oscillerebbe ai bordi.

---

## Slide 9 — TV: Risultati

**Sulla slide:** Tabella PSNR/SSIM, immagine qualitativa.

**Concetti chiave:**

**PSNR (Peak Signal-to-Noise Ratio)**
- *Cos'è:* Una metrica che confronta l'immagine ricostruita con l'originale pixel per pixel, in scala logaritmica
- *Cosa fa:* PSNR alto (>30 dB) = buona qualità. PSNR basso (<20 dB) = scarsa qualità. Ogni 3 dB significa che l'errore quadratico medio si è dimezzato

**SSIM (Structural Similarity Index)**
- *Cos'è:* Una metrica che confronta l'immagine ricostruita con l'originale in finestre locali 11×11, valutando luminanza, contrasto e struttura
- *Cosa fa:* Più allineato con la percezione umana del PSNR. SSIM=1 identiche, SSIM=0 completamente diverse. Lo staircasing fa crollare SSIM anche se PSNR è decente, perché la struttura locale è sbagliata

**dB (decibel)**
- *Cos'è:* Unità di misura logaritmica usata per PSNR e rapporti segnale-rumore
- *Cosa fa:* Una scala logaritmica comprime numeri grandi. 30 dB significa errore 1000× più piccolo del segnale. 20 dB = 100× più piccolo

**Esempio:** Avete una foto di famiglia sgranata. TV la ripulisce bene (PSNR 32 dB). Ma se aumentate il rumore, la foto diventa "a mosaico": le guance lisce sembrano fatte di quadratini (SSIM 0.586). PSNR dice "i pixel sono simili", SSIM dice "ma la disposizione è sbagliata". PSNR controlla se avete i mattoni giusti, SSIM controlla se la casa è costruita bene. TV a sigma=0.1 ha i colori giusti (PSNR 26.54) ma la struttura a blocchi (SSIM 0.586).

**Riferimento teorico:** SSIM si calcola su finestre 11×11 e confronta tre componenti: luminanza (media), contrasto (varianza), struttura (covarianza). PSNR = 10·log₁₀(MAX²/MSE). Per i nostri dati con MAX=2 (range [-1,1]), PSNR = 10·log₁₀(4/MSE).

---

## Slide 10 — UNet: Architettura

**Sulla slide:** Diagramma a U, canali, skip connections, dettagli architettura.

**Concetti chiave:**

**Encoder**
- *Cos'è:* La parte della UNet che comprime l'immagine per estrarre le caratteristiche semantiche
- *Cosa fa:* Dimezza la risoluzione spaziale e raddoppia i canali a ogni livello. Parte da 16 canali e arriva a 256. Impara caratteristiche sempre più astratte: prima bordi semplici, poi texture, poi forme complesse

**Decoder**
- *Cos'è:* La parte della UNet che ricostruisce l'immagine dalle caratteristiche estratte
- *Cosa fa:* Raddoppia la risoluzione spaziale e dimezza i canali a ogni livello. Ricostruisce l'immagine finale con la stessa risoluzione dell'input (256×256)

**Skip connection (connessione di salto)**
- *Cos'è:* Un collegamento diretto che porta i dettagli fini dall'encoder al decoder allo stesso livello di risoluzione
- *Cosa fa:* Senza skip connection, il decoder saprebbe COSA c'è nell'immagine ma non DOVE. Le skip connection preservano la localizzazione spaziale dei dettagli (nuclei, bordi cellulari)

**DoubleConv**
- *Cos'è:* Blocco base della UNet: due convoluzioni 3×3 in sequenza, ognuna seguita da GroupNorm e ReLU
- *Cosa fa:* Ogni blocco impara features più complesse combinando informazioni da pixel vicini. Due convoluzioni permettono di catturare pattern più ricchi di una singola

**GroupNorm**
- *Cos'è:* Una tecnica di normalizzazione che divide i canali in gruppi e normalizza ogni gruppo a media zero e varianza uno
- *Cosa fa:* Funziona indipendentemente dalla batch size. BatchNorm richiederebbe batch grandi (32+) per stimare bene media e varianza; con batch 16, la stima sarebbe rumorosa

**Collo di bottiglia (bottleneck)**
- *Cos'è:* Il punto più stretto della UNet, con 256 canali a risoluzione 16×16
- *Cosa fa:* Ha la massima capacità rappresentativa (256 canali) ma la minima risoluzione spaziale. Rappresenta l'informazione semantica più astratta. Da qui, il decoder ricostruisce

**1.9 milioni di parametri**
- *Cos'è:* Il numero totale di pesi numerici che la rete impara durante il training
- *Cosa fa:* Definisce la capacità della rete. Pochi parametri = non impara abbastanza. Troppi = overfitting. 1.9M è un buon compromesso per questo problema

**Input a 4 canali**
- *Cos'è:* L'input della UNet è composto da 3 canali RGB + 1 canale aggiuntivo per la mappa del rumore
- *Cosa fa:* Il 4° canale dice alla rete "quanto rumore c'è in questa immagine". Permette alla rete di comportarsi diversamente in base al livello di rumore

**Esempio:** Immaginate di dover ridisegnare un ritratto partendo da uno schizzo sporco. Prima lo guardate da lontano per capire la struttura generale (encoder). Poi lo ridisegnate nei dettagli (decoder). Ma mentre ridisegnate, tenete sempre lo schizzo originale accanto (skip connection) per copiare i dettagli precisi: la forma degli occhi, la piega dei capelli. Senza lo schizzo, disegnereste una faccia generica. Con lo schizzo, i dettagli tornano al loro posto.

---

## Slide 11 — UNet: Training

**Sulla slide:** Tabella parametri, pseudo-codice training.

**Concetti chiave:**

**Loss L1 (MAE — Mean Absolute Error)**
- *Cos'è:* Funzione di perdita che calcola la differenza assoluta media tra pixel dell'immagine ricostruita e dell'originale
- *Cosa fa:* Penalizza l'errore linearmente. Preserva i bordi perché non amplifica gli errori grandi (come farebbe MSE). L'errore su un bordo netto è uguale all'errore su una zona uniforme

**MSE (L2 — Mean Squared Error)**
- *Cos'è:* Funzione di perdita che calcola la differenza quadratica media
- *Cosa fa:* Penalizza l'errore quadraticamente. Un errore doppio viene penalizzato 4 volte di più. Porta a soluzioni conservative (immagini più sfocate). L'ottimo di MSE è la media condizionale — che è sempre più liscia del vero valore

**Learning rate (tasso di apprendimento)**
- *Cos'è:* Quanto grande è ogni passo di aggiornamento dei parametri della rete
- *Cosa fa:* 10⁻⁴ = 0.0001 è un passo piccolo e sicuro. Troppo grande (es. 0.01) fa divergere l'addestramento. Troppo piccolo (es. 10⁻⁶) rende l'apprendimento lentissimo

**Batch (lotto)**
- *Cos'è:* Il numero di immagini elaborate simultaneamente in una singola iterazione di training
- *Cosa fa:* Batch=16 significa che la rete vede 16 immagini alla volta, calcola l'errore medio su tutte, e aggiorna i pesi. Batch più grande = stima del gradiente più accurata, ma più memoria necessaria

**Epoca**
- *Cos'è:* Un passaggio completo attraverso tutto il dataset di training
- *Cosa fa:* 50 epoche = la rete vede ogni immagine 50 volte. Dopo 50 epoche, la loss sul validation set smette di migliorare (convergenza)

**Multi-noise augmentation**
- *Cos'è:* Tecnica di addestramento che a ogni batch sceglie casualmente uno dei 4 livelli di rumore
- *Cosa fa:* La rete impara a gestire tutti i livelli di rumore contemporaneamente. Diventa robusta — non specializzata su un singolo livello. Senza, la rete sarebbe brava solo a un rumore specifico

**Forward pass**
- *Cos'è:* Il passaggio in avanti: l'immagine entra nella rete, passa attraverso tutti gli strati, ed esce il risultato
- *Cosa fa:* Produce l'immagine ricostruita a partire da quella degradata. È la fase di "inferenza" della rete

**Backward pass (backpropagation)**
- *Cos'è:* Il passaggio all'indietro: l'errore viene propagato dalla fine all'inizio della rete
- *Cosa fa:* Calcola quanto ogni parametro ha contribuito all'errore, e in che direzione modificarlo per ridurlo. È il cuore dell'apprendimento

**Checkpoint (punto di controllo)**
- *Cos'è:* Il salvataggio del modello quando raggiunge le migliori prestazioni sulla validazione
- *Cosa fa:* Conserva la versione migliore della rete. Se dopo 50 epoche la rete peggiora (overfitting), teniamo la versione precedente che era migliore

**Esempio:** È come imparare ad aggiustare foto con un tutor. Epoca 1: il tutor vi mostra 673 foto rovinate con la soluzione. Voi provate, sbagliate, lui vi corregge. Epoca 2: rifate tutto da capo. Ora sbagliate meno. Dopo 50 volte che avete visto ogni foto, siete bravi. Il multi-noise è il tutor che ogni volta vi dà una foto con sporco diverso: oggi polvere, domani acqua — così imparate a gestire TUTTO.

**Riferimento teorico:** L1 stima la mediana condizionale (preserva i bordi). MSE stima la media condizionale (tende a sfumare). Con 4 livelli di downsampling (stride 2 ciascuno), il collo di bottiglia della UNet a 16×16 ha un campo recettivo che copre l'intera immagine 256×256. La rete ha contesto globale.

---

## Slide 12 — UNet: Risultati

**Sulla slide:** Tabella PSNR/SSIM/tempo.

**Concetti chiave:**

**Stabilità (degrado di ~1 dB)**
- *Cos'è:* La capacità del modello di mantenere prestazioni simili a tutti i livelli di rumore
- *Cosa fa:* UNet varia solo 1.3 dB tra rumore basso (29.79 dB) e alto (28.46 dB). TV varia 5.5 dB. UNet è molto più robusta grazie al multi-noise augmentation

**Inflection point (punto di svolta)**
- *Cos'è:* Il livello di rumore (σ≈0.05) dove UNet supera TV in qualità
- *Cosa fa:* Segna il confine tra due regimi: sotto, il prior scritto a mano (TV) è sufficiente. Sopra, il prior imparato dai dati (UNet) diventa migliore

**Velocità di inferenza (0.03 secondi)**
- *Cos'è:* Il tempo necessario per processare una singola immagine dopo che la rete è stata addestrata
- *Cosa fa:* UNet fa tutto in una singola forward pass: l'immagine entra e l'immagine restaurata esce. TV fa 150 iterazioni, DiffPIR fa 15 passi. UNet è 200× più veloce di TV e 100× più veloce di DiffPIR

**Esempio:** UNet è come un autista medio che guida bene sia col sole che col temporale. TV è come un pilota professionista che sul bagnato sbanda. Il pilota (TV) è meglio col sole (basso rumore), ma l'autista medio (UNet) è più affidabile quando piove. E UNet ci mette 0.03 secondi a decidere come sterzare, TV ci mette 7 secondi — alla velocità di un'auto, 7 secondi sono un'eternità.

---

## Slide 13 — DiffPIR: Panoramica

**Sulla slide:** Blocco blue DiffPIR, blocco arancione LightUNet.

**Concetti chiave:**

**DiffPIR (Diffusion Plug-and-Play Image Restoration)**
- *Cos'è:* Metodo del 2023 (Zhu et al., CVPR) che combina un modello di diffusione con un framework Plug-and-Play per il restauro immagini
- *Cosa fa:* Alterna due passi: (1) denoising generativo con DDPM per rimuovere il rumore, (2) data fidelity risolta con FFT per garantire coerenza con l'immagine degradata. È modulare: puoi cambiare il denoiser senza modificare il resto

**DDPM (Denoising Diffusion Probabilistic Model)**
- *Cos'è:* Modello generativo che impara a rimuovere gradualmente il rumore da un'immagine, passando da rumore puro a immagine pulita in 1000 passi
- *Cosa fa:* Durante il training, impara a predire il rumore presente a ogni passo. Durante l'inferenza, parte da un'immagine rumorosa e la ripulisce progressivamente. È la tecnologia alla base di Dall-E e Stable Diffusion

**LightUNet**
- *Cos'è:* Una versione leggera della UNet con soli 1.26 milioni di parametri (5 MB)
- *Cosa fa:* Funge da "denoiser" all'interno del DDPM. Predice il rumore nell'immagine a ogni timestep. È specificamente addestrata su immagini di cellule cervicali, non su immagini generiche di Internet

**PnP (Plug-and-Play) framework**
- *Cos'è:* Un framework di ottimizzazione che separa il problema in due parti: un denoiser e una proiezione sui dati
- *Cosa fa:* Permette di "attaccare" qualsiasi denoiser (TV, UNet, DDPM) allo stesso modello di restauro. È come un impianto stereo dove puoi cambiare le casse senza cambiare l'amplificatore

**Esempio:** DiffPIR funziona come un restauratore con due strumenti: una gomma magica (DDPM) che toglie lo sporco, e una lente d'ingrandimento (data fidelity) che controlla se il risultato corrisponde alla foto originale rovinata. Li usa in alternanza: gomma, lente, gomma, lente, per 15 cicli. Il nostro restauratore ha studiato solo su foto di cellule cervicali (LightUNet specializzata), non su foto generiche.

---

## Slide 14 — DDPM e LightUNet

**Sulla slide:** Dettagli DDPM e LightUNet, formule forward/reverse.

**Concetti chiave:**

**1000 timestep (passi temporali)**
- *Cos'è:* Una sequenza di 1000 passi in cui si trasforma gradualmente un'immagine pulita in rumore puro
- *Cosa fa:* Ogni passo aggiunge un po' di rumore. Dopo 1000 passi, l'immagine non è più riconoscibile — solo rumore gaussiano. Durante il reverse, il modello impara a fare il percorso inverso

**Forward process (processo diretto)**
- *Cos'è:* La fase in cui si aggiunge rumore all'immagine, passando da x₀ (pulita) a x₁₀₀₀ (rumore puro)
- *Cosa fa:* Definisce come l'immagine si degrada progressivamente. È una catena di Markov: ogni passo dipende solo dal precedente. È FISSO — non si impara, si definisce a priori

**Reverse process (processo inverso)**
- *Cos'è:* La fase in cui si rimuove il rumore, passando da x₁₀₀₀ (rumore) a x₀ (pulita)
- *Cosa fa:* È quello che il modello impara a fare. Parte dal rumore puro e, passo dopo passo, predice come togliere il rumore per ricostruire l'immagine. È l'opposto del forward

**Predire il rumore (invece dell'immagine)**
- *Cos'è:* Il modello impara a prevedere QUANTO rumore è stato aggiunto a ogni passo, non l'immagine pulita finale
- *Cosa fa:* È molto più facile! Il rumore ha una distribuzione nota (gaussiana standard, media zero, varianza uno). L'immagine ha una distribuzione complessa e sconosciuta. È come insegnare a togliere macchie (facile) invece che a ridipingere (difficile)

**Time embedding sinusoidale**
- *Cos'è:* Una codifica del numero del timestep (0-1000) usando funzioni seno e coseno a diverse frequenze
- *Cosa fa:* Dice al modello "a che passo siamo". La stessa tecnica usata nei Transformers. A passi bassi (poco rumore) il modello si comporta come un raffinatore di dettagli. A passi alti (tanto rumore) come un generatore creativo

**DDIM (Denoising Diffusion Implicit Models)**
- *Cos'è:* Una variante del DDPM che rende il processo di denoising deterministico invece che stocastico
- *Cosa fa:* Permette di fare 15 passi invece di 1000 con la stessa qualità. È 60× più veloce. Risultato: stesso punto di arrivo, percorso diverso. Il modello non cambia — basta cambiare il campionatore

**Score matching**
- *Cos'è:* La teoria che collega la predizione del rumore al gradiente della densità di probabilità dei dati
- *Cosa fa:* Imparare a predire il rumore equivale a imparare ∇log p(x) — la direzione in cui la probabilità dei dati aumenta più velocemente. È il "gradiente" che punta verso immagini realistiche

**Esempio:** Prendete una foto nitida. Sopra mettete un foglio di carta da lucido leggermente opaco (passo 1). Ancora un foglio (passo 2). Dopo 1000 fogli, non si vede più niente. Il modello impara a togliere i fogli uno a uno. Ma togliere 1000 fogli è lento. DDIM insegna al modello a toglierne 15 alla volta — stesso risultato, 60× più veloce. Predire il rumore invece dell'immagine è come insegnare a togliere macchie da un vestito invece di ricucire il vestito da zero.

**Riferimento teorico:** Ho et al. (2020): la forma chiusa x_t = √ᾱ·x₀ + √(1-ᾱ)·ε permette di saltare direttamente a qualsiasi t senza iterare. Predire ε equivale a score matching: ε_θ ≈ -∇log p(x_t). Song et al. (2021): DDIM rimuove la componente stocastica, permettendo di subsample i timestep (15 invece di 1000) senza perdita di qualità.

---

## Slide 15 — DiffPIR: Algoritmo

**Sulla slide:** Pseudo-codice a sinistra, formule a destra.

**Concetti chiave:**

**FFT (Fast Fourier Transform)**
- *Cos'è:* Un algoritmo veloce che trasforma un'immagine dal dominio spaziale (pixel) al dominio delle frequenze
- *Cosa fa:* Nel dominio delle frequenze, la sfocatura diventa una semplice moltiplicazione punto a punto invece di una convoluzione complicata. Permette di risolvere il problema di data fidelity in O(N log N) invece di O(N³)

**Data fidelity (via FFT)**
- *Cos'è:* Il passo che costringe l'immagine ricostruita a essere compatibile con l'osservazione degradata, risolto in frequenza
- *Cosa fa:* Combina l'estimato del DDPM con l'immagine degradata, pesandoli in base alla loro affidabilità. La formula è come un Wiener filter: dà più peso al dato dove il kernel di blur ha più energia (basse frequenze), più peso al prior dove il kernel è debole (alte frequenze)

**rho_t (ρ_t) dinamico**
- *Cos'è:* Un peso che cambia a ogni timestep per bilanciare l'influenza del prior generativo e della fedeltà ai dati
- *Cosa fa:* All'inizio (t alto, tanto rumore) ρ_t è grande: la fedeltà ai dati domina. Alla fine (t basso, pochi dettagli) ρ_t è piccolo: il prior generativo domina. È un arbitro che cambia le regole durante la partita

**t_start=50**
- *Cos'è:* Il timestep da cui iniziamo la diffusione inversa, invece di partire da t=1000 (rumore puro)
- *Cosa fa:* Partiamo da un'immagine già degradata (non da rumore puro). A t=50, l'immagine è ancora al 97% integra (ᾱ≈0.97). L'amplificazione dell'errore è solo del 3%. A t=1000 sarebbe 150×. È numericamente stabile

**Wiener filter**
- *Cos'è:* Un filtro ottimo nel dominio delle frequenze che minimizza l'errore quadratico medio tra stima e vero valore
- *Cosa fa:* La formula di data fidelity di DiffPIR è una versione del filtro di Wiener, dove ρ_t agisce come termine di regolarizzazione. Dove il blur è forte (basse frequenze) si fida dei dati. Dove il blur è debole (alte frequenze) si fida del prior

**DPS (Diffusion Posterior Sampling) vs DiffPIR**
- *Cos'è:* Due approcci diversi per usare la diffusione nei problemi inversi
- *Cosa fa:* DPS modifica la traiettoria di diffusione aggiungendo un gradiente della verosimiglianza (costoso, richiede autograd). DiffPIR alterna denoising e data-fidelity in modo indipendente (modulare, senza autograd). DPS è più accurato ma più lento. DiffPIR è più efficiente

**Esempio:** rho_t è come un arbitro di calcio che cambia comportamento durante la partita. All'inizio (t alto, tanto rumore) l'arbitro fischia poco: lascia giocare il modello generativo. Alla fine (t basso, pochi dettagli) fischia molto: pretende che i dati siano rispettati. t_start=50: è come iniziare una maratona al km 42 invece che al km 0. Tanto la parte iniziale (0→42) è solo rumore — non ci serve.

**Riferimento teorico:** La FFT risolve il problema di data fidelity in O(N log N) invece di O(N³) — differenza enorme (256²=65.536 vs 256⁶=2.8×10¹⁴). t_start=50: ᾱ₅₀≈0.97, 1/√ᾱ≈1.03. A t=1000: ᾱ≈4.5×10⁻⁵, 1/√ᾱ≈150. DPS usa ∇_x_t ||H·x̂₀(x_t) - y||² (richiede backprop), DiffPIR risolve (HᵀH+ρI)x = Hᵀy+ρ·x̂₀ con FFT (analitico).

---

## Slide 16 — DiffPIR: Risultati

**Sulla slide:** Tabella parametri, tabella risultati, immagine, osservazioni.

**Concetti chiave:**

**Hallucination (allucinazione)**
- *Cos'è:* Quando il modello generativo aggiunge dettagli realistici ma NON presenti nell'immagine originale
- *Cosa fa:* A σ=0.005, l'immagine è quasi pulita, ma DiffPIR ci "inventa sopra" texture, bordi, strutture che sembrano reali ma sono false (PSNR 15.78). È il costo di un prior generativo troppo forte

**Andamento invertito (PSNR cresce col rumore)**
- *Cos'è:* Contrariamente a TV e UNet, DiffPIR migliora quando il rumore aumenta
- *Cosa fa:* Da 15.78 dB (σ=0.005) a 25.46 dB (σ=0.1). Perché quando i dati sono molto rovinati, il prior generativo non è più un limite ma un aiuto: "riempie" i dettagli mancanti con strutture plausibili

**Bias (distorsione)**
- *Cos'è:* L'errore dovuto alle assunzioni del modello, quanto il modello è "fissato" sulle sue convinzioni
- *Cosa fa:* DiffPIR ha ALTA bias: il prior generativo è molto forte e tende a sovrascrivere i dati. TV ha bias media (assunzione piecewise-smooth). UNet ha bias bassa (impara dai dati). Bias alta è negativa quando i dati sono buoni

**Varianza**
- *Cos'è:* Quanto il risultato cambia se cambia l'input o se si ripete l'esperimento
- *Cosa fa:* DiffPIR ha BASSA varianza (DDIM deterministico — stesso input = stesso output). TV ha varianza zero (deterministico per definizione). UNet ha varianza moderata (dipende dal training). Bassa varianza è positiva quando i dati sono cattivi

**Bias-variance trade-off**
- *Cos'è:* Il compromesso fondamentale: non si possono minimizzare bias e varianza contemporaneamente
- *Cosa fa:* A basso rumore, la bias di DiffPIR domina (allucina). Ad alto rumore, la varianza degli altri metodi aumenta (rumore residuo), mentre DiffPIR con la sua bassa varianza e alta bias "regolarizza" meglio. TV ha bias media e varianza zero. UNet ha bassa bias e varianza moderata

**Esempio:** DiffPIR è come un bambino con troppa immaginazione. Gli mostrate una foto quasi perfetta: lui ci aggiunge un terzo occhio — la sua immaginazione (bias) è così forte che vuole "migliorare" anche quello che è già buono. Gli mostrate una foto molto rovinata: qui la sua immaginazione diventa un superpotere — dove gli altri vedono scarabocchi, lui ricostruisce una faccia.

**Riferimento teorico:** L'errore di ricostruzione si scompone in tre parti: Bias² (errore da assunzioni sbagliate), Varianza (sensibilità al rumore), Rumore irriducibile. DiffPIR: alta bias, bassa varianza. TV: bias media, varianza zero. UNet: bassa bias, varianza moderata. A σ=0.005: bias domina → DiffPIR peggiore. A σ=0.1: varianza altrui aumenta → DiffPIR recupera.

---

## Slide 17 — Implementazione

**Sulla slide:** Nella presentazione non c'è una slide specifica — il copione spiega la struttura del codice.

**Concetti chiave:**

**Struttura modulare**
- *Cos'è:* Ogni metodo è in una cartella separata (src/methods/tv, src/methods/unet, src/methods/diffpir)
- *Cosa fa:* Si possono modificare o aggiungere metodi indipendentemente senza rompere gli altri. Ogni modulo ha la sua logica, i suoi parametri, i suoi file

**Unit test (34 test)**
- *Cos'è:* Piccoli pezzi di codice che verificano automaticamente che ogni funzione faccia quello che deve
- *Cosa fa:* Se qualcuno modifica il codice e rompe qualcosa, i test lo segnalano immediatamente. Garantiscono che degradazione, metriche e core logic funzionino sempre correttamente

**Riproducibilità (Seed 42)**
- *Cos'è:* La proprietà per cui chiunque esegua il codice ottiene ESATTAMENTE gli stessi risultati
- *Cosa fa:* Garantita a tre livelli: (1) seed 42 per tutte le operazioni casuali, (2) split identico dei dati, (3) stesse immagini degradate per tutti i metodi. Se qualcuno esegue il codice su un altro computer, ottiene i nostri stessi numeri

**Esempio:** È come un'officina con tre banchi separati: uno per TV, uno per UNet, uno per DiffPIR. Ogni banco ha i suoi strumenti, ma tutti usano lo stesso armadietto dei pezzi (dataset) e lo stesso manuale delle misure (metriche). Se qualcuno sposta un cacciavite sul banco TV, gli altri banchi non vengono toccati. E c'è un ispettore (34 test) che ogni mattina controlla che tutti gli attrezzi funzionino.

---

## Slide 18 — Confronto PSNR/SSIM

**Sulla slide:** Tabella principale con tutti i numeri. Blocco Observations.

**Concetti chiave:**

**TV domina a σ≤0.01 (PSNR > 32 dB, SSIM > 0.9)**
- *Cos'è:* TV è il migliore a basso rumore, supera UNet di oltre 2 dB
- *Cosa fa:* Se il rumore è trascurabile, il prior TV (piecewise-smooth) è sufficiente. Non serve training — la formula scritta a mano basta e avanza. È la scelta più semplice ed efficace per questo regime

**UNet miglior trade-off (degrado solo 1 dB)**
- *Cos'è:* UNet perde solo 1.3 dB passando da σ=0.005 a σ=0.1
- *Cosa fa:* Nessun altro metodo è così stabile. UNet non vince mai in modo netto, ma è sempre secondo in ogni condizione. È la scelta più pragmatica se non si conosce il livello di rumore a priori

**DiffPIR cresce col rumore (15.78 → 25.46 dB)**
- *Cos'è:* L'unico metodo che migliora quando il rumore aumenta
- *Cosa fa:* Il prior generativo è un'arma a doppio taglio: a basso rumore peggiora (allucina), ad alto rumore aiuta (ricostruisce dettagli persi). È il metodo più interessante concettualmente ma il meno pratico in questo contesto

**Inflection point (punto di incrocio)**
- *Cos'è:* Il livello di rumore (σ≈0.05) dove TV e UNet si incrociano in SSIM (0.837 vs 0.864)
- *Cosa fa:* Segna il confine tra regimi. Sotto questo punto, il prior analitico vince. Sopra, il prior imparato dai dati vince. È la dimostrazione sperimentale che ogni prior ha il suo dominio di validità

**Esempio:** Tre ristoranti e un piatto di pasta. TV: chef stellato. Se la pasta è quasi perfetta (poco rumore), fa un piatto da 10 e lode. Se la pasta è bruciata, fa un piatto mediocre. UNet: bravo cuoco di trattoria. La pasta non è mai eccelsa ma non è mai immangiabile. DiffPIR: cuoco creativo. Se la pasta è buona, lui aggiunge ingredienti strani (allucina) e la rovina. Se è bruciata, lui la trasforma in un piatto creativo sorprendente. Quale scegliere? Dipende dalla pasta.

**Riferimento teorico:** PSNR = 10·log₁₀(MAX²/MSE). SSIM combina in finestre 11×11: luminanza (media), contrasto (varianza), struttura (covarianza). Ogni 3 dB in PSNR = metà dell'MSE.

---

## Slide 19 — Grafico comparativo

**Sulla slide:** Grafico PSNR/SSIM vs rumore.

**Concetti chiave:**

**Grafico PSNR/SSIM vs sigma rumore**
- *Cos'è:* Un grafico che mostra come PSNR e SSIM cambiano al variare del livello di rumore per i tre metodi
- *Cosa fa:* Visualizza le tendenze della tabella. TV e UNet scendono (leggermente per UNet, più per TV). DiffPIR sale. L'incrocio TV-UNet è chiaramente visibile nel grafico SSIM

**Esempio:** Il grafico è come un termometro della febbre dei metodi. TV ha la febbre che sale (peggiora) col rumore. UNet ha la febbre quasi costante. DiffPIR ha la febbre che scende (migliora) col rumore. Si mostra il grafico per far capire subito il trend senza leggere i numeri uno per uno.

---

## Slides 20-23 — Risultati qualitativi

**Sulla slide:** Immagini dei risultati a vari livelli di rumore, mappe di differenza.

**Concetti chiave:**

**Mappe di differenza (difference maps)**
- *Cos'è:* Immagini che mostrano, pixel per pixel, quanto la ricostruzione differisce dall'originale (bianco = errore grande, nero = errore piccolo)
- *Cosa fa:* Rivelano DOVE e COME ogni metodo sbaglia. TV: errori a blocchi (staircasing sistematico). UNet: errori piccoli e sparsi (uniformi, senza pattern). DiffPIR: errori concentrati sui bordi (ha spostato i contorni delle cellule)

**Artefatto**
- *Cos'è:* Un difetto visibile nell'immagine ricostruita che non era presente nell'originale
- *Cosa fa:* TV produce artefatti a scalini (staircasing). UNet produce artefatti minimi. DiffPIR produce artefatti "realistici" (hallucination) — texture che sembrano vere ma non lo sono

**Crop (ritaglio ingrandito)**
- *Cos'è:* Una porzione ingrandita dell'immagine per mostrare i dettagli a livello di pixel
- *Cosa fa:* Permette di vedere le differenze sui nuclei cellulari (la parte più importante per la diagnosi) a un livello di dettaglio che l'immagine intera non mostrerebbe

**Esempio:** È come guardare tre radiografie di un gatto: a basso rumore TV mostra un gatto perfetto, UNet un gatto buono, DiffPIR un gatto con 6 zampe (allucina). A medio rumore: TV mostra un gatto a quadretti (staircasing), UNet ancora riconoscibile, DiffPIR con occhi strani. Ad alto rumore: TV è irriconoscibile, UNet è il migliore, DiffPIR ha colori realistici ma testa leggermente diversa. Le mappe di differenza sono come le radiografie: TV ha fratture ovunque, UNet ha microfratture, DiffPIR ha fratture in punti precisi.

---

## Slide 24 — Confronto famiglie

**Sulla slide:** Tre blocchi PRO/CONTRO con vantaggi e svantaggi.

**Concetti chiave:**

**Interpretabilità**
- *Cos'è:* Quanto è chiaro e comprensibile il funzionamento interno di un metodo
- *Cosa fa:* TV è completamente interpretabile: la formula è scritta, si sa esattamente cosa fa. UNet ha bassa interpretabilità: è una scatola nera, non si sa esattamente quali caratteristiche usa. DiffPIR ha interpretabilità media: i due passi (denoising e data fidelity) sono separati e comprensibili

**Generalizzazione**
- *Cos'è:* La capacità di un metodo di funzionare su dati diversi da quelli visti durante il training
- *Cosa fa:* TV generalizza a qualsiasi immagine perché non ha training. UNet generalizza male fuori dal dominio delle immagini mediche. DiffPIR generalizza meglio perché il DDPM impara la distribuzione delle immagini pulite, non solo la mappatura sporco→pulito

**Velocità di inferenza**
- *Cos'è:* Il tempo necessario per processare una singola immagine a metodo addestrato
- *Cosa fa:* UNet (0.03 s) è 200× più veloce di TV (7 s) e 100× più veloce di DiffPIR (3 s). Per applicazioni reali (es. 10.000 immagini), UNet è l'unica scelta pratica (5 minuti contro 19 ore di TV o 8 ore di DiffPIR)

**Esempio:** È come scegliere un mezzo di trasporto. TV = bicicletta: semplice, economica, la capisci subito. Se la strada è in pianura (basso rumore) arrivi primo. Se è in salita (tanto rumore), fatichi. UNet = automobile: costa (training), ma è veloce e affronta qualsiasi strada. Non è la migliore in tutto ma è il miglior compromesso. DiffPIR = fuoristrada: supera buche e dossi (ricostruisce dettagli persi) ma consuma tanto (lento) e in autostrada (poco rumore) è rumoroso (allucina).

---

## Slide 25 — Regimi operativi

**Sulla slide:** Tabella scenari (basso/medio/alto rumore, velocità).

**Concetti chiave:**

**Regime operativo**
- *Cos'è:* L'insieme di condizioni (livello di rumore, budget di tempo, priorità di qualità) in cui un metodo deve operare
- *Cosa fa:* Determina quale metodo è più adatto. Non esiste "il metodo migliore" in assoluto — esiste il metodo più adatto per ogni regime

**Trade-off (compromesso)**
- *Cos'è:* La necessità di bilanciare obiettivi contrastanti (qualità vs velocità, interpretabilità vs potenza)
- *Cosa fa:* Scegliere un metodo significa accettare i suoi svantaggi in cambio dei suoi vantaggi. TV è lento ma non richiede training. UNet è veloce e robusto ma è una scatola nera. DiffPIR è creativo ma lento e allucina

**Esempio:** Dovete restaurare 10.000 foto storiche in un archivio. Il rumore è medio. Se usate TV (7 secondi l'una) = 19 ore. Se usate UNet (0.03 secondi l'una) = 5 minuti. Se usate DiffPIR (3 secondi l'una) = 8 ore. Per un archivio digitale, UNet è l'unica scelta pratica — la qualità è buona e il tempo è accettabile. Ma se le foto hanno rumore molto basso, TV dà qualità migliore — se avete tempo.

---

## Slide 26 — Conclusioni

**Sulla slide:** Tre blocchi: Main Results, Lessons Learned, Future Directions.

**Concetti chiave:**

**Lambda adattivo (λ(σₙ))**
- *Cos'è:* Un'idea per far sì che il parametro λ della TV cambi automaticamente in base al livello di rumore
- *Cosa fa:* Invece di usare λ=0.005 fisso per tutti i rumori, si potrebbe calcolare λ in funzione di σₙ: più rumore = λ più forte (più regolarizzazione). Migliorerebbe i risultati di TV ad alto rumore

**Validazione clinica**
- *Cos'è:* Testare i metodi su immagini reali provenienti da un ospedale, non degradate artificialmente
- *Cosa fa:* Le immagini reali hanno degradazioni più complesse di una semplice sfocatura gaussiana + AWGN. La validazione clinica direbbe se i metodi funzionano davvero in un ambiente reale

**Confronto equo (fair comparison)**
- *Cos'è:* La condizione in cui tutti i metodi vengono valutati con gli stessi dati, stesse metriche, stesse condizioni
- *Cosa fa:* Elimina fattori confondenti. Molti articoli confrontano metodi su dataset diversi, con preprocessing diversi, o metriche diverse. Il nostro setup elimina tutto questo: le differenze sono solo nei metodi

**Esempio:** Il progetto è come un test su tre medicine per lo stesso sintomo. Abbiamo dato a tutti gli stessi pazienti (stesse immagini), la stessa dose (stessa degradazione), e misurato con gli stessi strumenti. Risultato: non esiste la medicina universale. L'aspirina (TV) funziona per il mal di testa leggero, l'antibiotico (UNet) è più robusto ma va prescritto, l'omeopatia (DiffPIR) in alcuni casi funziona ma in altri inventa effetti. Il dottore sceglie in base al paziente.

---

## Slide 27 — Bibliografia

**Sulla slide:** Sei riferimenti accademici.

**Concetti chiave:**

**Riferimento bibliografico**
- *Cos'è:* Un articolo scientifico che ha introdotto un metodo fondamentale
- *Cosa fa:* Ogni riferimento è una tappa nella storia del restauro immagini. Rudin-Osher-Fatemi (TV, 1992) ha fondato l'approccio variazionale. Ronneberger (UNet, 2015) ha portato il deep learning nelle immagini mediche. Ho (DDPM, 2020) ha inventato la diffusione, Song (DDIM, 2021) l'ha resa pratica, Zhu (DiffPIR, 2023) l'ha applicata al restauro

**Esempio:** Questi sei articoli sono come i fondatori di una scuola di pensiero. Ognuno ha costruito sulle spalle del precedente. La storia della ricerca: qualcuno inventa un'idea (TV 1992), poi qualcuno la supera (UNet 2015), poi arriva un paradigma completamente nuovo (DDPM 2020), e infine qualcuno li combina (DiffPIR 2023).

Grazie per l'attenzione.
