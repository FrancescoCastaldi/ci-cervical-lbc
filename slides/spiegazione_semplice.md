# Spiegazione del progetto

> Da leggere alla prof seguendo le slide.

---

## Slide 1 — Copertina

**Sulla slide:** Titolo, nomi, logo UNIBO, link GitHub.

**Cosa significa:** Siamo due studenti che hanno fatto un progetto di restauro immagini. Abbiamo preso foto di cellule cervicali rovinate e le abbiamo riparate con tre metodi diversi per vedere qual è il migliore.

---

## Slide 2 — Il problema inverso

**Sulla slide:** Formula y = H(x) + n. Riquadro arancione "Why it is difficult".

**Cosa significa:**
- **Problema inverso:** invece di "ho una foto pulita, la rovino" (problema diretto), facciamo il contrario: "ho una foto rovinata, voglio recuperare quella pulita" (problema inverso).
- **Mal posto (ill-posed):** non si può semplicemente invertire la sfocatura perché il rumore esplode. È come cercare di rileggere un foglio scritto a matita dopo che qualcuno ci ha passato la gomma — se provi a "invertire" la gomma, ottieni solo macchie.
- **Prior:** una regola che dice "le immagini pulite di solito sono fatte così". Serve per scegliere tra tutte le possibili soluzioni.
- **AWGN:** rumore bianco gaussiano additivo. È il tipo di rumore più comune, come la "neve" sulla TV.
- **Gaussian blur:** sfocatura a forma di campana, come mettere una foto fuori fuoco.

**Riferimento teorico:** Hadamard (1902) ha definito le condizioni per un problema ben posto. Il nostro non le soddisfa. Con la SVD (decomposizione in valori singolari) si vede che quando il valore singolare sigma è piccolo, il fattore 1/sigma amplifica il rumore.

---

## Slide 3 — I tre metodi scelti

**Sulla slide:** Tre colonne con immagini: TV (Variazionale), UNet (Deep Learning), DiffPIR (Generativo).

**Cosa significa:**
- **Variazionale:** scriviamo una formula matematica a mano e la minimizziamo. TV dice "le immagini pulite hanno pochi bordi netti". Non serve training.
- **Deep Learning:** mostriamo tanti esempi al computer e lui impara da solo. UNet dice "ho visto migliaia di cellule, so come devono essere".
- **Generativo:** il modello sa generare immagini pulite da zero, come un artista. Poi le adatta ai dati che ha. DiffPIR dice "so com'è una cellula vera, la ricostruisco anche se mancano dettagli".
- **Lambda = 0.005:** parametro che dice quanto è forte la regolarizzazione. Più è alto, più l'immagine viene "lisciata".

---

## Slide 4 — Il dataset: Mendeley LBC Cervical Cancer

**Sulla slide:** Esempio immagine, tabella classi.

**Cosa significa:**
- **962 immagini:** non tantissime, ma sufficienti per un progetto su CPU. Ogni immagine è una diversa cellula cervicale.
- **256×256:** l'immagine originale era enorme (2048×1536). L'abbiamo rimpicciolita per farla elaborare al computer in tempi ragionevoli. I nuclei delle cellule (la parte importante per la diagnosi) a 256 pixel si vedono ancora bene.
- **NILM, LSIL, HSIL, SCC:** tipi di cellule dal normale al tumorale. Ma noi non classifichiamo — ripariamo. Quindi la classe non ci interessa.
- **Normalizzazione [-1,1]:** trasformiamo i pixel in numeri tra -1 e 1 perché le reti neurali funzionano meglio con numeri piccoli e bilanciati.

---

## Slide 5 — Preprocessing e split

**Sulla slide:** Tabella pipeline, tabella split.

**Cosa significa:**
- **Preprocessing:** le cose che si fanno prima di iniziare il vero lavoro. Come preparare gli ingredienti prima di cucinare.
- **Resize:** ridimensionare le immagini. Da 2048×1536 a 256×256 pixel.
- **ToTensor:** convertire in numeri che il computer può elaborare. Un'immagine in bianco e nero diventa una griglia di numeri.
- **Normalizzazione:** portare i numeri in un range uniforme [-1,1]. Se i numeri fossero tra 0 e 255, alcuni sarebbero grandi e altri piccoli — la rete farebbe fatica.
- **Split 70/15/15:** dividiamo i dati in tre gruppi. 70% per imparare (training), 15% per controllare durante l'apprendimento (validazione), 15% per il test finale.
- **Seed 42:** un numero fisso che garantisce che ogni volta lo split sia identico. Così tutti i metodi vedono le stesse immagini.
- **Stratificato:** manteniamo le stesse proporzioni di classi in ogni gruppo. Altrimenti potrebbe capitare che tutte le cellule tumorali vadano nel test.

---

## Slide 6 — La degradazione

**Sulla slide:** Due colonne blur/rumore, pipeline, strip immagini degradate.

**Cosa significa:**
- **Degradazione:** il processo che rovina l'immagine. Noi lo facciamo artificialmente perché vogliamo sapere com'era quella pulita (per confrontare).
- **Kernel 9×9:** una matrice 9×9 che dice come ogni pixel si "spalma" sui vicini. Più è grande, più la sfocatura è forte.
- **Sigma=2:** quanto è larga la sfocatura. Sigma più grande = più sfocato.
- **4 livelli di rumore:** 0.005 (quasi niente), 0.01 (poco), 0.05 (medio), 0.1 (tanto). Vogliamo vedere come i metodi si comportano con diversa quantità di rumore.
- **Seed 42:** lo stesso rumore per tutti i metodi. Confronto equo.

**Riferimento teorico:** Il kernel gaussiano è separabile: invece di 9×9=81 operazioni per pixel, facciamo 9+9=18. Fattore 4.5× più veloce.

---

## Slide 7 — TV: Teoria

**Sulla slide:** Formula TV, formula TV(x), spiegazione L1 vs Tikhonov, parametri.

**Cosa significa:**
- **TV (Total Variation):** misura quanto un'immagine "varia" da un pixel al successivo. Più è bassa, più l'immagine è uniforme.
- **Data fidelity:** il primo pezzo della formula, ||Hx - y||². Dice "l'immagine che ricostruisco, dopo sfocatura, deve assomigliare a quella degradata".
- **Regularization:** il secondo pezzo, lambda·TV(x). Dice "l'immagine deve essere regolare, senza troppi salti bruschi".
- **L1 vs L2:** L1 = valore assoluto, L2 = quadrato. L1 preserva i bordi perché un bordo netto costa come una rampa graduale. L2 preferisce le rampe (sfocate) perché il quadrato penalizza meno i gradienti piccoli.
- **Staircasing:** effetto a scalini. Quando lambda è troppo alto, l'immagine diventa "a blocchetti" invece che liscia. Come disegnare una linea inclinata su un foglio a quadretti.
- **Convesso:** la funzione ha un solo minimo globale, non ci sono "trappole". Lo trovi sempre, qualunque sia il punto di partenza.

**Riferimento teorico:** Interpretazione Bayesiana di Rudin-Osher-Fatemi (1992): i gradienti dell'immagine seguono una distribuzione di Laplace, che favorisce valori piccoli ma permette picchi grandi (bordi). La funzione è convessa — minimo globale unico garantito. Le reti neurali non hanno questa proprietà.

---

## Slide 8 — TV: Algoritmo

**Sulla slide:** Pseudo-codice, dettagli ottimizzazione, tabella scelta lambda.

**Cosa significa:**
- **Inizializzazione x = y:** partiamo dall'immagine degradata e la modifichiamo poco alla volta.
- **150 iterazioni:** ripetiamo il processo 150 volte. Ogni volta l'immagine migliora un po'. Dopo 150 si stabilizza.
- **Adam:** un ottimizzatore che adatta il passo di aggiornamento per ogni pixel. Accelera nelle zone uniformi (dove il gradiente è piccolo), rallenta ai bordi (dove il gradiente è grande).
- **Lambda=0.005:** il valore che dà il miglior equilibrio tra togliere rumore e preservare dettagli.
- **Backpropagation:** calcola di quanto e come muovere ogni pixel per far diminuire la loss.
- **Clamp [-1,1]:** dopo ogni aggiornamento, riportiamo i pixel nell'intervallo valido. Se un pixel va a 2, lo riportiamo a 1.

**Riferimento teorico:** Il gradiente di TV è 0 nelle zone piatte e ha salti discreti agli edge. Adam adatta il passo di apprendimento per ogni parametro — accelera nelle zone piatte e rallenta ai bordi. SGD con passo fisso sarebbe subottimale.

---

## Slide 9 — TV: Risultati

**Sulla slide:** Tabella PSNR/SSIM, immagine qualitativa.

**Cosa significa:**
- **PSNR (Peak Signal-to-Noise Ratio):** misura la qualità pixel per pixel. Più alto = più simile all'originale.
- **SSIM (Structural Similarity):** misura la qualità strutturale. Confronta non singoli pixel ma gruppi di pixel (finestre 11×11). Più allineato con ciò che l'occhio umano vede.
- **32 dB:** buono. Tipicamente sopra i 30 dB significa "buona qualità".
- **0.586 a sigma=0.1:** SSIM crolla perché lo staircasing distrugge la struttura locale. Le cellule sembrano fatte di mattoncini.
- **7 secondi per immagine:** è lento perché fa 150 iterazioni, ognuna con una convoluzione.

**Riferimento teorico:** SSIM confronta tre componenti in finestre 11x11: luminanza (luminosità media), contrasto (varianza), struttura (correlazione). Più fedele alla percezione umana del PSNR. PSNR guarda pixel per pixel — ogni 3 dB significa metà dell'errore quadratico.

---

## Slide 10 — UNet: Architettura

**Sulla slide:** Diagramma a U, canali, skip connections, dettagli architettura.

**Cosa significa:**
- **Encoder:** comprime l'immagine per estrarre le caratteristiche importanti (bordi, texture, forme). Ogni passaggio dimezza la risoluzione ma raddoppia il numero di canali.
- **Decoder:** ricostruisce l'immagine dalle caratteristiche estratte. Ogni passaggio raddoppia la risoluzione e dimezza i canali.
- **Skip connections:** "ponti" che portano i dettagli fini dall'encoder al decoder. Senza, il decoder saprebbe "cosa" c'è nell'immagine ma non "dove".
- **DoubleConv:** due convoluzioni 3×3 in sequenza, ognuna seguita da GroupNorm e ReLU. Ogni blocco impara features più complesse.
- **GroupNorm:** normalizza i dati divisi in gruppi di canali. Funziona bene anche con batch piccoli. BatchNorm richiederebbe batch grandi per stimare media e varianza.
- **1.9M parametri:** circa 1.9 milioni di numeri che la rete impara durante il training. Non tantissimi per una rete neurale (alcune ne hanno miliardi).
- **Input 4 canali:** RGB (3 canali colore) + mappa del rumore (1 canale). La mappa del rumore dice alla rete quanto rumore c'è.

---

## Slide 11 — UNet: Training

**Sulla slide:** Tabella parametri, pseudo-codice training.

**Cosa significa:**
- **Loss L1:** la funzione che dice quanto la rete sta sbagliando. L1 = differenza assoluta. MSE = differenza al quadrato. L1 preserva i bordi perché non penalizza troppo gli errori grandi.
- **Learning rate 10^-4:** quanto è grande il passo di apprendimento. 0.0001 è piccolo e sicuro. Troppo grande la rete non converge. Troppo piccolo la rete impara lentamente.
- **Batch 16:** quante immagini elabora insieme. 16 è un numero piccolo (limite CPU). Con batch più grande il training sarebbe più stabile ma servirebbe più memoria.
- **50 epoche:** quante volte vede ogni immagine. 50 è sufficiente per imparare senza andare in overfitting.
- **Multi-noise augmentation:** a ogni batch sceglie un livello di rumore a caso. La rete impara a gestire TUTTI i livelli con un'unica rete.
- **Forward:** la rete processa l'immagine e produce un risultato.
- **Backprop:** calcola l'errore e lo propaga all'indietro per capire quali pesi aggiustare.
- **Checkpoint:** salva il modello quando la validazione dà il PSNR migliore. Così teniamo la versione migliore.

**Riferimento teorico:** L1 stima la mediana condizionale (preserva i bordi). MSE stima la media condizionale (tende a sfumare). Con 4 livelli di downsampling, il collo di bottiglia della UNet vede l'intera immagine — ha contesto globale per ricostruire.

---

## Slide 12 — UNet: Risultati

**Sulla slide:** Tabella PSNR/SSIM/tempo.

**Cosa significa:**
- **29.79 dB:** PSNR iniziale. Non eccellente come TV (32.09) ma molto buono.
- **28.46 a sigma=0.1:** solo 1.3 dB in meno del valore iniziale. TV perde 5.5 dB. UNet è STABILE.
- **0.03 secondi per immagine:** 200× più veloce di TV. È una singola forward pass — fa tutto in un colpo solo.
- **Supera TV a sigma=0.1 (28.46 vs 26.54):** quando il rumore è alto, aver visto tante immagini rumorose nel training ripaga. Il prior imparato è più ricco di quello scritto a mano.
- **SSIM 0.795:** tiene bene la struttura anche a rumore alto.

---

## Slide 13 — DiffPIR: Panoramica

**Sulla slide:** Blocco blue DiffPIR, blocco arancione LightUNet.

**Cosa significa:**
- **DiffPIR (Diffusion PnP Image Restoration):** metodo del 2023 che usa un modello di diffusione per restaurare immagini. "Plug-and-Play" = puoi attaccare diversi modelli e funziona.
- **DDPM (Denoising Diffusion Probabilistic Model):** modello che impara a rimuovere rumore gradualmente, passo dopo passo. Alla base di tecnologie come Dall-E e Stable Diffusion.
- **LightUNet:** una UNet piccola (1.26M parametri) specifica per le nostre immagini. 5 MB di peso. Abbastanza piccola per girare su CPU.
- **Why custom:** un modello pre-addestrato su ImageNet (2 GB, 500M parametri) non funzionerebbe su CPU e non saprebbe niente di cellule cervicali. Il nostro è piccolo e specializzato.
- **PnP framework:** separa il denoising (modello generativo) dalla fedeltà ai dati (formula matematica con FFT). Ogni parte fa il suo lavoro.

---

## Slide 14 — DDPM e LightUNet

**Sulla slide:** Dettagli DDPM e LightUNet, formule forward/reverse.

**Cosa significa:**
- **1000 timestep:** 1000 passi in cui si aggiunge rumore. Dal pulito al completamente rumoroso.
- **Forward:** aggiunge rumore. Se avete un video di un cubetto di ghiaccio che si scioglie, il forward è dal ghiaccio (immagine) all'acqua (rumore).
- **Reverse:** toglie rumore. Dal rumore all'immagine. Il modello impara questo.
- **Predire il rumore, non l'immagine:** è più facile. Il rumore ha una forma nota (campana gaussiana). L'immagine è complessa. È come insegnare a qualcuno a togliere le macchie da una foto (facile) invece che a ridipingere la foto da zero (difficile).
- **Time embedding sinusoidale:** dice al modello "a che passo siamo". Come un orologio. Se siamo al passo 10, c'è poco rumore. Se siamo al passo 900, c'è tanto rumore. Il modello si comporta diversamente.
- **GroupNorm + SiLU:** modi di normalizzare e attivare i neuroni. Tecniche moderne che funzionano bene.

**Riferimento teorico:** Ho et al. (2020) — la forma chiusa della diffusione permette di saltare direttamente a qualsiasi timestep senza iterare. Il modello predice il rumore perché è equivalente a fare score matching: impara la direzione in cui aumentare la probabilità. Song et al. (2021) — DDIM rende il processo deterministico, permettendo di fare 15 passi invece di 1000.

---

## Slide 15 — DiffPIR: Algoritmo

**Sulla slide:** Pseudo-codice a sinistra, formule a destra.

**Cosa significa:**
- **FFT (Fast Fourier Transform):** trasforma l'immagine dal dominio spaziale (pixel) al dominio delle frequenze. La sfocatura diventa una moltiplicazione semplice invece di una convoluzione complicata.
- **Data fidelity:** costringe il risultato a spiegare l'immagine degradata. "Va bene che il modello generativo ha la sua idea, ma deve anche essere compatibile con i dati che abbiamo".
- **rho_t dinamico:** un peso che cambia a ogni passo. All'inizio (tanto rumore) diamo più fiducia ai dati. Alla fine (poco rumore) diamo più fiducia al modello generativo. È come un arbitro che all'inizio della partita lascia giocare, alla fine fischia di più.
- **t_start=50:** non partiamo dal rumore puro (t=1000) ma da t=50. Perché a t=1000 un piccolo errore viene amplificato 150 volte. A t=50 viene amplificato solo del 3%. È più stabile e sicuro.
- **DDIM step:** un modo di fare il passo inverso della diffusione più veloce (deterministico invece che stocastico).

**Riferimento teorico:** La FFT risolve il problema di fedeltà ai dati in tempo O(N log N) invece di O(N³) — fondamentale per essere pratico. t_start=50: l'immagine è ancora al 97% originale, l'amplificazione dell'errore è solo del 3%. A t=1000 sarebbe 150x. DPS usa gradienti (costoso), DiffPIR fa operator splitting (modulare, senza autograd).

---

## Slide 16 — DiffPIR: Risultati

**Sulla slide:** Tabella parametri, tabella risultati, immagine, osservazioni.

**Cosa significa:**
- **15.78 dB a sigma=0.005:** molto basso. Il modello peggiora l'immagine quasi pulita. Perché "allucina" — inventa dettagli che non ci sono.
- **25.46 dB a sigma=0.1:** molto meglio. Il prior generativo aiuta a ricostruire.
- **Andamento invertito:** TV e UNet peggiorano col rumore. DiffPIR MIGLIORA. È controintuitivo ma ha senso.
- **Allucinazione (hallucination):** il modello aggiunge dettagli realistici ma non veri. Come se qualcuno vi desse una ricetta quasi perfetta e voi aggiungeste ingredienti che non ci sono "per migliorarla".
- **Bias-variance trade-off:** il prior forte (alta bias) è un problema quando i dati sono buoni, ma un vantaggio quando i dati sono rovinati.

**Riferimento teorico:** L'errore di ricostruzione si scompone in tre parti: bias (errore da assunzioni sbagliate), varianza (sensibilità al rumore), e rumore irriducibile. DiffPIR ha alta bias (prior forte) ma bassa varianza (DDIM deterministico). TV ha bias medio, varianza zero. UNet ha bassa bias, varianza moderata. A basso rumore la bias domina (DiffPIR peggiore). Ad alto rumore la bias aiuta a regolarizzare (DiffPIR recupera).

---

## Slide 17 — Implementazione

Nella presentazione non c'è una slide specifica — il copione spiega la struttura del codice.

**Cosa significa:**
- **Modulare:** ogni metodo in una cartella separata. Si possono modificare indipendentemente.
- **34 unit test:** pezzi di codice che verificano che ogni funzione faccia quello che deve. Se qualcuno modifica qualcosa e rompe un'altra parte, i test lo segnalano.
- **Seed 42 per la riproducibilità:** se qualcuno esegue il nostro codice, ottiene esattamente i nostri risultati.

---

## Slide 18 — Confronto PSNR/SSIM

**Sulla slide:** Tabella principale con tutti i numeri. Blocco Observations.

**Cosa significa:**
- **TV domina a sigma≤0.01:** se il rumore è poco, TV è il migliore. 32 dB è un ottimo PSNR.
- **UNet miglior trade-off:** non è mai il primo in classifica, ma è secondo ovunque e perde solo 1 dB in tutto lo spettro.
- **DiffPIR aumentA col rumore:** da 15.78 a 25.46 dB. Parte malissimo ma recupera.
- **Incrocio a sigma≈0.05:** TV ha PSNR migliore (30.42 vs 29.44) ma UNet ha SSIM migliore (0.864 vs 0.837). Questo è il punto di svolta.

**Riferimento teorico:** PSNR misura l'errore pixel per pixel. SSIM confronta strutture locali in finestre 11x11.

---

## Slide 19 — Grafico comparativo

**Sulla slide:** Grafico PSNR/SSIM vs rumore.

**Cosa significa:** Il grafico mostra visivamente quello che la tabella dice coi numeri. TV e UNet sono linee quasi dritte. DiffPIR sale. L'incrocio TV-UNet si vede chiaramente.

---

## Slides 20-23 — Risultati qualitativi

**Sulla slide:** Immagini dei risultati a vari livelli di rumore.

**Cosa significa:**
- **TV:** si vede subito com'è nitido a basso rumore (slide 21). Poi arriva lo staircasing a medio/alto rumore (slide 22-23). Le cellule sembrano fatte di quadratini.
- **UNet:** sempre uniforme. Non eccelle mai ma non sbaglia mai.
- **DiffPIR:** brutto a basso rumore (slide 21 — artefatti evidenti). Migliora visibilmente ad alto rumore (slide 23).
- **Mappe di differenza (slide 23):** mostrano DOVE ogni metodo sbaglia. Bianco = errore grande, nero = errore piccolo.
- **TV:** errori a blocchi (staircasing sistematico).
- **UNet:** errori piccoli e sparsi.
- **DiffPIR:** errori concentrati sui bordi (ha spostato i contorni delle cellule).

---

## Slide 24 — Confronto famiglie

**Sulla slide:** Tre blocchi PRO/CONTRO.

**Cosa significa:**
- **TV:** trasparente — si capisce esattamente cosa fa. Non serve training — funziona subito su qualsiasi immagine. Ma se il rumore è tanto, non basta.
- **UNet:** veloce — 0.03 secondi. Robusto — non gli importa quanto rumore c'è. Ma se gli date un'immagine diversa da quelle viste in training, potrebbe non funzionare.
- **DiffPIR:** creativo — ricostruisce dettagli che gli altri perdono. Ma lento (3 secondi), e a volte inventa cose che non esistono.
- **Non c'è un vincitore universale:** dipende da quanto rumore c'è, quanto tempo avete, e cosa vi serve.

---

## Slide 25 — Regimi operativi

**Sulla slide:** Tabella scenari.

**Cosa significa:**
- **Basso rumore:** scegli TV. Non serve training, fa 32 dB, è perfetto.
- **Medio rumore:** UNet è la scelta sicura. TV inizia a fare staircasing.
- **Alto rumore:** UNet è il migliore in assoluto. Se volete texture più realistiche, provate DiffPIR.
- **Velocità:** se dovete fare tante immagini (es. un video), UNet è l'unica scelta (0.03 s). Le altre sono troppo lente (3-7 s).

---

## Slide 26 — Conclusioni

**Sulla slide:** Tre blocchi: Main Results, Lessons Learned, Future Directions.

**Cosa significa:**
- **Confronto equo:** abbiamo dato a tutti gli stessi dati, stesse metriche, stesse condizioni. Le differenze sono solo nei metodi, non in fattori esterni. Molti articoli non lo fanno.
- **Nessun metodo perfetto:** non esiste "il migliore". Esiste il più adatto al vostro problema. Se l'immagine ha poco rumore, TV vince senza bisogno di training. Se ne ha tanto, serve deep learning.
- **Con GPU avremmo fatto di più:** una scheda grafica potente permetterebbe modelli più grandi, training più veloce, batch più grandi.
- **Lambda adattivo:** un lambda che cambia in base al rumore invece di essere fisso.
- **Validazione clinica:** testare su immagini reali (non degradate artificialmente) per vedere se funziona davvero in un ospedale.

---

## Slide 27 — Bibliografia

**Sulla slide:** Sei riferimenti.

**Cosa significa:** Ogni articolo è una tappa fondamentale:
- Rudin-Osher-Fatemi (TV, 1992): ha inventato il metodo variazionale con L1
- Ronneberger et al. (UNet, 2015): ha inventato la U-Net per immagini mediche
- Ho et al. (DDPM, 2020): ha introdotto i modelli di diffusione
- Song et al. (DDIM, 2021): li ha resi 60× più veloci
- Zhu et al. (DiffPIR, 2023): ha combinato diffusione + PnP per il restauro

Grazie per l'attenzione.
