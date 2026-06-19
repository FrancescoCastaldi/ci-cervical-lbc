# Spiegazione del progetto

> Da leggere alla prof seguendo le slide.

---

## Slide 1 — Copertina

**Sulla slide:** Titolo, nomi, logo UNIBO, link GitHub.

**Cosa significa:** Siamo due studenti che hanno fatto un progetto di restauro immagini. Abbiamo preso foto di cellule cervicali rovinate e le abbiamo riparate con tre metodi diversi per vedere qual è il migliore.

**Esempio:** È come portare tre foto rovinate in tre diversi laboratori di restauro: uno usa prodotti chimici (TV), uno usa un computer che ha visto migliaia di foto (UNet), uno usa un artista che sa ridipingere i dettagli (DiffPIR). Alla fine confrontiamo i risultati.

---

## Slide 2 — Il problema inverso

**Sulla slide:** Formula y = H(x) + n. Riquadro arancione "Why it is difficult".

**Cosa significa:**
- **Problema inverso:** invece di "ho una foto pulita, la rovino" (problema diretto), facciamo il contrario: "ho una foto rovinata, voglio recuperare quella pulita" (problema inverso).
- **Mal posto (ill-posed):** non si può semplicemente invertire la sfocatura perché il rumore esplode.
- **Prior:** una regola che dice "le immagini pulite di solito sono fatte così". Serve per scegliere tra tutte le possibili soluzioni.
- **AWGN:** rumore bianco gaussiano additivo. È il tipo di rumore più comune, come la "neve" sulla TV.
- **Gaussian blur:** sfocatura a forma di campana, come mettere una foto fuori fuoco.

**Esempio:** Avete scritto una lettera a matita. Passate la gomma (sfocatura) e versate del tè sopra (rumore). Ora rileggete. Non si può tornare indietro semplicemente — se provate a "invertire" la gomma, le macchie di tè diventano ancora più grandi. Dovete usare un metodo intelligente che indovini cosa c'era scritto basandosi su come sono fatte di solito le lettere. Quel "come sono fatte di solito" è il prior.

**Riferimento teorico:** Hadamard (1902) ha definito le condizioni per un problema ben posto. Il nostro non le soddisfa. Con la SVD si vede che quando il valore singolare sigma è piccolo, il fattore 1/sigma amplifica il rumore.

---

## Slide 3 — I tre metodi scelti

**Sulla slide:** Tre colonne con immagini: TV (Variazionale), UNet (Deep Learning), DiffPIR (Generativo).

**Cosa significa:**
- **Variazionale:** scriviamo una formula matematica a mano e la minimizziamo. TV dice "le immagini pulite hanno pochi bordi netti". Non serve training.
- **Deep Learning:** mostriamo tanti esempi al computer e lui impara da solo. UNet dice "ho visto migliaia di cellule, so come devono essere".
- **Generativo:** il modello sa generare immagini pulite da zero, come un artista. Poi le adatta ai dati che ha. DiffPIR dice "so com'è una cellula vera, la ricostruisco anche se mancano dettagli".
- **Lambda = 0.005:** parametro che dice quanto è forte la regolarizzazione. Più è alto, più l'immagine viene "lisciata".

**Esempio:** Tre cuochi con lo stesso piatto bruciato:
- TV: ha una ricetta scritta (formula matematica). Segue le istruzioni alla lettera.
- UNet: ha cucinato questo piatto mille volte. Sa già come aggiustarlo a occhio.
- DiffPIR: sa cucinare qualsiasi piatto da zero. Ricostruisce gli ingredienti mancanti come meglio crede.

---

## Slide 4 — Il dataset: Mendeley LBC Cervical Cancer

**Sulla slide:** Esempio immagine, tabella classi.

**Cosa significa:**
- **962 immagini:** non tantissime, ma sufficienti per un progetto su CPU.
- **256×256:** l'immagine originale era enorme (2048×1536). L'abbiamo rimpicciolita.
- **NILM, LSIL, HSIL, SCC:** tipi di cellule dal normale al tumorale. Ma noi non classifichiamo — ripariamo.
- **Normalizzazione [-1,1]:** trasformiamo i pixel in numeri tra -1 e 1.

**Esempio:** Immaginate di avere 962 fototessere. Alcune sono nitide, altre mosse. Volete insegnare a un programma a sistemarle tutte. Le fototessere sono grandi 20×15 cm — troppo grandi per lavorarci comodamente. Le riducete a 3×3 cm (256×256 pixel). I dettagli importanti (occhi, naso) si vedono ancora, ma ora ci potete lavorare.

---

## Slide 5 — Preprocessing e split

**Sulla slide:** Tabella pipeline, tabella split.

**Cosa significa:**
- **Preprocessing:** le cose che si fanno prima di iniziare il vero lavoro.
- **Resize:** ridimensionare le immagini. Da 2048×1536 a 256×256 pixel.
- **ToTensor:** convertire in numeri che il computer può elaborare.
- **Normalizzazione:** portare i numeri in un range uniforme [-1,1].
- **Split 70/15/15:** dividiamo in tre gruppi: 70% imparare, 15% controllare, 15% test finale.
- **Seed 42:** numero fisso per garantire split identico.
- **Stratificato:** manteniamo le proporzioni di classi in ogni gruppo.

**Esempio:** Avete 962 foto di cani e gatti. Per insegnare a un programma a riconoscerli, non potete mostrargliele tutte in una volta. Ne usate 674 per studiare (come i compiti a casa), 144 per le verifiche intermedie, e le ultime 145 per l'esame finale. Se fate lo split a caso, potrebbero capitare tutti i gatti nell'esame finale e nessuno nei compiti — il programma non imparerebbe mai a riconoscere i gatti! Per questo lo split è stratificato: manteniamo la stessa percentuale di cani e gatti in ogni gruppo.

---

## Slide 6 — La degradazione

**Sulla slide:** Due colonne blur/rumore, pipeline, strip immagini degradate.

**Cosa significa:**
- **Degradazione:** il processo che rovina l'immagine. Lo facciamo artificialmente perché vogliamo sapere com'era quella pulita.
- **Kernel 9×9:** matrice 9×9 che dice come ogni pixel si "spalma" sui vicini.
- **Sigma=2:** quanto è larga la sfocatura.
- **4 livelli di rumore:** 0.005, 0.01, 0.05, 0.1.
- **Seed 42:** lo stesso rumore per tutti.

**Esempio:** Prendete 4 copie della stessa foto. Sulla prima versate una goccia d'acqua (σ=0.005), sulla seconda mezzo bicchiere (σ=0.01), sulla terza un bicchiere pieno (σ=0.05), sulla quarta immergetela nella vasca da bagno (σ=0.1). Poi chiamate tre restauratori diversi e vedete chi la recupera meglio a ogni livello di bagnato.

**Riferimento teorico:** Il kernel gaussiano è separabile: invece di 9×9=81 operazioni per pixel, facciamo 9+9=18. Fattore 4.5× più veloce.

---

## Slide 7 — TV: Teoria

**Sulla slide:** Formula TV, formula TV(x), spiegazione L1 vs Tikhonov, parametri.

**Cosa significa:**
- **TV (Total Variation):** misura quanto un'immagine "varia" da un pixel al successivo.
- **Data fidelity:** "l'immagine ricostruita, dopo sfocatura, deve assomigliare a quella degradata".
- **Regularization:** "l'immagine deve essere regolare, senza troppi salti bruschi".
- **L1 vs L2:** L1 preserva i bordi, L2 li sfuma.
- **Staircasing:** effetto a scalini quando lambda è troppo alto.
- **Convesso:** la funzione ha un solo minimo globale.

**Esempio:** Prendete una foto di un gatto. Il bordo tra il gatto e lo sfondo è netto.
- Con L2 (Tikhonov): il bordo diventa sfumato, come se il gatto si sciogliesse nello sfondo. Perché L2 penalizza tanto i gradienti grandi, quindi preferisce "spalmare" il bordo su più pixel per ridurre il gradiente per pixel.
- Con L1 (TV): il bordo rimane netto. Perché L1 penalizza allo stesso modo un bordo netto e una rampa graduale — non ha motivo di preferire la rampa.

**Riferimento teorico:** Interpretazione Bayesiana di Rudin-Osher-Fatemi (1992): i gradienti dell'immagine seguono una distribuzione di Laplace, che favorisce valori piccoli ma permette picchi grandi (bordi). La funzione è convessa — minimo globale unico garantito.

---

## Slide 8 — TV: Algoritmo

**Sulla slide:** Pseudo-codice, dettagli ottimizzazione, tabella scelta lambda.

**Cosa significa:**
- **Inizializzazione x = y:** partiamo dall'immagine degradata.
- **150 iterazioni:** ripetiamo 150 volte. Ogni volta migliora un po'.
- **Adam:** ottimizzatore che adatta il passo per ogni pixel.
- **Lambda=0.005:** il miglior equilibrio.
- **Backpropagation:** calcola di quanto muovere ogni pixel.
- **Clamp [-1,1]:** riportiamo i pixel nell'intervallo valido.

**Esempio:** È come scolpire una statua partendo da un blocco di marmo già un po' rovinato. A ogni passaggio (iterazione) togliete un po' di marmo. Adam è come uno scalpello智能: toglie tanto dove il marmo è uniforme (zone piatte), toglie poco dove ci sono dettagli delicati (bordi). Dopo 150 passaggi la statua è pronta. Se togliete troppo (lambda alto) la statua diventa un cubo (staircasing). Se togliete troppo poco (lambda basso) restano le schegge (rumore residuo).

**Riferimento teorico:** Il gradiente di TV è 0 nelle zone piatte e ha salti discreti agli edge. Adam adatta il passo di apprendimento per ogni parametro. SGD con passo fisso sarebbe subottimale.

---

## Slide 9 — TV: Risultati

**Sulla slide:** Tabella PSNR/SSIM, immagine qualitativa.

**Cosa significa:**
- **PSNR:** misura la qualità pixel per pixel. Più alto = più simile.
- **SSIM:** misura la qualità strutturale. Confronta gruppi di pixel.
- **32 dB:** buona qualità.
- **SSIM 0.586 a sigma=0.1:** crolla perché lo staircasing distrugge la struttura locale.
- **7 secondi per immagine:** lento (150 iterazioni).

**Esempio:** Avete una foto di famiglia sgranata. TV la ripulisce bene (PSNR 32 dB — le facce si riconoscono). Ma se aumentate il rumore, la foto diventa "a mosaico": le guance lisce sembrano fatte di quadratini (SSIM 0.586 — la struttura è sbagliata anche se i colori sono giusti). PSNR dice "i pixel sono simili all'originale", SSIM dice "ma la disposizione è sbagliata". Per capire: PSNR controlla se avete i mattoni giusti, SSIM controlla se la casa è costruita bene.

**Riferimento teorico:** SSIM confronta tre componenti in finestre 11x11: luminanza, contrasto, struttura. Più fedele alla percezione umana del PSNR. Ogni 3 dB in PSNR significa metà dell'errore quadratico.

---

## Slide 10 — UNet: Architettura

**Sulla slide:** Diagramma a U, canali, skip connections, dettagli.

**Cosa significa:**
- **Encoder:** comprime per estrarre caratteristiche importanti.
- **Decoder:** ricostruisce dalle caratteristiche.
- **Skip connections:** portano dettagli fini dall'encoder al decoder.
- **GroupNorm:** normalizza per gruppi di canali. Funziona con batch piccoli.
- **1.9M parametri:** numeri che la rete impara.
- **Input 4 canali:** RGB + mappa del rumore.

**Esempio:** Immaginate di dover ridisegnare un ritratto partendo da uno schizzo sporco. Prima lo guardate da lontano per capire la struttura generale (encoder — chi è, che posa ha). Poi lo ridisegnate nei dettagli (decoder). Ma mentre ridisegnate, tenete sempre lo schizzo originale accanto (skip connections) per copiare i dettagli precisi: la forma degli occhi, la piega dei capelli. Senza lo schizzo, disegnereste una faccia generica simile ma non identica. Con lo schizzo, i dettagli tornano al loro posto.

---

## Slide 11 — UNet: Training

**Sulla slide:** Tabella parametri, pseudo-codice training.

**Cosa significa:**
- **Loss L1:** dice quanto la rete sbaglia. Differenza assoluta.
- **Learning rate 10^-4:** passo di apprendimento. 0.0001 piccolo e sicuro.
- **Batch 16:** quante immagini elabora insieme.
- **50 epoche:** quante volte vede ogni immagine.
- **Multi-noise augmentation:** sceglie rumore a caso per ogni batch.
- **Forward:** rete processa l'immagine.
- **Backprop:** propaga l'errore all'indietro.

**Esempio:** È come imparare ad aggiustare foto con un tutor.
- Epoca 1: il tutor vi mostra 673 foto rovinate con la soluzione. Voi provate, sbagliate, lui vi corregge.
- Epoca 2: rifate tutto da capo. Ora sbagliate meno.
- Dopo 50 volte che avete visto ogni foto, siete diventati bravi.
- Il multi-noise è il tutor che ogni volta vi dà una foto con un diverso tipo di sporco: oggi polvere, domani acqua, dopo domani graffi. Così imparate a gestire TUTTI i tipi di sporco, non uno solo.

**Riferimento teorico:** L1 stima la mediana condizionale (preserva i bordi). MSE stima la media condizionale (tende a sfumare). Con 4 livelli di downsampling, il collo di bottiglia della UNet vede l'intera immagine.

---

## Slide 12 — UNet: Risultati

**Sulla slide:** Tabella PSNR/SSIM/tempo.

**Cosa significa:**
- **29.79 dB:** non eccellente come TV (32.09) ma molto buono.
- **28.46 a sigma=0.1:** solo 1.3 dB in meno. TV perde 5.5 dB. UNet è STABILE.
- **0.03 secondi per immagine:** 200× più veloce di TV.
- **Supera TV a sigma=0.1 (28.46 vs 26.54).**
- **SSIM 0.795:** tiene bene la struttura.

**Esempio:** UNet è come un autista medio che guida bene sia col sole che col temporale. TV è come un pilota professionista che sul bagnato sbanda. Il pilota (TV) è meglio col sole (basso rumore), ma l'autista medio (UNet) è più affidabile quando piove. E UNet ci mette 0.03 secondi a decidere come sterzare, TV ci mette 7 secondi — alla velocità di un'auto, 7 secondi sono un'eternità.

---

## Slide 13 — DiffPIR: Panoramica

**Sulla slide:** Blocco blue DiffPIR, blocco arancione LightUNet.

**Cosa significa:**
- **DiffPIR:** metodo del 2023 che usa un modello di diffusione.
- **DDPM:** modello che impara a rimuovere rumore gradualmente.
- **LightUNet:** UNet piccola (1.26M parametri, 5 MB) per CPU.
- **PnP framework:** separa denoising da fedeltà ai dati.

**Esempio:** DiffPIR funziona come un restauratore che ha due strumenti: una gomma magica (DDPM) che toglie lo sporco, e una lente d'ingrandimento (data fidelity) che controlla se il risultato corrisponde alla foto originale rovinata. Li usa in alternanza: gomma, lente, gomma, lente, per 15 cicli. Il nostro restauratore ha studiato solo su foto di cellule cervicali (LightUNet specializzata), non su foto generiche di Internet.

---

## Slide 14 — DDPM e LightUNet

**Sulla slide:** Dettagli DDPM e LightUNet, formule forward/reverse.

**Cosa significa:**
- **1000 timestep:** 1000 passi in cui si aggiunge rumore.
- **Forward:** aggiunge rumore gradualmente.
- **Reverse:** toglie rumore gradualmente. Il modello impara questo.
- **Predire il rumore, non l'immagine:** è più facile.
- **Time embedding sinusoidale:** dice al modello "a che passo siamo".
- **DDIM:** riduce 1000 passi a 15.

**Esempio:** Prendete una foto nitida. Sopra mettete un foglio di carta da lucido leggermente opaco (passo 1). Ancora sopra un altro foglio (passo 2). Dopo 1000 fogli, non si vede più niente — solo bianco. Il modello impara a togliere i fogli uno a uno, partendo dal bianco e arrivando alla foto nitida.

Ma togliere 1000 fogli è lento. DDIM insegna al modello a toglierne 15 alla volta invece di 1 per volta — salta gruppi di fogli. Risultato: stesso punto di arrivo, 60× più veloce.

Predire il rumore (non l'immagine) è come insegnare a qualcuno a togliere le macchie da un vestito (rumore = macchia) invece che a ricucire il vestito da zero (immagine = vestito). È molto più facile.

**Riferimento teorico:** Ho et al. (2020) — la forma chiusa della diffusione permette di saltare direttamente a qualsiasi timestep. Il modello predice il rumore perché fa score matching. Song et al. (2021) — DDIM rende il processo deterministico, 15 passi invece di 1000.

---

## Slide 15 — DiffPIR: Algoritmo

**Sulla slide:** Pseudo-codice a sinistra, formule a destra.

**Cosa significa:**
- **FFT:** trasforma l'immagine nel dominio delle frequenze.
- **Data fidelity:** costringe il risultato a spiegare i dati.
- **rho_t dinamico:** peso che cambia a ogni passo.
- **t_start=50:** non partiamo dal rumore puro ma da t=50.
- **DDIM step:** passo inverso veloce e deterministico.

**Esempio:** rho_t è come un arbitro di calcio che cambia il suo comportamento durante la partita. All'inizio (t alto, tanto rumore — primo tempo), l'arbitro fischia poco: lascia giocare il modello generativo. Alla fine (t basso, pochi dettagli da sistemare — finale), l'arbitro fischia molto: pretende che i dati siano rispettati.

t_start=50: è come iniziare una maratona al chilometro 42 invece che al chilometro 0. Tanto la parte iniziale (da 0 a 42) è solo rumore nella nostra applicazione — non ci serve. Partiamo da dove l'immagine è ancora riconoscibile (97% integra), risparmiando tempo e fatica.

**Riferimento teorico:** La FFT risolve il problema in tempo O(N log N) invece di O(N³). t_start=50: l'immagine è ancora al 97% originale, amplificazione errore solo 3%. A t=1000 sarebbe 150×.

---

## Slide 16 — DiffPIR: Risultati

**Sulla slide:** Tabella parametri, tabella risultati, immagine, osservazioni.

**Cosa significa:**
- **15.78 dB a sigma=0.005:** molto basso. Il modello "allucina".
- **25.46 dB a sigma=0.1:** molto meglio.
- **Andamento invertito:** migliora col rumore.
- **Allucinazione:** il modello aggiunge dettagli realistici ma falsi.
- **Bias-variance trade-off:** prior forte aiuta quando i dati sono rovinati.

**Esempio:** DiffPIR è come quel bambino che ha un'immaginazione troppo vivace.
- Gli mostrate una foto quasi perfetta (poco rumore): lui ci aggiunge dettagli che non ci sono — un terzo occhio, un sorriso strano. Perché? La sua immaginazione (prior) è così forte che vuole "migliorare" anche quello che è già buono. Risultato: la foto peggiora.
- Gli mostrate una foto molto rovinata (tanto rumore): qui la sua immaginazione è un superpotere. Dove gli altri vedono uno scarabocchio, lui ricostruisce una faccia intera. Perché i pochi dettagli che restano vengono amplificati dalla sua fantasia.

**Riferimento teorico:** L'errore si scompone in bias (assunzioni sbagliate) + varianza (sensibilità al rumore) + rumore irriducibile. DiffPIR ha alta bias (prior forte) ma bassa varianza (DDIM deterministico). A basso rumore la bias domina. Ad alto rumore la bias aiuta.

---

## Slide 17 — Implementazione

Nella presentazione non c'è una slide specifica — il copione spiega la struttura del codice.

**Cosa significa:**
- **Modulare:** ogni metodo in una cartella separata.
- **34 unit test:** verificano che ogni funzione funzioni.
- **Seed 42 per riproducibilità.**

**Esempio:** È come un'officina con tre banchi da lavoro separati: uno per la TV, uno per UNet, uno per DiffPIR. Ogni banco ha i suoi strumenti, ma tutti usano lo stesso armadietto dei pezzi di ricambio (dataset) e lo stesso manuale delle misure (metriche). Se qualcuno sposta un cacciavite sul banco TV, gli altri banchi non vengono toccati (modulare). E c'è un ispettore (34 test) che ogni giorno controlla che tutti gli attrezzi funzionino.

---

## Slide 18 — Confronto PSNR/SSIM

**Sulla slide:** Tabella principale con tutti i numeri. Blocco Observations.

**Cosa significa:**
- **TV domina a sigma≤0.01:** 32 dB, ottimo.
- **UNet miglior trade-off:** perde solo 1 dB in tutto lo spettro.
- **DiffPIR:** da 15.78 a 25.46 dB.
- **Incrocio a sigma≈0.05:** punto di svolta.

**Esempio:** Tre ristoranti e un piatto di pasta:
- TV: chef stellato. Se la pasta è quasi perfetta (poco rumore), fa un piatto da 10 e lode. Se la pasta è bruciata, fa un piatto mediocre.
- UNet: bravo cuoco di trattoria. La pasta non è mai eccelsa ma non è mai immangiabile. Qualunque cosa gli diate, tirate fuori un piatto decente.
- DiffPIR: cuoco creativo. Se la pasta è quasi perfetta, lui ci aggiunge ingredienti strani (allucina) e la rovina. Se la pasta è bruciata, lui la trasforma in un piatto creativo sorprendente.

Quale scegliere? Se la pasta è buona, lo chef stellato (TV). Se non sapete com'è, la trattoria (UNet). Se è bruciata, il creativo (DiffPIR).

---

## Slide 19 — Grafico comparativo

**Sulla slide:** Grafico PSNR/SSIM vs rumore.

**Cosa significa:** Il grafico mostra visivamente quello che la tabella dice coi numeri.

**Esempio:** Il grafico è come un termometro della febbre dei metodi. TV ha la febbre che sale (peggiora) col rumore. UNet ha la febbre quasi costante. DiffPIR ha la febbre che scende (migliora) col rumore. Il grafico lo si mostra per far capire subito senza leggere numeri.

---

## Slides 20-23 — Risultati qualitativi

**Sulla slide:** Immagini dei risultati a vari livelli di rumore.

**Cosa significa:**
- **TV:** nitido a basso rumore, poi staircasing.
- **UNet:** sempre uniforme.
- **DiffPIR:** brutto a basso rumore, migliora ad alto.
- **Mappe di differenza:** mostrano DOVE ogni metodo sbaglia (bianco = errore grande).

**Esempio:** È come guardare tre foto di un gatto:
- A basso rumore (slide 21): TV mostra un gatto perfetto, UNet un gatto buono, DiffPIR un gatto con 6 zampe (allucinazione).
- A medio rumore (slide 22): TV mostra un gatto a quadretti (staircasing), UNet un gatto ancora riconoscibile, DiffPIR un gatto con gli occhi un po' strani.
- Ad alto rumore (slide 23): TV è irriconoscibile, UNet è il gatto più somigliante, DiffPIR ha un gatto dai colori realistici ma con la testa leggermente diversa.

Le mappe di differenza sono come le radiografie: mostrano le ossa rotte. TV ha fratture dappertutto (staircasing sistematico), UNet ha qualche microfrattura sparsa, DiffPIR ha fratture concentrate in punti precisi (dove ha allucinato).

---

## Slide 24 — Confronto famiglie

**Sulla slide:** Tre blocchi PRO/CONTRO.

**Cosa significa:**
- **TV:** trasparente, non serve training. Ma se il rumore è tanto, non basta.
- **UNet:** veloce (0.03 s), robusto. Ma serve training.
- **DiffPIR:** ricostruisce dettagli, ma lento e allucina.
- **Non c'è un vincitore universale.**

**Esempio:** È come scegliere un mezzo di trasporto:
- TV = bicicletta. Semplice, economica, la capisci subito. Se la strada è in pianura (basso rumore) arrivi primo. Se è in salita (tanto rumore), fatichi.
- UNet = automobile. Costa (training), ma è veloce e affronta qualsiasi strada. Non è la migliore in tutto (non va in montagna come un fuoristrada, non è agile come una bici), ma è il miglior compromesso.
- DiffPIR = fuoristrada. Supera dossi e buche (ricostruisce dettagli persi) ma consuma tanto (lento) e in autostrada (poco rumore) è scomodo e rumoroso (allucina).

---

## Slide 25 — Regimi operativi

**Sulla slide:** Tabella scenari.

**Cosa significa:**
- **Basso rumore:** scegli TV.
- **Medio rumore:** UNet è la scelta sicura.
- **Alto rumore:** UNet migliore o DiffPIR se volete texture realistiche.
- **Velocità:** UNet (0.03 s) molto più veloce delle altre.

**Esempio:** Dovete restaurare 10.000 foto storiche in un archivio. Il rumore è medio. Se usate TV (7 secondi l'una) = 19 ore. Se usate UNet (0.03 secondi l'una) = 5 minuti. Se usate DiffPIR (3 secondi l'una) = 8 ore. Per un archivio digitale, UNet è l'unica scelta pratica — la qualità è buona e il tempo è accettabile.

---

## Slide 26 — Conclusioni

**Sulla slide:** Tre blocchi: Main Results, Lessons Learned, Future Directions.

**Cosa significa:**
- **Confronto equo:** stessi dati, stesse metriche, stesse condizioni.
- **Nessun metodo perfetto:** il più adatto dipende dal problema.
- **Con GPU avremmo fatto di più.**
- **Lambda adattivo, validazione clinica:** possibili miglioramenti.

**Esempio:** Il progetto è come un test su tre medicine per lo stesso sintomo. Abbiamo dato a tutti gli stessi pazienti (stesse immagini), la stessa dose (stessa degradazione), e misurato con gli stessi strumenti. Risultato: non esiste la medicina universale. L'aspirina (TV) funziona per il mal di testa leggero, l'antibiotico (UNet) è più robusto ma va prescritto, l'omeopatia (DiffPIR) in alcuni casi funziona bene ma in altri inventa effetti. Il dottore deve scegliere in base al paziente.

---

## Slide 27 — Bibliografia

**Sulla slide:** Sei riferimenti.

**Cosa significa:** Ogni articolo è una tappa fondamentale.

**Esempio:** Questi sei articoli sono come i fondatori di una scuola di pensiero. Rudin-Osher-Fatemi (TV, 1992) ha fondato il metodo classico. Ronneberger (UNet, 2015) ha portato il deep learning nelle immagini mediche. Ho (DDPM, 2020) ha inventato la diffusione, Song (DDIM, 2021) l'ha resa pratica, Zhu (DiffPIR, 2023) l'ha applicata al restauro. È la storia di come si evolve la ricerca: uno costruisce sulle spalle del precedente.

Grazie per l'attenzione.
