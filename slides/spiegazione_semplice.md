# Spiegazione del progetto — descrizioni ed esempi

> Da leggere alla prof seguendo le slide.

---

## Slide 1 — Copertina

Questo è il nostro progetto di Computational Imaging. Computational Imaging significa "immagini calcolate dal computer" — cioè usiamo il computer non solo per vedere le immagini, ma per migliorarle, ripararle, ricostruirle.

Noi abbiamo lavorato sul restauro di immagini mediche. In particolare immagini di cellule cervicali, quelle che si prelevano durante il pap-test per la diagnosi del cancro al collo dell'utero. Quando queste immagini vengono scattate al microscopio, spesso sono sfocate o rumorose per vari motivi tecnici. Il nostro compito era prenderle e ripulirle.

Abbiamo confrontato tre metodi diversi per capire quale funziona meglio: TV, UNet e DiffPIR. Ogni metodo appartiene a una famiglia diversa di approcci matematici e informatici. TV è un metodo classico, basato su formule scritte a mano. UNet è una rete neurale, che impara dagli esempi. DiffPIR è un modello generativo, che sa ricostruire dettagli persi.

Il nostro obiettivo era semplice: prendere una foto brutta e renderla bella. Ma abbiamo scoperto che non esiste un metodo perfetto per tutto — ogni metodo ha i suoi punti di forza e i suoi momenti migliori.

**Esempio:** È come avere tre fotografi diversi. Uno è bravo con poca luce, uno è bravo con i ritratti, uno è bravo con i paesaggi. Se li mettete alla prova con la stessa foto rovinata, ognuno la sistema in modo diverso. Noi abbiamo fatto esattamente questo: abbiamo dato la stessa foto rovinata a tutti e tre e abbiamo confrontato i risultati.

---

## Slide 2 — Il problema inverso

Partiamo dal problema di base. Noi abbiamo la foto degradata — cioè un'immagine che è stata rovinata da sfocatura e rumore. Vogliamo recuperare l'immagine originale, quella pulita, quella che il microscopio avrebbe dovuto catturare.

Questo si chiama problema inverso. Nella vita reale, di solito affrontiamo problemi diretti: abbiamo una causa e vogliamo prevedere l'effetto. Per esempio: se lascio cadere un uovo, cosa succede? Si rompe. Il problema inverso sarebbe: ho un uovo rotto, com'era prima? È molto più difficile, perché ci sono tanti modi in cui un uovo può rompersi e non possiamo sapere esattamente com'era prima.

Nel nostro caso il problema diretto è: abbiamo un'immagine pulita, la sfocchiamo e aggiungiamo rumore, cosa otteniamo? La foto degradata. Facile. Il problema inverso è: abbiamo la foto degradata, com'era l'immagine pulita originale? Difficile.

Il problema è che non possiamo semplicemente invertire la sfocatura. Perché? Perché la sfocatura è come una media tra pixel vicini — se provate a fare l'operazione contraria, il rumore viene amplificato talmente tanto che l'immagine diventa completamente bianca e nera, inutilizzabile. È come se aveste un messaggio scritto a matita, qualcuno ci passa la gomma sopra, e voi provate a rileggere quello che c'era scritto. Se provate a "invertire" la gomma, ottenete solo macchie.

Per risolvere questo problema, dobbiamo aggiungere un vincolo, una regola che dice "le immagini vere di solito sono fatte così". Questo vincolo si chiama regolarizzazione. È come dire: "Non so esattamente com'era l'uovo prima di rompersi, ma so che era intero, aveva il guscio, e il tuorlo era dentro". Con queste informazioni, posso fare un'ipotesi migliore.

**Esempio:** È come quando la polizia scientifica ricostruisce un volto da un cranio. Non sanno esattamente com'era la persona, ma sanno che in media gli occhi sono a una certa distanza, il naso ha una certa forma, le labbra hanno un certo spessore. Usano queste informazioni per fare una ricostruzione plausibile. La regolarizzazione fa la stessa cosa per le immagini.

---

## Slide 3 — I tre metodi scelti

Il corso ci proponeva quattro metodi. Noi ne abbiamo scelti tre, uno per ogni famiglia di approcci. Questo ci permette di confrontare strategie completamente diverse tra loro.

TV è un metodo variazionale. Variazionale significa che si basa su una funzione matematica che descrive quanto un'immagine è "buona". Noi scriviamo a mano le regole: "un'immagine pulita ha pochi bordi netti e tanto sfondo uniforme". Poi il computer modifica l'immagine poco alla volta per soddisfare queste regole, bilanciandole con la fedeltà ai dati originali. È trasparente: si capisce esattamente cosa fa e perché.

UNet è deep learning. Deep learning significa che usiamo una rete neurale, cioè un programma che imita il cervello umano con tanti piccoli "neuroni" artificiali collegati tra loro. Non scriviamo regole: mostriamo alla rete tanti esempi di foto sporche e foto pulite, e lei impara da sola come fare. È come insegnare a un bambino: non gli spieghi le regole grammaticali, gli fai leggere tanti libri e lui impara per osmosi.

DiffPIR è un modello generativo. Generativo significa che non solo sa ripulire le immagini, ma sa anche generare immagini pulite da zero, partendo da rumore puro. È come un artista che ha visto così tanti dipinti che è in grado di dipingere un quadro nuovo di zecca. Nel nostro caso, usa questa capacità per "riempire" i dettagli che mancano nell'immagine degradata.

**Esempio di confronto quotidiano:** TV è come un cuoco che segue una ricetta precisa: "30 grammi di questo, 5 minuti di quello". Se la ricetta è buona, il piatto viene buono. Ma se gli ingredienti sono diversi (per esempio il rumore è di un tipo che la ricetta non prevede), il piatto può venire male.

UNet è come un cuoco che ha lavorato in cucina per dieci anni e ha visto mille piatti. Non ha bisogno di ricetta: a occhio sa come bilanciare i sapori. Se gli date ingredienti nuovi, si adatta in fretta perché ha esperienza.

DiffPIR è come un cuoco creativo, uno chef stellato. Se manca un ingrediente, non si ferma: lo sostituisce con qualcosa di simile e il piatto viene comunque buono. Ma a volte, se l'ingrediente mancante era fondamentale, il risultato può essere strano, diverso da quello che vi aspettavate.

---

## Slide 4 — Il dataset

Per fare i nostri esperimenti, avevamo bisogno di immagini di cellule cervicali. Abbiamo usato 962 immagini da Mendeley, che è un archivio pubblico dove i ricercatori condividono i loro dati. Questo è importante perché i nostri risultati possono essere riprodotti da altri: se qualcuno vuole ripetere i nostri esperimenti, può scaricare le stesse immagini.

Le immagini sono state scattate al microscopio durante i pap-test. Il pap-test è l'esame che si fa per diagnosticare il cancro al collo dell'utero: si prelevano cellule dalla cervice e si osservano al microscopio per vedere se sono sane o anomale.

Ci sono quattro tipi di cellule nelle immagini. NILM sono cellule normali, sane. LSIL sono lesioni di basso grado, cioè cellule leggermente anomale ma non ancora tumorali. HSIL sono lesioni di alto grado, cellule molto anomale che potrebbero diventare cancerose. SCC è carcinoma, cioè cellule già tumorali. Abbiamo mantenuto le proporzioni originali nel training, cioè abbiamo usato le stesse percentuali di ogni classe che si trovano nel dataset originale.

Le immagini originali sono enormi: 2048×1536 pixel. Pixel sono i puntini che formano un'immagine digitale. Un'immagine 2048×1536 ha circa 3 milioni di puntini. Per farle elaborare al computer, che aveva solo una CPU (il processore normale) e non una GPU (scheda grafica potente), le abbiamo ridimensionate a 256×256 pixel. In pratica abbiamo ridotto ogni immagine a circa 65.000 puntini, 50 volte più piccola dell'originale. Abbiamo perso dettaglio fine, ma le caratteristiche principali delle cellule — forma, dimensione, texture del nucleo — si vedono ancora bene.

**Esempio:** È come prendere una foto 4K (ultra HD) e ridimensionarla a 720p (HD standard). Perdete un po' di qualità, ma il contenuto principale della foto rimane. I vostri amici nella foto sono ancora riconoscibili, solo i dettagli più fini (come le ciglia) si perdono. Per il nostro scopo (capire se il restauro funziona), 256×256 è sufficiente.

---

## Slide 5 — Preprocessing e split

Prima di lavorare con le immagini, abbiamo fatto quattro operazioni di preparazione. Si chiama preprocessing, cioè quello che si fa prima del vero e proprio lavoro.

Prima operazione: ridimensionamento. Tutte le immagini sono state portate a 256×256 pixel, così sono tutte uguali e il computer può elaborarle in blocco. Se fossero di dimensioni diverse, dovremmo scrivere programmi diversi per ciascuna.

Seconda operazione: conversione in tensori. Un tensore non è niente di spaventoso — è semplicemente una griglia di numeri. Un'immagine a colori è un tensore tridimensionale: larghezza × altezza × 3 (rosso, verde, blu). La nostra immagine in bianco e nero è larghezza × altezza × 1 (un solo valore per pixel, dal nero al bianco). I programmi di deep learning lavorano con numeri, non con immagini, quindi dobbiamo convertire tutto in numeri.

Terza operazione: normalizzazione. I pixel delle immagini hanno di solito valori da 0 (nero) a 255 (bianco). Noi li abbiamo portati nell'intervallo [-1,1]. Perché? Perché le reti neurali funzionano meglio quando i numeri non sono troppo grandi o troppo piccoli. Se i valori fossero tra 0 e 255, alcuni numeri sarebbero grandi e altri piccoli, e la rete farebbe fatica a imparare. Con valori tra -1 e 1, tutto è bilanciato.

Quarta operazione: divisione del dataset. Abbiamo diviso le 962 immagini in tre gruppi. Il 70% (673 immagini) è andato al training: qui i metodi imparano, vedono le coppie sporco-pulito e allenano i loro parametri. Il 15% (145 immagini) è andato alla validazione: durante l'addestramento, controlliamo ogni tanto come stanno andando le cose su immagini che il metodo non ha mai visto, per evitare che impari a memoria (si chiama overfitting, è come studiare le risposte del compito senza capire la materia). Il 15% finale (144 immagini) è andato al test: alla fine, valutiamo i risultati su immagini completamente nuove, che nessun metodo ha mai visto.

Tutti i metodi hanno visto le stesse identiche immagini di test. Questo è fondamentale. Se due metodi vedessero immagini diverse, non potremmo confrontare i risultati.

**Esempio:** È come un esame universitario. Il training sono le lezioni e lo studio sui libri. La validazione sono le esercitazioni in classe, dove il professore vi dà un problema e vedete se avete capito. Il test è l'esame finale: domande nuove che non avete mai visto prima. Se due studenti fanno esami diversi, non potete confrontare i loro voti. Noi abbiamo dato lo stesso identico esame a tutti e tre i metodi.

---

## Slide 6 — La degradazione

Per testare i nostri metodi, dovevamo avere immagini sporche da ripulire. Ma siccome volevamo anche sapere qual era l'immagine pulita originale (per calcolare se il restauro era stato buono), abbiamo preso le immagini pulite e le abbiamo danneggiate artificialmente.

Lo abbiamo fatto in due modi, combinati insieme. Primo: una sfocatura gaussiana. Gaussiana significa che la sfocatura segue una forma a campana, come quella che si vede quando mettete a fuoco male un obiettivo fotografico. Abbiamo usato un kernel (la "maschera" di sfocatura) di 9×9 pixel con deviazione standard σ=2. In pratica è come se ogni pixel si "spalmasse" sui vicini in un raggio di 4-5 pixel.

Secondo: rumore additivo gaussiano bianco. Questo è il tipo di rumore più comune, come la "neve" che si vede sulla TV quando non prende bene il canale. L'abbiamo aggiunto a quattro livelli di intensità: σ = 0.005 (poco rumore, immagine quasi ancora buona), σ = 0.01 (rumore leggero), σ = 0.05 (rumore medio, l'immagine inizia a essere disturbata), σ = 0.1 (tanto rumore, l'immagine è molto granulosa). Abbiamo usato lo stesso seme casuale (42) per tutti: significa che il rumore è identico per tutti i metodi, nessuno ha un vantaggio.

Tutti e tre i metodi hanno ricevuto le stesse immagini degradate. Se un metodo fa meglio di un altro, è perché il metodo in sé è migliore per quel livello di rumore, non perché ha avuto dati più facili.

**Esempio pratico:** Prendete una foto nitida del vostro gatto. Prima la sfuocate con Photoshop (mettete un po' di "Gaussian blur"). Poi aggiungete della grana (rumore) a vari livelli: un po', abbastanza, tanto, tantissimo. Alla fine avete quattro foto rovinate del gatto. Le date a tre programmi diversi e dite: "Sistematele". Chi le sistema meglio? Noi abbiamo fatto esattamente questo, ma con immagini di cellule invece che di gatti.

---

## Slide 7-9 — Total Variation (TV)

TV è il primo metodo che abbiamo testato. TV sta per Total Variation, che in italiano si può tradurre come "Variazione Totale". Il concetto è semplice: un'immagine pulita ha poca variazione tra pixel vicini, tranne dove ci sono bordi veri (i contorni delle cellule). Il metodo premia le immagini con pochi salti bruschi.

Come funziona nel dettaglio: partiamo dall'immagine degradata. Poi la modifichiamo passo dopo passo, cercando di bilanciare due obiettivi che sono in conflitto tra loro. Il primo obiettivo è la fedeltà ai dati: se prendiamo la nostra immagine ricostruita e la sfocchiamo di nuovo, deve assomigliare il più possibile alla foto degradata che abbiamo all'inizio. Il secondo obiettivo è la regolarizzazione: l'immagine deve avere poca "variazione totale", cioè pochi cambiamenti bruschi tra pixel vicini.

Questi due obiettivi sono in competizione. Se ci concentriamo solo sul primo, l'immagine sarà fedele ai dati ma rumorosa. Se ci concentriamo solo sul secondo, l'immagine sarà liscia ma potrebbe non assomigliare ai dati. Il trucco è trovare il giusto equilibrio.

Il parametro λ è la manopola che regola questo equilibrio. λ piccolo significa "dai più importanza ai dati, meno alla regolarità". λ grande significa "dai più importanza alla regolarità, meno ai dati". Noi abbiamo provato i valori λ = 0.001, 0.005, 0.01, 0.05 e 0.1.

Con λ = 0.001, la regolarizzazione è debole: il rumore resta visibile. Con λ = 0.005, il rumore viene rimosso bene ma i bordi delle cellule restano nitidi — è il punto ottimale. Con λ = 0.01 e sopra, inizia un problema chiamato staircasing: le zone che dovrebbero avere una sfumatura graduale (come il citoplasma, la parte interna della cellula) diventano a gradini, come una scala. Con λ = 0.1 l'immagine sembra fatta di mattoncini.

**Esempio di λ:** È come il controllo "riduzione rumore" in un programma di fotoritocco come Lightroom. Se lo mettete a zero, la foto è rumorosa ma i dettagli sono tutti lì. Se lo alzate un po', il rumore diminuisce e la foto è ancora nitida. Se lo alzate troppo, la foto diventa plastica, sembra dipinta, i capelli della persona sembrano un blocco unico senza singole ciocche. λ funziona esattamente così.

**Esempio di staircasing:** Immaginate di disegnare una linea inclinata su un foglio a quadretti. Se usate solo quadretti interi, la linea non sarà liscia, ma farà dei gradini: su, destra, su, destra. Questo è lo staircasing. Le immagini con TV a λ alto hanno questo effetto: i bordi inclinati diventano a scalini.

Il grande vantaggio di TV è che non ha bisogno di training. Potete buttargli qualsiasi immagine — una foto della luna, una radiografia, un disegno — e lui funziona subito, senza dover imparare niente prima. Non servono esempi, non servono dati di addestramento. È pronto all'uso.

I risultati quantitativi: TV è il migliore a basso rumore. A σ = 0.005 raggiunge 32.4 dB di PSNR. PSNR è una misura di qualità: più alto è, più l'immagine ricostruita è simile all'originale. 32 dB è considerato buono per immagini a 8 bit.

Ma quando il rumore sale, TV cala. A σ = 0.1 scende a 26.5 dB, una perdita di quasi 6 dB. Perché? Perché il suo "prior" (la regola scritta a mano) è troppo semplice: "le immagini hanno pochi bordi". Quando c'è tanto rumore, questa regola non basta più. Il rumore confonde il metodo: non capisce più cosa è rumore e cosa è dettaglio reale.

---

## Slide 10-12 — UNet

UNet è il secondo metodo, molto diverso da TV. Mentre TV usa regole scritte a mano, UNet impara dai dati. È una rete neurale, un programma ispirato al funzionamento del cervello umano, composta da tanti piccoli "neuroni" artificiali collegati in rete.

La struttura di UNet è particolare e ha dato il nome al metodo: la rete ha una forma a U. Nella parte sinistra della U c'è l'encoder: prende l'immagine e la comprime progressivamente, estraendo le caratteristiche più importanti. Prima identifica i bordi, poi le texture, poi le forme. Ogni passaggio riduce la dimensione dell'immagine ma aumenta il numero di caratteristiche estratte.

Nella parte destra della U c'è il decoder: prende queste caratteristiche e ricostruisce l'immagine, riportandola alla dimensione originale. È come se l'encoder facesse un riassunto dell'immagine e il decoder lo usasse per ricreare l'immagine pulita.

Le skip connections sono la caratteristica più importante di UNet. Sono dei "ponti" che collegano direttamente l'encoder al decoder, saltando la parte centrale. Perché servono? Quando l'encoder comprime l'immagine, perde i dettagli fini — la posizione esatta dei bordi, le piccole texture. Le skip connections portano questi dettagli direttamente al decoder, così la ricostruzione è più precisa. Senza skip connections, l'immagine ricostruita sarebbe sfocata.

**Esempio delle skip connections:** È come se doveste scrivere un riassunto di un libro di 300 pagine in 10 righe, e poi da quelle 10 righe doveste ricostruire il libro intero. Il riassunto cattura la trama principale, ma perdete i dettagli: i nomi dei personaggi secondari, le descrizioni dei luoghi. Le skip connections sono come avere il libro originale a fianco mentre riscrivete: ogni volta che dovete ricordare un dettaglio, lo guardate direttamente.

L'addestramento di UNet è stato fatto su 673 coppie di immagini: da un lato l'immagine sporca, dall'altro la corrispondente immagine pulita. La rete ha visto ogni coppia 50 volte. Ogni volta che vedeva una coppia, confrontava la sua ricostruzione con l'immagine pulita vera, calcolava l'errore, e regolava leggermente i suoi parametri interni per fare meglio la volta successiva. È un processo lento ma efficace: dopo 50 cicli completi (epoche), la rete aveva imparato a riconoscere e rimuovere il rumore.

Il training è durato circa 60 minuti su CPU. Con una GPU sarebbe stato molto più veloce (forse 5-10 minuti), ma non avevamo una GPU disponibile.

Durante il training, UNet ha usato due funzioni di perdita: L1 e MSE. L1 calcola l'errore come differenza assoluta tra pixel corrispondenti. MSE lo calcola come differenza al quadrato. MSE è più sensibile agli errori grandi: se un pixel viene ricostruito molto male, MSE lo penalizza tantissimo. L1 è più equilibrata. Noi abbiamo usato L1 perché preserva meglio i bordi. Con MSE, la rete tende a fare immagini più sfocate, perché preferisce "sbagliare un po' dappertutto" piuttosto che "sbagliare tanto in un punto solo".

**Esempio L1 vs MSE:** Se dovete indovinare un numero, e il numero vero è 10. Con L1, se dite 8 o 12, l'errore è 2 in entrambi i casi. Con MSE, se dite 8, l'errore è (10-8)² = 4. Se dite 12, l'errore è (10-12)² = 4. Ma se dite 5, l'errore è (10-5)² = 25 — molto più grande! MSE odia gli errori grandi, quindi la rete impara a non farli mai. Il risultato è che evita i bordi netti (dove l'errore potrebbe essere grande) e preferisce immagini più sfuocate.

UNet ha usato GroupNorm invece di BatchNorm per normalizzare i dati durante l'elaborazione. BatchNorm funziona bene quando avete tante immagini nel batch (gruppo), ma noi ne avevamo poche (16). GroupNorm raggruppa i canali invece delle immagini, e funziona bene anche con batch piccoli. È una scelta tecnica ma importante: con BatchNorm e batch piccolo, la rete avrebbe imparato peggio.

I risultati di UNet: è il metodo più robusto. Passando da σ = 0.005 a σ = 0.1, la qualità scende da 29.5 dB a 28.5 dB — solo 1 dB di differenza. Per darvi un'idea: TV perde 6 dB nello stesso intervallo. UNet è quasi insensibile al livello di rumore.

UNet è anche il più veloce: elabora un'immagine in 0.03 secondi. TV ne impiega 7. DiffPIR ne impiega 3. UNet è 200 volte più veloce di TV e 100 volte più veloce di DiffPIR. Se doveste restaurare un video (30 immagini al secondo), solo UNet sarebbe abbastanza veloce per funzionare in tempo reale.

A σ = 0.1, UNet supera TV: 28.5 dB contro 26.5 dB. La rete ha visto tante immagini rumorose durante il training e ha imparato a gestirle. Il suo prior imparato dai dati è più ricco e flessibile del prior scritto a mano di TV.

---

## Slide 13-16 — DiffPIR

DiffPIR è il terzo metodo, il più complesso e il più interessante. PIR sta per "Plug-and-Play Image Restoration" — cioè un metodo dove potete "attaccare" diversi modelli di prior e funziona comunque. Diff sta per "Diffusion" — si basa sui modelli di diffusione, una tecnologia molto recente (2023).

Per capire DiffPIR, dobbiamo prima capire i modelli di diffusione. Immaginate di avere un cubetto di ghiaccio e di filmarlo mentre si scioglie in acqua. Il video mostra il processo diretto: da ghiaccio (strutturato, ordinato) ad acqua (disordinata, caotica). Un modello di diffusione impara il processo inverso: dall'acqua caotica tornare al ghiaccio ordinato.

Nel nostro caso, il "ghiaccio" è l'immagine pulita e l'acqua" è rumore puro. Il modello impara a partire dal rumore e generare un'immagine pulita. Questo già da solo è impressionante: significa che il modello sa come sono fatte le immagini, anche senza vederle. Ha imparato la "distribuzione" delle immagini, cioè com'è fatta in media un'immagine di cellule cervicali.

Ma noi non vogliamo generare immagini a caso: vogliamo restaurare un'immagine specifica. DiffPIR combina due cose: il modello generativo (che sa come sono fatte le immagini in generale) e i dati specifici (l'immagine degradata che abbiamo). A ogni passo, dice: "Il modello pensa che l'immagine pulita sia questa, ma deve anche spiegare la foto degradata che ho qui. Troviamo un compromesso."

Il processo completo: partiamo dall'immagine degradata, aggiungiamo un po' di rumore per "entrare" nel processo di diffusione, e poi facciamo 15 passi di denoising (rimozione rumore). A ogni passo, il modello fa due operazioni: prima usa il modello generativo per stimare l'immagine pulita, poi usa i dati originali per correggere questa stima (si chiama data-fidelity, fedeltà ai dati). Dopo 15 passi, otteniamo l'immagine finale.

**Perché 15 passi e non di più?** Perché ogni passo richiede tempo e computing. Con 15 passi abbiamo un buon equilibrio tra qualità e velocità. Con più passi il risultato migliora poco ma il tempo aumenta molto.

Un aspetto tecnico importante: il modello parte dal passo 50 della diffusione, non dal passo 1000. Perché? Perché la diffusione è un processo che va da t=0 (immagine pulita) a t=1000 (rumore puro). A t=1000, l'immagine è completamente rumore. Se partissimo da lì, dovremmo fare 1000 passi per arrivare all'immagine pulita. Inoltre, a t=1000, un piccolo errore nella stima del rumore viene amplificato 150 volte — l'immagine esploderebbe, diventerebbe tutta bianca o tutta nera.

A t=50, invece, l'immagine è ancora molto simile all'originale (il 97% dell'immagine originale è ancora lì), e un errore viene amplificato solo del 3%. È molto più stabile e sicuro. Partiamo da t=50 perché nella restoration (restauro) non partiamo da rumore puro, ma dall'immagine degradata che già contiene informazioni utili.

**Esempio della scala:** È come stare su una scala a pioli. Se siete in cima a una scala di 1000 gradini, un passo falso e cadete e vi fate male. Se siete al gradino 50, un passo falso è solo un piccolo inciampo. Noi preferiamo stare al gradino 50 perché è più sicuro e arriviamo comunque a destinazione.

**Esempio del passepartout:** Pensate a DiffPIR come a un detective che deve ricostruire un volto da un identikit imperfetto. Il detective ha visto migliaia di volti in carriera (il modello generativo). Quando guarda l'identikit, dice: "Questo naso non mi convince, di solito i nasi sono fatti così. Ma devo anche rispettare l'identikit perché il testimone ha detto che il naso era proprio così." Trova un compromesso tra la sua esperienza e l'evidenza. A volte il compromesso è perfetto, a volte sbaglia.

I risultati di DiffPIR sono i più interessanti perché controintuitivi. Normalmente, ci aspetteremmo che un metodo funzioni peggio con più rumore. DiffPIR fa l'opposto: migliora quando il rumore aumenta.

A σ = 0.005 (poco rumore), DiffPIR fa solo 15.8 dB di PSNR. È un risultato bassissimo, peggio di non fare niente. Perché? Perché il modello "inventa" dettagli che non ci sono. L'immagine è già quasi buona, ma il modello ci mette del suo e la peggiora. Si chiama hallucination (allucinazione), un fenomeno ben noto nei modelli generativi: quando i dati sono già buoni, il modello aggiunge cose inesistenti.

**Esempio di allucinazione:** È come se qualcuno vi desse una ricetta quasi perfetta e voi diceste "No, secondo me manca questo ingrediente" e aggiungeste qualcosa che nella ricetta originale non c'era. Invece di migliorare, peggiorate. Il modello generativo fa la stessa cosa: vede un'immagine quasi pulita e pensa "Non è abbastanza realistica, aggiungo dettagli per renderla più vera" ma quei dettagli sono falsi.

A σ = 0.1 (tanto rumore), invece, DiffPIR arriva a 25.5 dB. Non supera UNet (28.5 dB), ma il miglioramento è enorme rispetto a σ=0.005. Perché? Perché quando c'è tanto rumore, il modello può usare la sua capacità generativa per "riempire" i dettagli persi senza rischiare di contraddire l'immagine originale — tanto l'originale è già molto danneggiata.

Il punto di forza di DiffPIR è che ricostruisce texture fini che TV e UNet perdono. Per esempio, i dettagli all'interno del nucleo della cellula, che sono importanti per la diagnosi. TV li liscia via (troppo uniforme), UNet a volte li perde, ma DiffPIR li ricostruisce in modo realistico. Il punto debole è che non ci si può fidare ciecamente: bisogna sempre controllare il risultato perché potrebbe aver aggiunto dettagli falsi.

---

## Slide 17-19 — Risultati quantitativi

Ora mettiamo tutto insieme e confrontiamo i numeri. Abbiamo una tabella che mostra il PSNR (qualità) di ogni metodo a ciascun livello di rumore. È il momento della verità.

TV a σ=0.005: 32.4 dB. È il migliore a questo livello. L'immagine è quasi perfetta. TV a σ=0.01: 31.1 dB. Ancora molto buono. TV a σ=0.05: 28.1 dB. Inizia a calare. TV a σ=0.1: 26.5 dB. Ha perso 6 dB rispetto al punto di partenza. TV parte fortissimo ma non regge all'aumentare del rumore.

UNet a σ=0.005: 29.5 dB. Parte leggermente sotto TV (3 dB di differenza) ma ancora buono. UNet a σ=0.01: 29.3 dB. Quasi identico. UNet a σ=0.05: 28.9 dB. Solo 0.6 dB in meno. UNet a σ=0.1: 28.5 dB. Ha perso solo 1 dB in tutto. È incredibilmente stabile.

DiffPIR a σ=0.005: 15.8 dB. Molto basso, il modello allucina. DiffPIR a σ=0.01: 18.2 dB. Ancora basso. DiffPIR a σ=0.05: 23.1 dB. Inizia a recuperare. DiffPIR a σ=0.1: 25.5 dB. Recupero notevole, ma ancora sotto UNet.

**Esempio della gara:** TV è come un velocista dei 100 metri: parte fortissimo, ma se la gara è più lunga si stanca e perde posizioni. UNet è come un maratoneta: tiene lo stesso ritmo per tutta la gara, non importa quanto è lunga. DiffPIR è come uno che inciampa alla partenza, perde terreno, ma poi accelera e recupera posizioni nel finale.

**Esempio delle vacanze:** TV è come andare in un ristorante che fa solo un piatto, ma lo fa benissimo. Se ordinate quel piatto, siete a cavallo. Se ordinate altro, non è altrettanto bravo. UNet è come un ristorante che fa tanti piatti discreti: nessuno è eccellente, ma nessuno è cattivo. DiffPIR è come uno chef sperimentale: a volte vi stupisce con un piatto incredibile, a volte vi serve qualcosa di immangiabile.

La regola pratica per scegliere: se l'immagine è poco rumorosa (σ ≤ 0.01), usate TV, è il migliore. Se l'immagine è molto rumorosa (σ ≥ 0.05), usate UNet, è il più affidabile. Se avete un'immagine molto degradata e volete provare a recuperare dettagli persi, potete provare DiffPIR, ma controllate sempre il risultato perché potrebbe aver allucinato.

---

## Slide 20-24 — Risultati qualitativi

I numeri dicono molto, ma le immagini dicono ancora di più. Guardando le foto dei risultati, si capiscono cose che i numeri non mostrano.

A σ = 0.005 (poco rumore), le differenze sono sottili. TV è perfetto: bordi netti, sfondo uniforme e pulito, il nucleo della cellula è ben definito. UNet è leggermente meno definito: c'è un filo di sfocatura in più, ma è comunque un buon risultato. DiffPIR invece è diverso: ha zone omogenee con chiazze strane, i bordi sono deformati, sembra che qualcuno abbia ritoccato l'immagine con un pennello grossolano. Si vede subito che ha "inventato" dei dettagli.

A σ = 0.01, la situazione è simile. TV ancora buono, UNet stabile, DiffPIR ancora in difficoltà.

A σ = 0.05, le cose cambiano. TV inizia a mostrare lo staircasing: nelle zone del citoplasma, dove dovrebbe esserci una sfumatura graduale dal nucleo al bordo della cellula, si vedono dei gradini, come una scala. È come se l'immagine fosse stata disegnata con un pennarello grosso. UNet rimane stabile: si vede un po' di rumore residuo (grana fine), ma la struttura dell'immagine è intatta, i bordi delle cellule sono riconoscibili. DiffPIR inizia a dare il meglio: le texture fini (la grana all'interno del nucleo, i piccoli dettagli) sono ricostruite meglio di TV e quasi quanto UNet.

A σ = 0.1, le differenze sono nette. TV è molto degradato: lo staircasing è ovunque, i dettagli fini sono persi, le cellule sembrano fatte di blocchetti. UNet è ancora buono: c'è più rumore residuo rispetto a σ=0.05, ma i contorni delle cellule sono ancora chiaramente visibili, la forma è corretta. DiffPIR sorprende: ricostruisce i dettagli del nucleo meglio di TV e quasi quanto UNet, con una texture più naturale. Sembra meno "processato" di TV.

Le mappe di differenza (slide 24) sono immagini che mostrano dove ogni metodo sbaglia. Si creano sottraendo l'immagine ricostruita dall'originale: dove il risultato è bianco, l'errore è grande; dove è nero, l'errore è piccolo.

Per TV, le mappe mostrano macchie bianche sparse dappertutto — è il rumore residuo che TV non è riuscito a rimuovere. Le macchie sono distribuite uniformemente su tutta l'immagine. Per UNet, le mappe sono più scure (meno errore) e le zone bianche sono poche e piccole. Per DiffPIR, le mappe mostrano qualcosa di diverso: le zone bianche non sono sparse, ma concentrate sui bordi delle cellule. Significa che DiffPIR ha spostato leggermente i contorni delle cellule, sbagliando la posizione precisa dei bordi.

**Esempio del disegno:** Prendete il disegno di un cerchio. TV a basso rumore lo disegna perfetto. TV ad alto rumore lo disegna come un poligono, con tanti lati dritti (staircasing). UNet lo disegna come un cerchio leggermente sfuocato: non perfetto, ma chiaramente un cerchio. DiffPIR a basso rumore disegna un cerchio con una protuberanza in più (allucinazione). DiffPIR ad alto rumore disegna un cerchio ben fatto ma leggermente spostato rispetto all'originale.

---

## Slide 25 — Discussione

Ora confrontiamo le tre famiglie di metodi a livello concettuale. Ogni famiglia ha il suo carattere, i suoi vantaggi e i suoi svantaggi.

La famiglia variazionale, rappresentata da TV, ha un grande vantaggio: non ha bisogno di training. Potete prendere il programma, installarlo e usarlo subito su qualsiasi immagine. È trasparente: potete aprire il codice e capire esattamente cosa fa. Dà sempre la stessa risposta con gli stessi dati — non c'è casualità. Il suo svantaggio è che il "prior" (la conoscenza) è scritto a mano: è una regola semplice ("le immagini hanno pochi bordi") che non basta per immagini molto degradate.

La famiglia deep learning, rappresentata da UNet, ha il vantaggio di imparare dai dati: più dati avete, meglio funziona. È veloce: una volta addestrata, una rete neurale elabora un'immagine in millisecondi. È robusta: la qualità non crolla quando le condizioni cambiano (più rumore, diversi tipi di immagini). Il suo svantaggio è che ha bisogno di una fase di training con esempi etichettati (coppie sporco-pulito), che richiede tempo e dati. Inoltre, se l'immagine da restaurare è molto diversa da quelle viste in training, potrebbe non funzionare bene.

La famiglia generativa, rappresentata da DiffPIR, ha il vantaggio di ricostruire dettagli realistici: dove gli altri metodi lasciano zone vuote o sfocate, DiffPIR le riempie in modo plausibile. Migliora con il rumore: più rumore c'è, meglio esprime la sua capacità generativa. Il suo svantaggio è che può allucinare: mostrare dettagli che non esistono nell'immagine originale. Inoltre è lento (3 secondi per immagine) e complesso da mettere a punto.

**Esempio dei tre attrezzi:** TV è come un martello. Semplice, affidabile, capite esattamente come funziona. Se dovete piantare un chiodo, è l'attrezzo giusto. Se dovete svitare una vite, non funziona.

UNet è come un cacciavite elettrico. Più complesso, ma versatile. Funziona per avvitare, svitare, forare leggero. Una volta che l'avete, lo usate per tante cose. Ma se la batteria si scarica (non avete dati di training), non potete usarlo.

DiffPIR è come un trapano professionale. Potente, preciso, fa buchi perfetti. Ma se lo usate male, buca dove non dovete (allucinazione). E ci vuole tempo per imparare a usarlo bene.

Nessun metodo vince su tutti i fronti. La scelta dipende dal contesto specifico: quanto rumore c'è, quanto tempo avete a disposizione, se avete dati di training, quanto è critico il risultato (se sbagliare una diagnosi è grave, meglio un metodo più conservativo).

Il nostro confronto è affidabile perché abbiamo usato la stessa pipeline per tutti e tre: stessi dati di partenza, stesse metriche, stesse condizioni sperimentali. Le differenze nei risultati dipendono solo dai metodi stessi, non da fattori esterni.

---

## Slide 26 — Conclusioni

Cosa abbiamo imparato da questo progetto? Tre cose principali.

Primo: il confronto è equo e affidabile. Abbiamo progettato tutto con cura per garantire che i tre metodi fossero valutati nelle stesse identiche condizioni: stessi dati di partenza, stessa pipeline di degradazione, stesse metriche di valutazione, stesso split training/validation/test. I risultati sono quindi confrontabili e le conclusioni sono solide.

Secondo: non esiste il metodo perfetto. TV è il migliore a basso rumore ma cala rapidamente. UNet è il più stabile e versatile. DiffPIR ha il comportamento più interessante ma è imprevedibile. La scelta del metodo dipende sempre dal problema specifico. Un medico che deve fare una diagnosi su un'immagine leggermente degradata userà TV. Un laboratorio che processa centinaia di immagini al giorno con vari livelli di rumore userà UNet. Un ricercatore che vuole recuperare dettagli persi proverà DiffPIR.

Terzo: con una GPU avremmo potuto fare molto di più. Una GPU (scheda grafica potente, come quelle usate per i videogiochi) accelera enormemente il deep learning. Con GPU avremmo potuto:
- Usare un batch size più grande (più immagini elaborate insieme) per un training più stabile
- Fare più epoche di training per modelli più accurati
- Provare modelli più grandi e complessi
- Ridurre i tempi di training da 60 minuti a pochi minuti
- Elaborare le immagini più velocemente

Per il futuro, abbiamo tre idee. Per TV: un λ che si adatta automaticamente al livello di rumore, invece di tenerlo fisso. Se l'immagine ha poco rumore, λ piccolo (regolarizzazione debole, si preservano i dettagli). Se ha molto rumore, λ grande (regolarizzazione forte, si rimuove il rumore ma si accetta lo staircasing).

Per UNet: training su GPU per provare architetture più grandi. Per esempio, una UNet con più canali, più strati, o varianti come Attention UNet che aggiunge un meccanismo di "attenzione" per focalizzarsi sulle zone più importanti dell'immagine.

Per tutti: validazione su immagini cliniche reali, non solo degradate artificialmente. Perché nella realtà, la degradazione non è mai perfettamente gaussiana. Ci possono essere artefatti diversi: macchie, pieghe del vetrino, illuminazione non uniforme. Solo testando su immagini cliniche vere possiamo capire se i metodi funzionano davvero in un ambiente reale.

**Esempio finale:** È come se avessimo costruito tre prototipi in laboratorio. I test in laboratorio sono stati superati: tutti e tre funzionano, ognuno con i suoi punti di forza. Il prossimo passo è portarli fuori dal laboratorio e testarli in un ospedale vero, con immagini vere di pazienti reali. Solo allora sapremo se sono davvero pronti per aiutare i medici nella diagnosi.

---

## Slide 27 — Bibliografia

Le slide finiscono con la bibliografia, cioè l'elenco degli articoli scientifici su cui si basa il progetto.

L'articolo principale per TV è "Nonlinear total variation based noise removal algorithms" di Rudin, Osher e Fatemi, pubblicato nel 1992. È uno degli articoli più citati nella storia del restauro di immagini. Ha introdotto il concetto di Total Variation per la rimozione del rumore e ha aperto una strada di ricerca che dura ancora oggi.

L'articolo principale per UNet è "U-Net: Convolutional Networks for Biomedical Image Segmentation" di Ronneberger, Fischer e Brox, pubblicato nel 2015. Questo articolo ha presentato l'architettura a U con skip connections ed è diventato uno standard nel campo delle immagini mediche. È stato citato decine di migliaia di volte.

L'articolo principale per DiffPIR è "DiffPIR: Denoising Diffusion Models for Plug-and-Play Image Restoration" di Zhu e colleghi, pubblicato nel 2023. È un articolo molto recente che combina i modelli di diffusione (la tecnologia alla base di Dall-E e Stable Diffusion) con il restauro di immagini. Il nome "Plug-and-Play" nel titolo significa che il metodo può funzionare con diversi modelli di prior, non solo uno specifico.
