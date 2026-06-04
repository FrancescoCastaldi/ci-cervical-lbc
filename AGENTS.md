# AGENTS.md — Guida operativa del progetto

Questo file descrive la struttura del progetto, la divisione dei task tra i due studenti,
la roadmap giorno per giorno e le convenzioni da seguire.

---

## Componenti del gruppo

| Studente | GitHub | Ruolo principale |
|---|---|---|
| Francesco Castaldi | @FrancescoCastaldi | TV + Evaluation + Report |
| Paolo Fusco | @PaoloFusco19 | UNet + DiffPir + Slides |

---

## Stack tecnologico

- Python 3.10+
- PyTorch >= 2.0
- scikit-image (PSNR, SSIM)
- matplotlib, pandas
- tqdm
- Jupyter
- Git + GitHub

---

## Roadmap giorno per giorno

### Giorni 1-2 — Dataset e preprocessing
**Owner: Francesco**

- [ ] Scaricare dataset Mendeley LBC da https://data.mendeley.com/datasets/zddtpgzv63/2
- [ ] Mettere le immagini in `data/raw/` con struttura per classe
- [ ] Lanciare `python scripts/preprocess.py` e verificare output
- [ ] Aprire `notebooks/01_eda.ipynb` e eseguire tutte le celle
- [ ] Verificare split: `data/splits/train.txt`, `val.txt`, `test.txt`
- [ ] Verificare esempi visivi in `data/degraded/examples/`
- [ ] Committare split e notebook output

**Deliverable:** Split fissi pronti, EDA completata, esempi degraded visivi salvati.

---

### Giorni 3-4 — Degradazione
**Owner: Francesco**

- [ ] Verificare `src/degradation/degradation.py` (blur gaussiano sigma=2, kernel=9)
- [ ] Verificare che tutti i 4 noise levels siano generati: 0.005, 0.01, 0.05, 0.1
- [ ] Confermare che train/val/test abbiano tutti i file `.pt` generati
- [ ] Salvare almeno 6 esempi visivi GT vs degraded per ogni noise level
- [ ] Documentare le scelte di degradazione nel report (sezione setup)

**Deliverable:** `data/degraded/` completo, esempi visivi pronti per il report.

---

### Giorni 5-6 — Metodo TV
**Owner: Francesco**

- [ ] Rivedere implementazione in `src/methods/tv/tv.py`
- [ ] Lanciare `python scripts/run_tv.py`
- [ ] Testare lambda_reg su piccolo subset (10-20 immagini) con valori: 0.01, 0.05, 0.1, 0.5
- [ ] Scegliere il lambda migliore osservando PSNR e visivamente
- [ ] Aggiornare `configs/experiment.yaml` con il valore scelto
- [ ] Lanciare TV su tutto il test set
- [ ] Salvare `results/tv_results.csv`
- [ ] Salvare 6 immagini qualitative in `results/qualitative/tv/`
- [ ] Documentare la scelta del parametro

**Deliverable:** `results/tv_results.csv` con PSNR/SSIM per tutti i noise levels.

---

### Giorni 7-9 — UNet
**Owner: Paolo**

- [ ] Rivedere architettura in `src/methods/unet/unet.py`
- [ ] Configurare iperparametri in `configs/experiment.yaml` (lr, batch_size, epochs)
- [ ] Lanciare training: `python scripts/run_unet.py`
- [ ] Monitorare loss su train e val
- [ ] Se overfitting: ridurre epochs o aggiungere dropout
- [ ] Salvare checkpoint in `results/unet_weights.pth`
- [ ] Eseguire evaluation su test set
- [ ] Salvare `results/unet_results.csv`
- [ ] Salvare 6 immagini qualitative in `results/qualitative/unet/`

**Deliverable:** `results/unet_results.csv` + checkpoint salvato.

---

### Giorni 10-11 — DiffPir
**Owner: Paolo**

- [ ] Clonare repo ufficiale DiffPIR: https://github.com/yuanzhi-zhu/DiffPIR
- [ ] Scaricare pesi pretrained indicati nel README di DiffPIR
- [ ] Adattare il wrapper in `src/methods/diffpir/diffpir.py`
- [ ] Testare su 5-10 immagini per verificare il funzionamento
- [ ] Scegliere numero di step (punto di partenza: 100, testare anche 50 e 200)
- [ ] Eseguire evaluation su test set (solo noise levels 0.01 e 0.1 se troppo lento)
- [ ] Salvare `results/diffpir_results.csv`
- [ ] Salvare 6 immagini qualitative in `results/qualitative/diffpir/`
- [ ] Documentare configurazione usata

**Deliverable:** `results/diffpir_results.csv` + note sulla configurazione.

---

### Giorni 12-13 — Evaluation e confronto
**Owner: entrambi**

- [ ] Unire i tre CSV in `results/all_results.csv`
- [ ] Generare tabella PSNR/SSIM per metodo e noise level
- [ ] Generare plot PSNR vs noise level (tutti i metodi sulla stessa figura)
- [ ] Generare plot SSIM vs noise level
- [ ] Selezionare 4-6 casi significativi (successi e fallimenti)
- [ ] Creare figura comparativa: degraded | TV | UNet | DiffPir | GT
- [ ] Salvare tutto in `results/figures/`
- [ ] Scrivere discussione critica: quando vince ciascun metodo?

**Deliverable:** `results/all_results.csv`, `results/figures/` completa.

---

### Giorno 14 — Report, slide, repo
**Owner: Francesco → report | Paolo → slides**

**Report (`report/`):**
- [ ] Sezione 1: Introduzione e formulazione del problema
- [ ] Sezione 2: Dataset e preprocessing
- [ ] Sezione 3: Degradazione
- [ ] Sezione 4: Metodi (TV, UNet, DiffPir)
- [ ] Sezione 5: Setup sperimentale
- [ ] Sezione 6: Risultati (tabelle + figure)
- [ ] Sezione 7: Discussione e conclusioni

**Slides (`slides/`):**
- [ ] Slide 1: Titolo, gruppo, corso
- [ ] Slide 2: Problema e dataset
- [ ] Slide 3: Pipeline di degradazione
- [ ] Slide 4: Metodo TV
- [ ] Slide 5: Metodo UNet
- [ ] Slide 6: Metodo DiffPir
- [ ] Slide 7: Risultati quantitativi (tabella + plot)
- [ ] Slide 8: Risultati qualitativi (figura comparativa)
- [ ] Slide 9: Discussione critica
- [ ] Slide 10: Conclusioni

**Repo:**
- [ ] Aggiornare `README.md` con link GitHub e istruzioni finali
- [ ] Verificare che tutti gli script girino da zero su repo clonata
- [ ] Pulire file temporanei e non necessari
- [ ] Tag finale: `git tag v1.0.0 && git push --tags`

---

## Convenzioni Git

### Branch
```
main          → codice stabile
feature/tv    → sviluppo metodo TV
feature/unet  → sviluppo UNet
feature/diffpir → sviluppo DiffPir
```

### Commit message
```
feat: nuova funzionalità
fix: correzione bug
docs: documentazione
chore: manutenzione (gitignore, config, ecc.)
exp: risultati sperimentali
```

---

## Struttura output attesa

```
results/
├── tv_results.csv
├── unet_results.csv
├── diffpir_results.csv
├── all_results.csv
├── figures/
│   ├── psnr_vs_noise.png
│   ├── ssim_vs_noise.png
│   └── qualitative_comparison.png
└── qualitative/
    ├── tv/
    ├── unet/
    └── diffpir/
```

---

## Parametri da discutere all'orale

| Metodo | Parametro | Valore usato | Come scelto |
|---|---|---|---|
| TV | lambda_reg | da definire | Grid search su val set |
| UNet | lr, batch_size, epochs | da definire | Curva di loss |
| DiffPir | num_steps | da definire | Trade-off qualità/tempo |

---

## Checklist finale pre-orale

- [ ] Tutti i metodi girano con `python scripts/run_*.py`
- [ ] `results/all_results.csv` presente e completo
- [ ] Almeno 4 figure comparative pronte
- [ ] Report PDF finalizzato
- [ ] Slide pronte con tutti i deliverable
- [ ] Link GitHub funzionante e repo pubblica
- [ ] README aggiornato con istruzioni di setup
- [ ] Parametri di ogni metodo documentati e motivati
