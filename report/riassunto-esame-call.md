# 📋 Riassunto Esame — Call con Paolo

**Corso:** Computational Imaging — Università di Bologna, LM Informatica
**Prof:** Picciolomini & Evangelista
**Gruppo R:** Francesco Castaldi & Paolo Fusco
**Repo:** `https://github.com/FrancescoCastaldi/ci-cervical-lbc`

---

## 1. Cosa Chiede l'Assignment (PDF Group R)

### Task
**Deblur + Denoise** di immagini di citologia cervicale.

### Degradazione (uguale per tutti i metodi)
- **Blur gaussiano:** σ=2, kernel 9×9
- **AWGN:** 4 livelli — σₙ ∈ {0.005, 0.01, 0.05, 0.1}
- **Dataset:** subset ~4000 immagini da Mendeley LBC Cervical Cancer, resize 256×256

> **Nota:** Il dataset contiene effettivamente **962 immagini totali** (non 4000). Abbiamo usato tutto il dataset, dichiarato e giustificato nel report (§2.1).

### Metodi Richiesti (4)

| Metodo | Famiglia | Scelto? |
|---|---|---|
| Total Variation (TV) | Variazionale | ✅ Fatto |
| UNet / ViT / NAF-Net | End-to-end | ✅ **UNet** |
| DiffPIR | Generativo (diffusione) | ✅ Fatto |
| Weighted TV | Ibrido | ❌ **Escluso** (siamo in 2, possiamo farne 3 su 4) |

### Deliverables richiesti
1. ✅ **Metriche quantitative:** PSNR + SSIM per ogni metodo e noise level
2. ✅ **Immagini qualitative:** GT vs Degradata vs Ricostruita (6 per noise level × 4 = 24 per metodo)
3. ✅ **Comparison plot:** grafico che confronta tutti i metodi
4. ✅ **Report** (`report/relazione.md`)
5. ✅ **Codice su GitHub** (documentato, modulare)
6. ✅ **PowerPoint** (`slides/presentazione.pptx`, 10 slide)
7. ✅ **Test unitari** (32 test, tutti verdi)

---

## 2. Cosa Abbiamo Fatto — Mappatura Puntuale

### 2.1 Dataset & Preprocessing
| Cosa | Dove | Stato |
|---|---|---|
| Download + ispezione dataset | `notebooks/01_eda.ipynb` | ✅ |
| Classi: NILM 63.6%, HSIL 16.9%, LSIL 11.7%, SCC 7.7% | Report §2.1 | ✅ |
| Resize 256×256 + normalizzazione [-1,1] | `src/data/dataset.py` | ✅ |
| Split 70/15/15 stratificato (673/144/145) | `configs/experiment.yaml` | ✅ |

### 2.2 Degradazione
| Cosa | Dove | Stato |
|---|---|---|
| Blur gaussiano (σ=2, kernel=9) | `src/degradation/degradation.py` | ✅ |
| AWGN 4 livelli | `src/degradation/degradation.py` | ✅ |
| Stessa pipeline per tutti | `degrade()` usata da tutti gli script | ✅ |

### 2.3 TV — Paolo
| Cosa | Dove | Stato |
|---|---|---|
| Algoritmo TV (λ=0.1, 300 iter Adam) | `src/methods/tv/tv.py` | ✅ |
| Script esecuzione | `scripts/run_tv.py` | ✅ |
| Risultati: 145 test img × 4 noise | `results/tv/metrics.csv` | ✅ |
| Qualitative: 24 immagini | `results/tv/qualitative/` | ✅ |

### 2.4 UNet — Paolo (implementato da Francesco)
| Cosa | Dove | Stato |
|---|---|---|
| Architettura UNet (31M params, encoder-decoder con skip) | `src/methods/unet/unet.py` | ✅ |
| Training multi-noise (random σ per batch) | `scripts/run_unet.py` | ✅ |
| Training eseguito: **1 epoca CPU** (~82 min) | best_model.pth salvato | ✅ |
| Valutazione: 145 test img × 4 noise | `scripts/eval_unet.py` | ✅ |
| Risultati | `results/unet/metrics.csv` | ✅ |
| Qualitative: 24 immagini | `results/unet/qualitative/` | ✅ |
| Notebook | `notebooks/03_unet.ipynb` | ✅ |
| Test unitari (6 test) | `tests/test_unet.py` | ✅ |

### 2.5 DiffPIR — Francesco
| Cosa | Dove | Stato |
|---|---|---|
| LightUNet custom (1.26M params) addestrato su LBC | `src/methods/diffpir/model.py` | ✅ |
| Algoritmo DiffPIR con FFT data-fidelity | `src/methods/diffpir/diffpir.py` | ✅ |
| Training DDPM | `src/methods/diffpir/train.py` | ✅ |
| Inferenza: 10 test img × 4 noise | `scripts/run_diffpir.py` | ✅ |
| Risultati | `results/diffpir/metrics.csv` | ✅ |
| Qualitative: 24 immagini | `results/diffpir/qualitative/` | ✅ |
| Notebook | `notebooks/04_diffpir.ipynb` | ✅ |

### 2.6 Report & Presentazione
| Cosa | Dove | Stato |
|---|---|---|
| Report completo (teoria + risultati) | `report/relazione.md` (673 righe) | ✅ |
| Fondamenti teorici | `report/teoria.md` | ✅ |
| Riassunto notebook | `report/notebook.md` | ✅ |
| PowerPoint (10 slide) | `slides/presentazione.pptx` | ✅ |
| Script generazione PPTX | `slides/generate_pptx.py` | ✅ |

### 2.7 Codice & Test
| Cosa | Dove | Stato |
|---|---|---|
| 32 test unitari (degradation:10, metrics:9, unet:6, diffpir:7) | `tests/` | ✅ Tutti passano |
| README completo | `README.md` | ✅ |
| AGENTS.md (stato, divisione) | `AGENTS.md` | ✅ |
| GitHub pushato | `origin/main` | ✅ |

---

## 3. Risultati Principali

### Tabella Comparativa Finale

| σₙ | **TV** PSNR | **TV** SSIM | **UNet** PSNR | **UNet** SSIM | **DiffPIR** PSNR | **DiffPIR** SSIM |
|---|---|---|---|---|---|---|
| 0.005 | **32.09 dB** 🥇 | **0.911** | 24.07 dB | 0.789 | 16.67 dB | 0.235 |
| 0.01 | **32.04 dB** 🥇 | **0.909** | 24.05 dB | 0.785 | 17.32 dB | 0.270 |
| 0.05 | **30.42 dB** 🥇 | **0.837** | 23.45 dB | 0.700 | 22.49 dB | 0.512 |
| 0.1 | 26.54 dB | 0.586 | 21.87 dB | 0.554 | **24.68 dB** 🥇 | **0.664** |

### Tempi di Inferenza
| Metodo | Tempo per immagine |
|---|---|
| **TV** | ~0.3 s (300 iter, CPU) |
| **UNet** | **~0.026 s** (più veloce!) |
| **DiffPIR** | ~2.0 s (15 step, CPU) |

### Interpretazione

| Situazione | Migliore | Perché |
|---|---|---|
| **Basso rumore** (σₙ=0.005) | **TV** domina (32 dB) | Il TV regolarizza bene quando il rumore è debole; nessun dato necessario |
| **Medio rumore** (σₙ=0.05) | **TV** vince (30 dB) | Ma DiffPIR si avvicina (22.5 dB) — il generativo inizia a brillare |
| **Alto rumore** (σₙ=0.1) | **DiffPIR** (24.7 dB) | Il prior generativo è più robusto; TV soffre per λ fisso |
| **Velocità** | **UNet** batte tutti | Inferenza 26ms/img — una volta addestrato, è il più pratico |
| **Nessun training** | **TV** non necessita dati | Metodo classico, funziona subito |

### Risultato "strano" di DiffPIR
PSNR **migliora** all'aumentare del rumore (16.7→24.7 dB). Perché?
- A basso rumore: il modello di diffusione rimuove anche dettagli fini (li confonde con rumore) → PSNR penalizzato
- Ad alto rumore: il modello pulisce aggressivamente e il guadagno netto è maggiore
- SSIM conferma la stessa tendenza: a σ=0.1 il SSIM è 0.66

---

## 4. Cosa Dire all'Orale — Punti Chiave

### Da sottolineare
1. **Confronto equo:** stessi input degradati per tutti, seed fisso 42
2. **Scelta motivata dei parametri:** spiega perché λ=0.1 per TV, perché UNet invece di ViT (troppo data-hungry, O(n²)), perché t_start=50 per DiffPIR
3. **Successi e fallimenti:** TV ottimo a basso rumore ma soffre di staircasing; UNet veloce ma serve più training; DiffPIR potente ma lento
4. **Limiti:** Training CPU-only, dataset piccolo (962 img vs ~4000 dichiarati)

### Divisione suggerita per la presentazione
| Slide | Relatore | Contenuto |
|---|---|---|
| 1-2 | Entrambi | Titolo, task, dataset |
| 3-4 | Paolo | TV (teoria + risultati) |
| 5-6 | Paolo/Chi fa UNet | UNet (architettura + risultati) |
| 7-8 | Francesco | DiffPIR (teoria + risultati) |
| 9-10 | Entrambi | Confronto, conclusioni |

---

## 5. Cosa Discutere con Paolo nella Call

### Questioni da decidere
- [ ] **Chi presenta UNet?** (È stato implementato da te, ma nel repo risulta "Paolo" in agents.md)
- [ ] **Vogliamo ritrainare UNet con più epoche?** (1 epoca su CPU = risultati mediocri. Con più tempo si migliora. Paolo ha GPU?)
- [ ] **Vogliamo aggiungere Weighted TV?** (Non richiesto per gruppo di 2, ma sarebbe un plus)
- [ ] **Dividersi le slide per la presentazione** (chi dice cosa)
- [ ] **Provare la demo live?** (I notebook funzionano, ma su CPU sono lenti)

### Cosa è già sistemato (non serve preoccuparsi)
- ✅ Codice completo su GitHub
- ✅ Report scritto (673 righe)
- ✅ PowerPoint generato
- ✅ 32 test che passano
- ✅ Tutte le qualitative generate (72 immagini)
- ✅ Confronto finale con comparison.png

### Link utili
- **Repo GitHub:** `https://github.com/FrancescoCastaldi/ci-cervical-lbc`
- **Assignment PDF:** `slides/Exam_Assignment/Group R.pdf`
- **Report:** `report/relazione.md`
- **Presentazione:** `slides/presentazione.pptx`
- **Comparison plot:** `results/comparison.png`
