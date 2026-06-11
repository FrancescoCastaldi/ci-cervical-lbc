# Riassunto Esame — Call con Paolo

**Corso:** Computational Imaging — Università di Bologna, LM Informatica
**Prof:** Picciolomini & Evangelista
**Gruppo R:** Francesco Castaldi & Paolo Fusco
**Repo:** `https://github.com/FrancescoCastaldi/ci-cervical-lbc`

---

## 1. Cosa Chiede l'Assignment

### Task
**Deblur + Denoise** di immagini di citologia cervicale — problema inverso.

### Degradazione (uguale per tutti i metodi)
- **Blur gaussiano:** σ=2, kernel 9×9
- **AWGN:** 4 livelli — σₙ ∈ {0.005, 0.01, 0.05, 0.1}
- **Dataset:** subset ~4000 immagini da Mendeley LBC Cervical Cancer, resize 256×256

> **Nota importante:** Il dataset contiene effettivamente **962 immagini totali** disponibili (non 4000). Abbiamo usato tutto il dataset, dichiarato e giustificato nel report (§2.1).

### Metodi Richiesti (4, noi ne scegliamo 3 perché siamo in 2)

| Metodo | Famiglia | Scelto? |
|---|---|---|
| Total Variation (TV) | Variazionale | ✅ Fatto da Paolo |
| UNet | End-to-end | ✅ Fatto (scelto vs ViT e NAF-Net) |
| DiffPIR | Generativo (diffusione) | ✅ Fatto da Francesco |
| Weighted TV | Ibrido | ❌ Escluso (siamo in 2) |

### Deliverables — tutti completati
1. ✅ **Metriche quantitative:** PSNR + SSIM per ogni metodo e noise level
2. ✅ **Immagini qualitative:** GT vs Degradata vs Ricostruita (24 per metodo = 72 totali)
3. ✅ **Comparison plot:** grafico a barre che confronta tutti i metodi
4. ✅ **Report** (`report/relazione.md`, 673 righe)
5. ✅ **Codice su GitHub** (modulare, documentato, con test)
6. ✅ **PowerPoint** (`slides/presentazione.pptx`, 10 slide)
7. ✅ **Test unitari** (34 test, tutti verdi)

---

## 2. Cosa Abbiamo Fatto

### 2.1 Dataset & Preprocessing
| Componente | File | Dettaglio |
|---|---|---|
| EDA | `notebooks/01_eda.ipynb` | 962 img, 4 classi (NILM 63.6%, HSIL 16.9%, LSIL 11.7%, SCC 7.7%) |
| Resize + norm | `src/data/dataset.py` | 256×256, normalizzazione [-1, 1] |
| Split | `configs/experiment.yaml` | 70/15/15 stratificato (673/144/145), seed 42 |

### 2.2 Degradazione (condivisa)
| Componente | File | Dettaglio |
|---|---|---|
| Blur + AWGN | `src/degradation/degradation.py` | σ=2 kernel=9, poi AWGN a 4 livelli |
| Stessa per tutti | Usata da TV, UNet, DiffPIR | Confronto equo garantito |

### 2.3 TV — Total Variation (Paolo)
| Componente | File | Dettaglio |
|---|---|---|
| Algoritmo | `src/methods/tv/tv.py` | λ=0.1, 300 iter Adam |
| Esecuzione | `scripts/run_tv.py` | 145 test img × 4 noise level |
| Risultati TV | σₙ=0.005: **32.09 dB** / 0.01: 32.04 / 0.05: 30.42 / 0.1: 26.54 |
| Qualitative | `results/tv/qualitative/` | 24 immagini |

### 2.4 UNet — End-to-End (implementato da Francesco, assegnato a Paolo)
| Componente | File | Dettaglio |
|---|---|---|
| Architettura | `src/methods/unet/unet.py` | **1.9M params** (features=[16,32,64,128]), GroupNorm, 4 canali input (RGB + noise map) |
| Training | `scripts/run_unet.py` | 50 epoche CPU, L1 loss, ReduceLROnPlateau, best model checkpoint |
| Eval | `scripts/eval_unet.py` | 145 test img × 4 noise level |
| Risultati UNet | σₙ=0.005: **29.89 dB** / 0.01: 29.89 / 0.05: 29.63 / 0.1: **28.93** |
| Qualitative | `results/unet/qualitative/` | 24 immagini |
| Notebook | `notebooks/03_unet.ipynb` | Completato |
| Test | `tests/test_unet.py` | 7 test (forward, gradient, params, doubleconv, seed, batch, noise cond) |

### 2.5 DiffPIR — Generativo (Francesco)
| Componente | File | Dettaglio |
|---|---|---|
| Modello DDPM | `src/methods/diffpir/model.py` | LightUNet custom (1.26M params) |
| Algoritmo | `src/methods/diffpir/diffpir.py` | FFT data-fidelity, 15 step DDIM, t_start=50 |
| Training | `src/methods/diffpir/train.py` | Addestrato su LBC |
| Risultati DiffPIR | σₙ=0.005: 16.67 / 0.01: 17.32 / 0.05: 22.49 / 0.1: **24.68** dB |
| Qualitative | `results/diffpir/qualitative/` | 24 immagini |
| Notebook | `notebooks/04_diffpir.ipynb` | Completato |

### 2.6 Report & Presentazione
| Componente | File | Dettaglio |
|---|---|---|
| Report principale | `report/relazione.md` | 7 sezioni, 673 righe, completa |
| Teoria | `report/teoria.md` | Fondamenti (problema inverso, metodi, metriche) |
| Riassunto notebook | `report/notebook.md` | Tabelle comparative |
| PowerPoint | `slides/presentazione.pptx` | 10 slide, generato via script Python |
| Script PPTX | `slides/generate_pptx.py` | Generazione automatica |

### 2.7 Codice & Test
| Componente | Dettaglio |
|---|---|
| **34 test unitari** | degradation:10, metrics:9, unet:7, diffpir:7, data:1 — **tutti passano** |
| README | `README.md` — completo |
| AGENTS.md | Divisione metodi e stato |
| GitHub | Pushato su `origin/master` |

---

## 3. Risultati Principali

### Tabella Comparativa Finale

| σₙ | **TV** PSNR | **TV** SSIM | **UNet** PSNR | **UNet** SSIM | **DiffPIR** PSNR | **DiffPIR** SSIM |
|---|---|---|---|---|---|---|
| **0.005** | **32.09 dB** 🥇 | 0.911 🥇 | 29.89 dB 🥈 | 0.894 🥈 | 16.67 dB | 0.235 |
| **0.01** | **32.04 dB** 🥇 | 0.909 🥇 | 29.89 dB 🥈 | 0.894 🥈 | 17.32 dB | 0.270 |
| **0.05** | **30.42 dB** 🥇 | 0.837 🥇 | 29.63 dB 🥈 | **0.875** 🥇 | 22.49 dB | 0.512 |
| **0.1** | 26.54 dB 🥉 | 0.586 | **28.93 dB** 🥇 | **0.830** 🥇 | 24.68 dB 🥈 | 0.664 🥈 |

### Chi Vince Cosa

| Categoria | Vincitore | Dettaglio |
|---|---|---|
| **Basso rumore** (0.005) | **TV** (32.09 dB) | Metodo classico eccelle quando il degrado è lieve |
| **Medio rumore** (0.05) | **TV** (30.42 dB) | Ma UNet è a solo 0.8 dB di distanza (29.63) |
| **Alto rumore** (0.1) | **UNet** (28.93 dB) | Supera TV di 2.4 dB! Il deep learning regge meglio il rumore forte |
| **Secondo ad alto rumore** | **DiffPIR** (24.68 dB) | Generativo più robusto del TV, ma sotto UNet |
| **Velocità inferenza** | **UNet** (0.035 s/img) | 10× più veloce di TV, 60× più veloce di DiffPIR |
| **Nessun training** | **TV** | Funziona subito, nessun dato necessario |

### Evoluzione UNet durante il training (50 epoche CPU)

Il training è partito da Val PSNR 13.44 dB (epoca 1) ed è migliorato costantemente fino a **29.84 dB** (epoca 45).

- **Prima ottimizzazione:** features=[16,32,64,128] (1.9M params vs vecchi 31M) → 60× più piccolo, training 50 epoche in ~85 min invece di 1 epoca in 82 min
- **Miglioramenti architetturali:** GroupNorm invece di BatchNorm, L1 loss invece di MSE, condizionamento noise level (4 canali input)
- **Risultato:** da 24 dB (1 epoca) a **29.89 dB** (50 epoche) — netto miglioramento

### Il "Comportamento Strano" di DiffPIR

PSNR **migliora** all'aumentare del rumore (16.7→24.7 dB). Sembra controintuitivo ma è normale:
- **A basso rumore:** il modello di diffusione rimuove anche dettagli fini (li scambia per rumore) → PSNR penalizzato rispetto a GT
- **Ad alto rumore:** il modello pulisce aggressivamente e il guadagno netto è maggiore → PSNR più alto

---

## 4. Cosa Dire all'Orale

### Punti da sottolineare
1. **Confronto equo:** stessi input degradati (stessa funzione `degrade()`), seed fisso 42, stesse metriche
2. **Scelte motivate dei parametri:**
   - **TV:** λ=0.1 determinato euristicamente (trade-off data fidelity / regolarizzazione)
   - **UNet:** scelto su ViT (troppo data-hungry, O(n²) attention su 256×256) e NAF-Net (più complesso, no benefici chiari)
   - **DiffPIR:** t_start=50 (non 1000, per stabilità numerica: a t=1000 il fattore di amplificazione rumore è 150×)
3. **Successi e fallimenti:**
   - TV: eccellente a basso rumore, soffre di staircasing ad alto rumore e λ fisso
   - UNet: quasi alla pari col TV dopo 50 epoche CPU, batte TV a σ=0.1
   - DiffPIR: potente ad alto rumore ma lento (2s/img) e debole a basso rumore
4. **Limiti:** training CPU-only, dataset 962 img (non 4000 come da specifica)

### Divisione suggerita slide
| Slide | Relatore | Contenuto |
|---|---|---|
| 1-2 | Entrambi | Titolo, task, dataset, degradazione |
| 3-4 | **Paolo** | TV — teoria, implementazione, risultati |
| 5-6 | **Paolo** | UNet — architettura, training, risultati |
| 7-8 | **Francesco** | DiffPIR — teoria, implementazione, risultati |
| 9-10 | Entrambi | Confronto, conclusioni, domande |

---

## 5. Cosa Discutere con Paolo nella Call

### Decisioni da prendere
- [ ] **Chi presenta UNet?** Nel repo risulta assegnato a Paolo, ma l'ho implementato io. Vi mettete d'accordo.
- [ ] **Dividersi le slide** — chi prepara quali slide? Io ho già generato il PowerPoint di base (10 slide) con script Python. Vogliamo personalizzarlo?
- [ ] **Provare la demo live?** I notebook funzionano. DiffPIR impiega ~20s per 10 immagini, UNet è istantaneo. Possiamo mostrare 1-2 esempi dal vivo.
- [ ] **Vogliamo aggiungere Weighted TV come extra?** Non richiesto, ma sarebbe un plus per l'orale.

### Cosa è già pronto (non preoccuparsi)
- ✅ **Codice completo** su GitHub
- ✅ **Report** (`report/relazione.md`, 673 righe)
- ✅ **PowerPoint** generato (10 slide)
- ✅ **34 test** che passano
- ✅ **72 immagini qualitative** (24 per metodo)
- ✅ **Comparison plot** (`results/comparison.png`)
- ✅ **Tutti i risultati** in metrics.csv per ogni metodo
- ✅ **Notebook** 01, 02, 03, 04 completi

### Cose da verificare insieme
- Il **README su GitHub** mostra ancora UNet come "Da eseguire" nella vista iniziale — potrebbe essere un cache di GitHub. Il commit `fedff52` ha già aggiornato tutto.
- La **comparison.png** va rigenerata al volo se si cambiano risultati.
- Il **PowerPoint** può essere aperto con PowerPoint o Google Slides.

### Link utili
| Cosa | Dove |
|---|---|
| **Repo GitHub** | `https://github.com/FrancescoCastaldi/ci-cervical-lbc` |
| **Assignment PDF** | `slides/Exam_Assignment/Group R.pdf` |
| **Report completo** | `report/relazione.md` |
| **Presentazione** | `slides/presentazione.pptx` |
| **Comparison plot** | `results/comparison.png` |
| **Dataset (Mendeley)** | `https://data.mendeley.com/datasets/zddtpgzv63/2` |

---

*Generato il 10/06/2026 — se i risultati cambiano, rigenerare con: `python scripts/run_unet.py && python scripts/run_diffpir.py && python scripts/eval_unet.py && python scripts/plot_results.py`*
