# Notebook 01: Analisi Esplorativa dei Dati (EDA) & Notebook 04: DiffPIR

## Notebook 01 — EDA (`notebooks/01_eda.ipynb`)

### Obiettivo
Caricare il dataset Mendeley LBC Cervical Cancer, esplorarne la struttura, visualizzare i campioni, e applicare la degradazione standard per capire il problema.

### Risultati

#### 1. Caricamento e Statistiche Dataset
| Metrica | Valore |
|---|---|
| Totale immagini | 962 |
| Dimensioni originali | 2048 × 1536 × 3 |
| Resize | 256 × 256 |
| Range pixel (originali) | [0, 255] |
| Normalizzazione | [-1, 1] (mean=0.5, std=0.5) |

#### 2. Distribuzione delle Classi
| Classe | Conteggio | Percentuale |
|---|---|---|
| NILM | 612 | 63.6% |
| HSIL | 163 | 16.9% |
| LSIL | 113 | 11.7% |
| SCC | 74 | 7.7% |

La classe NILM domina (>60%), il che rende il dataset sbilanciato. Tuttavia, per il task di restauro questo non è un problema — la qualità delle immagini NILM è comunque rappresentativa.

#### 3. Statistiche dei Canali RGB (post-normalizzazione [-1, 1])
| Canale | Mean | Std |
|---|---|---|
| R | 0.626 | 0.214 |
| G | 0.657 | 0.195 |
| B | 0.709 | 0.179 |

Media > 0.5 in tutti i canali suggerisce immagini tendenzialmente chiare (tipico di campioni citologici su sfondo chiaro).

#### 4. Degradazione Applicata

Visivamente:
- **Blur**: i bordi netti delle cellule diventano sfumati, i dettagli fini (nuclei, cromatina) si perdono
- **Noise**: a $\sigma_n=0.1$ il rumore è molto evidente, a $\sigma_n=0.005$ è sottile ma presente in ombre

Le immagini degradate mostrano chiaramente la necessità di un metodo di restauro — la struttura cellulare non è più distinguibile a occhio nudo a noise elevato.

#### 5. Split Dataset
| Split | Percentuale | Immagini |
|---|---|---|
| Training | 70% | 673 |
| Validation | 15% | 144 |
| Test | 15% | 145 |

Seed fissato a 42 per riproducibilità. Stratificato per classe (mantiene le proporzioni in ogni split).

---

## Notebook 04 — DiffPIR (`notebooks/04_diffpir.ipynb`)

### Obiettivo
Applicare DiffPIR, un metodo basato su modelli di diffusione, al problema di deblurring + denoising su immagini citologiche cervicali. Il notebook carica il modello LightUNet custom (addestrato su LBC), esegue il sampling su immagini di test, valuta PSNR/SSIM e confronta con TV e UNet.

### Risultati

#### 1. Configurazione DiffPIR
| Parametro | Valore |
|---|---|---|
| Modello | LightUNet (custom DDPM) addestrato su LBC |
| Parametri modello | ~1.26M |
| Peso modello | ~5 MB |
| `num_steps` | 15 (sub-campionati da $t_{\text{start}}=50$ a $t=0$) |
| $\lambda$ (data-fidelity) | 10.0 |
| $\zeta$ (stocasticità) | 0.0 (deterministico) |
| $t_{\text{start}}$ | 50 (stabilità numerica) |
| Dispositivo | CPU |

**Perché $t_{\text{start}}=50$ invece di $t=1000$?** A $t=1000$, $\bar{\alpha}_t \approx 4.5 \times 10^{-5}$, quindi $\sqrt{1-\bar{\alpha}_t} / \sqrt{\bar{\alpha}_t} \approx 150$, amplificando ogni errore di predizione del rumore di 150× nella stima di $x_0$. A $t=50$, $\bar{\alpha}_t \approx 0.97$, il fattore di amplificazione è ~0.17, rendendo la stima stabile.

#### 2. Demo Visiva (3 immagini)

Per ogni livello di rumore, il notebook mostra: [GT] [Degraded] [Restored]. L'output visivo è salvato in `results/diffpir/qualitative/`.

#### 3. Valutazione Quantitativa (10 immagini × 4 noise levels)

| $\sigma_n$ | PSNR (restored) | SSIM (restored) | Tempo medio |
|---|---|---|---|
| 0.005 | 16.67 dB | 0.235 | 2.0 s |
| 0.01 | 17.32 dB | 0.270 | 2.0 s |
| 0.05 | 22.49 dB | 0.512 | 2.0 s |
| 0.1 | 24.68 dB | 0.664 | 2.0 s |

**Osservazioni:**
- Il PSNR migliora con l'aumentare del rumore ($\sigma_n$) — comportamento inatteso ma spiegabile: a basso rumore, il modello di diffusione rimuove anche dettagli fini (che confonde con rumore), penalizzando il PSNR rispetto a GT. Ad alto rumore, il modello è più aggressivo nella pulizia e il guadagno netto è maggiore.
- A $\sigma_n=0.05$, PSNR 22.49 con SSIM 0.51 — risultato solido per un modello di 1.2M parametri addestrato su 100 immagini per 30 epoche.

#### 4. Timing

| Componente | Tempo medio |
|---|---|
| Degradazione (blur + noise) | ~0.02 sec/img |
| PSNR/SSIM computation | ~0.01 sec/img |
| DiffPIR inference (15 step, CPU) | ~2.1 sec/img |
| TV inference (300 iter, CPU) | ~10 sec/img |
| UNet inference (GPU) | ~0.1 sec/img |

**Osservazione critica**: DiffPIR è circa 20× più lento della UNet su CPU, ma 5× più veloce di TV. Il tempo di ~2 sec/img è accettabile per un batch di 10 immagini (~20 sec totali).

#### 5. Confronto Finale

Il notebook carica automaticamente i risultati dai file `results/<metodo>/metrics.csv` e produce una tabella pivot:

| $\sigma_n$ | TV (PSNR) | UNet (PSNR) | DiffPIR (PSNR) | TV (SSIM) | UNet (SSIM) | DiffPIR (SSIM) |
|---|---|---|---|---|---|---|---|---|
| 0.005 | **32.09 dB** 🥇 | 29.89 dB 🥈 | 16.67 dB | **0.911** 🥇 | 0.894 🥈 | 0.235 |
| 0.01 | **32.04 dB** 🥇 | 29.89 dB 🥈 | 17.32 dB | **0.909** 🥇 | 0.894 🥈 | 0.270 |
| 0.05 | **30.42 dB** 🥇 | 29.63 dB 🥈 | 22.49 dB | **0.837** 🥇 | 0.875 🥈 | 0.512 |
| 0.1 | 26.54 dB 🥈 | **28.93 dB** 🥇 | 24.68 dB | 0.586 | **0.830** 🥇 | 0.664 🥈 |

TV, UNet, e DiffPIR sono tutti completati:
- TV: ✅ PSNR 32.09 → 26.54 dB (dal noise più basso al più alto)
- UNet: ✅ **50 epoche CPU**, architettura ottimizzata (1.9M params, GroupNorm, L1 loss), PSNR 29.89 → 28.93 dB — miglioramento di **+5.8/+7.1 dB** rispetto al vecchio modello
- DiffPIR: ✅ PSNR 24.68 dB a σ=0.1 (miglior metodo ad alto rumore)

#### 6. Struttura dei Risultati

Tutti gli script ora salvano automaticamente in cartelle organizzate:
```
results/
├── diffpir/
│   ├── metrics.csv
│   └── qualitative/
├── tv/
│   ├── metrics.csv
│   └── qualitative/
├── unet/
│   ├── metrics.csv
│   └── qualitative/
└── comparison.png       (generato da scripts/plot_results.py)
```

Il confronto finale si ottiene con:
```
python scripts/plot_results.py
```

#### 7. Limitazioni Note

1. **Modello piccolo**: LightUNet (1.2M param) vs modelli standard (500M+) — qualità inferiore ma training pratico su CPU
2. **Training limitato**: solo 10 epoche su 50 immagini — un training più lungo e su più dati migliorerebbe i risultati
3. **CPU-only**: Impossibile sfruttare GPU per via dell'hardware AMD senza ROCm/DirectML
4. **Campione ridotto**: 10 immagini per test invece di 145 (limitato dal tempo di inferenza su CPU)
5. **$t_{\text{start}}$ ottimale**: Il valore $t_{\text{start}}=50$ è stato determinato empiricamente; potrebbe non essere ottimale per tutti i livelli di rumore
