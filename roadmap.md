# Roadmap — Verifica Requisiti Esame (Gruppo R)

Analisi puntuale delle richieste del professore e verifica della loro presenza nel progetto.
Ogni voce include: **richiesta** → **stato** → **posizione nel progetto** → **giustificazione**.

---

## 1. Dataset

| # | Richiesta del Professore | Stato | Dove si trova | Giustificazione |
|---|---|---|---|---|
| 1.1 | Usare il dataset **Mendeley LBC cervical cancer** | ✅ | `data/README.md`, `notebooks/01_eda.ipynb` | Dataset scaricato da Mendeley, collocato in `data/raw/` (non tracciato per dimensioni) |
| 1.2 | Ispezionare il dataset, comprenderne struttura, classi e immagini corrotte | ✅ | `notebooks/01_eda.ipynb`, `src/data/dataset.py:27-33` | EDA completo: 4 classi (NILM 63.6%, HSIL 16.9%, LSIL 11.7%, SCC 7.7%), verifica corruzione con `is_valid_image()`, statistiche dettagliate |
| 1.3 | Considerare un **subset di ~4000 immagini** | ⚠️ Parziale | `configs/experiment.yaml:8` subset_size=4000 | Il dataset Mendeley contiene **962 immagini totali**, non 4000. Il codice tenta di usarne 4000 ma `build_splits()` fa `min(subset_size, len(valid_images))` quindi usa tutte le 962 disponibili. Giustificato in `report/report.md §2.1` |
| 1.4 | **Resize a 256×256** | ✅ | `src/data/dataset.py:77-81` | `T.Resize((256, 256))` in `LBCDataset.transform` |
| 1.5 | Pipeline di preprocessing: **normalizzazione, split** | ✅ | `src/data/dataset.py:71-91` | `T.ToTensor()` + `T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])` → range [-1, 1]. Split 70/15/15 con `build_splits()` |
| 1.6 | **Scelte di preprocessing giustificate e documentate** | ✅ | `report/report.md §2.2` | Tabella con ogni passo: resize (carico computazionale), ToTensor (compatibilità PyTorch), normalizzazione [-1,1] (DDPM richiede questo range), split stratificato (mantiene proporzioni classi) |
| 1.7 | **Split fisso** per riproducibilità | ✅ | `configs/experiment.yaml:1` seed=42, `src/data/dataset.py:47` | `random.seed(config["seed"])` prima dello shuffle, split identico ad ogni esecuzione |

---

## 2. Task Definition

| # | Richiesta del Professore | Stato | Dove si trova | Giustificazione |
|---|---|---|---|---|
| 2.1 | **Target**: deblur + denoise | ✅ | `report/report.md §3`, `report/theory.md` | Formulato come problema inverso: $y = (k * x) + n$ |
| 2.2 | Formulazione come **problema inverso** | ✅ | `report/report.md §3.1`, `report/theory.md §2` | $y = \mathcal{H}(x) + n$, problema mal posto (ill-posed), serve regolarizzazione o prior appresi |
| 2.3 | **Blur gaussiano**: $\sigma=2$, kernel size $=9$ | ✅ | `configs/experiment.yaml:15-16`, `src/degradation/degradation.py:10-15` | `gaussian_kernel(kernel_size=9, sigma=2.0)` |
| 2.4 | **AWGN** a 4 livelli: $\sigma_n \in \{0.005, 0.01, 0.05, 0.1\}$ | ✅ | `configs/experiment.yaml:17`, `src/degradation/degradation.py:22-24` | `noise = torch.randn_like(blurred) * noise_level` |
| 2.5 | **Stessi input degradati** per tutti i metodi | ✅ | `src/degradation/degradation.py` | Singola funzione `degrade()` chiamata da tutti gli script `run_*.py`. Seed fisso 42 garantisce stessa sequenza di rumore |
| 2.6 | **Controllo di riproducibilità** della degradazione | ✅ | `report/report.md §3.3`, `configs/experiment.yaml:1` | Seed 42 per PyTorch (`torch.manual_seed`) e Python `random.seed` |

---

## 3. Metodo Variazionale — Total Variation (TV)

| # | Richiesta del Professore | Stato | Dove si trova | Giustificazione |
|---|---|---|---|---|
| 3.1 | Implementare **regolarizzazione Total Variation** | ✅ | `src/methods/tv/tv.py` | `tv_restore()` minimizza $\|\|Hx - y\|\|^2 + \lambda \cdot TV(x)$ con Adam |
| 3.2 | **Parametri scelti euristicamente** | ✅ | `report/report.md §4.2` | $\lambda_{reg}=0.005$, 150 iterazioni Adam, lr=0.001 |
| 3.3 | **Discussione della scelta dei parametri** | ✅ | `report/report.md §4.2, §6.4` | $\lambda$ testato su $\{0.001, 0.005, 0.01, 0.05, 0.1\}$: con $\lambda=0.001$ troppo rumore residuo, con $\lambda=0.1$ staircasing eccessivo. $\lambda=0.005$ è il bilanciamento ottimale |
| 3.4 | **Risultati quantitativi** (PSNR, SSIM) | ✅ | `results/tv/metrics.csv`, `report/report.md §5.3` | 145 test images × 4 noise level: PSNR 26.54–32.09 dB, SSIM 0.586–0.911 |
| 3.5 | **Risultati qualitativi** (immagini ricostruite) | ✅ | `results/tv/qualitative/` | 24 immagini (6 per noise level), confronto GT vs degraded vs restored |

---

## 4. Metodo End-to-End — UNet

| # | Richiesta del Professore | Stato | Dove si trova | Giustificazione |
|---|---|---|---|---|
| 4.1 | **Scegliere** uno tra UNet, ViT, NAF-Net | ✅ | `report/report.md §4.3.1` | Scelto **UNet** per 4 ragioni: (1) adatta a image-to-image translation, (2) dataset limitato (673 train), (3) efficienza CPU ($O(N^2)$ self-attention di ViT proibitiva), (4) semplicità e riproducibilità |
| 4.2 | **Architettura motivata** in relazione al task | ✅ | `report/report.md §4.3.2`, `src/methods/unet/unet.py` | Encoder-decoder con skip connections: encoder 16→32→64→128 canali, bottleneck 128→256, decoder simmetrico. Skip connections preservano dettagli spaziali fini (nuclei cellulari). Output tanh per range [-1,1] |
| 4.3 | **Training configurato e giustificato** | ✅ | `report/report.md §4.3.3`, `scripts/run_unet.py` | Loss L1 (preserva bordi meglio di MSE), Adam lr=$10^{-4}$, batch=16 (max su CPU), 50 epoche, multi-noise augmentation (σ random per batch) |
| 4.4 | **Validation + best model saving** | ✅ | `scripts/run_unet.py:160-195` | Validation su 25% del validation set (36 immagini) con PSNR medio su noise level random. Salva `best_model.pth` quando Val PSNR migliora |
| 4.5 | **Risultati quantitativi** (PSNR, SSIM, tempo) | ✅ | `results/unet/metrics.csv`, `report/report.md §5.4` | 145 test images × 4 noise level: PSNR 28.93–29.89 dB, SSIM 0.830–0.894, tempo ~0.035 s/img |
| 4.6 | **Risultati qualitativi** | ✅ | `results/unet/qualitative/` | 24 immagini (6 per noise level) |

---

## 5. Metodo Generativo — DiffPIR

| # | Richiesta del Professore | Stato | Dove si trova | Giustificazione |
|---|---|---|---|---|
| 5.1 | Implementare **DiffPIR** (Denoising Diffusion Models for Plug-and-Play Image Restoration) | ✅ | `src/methods/diffpir/diffpir.py` | Algoritmo completo: stima $\hat{x}_0$ dal denoiser, FFT data-fidelity nel dominio frequenziale, DDIM sampling deterministico |
| 5.2 | **Adattato al task** (non ImageNet pre-trained) | ✅ | `src/methods/diffpir/model.py`, `train.py` | LightUNet custom (1.26M params) addestrato su 673 immagini LBC cervicali. Scelta motivata: modello ImageNet non cattura morfologia cellulare, 1.26M vs 500M params = possibile su CPU |
| 5.3 | **DDPM training** descritto | ✅ | `report/report.md §4.4.2`, `src/methods/diffpir/train.py` | 1000 timestep, scheduler lineare $\beta_1=10^{-4}, \beta_T=0.02$, MSE sulla predizione del rumore $\varepsilon_\theta(x_t, t)$, 30 epoche |
| 5.4 | **Parametri euristici** discussi | ✅ | `report/report.md §4.4.4, §6.4` | $t_{start}=50$ testato $\{10,30,50,100,200\}$ (stabilità numerica), $\lambda=10$ testato $\{0.1,1,5,10,50\}$ (bilanciamento data-fidelity/prior), $num\_steps=15$ testato $\{5,10,15,30\}$ (qualità/tempo), $\zeta=0$ (riproducibilità DDIM) |
| 5.5 | **Spiegazione dell'uso** di DiffPIR nel contesto | ✅ | `report/report.md §4.4` | Alternanza denoising (prior generativo) e data-fidelity (FFT). Peso dinamico $\rho_t$ che bilancia i due termini in base al timestep |
| 5.6 | **Risultati quantitativi** (PSNR, SSIM, tempo) | ✅ | `results/diffpir/metrics.csv`, `report/report.md §5.2` | 10 test images × 4 noise level: PSNR 16.67–24.68 dB (cresce col rumore), tempo ~2 s/img |
| 5.7 | **Risultati qualitativi** | ✅ | `results/diffpir/qualitative/` | 24 immagini (6 per noise level) |

---

## 6. Metodo Ibrido — Weighted TV

| # | Richiesta del Professore | Stato | Dove si trova | Giustificazione |
|---|---|---|---|---|
| 6.1 | Implementare **Weighted TV** (Morotti et al., 2025) | ❌ Escluso | `report/report.md §4.1` | Il gruppo R è composto da **2 studenti** → la consegna permette di scegliere 3 metodi su 4. Abbiamo escluso Weighted TV perché TV (variazionale) + UNet (deep learning) + DiffPIR (generativo) coprono già **3 famiglie metodologiche distinte**, dando un confronto più ricco dal punto di vista didattico. |

---

## 7. Deliverables — Confronto e Valutazione

| # | Richiesta del Professore | Stato | Dove si trova | Giustificazione |
|---|---|---|---|---|
| 7.1 | **Confronto quantitativo**: PSNR, SSIM | ✅ | `src/eval/metrics.py`, `results/*/metrics.csv` | `compute_psnr()`, `compute_ssim()` via `skimage.metrics`. Calcolati su immagini normalizzate in $[0,1]$ |
| 7.2 | **Confronto visivo**: immagini ricostruite | ✅ | `results/*/qualitative/`, `src/plots/visualize.py:show_comparison()` | Griglia GT + degraded + restored per ogni metodo e noise level |
| 7.3 | **Grafico comparativo** tra tutti i metodi | ✅ | `results/comparison.png`, `scripts/plot_results.py`, `src/plots/visualize.py:plot_metrics()` | PSNR e SSIM per metodo e noise level in un unico plot |
| 7.4 | **Report** con metodi, setup e discussione | ✅ | `report/report.md` (673 righe) | Sezioni: dataset, task, metodi (TV/UNet/DiffPIR), risultati, discussione, parametri euristici, conclusioni |
| 7.5 | **Fondamenti teorici** | ✅ | `report/theory.md` | Problema inverso, regolarizzazione TV, UNet, modelli diffusivi, metriche PSNR/SSIM |
| 7.6 | **Codice documentato** su GitHub | ✅ | Intero repository, README.md, README in ogni cartella, `src/methods/*/README.md`, `scripts/README.md` | Struttura modulare, docstring, README esplicativi per ogni modulo |
| 7.7 | **Presentazione PowerPoint/Beamer** | ✅ | `slides/presentazione.pptx` | 10 slide: copertina, problema, dataset, metodi (TV/UNet/DiffPIR), risultati, confronto, discussioni |
| 7.8 | **Link GitHub** consegnato all'orale | ✅ | `report/report.md:12`, `README.md` | `https://github.com/FrancescoCastaldi/ci-cervical-lbc` |

---

## 8. Extra — Qualità del Progetto

| # | Elemento | Stato | Dove si trova | Giustificazione |
|---|---|---|---|---|
| 8.1 | **Test unitari** | ✅ 42 test | `tests/` | 5 file: degradation (10), metrics (9), diffpir (7), unet (8), tv (8). Verificano forme, range, convergenza, gradienti |
| 8.2 | **Notebook interattivi** | ✅ 5 notebook | `notebooks/` | `01_eda` (EDA), `02_tv` (TV demo), `03_unet` (UNet training), `04_diffpir` (DiffPIR demo), `05_full_pipeline` (pipeline end-to-end) |
| 8.3 | **Riproducibilità** | ✅ | `configs/experiment.yaml`, seed 42 | Seed fisso per Python, PyTorch, NumPy. Parametri centralizzati. Pipeline deterministiche |
| 8.4 | **Parametri euristici documentati** | ✅ | `report/report.md §6.4` | Tabella riassuntiva con tutti i parametri, valori testati, e criterio di scelta finale |
| 8.5 | **Discussione successi e fallimenti** | ✅ | `report/report.md §5-6` | TV: staircasing e λ fisso. UNet: degradazione minima (+1dB). DiffPIR: allucinazioni a basso rumore, migliora ad alto rumore |
| 8.6 | **Organizzazione modulare del codice** | ✅ | `src/methods/<metodo>/` | Ogni metodo è indipendente. Condivisi solo data, degradation, eval, plots |

---

## Riepilogo Finale

| Area | Totale Richieste | ✅ | ⚠️ | ❌ |
|---|---|---|---|---|
| 1. Dataset | 7 | 6 | 1 (subset 4000) | 0 |
| 2. Task Definition | 6 | 6 | 0 | 0 |
| 3. TV (variazionale) | 5 | 5 | 0 | 0 |
| 4. UNet (end-to-end) | 6 | 6 | 0 | 0 |
| 5. DiffPIR (generativo) | 7 | 7 | 0 | 0 |
| 6. Weighted TV (ibrido) | 1 | 0 | 0 | 1 ✅ esclusione giustificata |
| 7. Deliverables | 8 | 8 | 0 | 0 |
| 8. Extra (qualità) | 6 | 6 | 0 | 0 |
| **Totale** | **46** | **44** | **1** | **1** (esclusione consentita) |
