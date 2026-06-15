# tests/ — Test Unitari

34 test unitari che verificano la correttezza delle implementazioni.

## Utilità per l'esame

- **Affidabilità del codice**: ogni metodo ha test specifici che ne verificano il funzionamento
- **Riproducibilità**: i test garantiscono che le implementazioni siano numericamente stabili e corrette
- **Copertura**: degradation (10), metrics (9), diffpir (7), unet (8), tv (8)

## File

| File | Test | Cosa verifica |
|---|---|---|
| `test_degradation.py` | 10 | Kernel gaussiano, blur, AWGN, pipeline completa |
| `test_metrics.py` | 9 | PSNR, SSIM, edge case (immagini identiche, nere, diverse) |
| `test_diffpir.py` | 7 | FFT data-fidelity, DDIM sampling, shape output |
| `test_unet.py` | 8 | Forward pass, gradienti, range output, condizionamento noise |
| `test_tv.py` | 8 | TV loss, kernel, blur, range output, convergenza |

## Esecuzione

```bash
# Tutti i test
python -m pytest tests/ -v

# Singolo file
python -m pytest tests/test_tv.py -v

# Singolo test
python -m pytest tests/test_tv.py::test_tv_loss -v

# Con output dettagliato
python -m pytest tests/ -v --tb=short
```

## Struttura dei test

I test usano `pytest` con funzioni `test_*` indipendenti (nessuna classe). Ogni test crea input sintetici con `torch.randn` e verifica l'output con `torch.allclose`.

**Esempio concreto** (da `test_degradation.py`):

```python
def test_gaussian_kernel_sum():
    kernel = gaussian_kernel(kernel_size=9, sigma=2.0)
    assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)
    assert kernel.shape == (1, 1, 9, 9)
```

**Esempio metriche** (da `test_metrics.py`):

```python
def test_psnr_identical():
    x = torch.randn(1, 3, 64, 64)
    psnr = compute_psnr(x, x)
    assert psnr == float("inf")  # immagini identiche → PSNR infinito
```

## Cosa viene testato per modulo

| Modulo | Copertura specifica |
|---|---|
| **degradation** | normalizzazione kernel, blur conserva media, AWGN ha σ corretto, pipeline end-to-end riproducibile con seed |
| **metrics** | PSNR infinito per input identici, PSNR = 0 per immagini opposte, SSIM = 1 per identiche, SSIM < 1 per diverse, range [0,1] e [-1,1] |
| **diffpir** | FFT data-fidelity term, DDIM sampling mantiene shape e range [-1,1], zeta=0 coincide con denoising puro |
| **unet** | forward pass con e senza noise conditioning, gradienti fluiscono, output in [-1,1], shape preservato, skip connections attive |
| **tv** | TV loss è sempre ≥ 0, kernel gaussiano normalizzato, blur riduce varianza, output in [-1,1], convergenza iterativa |

## Aggiungere nuovi test

1. Creare `tests/test_<modulo>.py`
2. Importare le funzioni da testare
3. Scrivere funzioni `test_*` senza classe
4. Usare `torch.allclose()` con `atol=1e-4` per tolleranza numerica
5. Eseguire con `python -m pytest tests/test_<modulo>.py -v`

## Dipendenze

- `pytest` — test runner
- `torch` — tensori e operazioni
- `numpy` — metriche aggiuntive
- `scikit-image` — SSIM di riferimento per cross-validazione
