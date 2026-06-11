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
python -m pytest tests/ -v
```
