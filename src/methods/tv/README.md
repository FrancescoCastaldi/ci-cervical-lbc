# TV — Total Variation (Metodo Variazionale)

Implementazione della regolarizzazione Total Variation per deblur + denoise.

## Utilità per l'esame

- **Baseline classica**: metodo variazionale semplice e interpretabile — non richiede training né dati
- **Termine di regolarizzazione**: penalizza le variazioni locali favorendo regioni uniformi con bordi netti
- **Confronto**: ideale per mostrare i limiti dei metodi classici rispetto a deep learning e generativi

## File

| File | Contenuto |
|---|---|
| `tv.py` | `tv_restore()`, `tv_loss()`, `gaussian_kernel_tensor()`, `apply_blur()` |

## Formulazione

Minimizza: `||H*x - y||² + λ_reg * TV(x)`

con `TV(x) = |∇_h x| + |∇_v x|` (norma L1 del gradiente).

## Parametri

| Parametro | Valore | Ruolo |
|---|---|---|
| `λ_reg` | 0.005 | Peso regolarizzazione TV |
| `max_iter` | 150 | Iterazioni Adam |
| `lr` | 0.001 | Learning rate ottimizzatore |

## Risultati (145 test images × 4 noise level)

| σₙ | PSNR | SSIM |
|---|---|---|
| 0.005 | 32.09 dB | 0.911 |
| 0.01 | 32.04 dB | 0.909 |
| 0.05 | 30.42 dB | 0.837 |
| 0.1 | 26.54 dB | 0.586 |
