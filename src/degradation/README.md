# src/degradation/ — Pipeline di Degradazione

Implementa la degradazione blur + rumore, **identica per tutti i metodi**.

## Utilità per l'esame

- **Confronto equo**: stessa funzione `degrade()` per TV, UNet e DiffPIR — le differenze nei risultati sono solo dovute ai metodi
- **Parametri fissi**: blur gaussiano (σ=2, kernel=9) e 4 livelli di AWGN (0.005, 0.01, 0.05, 0.1)
- **Seed controllato**: seed 42 per riproducibilità

## File

| File | Contenuto |
|---|---|
| `degradation.py` | `degrade()`, `gaussian_kernel()`, `apply_blur()` |

## Funzioni principali

| Funzione | Ruolo |
|---|---|
| `gaussian_kernel()` | Crea kernel gaussiano 2D normalizzato |
| `degrade()` | Applica blur (`F.conv2d`) + AWGN, ritorna immagine in [-1, 1] |

## Parametri di degradazione

| Parametro | Valore |
|---|---|
| Blur | Gaussiano, σ=2, kernel 9×9 |
| Noise level 1 | σₙ = 0.005 |
| Noise level 2 | σₙ = 0.01 |
| Noise level 3 | σₙ = 0.05 |
| Noise level 4 | σₙ = 0.1 |
