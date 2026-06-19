# TV -- Total Variation (Metodo Variazionale)

Implementazione della regolarizzazione Total Variation per deblur + denoise su immagini LBC cervicali.

## Utilita per l'esame

- **Baseline classica**: metodo variazionale semplice e interpretabile -- non richiede training ne dati
- **Termine di regolarizzazione**: penalizza le variazioni locali favorendo regioni uniformi con bordi netti
- **Confronto**: ideale per mostrare i limiti dei metodi classici rispetto a deep learning e generativi

## File

| File | Contenuto |
|---|---|
| tv.py | tv_restore(), tv_loss(), gaussian_kernel_tensor(), apply_blur() |

## Formulazione Matematica

Il problema inverso si formalizza come minimizzazione del funzionale:

```
min_x  J(x) = ||Hx - y||^2_2 + lambda * TV(x)
```

dove **H** e l'operatore di blur (convoluzione gaussiana), **y** e l'immagine degradata, **lambda** e il peso di regolarizzazione.

### Total Variation (anisotropica)

```
TV(x) = |nabla_h x| + |nabla_v x|
      = sum_{i,j} |x(i+1,j) - x(i,j)| + |x(i,j+1) - x(i,j)|
```

Usa norma L1 sulle componenti del gradiente separatamente (vs. isotropica sqrt(nabla_h^2 + nabla_v^2)), favorendo bordi diagonali netti.

### Calcolo della loss (tv.py:5-8)

```python
def tv_loss(x):
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()
```

diff_h = differenze verticali (gradiente orizzontale), diff_w = differenze orizzontali (gradiente verticale). La .mean() normalizza rispetto alla dimensione.

### Ottimizzazione con Adam

Il gradiente del data-fidelity e 2H^T(Hx - y); il gradiente di TV e calcolato via autograd. Si usa **Adam** invece di SGD per:

- Convergenza piu rapida (~150 iter vs. ~500 con SGD)
- Learning rate adattivo per parametro
- Minore sensibilita al tuning di lambda

**Clamping**: dopo ogni passo x.clamp_(-1.0, 1.0) mantiene il range [-1, 1].

## Uso

### Import e chiamata singola

```python
from src.methods.tv.tv import tv_restore

restored = tv_restore(
    degraded,                    # tensore [C, H, W] in [-1, 1]
    kernel_size=9, sigma=2.0,   # kernel blur (stesso degradation)
    lambda_reg=0.005,           # peso TV
    max_iter=150,               # iterazioni Adam
    lr=0.001                    # learning rate
)
```

### Esecuzione completa

```bash
python scripts/run_tv.py
```

Valuta tutte le 145 immagini di test per i 4 noise level (0.005, 0.01, 0.05, 0.1).

## Parametri -- Scelte Motivate

| Parametro | Valore | Motivazione |
|---|---|---|
| lambda_reg | 0.005 | Bilanciato: sopprime rumore senza eccessivo staircasing |
| max_iter | 150 | Convergenza entro ~100 iter; 150 per safety margin |
| lr | 0.001 | Default Adam per ottimizzazione immagini |

### Tuning di lambda

- **lambda < 0.001**: rumore residuo, data-fidelity domina
- **lambda > 0.05**: eccessivo smoothing, perdita dettagli, staircasing
- **lambda = 0.005**: ottimo empirico per sigma_n in [0.005, 0.1] su immagini LBC

### Perche 150 iterazioni?

La loss scende rapidamente nelle prime 50 iter, poi si stabilizza. Oltre 200 iterazioni si osserva overfitting sul rumore (data-fidelity continua a scendere, SSIM degrada).

## Staircasing Effect

TV-L1 produce **staircasing**: regioni uniformi con gradini artificiali (gradienti costanti a tratti). Su immagini LBC:

- Zone piatte con falsi contorni in aree citoplasmatiche omogenee
- Perdita di texture sottili

Per sigma_n = 0.1 il PSNR cala a 26.54 dB e lo staircasing e visibile. E il principale limite del metodo vs UNet/DiffPIR.

## Confronto con Altri Metodi Variazionali

| Metodo | Caratteristica |
|---|---|
| **TV-L1** (questo) | Norma L1 sul gradiente -- preserva bordi, ma staircasing |
| **TV-L2** | Norma L2 -- meno staircasing, ma bordi sfumati |
| **Bilateral TV** | Penalizza gradienti per intensita -- meglio su texture |
| **Tikhonov** | Regolarizzazione L2 -- oversmoothing generalizzato |

## Kernel di Blur

Identico a src/degradation/degradation.py per confronto equo:

- **Tipo**: gaussiano 9x9, sigma = 2.0
- **Canali**: separato per canale (groups=3)
- **Padding**: riflesso

## Risultati (145 test images x 4 noise level)

| sigma_n | PSNR | SSIM |
|---|---|---|
| 0.005 | 32.09 dB | 0.911 |
| 0.01 | 32.04 dB | 0.909 |
| 0.05 | 30.42 dB | 0.837 |
| 0.1 | 26.54 dB | 0.586 |

Per sigma_n bassi (0.005-0.01) TV compete con UNet (~32 vs 29.89 dB). Per sigma_n alti (0.1) crolla a 26.54 dB vs 28.93 dB di UNet -- il deep learning generalizza meglio.
