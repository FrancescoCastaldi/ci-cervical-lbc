# src/methods/ — Metodi di Restauro

Implementazioni dei tre metodi di restauro: variazionale (TV), end-to-end (UNet), generativo (DiffPIR).

## Utilità per l'esame

- **Copertura metodologica**: tre famiglie diverse — variazionale, deep learning, generativo
- **Indipendenza**: ogni metodo è un modulo separato con il proprio codice
- **Confronto critico**: stessi input, stesse metriche — si valutano punti di forza e limiti di ciascuno

## Metodi

| Modulo | Metodo | Famiglia | File principale |
|---|---|---|---|
| `tv/` | Total Variation | Variazionale | `tv.py` |
| `unet/` | UNet | Deep Learning (end-to-end) | `unet.py` |
| `diffpir/` | DiffPIR | Generativo (diffusion) | `diffpir.py` |

## Perché questi tre?

| Metodo | Vantaggio | Limite |
|---|---|---|
| **TV** | Non serve training, interpretabile | Qualità limitata, staircasing |
| **UNet** | Veloce in inferenza, robusto | Serve training, generalizza fuori distribuzione |
| **DiffPIR** | Qualità generativa, ricostruisce dettagli | Lento, allucinazioni a basso rumore |
