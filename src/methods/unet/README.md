# UNet -- Metodo Deep Learning End-to-End

Architettura UNet con condizionamento del noise level per deblur + denoise su immagini LBC cervicali.

## Utilita per l'esame

- **Deep learning end-to-end**: impara direttamente la mappatura degraded -> restored dai dati
- **Architettura snella**: ~1.9M parametri (vs 31M originale) -- addestrabile su CPU
- **Multi-noise augmentation**: training con noise level random per batch -- generalizza su tutti i livelli

## File

| File | Contenuto |
|---|---|
| unet.py | UNet, DoubleConv -- architettura encoder-decoder con skip connections |

## Architettura Dettagliata

| Componente | Dettaglio |
|---|---|
| Input | 4 canali (RGB + noise map), 256x256 |
| Encoder | 4 livelli: 16 -> 32 -> 64 -> 128 canali |
| Decoder | 4 livelli simmetrici con upsampling |
| Skip connections | Connessioni encoder -> decoder |
| Output | 3 canali [-1, 1] |
| Parametri | ~1.9M |

### DoubleConv

Ogni livello usa DoubleConv (unet.py:7-20): due convoluzioni 3x3 con padding=1, ciascuna seguita da **GroupNorm** (num_groups=8) e **ReLU**. Nessun bias nelle convoluzioni (la norm lo rende ridondante). GroupNorm e preferita a BatchNorm per batch size piccoli (16) e stabilita su CPU.

```python
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )
```

### Condizionamento del Noise Level

Il noise level sigma_n e passato come **mappa costante 256x256** concatenata al 4 canale d'ingresso. Cio permette al modello di:

- Adattare la forza di denoising in base al rumore presente
- Generalizzare su noise level non visti
- Usare lo stesso modello per tutti i 4 livelli senza ricarica pesi

Generazione della noise map (run_unet.py:41-44):

```python
def make_noise_map(batch_size, image_size, noise_level, device):
    return torch.full((batch_size, 1, image_size, image_size),
                      noise_level, device=device)
```

### Skip Connections

Collegamenti encoder -> decoder concatenano le feature map dello stesso livello spaziale, permettendo al decoder di recuperare dettagli fini persi nel downsampling. L'encoder produce 4 skip connection salvate in lista e invertite nel decoder.

## Training

### Parametri

| Parametro | Valore |
|---|---|
| Loss | L1 (MSE meno blur) |
| Ottimizzatore | Adam, lr=10^-4 |
| Scheduler | ReduceLROnPlateau, patience=5, factor=0.5 |
| Batch size | 16 |
| Epoche | 50 |
| Augmentation | sigma_n random per batch tra {0.005, 0.01, 0.05, 0.1} |

### Perche L1 invece di L2?

- **L1** (MAE): minore penalizzazione dei grandi errori -> bordi piu netti, meno blur
- **L2** (MSE): tende a sfumare i bordi (penalizza quadraticamente i gradienti)

Su immagini LBC, L1 produce ~0.5 dB in piu di PSNR rispetto a L2.

### Multi-Noise Augmentation

Ad ogni batch, il noise level e campionato casualmente tra i 4 valori del dataset:

```python
noise = np.random.choice(noise_levels)
degraded = degrade_batch(gt, kernel_size=9, sigma=2.0, noise_level=noise)
```

Questo costringe il modello a operare su diversi livelli di rumore, migliorando la generalizzazione. Senza, il modello overfitterebbe su un singolo sigma_n.

### Model Checkpointing

```python
if avg_val_psnr > best_val_psnr:
    best_val_psnr = avg_val_psnr
    torch.save(model.state_dict(), "results/unet/best_model.pth")
```

Il modello con miglior PSNR di validazione e salvato. La validation usa un subset random del 25% su tutti i noise level, con lo stesso meccanismo di multi-noise.

### Weight Initialization

Default PyTorch (Kaiming Uniform per Conv2d). Non e stata applicata inizializzazione personalizzata -- per architetture con GroupNorm e ReLU, Kaiming Uniform e sufficiente.

## Uso

### Inferenza su singola immagine

```python
import torch
from src.methods.unet.unet import UNet

model = UNet(in_channels=4, out_channels=3, features=(16, 32, 64, 128))
model.load_state_dict(torch.load("results/unet/best_model.pth"))
model.eval()

# Crea input: [degraded RGB (3 canali) + noise map (1 canale)]
noise_map = torch.full((1, 1, 256, 256), noise_level)
model_input = torch.cat([degraded.unsqueeze(0), noise_map], dim=1)

with torch.no_grad():
    restored = model(model_input).squeeze(0)
```

### Training completo

```bash
python scripts/run_unet.py
```

Esegue 50 epoche su CPU (~20 minuti), validation ogni epoca, salvataggio best model, valutazione finale su test set.

### Confronto con UNet Originale

| Caratteristica | UNet Originale | Questa Implementazione |
|---|---|---|
| Parametri | 31M | ~1.9M |
| Canali encoder | 64->128->256->512 | 16->32->64->128 |
| Normalizzazione | BatchNorm | GroupNorm (gruppi=8) |
| Attivazione | ReLU | ReLU |
| Condizionamento | Assente | Noise map concatenata |
| Upsampling | ConvTranspose2d | ConvTranspose2d |

La riduzione dei canali (fattore 4x) e l'uso di GroupNorm rendono il modello addestrabile su CPU in tempi ragionevoli.

## Risultati (145 test images x 4 noise level)

| sigma_n | PSNR | SSIM | Tempo |
|---|---|---|---|
| 0.005 | 29.89 dB | 0.894 | 0.035 s |
| 0.01 | 29.89 dB | 0.894 | 0.034 s |
| 0.05 | 29.63 dB | 0.875 | 0.034 s |
| 0.1 | 28.93 dB | 0.830 | 0.036 s |

Prestazioni molto stabili su tutti i noise level (29.89 -> 28.93 dB, differenza <1 dB), a differenza di TV (32.09 -> 26.54 dB, differenza ~5.5 dB). Tempo di inferenza ~35 ms/immagine su CPU (vs 3 secondi di DiffPIR).
