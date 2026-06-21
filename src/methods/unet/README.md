# UNet — End-to-End Deep Learning Method

UNet architecture with noise level conditioning for deblur + denoise on cervical LBC images.

## Exam Utility

- **End-to-end deep learning**: learns the degraded -> restored mapping directly from data
- **Lightweight architecture**: ~1.9M parameters (vs 31M original) — trainable on CPU
- **Multi-noise augmentation**: training with random noise level per batch — generalizes across all levels

## Files

| File | Content |
|---|---|
| unet.py | UNet, DoubleConv — encoder-decoder architecture with skip connections |

## Detailed Architecture

| Component | Detail |
|---|---|
| Input | 4 channels (RGB + noise map), 256x256 |
| Encoder | 4 levels: 16 -> 32 -> 64 -> 128 channels |
| Decoder | 4 symmetric levels with upsampling |
| Skip connections | Encoder -> decoder connections |
| Output | 3 channels [-1, 1] |
| Parameters | ~1.9M |

### DoubleConv

Each level uses DoubleConv (unet.py:7-20): two 3x3 convolutions with padding=1, each followed by **GroupNorm** (num_groups=8) and **ReLU**. No bias in convolutions (norm makes it redundant). GroupNorm is preferred over BatchNorm for small batch sizes (16) and CPU stability.

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

### Noise Level Conditioning

The noise level sigma_n is passed as a **constant 256x256 map** concatenated as the 4th input channel. This allows the model to:

- Adapt denoising strength based on the present noise
- Generalize to unseen noise levels
- Use the same model for all 4 levels without weight reloading

Noise map generation (run_unet.py:41-44):

```python
def make_noise_map(batch_size, image_size, noise_level, device):
    return torch.full((batch_size, 1, image_size, image_size),
                      noise_level, device=device)
```

### Skip Connections

Encoder -> decoder connections concatenate feature maps at the same spatial level, allowing the decoder to recover fine details lost in downsampling. The encoder produces 4 skip connections stored in a list and reversed in the decoder.

## Training

### Parameters

| Parameter | Value |
|---|---|
| Loss | L1 (MSE is more blurry) |
| Optimizer | Adam, lr=10^-4 |
| Scheduler | ReduceLROnPlateau, patience=5, factor=0.5 |
| Batch size | 16 |
| Epochs | 50 |
| Augmentation | random sigma_n per batch from {0.005, 0.01, 0.05, 0.1} |

### Why L1 instead of L2?

- **L1** (MAE): less penalization of large errors -> sharper edges, less blur
- **L2** (MSE): tends to blur edges (quadratically penalizes gradients)

On LBC images, L1 yields ~0.5 dB more PSNR compared to L2.

### Multi-Noise Augmentation

At each batch, the noise level is randomly sampled from the 4 dataset values:

```python
noise = np.random.choice(noise_levels)
degraded = degrade_batch(gt, kernel_size=9, sigma=2.0, noise_level=noise)
```

This forces the model to operate on different noise levels, improving generalization. Without it, the model would overfit to a single sigma_n.

### Model Checkpointing

```python
if avg_val_psnr > best_val_psnr:
    best_val_psnr = avg_val_psnr
    torch.save(model.state_dict(), "results/unet/best_model.pth")
```

The model with the best validation PSNR is saved. Validation uses a random 25% subset across all noise levels, with the same multi-noise mechanism.

### Weight Initialization

Default PyTorch (Kaiming Uniform for Conv2d). No custom initialization was applied — for architectures with GroupNorm and ReLU, Kaiming Uniform is sufficient.

## Usage

### Inference on a single image

```python
import torch
from src.methods.unet.unet import UNet

model = UNet(in_channels=4, out_channels=3, features=(16, 32, 64, 128))
model.load_state_dict(torch.load("results/unet/best_model.pth"))
model.eval()

# Create input: [degraded RGB (3 channels) + noise map (1 channel)]
noise_map = torch.full((1, 1, 256, 256), noise_level)
model_input = torch.cat([degraded.unsqueeze(0), noise_map], dim=1)

with torch.no_grad():
    restored = model(model_input).squeeze(0)
```

### Full training

```bash
python scripts/run_unet.py
```

Runs 50 epochs on CPU (~20 minutes), validation every epoch, saves best model, final evaluation on test set.

### Comparison with Original UNet

| Feature | Original UNet | This Implementation |
|---|---|---|
| Parameters | 31M | ~1.9M |
| Encoder channels | 64->128->256->512 | 16->32->64->128 |
| Normalization | BatchNorm | GroupNorm (groups=8) |
| Activation | ReLU | ReLU |
| Conditioning | None | Concatenated noise map |
| Upsampling | ConvTranspose2d | ConvTranspose2d |

The channel reduction (4x factor) and use of GroupNorm make the model trainable on CPU in reasonable time.

## Results (145 test images x 4 noise levels)

| sigma_n | PSNR | SSIM | Time |
|---|---|---|---|
| 0.005 | 29.89 dB | 0.894 | 0.035 s |
| 0.01 | 29.89 dB | 0.894 | 0.034 s |
| 0.05 | 29.63 dB | 0.875 | 0.034 s |
| 0.1 | 28.93 dB | 0.830 | 0.036 s |

Very stable performance across all noise levels (29.89 -> 28.93 dB, difference <1 dB), unlike TV (32.09 -> 26.54 dB, difference ~5.5 dB). Inference time ~35 ms/image on CPU (vs 3 seconds for DiffPIR).
