"""
Valuta il modello UNet già addestrato sul test set.
Carica best_model.pth, esegue inferenza su tutti i noise level, salva metrics.csv.
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import gaussian_kernel
from src.methods.unet.unet import UNet
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison


def degrade_batch(images, kernel_size=9, sigma=2.0, noise_level=0.01):
    kernel = gaussian_kernel(kernel_size, sigma).to(images.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(images.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel, padding=padding, groups=images.shape[1])
    noise = torch.randn_like(blurred) * noise_level
    return torch.clamp(blurred + noise, -1.0, 1.0)


# ── Setup ─────────────────────────────────────────────────────────────────────
config = load_config()
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.manual_seed(config["seed"])

noise_levels = config["degradation"]["noise_levels"]
kernel_size = config["degradation"]["kernel_size"]
blur_sigma = config["degradation"]["blur_sigma"]
batch_size = config["unet"]["batch_size"]

output_dir = Path(config["eval"]["results_dir"]) / "unet"
qual_dir = output_dir / "qualitative"
output_dir.mkdir(parents=True, exist_ok=True)
qual_dir.mkdir(parents=True, exist_ok=True)

# ── Carica modello ────────────────────────────────────────────────────────────
model = UNet().to(device)
ckpt = torch.load(output_dir / "best_model.pth", map_location=device)
model.load_state_dict(ckpt)
model.eval()
print(f"Modello caricato da {output_dir / 'best_model.pth'}")
print(f"Device: {device}")

# ── Carica test set ───────────────────────────────────────────────────────────
test_dataset = LBCDataset("data/splits/test.txt", config["dataset"]["image_size"])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Test set: {len(test_dataset)} immagini")

# ── Valutazione ───────────────────────────────────────────────────────────────
rows = []
for noise_level in noise_levels:
    psnr_total = 0.0
    ssim_total = 0.0
    time_list = []
    first_batch = True

    for gt in test_loader:
        gt = gt.to(device)
        degraded = degrade_batch(gt, kernel_size=kernel_size,
                                 sigma=blur_sigma, noise_level=noise_level)

        t0 = time.time()
        with torch.no_grad():
            pred = model(degraded)
        elapsed = time.time() - t0
        time_list.append(elapsed)

        # Valuta immagine per immagine
        for j in range(gt.size(0)):
            m = evaluate(pred[j].cpu(), gt[j].cpu())
            psnr_total += m["psnr"]
            ssim_total += m["ssim"]

        # Salva qualitative (primo batch)
        if first_batch:
            first_batch = False
            for j in range(min(3, gt.size(0))):
                show_comparison(
                    {"GT": gt[j].cpu(), "Degraded": degraded[j].cpu(),
                     "UNet": pred[j].cpu()},
                    save_path=qual_dir / f"noise_{noise_level}_img{j}.png"
                )

    n = len(test_dataset)
    avg_psnr = psnr_total / n
    avg_ssim = ssim_total / n
    avg_time = sum(time_list) / len(time_list)

    rows.append({
        "method": "unet",
        "noise_level": noise_level,
        "psnr": round(avg_psnr, 2),
        "ssim": round(avg_ssim, 4),
        "avg_inference_time": round(avg_time, 4),
    })
    print(f"[UNet] noise={noise_level} | PSNR={avg_psnr:.2f} dB | "
          f"SSIM={avg_ssim:.4f} | Time={avg_time:.4f}s")

# ── Salva risultati ───────────────────────────────────────────────────────────
csv_path = output_dir / "metrics.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"\nRisultati salvati in {csv_path}")
print("Fatto!")
