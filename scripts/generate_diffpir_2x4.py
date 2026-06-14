"""
Genera figura DiffPIR: 4 colonne (GT | Degraded | DiffPIR | PSNR/SSIM)
x 4 righe (noise level).
"""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import degrade
from src.methods.diffpir.diffpir import run_diffpir

config = load_config()
device = torch.device("cpu")
torch.manual_seed(config["seed"])

noise_levels = config["degradation"]["noise_levels"]
kernel_size = config["degradation"]["kernel_size"]
blur_sigma = config["degradation"]["blur_sigma"]
img_size = config["dataset"]["image_size"]

SAVE_DIR = Path("results/qualitative_slides")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
})

def tensor_to_np(t):
    arr = t.detach().cpu().permute(1, 2, 0).clamp(-1, 1).numpy()
    return (arr * 0.5 + 0.5).astype(np.float64)

# Carica sample
dataset = LBCDataset("data/splits/test.txt", img_size)
gt = dataset[0]
gt_np = tensor_to_np(gt)
print(f"Sample caricato: shape={gt.shape}")

# DiffPIR weights
weights_path = Path(config["diffpir"]["weights"])
if not weights_path.is_absolute():
    weights_path = Path(__file__).resolve().parent.parent / weights_path

# Processa tutti e 4 i noise level
entries = []
for nl in noise_levels:
    print(f"  sigma={nl}...", end=" ", flush=True)
    degraded = degrade(gt, kernel_size=kernel_size,
                       sigma=blur_sigma, noise_level=nl)
    restored, inference_time = run_diffpir(
        degraded,
        num_steps=config["diffpir"]["num_steps"],
        noise_level=nl,
        weights_path=str(weights_path),
        kernel_size=kernel_size,
        blur_sigma=blur_sigma,
        lambda_=config["diffpir"]["lambda"],
        zeta=config["diffpir"]["zeta"],
        t_start=config["diffpir"].get("t_start", None),
        return_timing=True,
    )
    deg_np = tensor_to_np(degraded)
    res_np = tensor_to_np(restored)
    psnr = psnr_fn(gt_np, res_np, data_range=1.0)
    ssim = ssim_fn(gt_np, res_np, data_range=1.0, channel_axis=-1)
    entries.append((deg_np, res_np, psnr, ssim, inference_time))
    print(f"PSNR={psnr:.1f} SSIM={ssim:.3f} Tempo={inference_time:.2f}s")

# Crea figura: 4 righe x 4 colonne
COL_LABELS = ["GT (Originale)", "Degradata (Input)", "DiffPIR (Restored)", "Risultati"]
COL_COLORS = ["#2ecc71", "#e74c3c", "#ff8c00", "#555"]
fig, axes = plt.subplots(4, 4, figsize=(20, 12))
fig.patch.set_facecolor("white")

for row, (nl, (deg_np, res_np, psnr, ssim, tempo)) in enumerate(zip(noise_levels, entries)):
    for col in range(4):
        ax = axes[row, col]

        if col == 0:          # GT
            ax.imshow(gt_np)
            ax.set_ylabel(f"s_n = {nl}", fontsize=12, fontweight="bold", color="#333")
        elif col == 1:        # Degraded
            ax.imshow(deg_np)
        elif col == 2:        # DiffPIR
            ax.imshow(res_np)
        else:                 # Metrics
            ax.axis("off")
            # Sfondo carta
            ax.add_patch(plt.Rectangle((0.04, 0.04), 0.92, 0.92, fill=True,
                                        facecolor="#fafafa", edgecolor="#ddd", linewidth=2, linestyle="--",
                                        transform=ax.transAxes, zorder=0))
            # sigma_n in alto
            ax.text(0.5, 0.92, f"s_n = {nl}",
                    ha="center", va="top", fontsize=14, fontweight="bold", color="#ff8c00",
                    transform=ax.transAxes)
            # PSNR
            ax.text(0.5, 0.63, f"PSNR\n{psnr:.1f} dB",
                    ha="center", va="center", fontsize=15, fontweight="bold", color="#2c3e50",
                    transform=ax.transAxes)
            # SSIM
            ax.text(0.5, 0.38, f"SSIM\n{ssim:.3f}",
                    ha="center", va="center", fontsize=15, fontweight="bold", color="#2c3e50",
                    transform=ax.transAxes)
            # Tempo
            ax.text(0.5, 0.14, f"Tempo\n{tempo:.2f}s",
                    ha="center", va="center", fontsize=13, fontweight="bold", color="#888",
                    transform=ax.transAxes)

        ax.axis("off")

        # Intestazioni colonne (solo prima riga)
        if row == 0:
            ax.set_title(COL_LABELS[col], fontsize=13, fontweight="bold",
                          color=COL_COLORS[col], pad=8)

plt.subplots_adjust(left=0.03, right=0.99, bottom=0.02, top=0.92, wspace=0.02, hspace=0.06)

fig.text(0.5, 0.95, "DiffPIR — GT | Degradata | Restored | Metriche",
         ha="center", fontsize=18, fontweight="bold", color="#222")
fig.text(0.5, 0.91, "4 livelli di rumore con PSNR, SSIM e Tempo di inferenza",
         ha="center", fontsize=11, color="#888")

filepath = SAVE_DIR / "diffpir_2x4.png"
plt.savefig(filepath, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSalvato: {filepath}")
