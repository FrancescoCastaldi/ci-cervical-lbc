"""
Genera noise_strip_crops.png per la presentazione.
Mostra: GT | Blur only | Blur + noise a 4 livelli (σₙ = 0.005, 0.01, 0.05, 0.1).
Non richiede modelli ML — solo la pipeline di degradazione.

Esegui con: python scripts/generate_noise_strip.py
"""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import degrade, apply_blur

config = load_config()
device = torch.device("cpu")
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

noise_levels = config["degradation"]["noise_levels"]
kernel_size = config["degradation"]["kernel_size"]
blur_sigma = config["degradation"]["blur_sigma"]
img_size = config["dataset"]["image_size"]

SAVE_DIR = Path("results/qualitative_slides")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Carica sample
dataset = LBCDataset("data/splits/test.txt", img_size)
gt = dataset[0]  # [C, H, W] in [-1, 1]

def tensor_to_np(t):
    arr = t.detach().cpu().permute(1, 2, 0).clamp(-1, 1).numpy()
    return (arr * 0.5 + 0.5).astype(np.float64)

# Genera strip: GT, blur only, poi blur + noise ai 4 livelli
gt_np = tensor_to_np(gt)
blur_only = apply_blur(gt, kernel_size=kernel_size, sigma=blur_sigma)
blur_only_np = tensor_to_np(blur_only)

strips = [("GT (Originale)", gt_np), ("Blur only (σ=2)", blur_only_np)]
for nl in noise_levels:
    degraded = degrade(gt, kernel_size=kernel_size, sigma=blur_sigma, noise_level=nl)
    strips.append((f"Blur + AWGN σ={nl}", tensor_to_np(degraded)))

# Crea figura orizzontale
n = len(strips)
fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
fig.patch.set_facecolor("white")

for i, (label, img_np) in enumerate(strips):
    ax = axes[i]
    ax.imshow(img_np)
    ax.set_title(label, fontsize=10, fontweight="bold", color="#333", pad=8)
    ax.axis("off")

plt.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.88, bottom=0.02)
fig.text(0.5, 0.95, "Pipeline di Degradazione — Blur Gaussiano + AWGN a 4 livelli",
         ha="center", fontsize=14, fontweight="bold", color="#222")

filepath = SAVE_DIR / "noise_strip_crops.png"
plt.savefig(filepath, dpi=200, bbox_inches="tight")
plt.close()
print(f"Salvato: {filepath}")
