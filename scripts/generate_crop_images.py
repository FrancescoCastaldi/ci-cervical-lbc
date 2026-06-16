"""
Genera le immagini crop/diff per la presentazione (diapositiva Confronto).
Richiede i modelli UNet e DiffPIR addestrati.

Immagini generate:
  - results/qualitative_slides/crop_comparison.png
  - results/qualitative_slides/crop_diff_comparison.png
  - results/diff/all_diffs_0.1.png

Esegui con: python scripts/generate_crop_images.py
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
from src.methods.tv.tv import tv_restore
from src.methods.unet.unet import UNet
from src.methods.diffpir.diffpir import run_diffpir

config = load_config()
device = torch.device("cpu")
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

noise_levels = config["degradation"]["noise_levels"]
kernel_size = config["degradation"]["kernel_size"]
blur_sigma = config["degradation"]["blur_sigma"]
img_size = config["dataset"]["image_size"]

QUAL_DIR = Path("results/qualitative_slides")
DIFF_DIR = Path("results/diff")
QUAL_DIR.mkdir(parents=True, exist_ok=True)
DIFF_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
COLORS = {"GT": "#2ecc71", "TV": "#3498db", "UNet": "#9b59b6", "DiffPIR": "#ff8c00"}


def tensor_to_np(t):
    arr = t.detach().cpu().permute(1, 2, 0).clamp(-1, 1).numpy()
    return (arr * 0.5 + 0.5).astype(np.float64)


def compute_psnr_np(pred, gt):
    return psnr_fn(gt, pred, data_range=1.0)


def compute_ssim_np(pred, gt):
    return ssim_fn(gt, pred, data_range=1.0, channel_axis=-1)


def make_noise_map(batch_size, image_size, noise_level):
    return torch.full((batch_size, 1, image_size, image_size), noise_level)


def draw_image(ax, img_np, title="", metrics_text="", color="#333"):
    ax.imshow(img_np)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=color, pad=6)
    ax.axis("off")
    if metrics_text:
        ax.set_xlabel(metrics_text, fontsize=8, color="#555", labelpad=2)


# ── Carica sample ───────────────────────────────────────────────────────
dataset = LBCDataset("data/splits/test.txt", img_size)
sample_idx = 0
gt = dataset[sample_idx]
gt_np = tensor_to_np(gt)
print(f"Sample caricato: idx={sample_idx}, shape={gt.shape}")

# ── Carica modelli ──────────────────────────────────────────────────────
print("Caricamento modelli...")
unet_model = UNet(in_channels=4, out_channels=3, features=(16, 32, 64, 128))
unet_weights = Path("results/unet/best_model.pth")
if unet_weights.exists():
    unet_model.load_state_dict(torch.load(unet_weights, map_location="cpu", weights_only=True))
    unet_model.eval()
    print(f"  UNet: OK ({unet_weights})")
else:
    print(f"  UNet: peso non trovato ({unet_weights})")
    unet_model = None

diffpir_weights_path = Path(config["diffpir"]["weights"])
if not diffpir_weights_path.is_absolute():
    diffpir_weights_path = Path(__file__).resolve().parent.parent / diffpir_weights_path
diffpir_loaded = diffpir_weights_path.exists()
if diffpir_loaded:
    print(f"  DiffPIR: OK ({diffpir_weights_path})")
else:
    print(f"  DiffPIR: peso non trovato ({diffpir_weights_path})")

# ── Processa σ_n = 0.01 e 0.1 ──────────────────────────────────────────
results = {}
for nl in [0.01, 0.1]:
    print(f"\nProcessing sigma_n = {nl}...")
    degraded = degrade(gt, kernel_size=kernel_size, sigma=blur_sigma, noise_level=nl)
    deg_np = tensor_to_np(degraded)
    entry = {"gt": gt_np, "degraded": deg_np}

    # TV
    restored_tv = tv_restore(degraded, kernel_size=kernel_size,
                              sigma=blur_sigma,
                              lambda_reg=config["tv"]["lambda_reg"],
                              max_iter=config["tv"]["max_iter"])
    entry["TV"] = tensor_to_np(restored_tv)
    entry["psnr_tv"] = compute_psnr_np(entry["TV"], gt_np)
    entry["ssim_tv"] = compute_ssim_np(entry["TV"], gt_np)
    print(f"  TV:     PSNR={entry['psnr_tv']:.2f} SSIM={entry['ssim_tv']:.4f}")

    # UNet
    if unet_model is not None:
        with torch.no_grad():
            degraded_batch = degraded.unsqueeze(0)
            noise_map = make_noise_map(1, img_size, nl)
            model_input = torch.cat([degraded_batch, noise_map], dim=1)
            restored_unet = unet_model(model_input).squeeze(0).cpu()
        entry["UNet"] = tensor_to_np(restored_unet)
        entry["psnr_unet"] = compute_psnr_np(entry["UNet"], gt_np)
        entry["ssim_unet"] = compute_ssim_np(entry["UNet"], gt_np)
        print(f"  UNet:   PSNR={entry['psnr_unet']:.2f} SSIM={entry['ssim_unet']:.4f}")
    else:
        entry["UNet"] = np.zeros_like(gt_np)
        entry["psnr_unet"] = 0
        entry["ssim_unet"] = 0

    # DiffPIR
    if diffpir_loaded:
        restored_diffpir, _ = run_diffpir(
            degraded,
            num_steps=config["diffpir"]["num_steps"],
            noise_level=nl,
            weights_path=str(diffpir_weights_path),
            kernel_size=kernel_size,
            blur_sigma=blur_sigma,
            lambda_=config["diffpir"]["lambda"],
            zeta=config["diffpir"]["zeta"],
            t_start=config["diffpir"].get("t_start", None),
            return_timing=True,
        )
        entry["DiffPIR"] = tensor_to_np(restored_diffpir)
        entry["psnr_diffpir"] = compute_psnr_np(entry["DiffPIR"], gt_np)
        entry["ssim_diffpir"] = compute_ssim_np(entry["DiffPIR"], gt_np)
        print(f"  DiffPIR: PSNR={entry['psnr_diffpir']:.2f} SSIM={entry['ssim_diffpir']:.4f}")
    else:
        entry["DiffPIR"] = np.zeros_like(gt_np)
        entry["psnr_diffpir"] = 0
        entry["ssim_diffpir"] = 0

    results[nl] = entry


# ── Helper: crop ─────────────────────────────────────────────────────────
def crop_center(img_np, crop_size=112):
    """Ritaglia un quadrato centrale dell'immagine."""
    h, w = img_np.shape[:2]
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return img_np[y0:y0+crop_size, x0:x0+crop_size]


# ══════════════════════════════════════════════════════════════════════════
# FIGURA 1: crop_comparison.png (σ_n=0.01) — crop raffinato
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating: crop_comparison.png...")
entry = results[0.01]
METHODS = [("GT", entry["gt"], "#2ecc71", ""),
           ("Degradata", entry["degraded"], "#e74c3c", ""),
           ("TV", entry["TV"], "#3498db",
            f"PSNR {entry['psnr_tv']:.1f}  SSIM {entry['ssim_tv']:.3f}"),
           ("UNet", entry["UNet"], "#9b59b6",
            f"PSNR {entry['psnr_unet']:.1f}  SSIM {entry['ssim_unet']:.3f}"),
           ("DiffPIR", entry["DiffPIR"], "#ff8c00",
            f"PSNR {entry['psnr_diffpir']:.1f}  SSIM {entry['ssim_diffpir']:.3f}")]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.patch.set_facecolor("white")

for col, (name, img_np, color, metric) in enumerate(METHODS):
    # Riga 0: immagine intera
    draw_image(axes[0, col], img_np,
               title=name if col == 0 else "",
               metrics_text="", color=color)
    if col == 0:
        axes[0, col].set_ylabel("Full (256×256)", fontsize=10, fontweight="bold", color="#333")

    # Riga 1: crop centrale
    cropped = crop_center(img_np, 112)
    draw_image(axes[1, col], cropped,
               title=name if col == 0 else "",
               metrics_text=metric, color=color)
    if col == 0:
        axes[1, col].set_ylabel("Crop (112×112)", fontsize=10, fontweight="bold", color="#333")

plt.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.92, wspace=0.04, hspace=0.15)
fig.text(0.5, 0.95, "Confronto Rapido — σₙ = 0.01  |  Full vs Crop Centrale",
         ha="center", fontsize=14, fontweight="bold", color="#222")
fig.text(0.5, 0.04, "Crop: dettaglio 112×112 pixel al centro dell'immagine",
         ha="center", fontsize=9, color="#888")

plt.savefig(QUAL_DIR / "crop_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {QUAL_DIR / 'crop_comparison.png'}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURA 2: crop_diff_comparison.png (σ_n=0.1) — differenze
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating: crop_diff_comparison.png...")
entry = results[0.1]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.patch.set_facecolor("white")

TARGETS = [("GT", entry["gt"], "#2ecc71"),
           ("TV", entry["TV"], "#3498db"),
           ("UNet", entry["UNet"], "#9b59b6"),
           ("DiffPIR", entry["DiffPIR"], "#ff8c00")]

for col, (name, img_np, color) in enumerate(TARGETS):
    # Riga 0: crop
    cropped = crop_center(img_np, 112)
    metric = ""
    if name == "TV":
        metric = f"PSNR {entry['psnr_tv']:.1f}  SSIM {entry['ssim_tv']:.3f}"
    elif name == "UNet":
        metric = f"PSNR {entry['psnr_unet']:.1f}  SSIM {entry['ssim_unet']:.3f}"
    elif name == "DiffPIR":
        metric = f"PSNR {entry['psnr_diffpir']:.1f}  SSIM {entry['ssim_diffpir']:.3f}"
    draw_image(axes[0, col], cropped, title=name, metrics_text=metric, color=color)

    # Riga 1: mappa differenza (differenza assoluta dal GT, amplificata)
    gt_crop = crop_center(entry["gt"], 112)
    diff_map = np.abs(gt_crop.astype(float) - cropped.astype(float))
    diff_map = np.clip(diff_map * 5, 0, 1)  # amplifica 5x per visibilità

    axes[1, col].imshow(diff_map)
    axes[1, col].set_title("Diff ×5" if col == 0 else "", fontsize=11,
                            fontweight="bold", color="#e74c3c", pad=6)
    axes[1, col].axis("off")

plt.subplots_adjust(left=0.03, right=0.98, bottom=0.05, top=0.93, wspace=0.04, hspace=0.12)
fig.text(0.5, 0.96, "Dettaglio Crop + Mappe Differenza — σₙ = 0.1",
         ha="center", fontsize=14, fontweight="bold", color="#222")
fig.text(0.5, 0.03, "Differenza assoluta amplificata 5× per visibilità",
         ha="center", fontsize=9, color="#888")

plt.savefig(QUAL_DIR / "crop_diff_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {QUAL_DIR / 'crop_diff_comparison.png'}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURA 3: all_diffs_0.1.png — mappe complete
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating: all_diffs_0.1.png...")
entry = results[0.1]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.patch.set_facecolor("white")

METHOD_KEYS = [("TV", "TV", "#3498db"),
               ("UNet", "UNet", "#9b59b6"),
               ("DiffPIR", "DiffPIR", "#ff8c00")]

for col in range(4):
    ax = axes[col]
    if col == 0:
        ax.imshow(entry["gt"])
        ax.set_title("GT (Originale)", fontsize=13, fontweight="bold", color="#2ecc71", pad=8)
    else:
        name, key, color = METHOD_KEYS[col - 1]
        diff_map = np.abs(entry["gt"].astype(float) - entry[key].astype(float))
        diff_map = np.clip(diff_map * 5, 0, 1)
        im = ax.imshow(diff_map, cmap="hot")
        psnr = entry[f"psnr_{key.lower()}"]
        ssim = entry[f"ssim_{key.lower()}"]
        ax.set_title(f"{name} — Diff ×5", fontsize=13, fontweight="bold", color=color, pad=8)
        ax.set_xlabel(f"PSNR {psnr:.1f}  SSIM {ssim:.3f}", fontsize=9, color="#555")
    ax.axis("off")

plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.88, bottom=0.12)
fig.text(0.5, 0.95, "Mappe di Differenza (×5) — σₙ = 0.1",
         ha="center", fontsize=15, fontweight="bold", color="#222")

plt.savefig(DIFF_DIR / "all_diffs_0.1.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {DIFF_DIR / 'all_diffs_0.1.png'}")

print("\nTutte le immagini crop/diff sono state generate.")
