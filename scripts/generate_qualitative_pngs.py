"""
Genera PNG di alta qualita per la presentazione.
Legge il dataset, applica degradation e tutti e 3 i metodi,
produce figure professionali salvate in results/qualitative_slides/.
"""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import degrade
from src.methods.tv.tv import tv_restore
from src.methods.unet.unet import UNet
from src.methods.diffpir.diffpir import run_diffpir
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim

# ── Config ──────────────────────────────────────────────────────────────
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

# ── Style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
TITLE_COLORS = {"GT": "#2ecc71", "Degraded": "#e74c3c",
                "TV": "#3498db", "UNet": "#9b59b6", "DiffPIR": "#ff8c00"}


# ── Helpers ─────────────────────────────────────────────────────────────
def tensor_to_np(t):
    """Converte tensore [C,H,W] in [-1,1] a numpy [H,W,3] in [0,1]."""
    arr = t.detach().cpu().permute(1, 2, 0).clamp(-1, 1).numpy()
    return (arr * 0.5 + 0.5).astype(np.float64)


def compute_psnr_np(pred, gt):
    return _psnr(gt, pred, data_range=1.0)


def compute_ssim_np(pred, gt):
    return _ssim(gt, pred, data_range=1.0, channel_axis=-1)


def make_noise_map(batch_size, image_size, noise_level):
    return torch.full((batch_size, 1, image_size, image_size), noise_level)


def add_subtitle(fig, text, y=0.92, fontsize=14):
    """Aggiunge un sottotitolo centrato sopra la figura."""
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=fontsize, fontweight="bold", color="#333")


def draw_image(ax, img_np, title, metrics_text=""):
    """Disegna un'immagine con titolo colorato e metriche sottostanti."""
    ax.imshow(img_np)
    ax.set_title(title, fontsize=12, fontweight="bold",
                  color=TITLE_COLORS.get(title, "#333"), pad=6)
    ax.axis("off")
    if metrics_text:
        ax.set_xlabel(metrics_text, fontsize=8, color="#555", labelpad=2)


# ── Load models ─────────────────────────────────────────────────────────
print("Caricamento modelli...")

# UNet
unet_model = UNet(in_channels=4, out_channels=3, features=(16, 32, 64, 128))
unet_weights = Path("results/unet/best_model.pth")
if unet_weights.exists():
    unet_model.load_state_dict(torch.load(unet_weights, map_location="cpu", weights_only=True))
    unet_model.eval()
    print(f"  UNet: OK ({unet_weights})")
else:
    print(f"  UNet: peso non trovato ({unet_weights}), skip")
    unet_model = None

# DiffPIR weights path
diffpir_weights = Path(config["diffpir"]["weights"])
if not diffpir_weights.is_absolute():
    diffpir_weights = Path(__file__).resolve().parent.parent / diffpir_weights

print(f"  DiffPIR weights: {diffpir_weights}")
diffpir_loaded = diffpir_weights.exists()
if not diffpir_loaded:
    print(f"  DiffPIR: peso non trovato, skip")

# ── Load sample ─────────────────────────────────────────────────────────
dataset = LBCDataset("data/splits/test.txt", img_size)
sample_idx = 0
gt = dataset[sample_idx]  # [C, H, W] in [-1, 1]
print(f"Sample caricato: idx={sample_idx}, shape={gt.shape}")

# ── Process ─────────────────────────────────────────────────────────────
results = {}  # noise_level -> {"gt": np, "degraded": np, "TV": np, "UNet": np, "DiffPIR": np, "metrics": {...}}

for nl in noise_levels:
    print(f"\nProcessing noise sigma={nl}...")
    degraded = degrade(gt, kernel_size=kernel_size,
                       sigma=blur_sigma, noise_level=nl)

    entry = {
        "gt": tensor_to_np(gt),
        "degraded": tensor_to_np(degraded),
    }

    # ── TV ──
    print("  TV...", end=" ", flush=True)
    restored_tv = tv_restore(degraded, kernel_size=kernel_size,
                              sigma=blur_sigma,
                              lambda_reg=config["tv"]["lambda_reg"],
                              max_iter=config["tv"]["max_iter"])
    entry["TV"] = tensor_to_np(restored_tv)
    entry["psnr_tv"] = compute_psnr_np(entry["TV"], entry["gt"])
    entry["ssim_tv"] = compute_ssim_np(entry["TV"], entry["gt"])
    print(f"PSNR={entry['psnr_tv']:.2f} SSIM={entry['ssim_tv']:.4f}")

    # ── UNet ──
    if unet_model is not None:
        print("  UNet...", end=" ", flush=True)
        with torch.no_grad():
            degraded_batch = degraded.unsqueeze(0)
            noise_map = make_noise_map(1, img_size, nl)
            model_input = torch.cat([degraded_batch, noise_map], dim=1)
            restored_unet = unet_model(model_input).squeeze(0).cpu()
        entry["UNet"] = tensor_to_np(restored_unet)
        entry["psnr_unet"] = compute_psnr_np(entry["UNet"], entry["gt"])
        entry["ssim_unet"] = compute_ssim_np(entry["UNet"], entry["gt"])
        print(f"PSNR={entry['psnr_unet']:.2f} SSIM={entry['ssim_unet']:.4f}")
    else:
        entry["UNet"] = np.zeros_like(entry["gt"])
        entry["psnr_unet"] = 0
        entry["ssim_unet"] = 0

    # ── DiffPIR ──
    if diffpir_loaded:
        print("  DiffPIR...", end=" ", flush=True)
        restored_diffpir, _ = run_diffpir(
            degraded,
            num_steps=config["diffpir"]["num_steps"],
            noise_level=nl,
            weights_path=str(diffpir_weights),
            kernel_size=kernel_size,
            blur_sigma=blur_sigma,
            lambda_=config["diffpir"]["lambda"],
            zeta=config["diffpir"]["zeta"],
            t_start=config["diffpir"].get("t_start", None),
            return_timing=True,
        )
        entry["DiffPIR"] = tensor_to_np(restored_diffpir)
        entry["psnr_diffpir"] = compute_psnr_np(entry["DiffPIR"], entry["gt"])
        entry["ssim_diffpir"] = compute_ssim_np(entry["DiffPIR"], entry["gt"])
        print(f"PSNR={entry['psnr_diffpir']:.2f} SSIM={entry['ssim_diffpir']:.4f}")
    else:
        entry["DiffPIR"] = np.zeros_like(entry["gt"])
        entry["psnr_diffpir"] = 0
        entry["ssim_diffpir"] = 0

    results[nl] = entry


# ── Figure 1: Griglia completa 4×5 ──────────────────────────────────────
print("\nGenerating: griglia 4x5...")
COLUMNS = ["GT", "Degraded", "TV", "UNet", "DiffPIR"]
n_cols = len(COLUMNS)
n_rows = len(noise_levels)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
fig.patch.set_facecolor("white")

for row, nl in enumerate(noise_levels):
    entry = results[nl]
    for col, method in enumerate(COLUMNS):
        ax = axes[row, col]
        img_np = entry[method.lower() if method.lower() in entry else method]

        # Titolo colonna solo in prima riga
        if row == 0:
            title = method
        else:
            title = ""

        # Metriche per metodi di restauro
        metrics_text = ""
        if method == "TV":
            metrics_text = f"PSNR {entry['psnr_tv']:.1f}  SSIM {entry['ssim_tv']:.3f}"
        elif method == "UNet":
            metrics_text = f"PSNR {entry['psnr_unet']:.1f}  SSIM {entry['ssim_unet']:.3f}"
        elif method == "DiffPIR":
            metrics_text = f"PSNR {entry['psnr_diffpir']:.1f}  SSIM {entry['ssim_diffpir']:.3f}"

        draw_image(ax, img_np, title, metrics_text)

        # Noise level label a sinistra
        if col == 0:
            ax.set_ylabel(f"σ = {nl}", fontsize=14, fontweight="bold", color="#333")

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.03, top=0.94,
                    wspace=0.06, hspace=0.15)

# Titolo generale
method_labels = {
    "GT": "Ground Truth", "Degraded": "Degradata (Blur + AWGN)",
    "TV": "TV (Variazionale)", "UNet": "UNet (Deep Learning)",
    "DiffPIR": "DiffPIR (Generativo)"
}
col_labels = "  |  ".join([f"{k}: {v}" for k, v in method_labels.items()])
fig.text(0.5, 0.96, f"Confronto Qualitativo — Sample #{sample_idx}",
         ha="center", fontsize=18, fontweight="bold", color="#222")
fig.text(0.5, 0.925, col_labels,
         ha="center", fontsize=9, color="#888")

plt.savefig(SAVE_DIR / "all_methods_grid.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {SAVE_DIR / 'all_methods_grid.png'}")


# ── Figure 2 per metodo: 4 righe × 2 colonne (Degraded | Restored) ─────
print("\nGenerating: per-method comparison...")
methods_info = [
    ("TV",     "TV (Total Variation)",    "#3498db"),
    ("UNet",   "UNet (Deep Learning)",    "#9b59b6"),
    ("DiffPIR", "DiffPIR (Generativo)",   "#ff8c00"),
]

for method_key, method_title, color in methods_info:
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    fig.patch.set_facecolor("white")

    for row, nl in enumerate(noise_levels):
        entry = results[nl]
        gt_np = entry["gt"]
        deg_np = entry["degraded"]
        # Se il metodo non e disponibile, mostra placeholder
        rest_np = entry.get(method_key, np.zeros_like(gt_np))

        for col, (img, label, label_color) in enumerate([
            (gt_np, "GT", "#2ecc71"),
            (deg_np, "Degradata", "#e74c3c"),
            (rest_np, method_key.split("_")[0], color),
        ]):
            ax = axes[row, col]
            metric_text = ""
            if col == 2:
                psnr_key = f"psnr_{method_key.lower()}"
                ssim_key = f"ssim_{method_key.lower()}"
                if psnr_key in entry:
                    metric_text = f"PSNR {entry[psnr_key]:.1f}  SSIM {entry[ssim_key]:.3f}"
            draw_image(ax, img, label if row == 0 else "", metric_text)

            # Noise level a sinistra
            if col == 0:
                ax.set_ylabel(f"σ = {nl}", fontsize=13, fontweight="bold", color="#333")

    plt.subplots_adjust(left=0.07, right=0.97, bottom=0.03, top=0.93,
                        wspace=0.04, hspace=0.12)
    fig.text(0.5, 0.95, f"Metodo: {method_title}",
             ha="center", fontsize=16, fontweight="bold", color=color)

    fname = method_key.lower().replace(" ", "_")
    plt.savefig(SAVE_DIR / f"{fname}_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> {SAVE_DIR / f'{fname}_comparison.png'}")


# ── Figure 3: Un metodo per noise level, tutti i metodi affiancati ─────
print("\nGenerating: noise-level panels...")
for nl in noise_levels:
    entry = results[nl]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.patch.set_facecolor("white")

    images = [
        ("GT", entry["gt"], "#2ecc71"),
        ("Degradata", entry["degraded"], "#e74c3c"),
        ("TV", entry["TV"], "#3498db"),
        ("UNet", entry["UNet"], "#9b59b6"),
        ("DiffPIR", entry["DiffPIR"], "#ff8c00"),
    ]

    met_keys = {"TV": ("psnr_tv", "ssim_tv"),
                "UNet": ("psnr_unet", "ssim_unet"),
                "DiffPIR": ("psnr_diffpir", "ssim_diffpir")}

    for col, (name, img_np, clr) in enumerate(images):
        ax = axes[col]
        metric = ""
        if name in met_keys:
            pk, sk = met_keys[name]
            metric = f"PSNR {entry[pk]:.1f}  SSIM {entry[sk]:.3f}"
        draw_image(ax, img_np, name, metric)

    plt.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.88, bottom=0.08)
    fig.text(0.5, 0.94, f"Confronto — σ = {nl}",
             ha="center", fontsize=16, fontweight="bold", color="#222")

    plt.savefig(SAVE_DIR / f"noise_{nl}_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> {SAVE_DIR / f'noise_{nl}_comparison.png'}")


# ── Figure 4: Grid 2×2 solo metodi (TV, UNet, DiffPIR + row label) ─────
# Un singolo noise level con tutti e 3 i metodi in griglia
print("\nGenerating: metodo grid single noise...")
for nl in [0.05, 0.1]:  # i due noise level piu significativi
    entry = results[nl]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("white")

    for col, (name, key, clr) in enumerate([
        ("GT", "gt", "#2ecc71"),
        ("TV", "TV", "#3498db"),
        ("UNet", "UNet", "#9b59b6"),
        ("DiffPIR", "DiffPIR", "#ff8c00"),
    ]):
        ax = axes[col]
        img = entry[key]
        metric = ""
        if key == "TV":
            metric = f"{entry['psnr_tv']:.1f} dB / {entry['ssim_tv']:.3f}"
        elif key == "UNet":
            metric = f"{entry['psnr_unet']:.1f} dB / {entry['ssim_unet']:.3f}"
        elif key == "DiffPIR":
            metric = f"{entry['psnr_diffpir']:.1f} dB / {entry['ssim_diffpir']:.3f}"

        draw_image(ax, img, name, metric)

    plt.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.88, bottom=0.08)
    fig.text(0.5, 0.94, f"Metodi a confronto — σ = {nl}",
             ha="center", fontsize=16, fontweight="bold", color="#222")

    plt.savefig(SAVE_DIR / f"methods_noise_{nl}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> {SAVE_DIR / f'methods_noise_{nl}.png'}")

print(f"\nTutti i PNG salvati in {SAVE_DIR.resolve()}/")
