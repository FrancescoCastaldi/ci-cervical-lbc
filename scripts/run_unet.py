"""
UNet Snella - Deblur & Denoise (End-to-End)
============================================
Architettura ridotta (~500K params) con condizionamento sul noise level
per training su CPU (50 epoche in ~20 minuti).

Uso:
    python scripts/run_unet.py
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import gaussian_kernel
from src.methods.unet.unet import UNet
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison


# ==============================================================================
# Batch degradation helper (vettorizzata per CPU)
# ==============================================================================
def degrade_batch(images, kernel_size=9, sigma=2.0, noise_level=0.01):
    """Degrada un batch di immagini (B, C, H, W) in una sola passata."""
    kernel = gaussian_kernel(kernel_size, sigma).to(images.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(images.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel, padding=padding, groups=images.shape[1])
    noise = torch.randn_like(blurred) * noise_level
    return torch.clamp(blurred + noise, -1.0, 1.0)


def make_noise_map(batch_size, image_size, noise_level, device):
    """Crea una mappa di noise level costante per il condizionamento."""
    return torch.full((batch_size, 1, image_size, image_size),
                      noise_level, device=device)


# ==============================================================================
# Setup
# ==============================================================================
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
epochs = config["unet"]["epochs"]
lr = config["unet"]["lr"]
img_size = config["dataset"]["image_size"]

method_name = "unet"
output_dir = Path(config["eval"]["results_dir"]) / method_name
qual_dir = output_dir / "qualitative"
output_dir.mkdir(parents=True, exist_ok=True)
qual_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("UNet Snella - Deblur & Denoise (End-to-End)")
print("=" * 60)
print(f"Device     : {device}")
print(f"Noise      : {noise_levels}")
print(f"Epochs     : {epochs}")
print(f"Batch size : {batch_size}")
print(f"LR         : {lr}")
print(f"Output     : {output_dir}")
print("=" * 60)


# ==============================================================================
# Caricamento dati
# ==============================================================================
train_dataset = LBCDataset("data/splits/train.txt", img_size)
val_dataset = LBCDataset("data/splits/val.txt", img_size)
test_dataset = LBCDataset("data/splits/test.txt", img_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"\nTrain : {len(train_dataset)} img")
print(f"Val   : {len(val_dataset)} img")
print(f"Test  : {len(test_dataset)} img")


# ==============================================================================
# Modello snello (~500K params)
# ==============================================================================
model = UNet(in_channels=4, out_channels=3, features=(16, 32, 64, 128)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parametri: {n_params:,} (~{n_params/1e6:.1f}M)")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

# L1 loss invece di MSE: meno blur, preserva meglio i bordi
criterion = nn.L1Loss()


# ==============================================================================
# Training con multi-noise augmentation + validation su tutti i noise levels
# ==============================================================================
print("\n" + "-" * 60)
print("TRAINING")
print("-" * 60)

best_val_psnr = -float("inf")
best_epoch = 0
train_losses = []
val_psnrs = []

for epoch in range(epochs):
    # --- Training ---
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                unit="batch", leave=False)
    for gt in pbar:
        gt = gt.to(device)

        # Multi-noise: campiona un noise level random per batch
        noise = np.random.choice(noise_levels)
        degraded = degrade_batch(gt, kernel_size=kernel_size,
                                 sigma=blur_sigma, noise_level=noise)

        # Condizionamento: concatena mappa del noise level
        noise_map = make_noise_map(gt.size(0), img_size, noise, device)
        model_input = torch.cat([degraded, noise_map], dim=1)

        pred = model(model_input)
        loss = criterion(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", noise=f"s={noise}")

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # --- Validation su tutti i 4 noise levels (25% subset per velocit�) ---
    model.eval()
    val_half = max(1, len(val_dataset) // 4)
    val_indices = torch.randperm(len(val_dataset))[:val_half].tolist()
    val_psnr_avg = 0.0
    val_count = 0

    with torch.no_grad():
        for idx in val_indices:
            gt = val_dataset[idx].unsqueeze(0).to(device)
            # Campiona un noise level random per validazione
            val_noise = np.random.choice(noise_levels)
            degraded = degrade_batch(gt, kernel_size=kernel_size,
                                     sigma=blur_sigma, noise_level=val_noise)
            noise_map = make_noise_map(1, img_size, val_noise, device)
            model_input = torch.cat([degraded, noise_map], dim=1)
            pred = model(model_input)
            m = evaluate(pred[0].cpu(), gt[0].cpu())
            val_psnr_avg += m["psnr"]
            val_count += 1

    avg_val_psnr = val_psnr_avg / val_count
    val_psnrs.append(avg_val_psnr)

    scheduler.step(avg_val_psnr)

    improved = avg_val_psnr > best_val_psnr
    marker = " *" if improved else ""
    print(f"Epoch {epoch+1:2d}/{epochs} | "
          f"Train Loss: {avg_loss:.4f} | "
          f"Val PSNR: {avg_val_psnr:.2f} dB{marker}")

    if improved:
        best_val_psnr = avg_val_psnr
        best_epoch = epoch + 1
        torch.save(model.state_dict(), output_dir / "best_model.pth")

print(f"\nMiglior modello: epoch {best_epoch} (Val PSNR: {best_val_psnr:.2f} dB)")


# ==============================================================================
# Valutazione su test set (tutti i 4 noise level, tutte le 145 immagini)
# ==============================================================================
print("\n" + "-" * 60)
print("VALUTAZIONE SU TEST SET")
print("-" * 60)

model.load_state_dict(torch.load(output_dir / "best_model.pth", weights_only=True))
model.eval()

rows = []
for noise_level in noise_levels:
    psnr_list, ssim_list, time_list = [], [], []
    pbar = tqdm(enumerate(test_dataset), total=len(test_dataset),
                desc=f"[UNet] noise={noise_level}", unit="img")
    for i, gt in pbar:
        gt_t = gt.unsqueeze(0).to(device)

        # Degrada
        degraded = degrade_batch(gt_t, kernel_size=kernel_size,
                                 sigma=blur_sigma, noise_level=noise_level)

        # Condizionamento noise level
        noise_map = make_noise_map(1, img_size, noise_level, device)
        model_input = torch.cat([degraded, noise_map], dim=1)

        t0 = time.time()
        with torch.no_grad():
            restored = model(model_input).squeeze(0).cpu()
        inference_time = time.time() - t0

        m = evaluate(restored, gt)
        psnr_list.append(m["psnr"])
        ssim_list.append(m["ssim"])
        time_list.append(inference_time)

        pbar.set_postfix(psnr=f"{m['psnr']:.2f}", ssim=f"{m['ssim']:.4f}")

        if i < config["eval"]["save_qualitative"]:
            show_comparison(
                {"Degraded": degraded.squeeze(0).cpu(),
                 "UNet": restored, "GT": gt},
                save_path=qual_dir / f"noise_{noise_level}_sample{i}.png"
            )

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_time = sum(time_list) / len(time_list)
    rows.append({
        "method": "unet",
        "noise_level": noise_level,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "avg_inference_time": avg_time,
    })
    print(f"[UNet] noise={noise_level} | "
          f"PSNR={avg_psnr:.2f} dB | "
          f"SSIM={avg_ssim:.4f} | "
          f"Time={avg_time:.4f}s")


# ==============================================================================
# Salvataggio risultati
# ==============================================================================
csv_path = output_dir / "metrics.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)

print(f"\n{'=' * 60}")
print(f"Risultati salvati in {csv_path}")
print(f"Qualitative salvate in  {qual_dir}/")
print(f"Modello salvato in      {output_dir / 'best_model.pth'}")
print(f"{'=' * 60}")
