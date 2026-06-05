import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import degrade
from src.methods.unet.unet import UNet
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
noise_levels = config["degradation"]["noise_levels"]
results_dir = Path(config["eval"]["results_dir"])
results_dir.mkdir(parents=True, exist_ok=True)
kernel_size = config["degradation"]["kernel_size"] 
blur_sigma  = config["degradation"]["blur_sigma"] 

train_dataset = LBCDataset("data/splits/train.txt", config["dataset"]["image_size"])
train_loader = DataLoader(train_dataset, batch_size=config["unet"]["batch_size"], shuffle=True)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["unet"]["lr"])
criterion = nn.MSELoss()

for epoch in range(config["unet"]["epochs"]):
    model.train()
    total_loss = 0
    for batch_idx,gt in enumerate(train_loader):
        print(f"Epoca {epoch+1} | Elaborando il batch {batch_idx+1}/{len(train_loader)}...") # <--- Riga spia
        degraded = torch.stack([
            degrade(g, kernel_size=kernel_size, sigma=blur_sigma, noise_level=0.01) 
            for g in gt
        ])
        gt = gt.to(device)
        degraded = degraded.to(device)
        pred = model(degraded)
        loss = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{config['unet']['epochs']} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), results_dir / "unet_weights.pth")

model.eval()
test_dataset = LBCDataset("data/splits/test.txt", config["dataset"]["image_size"])
rows = []
for noise_level in noise_levels:
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for i, gt in enumerate(test_dataset):
            degraded = degrade(gt, kernel_size=kernel_size, sigma=blur_sigma, 
                   noise_level=noise_level).unsqueeze(0).to(device)
            restored = model(degraded).squeeze(0).cpu()
            m = evaluate(restored, gt)
            psnr_list.append(m["psnr"])
            ssim_list.append(m["ssim"])
            if i < config["eval"]["save_qualitative"]:
                show_comparison(
                    {"Degraded": degraded.squeeze(0).cpu(), "UNet": restored, "GT": gt},
                    save_path=results_dir / f"unet_noise{noise_level}_sample{i}.png"
                )
    rows.append({"method": "unet", "noise_level": noise_level,
                 "psnr": sum(psnr_list)/len(psnr_list),
                 "ssim": sum(ssim_list)/len(ssim_list)})
    print(f"[UNet] noise={noise_level} | PSNR={rows[-1]['psnr']:.2f} | SSIM={rows[-1]['ssim']:.4f}")

pd.DataFrame(rows).to_csv(results_dir / "unet_results.csv", index=False)
