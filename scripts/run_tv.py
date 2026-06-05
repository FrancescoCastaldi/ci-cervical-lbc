import sys
sys.path.insert(0, ".")

import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import degrade
from src.methods.tv.tv import tv_restore
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison

config = load_config()
noise_levels = config["degradation"]["noise_levels"]
kernel_size   = config["degradation"]["kernel_size"] 
blur_sigma    = config["degradation"]["blur_sigma"]    
lambda_reg = config["tv"]["lambda_reg"]
max_iter = config["tv"]["max_iter"]
results_dir = Path(config["eval"]["results_dir"]) / "tv"
results_dir.mkdir(parents=True, exist_ok=True)

dataset = LBCDataset("data/splits/test.txt", config["dataset"]["image_size"])

rows = []
for noise_level in noise_levels:
    psnr_list, ssim_list = [], []
    for i, gt in enumerate(dataset):
        degraded = degrade(gt, kernel_size=kernel_size,
                           sigma=blur_sigma, 
                           noise_level=noise_level)
        restored = tv_restore(degraded, kernel_size=kernel_size,
                              sigma=blur_sigma,            
                              lambda_reg=lambda_reg,
                              max_iter=max_iter)
        m = evaluate(restored, gt)
        psnr_list.append(m["psnr"])
        ssim_list.append(m["ssim"])
        if i < config["eval"]["save_qualitative"]:
            show_comparison(
                {"Degraded": degraded, "TV Restored": restored, "GT": gt},
                save_path=results_dir / f"tv_noise{noise_level}_sample{i}.png"
            )
    rows.append({"method": "tv", "noise_level": noise_level,
                 "psnr": sum(psnr_list)/len(psnr_list),
                 "ssim": sum(ssim_list)/len(ssim_list)})
    print(f"[TV] noise={noise_level} | PSNR={rows[-1]['psnr']:.2f} | SSIM={rows[-1]['ssim']:.4f}")

pd.DataFrame(rows).to_csv(results_dir / "tv_results.csv", index=False)
print("Salvato results/tv_results.csv")
