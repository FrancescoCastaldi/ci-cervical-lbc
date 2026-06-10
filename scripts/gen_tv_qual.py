"""
Genera immagini qualitative per TV (145 img test, 4 noise level).
Esegue solo le prime 6 immagini per noise level (indicizzate 0-5).
"""
import sys
sys.path.insert(0, ".")

import torch
from pathlib import Path
from tqdm import tqdm

from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import degrade
from src.methods.tv.tv import tv_restore
from src.plots.visualize import show_comparison

config = load_config()
noise_levels = config["degradation"]["noise_levels"]
kernel_size = config["degradation"]["kernel_size"]
blur_sigma = config["degradation"]["blur_sigma"]
lambda_reg = config["tv"]["lambda_reg"]
max_iter = config["tv"]["max_iter"]
n_qual = config["eval"]["save_qualitative"]  # 6

output_dir = Path(config["eval"]["results_dir"]) / "tv"
qual_dir = output_dir / "qualitative"
qual_dir.mkdir(parents=True, exist_ok=True)

dataset = LBCDataset("data/splits/test.txt", config["dataset"]["image_size"])

print(f"Generating TV qualitative: {n_qual} img × {len(noise_levels)} noise levels")
print(f"Saving to: {qual_dir}")

for noise_level in noise_levels:
    for i in tqdm(range(n_qual), desc=f"TV noise={noise_level}"):
        gt = dataset[i]
        degraded = degrade(gt, kernel_size=kernel_size,
                           sigma=blur_sigma, noise_level=noise_level)
        restored = tv_restore(degraded, kernel_size=kernel_size,
                              sigma=blur_sigma, lambda_reg=lambda_reg,
                              max_iter=max_iter)
        show_comparison(
            {"GT": gt, "Degraded": degraded, "TV Restored": restored},
            save_path=qual_dir / f"noise_{noise_level}_sample{i}.png"
        )

print(f"\n✅ TV qualitative complete! {n_qual * len(noise_levels)} images saved.")
