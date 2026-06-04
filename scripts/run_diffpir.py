import sys
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import degrade
from src.methods.diffpir.diffpir import run_diffpir
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison

config = load_config()
noise_levels = config["degradation"]["noise_levels"]
num_steps = config["diffpir"]["num_steps"]
results_dir = Path(config["eval"]["results_dir"])
results_dir.mkdir(parents=True, exist_ok=True)

test_dataset = LBCDataset("data/splits/test.txt", config["dataset"]["image_size"])

rows = []
for noise_level in noise_levels:
    psnr_list, ssim_list = [], []
    for i, gt in enumerate(test_dataset):
        degraded = degrade(gt, noise_level=noise_level)
        restored = run_diffpir(degraded, num_steps=num_steps, noise_level=noise_level)
        m = evaluate(restored, gt)
        psnr_list.append(m["psnr"])
        ssim_list.append(m["ssim"])
        if i < config["eval"]["save_qualitative"]:
            show_comparison(
                {"Degraded": degraded, "DiffPir": restored, "GT": gt},
                save_path=results_dir / f"diffpir_noise{noise_level}_sample{i}.png"
            )
    rows.append({"method": "diffpir", "noise_level": noise_level,
                 "psnr": sum(psnr_list)/len(psnr_list),
                 "ssim": sum(ssim_list)/len(ssim_list)})
    print(f"[DiffPir] noise={noise_level} | PSNR={rows[-1]['psnr']:.2f} | SSIM={rows[-1]['ssim']:.4f}")

pd.DataFrame(rows).to_csv(results_dir / "diffpir_results.csv", index=False)
