import sys
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.data.dataset import load_config, LBCDataset
from src.degradation.degradation import degrade
from src.methods.diffpir.diffpir import run_diffpir
from src.eval.metrics import evaluate
from src.plots.visualize import show_comparison

config = load_config()
noise_levels = config["degradation"]["noise_levels"]
num_steps = config["diffpir"]["num_steps"]
lambda_ = config["diffpir"]["lambda"]
zeta = config["diffpir"]["zeta"]
t_start = config["diffpir"].get("t_start", None)
weights_path = Path(config["diffpir"]["weights"])
if not weights_path.is_absolute():
    weights_path = Path(__file__).resolve().parent.parent / weights_path
kernel_size = config["degradation"]["kernel_size"]
blur_sigma = config["degradation"]["blur_sigma"]

method_name = "diffpir"
output_dir = Path(config["eval"]["results_dir"]) / method_name
qual_dir = output_dir / "qualitative"
output_dir.mkdir(parents=True, exist_ok=True)
qual_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DiffPIR — Deblur & Denoise")
print("=" * 60)
print(f"Noise levels: {noise_levels}")
print(f"Steps: {num_steps}")
print(f"Kernel: size={kernel_size}, sigma={blur_sigma}")
print(f"Output: {output_dir}")
print("=" * 60)

test_dataset = LBCDataset("data/splits/test.txt", config["dataset"]["image_size"])
max_test = config["diffpir"].get("max_test_images", len(test_dataset))
print(f"Immagini di test: {len(test_dataset)} (uso {max_test})")
print()

rows = []
for noise_level in noise_levels:
    psnr_list, ssim_list, time_list = [], [], []
    pbar = tqdm(
        enumerate(test_dataset),
        total=max_test,
        desc=f"[DiffPir] noise={noise_level}",
        unit="img",
    )
    for i, gt in pbar:
        if i >= max_test:
            break
        degraded = degrade(gt, kernel_size=kernel_size, sigma=blur_sigma, noise_level=noise_level)
        restored, inference_time = run_diffpir(
            degraded,
            num_steps=num_steps,
            noise_level=noise_level,
            weights_path=weights_path,
            kernel_size=kernel_size,
            blur_sigma=blur_sigma,
            lambda_=lambda_,
            zeta=zeta,
            t_start=t_start,
            return_timing=True,
        )
        m = evaluate(restored, gt)
        psnr_list.append(m["psnr"])
        ssim_list.append(m["ssim"])
        time_list.append(inference_time)
        pbar.set_postfix(psnr=f"{m['psnr']:.2f}", ssim=f"{m['ssim']:.4f}", time=f"{inference_time:.1f}s")
        if i < config["eval"]["save_qualitative"]:
            show_comparison(
                {"Degraded": degraded, "DiffPir": restored, "GT": gt},
                save_path=qual_dir / f"noise_{noise_level}_sample{i}.png"
            )
    rows.append({"method": "diffpir", "noise_level": noise_level,
                 "psnr": sum(psnr_list)/len(psnr_list),
                 "ssim": sum(ssim_list)/len(ssim_list),
                 "avg_inference_time": sum(time_list)/len(time_list)})
    print(f"[DiffPir] noise={noise_level} | PSNR={rows[-1]['psnr']:.2f} | SSIM={rows[-1]['ssim']:.4f} | Time={rows[-1]['avg_inference_time']:.1f}s")

csv_path = output_dir / "metrics.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)
print()
print(f"Risultati salvati in {csv_path}")
