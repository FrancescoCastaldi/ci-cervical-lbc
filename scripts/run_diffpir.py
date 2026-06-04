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
kernel_size = config["degradation"]["kernel_size"]
blur_sigma = config["degradation"]["blur_sigma"]
results_dir = Path(config["eval"]["results_dir"])
results_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DiffPIR — Deblur & Denoise")
print("=" * 60)
print(f"Noise levels: {noise_levels}")
print(f"Steps: {num_steps}")
print(f"Kernel: size={kernel_size}, sigma={blur_sigma}")
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
        degraded = degrade(gt, noise_level=noise_level)
        restored, inference_time = run_diffpir(
            degraded,
            num_steps=num_steps,
            noise_level=noise_level,
            kernel_size=kernel_size,
            blur_sigma=blur_sigma,
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
                save_path=results_dir / f"diffpir_noise{noise_level}_sample{i}.png"
            )
    rows.append({"method": "diffpir", "noise_level": noise_level,
                 "psnr": sum(psnr_list)/len(psnr_list),
                 "ssim": sum(ssim_list)/len(ssim_list),
                 "avg_inference_time": sum(time_list)/len(time_list)})
    print(f"[DiffPir] noise={noise_level} | PSNR={rows[-1]['psnr']:.2f} | SSIM={rows[-1]['ssim']:.4f} | Time={rows[-1]['avg_inference_time']:.1f}s")

pd.DataFrame(rows).to_csv(results_dir / "diffpir_results.csv", index=False)
print()
print(f"Risultati salvati in {results_dir / 'diffpir_results.csv'}")
