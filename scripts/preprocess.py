import sys
sys.path.insert(0, ".")

import torch
from pathlib import Path
from tqdm import tqdm
from src.data.dataset import load_config, build_splits, LBCDataset
from src.degradation.degradation import degrade

config = load_config()

print("=== STEP 1: Build splits ===")
splits = build_splits(config)

image_size = config["dataset"]["image_size"]
kernel_size = config["degradation"]["kernel_size"]
blur_sigma = config["degradation"]["blur_sigma"]
noise_levels = config["degradation"]["noise_levels"]
splits_dir = Path(config["dataset"]["splits"])

print("\n=== STEP 2: Genera immagini degradate su disco ===")
for split in ["train", "val", "test"]:
    dataset = LBCDataset(splits_dir / f"{split}.txt", image_size=image_size)
    gt_dir = Path(f"data/degraded/{split}/gt")
    gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{split}] Salvo ground truth...")
    for i, img in enumerate(tqdm(dataset)):
        torch.save(img, gt_dir / f"{i:05d}.pt")

    for noise_level in noise_levels:
        deg_dir = Path(f"data/degraded/{split}/noise_{noise_level}")
        deg_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{split}] Genero degraded con noise={noise_level}...")
        for i, img in enumerate(tqdm(dataset)):
            degraded = degrade(img, kernel_size=kernel_size, sigma=blur_sigma, noise_level=noise_level)
            torch.save(degraded, deg_dir / f"{i:05d}.pt")

print("\n=== STEP 3: Salva qualche esempio visivo ===")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

example_dataset = LBCDataset(splits_dir / "test.txt", image_size=image_size)
examples_dir = Path("data/degraded/examples")
examples_dir.mkdir(parents=True, exist_ok=True)

for i in range(min(6, len(example_dataset))):
    gt = example_dataset[i]
    fig, axes = plt.subplots(1, len(noise_levels) + 1, figsize=(4 * (len(noise_levels) + 1), 4))
    def to_img(t):
        return (t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    axes[0].imshow(to_img(gt))
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    for j, noise_level in enumerate(noise_levels):
        deg = degrade(gt, kernel_size=kernel_size, sigma=blur_sigma, noise_level=noise_level)
        axes[j + 1].imshow(to_img(deg))
        axes[j + 1].set_title(f"noise={noise_level}")
        axes[j + 1].axis("off")
    plt.tight_layout()
    plt.savefig(examples_dir / f"example_{i:02d}.png", dpi=100)
    plt.close()

print(f"Esempi visivi salvati in data/degraded/examples/")
print("\nPreprocessing completato.")
