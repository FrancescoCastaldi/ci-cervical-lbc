import matplotlib.pyplot as plt
import torch
import pandas as pd
from pathlib import Path


def show_comparison(images_dict, save_path=None):
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (title, img) in zip(axes, images_dict.items()):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metrics(results_dir, save_path=None):
    results_dir = Path(results_dir)
    dfs = []
    for method_dir in sorted(results_dir.iterdir()):
        csv_file = method_dir / "metrics.csv"
        if csv_file.exists():
            dfs.append(pd.read_csv(csv_file))
    if not dfs:
        print(f"Nessun risultato trovato in {results_dir}")
        return
    df = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["psnr", "ssim"]):
        for method, group in df.groupby("method"):
            ax.plot(group["noise_level"], group[metric], marker="o", label=method)
        ax.set_xlabel("Noise level")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs Noise level")
        ax.legend()
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
