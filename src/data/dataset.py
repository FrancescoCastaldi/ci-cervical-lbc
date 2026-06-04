import yaml
import random
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path):
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_config(path=None):
    if path is None:
        path = PROJECT_ROOT / "configs" / "experiment.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def build_splits(config):
    raw_dir = _resolve(config["dataset"]["root"])
    splits_dir = _resolve(config["dataset"]["splits"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.jpg"))
    print(f"Immagini trovate: {len(all_images)}")

    valid_images = [p for p in all_images if is_valid_image(p)]
    print(f"Immagini valide: {len(valid_images)} | Corrotte/scartate: {len(all_images) - len(valid_images)}")

    random.seed(config["seed"])
    random.shuffle(valid_images)

    n = min(config["dataset"]["subset_size"], len(valid_images))
    valid_images = valid_images[:n]

    n_train = int(n * config["dataset"]["train_ratio"])
    n_val = int(n * config["dataset"]["val_ratio"])

    splits = {
        "train": valid_images[:n_train],
        "val": valid_images[n_train:n_train + n_val],
        "test": valid_images[n_train + n_val:],
    }

    for split, paths in splits.items():
        with open(splits_dir / f"{split}.txt", "w") as f:
            for p in paths:
                f.write(str(p) + "\n")
        print(f"{split}: {len(paths)} immagini")

    return splits


class LBCDataset(Dataset):
    def __init__(self, split_file, image_size=256, return_path=False):
        with open(_resolve(split_file)) as f:
            self.paths = [_resolve(l.strip()) for l in f.readlines() if l.strip()]
        self.image_size = image_size
        self.return_path = return_path
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        tensor = self.transform(img)
        if self.return_path:
            return tensor, str(self.paths[idx])
        return tensor


class DegradedDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir):
        self.degraded_paths = sorted(_resolve(degraded_dir).glob("*.pt"))
        self.gt_dir = _resolve(gt_dir)

    def __len__(self):
        return len(self.degraded_paths)

    def __getitem__(self, idx):
        import torch
        deg_path = self.degraded_paths[idx]
        gt_path = self.gt_dir / deg_path.name
        return torch.load(deg_path), torch.load(gt_path)
