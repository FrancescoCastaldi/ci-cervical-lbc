import os
import yaml
import random
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


def load_config(path="configs/experiment.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_splits(config):
    raw_dir = Path(config["dataset"]["root"])
    splits_dir = Path(config["dataset"]["splits"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.jpg"))
    random.seed(config["seed"])
    random.shuffle(all_images)

    n = config["dataset"]["subset_size"]
    all_images = all_images[:n]

    n_train = int(n * config["dataset"]["train_ratio"])
    n_val = int(n * config["dataset"]["val_ratio"])

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    for split, paths in splits.items():
        with open(splits_dir / f"{split}.txt", "w") as f:
            for p in paths:
                f.write(str(p) + "\n")

    return splits


class LBCDataset(Dataset):
    def __init__(self, split_file, image_size=256):
        with open(split_file) as f:
            self.paths = [Path(l.strip()) for l in f.readlines()]
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)
