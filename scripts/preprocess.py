import sys
sys.path.insert(0, ".")

from src.data.dataset import load_config, build_splits

config = load_config()
splits = build_splits(config)
for split, paths in splits.items():
    print(f"{split}: {len(paths)} images")
