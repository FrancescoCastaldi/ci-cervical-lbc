# Data

This folder is **not tracked by git** (raw data and processed files are excluded via `.gitignore`).

## Download

Download the Mendeley LBC Cervical Cancer dataset from:  
https://data.mendeley.com/datasets/zddtpgzv63/2

## Structure

After download, place files as follows:

```
data/
├── raw/          # original images as downloaded
├── processed/    # resized and normalized images (256x256)
└── splits/       # train.txt, val.txt, test.txt with image paths
```

Run the preprocessing script to populate `processed/` and `splits/`:

```bash
python scripts/preprocess.py
```
