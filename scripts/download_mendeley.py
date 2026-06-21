#!/usr/bin/env python3
"""
Download the Mendeley LBC Cervical Cancer dataset.
Downloads all 973 images from the public API, organized into class folders.

Usage:
    python scripts/download_mendeley.py
    python scripts/download_mendeley.py --max-files 50   # download only first 50 for testing
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

API_URL = "https://data.mendeley.com/public-api/datasets/zddtpgzv63"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def get_file_list():
    """Fetch the full file listing from the Mendeley API using curl."""
    result = subprocess.run(
        ["curl", "-sL", API_URL],
        capture_output=True, text=True, timeout=30
    )
    data = json.loads(result.stdout)
    return data.get("files", [])


def map_folder_to_class(files):
    """Map folder IDs to class names based on filename prefixes."""
    folder_classes = {}
    for f in files:
        name = f["filename"]
        fid = f.get("folder_id", "unknown")
        for cls in ["HSIL", "LSIL", "NILM", "SCC"]:
            if name.startswith(cls):
                folder_classes[fid] = cls
                break
    # Fallback for any unmapped folders
    unmapped = set(f.get("folder_id", "unknown") for f in files) - set(folder_classes.keys())
    for fid in unmapped:
        folder_classes[fid] = f"unknown_{fid[:8]}"
    return folder_classes


def download_file(file_info, dest_dir):
    """Download a single file. Returns (filename, success, error_msg)."""
    filename = file_info["filename"]
    download_url = file_info["content_details"]["download_url"]
    dest_path = dest_dir / filename

    if dest_path.exists():
        return filename, True, "already exists"

    try:
        req = Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
        dest_path.write_bytes(data)
        return filename, True, None
    except Exception as e:
        return filename, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Download Mendeley LBC dataset")
    parser.add_argument("--max-files", type=int, default=None, help="Max files to download (testing)")
    parser.add_argument("--parallel", type=int, default=8, help="Parallel downloads")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between batches (seconds)")
    args = parser.parse_args()

    print("📡 Fetching file list from Mendeley API...")
    files = get_file_list()
    print(f"📁 File totali: {len(files)}")

    if args.max_files:
        files = files[:args.max_files]
        print(f"  (limitato a {args.max_files} file per test)")

    folder_classes = map_folder_to_class(files)
    print(f"\n📂 Classi trovate:")
    for fid, cls in sorted(folder_classes.items(), key=lambda x: x[1]):
        count = sum(1 for f in files if f.get("folder_id") == fid)
        print(f"  {cls}: {count} file")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Group files by class for organized download
    class_files = {}
    for f in files:
        fid = f.get("folder_id", "unknown")
        cls = folder_classes.get(fid, "unknown")
        class_files.setdefault(cls, []).append(f)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for cls, cls_file_list in sorted(class_files.items()):
        dest_dir = RAW_DIR / cls
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Count existing files
        existing = len(list(dest_dir.glob("*.jpg")))
        if existing >= len(cls_file_list):
            print(f"\n✅ {cls}: già completo ({existing}/{len(cls_file_list)})")
            total_skipped += len(cls_file_list)
            continue

        print(f"\n⬇️  {cls}: scarico {len(cls_file_list)} file in {dest_dir} ({existing} già presenti)...")

        batch_downloaded = 0
        batch_skipped = 0
        batch_failed = 0

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(download_file, f, dest_dir): f for f in cls_file_list}

            for i, future in enumerate(as_completed(futures), 1):
                filename, success, error = future.result()
                if success and error is None:
                    batch_downloaded += 1
                elif success and error == "already exists":
                    batch_skipped += 1
                else:
                    batch_failed += 1
                    print(f"  ❌ {filename}: {error}")

                if i % 20 == 0 or i == len(cls_file_list):
                    print(f"  {i}/{len(cls_file_list)} files...")

        print(f"  ✅ {cls}: {batch_downloaded} nuovi, {batch_skipped} già presenti, {batch_failed} falliti")
        total_downloaded += batch_downloaded
        total_skipped += batch_skipped
        total_failed += batch_failed

        if args.delay and cls != sorted(class_files.keys())[-1]:
            time.sleep(args.delay)

    print(f"\n{'='*50}")
    print(f"📊 Riepilogo:")
    print(f"  Scaricati: {total_downloaded}")
    print(f"  Già presenti: {total_skipped}")
    print(f"  Falliti: {total_failed}")
    print(f"  Destinazione: {RAW_DIR}")
    print(f"{'='*50}")

    # Verify structure
    print("\n📂 Struttura finale:")
    for cls_dir in sorted(RAW_DIR.iterdir()):
        if cls_dir.is_dir():
            n_files = len(list(cls_dir.glob("*.jpg")))
            print(f"  {cls_dir.name}/: {n_files} file")


if __name__ == "__main__":
    main()
