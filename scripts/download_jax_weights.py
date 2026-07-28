"""Download pi05_base JAX weights from GCS using requests (with proxy support)."""

import os
from pathlib import Path

import requests
from tqdm import tqdm

GCS_BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_base/params/"
LOCAL_DIR = Path("./checkpoints/pi05_base_jax/params")


def list_files():
    url = f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o"
    params = {"prefix": PREFIX, "fields": "items(name,size),nextPageToken", "maxResults": 1000}
    items = []
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items.extend(data.get("items", []))
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return items


def download_file(name, size, pbar):
    dl_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{name}"
    rel_path = name[len(PREFIX):]
    dest = LOCAL_DIR / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == size:
        pbar.update(size)
        return

    r = requests.get(dl_url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print("Listing files...")
    items = list_files()
    total_size = sum(int(i["size"]) for i in items)
    print(f"Found {len(items)} files, total {total_size / 1e9:.2f} GB")

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for item in items:
            download_file(item["name"], int(item["size"]), pbar)

    print(f"\nDone! Weights saved to {LOCAL_DIR}")


if __name__ == "__main__":
    main()
