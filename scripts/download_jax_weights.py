"""Download pi05_base JAX weights from GCS using requests (with proxy support)."""

import argparse
import concurrent.futures
from pathlib import Path

import requests
from tqdm import tqdm

GCS_BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_base/params/"
LOCAL_DIR = Path("./checkpoints/pi05_base_jax/params")
DEFAULT_WORKERS = 8


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

    existing_size = dest.stat().st_size if dest.exists() else 0
    if existing_size == size:
        return
    if existing_size > size:
        dest.unlink()
        pbar.update(-size)
        existing_size = 0

    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    with requests.get(dl_url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if existing_size and response.status_code != 206:
            # The server ignored Range. Restart this file without counting the stale partial bytes.
            pbar.update(-existing_size)
            existing_size = 0
        mode = "ab" if existing_size else "wb"
        with dest.open(mode) as file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

    actual_size = dest.stat().st_size
    if actual_size != size:
        raise RuntimeError(f"Incomplete download for {name}: expected {size} bytes, got {actual_size}")


def main(workers: int = DEFAULT_WORKERS):
    if workers <= 0:
        raise ValueError("--workers must be positive")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print("Listing files...")
    items = list_files()
    total_size = sum(int(i["size"]) for i in items)
    print(f"Found {len(items)} files, total {total_size / 1e9:.2f} GB")

    downloaded_size = sum(
        min((LOCAL_DIR / item["name"][len(PREFIX):]).stat().st_size, int(item["size"]))
        for item in items
        if (LOCAL_DIR / item["name"][len(PREFIX):]).exists()
    )
    with (
        tqdm(total=total_size, initial=downloaded_size, unit="B", unit_scale=True, desc="Downloading") as pbar,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        futures = [
            executor.submit(download_file, item["name"], int(item["size"]), pbar)
            for item in items
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print(f"\nDone! Weights saved to {LOCAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    main(args.workers)
