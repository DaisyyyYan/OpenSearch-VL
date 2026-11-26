"""Download the catalog images listed in products.jsonl from the public ABO bucket.

Only the selected images are fetched (a few thousand small JPEGs), not the 3 GB
archive. Files already present are skipped, so the script can be re-run safely.

Usage:
    python download_images.py --catalog /d/projects/data/catalog
Result: <catalog>/images/<path>   (same relative path as in ABO)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/"


def wanted_paths(products_path: str):
    seen = set()
    with open(products_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            for img in [p["main_image"]] + p.get("other_images", []):
                if img["path"] not in seen:
                    seen.add(img["path"])
                    yield img["path"]


def fetch(rel: str, out_root: str, session: requests.Session) -> str:
    dest = os.path.join(out_root, "images", rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for _ in range(3):
        try:
            r = session.get(BASE + rel, timeout=30)
            if r.status_code == 200 and r.content:
                with open(dest, "wb") as f:
                    f.write(r.content)
                return "ok"
            if r.status_code in (403, 404):
                return f"missing:{r.status_code}"
        except requests.RequestException:
            pass
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    paths = list(wanted_paths(os.path.join(args.catalog, "products.jsonl")))
    print(f"{len(paths)} images to check")
    counts = {}
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, p, args.catalog, session) for p in paths]
        for i, fu in enumerate(as_completed(futs), 1):
            key = fu.result().split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if i % 500 == 0:
                print(f"  {i}/{len(paths)} {counts}")
    print("done:", counts)
    if counts.get("fail") or counts.get("missing"):
        print("Some images were not downloaded; re-run to retry, or tell me the counts.", file=sys.stderr)


if __name__ == "__main__":
    main()
