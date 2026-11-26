"""Build the product catalog used by the e-commerce customer-service project.

Input : ABO listings (listings/metadata/*.json.gz) + images/metadata/images.csv.gz
Output: <out>/products.jsonl   one product per line, with a train/test split

Selection (deterministic, seed=42):
  * English title and a main image of at least MIN_SIDE pixels
  * only the product types in TYPE_CAPS, at most `cap` products per type
  * duplicate titles removed
  * 80/20 train/test split BY PRODUCT (stratified by type), so no test product
    is ever seen during training

Data source: Amazon Berkeley Objects (ABO), CC BY 4.0
(Collins et al., "ABO: Dataset and Benchmarks for Real-World 3D Object
Understanding", CVPR 2022). ABO has no price field; prices/offers are
generated separately and are synthetic.

Usage:
    python prepare_catalog.py --abo /d/projects/data/abo --out /d/projects/data/catalog
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

SEED = 42
MIN_SIDE = 200
TEST_FRACTION = 0.2

TYPE_CAPS = {
    # second-hand authentication friendly (brand / model labels matter)
    "SHOES": 300, "BOOT": 300, "SANDAL": 300, "HANDBAG": 300, "HAT": 300,
    "FINERING": 300, "FINENECKLACEBRACELETANKLET": 300, "FINEEARRING": 300,
    # price comparison / after-sales
    "CHAIR": 300, "SOFA": 300, "TABLE": 300, "LAMP": 300, "RUG": 300,
    "OFFICE_PRODUCTS": 300, "SPORTING_GOODS": 300, "SUITCASE": 250, "KITCHEN": 300,
    # many near-identical items; capped on purpose
    "CELLULAR_PHONE_CASE": 150,
}


def en_value(entries: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Pick the English value (en_US first, then any en_*) from a language-tagged list."""
    if not entries:
        return None
    english = [e for e in entries if str(e.get("language_tag", "")).startswith("en")]
    if not english:
        return None
    for e in english:
        if e.get("language_tag") == "en_US":
            return str(e.get("value", "")).strip() or None
    return str(english[0].get("value", "")).strip() or None


def plain_value(entries: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Values without language tags (model_number, product_type, ...)."""
    if not entries:
        return None
    v = entries[0].get("value")
    return None if v is None else str(v).strip() or None


def load_image_index(path: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            index[row["image_id"]] = {
                "h": int(row["height"]),
                "w": int(row["width"]),
                "path": row["path"],
            }
    return index


def build(abo_root: str, out_dir: str) -> None:
    images = load_image_index(os.path.join(abo_root, "images.csv.gz"))
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_titles = set()

    for path in sorted(glob.glob(os.path.join(abo_root, "listings", "metadata", "*.json.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ptype = plain_value(r.get("product_type"))
                if ptype not in TYPE_CAPS:
                    continue
                title = en_value(r.get("item_name"))
                main_id = r.get("main_image_id")
                if not title or len(title) < 10 or not main_id:
                    continue
                img = images.get(main_id)
                if not img or max(img["h"], img["w"]) < MIN_SIDE:
                    continue
                key = (ptype, title.lower())
                if key in seen_titles:
                    continue
                seen_titles.add(key)

                other = []
                for oid in r.get("other_image_id", []) or []:
                    oi = images.get(oid)
                    if oi and max(oi["h"], oi["w"]) >= MIN_SIDE:
                        other.append({"image_id": oid, "path": oi["path"]})
                bullets = [
                    b["value"].strip() for b in (r.get("bullet_point") or [])
                    if str(b.get("language_tag", "")).startswith("en") and b.get("value")
                ]
                by_type[ptype].append({
                    "item_id": r["item_id"],
                    "product_type": ptype,
                    "title": title,
                    "brand": en_value(r.get("brand")),
                    "color": en_value(r.get("color")),
                    "model_name": en_value(r.get("model_name")),
                    "model_number": plain_value(r.get("model_number")),
                    "model_year": plain_value(r.get("model_year")),
                    "bullet_points": bullets[:6],
                    "main_image": {"image_id": main_id, "path": img["path"]},
                    "other_images": other[:3],
                })

    rng = random.Random(SEED)
    selected: List[Dict[str, Any]] = []
    for ptype in sorted(by_type):
        items = sorted(by_type[ptype], key=lambda x: x["item_id"])
        rng.shuffle(items)
        items = items[: TYPE_CAPS[ptype]]
        n_test = max(1, round(len(items) * TEST_FRACTION)) if len(items) >= 5 else 0
        for i, it in enumerate(items):
            it["split"] = "test" if i < n_test else "train"
        selected.extend(items)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "products.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for it in selected:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    n_test = sum(1 for x in selected if x["split"] == "test")
    print(f"wrote {len(selected)} products to {out_path} (train {len(selected) - n_test}, test {n_test})")
    for ptype in sorted(TYPE_CAPS):
        have = sum(1 for x in selected if x["product_type"] == ptype)
        print(f"  {ptype:28s} {have:4d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--abo", required=True, help="folder with listings/ and images.csv.gz")
    ap.add_argument("--out", required=True, help="output folder (outside the git repo)")
    args = ap.parse_args()
    build(args.abo, args.out)
