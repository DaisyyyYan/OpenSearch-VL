"""Convert clean trajectories to a LLaMA-Factory dataset (OpenAI-style messages + images).

Output folder (upload this whole folder to the GPU machine):
    <out>/images/<file>.png        only the images that are used
    <out>/ecom_sft_train.json
    <out>/ecom_sft_val.json        small held-out slice, used only to watch the loss
    <out>/dataset_info.json

Image paths in the json are relative to <out>, so the folder works anywhere.
Loss is computed on assistant turns only (LLaMA-Factory default); observations are user turns.

    python ecommerce/to_sft.py --inp D:/projects/data/traj/train_clean.jsonl --out D:/projects/data/sft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil


def is_val(qid: str, frac: float) -> bool:
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % 10000 < frac * 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.04)
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    train, val = [], []
    for line in open(args.inp, encoding="utf-8"):
        r = json.loads(line)
        src = r["images"][0]
        name = os.path.basename(src)
        dst = os.path.join(img_dir, name)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        first_user = r["messages"][1]["content"]
        if first_user.count("<image>") != 1:
            raise SystemExit(f"{r['id']}: expected exactly one <image> tag")
        row = {"messages": r["messages"], "images": [f"images/{name}"]}
        (val if is_val(r["id"], args.val_frac) else train).append(row)

    for name, rows in (("ecom_sft_train.json", train), ("ecom_sft_val.json", val)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
    cols = {"messages": "messages", "images": "images"}
    tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user",
            "assistant_tag": "assistant", "system_tag": "system"}
    info = {n: {"file_name": f, "formatting": "sharegpt", "columns": cols, "tags": tags}
            for n, f in (("ecom_sft_train", "ecom_sft_train.json"), ("ecom_sft_val", "ecom_sft_val.json"))}
    with open(os.path.join(args.out, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"train {len(train)} | val {len(val)} | images copied to {img_dir}")


if __name__ == "__main__":
    main()
