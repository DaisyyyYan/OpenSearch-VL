"""Pack everything needed to EVALUATE on the rented GPU machine (test split only, no train data).

Creates <out>.tar.gz containing, under data/:
    questions/test.jsonl      image paths rewritten to <remote_root>/questions/images/<name>
    questions/images/         only the 180 test images
    world/rules.json, world/corpus_full.jsonl, world/offers.jsonl (if present)
    clip_index/               CLIP embeddings + metadata (full scope)

    python ecommerce/make_eval_package.py --data-root D:/projects/data --out D:/projects/data/eval_data.tar.gz
Upload it to the machine, then:  mkdir -p /workspace && tar -xzf eval_data.tar.gz -C /workspace
(the archive unpacks to /workspace/data, matching the default --remote-root).
"""

import argparse
import json
import os
import tarfile
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="D:/projects/data")
    ap.add_argument("--out", default="D:/projects/data/eval_data.tar.gz")
    ap.add_argument("--remote-root", default="/workspace/data")
    args = ap.parse_args()

    qdir = os.path.join(args.data_root, "questions")
    rows = [json.loads(l) for l in open(os.path.join(qdir, "test.jsonl"), encoding="utf-8")]
    tmp = tempfile.mkdtemp(prefix="evalpkg_")
    test_jsonl = os.path.join(tmp, "test.jsonl")
    with open(test_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            r["images"] = [f"{args.remote_root}/questions/images/{os.path.basename(p)}" for p in r["images"]]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with tarfile.open(args.out, "w:gz") as tar:
        tar.add(test_jsonl, arcname="data/questions/test.jsonl")
        n_img = 0
        for r in json.loads(json.dumps(rows)):
            pass
        for line in open(os.path.join(qdir, "test.jsonl"), encoding="utf-8"):
            for p in json.loads(line)["images"]:
                tar.add(p, arcname=f"data/questions/images/{os.path.basename(p)}")
                n_img += 1
        for name in ("rules.json", "corpus_full.jsonl", "offers.jsonl"):
            p = os.path.join(args.data_root, "world", name)
            if os.path.exists(p):
                tar.add(p, arcname=f"data/world/{name}")
            elif name != "offers.jsonl":
                raise SystemExit(f"missing {p}")
        cdir = os.path.join(args.data_root, "clip_index")
        for name in os.listdir(cdir):
            tar.add(os.path.join(cdir, name), arcname=f"data/clip_index/{name}")
    print(f"{len(rows)} test questions, {n_img} images -> {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
