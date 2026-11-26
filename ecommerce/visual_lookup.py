"""Offline image search over the product catalog (CLIP embeddings).

Replaces the external reverse-image-search service used by ``image_search``.

Build the index once (about 15k small images; CPU is fine, ~10-20 minutes):

    python -m pip install torch transformers pillow numpy
    python ecommerce/visual_lookup.py build --catalog D:/projects/data/catalog --out D:/projects/data/clip_index

Try a query:

    python ecommerce/visual_lookup.py query --index D:/projects/data/clip_index --scope full --image some.png

``scope``:
  train : only train products are searchable (used when generating TRAINING
          trajectories, so test products never appear in training data)
  full  : all products (used for evaluation)

The lookup returns ``[{"title", "source", "match_score"}]``: ``source`` is the
knowledge-base passage id (``prod-<item_id>``) and ``match_score`` the cosine
similarity, so the agent can tell a strong match from a weak one.

Limitation (state it in the README): query images are augmented copies of
catalog images, so this is a near-duplicate lookup, easier than matching a
real customer photo against a catalog.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

MODEL_NAME = "openai/clip-vit-base-patch32"
EMB_FILE = "clip_embeddings.npz"
META_FILE = "clip_meta.json"


class ClipEmbedder:
    """Wraps a Hugging Face CLIP model; ``embed(list_of_PIL) -> (N, D)`` L2-normalised."""

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = [im.convert("RGB") for im in images[i:i + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            if not torch.is_tensor(feats):  # newer transformers versions wrap the output
                feats = getattr(feats, "image_embeds", None) or getattr(feats, "pooler_output")
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


def build_index(catalog_dir: str, out_dir: str, embedder=None, limit: Optional[int] = None,
                log: Callable[[str], None] = print) -> int:
    products = [json.loads(l) for l in open(os.path.join(catalog_dir, "products.jsonl"), encoding="utf-8")]
    entries: List[Dict[str, str]] = []
    for p in products:
        for img in [p["main_image"]] + p.get("other_images", []):
            entries.append({"item_id": p["item_id"], "title": p["title"], "split": p["split"],
                            "path": os.path.join(catalog_dir, "images", img["path"])})
    if limit:
        entries = entries[:limit]
    embedder = embedder or ClipEmbedder()

    embs, kept = [], []
    chunk = 256
    for i in range(0, len(entries), chunk):
        part, images = [], []
        for e in entries[i:i + chunk]:
            try:
                with Image.open(e["path"]) as im:
                    images.append(im.convert("RGB"))
                part.append(e)
            except Exception:
                continue
        if images:
            embs.append(embedder.embed(images))
            kept.extend(part)
        log(f"  embedded {min(i + chunk, len(entries))}/{len(entries)}")
    matrix = np.concatenate(embs, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, EMB_FILE),
        emb=matrix,
        item_ids=np.array([e["item_id"] for e in kept]),
        splits=np.array([e["split"] for e in kept]),
    )
    titles = {e["item_id"]: e["title"] for e in kept}
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "count": len(kept), "titles": titles}, f)
    log(f"index built: {len(kept)} images, {len(titles)} products -> {out_dir}")
    return len(kept)


class CatalogLookup:
    """Callable used as ``visual_lookup(image_url=<local path>)`` by the image_search tool."""

    def __init__(self, index_dir: str, scope: str = "full", top_k: int = 3, embedder=None) -> None:
        if scope not in ("train", "full"):
            raise ValueError("scope must be 'train' or 'full'")
        data = np.load(os.path.join(index_dir, EMB_FILE))
        meta = json.load(open(os.path.join(index_dir, META_FILE), encoding="utf-8"))
        mask = (data["splits"] == "train") if scope == "train" else np.ones(len(data["splits"]), dtype=bool)
        self.emb = data["emb"][mask]
        self.item_ids = data["item_ids"][mask]
        self.titles: Dict[str, str] = meta["titles"]
        self.top_k = top_k
        self._embedder = embedder
        self._lazy_lock = None

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = ClipEmbedder()
        return self._embedder

    def __call__(self, image_url: Optional[str] = None, **_: Any):
        path = image_url or ""
        if not path or path.startswith(("http://", "https://")) or not os.path.exists(path):
            return {"error": f"image_search needs an existing local image; got {path!r}"}
        try:
            with Image.open(path) as im:
                query = self._get_embedder().embed([im.convert("RGB")])[0]
        except Exception as exc:
            return {"error": f"could not read image: {exc}"}
        sims = self.emb @ query
        best: Dict[str, float] = {}
        for idx in np.argsort(-sims)[: max(50, self.top_k * 10)]:
            iid = str(self.item_ids[idx])
            if iid not in best:
                best[iid] = float(sims[idx])
            if len(best) >= self.top_k:
                break
        return [{"title": self.titles[iid], "source": f"prod-{iid}", "match_score": round(s, 3)}
                for iid, s in best.items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--catalog", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--limit", type=int, default=None, help="only embed the first N images (smoke test)")
    q = sub.add_parser("query")
    q.add_argument("--index", required=True)
    q.add_argument("--scope", default="full", choices=["train", "full"])
    q.add_argument("--image", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build_index(args.catalog, args.out, limit=args.limit)
    else:
        print(json.dumps(CatalogLookup(args.index, args.scope)(image_url=args.image), indent=2))


if __name__ == "__main__":
    main()
