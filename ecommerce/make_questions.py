"""Generate the question sets (train / test) for the three e-commerce tasks.

Tasks
  price_compare : photo of a product -> cheapest seller and total price
  authenticate  : photo of a second-hand item label -> authentic / likely
                  counterfeit / cannot verify
  after_sales   : screenshot of an order + customer message -> eligible /
                  not eligible / cannot determine

Ground truth is computed from the catalog, the synthetic offers and
rules.json - never by an LLM.

Train questions use ONLY train products, test questions ONLY test products
(the split was made by product in prepare_catalog.py).

Product names, brands, model numbers and images come from ABO (CC BY 4.0).
Label / order-screenshot images are rendered here and are synthetic.

Usage:
    python make_questions.py --catalog /d/projects/data/catalog \
        --world /d/projects/data/world --out /d/projects/data/questions
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SEED = 42

# ---------------------------------------------------------------------------
# Ground-truth logic (shared with the evaluation code)
# ---------------------------------------------------------------------------


def normal_return(spec: Dict[str, Any], days: int, condition: str) -> str:
    """Change-of-mind return under the category policy."""
    if spec["final_sale"]:
        return "not eligible"
    if days > spec["return_days"]:
        return "not eligible"
    if condition != "unused":
        return "not eligible"
    return "eligible"


def decide_after_sales(rules: Dict[str, Any], group: str, preowned: bool, issue: str,
                       days: Optional[int], hours: Optional[int], condition: str) -> str:
    """days is None when the item has not been delivered yet."""
    if days is None:
        return "cannot determine"
    spec = rules["groups"][group]
    g = rules["global"]
    if preowned:
        if issue == "not_as_described":
            return "eligible" if days <= g["preowned_return_days"] else "not eligible"
        return "not eligible"
    if issue == "damaged":
        if hours is not None and hours <= g["damaged_report_hours"]:
            return "eligible"
        return normal_return(spec, days, "unused")
    if issue == "wrong_item":
        if days <= g["wrong_item_days"]:
            return "eligible"
        return normal_return(spec, days, "unused")
    return normal_return(spec, days, condition)


# ---------------------------------------------------------------------------
# Fonts and image helpers
# ---------------------------------------------------------------------------
_FONT_CACHE: Dict[Tuple[int, bool], ImageFont.ImageFont] = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"])
    f = None
    for n in names:
        try:
            f = ImageFont.truetype(n, size)
            break
        except OSError:
            continue
    if f is None:
        try:
            f = ImageFont.load_default(size=size)
        except TypeError:
            f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def photo_effect(img: Image.Image, rng: random.Random, rot: float = 5.0, blur=(0.6, 1.4),
                 noise: float = 4.0, bg=(150, 140, 125)) -> Image.Image:
    """Make a flat rendering look like a hand-held phone photo (tilt, perspective, blur, noise)."""
    w, h = img.size
    pad = int(0.12 * max(w, h))
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), bg)
    canvas.paste(img.convert("RGB"), (pad, pad))
    cw, ch = canvas.size
    j = 0.04
    src = [(0, 0), (cw, 0), (cw, ch), (0, ch)]
    dst = [(rng.uniform(-j, j) * cw + x, rng.uniform(-j, j) * ch + y) for x, y in src]
    coeffs = _perspective_coeffs(dst, src)
    out = canvas.transform((cw, ch), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=bg)
    out = out.rotate(rng.uniform(-rot, rot), resample=Image.BICUBIC, fillcolor=bg)
    out = out.filter(ImageFilter.GaussianBlur(rng.uniform(*blur)))
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.1))
    arr = np.asarray(out).astype(np.float32)
    arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, noise, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _perspective_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(pb).reshape(8)
    return np.linalg.solve(a, b).tolist()


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_label(brand: str, model: str, extra: Dict[str, str], hide_model: bool, rng: random.Random) -> Image.Image:
    w, h = 640, 380
    img = Image.new("RGB", (w, h), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([6, 6, w - 7, h - 7], outline=(40, 40, 40), width=3)
    d.text((28, 22), brand.upper(), fill=(20, 20, 20), font=font(44, True))
    y = 100
    model_box = None
    d.text((28, y), "Model No.:", fill=(30, 30, 30), font=font(30))
    mx = 28 + int(d.textlength("Model No.: ", font=font(30)))
    d.text((mx, y), model, fill=(10, 10, 10), font=font(32, True))
    model_box = (mx - 4, y - 4, mx + int(d.textlength(model, font=font(32, True))) + 10, y + 42)
    y += 62
    for k, v in extra.items():
        d.text((28, y), f"{k}: {v}", fill=(30, 30, 30), font=font(26))
        y += 44
    x = 28
    while x < w - 40:  # decorative barcode
        bw = rng.choice([2, 3, 5])
        d.rectangle([x, h - 78, x + bw, h - 24], fill=(15, 15, 15))
        x += bw + rng.choice([2, 3, 4])
    if hide_model:
        d.rectangle(model_box, fill=(0, 0, 0))
    return img


def wrap(text: str, width: int) -> List[str]:
    return textwrap.wrap(text, width=width)[:3]


def render_order(order: Dict[str, Any]) -> Image.Image:
    w, h = 900, 640
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 78], fill=(22, 62, 98))
    d.text((30, 20), "Harbor Market  |  Order Details", fill=(255, 255, 255), font=font(34, True))
    y = 108
    d.text((30, y), f"Order #{order['order_id']}", fill=(20, 20, 20), font=font(30, True))
    y += 58
    for line in wrap(order["title"], 46):
        d.text((30, y), line, fill=(20, 20, 20), font=font(28))
        y += 38
    y += 16
    rows = [("Sold by", order["seller"]), ("Condition", order["condition"]),
            ("Item price", f"${order['price']:.2f}"), ("Order date", order["order_date"]),
            ("Status", order["status"])]
    for k, v in rows:
        d.text((30, y), f"{k}:", fill=(90, 90, 90), font=font(27))
        d.text((250, y), v, fill=(15, 15, 15), font=font(27, True))
        y += 44
    return img


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(catalog: str, world: str):
    products = [json.loads(l) for l in open(os.path.join(catalog, "products.jsonl"), encoding="utf-8")]
    offers = {}
    for l in open(os.path.join(world, "offers.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        offers[r["item_id"]] = r["offers"]
    rules = json.load(open(os.path.join(world, "rules.json"), encoding="utf-8"))
    return products, offers, rules


def fmt_date(d: dt.date) -> str:
    return d.strftime("%b %d, %Y").replace(" 0", " ")


# ---------------------------------------------------------------------------
# Task builders. Each returns (row, image) or None.
# ---------------------------------------------------------------------------


def build_price(p, offers, rng):
    imgs = [p["main_image"]] + p.get("other_images", [])
    src = rng.choice(imgs)
    path = os.path.join(CATALOG, "images", src["path"])
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    w, h = im.size
    c = rng.uniform(0.85, 1.0)
    cw, ch = int(w * c), int(h * c)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = im.crop((x0, y0, x0 + cw, y0 + ch)).rotate(rng.uniform(-4, 4), expand=True, fillcolor=(255, 255, 255))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.12))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.9, 1.1))
    scale = 384 / max(im.size)
    im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)), Image.BICUBIC)

    only_new = rng.random() < 0.6
    pool = [o for o in offers if o["condition"] == "new"] if only_new else list(offers)
    if len(pool) < 2:
        only_new, pool = False, list(offers)
    best = min(pool, key=lambda o: o["total"])
    q = rng.choice([
        "A customer sent this photo and asks: \"What is the lowest price I can pay for this item, shipping included?\"",
        "The customer attached this photo and wants to know where to buy this exact item cheapest, shipping included.",
        "A customer shared this picture: \"Which seller has this for the best total price?\"",
    ])
    q += (" Consider new items only." if only_new else " Consider all listed offers, including pre-owned.")
    q += " Answer with the seller name and the total price (price + shipping)."
    answer = f"{best['seller']}, ${best['total']:.2f}"
    return {"task": "price_compare", "question": q, "answer": answer,
            "meta": {"item_id": p["item_id"], "only_new": only_new, "seller": best["seller"], "total": best["total"]}}, im


def mutate_brand(brand: str, rng: random.Random) -> str:
    letters = [i for i, ch in enumerate(brand) if ch.isalpha()]
    for _ in range(20):
        i = rng.choice(letters)
        mode = rng.choice(["swap", "double", "replace"])
        s = list(brand)
        if mode == "swap" and i + 1 < len(s) and s[i + 1].isalpha() and s[i] != s[i + 1]:
            s[i], s[i + 1] = s[i + 1], s[i]
        elif mode == "double":
            s.insert(i, s[i])
        else:
            s[i] = rng.choice([c for c in "aeioulnrst" if c != s[i].lower()])
        out = "".join(s)
        if out.lower() != brand.lower():
            return out
    return brand + "e"


def mutate_model(model: str, rng: random.Random) -> str:
    idx = [i for i, ch in enumerate(model) if ch.isalnum()]
    for _ in range(20):
        i = rng.choice(idx)
        s = list(model)
        s[i] = rng.choice("0123456789") if s[i].isdigit() else rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ")
        out = "".join(s)
        if out != model:
            return out
    return model + "1"


def build_auth(p, offers, rng, target: str, no_model_pool):
    brand = (p.get("brand") or "").strip()
    model = (p.get("model_number") or "").strip()
    if not brand or len(brand) > 28:
        return None
    extra_pool = [("Size", rng.choice(["S", "M", "L", "38", "40", "7", "9", "One Size"])),
                  ("Made in", rng.choice(["Vietnam", "China", "India", "Turkey"])),
                  ("Color", p.get("color") or "Assorted")]
    extra = dict(extra_pool[: rng.randint(2, 3)])
    kind = None
    if target == "authentic":
        if not model or len(model) > 22:
            return None
        shown_brand, shown_model, hide = brand, model, False
    elif target == "likely counterfeit":
        if not model or len(model) > 22:
            return None
        kind = rng.choice(["brand", "model"])
        shown_brand = mutate_brand(brand, rng) if kind == "brand" else brand
        shown_model = mutate_model(model, rng) if kind == "model" else model
        hide = False
    else:  # cannot verify
        if model and rng.random() < 0.6:
            kind, shown_brand, shown_model, hide = "occluded", brand, model, True
        elif not model:
            kind, shown_brand, shown_model, hide = "no_catalog_model", brand, "".join(
                rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(8)), False
        else:
            return None
    label = render_label(shown_brand, shown_model, extra, hide, rng)
    im = photo_effect(label, rng, rot=6, blur=(0.5, 1.1), noise=3.5)
    q = rng.choice([
        f"A buyer is considering a second-hand listing titled \"{p['title']}\". The seller sent this photo of the item's label. "
        "Is the item authentic?",
        f"Second-hand listing: \"{p['title']}\". Here is the label photo provided by the seller. "
        "Can you confirm whether this item is authentic?",
    ])
    q += " Answer with one of: authentic, likely counterfeit, cannot verify."
    return {"task": "authenticate", "question": q, "answer": target,
            "meta": {"item_id": p["item_id"], "variant": kind or "match", "shown_brand": shown_brand,
                     "shown_model": shown_model, "catalog_model": model}}, im


def build_after_sales(p, offers, rules, rng, target: str):
    group = rules["type_to_group"][p["product_type"]]
    spec = rules["groups"][group]
    offer = rng.choice(offers)
    preowned = offer["condition"] == "pre-owned"
    today = dt.date(2026, 12, 1) + dt.timedelta(days=rng.randint(0, 60))

    for _ in range(80):
        if target == "cannot determine":
            issue, condition, days, hours = rng.choice(["change_of_mind", "damaged", "wrong_item"]), "unused", None, None
            if preowned:
                issue = "not_as_described"
        else:
            if preowned:
                issue = rng.choice(["change_of_mind", "not_as_described", "not_as_described"])
            else:
                issue = rng.choice(["change_of_mind"] * 3 + ["damaged", "wrong_item"])
            condition = "unused"
            if issue == "change_of_mind":
                condition = rng.choice(["unused", "unused", "used"] + (["assembled"] if group == "furniture" else []))
            hours = None
            win = max(spec["return_days"], 7)
            days = rng.choice([rng.randint(1, max(1, min(win - 2, 6)) if win > 3 else 3),
                              rng.randint(1, max(1, win - 2)), rng.randint(win + 2, win + 25)])
            if issue == "damaged":
                hours = rng.choice([rng.randint(2, 44), rng.randint(52, 240)])
                days = hours // 24
                if days < 1:
                    days = 0
        out = decide_after_sales(rules, group, preowned, issue, days, hours, condition)
        if out == target:
            break
    else:
        return None

    if days is None:
        order_date = today - dt.timedelta(days=rng.randint(2, 4))
        status = f"Shipped - arriving {fmt_date(today + dt.timedelta(days=rng.randint(1, 3)))}"
    else:
        deliver = today - dt.timedelta(days=days)
        order_date = deliver - dt.timedelta(days=rng.randint(2, 6))
        status = f"Delivered on {fmt_date(deliver)}"
    order = {"order_id": f"HM-{rng.randint(100000, 999999)}", "title": p["title"], "seller": offer["seller"],
             "condition": "Pre-owned" if preowned else "New", "price": offer["price"],
             "order_date": fmt_date(order_date), "status": status}
    im = photo_effect(render_order(order), rng, rot=4, blur=(0.7, 1.5), noise=4.0, bg=(170, 165, 155))

    msgs = {
        "change_of_mind": {"unused": ["I changed my mind and want to return this. I haven't used it.",
                                      "I'd like to send this back, it is still unused."],
                           "used": ["I want to return this. I wore/used it a few times but it isn't what I wanted."],
                           "assembled": ["I want to return this. I already assembled it but I don't like it."]},
        "damaged": ["The item arrived damaged and I'd like a refund."],
        "wrong_item": ["I received the wrong item and want to return it."],
        "not_as_described": ["The item is not as described in the listing and I want to return it."],
    }
    m = msgs[issue][condition] if issue == "change_of_mind" else msgs[issue]
    msg = rng.choice(m)
    if issue == "damaged" and hours is not None:
        msg += f" I am reporting this about {hours} hours after delivery."
    if days is None:
        msg = rng.choice(["I want to return this order.", "Please start a return for this item."])
    q = (f"Customer message: \"{msg}\" Today's date is {today.isoformat()}. "
         "Using the order screenshot and the store policy, is this request eligible for a return/refund? "
         "Answer with one of: eligible, not eligible, cannot determine.")
    return {"task": "after_sales", "question": q, "answer": target,
            "meta": {"item_id": p["item_id"], "group": group, "preowned": preowned, "issue": issue,
                     "condition": condition, "days_since_delivery": days, "hours": hours,
                     "today": today.isoformat()}}, im


# ---------------------------------------------------------------------------
CATALOG = ""


def split_targets(n: int, weights: Dict[str, float], rng: random.Random) -> List[str]:
    labels = []
    for k, wgt in weights.items():
        labels += [k] * int(round(n * wgt))
    while len(labels) < n:
        labels.append(rng.choice(list(weights)))
    labels = labels[:n]
    rng.shuffle(labels)
    return labels


def generate(split: str, n_per_task: int, products, offers, rules, out_dir: str, rng: random.Random):
    pool = [p for p in products if p["split"] == split]
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    counter = {"price_compare": 0, "authenticate": 0, "after_sales": 0}

    def add(res, task):
        if res is None:
            return False
        row, im = res
        counter[task] += 1
        qid = f"{task}-{split}-{counter[task]:04d}"
        path = os.path.abspath(os.path.join(img_dir, f"{qid}.png"))
        im.save(path)
        row.update({"id": qid, "split": split, "images": [path]})
        rows.append(row)
        return True

    plans = {
        "price_compare": [None] * n_per_task,
        "authenticate": split_targets(n_per_task, {"authentic": 0.40, "likely counterfeit": 0.35, "cannot verify": 0.25}, rng),
        "after_sales": split_targets(n_per_task, {"eligible": 0.40, "not eligible": 0.45, "cannot determine": 0.15}, rng),
    }
    for task, targets in plans.items():
        for target in targets:
            for _ in range(200):
                p = rng.choice(pool)
                offs = offers[p["item_id"]]
                if task == "price_compare":
                    res = build_price(p, offs, rng)
                elif task == "authenticate":
                    res = build_auth(p, offs, rng, target, None)
                else:
                    res = build_after_sales(p, offs, rules, rng, target)
                if add(res, task):
                    break
    rng.shuffle(rows)
    path = os.path.join(out_dir, f"{split}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{split}: {len(rows)} questions -> {path}  {counter}")


def main():
    global CATALOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=300, help="questions per task")
    ap.add_argument("--n-test", type=int, default=60, help="questions per task")
    args = ap.parse_args()
    CATALOG = args.catalog
    products, offers, rules = load_data(args.catalog, args.world)
    os.makedirs(args.out, exist_ok=True)
    generate("train", args.n_train, products, offers, rules, args.out, random.Random(SEED))
    generate("test", args.n_test, products, offers, rules, args.out, random.Random(SEED + 1))


if __name__ == "__main__":
    main()
