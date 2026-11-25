"""Build the synthetic 'store world' around the real ABO catalog.

ABO provides real products (title, brand, model, images) but no prices, sellers
or policies. This script generates, deterministically (seed 42):

  * offers.jsonl      3-5 SYNTHETIC seller offers per product (price, shipping,
                      condition) -> used by price-comparison tasks
  * rules.json        SYNTHETIC store policy rules of a fictional marketplace
                      ("Harbor Market") -> single source of truth for after-sales
                      and authentication ground truth
  * corpus_train.jsonl  knowledge base for generating TRAINING trajectories
                      (train products only + policies)
  * corpus_full.jsonl   knowledge base for EVALUATION (all products + policies)

Everything price/seller/policy related is synthetic and must be described as
such in the README. Product names, brands, model numbers and images are real
ABO data (CC BY 4.0).

Usage:
    python build_world.py --catalog /d/projects/data/catalog --out /d/projects/data/world
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List

SEED = 42

# ---------------------------------------------------------------------------
# Sellers (fictional)
# ---------------------------------------------------------------------------
SELLERS = [
    # name, flat shipping, free-shipping threshold
    ("BrightCart", 4.99, 35.0),
    ("Northline Outlet", 5.99, 50.0),
    ("Maple Supply Co", 3.99, 25.0),
    ("ValueNest", 6.99, 75.0),
    ("Cedar Street Goods", 4.49, 40.0),
    ("PrimePick Direct", 0.00, 0.0),
    ("Harbor Trading", 5.49, 60.0),
    ("Greenfield Mart", 3.49, 30.0),
    ("Urban Bazaar", 7.99, 100.0),
    ("Lakeside Wholesale", 4.99, 45.0),
    ("Summit Deals", 5.99, 55.0),
    ("Willow Market", 2.99, 20.0),
]

# (low, high) base price in USD, sampled log-uniformly
PRICE_RANGE = {
    "SHOES": (25, 150), "BOOT": (40, 200), "SANDAL": (15, 90), "HANDBAG": (30, 400),
    "HAT": (8, 60), "FINERING": (30, 600), "FINENECKLACEBRACELETANKLET": (25, 500),
    "FINEEARRING": (15, 300), "CHAIR": (60, 500), "SOFA": (300, 1800), "TABLE": (80, 900),
    "LAMP": (20, 250), "RUG": (25, 600), "OFFICE_PRODUCTS": (8, 150),
    "SPORTING_GOODS": (10, 200), "SUITCASE": (40, 350), "KITCHEN": (10, 150),
    "CELLULAR_PHONE_CASE": (6, 40),
}

# ---------------------------------------------------------------------------
# Policy rules (fictional marketplace "Harbor Market")
# ---------------------------------------------------------------------------
STORE = "Harbor Market"

GROUPS = {
    "footwear": {
        "types": ["SHOES", "BOOT", "SANDAL"],
        "label": "Footwear",
        "return_days": 30, "restocking_pct": 0, "returnable_if_worn": False,
        "condition_text": "unworn, with the original box and tags",
        "final_sale": False,
        "auth_where": "the insole or inner tongue label",
    },
    "bags_hats": {
        "types": ["HANDBAG", "HAT", "SUITCASE"],
        "label": "Bags, hats and luggage",
        "return_days": 30, "restocking_pct": 0, "returnable_if_worn": False,
        "condition_text": "unused, with all tags attached",
        "final_sale": False,
        "auth_where": "the interior brand tag",
    },
    "rings_necklaces": {
        "types": ["FINERING", "FINENECKLACEBRACELETANKLET"],
        "label": "Rings, necklaces and bracelets",
        "return_days": 14, "restocking_pct": 0, "returnable_if_worn": False,
        "condition_text": "unworn, in the original pouch or box",
        "final_sale": False,
        "auth_where": "the box label or hallmark card",
    },
    "earrings": {
        "types": ["FINEEARRING"],
        "label": "Earrings",
        "return_days": 0, "restocking_pct": 0, "returnable_if_worn": False,
        "condition_text": "not returnable for change of mind (hygiene); defective or wrong items only",
        "final_sale": True,
        "auth_where": "the box label or hallmark card",
    },
    "furniture": {
        "types": ["CHAIR", "SOFA", "TABLE"],
        "label": "Furniture",
        "return_days": 45, "restocking_pct": 15, "returnable_if_worn": False,
        "condition_text": "unassembled and in the original packaging; assembled items are returnable only if defective",
        "final_sale": False,
        "auth_where": "the manufacturer label under the seat or tabletop",
    },
    "home": {
        "types": ["LAMP", "RUG", "KITCHEN"],
        "label": "Home and kitchen",
        "return_days": 30, "restocking_pct": 0, "returnable_if_worn": False,
        "condition_text": "unused and in the original packaging",
        "final_sale": False,
        "auth_where": "the product label on the base or packaging",
    },
    "office_sports": {
        "types": ["OFFICE_PRODUCTS", "SPORTING_GOODS"],
        "label": "Office and sporting goods",
        "return_days": 30, "restocking_pct": 0, "returnable_if_worn": False,
        "condition_text": "unused and in the original packaging",
        "final_sale": False,
        "auth_where": "the product label on the packaging",
    },
    "phone_cases": {
        "types": ["CELLULAR_PHONE_CASE"],
        "label": "Phone cases",
        "return_days": 30, "restocking_pct": 0, "returnable_if_worn": False,
        "condition_text": "unused and in the original packaging",
        "final_sale": False,
        "auth_where": "the packaging label",
    },
}

GLOBAL_RULES = {
    "damaged_report_hours": 48,
    "wrong_item_days": 7,
    "refund_business_days": "5-7",
    "preowned_return_days": 7,
    "return_shipping": "The buyer pays return shipping unless the item arrived damaged, defective or wrong.",
}

TYPE_TO_GROUP = {t: g for g, spec in GROUPS.items() for t in spec["types"]}


# ---------------------------------------------------------------------------
# Offers
# ---------------------------------------------------------------------------
def make_offers(product: Dict[str, Any]) -> List[Dict[str, Any]]:
    rng = random.Random(f"{SEED}-{product['item_id']}")
    lo, hi = PRICE_RANGE[product["product_type"]]
    base = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    sellers = rng.sample(SELLERS, rng.randint(3, 5))
    offers = []
    for name, flat, threshold in sellers:
        condition = "pre-owned" if rng.random() < 0.3 else "new"
        factor = rng.uniform(0.45, 0.75) if condition == "pre-owned" else rng.uniform(0.9, 1.15)
        price = round(base * factor, 2)
        price = math.floor(price) + rng.choice([0.0, 0.49, 0.95, 0.99]) if price >= 10 else price
        ship = 0.0 if price >= threshold else flat
        offers.append({
            "seller": name,
            "condition": condition,
            "price": round(price, 2),
            "shipping": round(ship, 2),
            "total": round(price + ship, 2),
        })
    # make all totals distinct so "cheapest" always has one answer
    seen = set()
    for o in sorted(offers, key=lambda x: x["total"]):
        while o["total"] in seen:
            o["price"] = round(o["price"] + 0.01, 2)
            o["total"] = round(o["price"] + o["shipping"], 2)
        seen.add(o["total"])
    return offers


# ---------------------------------------------------------------------------
# Knowledge base text
# ---------------------------------------------------------------------------
def money(x: float) -> str:
    return f"${x:,.2f}"


def product_entry(p: Dict[str, Any], offers: List[Dict[str, Any]]) -> Dict[str, str]:
    model = " / ".join(x for x in [p.get("model_name"), p.get("model_number")] if x) or "not listed"
    head = (
        f"Product ID: {p['item_id']} | Type: {p['product_type']} | Brand: {p.get('brand') or 'not listed'} | "
        f"Model: {model} | Color: {p.get('color') or 'not listed'}"
    )
    lines = ["Offers (price + shipping = total):"]
    for o in offers:
        lines.append(
            f"- {o['seller']} ({o['condition']}): {money(o['price'])} + {money(o['shipping'])} = {money(o['total'])}"
        )
    feats = "; ".join(p.get("bullet_points", [])[:3])
    text = head + "\n" + "\n".join(lines) + (f"\nFeatures: {feats}" if feats else "")
    return {"id": f"prod-{p['item_id']}", "type": "product", "title": p["title"][:160], "text": text}


def policy_entries() -> List[Dict[str, str]]:
    g = GLOBAL_RULES
    out: List[Dict[str, str]] = []
    for key, spec in GROUPS.items():
        types = ", ".join(spec["types"])
        if spec["final_sale"]:
            body = (
                f"{STORE} return policy - {spec['label']} (product types: {types}). "
                f"These items are final sale: they cannot be returned for change of mind. "
                f"They can be returned or replaced only if they arrive damaged, defective, or are the wrong item "
                f"(see 'Damaged or wrong items')."
            )
        else:
            fee = (
                f" A {spec['restocking_pct']}% restocking fee is deducted from the refund unless the item is defective."
                if spec["restocking_pct"] else " No restocking fee applies."
            )
            body = (
                f"{STORE} return policy - {spec['label']} (product types: {types}). "
                f"Items can be returned within {spec['return_days']} days of delivery for a refund if they are "
                f"{spec['condition_text']}.{fee} Items that do not meet the condition requirement are not eligible "
                f"for a change-of-mind return."
            )
        out.append({"id": f"policy-return-{key}", "type": "policy",
                    "title": f"{STORE} Return Policy: {spec['label']}", "text": body})
    out.append({
        "id": "policy-damaged", "type": "policy", "title": f"{STORE} Policy: Damaged or wrong items",
        "text": (f"If an item arrives damaged or defective, report it within {g['damaged_report_hours']} hours of delivery "
                 f"with photos. A confirmed damaged or defective item is eligible for a replacement or full refund even "
                 f"when the normal return window or condition rules would not allow a return, and return shipping is free. "
                 f"If you received the wrong item, report it within {g['wrong_item_days']} days of delivery for a full refund "
                 f"or the correct item. Reports made after these deadlines follow the normal return policy for the category."),
    })
    out.append({
        "id": "policy-refunds", "type": "policy", "title": f"{STORE} Policy: Refunds and return shipping",
        "text": (f"Approved refunds are issued to the original payment method within {g['refund_business_days']} business days "
                 f"after the returned item is received. {g['return_shipping']} Shipping fees paid at purchase are refunded "
                 f"only when the return is due to a damaged, defective or wrong item."),
    })
    out.append({
        "id": "policy-preowned", "type": "policy", "title": f"{STORE} Policy: Pre-owned items",
        "text": (f"Items listed as 'pre-owned' can be returned within {g['preowned_return_days']} days of delivery, and only if "
                 f"they are materially different from the listing description. Pre-owned items are not eligible for "
                 f"change-of-mind returns, and the category return windows do not apply to them."),
    })
    for key, spec in GROUPS.items():
        types = ", ".join(spec["types"])
        out.append({
            "id": f"policy-auth-{key}", "type": "policy",
            "title": f"{STORE} Authentication Standard: {spec['label']}",
            "text": (f"Authentication check for second-hand listings (product types: {types}). Read {spec['auth_where']} and "
                     f"compare it with the catalog entry. Authentic: the brand is spelled exactly as in the catalog AND the model "
                     f"number is identical to the catalog model number. Likely counterfeit: the brand spelling differs or the model "
                     f"number does not match the catalog. Cannot verify: the photo does not show the brand and model number clearly "
                     f"enough to read, or the catalog entry has no model number. Always look up the catalog entry first; "
                     f"never judge from appearance alone."),
        })
    return out


# ---------------------------------------------------------------------------
def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    products = [json.loads(l) for l in open(os.path.join(args.catalog, "products.jsonl"), encoding="utf-8")]
    os.makedirs(args.out, exist_ok=True)

    offers_rows, prod_entries = [], {}
    for p in products:
        offers = make_offers(p)
        offers_rows.append({"item_id": p["item_id"], "offers": offers})
        prod_entries[p["item_id"]] = product_entry(p, offers)

    policies = policy_entries()
    train = [prod_entries[p["item_id"]] for p in products if p["split"] == "train"] + policies
    full = [prod_entries[p["item_id"]] for p in products] + policies

    write_jsonl(os.path.join(args.out, "offers.jsonl"), offers_rows)
    write_jsonl(os.path.join(args.out, "corpus_train.jsonl"), train)
    write_jsonl(os.path.join(args.out, "corpus_full.jsonl"), full)
    with open(os.path.join(args.out, "rules.json"), "w", encoding="utf-8") as f:
        json.dump({"store": STORE, "seed": SEED, "groups": GROUPS, "global": GLOBAL_RULES,
                   "type_to_group": TYPE_TO_GROUP}, f, indent=2)

    print(f"products: {len(products)} | policies: {len(policies)}")
    print(f"corpus_train: {len(train)} entries | corpus_full: {len(full)} entries")
    print("wrote offers.jsonl, rules.json, corpus_train.jsonl, corpus_full.jsonl to", args.out)


if __name__ == "__main__":
    main()
