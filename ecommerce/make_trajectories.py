"""Programmatic expert trajectories (no LLM involved).

For every question the script runs the REAL offline tools (local OCR, BM25
knowledge base, CLIP catalog lookup), parses what they return, applies the
store rules, and writes the conversation an ideal agent would have had:

    system -> user(images + <tools> + question)
           -> assistant(<think> + <tool_call>) -> user(<observation>) -> ...
           -> assistant(<think> + <response> ... \\boxed{verdict} </response>)

A trajectory is kept only if the verdict computed from the *observations*
equals the ground truth in the question file, so tool failures (OCR typos,
wrong CLIP match, KB miss) simply drop that question instead of producing a
wrong example. The rule-based "expert" is deterministic; the <think> text is
templated with a few phrasings (seeded) to avoid a single fixed sentence.

Run from the repository root:

    python ecommerce/make_trajectories.py --split train --limit 20 \
        --out D:/projects/data/traj/train_20.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


class Skip(Exception):
    """The tools did not give the evidence needed; drop this question."""


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------


def passages(obs: str) -> List[Dict[str, str]]:
    """Split a text_search observation into passages: id (URL line), title, text."""
    out = []
    for block in re.split(r"\[Passage \d+\]", obs)[1:]:
        m_title = re.search(r"Title:\s*(.*)", block)
        m_id = re.search(r"URL:\s*(\S+)", block)
        m_text = re.search(r"Summary:\s*(.*)", block, re.S)
        out.append({"id": m_id.group(1) if m_id else "", "title": m_title.group(1).strip() if m_title else "",
                    "text": (m_text.group(1) if m_text else "").strip()})
    return out


def find_passage(obs: str, pid: str) -> Dict[str, str]:
    for p in passages(obs):
        if p["id"] == pid:
            return p
    raise Skip(f"passage {pid} not in search result")


def short(text: str, n: int = 70) -> str:
    """Cut at a word boundary and mark the cut."""
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(",;:-") + "..."


def money(s: str) -> float:
    return float(s.replace("$", "").replace(",", ""))


def parse_product(text: str) -> Dict[str, Any]:
    m = re.search(r"Type:\s*(.*?)\s*\|\s*Brand:\s*(.*?)\s*\|\s*Model:\s*(.*?)\s*\|\s*Color:\s*(.*)", text)
    if not m:
        raise Skip("cannot parse product header")
    offers = []
    for seller, cond, price, ship, total in re.findall(
            r"^- (.+?) \((new|pre-owned)\): (\$[\d,\.]+) \+ (\$[\d,\.]+) = (\$[\d,\.]+)", text, re.M):
        offers.append({"seller": seller, "condition": cond, "price": money(price),
                       "shipping": money(ship), "total": money(total)})
    return {"type": m.group(1), "brand": m.group(2), "model": m.group(3), "color": m.group(4).split("\n")[0],
            "offers": offers}


def parse_order_ocr(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title_lines, seen_order = [], False
    for ln in lines:
        if re.match(r"Order\s*#", ln, re.I):
            seen_order = True
            continue
        if seen_order:
            if re.match(r"Sold\s*by", ln, re.I):
                break
            title_lines.append(ln)
    if not title_lines:
        raise Skip("order title not found in OCR")
    joined = "\n".join(lines)

    def field(name):
        m = re.search(rf"{name}\s*:?\s*(.+)", joined, re.I)
        return m.group(1).strip() if m else None

    status = field("Status") or ""
    cond = (field("Condition") or "").lower()
    out = {"title": " ".join(title_lines), "seller": field(r"Sold\s*by"),
           "preowned": cond.startswith("pre"), "condition_text": cond, "status": status, "delivered": None}
    m = re.search(r"Delivered on\s+([A-Za-z0]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})", status)
    mon = m.group(1).replace("0", "o").title() if m else None  # OCR often reads "Oct" as "0ct"
    if m and mon in MONTHS:
        out["delivered"] = dt.date(int(m.group(3)), MONTHS[mon], int(m.group(2)))
    elif not re.search(r"ship|arriv", status, re.I):
        raise Skip("cannot read order status")
    if cond[:3] not in ("new", "pre"):
        raise Skip("cannot read order condition")
    return out


def parse_customer(question: str) -> Dict[str, Any]:
    m = re.search(r"Customer message: \"(.*?)\" Today's date is (\d{4}-\d{2}-\d{2})", question, re.S)
    if not m:
        raise Skip("cannot parse customer message")
    msg, today = m.group(1), dt.date.fromisoformat(m.group(2))
    low = msg.lower()
    hours = None
    hm = re.search(r"about (\d+) hours after delivery", low)
    if hm:
        hours = int(hm.group(1))
    if "damaged" in low:
        issue, cond = "damaged", "unused"
    elif "wrong item" in low:
        issue, cond = "wrong_item", "unused"
    elif "not as described" in low:
        issue, cond = "not_as_described", "unused"
    else:
        issue = "change_of_mind"
        if "assembled" in low:
            cond = "assembled"
        elif "unused" in low or "haven't used" in low:
            cond = "unused"
        elif "used" in low or "wore" in low:
            cond = "used"
        else:
            cond = "unused"  # generic "start a return" (order not delivered yet)
    return {"message": msg, "today": today, "hours": hours, "issue": issue, "condition": cond}


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------


class Builder:
    """Collects assistant/observation turns while calling the real tools."""

    def __init__(self, run_tool: Callable[[Dict[str, Any], Dict[str, str], int], Tuple[str, Dict[str, str]]],
                 first_image: str, rng: random.Random):
        self.run_tool = run_tool
        self.images: Dict[str, str] = {"img_1": first_image}
        self.turns: List[Dict[str, str]] = []  # alternating assistant / user(observation)
        self.rng = rng
        self.n_calls = 0

    def phrase(self, options: List[str]) -> str:
        return self.rng.choice(options)

    def call(self, think: str, name: str, arguments: Dict[str, Any]) -> str:
        self.n_calls += 1
        payload = {"name": name, "arguments": arguments}
        assistant = f"<think>{think}</think>\n<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"
        run = {"name": name, "parameters": arguments}
        obs, new_images = self.run_tool(run, self.images, self.n_calls)
        if obs.startswith("Tool execution error"):
            raise Skip(f"{name} failed: {obs[:120]}")
        for k, v in (new_images or {}).items():
            self.images[k] = v
        self.turns.append({"role": "assistant", "content": assistant})
        self.turns.append({"role": "user", "content": f"<observation>\n{obs}\n</observation>"})
        return obs

    def finish(self, think: str, response: str, verdict: str) -> None:
        text = f"<think>{think}</think>\n<response>\n{response}\n\\boxed{{{verdict}}}\n</response>"
        self.turns.append({"role": "assistant", "content": text})


# ---------------------------------------------------------------------------
# The three experts
# ---------------------------------------------------------------------------


def expert_after_sales(b: Builder, q: Dict[str, Any], rules: Dict[str, Any]) -> str:
    cust = parse_customer(q["question"])
    obs = b.call(b.phrase([
        "The customer's order details are in the screenshot, so I will read the text with layout parsing before checking any policy.",
        "I need the order title, condition and delivery status from the screenshot. I will extract the text first.",
    ]), "layout_parsing", {"image": "img_1"})
    order = parse_order_ocr(obs)
    meta = q["meta"]

    # --- the screenshot must agree with the question's ground truth, otherwise OCR was wrong
    if order["preowned"] != meta["preowned"]:
        raise Skip("OCR condition disagrees with meta")
    days = None
    if order["delivered"] is not None:
        days = (cust["today"] - order["delivered"]).days
    if days != meta["days_since_delivery"]:
        raise Skip(f"days mismatch {days} vs {meta['days_since_delivery']}")
    if days is not None:  # undelivered orders carry a generic message; meta issue is a placeholder
        if cust["issue"] != meta["issue"] or (meta["issue"] == "change_of_mind" and cust["condition"] != meta["condition"]):
            raise Skip("customer message parsed differently from meta")
        if meta["hours"] != cust["hours"]:
            raise Skip("hours mismatch")

    obs = b.call(b.phrase([
        f"The order is for \"{short(order['title'])}\". I will look it up in the catalog to learn its product type, which decides the return rules.",
        "Now I need the product type of this item. I will search the catalog with the title from the screenshot.",
    ]), "text_search", {"q": order["title"]})
    prod_pid = f"prod-{meta['item_id']}"
    prod = parse_product(find_passage(obs, prod_pid)["text"])
    group = rules["type_to_group"].get(prod["type"])
    if group != meta["group"]:
        raise Skip("group mismatch")
    spec, g = rules["groups"][group], rules["global"]
    label = spec["label"]

    status_txt = f"delivered on {order['delivered'].strftime('%b %d, %Y').replace(' 0', ' ')}, which is {days} days before today ({cust['today'].isoformat()})" \
        if days is not None else "not delivered yet"

    def need(policy_id: str, query: str, think: str) -> str:
        o = b.call(think, "text_search", {"q": query})
        p = find_passage(o, policy_id)
        return p["text"]

    def check_nums(text: str, nums: List[int]):
        for n in nums:
            if str(n) not in text:
                raise Skip(f"policy text does not contain {n}")

    def normal(days: int, condition: str) -> Tuple[str, str]:
        if spec["final_sale"]:
            return "not eligible", f"{label} are final sale, so a change-of-mind return is not allowed"
        if days > spec["return_days"]:
            return "not eligible", f"{days} days is past the {spec['return_days']}-day return window for {label.lower()}"
        if condition != "unused":
            return "not eligible", "the item is not unused, and the policy requires unused condition"
        return "eligible", f"{days} days is within the {spec['return_days']}-day window and the item is unused"

    group_policy_id = f"policy-return-{group}"
    group_query = f"return policy {label}"

    if days is None:
        # Not delivered: still read the category policy so the answer is grounded.
        need(group_policy_id, group_query,
             f"The product type is {prod['type']}, so the category is {label}. I will read its return policy.")
        verdict = "cannot determine"
        reason = "The order has not been delivered yet, so the return window has not started and there is nothing to evaluate."
        why = "The screenshot status shows the order is still in transit."
    elif order["preowned"]:
        t = need("policy-preowned", "Harbor Market Policy: Pre-owned items, return only if materially different from the listing",
                 "This order is pre-owned, and pre-owned items follow their own policy rather than the category windows. I will read it.")
        check_nums(t, [g["preowned_return_days"]])
        if cust["issue"] == "not_as_described" and days <= g["preowned_return_days"]:
            verdict, why = "eligible", f"Pre-owned items can be returned within {g['preowned_return_days']} days if they differ from the listing, and this was delivered {days} days ago with a not-as-described claim."
            reason = why
        elif cust["issue"] == "not_as_described":
            verdict, why = "not eligible", f"The not-as-described claim comes {days} days after delivery, past the {g['preowned_return_days']}-day pre-owned limit."
            reason = why
        else:
            verdict, why = "not eligible", "Pre-owned items are only returnable when materially different from the listing, and this request is not that."
            reason = why
    elif cust["issue"] == "damaged":
        t = need("policy-damaged", "Harbor Market Policy: Damaged or wrong items",
                 "The customer reports damage, so I will read the damaged and wrong item policy.")
        check_nums(t, [g["damaged_report_hours"], g["wrong_item_days"]])
        if cust["hours"] is not None and cust["hours"] <= g["damaged_report_hours"]:
            verdict = "eligible"
            why = f"Damage was reported {cust['hours']} hours after delivery, within the {g['damaged_report_hours']}-hour limit."
            reason = why
        else:
            t2 = need(group_policy_id, group_query,
                      "The report is outside the damage window, so the normal category return policy applies. I will read it.")
            check_nums(t2, [spec["return_days"]] if not spec["final_sale"] else [])
            verdict, detail = normal(days, "unused")
            why = f"The damage report came after the {g['damaged_report_hours']}-hour limit, so the category policy applies: {detail}."
            reason = why
    elif cust["issue"] == "wrong_item":
        t = need("policy-damaged", "Harbor Market Policy: Damaged or wrong items",
                 "The customer says they received the wrong item, so I will read the damaged and wrong item policy.")
        check_nums(t, [g["wrong_item_days"]])
        if days <= g["wrong_item_days"]:
            verdict = "eligible"
            why = f"A wrong item reported {days} days after delivery is within the {g['wrong_item_days']}-day limit."
            reason = why
        else:
            t2 = need(group_policy_id, group_query,
                      "The wrong-item window has passed, so the normal category return policy applies. I will read it.")
            check_nums(t2, [spec["return_days"]] if not spec["final_sale"] else [])
            verdict, detail = normal(days, "unused")
            why = f"{days} days is past the {g['wrong_item_days']}-day wrong-item limit, so the category policy applies: {detail}."
            reason = why
    else:
        t = need(group_policy_id, group_query,
                 f"The product type is {prod['type']}, so the category is {label}. This is a change-of-mind return, so I will read the category return policy.")
        check_nums(t, [spec["return_days"]] if not spec["final_sale"] else [])
        verdict, detail = normal(days, cust["condition"])
        why = f"For {label.lower()}: {detail}."
        reason = why

    if verdict != q["answer"]:
        raise Skip(f"computed {verdict!r} != truth {q['answer']!r}")

    think = (f"The order is {status_txt}; the item is "
             f"{'pre-owned' if order['preowned'] else 'new'}, in category {label}. {why}")
    ending = {"eligible": "This request is eligible for a return.",
              "not eligible": "This request is not eligible for a return.",
              "cannot determine": "I cannot determine eligibility yet."}[verdict]
    response = f"{why[0].upper() + why[1:]} {ending}"
    b.finish(think, response, verdict)
    return verdict


def expert_authenticate(b: Builder, q: Dict[str, Any], rules: Dict[str, Any]) -> str:
    meta = q["meta"]
    m = re.search(r"titled \"(.*?)\"|Second-hand listing: \"(.*?)\"", q["question"])
    if not m:
        raise Skip("cannot parse listing title")
    title = m.group(1) or m.group(2)

    obs = b.call(b.phrase([
        "I must read the brand and model number from the label photo itself, so I will run layout parsing on it.",
        "The label text decides this case. I will extract it with layout parsing instead of guessing from the picture.",
    ]), "layout_parsing", {"image": "img_1"})
    lines = [l.strip() for l in obs.split("ALL RECOGNIZED TEXT:")[-1].splitlines() if l.strip()]
    if not lines:
        raise Skip("no OCR text")
    shown_brand = lines[0]
    mm = re.search(r"Model\s*No\.?\s*:?\s*(\S.*)", "\n".join(lines), re.I)
    shown_model = mm.group(1).strip() if mm else None

    variant = meta["variant"]
    if variant == "occluded":
        # the model number was blacked out: OCR must NOT have read the true model
        if shown_model and meta["catalog_model"] and meta["catalog_model"].lower() in shown_model.lower():
            raise Skip("model unexpectedly readable")
        shown_model = None
    elif variant in ("match", "brand", "model"):
        if shown_model is None or shown_model != meta["shown_model"] or shown_brand.lower() != meta["shown_brand"].lower():
            raise Skip(f"OCR read {shown_brand!r}/{shown_model!r}, expected {meta['shown_brand']!r}/{meta['shown_model']!r}")
        shown_brand = meta["shown_brand"] if shown_brand.upper() == meta["shown_brand"].upper() else shown_brand
    # variant no_catalog_model: label model is random, readable; verified below against catalog.

    obs = b.call(
        f"The label reads brand \"{shown_brand}\"" + (f" and model \"{shown_model}\"" if shown_model else " but the model number is not readable") +
        f". Next I will look up the listing \"{short(title, 60)}\" in the catalog to get the official brand and model number.",
        "text_search", {"q": title})
    prod = parse_product(find_passage(obs, f"prod-{meta['item_id']}")["text"])
    group = rules["type_to_group"].get(prod["type"])
    if not group:
        raise Skip("unknown product type")
    catalog_numbers = [x.strip() for x in prod["model"].split("/")] if prod["model"] != "not listed" else []
    cat_model = meta["catalog_model"]
    if cat_model:
        if cat_model not in catalog_numbers:
            raise Skip("catalog model number not in entry")
    else:
        if prod["model"] != "not listed":
            raise Skip("entry shows a model name only; ambiguous 'no model number' case")

    label = rules["groups"][group]["label"]
    o = b.call("I have the catalog entry. Before comparing, I will read the authentication standard for this product category.",
               "text_search", {"q": f"authentication standard {label}"})
    find_passage(o, f"policy-auth-{group}")

    if cat_model == "" or not catalog_numbers:
        verdict = "cannot verify"
        why = "The catalog entry for this product has no model number to compare against"
    elif shown_model is None:
        verdict = "cannot verify"
        why = "The model number on the label is covered, so it cannot be compared with the catalog"
    elif shown_brand.lower() != prod["brand"].lower():
        verdict = "likely counterfeit"
        why = f"The label brand \"{shown_brand}\" is not spelled like the catalog brand \"{prod['brand']}\""
    elif shown_model not in catalog_numbers:
        verdict = "likely counterfeit"
        why = f"The label model number {shown_model} does not match the catalog model number {cat_model}"
    else:
        verdict = "authentic"
        why = f"The label brand \"{shown_brand}\" and model number {shown_model} are identical to the catalog entry"
    if verdict != q["answer"]:
        raise Skip(f"computed {verdict!r} != truth {q['answer']!r}")
    b.finish(f"{why}. By the authentication standard this gives: {verdict}.",
             f"{why}. By the store's authentication standard the verdict is: {verdict}.", verdict)
    return verdict


def expert_price(b: Builder, q: Dict[str, Any], rules: Dict[str, Any]) -> str:
    """Identify the product from the top-3 image matches, checking each candidate's catalog entry.

    The generator cannot see the photo, so it uses the true product's catalog Color (which the photo
    shows) as the visual cue: candidates are opened in rank order and the first whose Color matches
    is accepted. The trajectory is skipped when that does not single out the true product.
    """
    meta = q["meta"]
    obs = b.call(b.phrase([
        "I do not know which product this is yet, so I will match the photo against the catalog images.",
        "To compare prices I first need to identify the product. I will run an image search on the customer's photo.",
    ]), "image_search", {"image": "img_1"})
    try:
        hits = json.loads(obs.split("Tool execution result:", 1)[1].strip())
    except Exception:
        raise Skip("cannot parse image_search result")
    if isinstance(hits, dict):
        hits = [hits]
    hits = [h for h in hits if h.get("source", "").startswith("prod-")][:3]
    true_src = f"prod-{meta['item_id']}"
    if true_src not in [h["source"] for h in hits]:
        raise Skip("true product not among the top-3 image matches")

    # colour of the product in the photo (from the catalog record of the true product)
    true_idx = [h["source"] for h in hits].index(true_src)
    chosen, chosen_prod = None, None
    seen_colors: List[str] = []
    for rank, h in enumerate(hits):
        if rank == 0:
            think = (f"The photo is matched to {len(hits)} catalog items. I will open the best match, \"{short(h['title'])}\", "
                     "and check its entry against what the photo shows.")
        else:
            think = (f"That entry does not fit the photo (colour differs), so I will check the next candidate, \"{short(h['title'])}\".")
        obs = b.call(think, "text_search", {"q": h["title"]})
        prod = parse_product(find_passage(obs, h["source"])["text"])
        seen_colors.append(prod["color"])
        if h["source"] == true_src:
            chosen, chosen_prod = h, prod
            break
        # a wrong candidate is only rejectable if its colour differs from the true product's colour
    if chosen is None:
        raise Skip("true product not reached")
    true_color = chosen_prod["color"]
    if any(c.strip().lower() == true_color.strip().lower() for c in seen_colors[:-1]):
        raise Skip("an earlier candidate has the same colour; cannot reject it visually")

    only_new = "new items only" in q["question"].lower()
    pool = [o for o in chosen_prod["offers"] if o["condition"] == "new"] if only_new else chosen_prod["offers"]
    if not pool:
        raise Skip("no offers")
    best = min(pool, key=lambda o: o["total"])
    verdict = f"{best['seller']}, ${best['total']:.2f}"
    if verdict != q["answer"]:
        raise Skip(f"computed {verdict!r} != truth {q['answer']!r}")
    scope = "new offers only" if only_new else "all offers including pre-owned"
    if len(seen_colors) == 1:
        ident = f"The photo matches this item (colour {true_color})."
    else:
        ident = (f"The photo shows a {true_color} item. The first {len(seen_colors) - 1} candidate(s) had other colours "
                 f"({', '.join(seen_colors[:-1])}), so the match is \"{short(chosen['title'])}\".")
    b.finish(f"{ident} Comparing price plus shipping across {len(pool)} offers ({scope}), the lowest total is {best['seller']} at ${best['total']:.2f}.",
             f"The cheapest option ({scope}) is {best['seller']}, with a total of ${best['total']:.2f} including shipping.",
             verdict)
    return verdict


EXPERTS = {"after_sales": expert_after_sales, "authenticate": expert_authenticate, "price_compare": expert_price}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def conversation(system: str, tools_json: str, q: Dict[str, Any], turns: List[Dict[str, str]]) -> Dict[str, Any]:
    first = f"<image>\n<tools>\n{tools_json}\n</tools>\n\n{q['question']}"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": first}] + turns
    return {"id": q["id"], "task": q["task"], "answer": q["answer"], "images": q["images"][:1], "messages": msgs}


def generate(questions: List[Dict[str, Any]], rules: Dict[str, Any], run_tool, system: str, tools_json: str,
             seed: int = 7) -> Tuple[List[Dict[str, Any]], Counter, List[Tuple[str, str]]]:
    out, stats, skipped = [], Counter(), []
    for q in questions:
        rng = random.Random(f"{seed}-{q['id']}")
        b = Builder(run_tool, q["images"][0], rng)
        try:
            EXPERTS[q["task"]](b, q, rules)
        except Skip as e:
            stats[f"skip:{q['task']}"] += 1
            skipped.append((q["id"], str(e)))
            continue
        except Exception as e:  # tool crash etc.; never abort the whole run
            stats[f"error:{q['task']}"] += 1
            skipped.append((q["id"], f"{type(e).__name__}: {e}"))
            continue
        stats[f"ok:{q['task']}"] += 1
        out.append(conversation(system, tools_json, q, b.turns))
    return out, stats, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--data-root", default="D:/projects/data")
    ap.add_argument("--out", required=True, help="output .jsonl")
    ap.add_argument("--task", choices=list(EXPERTS), default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-task", type=int, default=None, help="take this many questions of every task (overrides --limit)")
    args = ap.parse_args()

    if args.split != "train":
        print("Trajectories are only generated for the TRAIN split. The test split stays sealed.")
        return 1

    qdir = os.path.join(args.data_root, "questions")
    wdir = os.path.join(args.data_root, "world")
    cdir = os.path.join(args.data_root, "clip_index")
    os.environ["FVQA_IMAGE_DIR"] = os.path.join(qdir, "images")
    os.environ["KB_PATH"] = os.path.join(wdir, "corpus_train.jsonl")
    os.environ.setdefault("SEARCH_BACKEND", "offline")
    os.environ.setdefault("LAYOUT_BACKEND", "local")
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    from opensearch_vl.opensearch_infer import tools as T  # noqa: E402
    from opensearch_vl.opensearch_infer.prompts import SYSTEM_PROMPT  # noqa: E402
    from visual_lookup import CatalogLookup  # noqa: E402

    rows = [json.loads(l) for l in open(os.path.join(qdir, "train.jsonl"), encoding="utf-8")]
    if args.task:
        rows = [r for r in rows if r["task"] == args.task]
    if args.per_task:
        picked, seen = [], Counter()
        for r in rows[args.start:]:
            if seen[r["task"]] < args.per_task:
                seen[r["task"]] += 1
                picked.append(r)
        rows = picked
    else:
        rows = rows[args.start:]
        rows = rows[:args.limit] if args.limit else rows
    rules = json.load(open(os.path.join(wdir, "rules.json"), encoding="utf-8"))
    lookup = CatalogLookup(cdir, scope="train")
    tmp = tempfile.mkdtemp(prefix="traj_tools_")

    def run_tool(call, images, turn):
        return T.execute_tool(call, images, "traj", 0, turn, tmp, visual_lookup=lookup)

    out, stats, skipped = generate(rows, rules, run_tool, SYSTEM_PROMPT, T.get_tools_definition())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(stats))
    for qid, why in skipped[:15]:
        print("  skipped", qid, "-", why)
    print(f"wrote {len(out)} trajectories -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
