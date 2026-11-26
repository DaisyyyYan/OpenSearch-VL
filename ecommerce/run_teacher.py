"""Run a teacher model (Claude) through the agent loop on our question set.

Replaces opensearch_vl/run_infer.py for this project: it reads our JSONL
questions directly (no parquet), wires the offline tools (local knowledge
base, CLIP image lookup, local OCR) and runs several questions in parallel.

Run from the repository root, with ANTHROPIC_API_KEY exported:

    python ecommerce/run_teacher.py --split train --limit 20 --out D:/projects/data/pilot

Trajectories are written as <out>/<question id>_trajectory.json; questions that
already have a file are skipped, so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def is_empty_trajectory(path):
    """A trajectory with <=1 turn and no final text means the model call itself failed."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return True
    turns = d.get("turns", []) if isinstance(d, dict) else []
    final = (d.get("final_response_text") or "") if isinstance(d, dict) else ""
    first = (turns[0].get("response_text") or "") if turns else ""
    return len(turns) <= 1 and not final.strip() and not first.strip()


def load_questions(path, task, start, limit):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if task:
        rows = [r for r in rows if r["task"] == task]
    rows = rows[start:]
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--data-root", default="D:/projects/data", help="folder holding questions/, world/, clip_index/")
    ap.add_argument("--out", required=True, help="output folder for trajectories")
    ap.add_argument("--model", default="claude-sonnet", help="name in MODEL_REGISTRY")
    ap.add_argument("--task", choices=["price_compare", "authenticate", "after_sales"], default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="check files and settings, call nothing")
    args = ap.parse_args()

    qdir = os.path.join(args.data_root, "questions")
    wdir = os.path.join(args.data_root, "world")
    cdir = os.path.join(args.data_root, "clip_index")
    scope = "train" if args.split == "train" else "full"
    kb = os.path.join(wdir, "corpus_train.jsonl" if scope == "train" else "corpus_full.jsonl")

    # The repo reads these at import time, so set them BEFORE importing it.
    os.environ["FVQA_IMAGE_DIR"] = os.path.join(qdir, "images")
    os.environ["KB_PATH"] = kb
    os.environ.setdefault("SEARCH_BACKEND", "offline")
    os.environ.setdefault("LAYOUT_BACKEND", "local")
    os.environ["AGENT_MAX_TURNS"] = str(args.max_turns)

    questions = load_questions(os.path.join(qdir, f"{args.split}.jsonl"), args.task, args.start, args.limit)
    os.makedirs(args.out, exist_ok=True)
    def finished(q):
        p = os.path.join(args.out, f"{q['id']}_trajectory.json")
        return os.path.exists(p) and not is_empty_trajectory(p)

    todo = [q for q in questions if not finished(q)]
    print(f"split={args.split} scope={scope} kb={os.path.basename(kb)} model={args.model}")
    print(f"{len(questions)} questions selected, {len(questions) - len(todo)} already done, {len(todo)} to run")

    missing = [q["id"] for q in todo if not os.path.exists(q["images"][0])]
    if missing or not os.path.exists(kb):
        print("missing files:", missing[:5], "kb exists:", os.path.exists(kb))
        return 1
    if args.dry_run:
        print("dry run OK")
        return 0
    if args.model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set in this terminal window. Run: export ANTHROPIC_API_KEY=your_key")
        return 1

    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    from opensearch_vl.opensearch_infer import pipeline, runners  # noqa: E402
    from visual_lookup import CatalogLookup  # noqa: E402

    runner = runners.build_runner(args.model)
    runner.load()
    lookup = CatalogLookup(cdir, scope=scope)
    cfg = runners.InferenceConfig(max_tokens=args.max_tokens)

    done, failed = 0, 0
    lock = threading.Lock()
    t0 = time.time()

    def work(idx_q):
        idx, q = idx_q
        row = {"data_id": q["id"], "category": q["task"], "data_source": "ecommerce-synthetic",
               "prompt": q["question"], "images": [], "answer": q["answer"]}
        pipeline.process_single_case(row=row, runner=runner, output_dir=args.out, case_idx=idx,
                                     dataset_type=args.split, visual_lookup=lookup, inference_cfg=cfg)
        path = os.path.join(args.out, f"{q['id']}_trajectory.json")
        if os.path.exists(path) and is_empty_trajectory(path):
            try:
                print(f"--- empty trajectory for {q['id']}: {open(path, encoding='utf-8').read()[:1500]}")
            except OSError:
                pass
            os.remove(path)  # so a re-run tries this question again
            raise RuntimeError("empty trajectory: the model call failed (see the log lines above)")
        return q["id"]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, (i, q)): q["id"] for i, q in enumerate(todo)}
        for fu in as_completed(futs):
            try:
                fu.result()
                with lock:
                    done += 1
            except Exception as exc:
                with lock:
                    failed += 1
                print(f"FAILED {futs[fu]}: {exc}")
                traceback.print_exc()
            n = done + failed
            if n % 5 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} done ({failed} failed), {time.time() - t0:.0f}s elapsed")
    print(f"finished: ok={done} failed={failed} out={args.out}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
