"""Validate and score teacher trajectories against the ground truth.

For every <id>_trajectory.json in a folder it checks

  format  : every tool turn has exactly one valid <tool_call> with a known tool
            name, the last turn is a <response> with boxed{...} and no tool
            call, at most MAX_TURNS turns
  process : image_search is followed by text_search; no tool errors
  answer  : the boxed verdict matches the ground truth in questions/<split>.jsonl

and prints accuracy and format-validity per task. Trajectories that are both
well-formed and correct are the ones kept for SFT (written to --keep-file).

Usage:
    python ecommerce/score_trajectories.py --traj D:/projects/data/pilot \
        --questions D:/projects/data/questions/train.jsonl --keep-file D:/projects/data/pilot_keep.txt
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

TOOLS = {"crop", "perspective_correct", "sharpen", "super_resolution", "layout_parsing",
         "text_search", "web_search", "image_search"}
MAX_TURNS = 12
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
BOXED = re.compile(r"\\?boxed\{([^{}]*)\}")


def parse_boxed(text: str) -> Optional[str]:
    found = BOXED.findall(text or "")
    return found[-1].strip() if found else None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().strip(". "))


def check_answer(task: str, truth: str, got: Optional[str]) -> bool:
    if got is None:
        return False
    if task != "price_compare":
        return norm(got) == norm(truth)
    t_seller, t_price = [x.strip() for x in truth.rsplit(",", 1)]
    m = re.search(r"\$?\s*([0-9][0-9,]*\.?[0-9]*)", got.rsplit(",", 1)[-1])
    if not m:
        return False
    try:
        price_ok = abs(float(m.group(1).replace(",", "")) - float(t_price.replace("$", "").replace(",", ""))) < 0.005
    except ValueError:
        return False
    return price_ok and norm(t_seller) in norm(got)


def check_format(turns: List[Dict[str, Any]], final_text: str) -> Tuple[List[str], List[str]]:
    problems: List[str] = []
    tools: List[str] = []
    if not turns:
        return ["no turns"], tools
    if len(turns) > MAX_TURNS:
        problems.append(f"too many turns ({len(turns)})")
    for i, t in enumerate(turns[:-1]):
        text = t.get("response_text", "") or ""
        calls = TOOL_CALL.findall(text)
        if len(calls) != 1:
            problems.append(f"turn {i}: {len(calls)} tool calls")
            continue
        try:
            call = json.loads(calls[0])
            name = call.get("name")
            if name not in TOOLS:
                problems.append(f"turn {i}: unknown tool {name!r}")
            elif not isinstance(call.get("arguments"), dict):
                problems.append(f"turn {i}: arguments is not an object")
            else:
                tools.append(name)
        except json.JSONDecodeError:
            problems.append(f"turn {i}: invalid tool_call JSON")
        if "<think>" not in text:
            problems.append(f"turn {i}: no <think>")
    last = turns[-1].get("response_text", "") or final_text or ""
    if "<response>" not in last or "</response>" not in last:
        problems.append("last turn has no <response>")
    if TOOL_CALL.search(last):
        problems.append("last turn still contains a tool call")
    if parse_boxed(last) is None:
        problems.append("no boxed{} verdict")
    return problems, tools


def check_process(turns: List[Dict[str, Any]], tools: List[str]) -> List[str]:
    notes = []
    if "image_search" in tools:
        after = tools[tools.index("image_search") + 1:]
        if "text_search" not in after and "web_search" not in after:
            notes.append("image_search not followed by text_search")
    errors = sum(1 for t in turns[:-1] if "Tool execution error" in str(t.get("tool_output", "")))
    if errors:
        notes.append(f"{errors} tool error(s)")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--keep-file", default=None, help="write ids that are well-formed AND correct")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    truth = {}
    for l in open(args.questions, encoding="utf-8"):
        r = json.loads(l)
        truth[r["id"]] = r
    stats: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.Counter())
    keep: List[str] = []
    rows = []
    for path in sorted(glob.glob(os.path.join(args.traj, "*_trajectory.json"))):
        d = json.load(open(path, encoding="utf-8"))
        cid = d.get("case_id") or os.path.basename(path).replace("_trajectory.json", "")
        q = truth.get(cid)
        if q is None:
            continue
        turns = d.get("turns", [])
        final_text = d.get("final_response_text", "") or ""
        fmt, tools = check_format(turns, final_text)
        proc = check_process(turns, tools)
        got = parse_boxed(final_text or (turns[-1].get("response_text", "") if turns else ""))
        ok = check_answer(q["task"], q["answer"], got)
        s = stats[q["task"]]
        s["n"] += 1
        s["well_formed"] += not fmt
        s["correct"] += ok
        usable = (not fmt) and ok and not any(n.startswith("image_search not followed") for n in proc)
        s["kept"] += usable
        s["turns"] += len(turns)
        if usable:
            keep.append(cid)
        rows.append((cid, q["answer"], got, ok, fmt, proc, tools))

    for cid, truth_a, got, ok, fmt, proc, tools in rows:
        flag = "OK " if ok and not fmt and not any(n.startswith("image_search not followed") for n in proc) else "BAD"
        print(f"{flag} {cid} | truth={truth_a!r} got={got!r} | tools={'>'.join(tools)}")
        if fmt:
            print("      format:", "; ".join(fmt))
        if proc:
            print("      process:", "; ".join(proc))
    print("\nSUMMARY")
    for task, s in sorted(stats.items()):
        n = int(s["n"])
        print(f"  {task:14s} n={n:4d}  well-formed {s['well_formed']/n:5.0%}  correct {s['correct']/n:5.0%}  "
              f"kept {s['kept']/n:5.0%}  avg turns {s['turns']/n:4.1f}")
    if args.keep_file:
        with open(args.keep_file, "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))
        print(f"kept {len(keep)} ids -> {args.keep_file}")


if __name__ == "__main__":
    main()
