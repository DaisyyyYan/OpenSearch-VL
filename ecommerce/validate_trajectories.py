"""Validate generated trajectories and print dataset statistics.

Checks every conversation in a trajectories .jsonl (output of make_trajectories.py):
  - roles alternate system, user, assistant, user, assistant, ...; last turn is assistant
  - every non-final assistant turn has <think> and exactly one valid <tool_call> (known tool, dict arguments)
  - the final assistant turn has <response> with boxed{...} and no tool call
  - at most MAX_TURNS assistant turns
  - the boxed verdict matches the ground truth (same rule as score_trajectories.py)

Writes the clean conversations to --out and prints counts, tool usage, turn and length statistics.

    python ecommerce/validate_trajectories.py --inp D:/projects/data/traj/train_all.jsonl \
        --out D:/projects/data/traj/train_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_trajectories import BOXED, MAX_TURNS, TOOL_CALL, TOOLS, check_answer, parse_boxed  # noqa: E402


def check(conv):
    problems, tools = [], []
    msgs = conv["messages"]
    if len(msgs) < 4 or msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
        return ["bad start"], tools
    body = msgs[2:]
    if body[-1]["role"] != "assistant":
        problems.append("last turn is not assistant")
    for i, m in enumerate(msgs[1:], start=1):
        if m["role"] != ("user" if i % 2 == 1 else "assistant"):
            problems.append("roles do not alternate")
            break
    assistants = [m["content"] for m in msgs if m["role"] == "assistant"]
    if len(assistants) > MAX_TURNS:
        problems.append(f"too many turns ({len(assistants)})")
    for j, text in enumerate(assistants[:-1]):
        calls = TOOL_CALL.findall(text)
        if len(calls) != 1:
            problems.append(f"turn {j}: {len(calls)} tool calls")
            continue
        try:
            c = json.loads(calls[0])
            if c.get("name") not in TOOLS or not isinstance(c.get("arguments"), dict):
                problems.append(f"turn {j}: bad tool call")
            else:
                tools.append(c["name"])
        except json.JSONDecodeError:
            problems.append(f"turn {j}: invalid JSON")
        if "<think>" not in text or "</think>" not in text:
            problems.append(f"turn {j}: no think")
    last = assistants[-1] if assistants else ""
    if "<response>" not in last or "</response>" not in last:
        problems.append("final: no response")
    if TOOL_CALL.search(last):
        problems.append("final: contains tool call")
    got = parse_boxed(last)
    if got is None:
        problems.append("final: no boxed")
    elif not check_answer(conv["task"], conv["answer"], got):
        problems.append(f"final: wrong answer {got!r} vs {conv['answer']!r}")
    return problems, tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    clean, bad = [], []
    per_task, tool_use, turns, chars = Counter(), defaultdict(Counter), defaultdict(list), defaultdict(list)
    for r in rows:
        problems, tools = check(r)
        if problems:
            bad.append((r["id"], problems[:2]))
            continue
        clean.append(r)
        per_task[r["task"]] += 1
        tool_use[r["task"]].update(tools)
        turns[r["task"]].append(len(tools) + 1)
        chars[r["task"]].append(sum(len(m["content"]) for m in r["messages"]))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} read, {len(clean)} clean, {len(bad)} rejected")
    for b in bad[:10]:
        print("  rejected", b)
    ans = Counter((r["task"], r["answer"] if r["task"] != "price_compare" else "-") for r in clean)
    for task in sorted(per_task):
        t = turns[task]
        print(f"\n[{task}] {per_task[task]} trajectories | assistant turns mean {statistics.mean(t):.1f} (min {min(t)}, max {max(t)})"
              f" | chars mean {statistics.mean(chars[task]):.0f}, max {max(chars[task])} (~{max(chars[task]) // 3} tokens)")
        print("  tools:", dict(tool_use[task]))
        if task != "price_compare":
            print("  answers:", {a: n for (tk, a), n in ans.items() if tk == task})
    allc = [c for v in chars.values() for c in v]
    if allc:
        print(f"\noverall max ~{max(allc) // 3} tokens (chars/3). Set cutoff_len above this plus image tokens.")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
