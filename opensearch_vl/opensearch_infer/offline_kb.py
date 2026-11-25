"""Offline product-catalog / policy knowledge base used by ``text_search``.

The original agent called Serper + Jina + an LLM summariser. This module
replaces that with retrieval over a frozen local corpus, so every run sees the
same evidence (reproducible evaluation, no web access, no per-call cost).

Corpus format: JSON Lines, one document per line::

    {"id": "catalog://product/P00123",
     "type": "product",                      # "product" or "policy"
     "title": "Acme Wireless Earbuds X2",
     "text": "Price: $42.99. Seller: ... Specs: ..."}

* ``id`` is the stable source identifier. It is shown to the agent in the
  ``URL:`` field so it can cite it in its final answer.
* ``type`` is informational; it is not used for ranking.
* ``text`` is returned to the agent verbatim (truncated to ``MAX_CHARS``).

Ranking is plain BM25 implemented here with no third-party dependency.

Environment:
    KB_PATH   path to the JSONL corpus (default: ``kb/corpus.jsonl``)

Command-line smoke test::

    python offline_kb.py --kb path/to/corpus.jsonl "return window electronics"
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

MAX_CHARS = 800          # characters of passage text returned per document
K1 = 1.5                 # BM25 term-frequency saturation
B = 0.75                 # BM25 length normalisation
SEPARATOR = "\n\n" + "=" * 60 + "\n\n"   # same separator the original tool used

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class KnowledgeBase:
    """In-memory BM25 index over a JSONL corpus."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.docs: List[Dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
                for field in ("id", "title", "text"):
                    if field not in doc or not isinstance(doc[field], str):
                        raise ValueError(f"{path}:{line_no}: missing string field {field!r}")
                self.docs.append(doc)
        if not self.docs:
            raise ValueError(f"{path}: corpus is empty")

        # The title is indexed together with the body so title words count.
        self._tf: List[Counter] = []
        self._len: List[int] = []
        df: Counter = Counter()
        for doc in self.docs:
            tokens = _tokenize(doc["title"] + " " + doc["text"])
            tf = Counter(tokens)
            self._tf.append(tf)
            self._len.append(len(tokens))
            df.update(tf.keys())
        n = len(self.docs)
        self._avgdl = sum(self._len) / n
        # BM25+ style idf, always positive.
        self._idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scored = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            dl = self._len[i]
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * dl / self._avgdl)
                score += self._idf.get(term, 0.0) * f * (K1 + 1) / denom
            if score > 0:
                scored.append((score, i))
        # Higher score first; ties broken by document id so results are deterministic.
        scored.sort(key=lambda s: (-s[0], self.docs[s[1]]["id"]))
        return [dict(self.docs[i], score=round(s, 4)) for s, i in scored[: max(1, top_k)]]


_KB: Optional[KnowledgeBase] = None
_KB_PATH: Optional[str] = None


def get_kb(path: Optional[str] = None) -> KnowledgeBase:
    """Load (and cache) the knowledge base."""
    global _KB, _KB_PATH
    path = path or os.environ.get("KB_PATH", "kb/corpus.jsonl")
    if _KB is None or _KB_PATH != path:
        _KB = KnowledgeBase(path)
        _KB_PATH = path
    return _KB


def format_passages(results: List[Dict]) -> str:
    """Render results in the same observation layout the original tool used."""
    blocks = []
    for idx, doc in enumerate(results, start=1):
        text = doc["text"].strip()
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS].rstrip() + " ..."
        blocks.append(
            f"[Passage {idx}]\n"
            f"Title: {doc['title']}\n"
            f"URL: {doc['id']}\n"
            f"Summary:\n{text}"
        )
    return "Tool execution result:\n" + SEPARATOR.join(blocks)


def text_search(query: str, top_k: int = 5, kb_path: Optional[str] = None) -> str:
    """Offline replacement for the web ``text_search`` tool (returns the observation text)."""
    try:
        kb = get_kb(kb_path)
    except Exception as exc:
        return f"Tool execution error:\nKnowledge base unavailable: {exc}"
    results = kb.search(query, top_k=top_k)
    if not results:
        return (
            "Tool execution result:\n"
            "No relevant passages found in the knowledge base for the query."
        )
    return format_passages(results)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the offline knowledge base.")
    ap.add_argument("query")
    ap.add_argument("--kb", default=None, help="path to the JSONL corpus")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()
    print(text_search(args.query, top_k=args.top_k, kb_path=args.kb))
