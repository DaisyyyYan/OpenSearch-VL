"""Local OCR used by the ``layout_parsing`` tool (offline, no endpoint needed).

Backed by RapidOCR (ONNX models bundled in the pip package):

    python -m pip install rapidocr_onnxruntime

The original project calls a PP-StructureV3-compatible HTTP endpoint. For a
reproducible, offline setup we replace it with plain OCR plus a simple
reading-order step: boxes are de-skewed, grouped into text lines and sorted
top-to-bottom / left-to-right, so "Sold by:" and its value end up on the same
line even when the photo is tilted.

The return value has the same shape as ``search.layout_parsing``:
``{"rec_texts", "formatted_text", "blocks", "error"}``.
"""

from __future__ import annotations

import math
import os
import statistics
import threading
from typing import Any, Dict, List, Tuple

_ENGINE = None
_LOCK = threading.Lock()


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR()
    return _ENGINE


def _empty(error: str) -> Dict[str, Any]:
    return {"error": error, "rec_texts": [], "formatted_text": "", "blocks": []}


def order_lines(items: List[Tuple[List[List[float]], str, float]]) -> List[str]:
    """Turn OCR boxes into text lines in reading order.

    ``items`` are ``(box, text, confidence)`` with ``box`` = 4 corner points.
    """
    if not items:
        return []
    # estimate the tilt from the top edge of every box and rotate it away
    angles = []
    for box, _, _ in items:
        dx, dy = box[1][0] - box[0][0], box[1][1] - box[0][1]
        if abs(dx) > 1:
            angles.append(math.atan2(dy, dx))
    tilt = statistics.median(angles) if angles else 0.0
    cos_t, sin_t = math.cos(-tilt), math.sin(-tilt)

    rows = []
    for box, text, _ in items:
        cx = sum(p[0] for p in box) / 4.0
        cy = sum(p[1] for p in box) / 4.0
        height = max(1.0, math.dist(box[0], box[3]))
        rx, ry = cx * cos_t - cy * sin_t, cx * sin_t + cy * cos_t
        rows.append({"x": rx, "y": ry, "h": height, "text": text})

    rows.sort(key=lambda r: r["y"])
    lines: List[List[Dict[str, float]]] = []
    for r in rows:
        placed = False
        for line in lines:
            mean_y = sum(i["y"] for i in line) / len(line)
            mean_h = sum(i["h"] for i in line) / len(line)
            if abs(r["y"] - mean_y) <= 0.6 * max(mean_h, r["h"]):
                line.append(r)
                placed = True
                break
        if not placed:
            lines.append([r])
    lines.sort(key=lambda ln: sum(i["y"] for i in ln) / len(ln))
    return [" ".join(i["text"] for i in sorted(ln, key=lambda i: i["x"])) for ln in lines]


def layout_parsing(file_path: str) -> Dict[str, Any]:
    """OCR a local image file. Never raises."""
    if not os.path.exists(file_path):
        return _empty(f"File not found: {file_path}")
    try:
        with _LOCK:
            result, _ = _engine()(file_path)
    except ImportError:
        return _empty("Local OCR is not installed. Run: python -m pip install rapidocr_onnxruntime")
    except Exception as exc:  # pragma: no cover - depends on the OCR runtime
        return _empty(f"OCR error: {exc}")

    items = [(box, str(text), float(conf)) for box, text, conf in (result or []) if str(text).strip()]
    lines = order_lines(items)
    blocks = [{"block_label": "text", "block_content": ln} for ln in lines]
    return {"rec_texts": lines, "formatted_text": "\n".join(lines), "blocks": blocks, "error": None}
