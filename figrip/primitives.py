"""Read a PDF page into normalised Paths and TextSpans (see model.py)."""
from __future__ import annotations

import math

import numpy as np
import pymupdf

from .model import Path, PageContent, TextSpan

_EPS = 1e-3          # points closer than this are "the same" when chaining items


def _bezier(p0, c1, c2, p3) -> np.ndarray:
    """Flatten a cubic bezier; returns points excluding p0."""
    pts = np.array([p0, c1, c2, p3], float)
    ctrl_len = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    n = int(min(64, max(4, math.ceil(ctrl_len / 0.5))))
    t = np.linspace(0, 1, n + 1)[1:, None]
    mt = 1 - t
    return (mt**3) * pts[0] + 3 * mt**2 * t * pts[1] + 3 * mt * t**2 * pts[2] + t**3 * pts[3]


def _xy(p) -> tuple[float, float]:
    return (float(p.x), float(p.y))


def _subpaths(items, close_last: bool):
    """Split a drawing's item list into polylines.

    PyMuPDF flattens moveto/lineto into a list of items; a new subpath starts
    whenever an item does not begin where the previous one ended.
    """
    subs: list[list[tuple[float, float]]] = []
    closed: list[bool] = []
    cur: list[tuple[float, float]] = []
    has_curves = False

    def flush(is_closed=False):
        nonlocal cur
        if len(cur) >= 1:
            subs.append(cur)
            closed.append(is_closed)
        cur = []

    for it in items:
        op = it[0]
        if op == "l":
            a, b = _xy(it[1]), _xy(it[2])
            if not cur or abs(cur[-1][0] - a[0]) > _EPS or abs(cur[-1][1] - a[1]) > _EPS:
                flush()
                cur = [a]
            cur.append(b)
        elif op == "c":
            has_curves = True
            a = _xy(it[1])
            if not cur or abs(cur[-1][0] - a[0]) > _EPS or abs(cur[-1][1] - a[1]) > _EPS:
                flush()
                cur = [a]
            cur.extend(map(tuple, _bezier(a, _xy(it[2]), _xy(it[3]), _xy(it[4]))))
        elif op == "re":
            flush()
            r = it[1]
            cur = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
            flush(True)
        elif op == "qu":
            flush()
            q = it[1]
            cur = [_xy(q.ul), _xy(q.ur), _xy(q.lr), _xy(q.ll), _xy(q.ul)]
            flush(True)
    flush()
    if close_last and closed:
        closed[-1] = True
    # an explicitly coincident first/last vertex is closed too
    for i, s in enumerate(subs):
        if len(s) > 2 and abs(s[0][0] - s[-1][0]) < _EPS and abs(s[0][1] - s[-1][1]) < _EPS:
            closed[i] = True
    return [np.array(s, float) for s in subs], closed, has_curves


def _color(c):
    if c is None:
        return None
    c = tuple(float(v) for v in c)
    if len(c) == 1:          # gray
        return (c[0], c[0], c[0])
    if len(c) == 4:          # cmyk -> rgb (naive)
        cc, m, y, k = c
        return ((1 - cc) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return c[:3]


def extract_page(page: pymupdf.Page) -> PageContent:
    paths: list[Path] = []
    for d in page.get_drawings():
        typ = d.get("type", "")
        subs, closed, curves = _subpaths(d["items"], bool(d.get("closePath")))
        if not subs:
            continue
        stroke = _color(d.get("color")) if "s" in typ else None
        fill = _color(d.get("fill")) if "f" in typ else None
        dashes = d.get("dashes")
        if dashes in ("[] 0", "[] 0.0", ""):
            dashes = None
        r = d["rect"]
        paths.append(Path(
            id=len(paths), subpaths=subs, closed=closed, stroke=stroke, fill=fill,
            width=float(d.get("width") or 0.0) if stroke is not None else 0.0,
            dashes=dashes, bbox=(r.x0, r.y0, r.x1, r.y1),
            seqno=int(d.get("seqno", len(paths))), has_curves=curves,
            n_segments=len(d["items"]),
        ))

    texts: list[TextSpan] = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if not s["text"].strip():
                    continue
                texts.append(TextSpan(
                    text=s["text"], bbox=tuple(s["bbox"]), size=float(s["size"]),
                    font=s["font"], dir=tuple(line["dir"]), origin=tuple(s["origin"]),
                ))

    images = []
    for info in page.get_image_info():
        images.append(tuple(info["bbox"]))

    r = page.rect
    return PageContent(page.number, (r.x0, r.y0, r.x1, r.y1), paths, texts, images)


def open_page(pdf_path: str, page_no: int) -> PageContent:
    with pymupdf.open(pdf_path) as doc:
        return extract_page(doc[page_no])
