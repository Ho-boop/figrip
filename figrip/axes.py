"""Find x-y plots on a page and calibrate their axes.

``find_panels(pc, page=None, hints=None) -> list[Panel]``

Algorithm
---------
1. **Segments.**  Every axis-aligned straight piece of ink becomes a `_Seg`:
   horizontal/vertical edges of stroked paths, *and* thin filled rectangles
   (journals often "outline" strokes, so an axis line or a tick mark arrives
   as a 0.36 pt wide filled box).  Both kinds are normalised to the same
   (orientation, fixed coordinate, lo, hi, thickness) form, so nothing later
   cares how the line was painted.
2. **Axis lines.**  Long segments are merged (duplicates and collinear
   pieces become one line).
3. **Plot boxes from corners.**  A horizontal and a vertical line whose *ends*
   meet form an L: that is the corner of a plot area.  Requiring that the
   ends meet (not just that the lines cross) rejects grid lines, zero lines
   and table rules.  Each L gives a box; duplicates (the 4 corners of a full
   frame) are merged.
4. **Ticks.**  Along each side of a box: short perpendicular segments that
   touch the side line.  Long ones are major ticks, short ones minor ticks.
5. **Tick labels.**  Numbers in a band just outside the side (beyond the
   outward ticks), from the first source that works:
     a. real text spans (superscript spans glued on: "10" + "-3" -> 10^-3);
     b. outlined glyphs, read by `ocr.glyph_words` (template matching);
     c. text in a font without a unicode map (garbage codes): its character
        boxes are rendered and template-matched (`ocr.rendered_words`);
     d. pixel OCR of the rendered band (`ocr.ocr_region`, optional
        dependency, rationed to OCR_BUDGET calls per page).
   Each label is attached to the nearest major tick, so the *tick mark*
   gives the exact position and the label only gives the value.
6. **Robust fit.**  (position, value) pairs must be linear (or log-linear)
   to within 5 % of the smallest label step.  If not, try to repair one
   misread label (flip its sign — a lost minus — or drop it) when enough
   labels remain to still prove the progression.
   A box becomes a `Panel` only when both x and y fit: this is what keeps
   legends, molecule drawings, energy diagrams, tables and photos out.
7. **Shared axes.**  A plot whose y (or x) labels are missing but which
   touches a calibrated neighbour with the same y (x) extent borrows the
   neighbour's calibration (broken axes, side-by-side panels).
8. Extras: offset/multiplier texts ("1e−4", "×10^3"), axis titles, panel
   letter, twin y axes (right axis with a different scale is reported in
   ``notes`` and ignored).

Hints (manual override)
-----------------------
``hints`` is a dict; every key is optional::

    {
      "bbox":    (x0, y0, x1, y1),   # page coords of the plot area (approximate is fine).
                                     # Only boxes overlapping it are kept; if no box is
                                     # found there, this rectangle itself is used.
      "x_ticks": [0, 0.2, 0.4],      # values of the labelled x ticks, LEFT -> RIGHT, or
                 [(pos, value), ...] # explicit (page x, value) pairs
      "y_ticks": [-20, -15, -10],    # values of the labelled y ticks, BOTTOM -> TOP, or
                 [(pos, value), ...] # explicit (page y, value) pairs
      "xscale":  "linear" | "log",   # force the scale (default: inferred)
      "yscale":  "linear" | "log",
    }

With plain values the tick *marks* are located automatically and paired in
order with the given values; the count must equal the number of major ticks
(or of all ticks).  Calibrations built from hints have ``source="manual"``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import ocr
from .model import AxisCalib, BBox, PageContent, Panel
from .ocr import Word

MIN_AXIS = 15.0      # pt; shorter lines cannot be an axis
MAX_THIN = 2.5       # pt; a filled rectangle thinner than this is a line
FIT_TOL = 0.05       # max fit residual, in tick steps, when positions come from tick marks
FIT_TOL_LOOSE = 0.15 # ... when positions come from label centres (no tick marks)
SIDES = ("bottom", "top", "left", "right")


# --------------------------------------------------------------------------- 1. segments
@dataclass
class _Seg:
    o: str          # 'h' (horizontal) or 'v' (vertical)
    c: float        # fixed coordinate: y for 'h', x for 'v'
    lo: float       # extent along the line
    hi: float
    t: float        # thickness
    pid: int = -1   # source path id (-1 for merged lines)

    @property
    def length(self) -> float:
        return self.hi - self.lo


def _stroke_segments(p) -> list[_Seg]:
    out = []
    if p.width > 4 or p.dashes:          # thick bars and dashed guides are never axes/ticks
        return out
    t = max(p.width, 0.1)
    for sp in p.subpaths:
        if len(sp) < 2:
            continue
        a, b = sp[:-1], sp[1:]
        dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        adx, ady = np.abs(dx), np.abs(dy)
        for i in np.nonzero((ady <= 0.02 * adx + 0.01) & (adx > 0.3))[0]:
            out.append(_Seg("h", (a[i, 1] + b[i, 1]) / 2, min(a[i, 0], b[i, 0]), max(a[i, 0], b[i, 0]), t, p.id))
        for i in np.nonzero((adx <= 0.02 * ady + 0.01) & (ady > 0.3))[0]:
            out.append(_Seg("v", (a[i, 0] + b[i, 0]) / 2, min(a[i, 1], b[i, 1]), max(a[i, 1], b[i, 1]), t, p.id))
    return out


def _fill_segments(p) -> list[_Seg]:
    """Thin axis-aligned filled rectangles -> segments (outlined strokes)."""
    out = []
    for sp in p.subpaths:
        pts = sp
        if len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=1e-3):
            pts = pts[:-1]
        if 6 < len(pts) <= 64:
            out += _rectilinear_strips(pts, p.id)
            continue
        if len(pts) < 4:
            continue
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        w, h = x1 - x0, y1 - y0
        thin, long_ = min(w, h), max(w, h)
        if thin > MAX_THIN or long_ < 0.5 or long_ < 1.4 * thin:   # stubby ticks are ~2:1
            continue
        # every vertex must sit on a bbox corner: rejects tilted quads (pieces of a curve)
        eps = max(0.03, 0.15 * thin)
        cx = np.minimum(np.abs(pts[:, 0] - x0), np.abs(pts[:, 0] - x1))
        cy = np.minimum(np.abs(pts[:, 1] - y0), np.abs(pts[:, 1] - y1))
        if np.any(cx > eps) or np.any(cy > eps):
            continue
        if w >= h:
            out.append(_Seg("h", (y0 + y1) / 2, x0, x1, h, p.id))
        else:
            out.append(_Seg("v", (x0 + x1) / 2, y0, y1, w, p.id))
    return out


def _rectilinear_strips(pts: np.ndarray, pid: int) -> list[_Seg]:
    """Thin strips inside a filled polygon made only of horizontal/vertical edges.

    An outlined *frame* often arrives as one polygon tracing the outer and the
    inner rectangle (16 vertices), and an axis with its ticks as a comb shape.
    Two parallel edges closer than MAX_THIN, overlapping over a stretch much
    longer than their distance, enclose a strip of ink: that strip is a line.
    """
    a, b = pts, np.roll(pts, -1, axis=0)
    d = b - a
    horiz = np.abs(d[:, 1]) < 1e-3
    vert = np.abs(d[:, 0]) < 1e-3
    if not np.all(horiz | vert):
        return []                                   # curved or slanted: not a frame
    out = []
    for o, mask, c_ax, r_ax in (("h", horiz, 1, 0), ("v", vert, 0, 1)):
        edges = [(a[i, c_ax], min(a[i, r_ax], b[i, r_ax]), max(a[i, r_ax], b[i, r_ax]))
                 for i in np.nonzero(mask)[0]]
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                c1, lo1, hi1 = edges[i]
                c2, lo2, hi2 = edges[j]
                gap = abs(c1 - c2)
                lo, hi = max(lo1, lo2), min(hi1, hi2)
                if 0.05 < gap <= MAX_THIN and hi - lo > 2 * gap:
                    out.append(_Seg(o, (c1 + c2) / 2, lo, hi, gap, pid))
    return out


def _segments(pc: PageContent) -> list[_Seg]:
    segs: list[_Seg] = []
    for p in pc.paths:
        if p.stroke is not None:
            segs += _stroke_segments(p)
        if p.fill is not None:
            segs += _fill_segments(p)
    return segs


# --------------------------------------------------------------------------- 2. axis lines
def _merge_lines(segs: list[_Seg]) -> list[_Seg]:
    """Merge long collinear segments (an axis drawn twice or in pieces) into lines."""
    out: list[_Seg] = []
    for o in "hv":
        cand = sorted((s for s in segs if s.o == o and s.length >= MIN_AXIS), key=lambda s: (s.c, s.lo))
        merged: list[_Seg] = []
        for s in cand:
            for m in merged:
                if abs(m.c - s.c) <= max(0.6, (m.t + s.t) / 2) and s.lo <= m.hi + 1.0 and s.hi >= m.lo - 1.0:
                    m.c = (m.c * m.length + s.c * s.length) / (m.length + s.length)
                    m.lo, m.hi, m.t = min(m.lo, s.lo), max(m.hi, s.hi), max(m.t, s.t)
                    break
            else:
                merged.append(_Seg(o, s.c, s.lo, s.hi, s.t))
        out += merged
    return out


# --------------------------------------------------------------------------- 3. boxes
@dataclass
class _Box:
    x0: float
    y0: float
    x1: float
    y1: float
    sides: dict = field(default_factory=dict)    # side name -> _Seg (axis line) or None

    @property
    def bbox(self) -> BBox:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


def _split_at_dividers(lines: list[_Seg], segs: list[_Seg]) -> list[_Seg]:
    """Split lines where a *ticked* perpendicular line ends on them.

    Panels drawn edge to edge (sharey with wspace=0, broken axes) share a spine,
    and their bottom spines merge into one long line with no corner in the
    middle.  The shared spine is recognisable: both its ends sit on lines and
    it carries tick marks (a grid line, which also spans frame to frame, does
    not).  Cutting the long lines there restores the two corners.
    """
    out = list(lines)
    for o, po in (("h", "v"), ("v", "h")):
        cross = [l for l in lines if l.o == po]
        for d in cross:
            ends_on = []
            for l in lines:
                if l.o != o:
                    continue
                tol = max(1.0, 2 * (l.t + d.t))
                interior = l.lo + tol < d.c < l.hi - tol
                if interior and (abs(d.lo - l.c) <= tol or abs(d.hi - l.c) <= tol):
                    ends_on.append(l)
            if len(ends_on) < 2:
                continue
            # tick marks on the divider: >= 3 short segments that END on it, evenly spaced
            # (legend samples or curve pieces touching a grid line are not regular)
            pos = sorted({round(t.c, 1) for t in segs if t.o == o and t.length <= 12
                          and d.lo < t.c < d.hi
                          and min(abs(t.lo - d.c), abs(t.hi - d.c)) <= max(0.5, d.t)})
            steps = np.diff(pos)
            if len(pos) < 3 or steps.min() <= 0 or steps.max() > 1.15 * steps.min():
                continue
            for l in ends_on:
                if l in out:
                    out.remove(l)
                    out.append(_Seg(o, l.c, l.lo, d.c, l.t))
                    out.append(_Seg(o, l.c, d.c, l.hi, l.t))
    return out


def _corner_boxes(lines: list[_Seg]) -> list[_Box]:
    H = [s for s in lines if s.o == "h"]
    V = [s for s in lines if s.o == "v"]
    if not H or not V:
        return []
    hc = np.array([s.c for s in H]); hlo = np.array([s.lo for s in H]); hhi = np.array([s.hi for s in H])
    ht = np.array([s.t for s in H])
    boxes: list[_Box] = []
    for v in V:
        tol = np.maximum(2.0, 2 * (ht + v.t))
        at_lo = np.abs(hlo - v.c) <= tol          # v sits at the left end of h
        at_hi = np.abs(hhi - v.c) <= tol          # ... or at its right end
        v_top = np.abs(v.lo - hc) <= tol          # h sits at the top end of v
        v_bot = np.abs(v.hi - hc) <= tol          # ... or at its bottom end
        for i in np.nonzero((at_lo | at_hi) & (v_top | v_bot))[0]:
            x0, x1 = (v.c, hhi[i]) if at_lo[i] else (hlo[i], v.c)
            y0, y1 = (hc[i], v.hi) if v_top[i] else (v.lo, hc[i])
            if x1 - x0 < MIN_AXIS or y1 - y0 < MIN_AXIS:
                continue
            boxes.append(_Box(x0, y0, x1, y1))
    # the corners of one frame give (nearly) the same box: keep one, average-free
    uniq: list[_Box] = []
    for b in boxes:
        if not any(max(abs(b.x0 - u.x0), abs(b.y0 - u.y0), abs(b.x1 - u.x1), abs(b.y1 - u.y1)) < 3.0
                   for u in uniq):
            uniq.append(b)
    return uniq


def _side_line(box: _Box, side: str, lines: list[_Seg]) -> Optional[_Seg]:
    """The axis line lying along `side` of `box`, if any."""
    o = "h" if side in ("bottom", "top") else "v"
    c = {"bottom": box.y1, "top": box.y0, "left": box.x0, "right": box.x1}[side]
    lo, hi = (box.x0, box.x1) if o == "h" else (box.y0, box.y1)
    best, best_ov = None, 0.0
    for s in lines:
        if s.o != o or abs(s.c - c) > max(2.0, 2 * s.t):
            continue
        ov = (min(hi, s.hi) - max(lo, s.lo)) / (hi - lo)
        if ov > 0.8 and ov > best_ov:
            best, best_ov = s, ov
    return best


# --------------------------------------------------------------------------- 4. ticks
@dataclass
class _Ticks:
    major: list[float]
    minor: list[float]
    out_len: float        # how far ticks stick out of the box (0 for inward ticks)
    from_grid: bool = False


def _side_geom(box: _Box, side: str):
    """(fixed coord of the side, along-range, outward sign in page coords, tick orientation)."""
    if side == "bottom":
        return box.y1, (box.x0, box.x1), +1, "v"
    if side == "top":
        return box.y0, (box.x0, box.x1), -1, "v"
    if side == "left":
        return box.x0, (box.y0, box.y1), -1, "h"
    return box.x1, (box.y0, box.y1), +1, "h"


def _find_ticks(box: _Box, side: str, segs: list[_Seg], line_t: float) -> _Ticks:
    C, (a, b), out, to = _side_geom(box, side)
    depth = box.h if to == "v" else box.w            # extent of the box across this side
    max_tick = min(20.0, 0.3 * depth)
    tt = max(1.0, 1.5 * line_t)
    cand = []                                         # (position, length, outward length)
    grid = []
    for s in segs:
        if s.o != to or not (a - 1.0 <= s.c <= b + 1.0):
            continue
        if not (s.lo <= C + tt and s.hi >= C - tt):   # must touch the side line
            continue
        out_len = max(0.0, s.hi - C) if out > 0 else max(0.0, C - s.lo)
        if s.length <= max_tick and s.length >= 0.3:
            cand.append((s.c, s.length, out_len))
        elif s.length >= 0.9 * depth:
            grid.append(s.c)
    if not cand:
        # no tick marks: grid lines (if any) mark the tick positions
        g = sorted(set(round(x, 2) for x in grid if a + 0.5 < x < b - 0.5))
        return _Ticks(g, [], 0.0, from_grid=bool(g))
    cand.sort()
    # the same tick drawn twice (or tick + its outline) -> keep the longest
    dedup = []
    for c in cand:
        if dedup and abs(c[0] - dedup[-1][0]) < 0.4:
            if c[1] > dedup[-1][1]:
                dedup[-1] = c
        else:
            dedup.append(c)
    lengths = np.array([c[1] for c in dedup])
    # major length = longest length shared by >= 2 ticks (a stray long piece must not set it)
    rounded = np.round(lengths, 1)
    vals, counts = np.unique(rounded, return_counts=True)
    shared = vals[counts >= 2]
    lm = float(shared.max()) if len(shared) else float(lengths.max())
    major = [c[0] for c in dedup if abs(c[1] - lm) <= 0.25 * lm + 0.05]
    minor = [c[0] for c in dedup if c[1] < 0.75 * lm and c[0] not in major]
    out_len = max([c[2] for c in dedup if c[0] in major] + [0.0])
    return _Ticks(major, minor, out_len)


# --------------------------------------------------------------------------- 5. labels
_MINUS = dict.fromkeys(map(ord, "−–—‒﹣－‐‑"), "-")
_NUM = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def parse_value(s: str) -> Optional[float]:
    """Tick-label text -> float, or None.  Handles U+2212 minus, '10^-3', '2×10^4', '1e-4', '50%'."""
    s = s.translate(_MINUS).replace(" ", "").replace(" ", "").rstrip("%")
    s = s.replace("×", "x").replace("⋅", "x").replace("*", "x")
    s = s.replace("{", "").replace("}", "").replace("$", "")
    if not s:
        return None
    m = re.fullmatch(r"([+-]?)10\^([+-]?\d+(?:\.\d+)?)", s)
    if m:
        return (-1 if m.group(1) == "-" else 1) * 10.0 ** float(m.group(2))
    m = re.fullmatch(r"([+-]?(?:\d+\.?\d*|\.\d+))x10\^([+-]?\d+)", s)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    if re.fullmatch(r"[+-]?\d+,\d{1,2}", s) or re.fullmatch(r"[+-]?\d+,\d{4,}", s):
        s = s.replace(",", ".")              # decimal comma ("0,5"); "1,000" stays a thousands sep
    s = s.replace(",", "")
    if _NUM.fullmatch(s):
        return float(s)
    return None


def _text_words(pc: PageContent) -> list[Word]:
    """Horizontal text spans -> words; superscript spans are glued on as '^...'."""
    spans = [t for t in pc.texts if abs(t.dir[0] - 1) < 0.05 and abs(t.dir[1]) < 0.05]
    spans.sort(key=lambda t: t.bbox[0])
    words: list[Word] = []
    sizes: list[float] = []      # font size of each word's first span
    bases: list[float] = []      # baseline (origin y) of each word's first span
    ends: dict[int, list[int]] = {}   # int(right edge) -> word indices, to find neighbours fast
    for t in spans:
        text = t.text
        glued = None
        # is there a word ending right where this span starts?
        cands = [k for e in range(int(t.bbox[0] - 6) - 1, int(t.bbox[0] + 0.5) + 1)
                 for k in ends.get(e, [])]
        for k in cands:
            w, size, base = words[k], sizes[k], bases[k]
            gap = t.bbox[0] - w.bbox[2]
            same_row = t.bbox[1] < w.bbox[3] and t.bbox[3] > w.bbox[1]
            if not (same_row and -0.5 < gap <= 0.3 * size and text.strip() == text):
                continue
            rise = base - t.origin[1]                      # > 0: this span sits higher
            if rise > 0.2 * size and t.size <= size and "^" not in w.text:
                glued = w.text + "^" + text                # 10 + superscript -3
            elif abs(t.size - size) < 0.15 * size and abs(rise) < 0.1 * size:
                glued = w.text + text                      # '−' and '20' in different fonts
            if glued is not None:
                ends[int(w.bbox[2])].remove(k)
                words[k] = Word(glued, (w.bbox[0], min(w.bbox[1], t.bbox[1]), t.bbox[2],
                                        max(w.bbox[3], t.bbox[3])), 1.0, "text")
                ends.setdefault(int(t.bbox[2]), []).append(k)
                break
        if glued is not None:
            continue
        # one span may hold several labels ("0.0 0.2 0.4"): split, estimating char widths
        x0, y0, x1, y1 = t.bbox
        n = max(len(text), 1)
        for m in re.finditer(r"\S+", text):
            bx0 = x0 + (x1 - x0) * m.start() / n
            bx1 = x0 + (x1 - x0) * m.end() / n
            ends.setdefault(int(bx1), []).append(len(words))
            words.append(Word(m.group(), (bx0, y0, bx1, y1), 1.0, "text"))
            sizes.append(t.size)
            bases.append(t.origin[1])
    return words


def _zone(box: _Box, side: str, ticks: _Ticks) -> BBox:
    """Band outside `side` where its tick labels must be."""
    C, (a, b), out, to = _side_geom(box, side)
    start = C + out * (ticks.out_len + 0.3)
    depth = 26.0 if to == "v" else 48.0              # y labels can be wide ("0.0001")
    far = start + out * depth
    if to == "v":
        return (a - 30, min(start, far), b + 30, max(start, far))
    return (min(start, far), a - 12, max(start, far), b + 12)


def _in(w: Word, z: BBox) -> bool:
    """Centre inside `z` and the box not sticking out much (text boxes include line spacing)."""
    m = 0.4 * w.h
    return (z[0] <= w.cx <= z[2] and z[1] <= w.cy <= z[3]
            and w.bbox[0] >= z[0] - m and w.bbox[2] <= z[2] + m and w.bbox[1] >= z[1] - m and w.bbox[3] <= z[3] + m)


def _nearest_row(words: list[Word], side: str, start: float) -> list[Word]:
    """Keep the row (x axis) / column (y axis) of labels closest to the axis."""
    if not words:
        return []
    def gap(w):
        return {"bottom": w.bbox[1] - start, "top": start - w.bbox[3],
                "left": start - w.bbox[2], "right": w.bbox[0] - start}[side]
    g = np.array([gap(w) for w in words])
    hmed = float(np.median([w.h for w in words]))
    if g.min() > max(6.0, 1.2 * hmed):
        return []          # too far out: these belong to a neighbouring plot
    keep = g <= g.min() + max(2.0, 0.7 * hmed)
    return [w for w, k in zip(words, keep) if k]


def _pair_labels(words: list[Word], side: str, ticks: _Ticks):
    """Attach each numeric label to a tick -> list of (page_pos, value, word)."""
    along = (lambda w: w.cx) if side in ("bottom", "top") else (lambda w: w.cy)
    major = sorted(ticks.major)
    alltk = sorted(set(ticks.major) | set(ticks.minor))
    def spacing(xs):
        return float(np.median(np.diff(xs))) if len(xs) >= 2 else math.inf
    sp_major, sp_all = spacing(major), spacing(alltk)
    best: dict[float, tuple[float, float, Word]] = {}
    for w in words:
        v = parse_value(w.text)
        if v is None:
            continue
        pos = along(w)
        if not alltk:                                 # no tick marks: label centre is all we have
            best[round(pos, 3)] = (0.0, v, w)
            continue
        tick = None
        if major:
            t = min(major, key=lambda x: abs(x - pos))
            if abs(t - pos) <= 0.3 * min(sp_major, 60.0):
                tick = t
        if tick is None:
            t = min(alltk, key=lambda x: abs(x - pos))
            if abs(t - pos) <= 0.45 * min(sp_all, 40.0):
                tick = t
        if tick is None:
            continue
        d = abs(tick - pos)
        if tick not in best or d < best[tick][0]:
            best[tick] = (d, v, w)
    return [(p, v, w) for p, (d, v, w) in sorted(best.items())]


def _log_minor_ticks(ticks: _Ticks) -> bool:
    """Do the minor ticks sit at log10(2..9) between two majors (a log axis)?"""
    major = sorted(ticks.major)
    for m1, m2 in zip(major, major[1:]):
        inside = [t for t in ticks.minor if min(m1, m2) < t < max(m1, m2)]
        if len(inside) < 4:
            continue
        d = m2 - m1
        fr = sorted((t - m1) / d for t in inside)
        if d < 0:
            fr = sorted(1 - f for f in fr)
        expect = np.log10(np.arange(2, 10))
        hits = sum(np.min(np.abs(expect - f)) < 0.03 for f in fr)
        return hits >= 0.75 * len(fr) and len(fr) <= 9
    return False


# --------------------------------------------------------------------------- 6. fitting
def _try_fit(orient, pairs, scale, tol, **kw) -> Optional[AxisCalib]:
    if len(pairs) < 2:
        return None
    vals = [v for _, v in pairs]
    if scale == "log" and min(vals) <= 0:
        return None
    if len(set(vals)) != len(vals) or len(set(p for p, _ in pairs)) != len(pairs):
        return None
    c = AxisCalib.fit(orient, pairs, scale=scale, **kw)
    if c.a == 0 or not math.isfinite(c.a):
        return None
    # judge the fit against the SMALLEST label step: AxisCalib.residual uses the median
    # step, which is lenient when the labelled values are unevenly spaced (0, 0.5, 2)
    y = np.log10(vals) if scale == "log" else np.asarray(vals, float)
    step = float(np.min(np.abs(np.diff(np.sort(y)))))
    err = float(np.max(np.abs(c.a * np.array([p for p, _ in pairs]) + c.b - y)))
    if not step or err / step > tol:
        return None
    return c


def fit_axis(orient, pairs, scale=None, log_hint=False, tol=FIT_TOL, **kw):
    """Robust calibration from (pos, value) pairs.  Returns (AxisCalib | None, notes)."""
    pairs = sorted(pairs)
    scales = [scale] if scale else (["log", "linear"] if log_hint else ["linear", "log"])
    for sc in scales:
        c = _try_fit(orient, pairs, sc, tol, **kw)
        if c is not None:
            return c, []
    # repair one bad label: a lost minus sign first, then dropping it.  Only with enough
    # labels left over to still prove the progression (3 points can fit almost anything
    # after a sign flip or a drop).
    n = len(pairs)
    if n >= 4:
        for sc in scales:
            for i in range(n):
                p, v = pairs[i]
                if v != 0:
                    trial = pairs[:i] + [(p, -v)] + pairs[i + 1:]
                    c = _try_fit(orient, trial, sc, tol, **kw)
                    if c is not None:
                        return c, [f"{orient}: tick label {v:g} read as {-v:g} (sign repaired)"]
    if n >= 5:
        for sc in scales:
            fits = []
            for i in range(n):
                c = _try_fit(orient, pairs[:i] + pairs[i + 1:], sc, tol, **kw)
                if c is not None:
                    fits.append((c.residual, i, c))
            if fits:
                _, i, c = min(fits, key=lambda f: f[0])
                return c, [f"{orient}: dropped inconsistent tick label {pairs[i][1]:g}"]
    return None, []


# --------------------------------------------------------------------------- 7. one side
@dataclass
class _SideResult:
    calib: Optional[AxisCalib]
    notes: list
    ticks: _Ticks
    words: list            # label words used
    zone: BBox


def _has_ink(pc: PageContent, zone: BBox) -> bool:
    """Anything drawn or written inside `zone`?  (Pixel OCR is slow: skip empty zones.)"""
    def inside(b):
        return b[0] >= zone[0] and b[2] <= zone[2] and b[1] >= zone[1] and b[3] <= zone[3]
    return any(inside(t.bbox) for t in pc.texts) or any(inside(p.bbox) for p in pc.paths)


def _ocr_zone(page, zone, notes) -> list[Word]:
    notes.append("ocr-called")          # tells the caller to charge the OCR budget
    return ocr.ocr_region(page, zone, dpi=400)


def _read_side(box, side, segs, pc, page, text_words, hint_vals=None, scale=None,
               allow_ocr=True) -> _SideResult:
    line = box.sides.get(side)
    ticks = _find_ticks(box, side, segs, line.t if line else 0.5)
    zone = _zone(box, side, ticks)
    orient = "x" if side in ("bottom", "top") else "y"
    notes: list[str] = []
    if ticks.from_grid:
        notes.append(f"{orient}: no tick marks on {side}; positions from grid lines")

    if hint_vals is not None:                         # ---- manual values
        if hint_vals and isinstance(hint_vals[0], (tuple, list)):
            pairs = [(float(p), float(v)) for p, v in hint_vals]
        else:
            vals = [float(v) for v in hint_vals]
            tk = sorted(ticks.major)
            if len(tk) != len(vals):
                tk = sorted(set(ticks.major) | set(ticks.minor))
            if len(tk) != len(vals):
                return _SideResult(None, [f"{orient}: hint gives {len(vals)} ticks but {len(ticks.major)} "
                                          f"major ticks found on {side}"], ticks, [], zone)
            if orient == "y":
                tk = tk[::-1]                         # values are given bottom -> top
            pairs = list(zip(tk, vals))
        c, n2 = fit_axis(orient, pairs, scale, tol=FIT_TOL_LOOSE, source="manual")
        return _SideResult(c, notes + n2, ticks, [], zone)

    start = _side_geom(box, side)[0] + _side_geom(box, side)[2] * (ticks.out_len + 0.3)
    sources = [("text", lambda: [w for w in text_words if _in(w, zone)]),
               ("glyph", lambda: ocr.glyph_words(pc, zone)),
               ("render", lambda: ocr.rendered_words(page, zone)
                if any(ocr.unreadable(t.text) and zone[0] <= t.cx <= zone[2] and zone[1] <= t.cy <= zone[3]
                       for t in pc.texts) else []),
               ("ocr", lambda: _ocr_zone(page, zone, notes)
                if page is not None and allow_ocr and len(ticks.major) >= 3 and _has_ink(pc, zone) else [])]
    for src, get in sources:
        words = get()
        if src == "ocr":
            # pixel OCR flattens superscripts: "10" + "15" comes back as "1015", and a log
            # axis 10^0..10^3 would read as a perfectly linear 100, 101, 102, 103.  When most
            # labels look like that, read them all as powers of ten.
            pw = [w for w in words if re.fullmatch(r"10[-−]?\d{1,2}", w.text)]
            num = [w for w in words if parse_value(w.text) is not None]
            if len(pw) >= 2 and len(pw) >= 0.5 * len(num):
                words = [Word("10^" + w.text[2:], w.bbox, w.score, w.source) for w in pw]
        words = [w for w in words if parse_value(w.text) is not None and _in(w, zone)]
        words = _nearest_row(words, side, start)
        if len(words) < 2:
            continue
        pairs3 = _pair_labels(words, side, ticks)
        if len(pairs3) < 2:
            continue
        has_ticks = bool(ticks.major or ticks.minor)
        if not has_ticks and len(pairs3) < 3:
            continue        # two label centres prove nothing (e.g. a legend box near labels)
        log_hint = any("^" in w.text for _, _, w in pairs3) or _log_minor_ticks(ticks)
        vals = sorted(v for _, v, _ in pairs3)
        inferred = []
        if (len(vals) == 2 and vals[0] > 0 and math.isclose(vals[1] / vals[0], 10.0)
                and math.isclose(math.log10(vals[0]), round(math.log10(vals[0])), abs_tol=1e-9)):
            # just "1" and "10": nobody labels a linear axis like that
            log_hint = True
            inferred.append(f"{orient}: log axis inferred from two labels {vals[0]:g}, {vals[1]:g}")
        c, n2 = fit_axis(orient, [(p, v) for p, v, _ in pairs3], scale, log_hint,
                         tol=FIT_TOL if has_ticks else FIT_TOL_LOOSE,
                         source="text" if src == "text" else "ocr")
        if c is None:
            continue
        if not has_ticks:
            n2.append(f"{orient}: no tick marks; positions from label centres")
            # label centres stand in for ticks: positions are less exact
        if src == "glyph":
            n2.append(f"{orient} calibrated from outlined glyphs")
        elif src == "render":
            n2.append(f"{orient} calibrated from rendered glyphs of an unmapped font")
        elif src == "ocr":
            n2.append(f"{orient} calibrated via OCR")
        if c.scale == "log":
            n2 += inferred or [f"{orient}: log axis"]
        return _SideResult(c, notes + n2, ticks, [w for _, _, w in pairs3], zone)
    return _SideResult(None, notes, ticks, [], zone)


# --------------------------------------------------------------------------- 8. extras
_MULT = re.compile(r"^(?:[x×]\s*10\^?([+-]?\d+)|1e([+-]?\d+))$")


def _multiplier(words: list[Word], used: list[Word], region: BBox,
                pc: PageContent) -> Optional[tuple[float, str]]:
    """Offset text like '1e−4' or '×10^3' inside `region` -> (factor, text).

    Outlined glyphs are tried too; there the '×' is an unknown glyph, so only
    '?10^n' is accepted (a bare '10^n' could be the top label of a log axis).
    """
    def overlaps_label(w):
        return any(w.bbox[0] < u.bbox[2] and u.bbox[0] < w.bbox[2] and w.bbox[1] < u.bbox[3]
                   and u.bbox[1] < w.bbox[3] for u in used)
    for w in words:
        if not _in(w, region) or overlaps_label(w):
            continue
        m = _MULT.match(w.text.translate(_MINUS).replace(" ", ""))
        if m:
            e = next(g for g in m.groups() if g is not None)
            return 10.0 ** int(e), w.text
    for w in ocr.glyph_words(pc, region):
        m = re.fullmatch(r"\?10\^(-?\d+)", w.text)
        if m and not overlaps_label(w):
            return 10.0 ** int(m.group(1)), "\u00d7" + w.text[1:]
    return None


def _scaled(c: AxisCalib, k: float) -> AxisCalib:
    return AxisCalib(c.orient, c.scale, c.a * k, c.b * k, [(p, v * k) for p, v in c.ticks],
                     c.residual, c.label, c.source)


def _join(spans, vertical: bool = False) -> Optional[str]:
    """Spans of one title line -> text, inserting a space where there is a visible gap."""
    if not spans:
        return None
    if vertical:   # rotated 90 deg CCW: reading order is bottom -> top
        spans = sorted(spans, key=lambda t: -t.bbox[3])
        gaps = [spans[i - 1].bbox[1] - spans[i].bbox[3] for i in range(1, len(spans))]
    else:
        spans = sorted(spans, key=lambda t: t.bbox[0])
        gaps = [spans[i].bbox[0] - spans[i - 1].bbox[2] for i in range(1, len(spans))]
    text = spans[0].text
    for s, g in zip(spans[1:], gaps):
        sep = " " if g > 0.15 * s.size and not text.endswith(" ") and not s.text.startswith(" ") else ""
        text += sep + s.text
    return text.strip() or None


def _titles(pc: PageContent, box: _Box, xr: Optional[_SideResult],
            yr: Optional[_SideResult]) -> tuple[Optional[str], Optional[str]]:
    """Axis titles (real text only; outlined titles stay None)."""
    # x title: horizontal text under the tick labels, overlapping the box horizontally
    below = max([w.bbox[3] for w in (xr.words if xr else [])] + [box.y1])
    xs = [t for t in pc.texts if abs(t.dir[0] - 1) < 0.05 and below - 1 <= t.bbox[1] <= below + 22
          and not ocr.unreadable(t.text)
          and box.x0 - 0.1 * box.w < t.cx < box.x1 + 0.1 * box.w
          and t.bbox[2] - t.bbox[0] < 1.3 * box.w]           # not a caption line
    if xs:
        row = min(t.bbox[1] for t in xs)
        xs = [t for t in xs if t.bbox[1] < row + 0.6 * max(t.size for t in xs)]
        # a line that runs far past the plot is a caption / body text, not an axis title
        oy = xs[0].origin[1]
        same = sorted((t for t in pc.texts if abs(t.dir[0] - 1) < 0.05 and abs(t.origin[1] - oy) < 1.0),
                      key=lambda t: t.bbox[0])
        line, x_end = [], -math.inf          # the run of spans (small gaps) containing xs[0]
        for t in same:
            if line and t.bbox[0] - x_end > 1.5 * t.size:
                if xs[0] in line:
                    break
                line = []
            line.append(t)
            x_end = max(x_end, t.bbox[2])
        if min(t.bbox[0] for t in line) < box.x0 - 0.25 * box.w or max(t.bbox[2] for t in line) > box.x1 + 0.25 * box.w:
            xs = []
    # y title: text rotated 90 deg, left of the tick labels
    left = min([w.bbox[0] for w in (yr.words if yr else [])] + [box.x0])
    ys = [t for t in pc.texts if abs(t.dir[0]) < 0.05 and abs(abs(t.dir[1]) - 1) < 0.05
          and not ocr.unreadable(t.text)
          and left - 25 <= t.bbox[0] and t.bbox[2] <= left + 1
          and t.bbox[3] > box.y0 and t.bbox[1] < box.y1]
    if ys:
        col = max(t.bbox[2] for t in ys)
        ys = [t for t in ys if t.bbox[2] > col - 0.8 * max(t.size for t in ys) - 2]
    return _join(xs), _join(ys, vertical=True)


_LETTER = re.compile(r"^\(?([a-hA-H])\)?[.)]?$")


def _panel_letter(pc: PageContent, box: _Box, left: float) -> Optional[str]:
    """Letter like "a", "(b)", "c)" near the top-left corner (text, else outlined glyphs)."""
    region = (left - 45, box.y0 - 35, box.x0 + 0.25 * box.w, box.y0 + 0.15 * box.h)
    cands = []
    for t in pc.texts:
        m = _LETTER.match(t.text.strip())
        if m and t.size >= 6 and region[0] <= t.bbox[0] <= region[2] and region[1] <= t.bbox[1] <= region[3]:
            cands.append((t.bbox, m.group(1)))
    if not cands:
        cands = [(w.bbox, w.text) for w in ocr.glyph_letters(pc, region)]
    if not cands:
        return None
    bb, letter = min(cands, key=lambda c: math.hypot(c[0][0] - box.x0, c[0][1] - box.y0))
    return letter.lower()


# --------------------------------------------------------------------------- 9. driver
def _iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _same_calib(c1: AxisCalib, c2: AxisCalib, lo: float, hi: float) -> bool:
    v1, v2 = c1.to_data([lo, hi]), c2.to_data([lo, hi])
    span = abs(v1[1] - v1[0]) or 1.0
    return bool(np.all(np.abs(v1 - v2) <= 0.01 * span)) and c1.scale == c2.scale


def _read_box(box, segs, pc, page, text_words, hints, res: dict, ocr_budget: Optional[list]) -> None:
    """Read the sides of `box` that are not calibrated yet into `res` (side -> _SideResult).

    `ocr_budget` is a one-element list counting the pixel-OCR calls still allowed on
    this page (None: no pixel OCR).  OCR costs ~1 s per call, so it is rationed.
    """
    ok = lambda s: s in res and res[s].calib is not None

    def read(side, allow_ocr=True):
        if ok(side):
            return
        use_ocr = allow_ocr and ocr_budget is not None and ocr_budget[0] > 0
        hv = hints.get("x_ticks") if side in ("bottom", "top") else hints.get("y_ticks")
        sc = hints.get("xscale") if side in ("bottom", "top") else hints.get("yscale")
        r = _read_side(box, side, segs, pc, page, text_words, hv, sc, use_ocr)
        if "ocr-called" in r.notes:
            ocr_budget[0] -= 1
            r.notes = [n for n in r.notes if n != "ocr-called"]
        res[side] = r

    # bottom/left are the usual axes; top/right are read when those fail, or (cheaply,
    # without pixel OCR) to spot a twin y axis
    read("bottom")
    if not ok("bottom") and box.sides.get("top") is not None:
        read("top")
    read("left")
    if box.sides.get("right") is not None:
        read("right", allow_ocr=not ok("left"))


def _axes_of(res: dict) -> tuple[Optional[str], Optional[str]]:
    """Which side calibrates x and which y (None if neither candidate worked)."""
    ok = lambda s: s in res and res[s].calib is not None
    xs = "bottom" if ok("bottom") else ("top" if ok("top") else None)
    ys = "left" if ok("left") else ("right" if ok("right") else None)
    return xs, ys


def _make_panel(box, res, pc, text_words, shared: Optional[dict] = None) -> Panel:
    """Assemble a Panel from calibrated sides (`shared` = calibration borrowed from a neighbour)."""
    xs, ys = _axes_of(res)
    shared = shared or {}
    ok = lambda s: s in res and res[s].calib is not None
    notes = []
    for side in (xs, ys):
        if side is not None:
            notes += res[side].notes
    if xs == "top":
        notes.append("x calibrated from the top axis")
    if ys == "right":
        notes.append("y calibrated from the right axis")
    if ys == "left" and ok("right") and not _same_calib(res["left"].calib, res["right"].calib, box.y0, box.y1):
        r = res["right"].calib
        notes.append(f"twin y axis on the right ({r.to_data(box.y1):.4g}..{r.to_data(box.y0):.4g}) ignored")
    notes += shared.get("notes", [])
    xr = res[xs] if xs else None
    yr = res[ys] if ys else None
    xc = xr.calib if xr else shared["x"]
    yc = yr.calib if yr else shared["y"]

    # offset / multiplier texts (matplotlib "1e−4" above the y axis, at the right end of x)
    used = [w for r in res.values() for w in r.words]
    if yr is not None:
        m = _multiplier(text_words, used, (box.x0 - 40, box.y0 - 22, box.x0 + 0.5 * box.w, box.y0 + 1), pc)
        if m and yc.source != "manual":
            yc = _scaled(yc, m[0])
            notes.append(f"y multiplied by {m[0]:g} (offset text '{m[1]}')")
    if xr is not None:
        xb = max([w.bbox[3] for w in xr.words] + [box.y1])
        m = _multiplier(text_words, used, (box.x1 - 0.5 * box.w, box.y1, box.x1 + 40, xb + 22), pc)
        if m and xc.source != "manual":
            xc = _scaled(xc, m[0])
            notes.append(f"x multiplied by {m[0]:g} (offset text '{m[1]}')")
    xt, yt = _titles(pc, box, xr, yr)
    if xr is not None:
        xc.label = xt
    if yr is not None:
        yc.label = yt
    for im in pc.images:
        if _iou(im, box.bbox) > 0.5 or (im[0] <= box.x0 and im[1] <= box.y0 and im[2] >= box.x1 and im[3] >= box.y1):
            notes.append("plot area is covered by a raster image")
            break
    left = min([w.bbox[0] for w in (yr.words if yr else [])] + [box.x0])
    return Panel(pc.page_no, box.bbox, xc, yc, label=_panel_letter(pc, box, left), notes=notes)


def _copy_calib(c: AxisCalib) -> AxisCalib:
    return AxisCalib(c.orient, c.scale, c.a, c.b, list(c.ticks), c.residual, c.label, c.source)


def _borrow(box: _Box, res: dict, made: list) -> Optional[dict]:
    """Shared axes: a plot without y labels right next to one with the same y extent
    reuses its y calibration (likewise x, for plots stacked above each other)."""
    xs, ys = _axes_of(res)
    # the plots must (nearly) touch: a gap wide enough for tick labels means the
    # unread axis has labels of its own, which must not be replaced by the neighbour's
    gap = 10.0
    for b, p in made:
        same_rows = abs(b.y0 - box.y0) < 1.0 and abs(b.y1 - box.y1) < 1.0
        same_cols = abs(b.x0 - box.x0) < 1.0 and abs(b.x1 - box.x1) < 1.0
        if xs and not ys and same_rows and (0 <= box.x0 - b.x1 < gap or 0 <= b.x0 - box.x1 < gap):
            return {"y": _copy_calib(p.y), "notes": ["y shared with the neighbouring plot"]}
        if ys and not xs and same_cols and (0 <= box.y0 - b.y1 < gap or 0 <= b.y0 - box.y1 < gap):
            return {"x": _copy_calib(p.x), "notes": ["x shared with the neighbouring plot"]}
    return None


OCR_BUDGET = 4       # max pixel-OCR calls per page (~1 s each)


def find_panels(pc: PageContent, page=None, hints: Optional[dict] = None) -> list[Panel]:
    """Every calibratable x-y plot on the page (see module docstring)."""
    hints = hints or {}
    segs = _segments(pc)
    lines = _merge_lines(segs)
    lines = _split_at_dividers(lines, segs)
    boxes = _corner_boxes(lines)
    hb = tuple(hints["bbox"]) if hints.get("bbox") else None
    if hb is not None:
        cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
        boxes = [b for b in boxes if _iou(b.bbox, hb) > 0.5 or (b.x0 < cx < b.x1 and b.y0 < cy < b.y1
                                                                and _iou(b.bbox, hb) > 0.2)]
        if not boxes:
            boxes = [_Box(*hb)]
    for b in boxes:
        for side in SIDES:
            b.sides[side] = _side_line(b, side, lines)
        # snap the box to the centre lines of the axes it has (outlined frames overshoot)
        for side, attr in (("left", "x0"), ("right", "x1"), ("top", "y0"), ("bottom", "y1")):
            if b.sides[side] is not None:
                setattr(b, attr, b.sides[side].c)
    text_words = _text_words(pc)

    # pass 1: text and outlined glyphs only (fast); pass 2: pixel OCR for what is left
    results = [{} for _ in boxes]
    for b, res in zip(boxes, results):
        _read_box(b, segs, pc, page, text_words, hints, res, None)
    if page is not None and ocr.ocr_available():
        budget = [OCR_BUDGET]
        for b, res in zip(boxes, results):
            if None in _axes_of(res):
                _read_box(b, segs, pc, page, text_words, hints, res, budget)

    made: list[tuple[_Box, Panel]] = []
    for b, res in zip(boxes, results):
        if None not in _axes_of(res):
            made.append((b, _make_panel(b, res, pc, text_words)))
    # pass 3: plots that share an axis with a neighbour (only one of their axes labelled)
    for b, res in zip(boxes, results):
        xs, ys = _axes_of(res)
        if (xs is None) != (ys is None):
            sh = _borrow(b, res, made)
            if sh is not None:
                made.append((b, _make_panel(b, res, pc, text_words, sh)))

    panels: list[Panel] = []
    for _, p in made:
        nt = len(p.x.ticks) + len(p.y.ticks)
        dup = next((q for q in panels if _iou(q.bbox, p.bbox) > 0.8), None)
        if dup is not None:
            if nt > len(dup.x.ticks) + len(dup.y.ticks):
                panels[panels.index(dup)] = p
            continue
        panels.append(p)
    panels.sort(key=lambda p: (round(p.bbox[1] / 20), p.bbox[0]))
    return panels
