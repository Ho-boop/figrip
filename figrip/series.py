"""Find the data series inside one plot area.

``extract_series(pc, panel) -> list[Series]``

Input is everything drawn on the page (``PageContent``) plus the plot area
``panel.bbox``.  We work purely in page coordinates; the pipeline converts
``Series.page_pts`` to data values with ``panel.to_data`` (we fill ``data`` too).

The difficulty is not reading coordinates (they are exact in a vector PDF) but
deciding *which* drawings are data.  A plot area also contains the frame,
ticks, grid and zero lines, a legend, text that may be drawn as glyph
outlines, error bars ... and papers draw curves in very different ways:

* matplotlib: one stroked path per line (maybe dashed), markers as small
  closed paths, all in the series colour.
* Origin: often one stroked or *filled* path per line segment.
* "outlined strokes" (Illustrator / publisher pipelines): every line is a
  thin filled polygon running along one side of the curve and back along the
  other side; its centreline must be recovered.

Algorithm (each step is one function below)
-------------------------------------------
1. ``_decompose``   every path near the panel becomes either a *piece* (a
   polyline that may be part of a curve: stroked subpaths, and the centreline
   of thin filled polygons, see ``_outline_to_line``) or a *blob* (a small
   compact closed shape: marker or glyph).
2. ``_chain``       pieces of the same colour whose ends touch are joined into
   ordered polylines ("chains").  Origin's segment-by-segment curves and
   matplotlib's split paths become one polyline here.  Curves of the *same*
   style stay separate unless they touch (Zheng 2018: three black curves).
3. ``_classify_blob`` blobs get a marker shape (circle, square, triangle...)
   or "other"; ``_merge_concentric`` merges a marker drawn in layers (filled
   body + outlined edge, Origin's shaded "3D balls"); ``_glyph_runs`` finds
   rows of glyph-sized shapes that contain at least two glyph-like ("other")
   members: that is outlined text -> removed.  A row of identical markers is
   never text.
4. ``_find_legend`` a legend entry = short horizontal sample line and/or one
   marker just left of a label (real text, or a glyph run).  Samples are
   removed from the data and remembered for naming.
5. ``_is_decoration`` & co.: frame, grid lines, zero/reference lines spanning
   the plot, boxes, ticks, error bars and annotation arrows are dropped, and
   so is everything inside an inset/legend box.  Default for reference lines
   (e.g. J = 0): *excluded* - any straight horizontal/vertical line spanning
   >= 60 % of the plot is treated as axis furniture, never as data.
6. a second ``_chain`` pass bridges small gaps (dashes of outlined dashed
   lines, Origin's "gap to symbol") when the directions line up.
7. clip everything to the plot area (+0.5 pt).  A curve leaving the plot ends
   at the frame (the clipped end point is interpolated on the last segment);
   a curve that leaves and re-enters is concatenated into one polyline.
8. ``_assemble``     markers sitting on the vertices of a line of the same
   colour make that line "line+markers"; other marker groups (>= 3 markers)
   become "markers" series; ``_name_series`` matches legend entries by style.

Series.style keys (documented contract)
---------------------------------------
``color`` (r,g,b) | ``width`` line width in pt (estimated for outlined lines,
None for marker-only) | ``dashes`` PDF dash string or None | ``outlined``
True if the line was a filled outline | ``marker`` shape name or None
("circle", "square", "diamond", "triangle_up/down/left/right", "pentagon",
"hexagon", "star", "plus", "x") | ``marker_size`` pt | ``marker_filled`` bool |
``marker_pts`` (M,2) page coords of the marker centres, also for
"line+markers" (where ``page_pts`` holds the line vertices, which usually
include more points than the markers, e.g. matplotlib ``markevery``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .model import BBox, Color, PageContent, Panel, Series

# --------------------------------------------------------------------------- tunables
SNAP = 0.08             # pt: ends closer than this are "the same point"
STRAIGHT_TOL = 0.15     # pt: max deviation from the chord for a "straight" chain
AXIS_TOL = 0.02         # rad (~1.1 deg): "horizontal"/"vertical"
COLOR_TOL = 0.06        # max abs channel difference for "same colour"
SPAN_FRAC = 0.6         # straight axis-aligned lines longer than this x plot size = grid/ref
MARKER_SHAPES = ("circle", "square", "diamond", "triangle_up", "triangle_down",
                 "triangle_left", "triangle_right", "pentagon", "hexagon", "star", "plus", "x")


# =========================================================================== small geometry helpers
def _area(r: np.ndarray) -> float:
    """Signed shoelace area of a closed ring (last point need not repeat the first)."""
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _seglen(p: np.ndarray) -> np.ndarray:
    return np.hypot(*np.diff(p, axis=0).T) if len(p) > 1 else np.zeros(0)


def _length(p: np.ndarray) -> float:
    return float(_seglen(p).sum())


def _ring(sub: np.ndarray) -> np.ndarray:
    """Closed subpath -> ring without repeated vertices and without the closing duplicate."""
    keep = np.r_[True, np.hypot(*np.diff(sub, axis=0).T) > 1e-6]
    r = sub[keep]
    if len(r) > 1 and np.hypot(*(r[0] - r[-1])) < 1e-6:
        r = r[:-1]
    return r


def _dedupe(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(p) < 2:
        return p
    return p[np.r_[True, np.hypot(*np.diff(p, axis=0).T) > eps]]


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull (Andrew's monotone chain), counter-clockwise in math orientation."""
    p = np.unique(np.round(pts, 6), axis=0)
    if len(p) < 3:
        return p

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 1e-12:
                    out.pop()
                else:
                    break
            out.append(q)
        return out
    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _rdp(p: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of an open polyline."""
    if len(p) < 3:
        return p
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d = _dist_to_line(p[i + 1:j], p[i], p[j])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return p[keep]


def _dist_to_line(pts, a, b) -> np.ndarray:
    """Distance of points to the infinite line a-b (to point a if a == b)."""
    ab = b - a
    n = math.hypot(*ab)
    if n < 1e-12:
        return np.hypot(*(pts - a).T)
    return np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n


def _clip_polyline(p: np.ndarray, box: BBox) -> list[np.ndarray]:
    """Liang-Barsky clip of a polyline to a rectangle -> list of visible pieces."""
    x0, y0, x1, y1 = box
    pieces: list[list] = []
    cur: list = []
    for a, b in zip(p[:-1], p[1:]):
        d = b - a
        t0, t1 = 0.0, 1.0
        ok = True
        for q, r in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
            if abs(q) < 1e-12:
                if r < 0:
                    ok = False
                    break
                continue
            t = r / q
            if q < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
        if not ok or t0 > t1:
            if cur:
                pieces.append(cur)
                cur = []
            continue
        pa, pb = a + t0 * d, a + t1 * d
        if cur and np.hypot(*(cur[-1] - pa)) < 1e-9:
            cur.append(pb)
        else:
            if cur:
                pieces.append(cur)
            cur = [pa, pb]
    if cur:
        pieces.append(cur)
    return [_dedupe(np.array(c)) for c in pieces]


def _same_color(a: Optional[Color], b: Optional[Color], tol: float = COLOR_TOL) -> bool:
    if a is None or b is None:
        return False
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol and abs(a[2] - b[2]) <= tol


def _bbox_of(p: np.ndarray) -> BBox:
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))


def _pairs_within(pts: np.ndarray, radius: float):
    """All index pairs (i < j) with |pts[i] - pts[j]| <= radius (grid hashing, O(n))."""
    cells: dict = {}
    for i, (x, y) in enumerate(pts):
        cells.setdefault((int(math.floor(x / radius)), int(math.floor(y / radius))), []).append(i)
    out = []
    for (cx, cy), idx in cells.items():
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand += cells.get((cx + dx, cy + dy), [])
        cand = np.array(cand)
        for i in idx:
            c = cand[cand > i]
            if len(c):
                d = np.hypot(*(pts[c] - pts[i]).T)
                out += [(i, int(j)) for j in c[d <= radius]]
    return out


# =========================================================================== internal records
@dataclass
class _Piece:
    """A polyline that may be (part of) a curve."""
    pts: np.ndarray
    color: Color
    width: float
    dashes: Optional[str]
    ids: set
    outlined: bool = False      # came from a filled outline polygon
    closed: bool = False


@dataclass(eq=False)
class _Blob:
    """A small closed shape: a marker, or a glyph of outlined text."""
    rings: list                 # outer ring first, then holes
    color: Color
    filled: bool
    ids: set
    bbox: BBox
    cls: str = "other"          # marker shape, or "other"
    hollow: bool = False        # drawn as a ring (open marker)
    centroid: Optional[np.ndarray] = None   # area centroid (set for triangles)
    use_centroid: bool = False  # anchor = centroid instead of bbox centre (see _assemble)
    centre_fix: Optional[np.ndarray] = None  # for shapes whose bbox centre is not their centre

    @property
    def pos(self) -> np.ndarray:
        """The data point this marker stands for."""
        return self.centroid if self.use_centroid and self.centroid is not None else self.centre

    @property
    def centre(self) -> np.ndarray:
        if self.centre_fix is not None:          # regular pentagon / star: vertex mean
            return self.centre_fix
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    @property
    def size(self) -> float:
        return max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


@dataclass
class _Chain:
    pts: np.ndarray
    color: Color
    width: float
    dashes: Optional[str]
    ids: set
    outlined: bool
    closed: bool
    n_pieces: int = 1


@dataclass
class _LegendEntry:
    label_bbox: BBox
    name: Optional[str]
    line: Optional[_Chain] = None
    marker: Optional[_Blob] = None


@dataclass
class _Ctx:
    """Panel geometry and the tolerances derived from it."""
    bbox: BBox
    w: float = 0.0
    h: float = 0.0
    region: BBox = (0, 0, 0, 0)      # where we look for drawings (plot area + legend margin)
    clip: BBox = (0, 0, 0, 0)        # plot area + clipping tolerance
    marker_max: float = 0.0

    def __post_init__(self):
        x0, y0, x1, y1 = self.bbox
        self.w, self.h = x1 - x0, y1 - y0
        m = 0.25 * max(self.w, self.h)
        self.region = (x0 - m, y0 - m, x1 + m, y1 + m)
        t = 0.5                     # points this close outside the frame still count as inside
        self.clip = (x0 - t, y0 - t, x1 + t, y1 + t)
        # markers are small compared with the plot; glyphs too
        self.marker_max = max(4.0, min(14.0, 0.12 * min(self.w, self.h)))


# =========================================================================== step 1: decompose paths
def _decompose(pc: PageContent, ctx: _Ctx) -> tuple[list[_Piece], list[_Blob]]:
    pieces: list[_Piece] = []
    blobs: list[_Blob] = []
    rx0, ry0, rx1, ry1 = ctx.region
    for p in pc.paths:
        bx0, by0, bx1, by1 = p.bbox
        if bx1 < rx0 or bx0 > rx1 or by1 < ry0 or by0 > ry1:
            continue
        if p.stroke is not None:
            _decompose_stroke(p, ctx, pieces, blobs)
        elif p.fill is not None:
            _decompose_fill(p, ctx, pieces, blobs)
    return pieces, blobs


def _is_white(c) -> bool:
    return c is not None and min(c) > 0.97


def _decompose_stroke(p, ctx, pieces, blobs):
    small = max(p.w, p.h) <= ctx.marker_max
    if small and any(p.closed) and all(len(s) >= 3 for s in p.subpaths):
        # small closed stroked shape -> marker (filled if the fill has the stroke colour)
        filled = p.fill is not None and not _is_white(p.fill)
        ring = _ring(p.subpaths[0])
        if len(ring) >= 3:
            blobs.append(_Blob([ring] + [_ring(s) for s in p.subpaths[1:]], p.stroke, filled,
                               {p.id}, p.bbox))
            return
    if small and len(p.subpaths) >= 2 and not any(p.closed) and _crossing(p.subpaths):
        # '+' / 'x' markers: two short strokes crossing in the middle
        segs = [s for s in p.subpaths]
        diag = all(abs(abs(d[0]) - abs(d[1])) < 0.3 * max(abs(d[0]), abs(d[1]), 1e-9)
                   for d in (s[-1] - s[0] for s in segs))
        b = _Blob([np.vstack(segs)], p.stroke, True, {p.id}, p.bbox, cls="x" if diag else "plus")
        blobs.append(b)
        return
    for sub, closed in zip(p.subpaths, p.closed):
        sub = _dedupe(sub)
        if closed and len(_ring(sub)) <= 2:
            # a "closed" single segment (a -> b -> a, the PDF 'h' operator after a line)
            sub, closed = _ring(sub), False
        if len(sub) >= 2:
            pts = np.vstack([sub, sub[:1]]) if closed and np.hypot(*(sub[0] - sub[-1])) > 1e-6 else sub
            pieces.append(_Piece(pts, p.stroke, p.width, p.dashes, {p.id}, closed=closed))


def _crossing(subs) -> bool:
    if len(subs) != 2:
        return False
    c = [s.mean(axis=0) for s in subs]
    size = max(np.ptp(np.vstack(subs), axis=0).max(), 1e-9)
    return float(np.hypot(*(c[0] - c[1]))) < 0.15 * size


def _split_rings(subpaths) -> list[list[np.ndarray]]:
    """Group the subpaths of a fill into shapes: [outer, hole, hole...] by bbox nesting."""
    rings = [_ring(s) for s in subpaths]
    rings = [r for r in rings if len(r) >= 3]
    order = sorted(range(len(rings)), key=lambda i: -abs(_area(rings[i])))
    shapes: list[list[np.ndarray]] = []
    boxes: list[BBox] = []
    for i in order:
        r = rings[i]
        bb = _bbox_of(r)
        for k, ob in enumerate(boxes):
            if bb[0] >= ob[0] - 1e-6 and bb[1] >= ob[1] - 1e-6 and bb[2] <= ob[2] + 1e-6 and bb[3] <= ob[3] + 1e-6:
                shapes[k].append(r)
                break
        else:
            shapes.append([r])
            boxes.append(bb)
    return shapes


def _decompose_fill(p, ctx, pieces, blobs):
    if _is_white(p.fill):
        return                          # backgrounds, legend box fill, marker "holes"
    for rings in _split_rings(p.subpaths):
        outer = rings[0]
        bb = _bbox_of(outer)
        size = max(bb[2] - bb[0], bb[3] - bb[1])
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        # A small *compact* shape is a marker or a glyph, even if it is thin (the outlined edge
        # of an open marker, letters like 'n').  Only elongated small shapes may be line bits
        # (one segment of an Origin curve, a dash, letters like 'l' which the glyph test removes).
        compact = size <= ctx.marker_max * 1.5 and min(bw, bh) >= 0.5 * size and len(outer) != 4
        if len(rings) == 1 and not compact:
            line = _outline_to_line(outer)
            if line is not None:
                pts, width = line
                pieces.append(_Piece(pts, p.fill, width, None, {p.id}, outlined=True))
                continue
        if size <= ctx.marker_max * 1.5 and min(bw, bh) < 0.25 * size and len(rings) == 1:
            # a short sliver whose outline we could not trace (error-bar pieces, ticks):
            # a straight line along its long axis
            if bw >= bh:
                pts = np.array([[bb[0], (bb[1] + bb[3]) / 2], [bb[2], (bb[1] + bb[3]) / 2]])
            else:
                pts = np.array([[(bb[0] + bb[2]) / 2, bb[1]], [(bb[0] + bb[2]) / 2, bb[3]]])
            pieces.append(_Piece(pts, p.fill, min(bw, bh), None, {p.id}, outlined=True))
        elif size <= ctx.marker_max * 1.5:
            blobs.append(_Blob(rings, p.fill, True, {p.id}, bb))
        # anything else (big filled areas: backgrounds, bars, patches) is not a curve


# =========================================================================== step 1b: outlined strokes
def _outline_to_line(r: np.ndarray):
    """If ring ``r`` is a thin filled outline of a stroked line, return (centreline, width).

    Fast path: Origin and many converters emit one 4-vertex rectangle per line
    segment -> the centreline is the segment joining the midpoints of the two
    short edges.  General case: ``_centreline``.
    """
    n = len(r)
    if n < 3:
        return None
    if n == 4:
        e = np.roll(r, -1, axis=0) - r                     # edge i: r[i] -> r[i+1]
        L = np.hypot(e[:, 0], e[:, 1])
        if L.min() < 1e-9:
            return None
        cosang = np.abs([(e[i] @ e[(i + 1) % 4]) / (L[i] * L[(i + 1) % 4]) for i in range(4)])
        if cosang.max() > 0.05:                            # not a rectangle (diamond marker, glyph part)
            return None
        if L[0] + L[2] <= L[1] + L[3]:                     # edges 0 and 2 are the short ones
            short, a, b = (L[0] + L[2]) / 2, (r[0] + r[1]) / 2, (r[2] + r[3]) / 2
        else:
            short, a, b = (L[1] + L[3]) / 2, (r[1] + r[2]) / 2, (r[3] + r[0]) / 2
        long_ = (L.sum() - 2 * short) / 2
        if long_ < 1.6 * short:                            # square-ish: a marker, not a line
            return None
        return np.array([a, b]), float(short)
    A, P = abs(_area(r)), float(np.hypot(*(np.roll(r, -1, axis=0) - r).T).sum())
    if P <= 0:
        return None
    w = 2 * A / P                                          # strip of length L, width w: A=Lw, P~2L
    if w > 0.15 * (P / 2) or w > 4.0 or w <= 1e-4:
        return None
    return _centreline(r, w)


def _centreline(r: np.ndarray, w0: float):
    """Centreline of a thin closed outline ``r`` (one side out, the other side back).

    1. find the two *caps*: places where the outline turns by ~180 deg within a
       short arc (~1.5 widths).  They are about half a perimeter apart.
    2. split the outline there into two sides running from cap A to cap B.
    3. pair every vertex of each side with the point straight across on the
       other side (``_pair_midpoints``); keep a pair only if its midpoint is
       ~w/2 from both sides.  The midpoints, ordered along side 1, are the
       centreline.  Validated on synthetic outlines (tests): error < 0.05 pt.
    """
    r = _despike(r, w0)
    n = len(r)
    if n < 4:
        return None
    if _area(r) < 0:                                   # make the turning angles positive overall
        r = r[::-1]
    e = np.roll(r, -1, axis=0) - r                     # edge i leaves vertex i
    L = np.hypot(e[:, 0], e[:, 1])
    P = float(L.sum())
    s = np.r_[0.0, np.cumsum(L)[:-1]]                  # arc position of every vertex
    ein = np.roll(e, 1, axis=0)                        # edge arriving at vertex i
    theta = np.arctan2(ein[:, 0] * e[:, 1] - ein[:, 1] * e[:, 0], (ein * e).sum(axis=1))
    # turning summed over a window of +-win arc length (unrolled copies handle the wrap-around)
    win = 1.5 * w0 + 1e-6
    s3 = np.r_[s - P, s, s + P]
    c3 = np.r_[0.0, np.cumsum(np.r_[theta, theta, theta])]
    lo = np.searchsorted(s3, s - win, side="left")      # absolute indices into the 3 copies
    hi = np.searchsorted(s3, s + win, side="right")
    T = c3[hi] - c3[lo]
    a = int(np.argmax(T))
    if T[a] < 0.75 * math.pi:
        return None
    da = np.abs(((s - s[a]) + P / 2) % P - P / 2)       # circular arc distance from cap A
    cand = np.where((da >= 0.3 * P) & (da <= 0.5 * P + 1e-9))[0]
    if len(cand) == 0:
        return None
    b = int(cand[np.argmax(T[cand])])
    if T[b] < 0.75 * math.pi:
        return None

    def cap_pos(i):                                    # turning-weighted arc position of the cap
        d = ((s - s[i]) + P / 2) % P - P / 2
        m = (np.abs(d) <= win) & (theta > 0)
        return (s[i] + float((theta[m] * d[m]).sum() / theta[m].sum())) % P

    sa, sb = cap_pos(a), cap_pos(b)
    side1 = _arc(r, s, P, sa, sb)                      # A -> B forwards
    side2 = _arc(r, s, P, sb, sa)[::-1]                # B -> A forwards, reversed: also A -> B
    # drop the interpolated cap points themselves: they are not on a side
    s1, s2 = side1[1:-1], side2[1:-1]
    if len(s1) < 2 or len(s2) < 2:
        return None
    # refine the width: typical distance between the sides, sampled evenly along side 1
    # (not per vertex: round caps put many vertices where the sides converge)
    w = float(np.median(_project(_points_at(s1, np.linspace(0.05, 0.95, 41)), s2)[0]))
    if w <= 1e-6:
        return None
    mids = []                                          # (position along side 1, point)
    for src, dst in ((s1, s2), (s2, s1)):
        m = _pair_midpoints(src, dst, w)
        m = m[np.isfinite(m[:, 0])]
        if len(m):
            pos = _project(m, s1)[1]
            mids += list(zip(pos, m))
    if len(mids) < 2:
        return None
    mids.sort(key=lambda t: t[0])
    # both sides usually yield the same vertex (up to the converter's rounding): average
    # consecutive midpoints closer than 0.15 w into one point
    groups: list[list[np.ndarray]] = []
    for _, m in mids:
        if groups and np.hypot(*(np.mean(groups[-1], axis=0) - m)) < 0.15 * w:
            groups[-1].append(m)
        else:
            groups.append([m])
    c = np.array([np.mean(g, axis=0) for g in groups])
    if len(c) < 2:
        return None
    # sanity: a strip of length L has perimeter ~2L, so the centreline must be ~P/2 long
    if abs(_length(c) - (P / 2 - w)) > 0.1 * P / 2:
        return None
    return c, w


def _points_at(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points at fractions ``t`` of the arc length of polyline ``p``."""
    cs = np.r_[0, np.cumsum(_seglen(p))]
    x = np.asarray(t) * cs[-1]
    k = np.clip(np.searchsorted(cs, x, side="right") - 1, 0, len(p) - 2)
    seg = cs[k + 1] - cs[k]
    f = np.where(seg > 0, (x - cs[k]) / np.where(seg > 0, seg, 1), 0.0)
    return p[k] + f[:, None] * (p[k + 1] - p[k])


def _project(pts: np.ndarray, poly: np.ndarray):
    """Distance of every point to polyline ``poly`` and the arc position of its foot point."""
    a = poly[:-1]
    ab = poly[1:] - a
    l2 = np.maximum((ab ** 2).sum(axis=1), 1e-24)
    cum = np.r_[0, np.cumsum(np.sqrt(l2))]
    dist = np.empty(len(pts))
    pos = np.empty(len(pts))
    for k in range(0, len(pts), 256):                  # chunks keep the (n, m) arrays small
        q = pts[k:k + 256]
        t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(axis=2) / l2[None], 0, 1)
        d = np.hypot(*(a[None] + t[..., None] * ab[None] - q[:, None, :]).transpose(2, 0, 1))
        i = np.argmin(d, axis=1)
        r = np.arange(len(q))
        dist[k:k + 256] = d[r, i]
        pos[k:k + 256] = cum[i] + t[r, i] * np.sqrt(l2[i])
    return dist, pos


def _despike(r: np.ndarray, w: float) -> np.ndarray:
    """Remove tiny out-and-back spikes from an outline.

    Some converters leave little "ears" at line joins: the outline steps out by
    a fraction of the width and straight back.  Each one turns by 180 deg in a
    tiny arc and would look like a line cap to ``_centreline``.
    """
    for _ in range(10):
        if len(r) < 5:
            return r
        e_out = np.roll(r, -1, axis=0) - r
        e_in = r - np.roll(r, 1, axis=0)
        lo, li = np.hypot(*e_out.T), np.hypot(*e_in.T)
        cos = (e_in * e_out).sum(axis=1) / np.maximum(lo * li, 1e-24)
        spike = (cos < -0.95) & (lo < 3 * w) & (li < 3 * w)
        if not spike.any():
            return r
        spike &= ~np.roll(spike, 1)                    # never drop two neighbours in one pass
        r = r[~spike]
        r = _ring(np.vstack([r, r[:1]]))               # merge the now coinciding vertices
    return r


def _pair_midpoints(own: np.ndarray, other: np.ndarray, w: float) -> np.ndarray:
    """For every vertex of side ``own``: the midpoint to its partner on side ``other``.

    The partner is found by shooting a ray from the vertex along the outline's
    normal (the bisector of its two edges).  At a miter join the inner corner
    lies exactly on that bisector, so outer + inner corner average to the
    original data vertex.  Fallback: the nearest vertex of the other side.
    A pair is kept only if its midpoint is ~w/2 from both sides; rows of
    rejected vertices are NaN.
    """
    n = len(own)

    def unit(d):
        return d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)[:, None]
    e = unit(np.diff(own, axis=0))                     # unit edge directions
    t = np.zeros((n, 2))
    t[:-1] += e
    t[1:] += e
    t = unit(t)
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    a, ab = other[:-1], np.diff(other, axis=0)
    out = np.full((n, 2), np.nan)
    for k in range(0, n, 256):
        v, nv = own[k:k + 256], nrm[k:k + 256]
        # solve v + s * n = a + u * ab for every (vertex, segment) pair
        den = nv[:, None, 0] * -ab[None, :, 1] + nv[:, None, 1] * ab[None, :, 0]
        good = np.abs(den) > 1e-12
        den = np.where(good, den, 1.0)
        rx = a[None, :, 0] - v[:, None, 0]
        ry = a[None, :, 1] - v[:, None, 1]
        sray = (rx * -ab[None, :, 1] + ry * ab[None, :, 0]) / den
        useg = (nv[:, None, 0] * ry - nv[:, None, 1] * rx) / den
        hit = (good & (useg >= -1e-9) & (useg <= 1 + 1e-9)
               & (np.abs(sray) >= 0.5 * w) & (np.abs(sray) <= 2.5 * w))   # partner ~w away
        sbest = np.where(hit, np.abs(sray), np.inf)
        col = np.argmin(sbest, axis=1)
        r = np.arange(len(v))
        has = np.isfinite(sbest[r, col])
        out[k:k + 256][has] = v[has] + (sray[r, col][has])[:, None] * nv[has] / 2
    ok = _midpoint_ok(out, own, other, w)
    # fallback for the rest: midpoint to the nearest vertex of the other side
    rest = np.where(~ok)[0]
    if len(rest):
        d = np.hypot(*(own[rest][:, None, :] - other[None]).transpose(2, 0, 1))
        j = np.argmin(d, axis=1)
        m = (own[rest] + other[j]) / 2
        good = (d[np.arange(len(rest)), j] <= 2.5 * w) & _midpoint_ok(m, own, other, w)
        out[rest] = np.where(good[:, None], m, np.nan)
    return out


def _midpoint_ok(m: np.ndarray, own: np.ndarray, other: np.ndarray, w: float) -> np.ndarray:
    """A centreline point is w/2 (+-15 %) away from both sides."""
    ok = np.isfinite(m[:, 0])
    if ok.any():
        d1 = _project(m[ok], own)[0]
        d2 = _project(m[ok], other)[0]
        ok[ok] = (np.abs(d1 - w / 2) <= 0.15 * w) & (np.abs(d2 - w / 2) <= 0.15 * w)
    return ok


def _arc(r, s, P, s_from, s_to):
    """Points of ring ``r`` walking forwards from arc position s_from to s_to (with wrap)."""
    span = (s_to - s_from) % P
    pts = [_at(r, s, P, s_from)]
    rel = (s - s_from) % P
    order = np.argsort(rel)
    for i in order:
        if 1e-9 < rel[i] < span - 1e-9:
            pts.append(r[i])
    pts.append(_at(r, s, P, s_to))
    return np.array(pts)


def _at(r, s, P, pos):
    pos = pos % P
    i = int(np.searchsorted(s, pos, side="right") - 1)
    j = (i + 1) % len(r)
    seg = (s[j] if j else P) - s[i]
    t = 0.0 if seg <= 0 else (pos - s[i]) / seg
    return r[i] + t * (r[j] - r[i])


# =========================================================================== step 2: chaining
def _style_ok(a, b) -> bool:
    if not _same_color(a.color, b.color) or a.dashes != b.dashes:
        return False
    wa, wb = max(a.width, 0.05), max(b.width, 0.05)
    return max(wa, wb) / min(wa, wb) <= 1.6


def _ends(p: np.ndarray):
    """Both ends of a polyline with their *outward* unit directions."""
    def unit(v):
        n = math.hypot(*v)
        return v / n if n > 1e-12 else np.zeros(2)
    return (p[0], unit(p[0] - p[1])), (p[-1], unit(p[-1] - p[-2]))


def _chain(items: list, gaps: bool = False, markers: Optional[list] = None) -> list[_Chain]:
    """Join open polylines (``_Piece`` or ``_Chain``) end to end.

    Without ``gaps`` only ends that coincide (<= SNAP) are joined, at any angle
    except a hairpin.  With ``gaps`` ends up to max(2 pt, 6 line widths) apart
    are bridged too (dashes), but only when the bridge continues both ends'
    directions (<= 25 deg): two curves that merely start near each other are
    never merged.  Centred on a marker of the same colour the bridge may be
    1.6 marker sizes long (Origin's "gap to symbol"); the marker centre is
    then inserted as the vertex it stands for.
    """
    n = len(items)
    pts = [np.asarray(it.pts) for it in items]
    closed = [it.closed or len(p) < 2 for it, p in zip(items, pts)]
    E, D, owner = [], [], []
    for i, p in enumerate(pts):
        if closed[i]:
            continue
        for side, (q, d) in enumerate(_ends(p)):
            E.append(q)
            D.append(d)
            owner.append((i, side))
    E = np.array(E).reshape(-1, 2)
    D = np.array(D).reshape(-1, 2)
    mk = list(markers or [])
    wmax = max((it.width for it in items), default=0.0)
    reach = SNAP if not gaps else max(2.0, 6 * wmax, max((1.6 * m.size + 2 * wmax for m in mk), default=0))
    cands = []
    vias: dict = {}                                    # bridged gap -> marker centre inside it
    pairs = [(ia, ib) for ia, ib in (_pairs_within(E, reach) if len(E) else [])
             if owner[ia][0] != owner[ib][0] and _style_ok(items[owner[ia][0]], items[owner[ib][0]])]
    # how many coinciding partners each end has: a sharp reversal (noise spike) is only a
    # problem when there is a choice, i.e. when 3+ ends meet in one point
    degree = np.zeros(len(E), int)
    for ia, ib in pairs:
        if math.hypot(*(E[ib] - E[ia])) <= SNAP:
            degree[ia] += 1
            degree[ib] += 1
    for ia, ib in pairs:
        pa = owner[ia][0]
        g = E[ib] - E[ia]
        dist = math.hypot(*g)
        if dist <= SNAP:
            hairpin = D[ia] @ D[ib] > math.cos(math.radians(30))   # both ends point the same way
            if hairpin and (degree[ia] > 1 or degree[ib] > 1):
                continue
            cands.append((dist + 0.01 * (1 + D[ia] @ D[ib]), ia, ib))
            continue
        if not gaps:
            continue
        u = g / dist
        if D[ia] @ u < math.cos(math.radians(25)) or -(D[ib] @ u) < math.cos(math.radians(25)):
            continue
        w = max(items[pa].width, items[owner[ib][0]].width)
        limit, via = max(2.0, 6 * w), None
        mid = (E[ia] + E[ib]) / 2
        for m in mk:
            if (_same_color(m.color, items[pa].color) and math.hypot(*(m.centre - mid)) <= 0.6 * m.size
                    and dist <= 1.6 * m.size + 2 * w):
                limit, via = max(limit, 1.6 * m.size + 2 * w), m.centre
        if dist <= limit:
            cands.append((1.0 + dist, ia, ib))
            if via is not None:
                vias[(ia, ib)] = vias[(ib, ia)] = via
    cands.sort()
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    link: dict = {}
    for _, ia, ib in cands:
        if ia in link or ib in link:
            continue
        ra, rb = find(owner[ia][0]), find(owner[ib][0])
        if ra == rb:                                     # would close a loop
            continue
        parent[ra] = rb
        link[ia], link[ib] = ib, ia
    end_index = {o: k for k, o in enumerate(owner)}
    seen = [False] * n
    out: list[_Chain] = []
    for i in range(n):
        if seen[i]:
            continue
        if closed[i]:
            seen[i] = True
            out.append(_as_chain([items[i]], [pts[i]]))
            continue
        # walk to one free end of this group, then traverse
        cur, side = i, 0
        guard = 0
        while (end_index[(cur, side)] in link) and guard <= n:
            other = owner[link[end_index[(cur, side)]]]
            cur, side = other[0], 1 - other[1]
            guard += 1
        members, polys = [], []
        enter = side                                      # we enter `cur` at its `side` end
        while True:
            seen[cur] = True
            p = pts[cur] if enter == 0 else pts[cur][::-1]
            members.append(items[cur])
            polys.append(p)
            k = end_index[(cur, 1 - enter)]
            if k not in link:
                break
            if (k, link[k]) in vias:                   # the data point hidden in the gap
                polys.append(np.array([vias[(k, link[k])]]))
            nxt = owner[link[k]]
            cur, enter = nxt[0], nxt[1]
            if seen[cur]:
                break
        out.append(_as_chain(members, polys))
    return out


def _as_chain(members, polys) -> _Chain:
    """Concatenate the oriented polylines of one chain (a shared joint point appears once)."""
    parts = [polys[0]]
    for p in polys[1:]:
        if math.hypot(*(parts[-1][-1] - p[0])) <= SNAP:
            parts.append(p[1:])
        else:
            parts.append(p)
    pts = _dedupe(np.vstack(parts))
    lens = np.array([_length(p) for p in polys if len(p) > 1]) + 1e-9
    width = float(np.average([m.width for m in members], weights=lens))
    ids = set().union(*[m.ids for m in members])
    closed = len(members) == 1 and members[0].closed
    return _Chain(pts, members[0].color, width, members[0].dashes, ids,
                  all(getattr(m, "outlined", False) for m in members), closed,
                  sum(getattr(m, "n_pieces", 1) for m in members))


# =========================================================================== step 3: blobs -> markers / glyphs
def _classify_blob(b: _Blob) -> None:
    """Set ``b.cls`` to a marker shape name or "other" (glyphs, odd shapes)."""
    if b.cls != "other":                             # already set ('+'/'x' strokes)
        return
    outer = b.rings[0]
    bw, bh = b.bbox[2] - b.bbox[0], b.bbox[3] - b.bbox[1]
    if min(bw, bh) <= 1e-6:
        return
    hull = _hull(outer)
    if len(hull) < 3:
        return
    hull_area = abs(_area(hull))
    solidity = abs(_area(outer)) / hull_area if hull_area > 0 else 0
    holes = sum(abs(_area(h)) for h in b.rings[1:])
    perim = float(_seglen(np.vstack([outer, outer[:1]])).sum())
    if len(b.rings) > 1 and holes > 0.15 * abs(_area(outer)):
        b.hollow = True                              # ring: open marker (or a glyph like 'o')
    elif len(b.rings) == 1 and 2 * abs(_area(outer)) / perim < 0.12 * max(bw, bh):
        # a thin closed band: outer and inner contour of an outlined marker edge joined
        # into one subpath -> shape of its hull, hollow
        b.hollow = True
    elif solidity < 0.85:
        b.cls = _concave_class(outer, bw, bh)
        if b.cls == "star":
            b.centre_fix = _simplify_closed(outer, 0.05 * max(bw, bh)).mean(axis=0)
        return
    aspect = bw / bh
    size = max(bw, bh)
    # resample the hull boundary uniformly to judge roundness independent of vertex spacing;
    # radii are measured after scaling the bbox to a square, because figures are sometimes
    # stretched non-uniformly when placed (circles become ellipses)
    hh = np.vstack([hull, hull[:1]])
    sl = _seglen(hh)
    ts = np.linspace(0, sl.sum(), 64, endpoint=False)
    cs = np.r_[0, np.cumsum(sl)]
    k = np.clip(np.searchsorted(cs, ts, side="right") - 1, 0, len(sl) - 1)
    t = (ts - cs[k]) / np.maximum(sl[k], 1e-12)
    samp = hh[k] + t[:, None] * (hh[k + 1] - hh[k])
    cc = np.array([(b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2])
    rad = np.hypot(*((samp - cc) / [bw / 2, bh / 2]).T)
    if 0.7 < aspect < 1.43 and rad.std() / rad.mean() < 0.04:
        b.cls = "circle"
        return
    simp = _simplify_closed(hull, 0.08 * size)
    k = len(simp)
    if k == 3:
        b.cls = _triangle_dir(simp)
        b.centroid = simp.mean(axis=0)               # a triangle's area centroid = vertex mean
    elif k == 4:
        e = np.roll(simp, -1, axis=0) - simp
        ang = np.abs(np.degrees(np.arctan2(e[:, 1], e[:, 0]))) % 90
        axis = np.minimum(ang, 90 - ang)
        if axis.max() < 8 and 0.7 < aspect < 1.43:
            b.cls = "square"
        elif np.all(np.abs(axis - 45) < 12):
            b.cls = "diamond"
    elif k in (5, 6) and 0.8 < aspect < 1.25 and _regular(simp):
        b.cls = "pentagon" if k == 5 else "hexagon"
        if k == 5:
            b.centre_fix = simp.mean(axis=0)      # a pentagon's bbox is not centred on it


def _regular(poly: np.ndarray) -> bool:
    """Regular polygon test (equal radii and equal edges), so glyphs like 'a' don't pass."""
    rad = np.hypot(*(poly - poly.mean(axis=0)).T)
    edges = np.hypot(*(np.roll(poly, -1, axis=0) - poly).T)
    return rad.std() / rad.mean() < 0.08 and edges.std() / edges.mean() < 0.15


def _simplify_closed(ring: np.ndarray, tol: float) -> np.ndarray:
    """RDP on a closed ring, starting from its two most distant vertices."""
    d = np.hypot(*(ring[:, None, :] - ring[None, :, :]).transpose(2, 0, 1))
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    i, j = min(i, j), max(i, j)
    a = _rdp(ring[i:j + 1], tol)
    b = _rdp(np.vstack([ring[j:], ring[:i + 1]]), tol)
    return np.vstack([a[:-1], b[:-1]])


def _triangle_dir(t: np.ndarray) -> str:
    """Name a triangle by where its apex points (page coords: y grows down)."""
    c = t.mean(axis=0)
    e = [(t[(i + 1) % 3] - t[i]) for i in range(3)]
    # the base is the most axis-aligned edge; the apex is the vertex opposite to it
    best, apex = 1e9, None
    for i, v in enumerate(e):
        a = abs(math.degrees(math.atan2(v[1], v[0]))) % 90
        dev = min(a, 90 - a)
        if dev < best:
            best, apex = dev, t[(i + 2) % 3]
    d = apex - c
    if abs(d[1]) >= abs(d[0]):
        return "triangle_up" if d[1] < 0 else "triangle_down"
    return "triangle_left" if d[0] < 0 else "triangle_right"


def _concave_class(outer, bw, bh) -> str:
    if not 0.8 < bw / bh < 1.25:
        return "other"
    size = max(bw, bh)
    simp = _simplify_closed(outer, 0.05 * size)
    k = len(simp)
    if k == 10:
        return "star"
    if k == 12:
        e = np.roll(simp, -1, axis=0) - simp
        ang = np.abs(np.degrees(np.arctan2(e[:, 1], e[:, 0]))) % 90
        axis = np.minimum(ang, 90 - ang)
        if axis.max() < 8:
            return "plus"
        if np.all(np.abs(axis - 45) < 10):
            return "x"
    return "other"


def _merge_concentric(blobs: list[_Blob]) -> list[_Blob]:
    """One marker drawn twice (filled body + outlined edge) -> one blob."""
    if not blobs:
        return blobs
    cen = np.array([b.centre for b in blobs])
    maxs = max(b.size for b in blobs)
    alive = [True] * len(blobs)
    for i, j in sorted(_pairs_within(cen, 0.2 * maxs)):
        if not (alive[i] and alive[j]):
            continue
        a, b = blobs[i], blobs[j]
        s = max(a.size, b.size)
        if not _same_color(a.color, b.color) or math.hypot(*(a.centre - b.centre)) > 0.05 * s:
            continue
        if not 0.6 <= a.size / b.size <= 1 / 0.6:
            continue
        if a.cls != b.cls and "other" not in (a.cls, b.cls):
            continue                                  # two different markers that overlap
        # keep the better-classified one; a solid body makes the marker filled
        (keep, drop), di = ((a, b), j) if _rank(a) >= _rank(b) else ((b, a), i)
        keep.filled = (a.filled and not a.hollow) or (b.filled and not b.hollow)
        keep.hollow = not keep.filled
        keep.ids |= drop.ids
        keep.bbox = (min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]),
                     max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3]))
        alive[di] = False
    out = [b for b, ok in zip(blobs, alive) if ok]
    # Shading inside a marker (Origin's "3D ball": grey and white discs inside a black one,
    # a dot inside a circle) belongs to that marker.
    out.sort(key=lambda b: -b.size)
    cell = max(maxs, 1e-3)
    grid: dict = {}                                   # marker-shaped hosts by grid cell
    keep: list[_Blob] = []
    for b in out:
        cx, cy = int(b.centre[0] // cell), int(b.centre[1] // cell)
        near = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1) for k in grid.get((cx + dx, cy + dy), ())]
        host = next((k for k in near if b.size <= 0.85 * k.size
                     and b.bbox[0] >= k.bbox[0] - 0.05 and b.bbox[1] >= k.bbox[1] - 0.05
                     and b.bbox[2] <= k.bbox[2] + 0.05 and b.bbox[3] <= k.bbox[3] + 0.05), None)
        if host is not None:
            host.ids |= b.ids
            continue
        keep.append(b)
        if b.cls != "other":
            grid.setdefault((cx, cy), []).append(b)
    return keep


def _rank(b: _Blob) -> int:
    return (2 if b.cls != "other" else 0) + (1 if not b.hollow else 0)


def _glyph_runs(blobs: list[_Blob], chains: list[_Chain]):
    """Find text drawn as glyph outlines.

    Glyph candidates are blobs plus tiny chains (strokes of letters such as
    'l', '1', '-').  Neighbours on a common text line (vertical overlap,
    horizontal gap < 0.6 x height) are connected; a connected group with at
    least two glyph-like ("other") blobs is a word -> every member is text.
    Rows of identical markers are therefore never mistaken for text.
    Returns (set of blob indices, set of chain indices, list of run bboxes).
    """
    other_h = [b.bbox[3] - b.bbox[1] for b in blobs if b.cls == "other"]
    if not other_h:
        return set(), set(), []
    H = float(np.median(other_h))
    Hmax = 2.0 * float(np.percentile(other_h, 90))      # fonts differ (legend vs panel label)
    items = [("b", i, b.bbox) for i, b in enumerate(blobs)]
    for i, c in enumerate(chains):
        hw = c.width / 2                               # a stroke has thickness: '-' is not 0 tall
        cb = _bbox_of(c.pts)
        cb = (cb[0] - hw, cb[1] - hw, cb[2] + hw, cb[3] + hw)
        if not c.closed and max(cb[2] - cb[0], cb[3] - cb[1]) <= Hmax and c.n_pieces <= 2:
            items.append(("c", i, cb))
    bb = np.array([it[2] for it in items])
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    hs = bb[:, 3] - bb[:, 1]
    for i in range(len(items)):
        ov = np.minimum(bb[i, 3], bb[:, 3]) - np.maximum(bb[i, 1], bb[:, 1])
        gap = np.maximum(bb[:, 0] - bb[i, 2], bb[i, 0] - bb[:, 2])
        hmax, hmin = np.maximum(hs[i], hs), np.minimum(hs[i], hs)
        ok = (ov >= 0.4 * hmin) & (ov > 0) & (gap <= 0.6 * hmax)
        # sub-/superscripts ('WO' + 'x'): a smaller glyph right next to, slightly below/above
        ok |= (gap <= 0.3 * hmax) & (ov >= -0.3 * hmax) & (hmin <= 0.7 * hmax)
        ok &= np.maximum(hs[i], hs) <= 2.5 * H                 # glyph-sized
        for j in np.where(ok)[0]:
            if j != i:
                parent[find(i)] = find(int(j))
    groups: dict = {}
    for k in range(len(items)):
        groups.setdefault(find(k), []).append(k)
    tb, tc, runs = set(), set(), []
    for members in groups.values():
        # glyph-like members: odd blobs, and bent letter strokes ('n', 'm', '1' ...).  Straight
        # strokes do not count, so a marker with its error bar is never taken for a word.
        n_other = sum(1 for k in members
                      if (items[k][0] == "b" and blobs[items[k][1]].cls == "other")
                      or (items[k][0] == "c" and not _straight(chains[items[k][1]].pts)
                          and bb[k, 3] - bb[k, 1] >= 0.5 * H))
        if n_other < 2:
            continue
        for k in members:
            (tb if items[k][0] == "b" else tc).add(items[k][1])
        g = bb[members]
        runs.append((float(g[:, 0].min()), float(g[:, 1].min()), float(g[:, 2].max()), float(g[:, 3].max())))
    return tb, tc, runs


# =========================================================================== step 4: legend
def _labels(pc: PageContent, ctx: _Ctx, runs: list) -> list[tuple[BBox, Optional[str]]]:
    """Text lines near the plot: real text spans merged per line, plus glyph runs."""
    rx0, ry0, rx1, ry1 = ctx.region
    spans = [t for t in pc.texts if abs(t.dir[0] - 1) < 0.05 and t.bbox[0] < rx1 and t.bbox[2] > rx0
             and t.bbox[1] < ry1 and t.bbox[3] > ry0]
    spans.sort(key=lambda t: (t.bbox[0]))
    lines: list[list] = []
    for t in spans:
        h = t.bbox[3] - t.bbox[1]
        for ln in lines:
            last = ln[-1]
            lh = last.bbox[3] - last.bbox[1]
            ov = min(t.bbox[3], last.bbox[3]) - max(t.bbox[1], last.bbox[1])
            if ov >= 0.4 * min(h, lh) and -0.5 * h <= t.bbox[0] - last.bbox[2] <= 1.0 * max(h, lh):
                ln.append(t)
                break
        else:
            lines.append([t])
    out = []
    for ln in lines:
        bb = (min(t.bbox[0] for t in ln), min(t.bbox[1] for t in ln),
              max(t.bbox[2] for t in ln), max(t.bbox[3] for t in ln))
        name = "".join(t.text for t in ln).strip()
        out.append((bb, name or None))
    out += [(r, None) for r in runs]
    return out


def _find_legend(labels, chains: list[_Chain], markers: list[_Blob], ctx: _Ctx):
    """Legend entries; returns (entries, used chain indices, used marker indices)."""
    entries, used_c, used_m = [], set(), set()
    # candidate sample lines: short, straight, horizontal (index, y, x_left, x_right, dy)
    samples = []
    for i, c in enumerate(chains):
        a, b = c.pts[0], c.pts[-1]
        if (not c.closed and abs(a[1] - b[1]) <= 0.5 and abs(a[0] - b[0]) <= 0.35 * ctx.w
                and _straight(c.pts)):
            samples.append((i, a[1], min(a[0], b[0]), max(a[0], b[0]), abs(a[1] - b[1])))
    S = np.array(samples, float).reshape(-1, 5)
    mc = np.array([m.centre for m in markers]).reshape(-1, 2)
    ms = np.array([m.size for m in markers])
    for bb, name in labels:
        lh = bb[3] - bb[1]
        if lh <= 0:
            continue
        yc = (bb[1] + bb[3]) / 2
        line_i = None
        if len(S):
            L = S[:, 3] - S[:, 2]
            ok = ((S[:, 4] <= 0.1 * lh) & (L >= 0.8 * lh) & (np.abs(S[:, 1] - yc) <= 0.5 * lh)
                  & (S[:, 3] >= bb[0] - 2.5 * lh) & (S[:, 3] <= bb[0] + 0.2 * lh))
            ok &= np.array([int(k) not in used_c for k in S[:, 0]])
            if ok.any():
                k = np.where(ok)[0][np.argmin(bb[0] - S[ok, 3])]     # the one closest to the text
                line_i = int(S[k, 0])
        mark_i = None
        if line_i is not None and len(mc):
            c = chains[line_i]
            x0, x1 = sorted((c.pts[0][0], c.pts[-1][0]))
            hit = ((mc[:, 0] >= x0 - 0.2 * ms) & (mc[:, 0] <= x1 + 0.2 * ms)
                   & (np.abs(mc[:, 1] - c.pts[0][1]) <= 0.3 * ms))
            mark_i = next((int(j) for j in np.where(hit)[0] if int(j) not in used_m), None)
        elif len(mc):
            dx = bb[0] - mc[:, 0]
            hit = (dx > 0) & (dx <= 3.0 * lh) & (np.abs(mc[:, 1] - yc) <= 0.5 * lh) & (ms <= 1.5 * lh)
            cands = sorted((float(dx[j]), int(j)) for j in np.where(hit)[0]
                           if int(j) not in used_m and _lonely(int(j), markers, chains))
            if cands:
                mark_i = cands[0][1]
        if line_i is None and mark_i is None:
            continue
        if line_i is not None:
            used_c.add(line_i)
        if mark_i is not None:
            used_m.add(mark_i)
        entries.append(_LegendEntry(bb, name, chains[line_i] if line_i is not None else None,
                                    markers[mark_i] if mark_i is not None else None))
    return entries, used_c, used_m


def _lonely(j: int, markers: list[_Blob], chains: list[_Chain]) -> bool:
    """A legend sample marker stands alone: no twin close by, not on a curve.

    (Otherwise a data marker that happens to sit left of an annotation would be
    taken for a legend entry.)"""
    m = markers[j]
    for k, o in enumerate(markers):
        if k != j and o.cls == m.cls and math.hypot(*(o.centre - m.centre)) < 3 * m.size:
            return False
    return not any(np.hypot(*(c.pts - m.centre).T).min() < 0.3 * m.size for c in chains
                   if not c.closed and _length(c.pts) > 3 * m.size)


# =========================================================================== step 5: decorations
def _straight(p: np.ndarray, tol: float = STRAIGHT_TOL) -> bool:
    return len(p) <= 2 or float(_dist_to_line(p, p[0], p[-1]).max()) <= tol


def _axis_dir(a, b) -> Optional[str]:
    d = b - a
    if math.hypot(*d) < 1e-9:
        return None
    ang = math.atan2(abs(d[1]), abs(d[0]))
    if ang <= AXIS_TOL:
        return "h"
    if ang >= math.pi / 2 - AXIS_TOL:
        return "v"
    return None


def _axis_fraction(p: np.ndarray) -> float:
    """Fraction of a polyline's length made of horizontal/vertical segments."""
    sl = _seglen(p)
    if sl.sum() <= 0:
        return 0.0
    ax = np.array([_axis_dir(a, b) is not None for a, b in zip(p[:-1], p[1:])])
    return float(sl[ax].sum() / sl.sum())


def _is_decoration(c: _Chain, ctx: _Ctx) -> bool:
    p = c.pts
    x0, y0, x1, y1 = ctx.bbox
    closed = c.closed or (len(p) > 3 and math.hypot(*(p[0] - p[-1])) <= max(1.0, 2 * c.width))
    afrac = _axis_fraction(p)
    if closed and afrac >= 0.75:
        return True                                   # frame / legend box / inset box
    if afrac >= 0.9 and _length(p) <= 0.1 * min(ctx.w, ctx.h):
        tol = 1.0 + c.width                           # tick (+ a bit of frame) on the frame
        if np.any((np.abs(p[:, 0] - x0) <= tol) | (np.abs(p[:, 0] - x1) <= tol)
                  | (np.abs(p[:, 1] - y0) <= tol) | (np.abs(p[:, 1] - y1) <= tol)):
            return True
    if _straight(p):
        d = _axis_dir(p[0], p[-1])
        if d == "h":
            if abs(p[0][0] - p[-1][0]) >= SPAN_FRAC * ctx.w:
                return True                           # grid / zero line / frame side
            if min(abs(p[0][1] - y0), abs(p[0][1] - y1)) <= 1.0 + c.width:
                return True                           # lies on the frame
        elif d == "v":
            if abs(p[0][1] - p[-1][1]) >= SPAN_FRAC * ctx.h:
                return True
            if min(abs(p[0][0] - x0), abs(p[0][0] - x1)) <= 1.0 + c.width:
                return True
    elif afrac >= 0.9:
        # a polyline of axis-aligned runs with one long run: frame drawn as one path
        for a, b in zip(p[:-1], p[1:]):
            d = _axis_dir(a, b)
            if (d == "h" and abs(b[0] - a[0]) >= SPAN_FRAC * ctx.w) or \
                    (d == "v" and abs(b[1] - a[1]) >= SPAN_FRAC * ctx.h):
                return True
    return False


def _is_inner_box(c: _Chain, ctx: _Ctx) -> bool:
    """A closed rectangle clearly smaller than the plot: an inset's frame or a legend box."""
    p = c.pts
    closed = c.closed or (len(p) > 3 and math.hypot(*(p[0] - p[-1])) <= max(1.0, 2 * c.width))
    if not closed or _axis_fraction(p) < 0.75:
        return False
    bx0, by0, bx1, by1 = _bbox_of(p)
    x0, y0, x1, y1 = ctx.bbox
    return ((bx1 - bx0) * (by1 - by0) < 0.7 * ctx.w * ctx.h and bx0 >= x0 - 1 and bx1 <= x1 + 1
            and by0 >= y0 - 1 and by1 <= y1 + 1)


def _side_boxes(chains: list[_Chain], ctx: _Ctx) -> list[BBox]:
    """Rectangles drawn as four separate sides (e.g. an inset frame of four outlined lines)."""
    hs, vs = [], []
    for c in chains:
        p = c.pts
        if c.closed or not _straight(p) or _length(p) < 3.0:
            continue
        d = _axis_dir(p[0], p[-1])
        if d == "h":
            hs.append((min(p[0][0], p[-1][0]), max(p[0][0], p[-1][0]), p[0][1], c.width))
        elif d == "v":
            vs.append((min(p[0][1], p[-1][1]), max(p[0][1], p[-1][1]), p[0][0], c.width))
    out = []
    for i, (a0, a1, ya, wa) in enumerate(hs):
        for b0, b1, yb, wb in hs[i + 1:]:
            tol = 0.3 + wa + wb                # sides meet at the corners (error-bar caps don't)
            if abs(a0 - b0) > tol or abs(a1 - b1) > tol or abs(ya - yb) < 3.0 or a1 - a0 < 3.0:
                continue
            top, bot = min(ya, yb), max(ya, yb)
            def side(x):
                return any(abs(xv - x) <= tol and v0 <= top + tol and v1 >= bot - tol
                           for v0, v1, xv, _ in vs)
            if side(min(a0, b0)) and side(max(a1, b1)):
                box = (min(a0, b0), top, max(a1, b1), bot)
                x0, y0, x1, y1 = ctx.bbox
                if ((box[2] - box[0]) * (box[3] - box[1]) < 0.7 * ctx.w * ctx.h
                        and box[0] >= x0 - 1 and box[2] <= x1 + 1 and box[1] >= y0 - 1 and box[3] <= y1 + 1):
                    out.append(box)
    return out


def _mostly_inside(p: np.ndarray, box: BBox, tol: float = 0.5) -> bool:
    """A curve belongs to the box: most of it lies inside, or it stays within the box's
    x-range and a good part of it is inside (inset curves may run out of the inset frame)."""
    mid = (p[:-1] + p[1:]) / 2
    sl = _seglen(p)
    if sl.sum() <= 0:
        return False
    inb = ((mid[:, 0] >= box[0] - tol) & (mid[:, 0] <= box[2] + tol)
           & (mid[:, 1] >= box[1] - tol) & (mid[:, 1] <= box[3] + tol))
    frac = float(sl[inb].sum() / sl.sum())
    within_x = p[:, 0].min() >= box[0] - 2 * tol - 0.5 and p[:, 0].max() <= box[2] + 2 * tol + 0.5
    return frac >= 0.6 or (within_x and frac >= 0.3)


def _inside(bb: BBox, box: BBox, tol: float = 0.5) -> bool:
    return bb[0] >= box[0] - tol and bb[1] >= box[1] - tol and bb[2] <= box[2] + tol and bb[3] <= box[3] + tol


def _drop_arrows(chains: list[_Chain], blobs: list[_Blob]):
    """Annotation arrows: a straight line with a filled head at exactly one end.

    Returns the remaining chains and the blobs that were arrow heads.
    """
    keep, heads = [], []
    bc = np.array([b.centre for b in blobs]).reshape(-1, 2)
    bs = np.array([b.size for b in blobs])
    for c in chains:
        p = c.pts
        if len(bc) and _straight(p):
            at_end = []
            for end in (p[0], p[-1]):
                near = np.where((np.hypot(*(bc - end).T) <= 0.75 * bs) & (_length(p) > 3 * bs))[0]
                at_end.append([blobs[k] for k in near if _same_color(blobs[k].color, c.color)])
            if bool(at_end[0]) != bool(at_end[1]):
                heads += at_end[0] + at_end[1]
                continue
        keep.append(c)
    return keep, heads


def _short_axis_groups(chains: list[_Chain], ctx: _Ctx) -> set:
    """Ticks, error bars and caps: short straight axis-aligned chains repeated >= 3 times per colour."""
    ext = min(ctx.w, ctx.h)
    cand = [i for i, c in enumerate(chains)
            if not c.closed and _straight(c.pts) and _axis_dir(c.pts[0], c.pts[-1])
            and _length(c.pts) <= 0.25 * ext]
    groups: list[list[int]] = []                  # candidates grouped by colour
    for i in cand:
        g = next((g for g in groups if _same_color(chains[g[0]].color, chains[i].color)), None)
        if g is None:
            groups.append([i])
        else:
            g.append(i)
    return {i for g in groups if len(g) >= 3 for i in g}


# =========================================================================== step 6: assemble
def _marker_groups(markers: list[_Blob]) -> list[list[_Blob]]:
    """Group markers by colour, shape, fill and size (within 20 %)."""
    groups: list[list[_Blob]] = []
    for m in markers:
        for g in groups:
            r = g[0]
            if r.cls == m.cls and r.filled == m.filled and _same_color(r.color, m.color) \
                    and 0.8 <= m.size / r.size <= 1.25:
                g.append(m)
                break
        else:
            groups.append([m])
    return groups


def _orient(p: np.ndarray) -> np.ndarray:
    """Function-like curves run left to right."""
    return p[::-1].copy() if p[-1][0] < p[0][0] else p


def _vertex_hits(c: _Chain, g: list[_Blob]) -> int:
    """How many markers of group ``g`` sit on a vertex of chain ``c``."""
    # Plotting programs put the marker exactly on a vertex (0.04 pt in real Origin files).
    # Keep this tight: a dense fit line (1000 vertices) passes *near* noisy markers too.
    tol = max(0.1, 0.05 * g[0].size)
    return sum(1 for m in g if np.hypot(*(c.pts - m.pos).T).min() <= tol)


def _assemble(chains: list[_Chain], groups: list[list[_Blob]]) -> list[Series]:
    """Chains (+ the marker group on their vertices) and lone marker groups -> Series.

    Plotting programs anchor markers differently: matplotlib puts the *bbox
    centre* of '^' on the data point, Origin the triangle's *centroid* (1/6 of
    the height lower).  For line+markers we simply try both anchors and keep
    the one that lands on the line's vertices; marker-only triangle groups then
    follow the anchor that this panel's attached groups used (default: bbox).
    """
    series: list[Series] = []
    attach: dict = {}                  # chain index -> (hits, group index)
    centroid_votes = 0
    for gi, g in enumerate(groups):
        best = None
        if len(g) < 2:                     # a lone symbol says nothing about a line
            continue
        anchors = (False, True) if g[0].centroid is not None else (False,)
        for ci, c in enumerate(chains):
            if not _same_color(c.color, g[0].color):
                continue
            for use_c in anchors:
                for m in g:
                    m.use_centroid = use_c
                hits = _vertex_hits(c, g)
                if hits >= max(1, 0.6 * len(g)) and (best is None or hits > best[0]):
                    best = (hits, ci, use_c)
        for m in g:
            m.use_centroid = bool(best and best[2])
        if best is not None:
            centroid_votes += 1 if best[2] else (-1 if g[0].centroid is not None else 0)
            prev = attach.get(best[1])
            if prev is None or best[0] > prev[0]:
                attach[best[1]] = (best[0], gi)
    used_groups = {gi for _, gi in attach.values()}
    for ci, c in enumerate(chains):
        pts = _orient(c.pts)
        style = {"color": tuple(c.color), "width": round(c.width, 3), "dashes": c.dashes,
                 "outlined": c.outlined, "marker": None, "marker_size": None,
                 "marker_filled": None, "marker_pts": None}
        ids = set(c.ids)
        kind = "line"
        if ci in attach:
            g = groups[attach[ci][1]]
            near = [int(np.argmin(np.hypot(*(pts - m.pos).T))) for m in g]   # order along the line
            order = np.argsort(near, kind="stable")
            style.update(marker=g[0].cls, marker_size=round(float(np.median([m.size for m in g])), 3),
                         marker_filled=g[0].filled,
                         marker_pts=np.array([g[k].pos for k in order]))
            ids |= set().union(*[m.ids for m in g])
            kind = "line+markers"
        series.append(Series(None, kind, style, pts, pts.copy(), sorted(ids)))
    for gi, g in enumerate(groups):
        if gi in used_groups or len(g) < 3:            # 1-2 lone symbols: annotations
            continue
        for m in g:
            m.use_centroid = centroid_votes > 0
        cen = np.array([m.pos for m in g])
        cen = cen[np.argsort(cen[:, 0], kind="stable")]
        style = {"color": tuple(g[0].color), "width": None, "dashes": None, "outlined": False,
                 "marker": g[0].cls, "marker_size": round(float(np.median([m.size for m in g])), 3),
                 "marker_filled": g[0].filled, "marker_pts": cen}
        series.append(Series(None, "markers", style, cen, cen.copy(),
                             sorted(set().union(*[m.ids for m in g]))))
    return series


def _name_series(series: list[Series], entries: list[_LegendEntry]) -> None:
    """Give each series the name of the legend entry whose sample has the same style."""
    def cost(e: _LegendEntry, s: Series):
        st = s.style
        col = e.line.color if e.line is not None else e.marker.color
        if not _same_color(col, st["color"], 0.1):
            return None
        c = 0.0
        if e.marker is not None:
            if st["marker"] != e.marker.cls or st["marker_filled"] != e.marker.filled:
                return None
        elif st["marker"] is not None:
            c += 1.0
        if e.line is not None:
            if s.kind == "markers" or e.line.dashes != st["dashes"]:
                return None
            c += abs(e.line.width - (st["width"] or 0)) / max(e.line.width, 0.1)
        elif s.kind != "markers":
            c += 1.0
        return c
    for e in entries:
        if e.name is None:
            continue
        costs = [(cost(e, s), i) for i, s in enumerate(series) if s.name is None]
        costs = sorted((c, i) for c, i in costs if c is not None)
        if not costs:
            continue
        if len(costs) > 1 and costs[1][0] - costs[0][0] < 0.05:
            continue                                     # two series look identical: ambiguous
        series[costs[0][1]].name = e.name


# =========================================================================== public entry point
def extract_series(pc: PageContent, panel: Panel) -> list[Series]:
    """Data series inside ``panel.bbox`` (see module docstring for the algorithm)."""
    ctx = _Ctx(tuple(panel.bbox))
    pieces, blobs = _decompose(pc, ctx)

    # 2. exact chaining (no gaps yet): curves drawn piecewise become one polyline
    chains = _chain(pieces)

    # 3. shapes of blobs; merge markers drawn in several layers; then remove outlined text
    for b in blobs:
        _classify_blob(b)
    blobs = _merge_concentric(blobs)
    glyph_b, glyph_c, runs = _glyph_runs(blobs, chains)
    blobs = [b for i, b in enumerate(blobs) if i not in glyph_b]
    chains = [c for i, c in enumerate(chains) if i not in glyph_c]
    markers = [b for b in blobs if b.cls in MARKER_SHAPES]

    # 5. legend (before the decoration filters, which would eat short sample lines)
    entries, used_c, used_m = _find_legend(_labels(pc, ctx, runs), chains, markers, ctx)
    chains = [c for i, c in enumerate(chains) if i not in used_c]
    markers = [m for i, m in enumerate(markers) if i not in used_m]

    # 6. frame, grid, reference lines, ticks, error bars, arrows; then whatever sits inside
    #    a box (inset plots, legend boxes)
    small = _short_axis_groups(chains, ctx)
    deco = [_is_decoration(c, ctx) for c in chains]
    boxes = [_bbox_of(c.pts) for c, d in zip(chains, deco) if d and _is_inner_box(c, ctx)]
    boxes += _side_boxes(chains, ctx)
    chains = [c for i, c in enumerate(chains) if i not in small and not deco[i]]
    chains, heads = _drop_arrows(chains, blobs)
    markers = [m for m in markers if m not in heads]
    chains = [c for c in chains if not any(_mostly_inside(c.pts, b) for b in boxes)]
    markers = [m for m in markers if not any(_inside(m.bbox, b) for b in boxes)]

    # 7. bridge small gaps (dashes, gap-to-symbol), then drop duplicates
    chains = (_chain([c for c in chains if not c.closed], gaps=True, markers=markers)
              + [c for c in chains if c.closed])
    chains = [c for c in chains if not _is_decoration(c, ctx)]

    # 8. clip to the plot area
    clipped = []
    for c in chains:
        parts = [q for q in _clip_polyline(c.pts, ctx.clip) if len(q) >= 2]
        if not parts:
            continue
        c.pts = _dedupe(np.vstack(parts))
        if len(c.pts) >= 2:
            clipped.append(c)
    cx0, cy0, cx1, cy1 = ctx.clip
    markers = [m for m in markers if cx0 <= m.centre[0] <= cx1 and cy0 <= m.centre[1] <= cy1]
    min_len = 0.04 * min(ctx.w, ctx.h)
    groups = _marker_groups(markers)
    chains = [c for c in _drop_duplicates(clipped)
              if _length(c.pts) >= min_len or _has_markers(c, markers)]

    # 9. series + names
    series = _assemble(chains, groups)
    _name_series(series, entries)
    for s in series:
        s.data = panel.to_data(s.page_pts)
    return series


def _has_markers(c: _Chain, markers: list[_Blob]) -> bool:
    return any(_same_color(m.color, c.color) and np.hypot(*(c.pts - m.centre).T).min() < 0.2 * m.size
               for m in markers)


def _drop_duplicates(chains: list[_Chain]) -> list[_Chain]:
    """The same curve painted twice -> keep one (same colour, length and end points)."""
    out: list[_Chain] = []
    lens: list[float] = []
    for c in chains:
        L = _length(c.pts)
        dup = None
        for o, Lo in zip(out, lens):
            if abs(Lo - L) > 0.05 * max(L, 1e-9) or not _same_color(o.color, c.color):
                continue
            ends = (np.hypot(*(o.pts[0] - c.pts[0])) + np.hypot(*(o.pts[-1] - c.pts[-1])),
                    np.hypot(*(o.pts[0] - c.pts[-1])) + np.hypot(*(o.pts[-1] - c.pts[0])))
            # same ends are not enough (EQE curves all start and end at 0): whole curve must match
            if (min(ends) < 0.2 and _project(c.pts, o.pts)[0].max() < 0.2
                    and _project(o.pts, c.pts)[0].max() < 0.2):
                dup = o
                break
        if dup is not None:
            dup.ids |= c.ids
        else:
            out.append(c)
            lens.append(L)
    return out
