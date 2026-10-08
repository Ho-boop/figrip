"""Read tick-label numbers that are *not* text.

Three engines, used by `axes.py` in this order of preference:

1. **Glyph-outline recogniser** (`glyph_words`; no extra dependencies).
   Many journals convert every glyph of a figure into a filled vector path,
   so `get_text()` returns nothing for the tick labels.  But a glyph outline
   is still an exact shape: we rasterise it ourselves into a small coverage
   grid (normalised to the glyph's own bounding box) and compare it with
   templates of the digits 0-9 rendered from a handful of common fonts
   (PDF base-14 Helvetica/Times, plus Arial/Times New Roman/Calibri/DejaVu
   when those font files exist on the machine).  Minus sign and decimal
   point are too small and featureless for template matching, so they are
   recognised by geometry (size + position relative to the digits), and so
   are superscripts (10^-3).  The caller validates the numbers with the
   arithmetic/geometric-progression test, so a rare misread digit is caught.

2. **Rendered characters of unmapped fonts** (`rendered_words`; needs the
   page).  Type-3 fonts without a unicode map extract as control codes, but
   the PDF still tells where each character is: render each character box
   and match it against the same templates.

3. **RapidOCR** (`ocr_region`; optional: ``pip install rapidocr-onnxruntime``).
   Generic pixel OCR of a rendered strip, last resort.  If the package is
   missing, `ocr_region` simply returns ``[]``.

Public API
----------
``glyph_words(pc, region) -> list[Word]``       outlined glyphs in `region`, grouped into words
``glyph_letters(pc, region) -> list[Word]``     outlined panel letters ("a", "(b)")
``rendered_words(page, region) -> list[Word]``  unmapped-font text, read from pixels
``ocr_region(page, clip, dpi=600) -> list[Word]``  pixel OCR (may be empty)
``ocr_available() -> bool``
``unreadable(text) -> bool``                    text extracted as garbage codes?
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from .model import BBox, PageContent

GRID = 16            # glyphs are compared as GRID x GRID coverage maps
SUPERSAMPLE = 3      # sample points per grid cell and axis (anti-aliasing)
MIN_SCORE = 0.70     # below this correlation a glyph is reported as '?'
DIGITS = "0123456789"


@dataclass
class Word:
    """A recognised string with its page-space bounding box."""
    text: str
    bbox: BBox
    score: float = 1.0           # worst glyph score in the word (1 = exact text)
    source: str = "glyph"        # "text", "glyph" or "ocr"

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def h(self) -> float:
        return self.bbox[3] - self.bbox[1]


# --------------------------------------------------------------------------- rasterising
def _sample_grid(box: BBox) -> tuple[np.ndarray, np.ndarray]:
    """Sample-point coordinates covering `box` (cell centres of the supersampled grid)."""
    n = GRID * SUPERSAMPLE
    x0, y0, x1, y1 = box
    xs = x0 + (np.arange(n) + 0.5) / n * (x1 - x0)
    ys = y0 + (np.arange(n) + 0.5) / n * (y1 - y0)
    return np.meshgrid(xs, ys)


def _to_grid(inside: np.ndarray) -> np.ndarray:
    """(n*ss, n*ss) booleans -> (n, n) coverage fractions."""
    s = SUPERSAMPLE
    return inside.reshape(GRID, s, GRID, s).mean(axis=(1, 3))


def rasterize(polys: list[np.ndarray], box: Optional[BBox] = None) -> np.ndarray:
    """Coverage map of a filled outline, stretched to fill its bounding box.

    Uses the even-odd rule: holes of '0', '6', '8', '9' are separate contours.
    (Font outlines never self-overlap, so even-odd and nonzero agree, and
    even-odd is also right for converters that do not reverse hole contours.)
    """
    if box is None:
        pts = np.vstack(polys)
        box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    px, py = _sample_grid(box)
    px, py = px.ravel()[:, None], py.ravel()[:, None]
    parity = np.zeros(px.shape[0], bool)
    for poly in polys:
        a = np.asarray(poly, float)
        b = np.roll(a, -1, axis=0)           # closes the contour implicitly
        ax_, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        straddle = (ay <= py) != (by <= py)  # edge crosses the horizontal ray's height
        with np.errstate(divide="ignore", invalid="ignore"):
            xcross = ax_ + (py - ay) * (bx - ax_) / (by - ay)
        hits = straddle & (px < xcross)
        parity ^= (hits.sum(axis=1) % 2).astype(bool)
    n = GRID * SUPERSAMPLE
    return _to_grid(parity.reshape(n, n))


# --------------------------------------------------------------------------- templates
def _font_sources() -> list[tuple[str, Optional[str]]]:
    """(base-14 name, None) or (alias, font file path) for the template fonts."""
    out: list[tuple[str, Optional[str]]] = [("helv", None), ("hebo", None), ("tiro", None), ("tibo", None)]
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for f in ("arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "calibri.ttf", "calibrib.ttf",
              "segoeui.ttf", "tahoma.ttf", "verdana.ttf"):
        p = os.path.join(windir, "Fonts", f)
        if os.path.exists(p):
            out.append((f.split(".")[0], p))
    try:  # matplotlib's default font: SVG-exported figures often carry DejaVu outlines
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        for f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
            if os.path.exists(os.path.join(d, f)):
                out.append((f.split(".")[0].replace("-", ""), os.path.join(d, f)))
    except Exception:
        pass
    return out


def _render_char(ch: str, fontname: str, fontfile: Optional[str]) -> Optional[tuple[np.ndarray, float]]:
    """Render one character big, return (coverage map, aspect w/h of its ink)."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=260, height=260)
    try:
        if fontfile:
            page.insert_text((30, 210), ch, fontsize=200, fontname=fontname, fontfile=fontfile)
        else:
            page.insert_text((30, 210), ch, fontsize=200, fontname=fontname)
        pix = page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY)
    except Exception:
        return None
    img = 1.0 - np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width) / 255.0
    ys, xs = np.nonzero(img > 0.5)
    if len(xs) == 0:
        return None
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    px, py = _sample_grid(box)
    inside = img[np.clip(py.astype(int), 0, pix.height - 1), np.clip(px.astype(int), 0, pix.width - 1)] > 0.5
    return _to_grid(inside), (box[2] - box[0]) / (box[3] - box[1])


def _normalise(cov: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-length vector: then a dot product is the Pearson correlation."""
    v = cov.ravel() - cov.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@lru_cache(maxsize=4)
def templates(chars: str = DIGITS) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Templates of `chars` in every template font: (labels, (N, GRID*GRID) vectors, aspects).

    Built once per process and character set (~0.5 s for the digits).
    """
    labels, vecs, aspects = [], [], []
    for name, path in _font_sources():
        for ch in chars:
            r = _render_char(ch, name, path)
            if r is not None:
                labels.append(ch)
                vecs.append(_normalise(r[0]))
                aspects.append(r[1])
    return labels, np.array(vecs), np.array(aspects)


def classify(cov: np.ndarray, aspect: float, chars: str = DIGITS) -> tuple[str, float]:
    """Best character for a coverage map.

    score = correlation with the template minus a penalty for a different
    width/height ratio (the coverage map itself is stretched to a square,
    so the aspect ratio is the only size information left).
    """
    labels, vecs, aspects = templates(chars)
    scores = vecs @ _normalise(cov) - 0.5 * np.abs(np.log(max(aspect, 1e-3) / aspects))
    i = int(np.argmax(scores))
    return labels[i], float(scores[i])


def classify_digit(cov: np.ndarray, aspect: float) -> tuple[str, float]:
    if aspect < 0.22:      # a bare vertical bar: '1' in many sans fonts, nothing else is that thin
        return "1", 0.9
    return classify(cov, aspect, DIGITS)


# --------------------------------------------------------------------------- glyphs -> words
@dataclass
class _Glyph:
    polys: list[np.ndarray]          # outline contours (empty for rendered glyphs)
    bbox: BBox                       # ink bounding box, page coords
    cov: Optional[np.ndarray] = None # precomputed coverage map (rendered glyphs)

    def coverage(self) -> np.ndarray:
        return self.cov if self.cov is not None else rasterize(self.polys, self.bbox)

    @property
    def w(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self):
        return self.bbox[3] - self.bbox[1]


def _inside(inner: BBox, outer: BBox, tol: float = 0.05) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _glyphs(pc: PageContent, region: BBox, max_h: float) -> list[_Glyph]:
    """Filled closed contours inside `region`, grouped into glyphs.

    One PDF path may hold a whole number ("0.2" = 4 contours) or one glyph;
    either way a glyph is an outer contour plus every contour lying inside its
    bounding box (the counters of 0/4/6/8/9).
    """
    rx0, ry0, rx1, ry1 = region
    contours: list[tuple[np.ndarray, BBox]] = []
    for p in pc.paths:
        if p.fill is None or (p.stroke is not None and p.width > 0.6 * max_h):
            continue                 # glyphs are filled (a thin outline stroke is tolerated)
        b = p.bbox
        if b[2] < rx0 or b[0] > rx1 or b[3] < ry0 or b[1] > ry1:
            continue
        for sp in p.subpaths:
            if len(sp) < 3:
                continue
            sb = (sp[:, 0].min(), sp[:, 1].min(), sp[:, 0].max(), sp[:, 1].max())
            if sb[3] - sb[1] > max_h or sb[2] - sb[0] > max_h:
                continue
            if sb[0] >= rx0 and sb[2] <= rx1 and sb[1] >= ry0 and sb[3] <= ry1:
                contours.append((sp, sb))
    # big contours first, so each smaller one can find the glyph that contains it
    contours.sort(key=lambda c: -(c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))
    glyphs: list[_Glyph] = []
    seen = set()
    for sp, sb in contours:
        # a glyph painted twice (fill pass + outline pass) must count once; a doubled
        # counter contour would even cancel the hole of a '0' under the even-odd rule
        key = (len(sp),) + tuple(round(v, 1) for v in sb)
        if key in seen:
            continue
        seen.add(key)
        for g in glyphs:
            if _inside(sb, g.bbox) and (sb[2] - sb[0]) * (sb[3] - sb[1]) < 0.9 * g.w * g.h:
                g.polys.append(sp)
                break
        else:
            glyphs.append(_Glyph([sp], sb))
    return glyphs


def _tiny(g: _Glyph, ref: float) -> bool:
    """A minus bar or a decimal point next to glyphs of height `ref`."""
    if g.h >= 0.4 * ref:
        return False
    flat = g.w > 1.6 * g.h                                   # minus
    dot = g.h < 0.3 * ref and 0.5 * g.h <= g.w <= 2 * g.h     # point (roughly square)
    return flat or dot


def _joinable(a: _Glyph, b: _Glyph) -> bool:
    """May `b` (to the right of `a`) belong to the same word as `a`?"""
    big, small = max(a.h, b.h), min(a.h, b.h)
    if small >= 0.5 * big:
        if big > 1.5 * small:
            # very different sizes: only a superscript (10^15) may follow a digit;
            # anything else is e.g. a bold panel letter next to tick digits
            if not (b.h < a.h and b.bbox[3] < a.bbox[3] - 0.3 * a.h):
                return False
        ref = small                      # gaps scale with the smaller glyph
    elif _tiny(a, big) or _tiny(b, big):
        ref = big                        # '-' or '.' next to a digit: scale with the digit
    else:
        return False
    gap = b.bbox[0] - a.bbox[2]
    # same line: vertical overlap, with room for superscripts (10^-3)
    return (-0.3 * ref <= gap <= 0.45 * ref
            and b.bbox[1] < a.bbox[3] and b.bbox[3] > a.bbox[1])


def _words(glyphs: list[_Glyph]) -> list[list[_Glyph]]:
    """Connected groups of joinable glyphs (union-find), each sorted left to right."""
    glyphs = sorted(glyphs, key=lambda g: g.bbox[0])
    parent = list(range(len(glyphs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(glyphs):
        for j in range(i + 1, len(glyphs)):
            b = glyphs[j]
            if b.bbox[0] > a.bbox[2] + 0.5 * max(a.h, b.h, 1.0):
                break                    # sorted by x0: nothing further can be adjacent
            if _joinable(a, b):
                parent[root(j)] = root(i)
    groups: dict[int, list[_Glyph]] = {}
    for i, g in enumerate(glyphs):
        groups.setdefault(root(i), []).append(g)
    return list(groups.values())


# Overlapping label zones see the same glyphs again: remember results by glyph
# geometry (only for the most recent page, so memory stays bounded).
_glyph_cache: dict = {"owner": None, "map": {}}
MAX_GLYPH_VERTICES = 400      # more than this is a drawing, not a digit: skip the raster work


def _classify_cached(g: _Glyph, aspect: float) -> tuple[str, float]:
    if g.cov is None and sum(len(p) for p in g.polys) > MAX_GLYPH_VERTICES:
        return "?", 0.0
    key = (round(g.bbox[0], 2), round(g.bbox[1], 2), round(g.bbox[2], 2), round(g.bbox[3], 2),
           sum(len(p) for p in g.polys))
    m = _glyph_cache["map"]
    if g.cov is not None or key not in m:
        res = classify_digit(g.coverage(), aspect)
        if g.cov is not None:
            return res
        m[key] = res
    return m[key]


def _read_word(word: list[_Glyph]) -> tuple[str, float]:
    """Glyphs of one word -> string; '^' marks the start of a superscript run."""
    H = max(g.h for g in word)                      # digit height ~ cap height
    tallest = max(word, key=lambda g: g.h)          # a full-size digit, never a superscript
    base, top = tallest.bbox[3], tallest.bbox[1]    # its baseline and top line
    chars, worst = [], 1.0
    in_sup = False
    for g in word:
        mid = (g.bbox[1] + g.bbox[3]) / 2
        small = g.h < 0.35 * H
        raised = False
        if small and g.w > 1.6 * g.h and top + 0.2 * H < mid < base - 0.2 * H:
            ch, s = "-", 1.0                        # minus: a flat bar at mid height
        elif small and g.w < 0.45 * H and g.bbox[3] > base - 0.15 * H:
            ch, s = ".", 1.0                        # point: small blob on the baseline
        elif small and g.w > 1.6 * g.h and mid <= top + 0.2 * H:
            ch, s, raised = "-", 0.9, True          # superscript minus (10^-3)
        elif small:
            ch, s = "?", 0.0
        else:
            aspect = g.w / g.h if g.h > 0 else 0
            ch, s = _classify_cached(g, aspect)
            if s < MIN_SCORE:
                ch = "?"
            # a digit whose bottom is well above the baseline is a superscript
            raised = g.bbox[3] < base - 0.3 * H and g.h < 0.85 * H
        if raised and not in_sup:
            chars.append("^")
            in_sup = True
        chars.append(ch)
        worst = min(worst, s)
    return "".join(chars), worst


def glyph_words(pc: PageContent, region: BBox, max_h: float = 16.0) -> list[Word]:
    """Recognise outlined glyph words inside `region` (page coords)."""
    if _glyph_cache["owner"] is not pc:
        _glyph_cache["owner"], _glyph_cache["map"] = pc, {}
    out = []
    for w in _words(_glyphs(pc, region, max_h)):
        text, score = _read_word(w)
        x0 = min(g.bbox[0] for g in w)
        y0 = min(g.bbox[1] for g in w)
        x1 = max(g.bbox[2] for g in w)
        y1 = max(g.bbox[3] for g in w)
        out.append(Word(text, (x0, y0, x1, y1), score, "glyph"))
    return out


def unreadable(text: str) -> bool:
    """Text that extracted as control / private-use / replacement characters
    (typically a Type-3 font without a unicode map)."""
    return any(ord(c) < 32 or 0xE000 <= ord(c) <= 0xF8FF or c == "\ufffd" for c in text)


def rendered_words(page, region: BBox) -> list[Word]:
    """Read text spans whose characters are unreadable codes by *looking* at them.

    The PDF still says where every character is (rawdict char boxes), so we render
    each box, take the ink inside it as one glyph and template-match it exactly like
    an outlined glyph.  Needs the `page`; returns [] without it.
    """
    if page is None:
        return []
    import pymupdf
    out = []
    try:
        raw = page.get_text("rawdict", clip=pymupdf.Rect(region))
    except Exception:
        return []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            if abs(line["dir"][0] - 1) > 0.05:
                continue
            for span in line["spans"]:
                text = "".join(c["c"] for c in span["chars"])
                if not span["chars"] or not unreadable(text):
                    continue
                sb = pymupdf.Rect(span["bbox"])
                if sb.height <= 0 or sb.width <= 0:
                    continue
                dpi = int(min(2400, 72 * 64 / sb.height))   # ~64 px per line of text
                pix = page.get_pixmap(clip=sb, dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
                img = 1.0 - np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width) / 255.0
                k = pix.width / sb.width                   # pixels per point
                glyphs = []
                for c in span["chars"]:
                    cx0 = int(round((c["bbox"][0] - sb.x0) * k))
                    cx1 = int(round((c["bbox"][2] - sb.x0) * k))
                    sub = img[:, max(cx0, 0):max(cx1, cx0 + 1)]
                    ys, xs = np.nonzero(sub > 0.5)
                    if len(xs) == 0:
                        continue                           # a space
                    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                    px, py = _sample_grid(box)
                    inside = sub[np.clip(py.astype(int), 0, sub.shape[0] - 1),
                                 np.clip(px.astype(int), 0, sub.shape[1] - 1)] > 0.5
                    bbox = (sb.x0 + (cx0 + box[0]) / k, sb.y0 + box[1] / k,
                            sb.x0 + (cx0 + box[2]) / k, sb.y0 + box[3] / k)
                    glyphs.append(_Glyph([], bbox, _to_grid(inside)))
                for w in _words(glyphs):
                    text, score = _read_word(w)
                    out.append(Word(text, (min(g.bbox[0] for g in w), min(g.bbox[1] for g in w),
                                           max(g.bbox[2] for g in w), max(g.bbox[3] for g in w)),
                                    score, "glyph"))
    return out


PANEL_LETTERS = "abcdefghABCDEFGH"


def glyph_letters(pc: PageContent, region: BBox, max_h: float = 24.0) -> list[Word]:
    """Outlined panel letters ("a", "(b)", "c)") inside `region`.

    Parentheses are recognised by shape (tall, thin, at the ends of the word)
    and dropped; the remaining single glyph is matched against letter templates.
    """
    out = []
    for w in _words(_glyphs(pc, region, max_h)):
        if len(w) > 3:
            continue
        core = list(w)
        while len(core) > 1 and core[-1].w < 0.5 * core[-1].h and core[-1].h >= 1.1 * min(g.h for g in core):
            core.pop()                               # closing parenthesis
        while len(core) > 1 and core[0].w < 0.5 * core[0].h and core[0].h >= 1.1 * min(g.h for g in core):
            core.pop(0)                              # opening parenthesis
        if len(core) != 1 or core[0].h < 3:
            continue
        g = core[0]
        ch, score = classify(g.coverage(), g.w / g.h, PANEL_LETTERS)
        if score >= 0.75:
            out.append(Word(ch, g.bbox, score, "glyph"))
    return out


# --------------------------------------------------------------------------- optional pixel OCR
@lru_cache(maxsize=1)
def _engine():
    try:
        from rapidocr_onnxruntime import RapidOCR  # optional dependency
    except Exception:
        return None
    try:
        return RapidOCR()
    except Exception:
        return None


def ocr_available() -> bool:
    return _engine() is not None


def ocr_region(page, clip: BBox, dpi: int = 600) -> list[Word]:
    """OCR the rendered `clip` of `page`; returns [] when no engine is installed."""
    eng = _engine()
    if eng is None or page is None:
        return []
    import pymupdf
    rect = pymupdf.Rect(clip) & page.rect
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return []
    pix = page.get_pixmap(clip=rect, dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    # small crops OCR badly: pad with white so the detector sees some margin
    pad = 32
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
    try:
        res, _ = eng(img)
    except Exception:
        return []
    out = []
    k = 72.0 / dpi
    for quad, text, conf in res or []:
        q = np.asarray(quad, float) - pad
        x0, y0 = q.min(axis=0) * k
        x1, y1 = q.max(axis=0) * k
        text = str(text)
        n = max(len(text), 1)
        # one detected line may hold several labels ("0.0 0.2 0.4"): split at spaces
        for m in re.finditer(r"\S+", text):
            bx0 = x0 + (x1 - x0) * m.start() / n
            bx1 = x0 + (x1 - x0) * m.end() / n
            out.append(Word(m.group(), (rect.x0 + bx0, rect.y0 + y0, rect.x0 + bx1, rect.y0 + y1),
                            float(conf), "ocr"))
    return out
