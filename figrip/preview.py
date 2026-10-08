"""Overlay preview: the page region around a panel with every extracted point drawn on top.

This image is how a user decides whether to trust an extraction, so it shows
(1) the extracted points as small coloured crosses, one colour per series,
(2) the calibrated tick positions *with the numeric values the program believes they have*,
(3) a legend (series name, point count) and a one-line summary of the axis ranges.

Implementation note - we never modify the user's PDF page.  Instead we build a fresh one-page
PDF, paste the cropped (vector!) source region into it with ``show_pdf_page`` and draw the
overlay there, leaving room on the right for the legend.  Rendering then is just ``get_pixmap``.
Only PyMuPDF is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pymupdf

from .export import series_names
from .model import FigureResult

# distinguishable on white paper and on typical coloured curves; (r, g, b) in 0..1
PALETTE = [
    (0.84, 0.10, 0.11), (0.12, 0.33, 0.85), (0.09, 0.62, 0.25), (0.80, 0.20, 0.80),
    (0.95, 0.55, 0.00), (0.00, 0.65, 0.70), (0.45, 0.25, 0.10), (0.35, 0.35, 0.35),
]
TICK_COLOR = (0.85, 0.35, 0.0)
MAX_MARKERS = 500            # draw at most this many crosses per series (thousands just paint a blob)


def _safe(text: str) -> str:
    """Built-in PDF fonts are Latin-1 only; replace anything else so insert_text never fails."""
    return text.encode("latin-1", "replace").decode("latin-1")


def _fmt_val(v: float) -> str:
    return f"{v:.4g}"


def render_overlay(pdf_path: Union[str, Path], result: FigureResult, out_png: Union[str, Path],
                   dpi: int = 200, margin: float = 36.0) -> Path:
    """Write ``out_png`` showing the panel (plus ``margin`` pt around it for tick labels) with the overlay.

    Returns the path of the PNG.
    """
    panel = result.panel
    x0, y0, x1, y1 = panel.bbox

    with pymupdf.open(str(pdf_path)) as src:
        spage = src[panel.page_no]
        # more room left/bottom for tick labels + axis titles
        clip = pymupdf.Rect(x0 - margin * 1.6, y0 - margin * 0.6, x1 + margin * 0.6, y1 + margin * 1.6)
        clip = clip & spage.rect
        cw, ch = clip.width, clip.height

        # ---- legend sizing -------------------------------------------------
        names = series_names(result)
        fs = 8.0
        font = "helv"
        lines = [f"{n}  ({len(s.data)} pts)" for n, s in zip(names, result.series)]
        legend_w = max([pymupdf.get_text_length(_safe(t), font, fs) for t in lines] + [110]) + 30
        title = _summary_lines(result)
        title_w = max(pymupdf.get_text_length(_safe(t), font, fs) for t in title) + 16
        legend_w = max(legend_w, min(title_w, 260))
        total_h = max(ch, 14 + 11 * (len(title) + len(lines) + 2))
        W = cw + legend_w + 8

        out = pymupdf.open()
        page = out.new_page(width=W, height=total_h)
        # white background so that transparent PDF pages do not render black
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
        page.show_pdf_page(pymupdf.Rect(0, 0, cw, ch), src, panel.page_no, clip=clip)

        def P(px, py):                          # source page coords -> new page coords
            return pymupdf.Point(px - clip.x0, py - clip.y0)

        # ---- plot-area frame -------------------------------------------------
        page.draw_rect(pymupdf.Rect(P(x0, y0), P(x1, y1)), color=(0.5, 0.5, 0.5), width=0.4, dashes="[2 2] 0")

        # ---- calibrated ticks + the values we assigned to them ------------------
        for pos, val in panel.x.ticks:          # x ticks sit on the bottom edge of the plot area
            _draw_tick(page, P(pos, y1), val, horizontal_axis=True)
        for pos, val in panel.y.ticks:          # y ticks sit on the left edge
            _draw_tick(page, P(x0, pos), val, horizontal_axis=False)

        # ---- extracted points -----------------------------------------------
        for i, s in enumerate(result.series):
            color = PALETTE[i % len(PALETTE)]
            pts = np.asarray(s.page_pts, float).reshape(-1, 2)
            if len(pts) > MAX_MARKERS:
                pts = pts[np.linspace(0, len(pts) - 1, MAX_MARKERS).astype(int)]
            shape = page.new_shape()
            r = 2.2                              # arm length in pt: visible but not covering the curve
            for px, py in pts:
                c = P(px, py)
                shape.draw_line(pymupdf.Point(c.x - r, c.y - r), pymupdf.Point(c.x + r, c.y + r))
                shape.draw_line(pymupdf.Point(c.x - r, c.y + r), pymupdf.Point(c.x + r, c.y - r))
            shape.finish(color=color, width=0.7)
            shape.commit()

        # ---- legend / summary on the right ------------------------------------
        lx, ly = cw + 8, 12
        for t in title:
            page.insert_text((lx, ly), _safe(t), fontname=font, fontsize=fs, color=(0.1, 0.1, 0.1))
            ly += 11
        ly += 6
        for i, t in enumerate(lines):
            color = PALETTE[i % len(PALETTE)]
            cx, cy = lx + 5, ly - 3
            page.draw_line((cx - 3, cy - 3), (cx + 3, cy + 3), color=color, width=1.2)
            page.draw_line((cx - 3, cy + 3), (cx + 3, cy - 3), color=color, width=1.2)
            page.insert_text((lx + 14, ly), _safe(t), fontname=font, fontsize=fs, color=(0.1, 0.1, 0.1))
            ly += 11
        if not lines:
            page.insert_text((lx, ly), "(no series found)", fontname=font, fontsize=fs, color=(0.8, 0, 0))

        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(dpi=dpi, alpha=False).save(str(out_png))
        out.close()
    return out_png


def _summary_lines(result: FigureResult) -> list[str]:
    p = result.panel
    xs = p.x.to_data([p.bbox[0], p.bbox[2]])
    ys = p.y.to_data([p.bbox[3], p.bbox[1]])
    tag = f"page {p.page_no + 1}" + (f" panel {p.label}" if p.label else "")
    out = [tag,
           f"x: {p.x.label or '?'}  [{_fmt_val(xs[0])} .. {_fmt_val(xs[1])}] ({p.x.scale}, {p.x.source})",
           f"y: {p.y.label or '?'}  [{_fmt_val(ys[0])} .. {_fmt_val(ys[1])}] ({p.y.scale}, {p.y.source})"]
    out += [f"! {n}" for n in p.notes]
    return out


def _draw_tick(page, at: pymupdf.Point, value: float, horizontal_axis: bool) -> None:
    """Orange tick mark just inside the frame and a small white-backed label with the calibrated value.

    Labels go *inside* the plot area so they do not collide with the paper's own tick labels:
    if the two numbers agree you trust the calibration at a glance.
    """
    L = 5.0
    txt = _fmt_val(value)
    fs = 6.5
    tw = pymupdf.get_text_length(txt, "helv", fs)
    if horizontal_axis:                          # tick at the bottom edge, pointing up
        page.draw_line(at, pymupdf.Point(at.x, at.y - L), color=TICK_COLOR, width=1.0)
        r = pymupdf.Rect(at.x - tw / 2 - 1, at.y - L - fs - 2, at.x + tw / 2 + 1, at.y - L)
        base = (r.x0 + 1, r.y1 - 1.5)
    else:                                        # tick at the left edge, pointing right
        page.draw_line(at, pymupdf.Point(at.x + L, at.y), color=TICK_COLOR, width=1.0)
        r = pymupdf.Rect(at.x + L + 1, at.y - fs / 2 - 1, at.x + L + tw + 3, at.y + fs / 2 + 1)
        base = (r.x0 + 1, r.y1 - 2)
    page.draw_rect(r, color=None, fill=(1, 1, 0.85), fill_opacity=0.85)
    page.insert_text(base, txt, fontname="helv", fontsize=fs, color=(0.6, 0.2, 0.0))
