"""Tests for figrip.series.extract_series.

Most tests draw a small figure with matplotlib (or PyMuPDF directly), so the
exact page position of every data point is known, and check that the series
come back with sub-0.1 pt accuracy.  Coordinates: matplotlib display units are
points at dpi=72, and PyMuPDF's page y grows downwards, so
``page = (x_disp, fig_height_pt - y_disp)``.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from figrip.model import AxisCalib, FigureResult, Panel
from figrip.primitives import open_page
from figrip.series import extract_series

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZHENG = Path("D:/Zotero/storage/QIZR5ARD/Zheng_et_al_2018_Adv_Mater_Fine_tuned_HTL_high_FF_OSCs.pdf")


# --------------------------------------------------------------------------- helpers
def _panel(bbox, x=None, y=None) -> Panel:
    """Panel with identity-ish calibration (data = page coords, y flipped)."""
    return Panel(0, tuple(bbox), x or AxisCalib("x", "linear", 1.0, 0.0),
                 y or AxisCalib("y", "linear", -1.0, 0.0))


class Fig:
    """A matplotlib figure saved to PDF, with helpers to map data -> page coords."""

    def __init__(self, tmp_path, w=4.0, h=3.0, name="fig.pdf"):
        plt.rcParams.update({"pdf.fonttype": 42, "figure.dpi": 72, "savefig.dpi": 72})
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        self.H = h * 72
        self.path = tmp_path / name

    def page(self, x, y) -> np.ndarray:
        d = self.ax.transData.transform(np.column_stack([x, y]))
        return np.column_stack([d[:, 0], self.H - d[:, 1]])

    def save(self):
        self.fig.canvas.draw()
        bb = self.ax.get_window_extent()
        self.bbox = (bb.x0, self.H - bb.y1, bb.x1, self.H - bb.y0)
        self.fig.savefig(self.path)          # no bbox_inches: keep the coordinates
        plt.close(self.fig)
        return self

    def extract(self):
        return extract_series(open_page(str(self.path), 0), _panel(self.bbox))


def _poly_dist(pts, poly) -> np.ndarray:
    """Distance from each point to polyline ``poly``."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(1), 1e-24)
    out = []
    for p in pts:
        t = np.clip(((p - a) * ab).sum(1) / l2, 0, 1)
        out.append(np.hypot(*(a + t[:, None] * ab - p).T).min())
    return np.array(out)


def _nearest(pts, ref) -> np.ndarray:
    """For each ref point the distance to the closest of ``pts``."""
    return np.array([np.hypot(*(pts - r).T).min() for r in ref])


def _by_color(series, rgb, tol=0.05):
    hits = [s for s in series if max(abs(a - b) for a, b in zip(s.style["color"], rgb)) <= tol]
    assert len(hits) == 1, f"expected one series of colour {rgb}, got {len(hits)}"
    return hits[0]


C0 = mpl.colors.to_rgb("C0")
C1 = mpl.colors.to_rgb("C1")
C3 = mpl.colors.to_rgb("C3")


# --------------------------------------------------------------------------- matplotlib basics
def test_single_line(tmp_path):
    f = Fig(tmp_path)
    x = np.linspace(0, 10, 80)
    y = np.sin(x) + 0.1 * x
    f.ax.plot(x, y, color="C0")
    f.save()
    (s,) = f.extract()
    assert s.kind == "line" and s.style["marker"] is None
    truth = f.page(x, y)
    assert len(s.page_pts) == len(truth)
    assert np.abs(s.page_pts - truth).max() < 0.1
    assert np.all(np.diff(s.page_pts[:, 0]) > 0)          # ordered by x


def test_multicolour_dashed_and_legend(tmp_path):
    f = Fig(tmp_path)
    x = np.linspace(0, 10, 60)
    ys = {"Sample A": (np.sin(x), "C0", "-"), "Sample B": (np.cos(x), "C3", "-"),
          "Baseline": (0.1 * x - 1, "C1", "--")}
    for name, (y, c, ls) in ys.items():
        f.ax.plot(x, y, color=c, ls=ls, label=name)
    f.ax.legend(loc="upper right")
    f.save()
    ss = f.extract()
    assert len(ss) == 3
    for name, (y, c, ls) in ys.items():
        s = _by_color(ss, mpl.colors.to_rgb(c))
        assert s.name == name
        assert (s.style["dashes"] is not None) == (ls == "--")
        assert np.abs(s.page_pts - f.page(x, y)).max() < 0.1


@pytest.mark.parametrize("marker,shape", [
    ("o", "circle"), ("s", "square"), ("^", "triangle_up"), ("v", "triangle_down"),
    ("<", "triangle_left"), (">", "triangle_right"), ("D", "diamond"), ("p", "pentagon"),
    ("h", "hexagon"), ("*", "star"), ("P", "plus"), ("X", "x"),
])
def test_marker_shapes(tmp_path, marker, shape):
    f = Fig(tmp_path)
    rng = np.random.default_rng(0)
    x = np.linspace(1, 9, 15)
    y = rng.normal(5, 2, x.size)
    f.ax.plot(x, y, marker, ms=6, color="C3")
    f.ax.set(xlim=(0, 10), ylim=(-2, 12))
    f.save()
    (s,) = f.extract()
    assert s.kind == "markers"
    assert s.style["marker"] == shape
    assert len(s.page_pts) == x.size
    assert _nearest(s.page_pts, f.page(x, y)).max() < 0.1


def test_open_markers(tmp_path):
    f = Fig(tmp_path)
    x = np.arange(1, 10)
    f.ax.plot(x, x * 0.5, "o", mfc="none", ms=6, color="C0", label="open")
    f.ax.plot(x, x * 0.5 + 2, "s", ms=6, color="C0", label="filled")
    f.ax.legend(loc="lower right")
    f.save()
    ss = f.extract()
    assert len(ss) == 2
    o = next(s for s in ss if s.style["marker"] == "circle")
    q = next(s for s in ss if s.style["marker"] == "square")
    assert o.style["marker_filled"] is False and q.style["marker_filled"] is True
    assert o.name == "open" and q.name == "filled"
    assert _nearest(o.page_pts, f.page(x, x * 0.5)).max() < 0.1


def test_line_markers_markevery(tmp_path):
    f = Fig(tmp_path)
    x = np.arange(300, 901, 5.0)
    y = 80 * np.exp(-((x - 600) / 200) ** 2)
    f.ax.plot(x, y, "-o", ms=3, markevery=4, color="C3", label="EQE")
    f.ax.legend(loc="lower center")
    f.ax.set(xlim=(300, 900), ylim=(0, 100))
    f.save()
    (s,) = f.extract()
    assert s.kind == "line+markers" and s.name == "EQE"
    assert s.style["marker"] == "circle"
    assert np.abs(s.page_pts - f.page(x, y)).max() < 0.1          # line: every point
    mk = s.style["marker_pts"]
    assert len(mk) == len(x[::4])
    assert np.abs(mk - f.page(x[::4], y[::4])).max() < 0.1          # markers: every 4th


def test_grid_zero_line_and_errorbars(tmp_path):
    f = Fig(tmp_path)
    x = np.arange(1, 13)
    y = 3 + 6 * (1 - np.exp(-x / 4.0)) - 4
    f.ax.errorbar(x, y, yerr=0.5, fmt="-o", ms=3.5, capsize=2.5, color="C0", label="Measured")
    f.ax.axhline(0, color="0.4", lw=0.6)
    f.ax.axvline(6.5, color="k", ls=":", lw=0.6)
    f.ax.grid(True)
    f.ax.minorticks_on()
    f.ax.tick_params(which="both", direction="in", top=True, right=True)
    f.ax.legend(loc="lower right")
    f.ax.set(xlim=(0, 13), ylim=(-3, 7))
    f.save()
    (s,) = f.extract()
    assert s.kind == "line+markers" and s.name == "Measured"
    assert np.abs(s.page_pts - f.page(x, y)).max() < 0.1


def test_line_leaving_plot_is_clipped(tmp_path):
    f = Fig(tmp_path)
    x = np.linspace(-0.2, 1.0, 50)
    y = -20 + 2e-6 * np.exp(x / 0.05)              # J-V like: shoots up past the top
    f.ax.plot(x, y, color="C0")
    f.ax.set(xlim=(-0.2, 1.0), ylim=(-25, 10))
    f.save()
    (s,) = f.extract()
    x0, y0, x1, y1 = f.bbox
    assert s.page_pts[:, 1].min() >= y0 - 0.51
    assert _poly_dist(s.page_pts, f.page(x, y)).max() < 0.1


# --------------------------------------------------------------------------- outlined strokes
def _offset_polygon(pts, width):
    """Outline of a stroked polyline with miter joins and butt caps (what 'outline stroke' does)."""
    d = np.diff(pts, axis=0)
    t = d / np.hypot(*d.T)[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    nv = np.empty_like(pts)
    nv[0], nv[-1] = n[0], n[-1]
    for i in range(1, len(pts) - 1):
        m = n[i - 1] + n[i]
        m /= np.hypot(*m)
        nv[i] = m / (m @ n[i])                      # miter: reach width/2 from both segments
    return np.vstack([pts + width / 2 * nv, (pts - width / 2 * nv)[::-1]])


def _jv_page_curve():
    v = np.linspace(0, 1, 70)
    j = 1 - 1e-6 * np.exp(v / 0.07)
    return np.column_stack([60 + 200 * v, 80 + 120 * (1 - j)])  # page coords, y down


@pytest.mark.parametrize("width", [0.5, 0.75, 1.0])
def test_outlined_stroke_centreline(tmp_path, width):
    curve = _jv_page_curve()
    curve = curve[curve[:, 1] < 250]
    doc = pymupdf.open()
    page = doc.new_page(width=320, height=300)
    sh = page.new_shape()
    sh.draw_polyline([tuple(p) for p in _offset_polygon(curve, width)])
    sh.finish(fill=(0.8, 0.1, 0.1), color=None, closePath=True)
    # a filled frame and a few filled ticks, the way outlined figures draw them
    for r in [(50, 60, 280, 60.4), (50, 260, 280, 260.4), (50, 60, 50.4, 260.4), (280, 60, 280.4, 260.4)]:
        sh.draw_rect(pymupdf.Rect(*r))
    for x in range(70, 280, 40):
        sh.draw_rect(pymupdf.Rect(x - 0.2, 256.5, x + 0.2, 260))
    sh.finish(fill=(0, 0, 0), color=None)
    sh.commit()
    p = tmp_path / "outl.pdf"
    doc.save(p)
    (s,) = extract_series(open_page(str(p), 0), _panel((50.2, 60.2, 280.2, 260.2)))
    assert s.style["outlined"] and abs(s.style["width"] - width) < 0.05
    inside = curve[(curve[:, 1] <= 260.2)]
    assert _poly_dist(s.page_pts, curve).max() < 0.05           # every point on the true curve
    assert _nearest(s.page_pts, inside).max() < 0.05             # every true vertex recovered


def test_origin_style_segments_and_outlined_markers(tmp_path):
    """Origin export: every segment a filled thin rectangle, markers = filled body + outlined edge."""
    doc = pymupdf.open()
    page = doc.new_page(width=320, height=300)
    sh = page.new_shape()
    x = np.linspace(60, 270, 30)
    curves = [np.column_stack([x, 100 + 0.002 * (x - 60) ** 2]),
              np.column_stack([x, 200 - 30 * np.sin((x - 60) / 40)])]
    w = 0.7
    for c in curves:                                  # same colour for both: separated by geometry
        for a, b in zip(c[:-1], c[1:]):
            t = (b - a) / np.hypot(*(b - a))
            nrm = np.array([-t[1], t[0]]) * w / 2
            sh.draw_polyline([tuple(b + nrm), tuple(a + nrm), tuple(a - nrm), tuple(b - nrm)])
            sh.finish(fill=(0.1, 0.1, 0.1), color=None, closePath=False)
    for c, kind in zip(curves, ("square", "circle")):
        for p in c[::3]:
            r = pymupdf.Rect(p[0] - 1.5, p[1] - 1.5, p[0] + 1.5, p[1] + 1.5)
            if kind == "square":
                sh.draw_rect(r)
            else:
                sh.draw_circle(pymupdf.Point(*p), 1.5)
            sh.finish(fill=(0.1, 0.1, 0.1), color=None)
            # the marker edge: a thin ring, outer and inner contour in one even-odd fill
            if kind == "square":
                sh.draw_rect(r + (-0.1, -0.1, 0.1, 0.1))
                sh.draw_rect(r + (0.1, 0.1, -0.1, -0.1))
            else:
                sh.draw_circle(pymupdf.Point(*p), 1.6)
                sh.draw_circle(pymupdf.Point(*p), 1.4)
            sh.finish(fill=(0.1, 0.1, 0.1), color=None, even_odd=True)
    sh.draw_rect(pymupdf.Rect(50, 60, 280, 60.4))      # frame top as an outlined line
    sh.draw_rect(pymupdf.Rect(50, 150, 280, 150.4))    # a full-width reference line
    sh.finish(fill=(0.1, 0.1, 0.1), color=None)
    sh.commit()
    p = tmp_path / "origin.pdf"
    doc.save(p)
    ss = extract_series(open_page(str(p), 0), _panel((50.2, 60.2, 280.2, 260.2)))
    assert len(ss) == 2
    for c, kind in zip(curves, ("square", "circle")):
        s = next(s for s in ss if s.style["marker"] == kind)
        assert s.kind == "line+markers" and s.style["marker_filled"]
        assert np.abs(s.page_pts - c).max() < 0.05
        assert np.abs(s.style["marker_pts"] - c[::3]).max() < 0.1


def _page_with(draw) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=320, height=300)
    sh = page.new_shape()
    draw(sh)
    sh.commit()
    return doc


def test_stroked_segments_gap_to_symbol_and_outlined_dashes(tmp_path):
    """Three more Origin/Illustrator habits in one plot (all in the same colour):

    * a curve stroked segment by segment (one path per segment),
    * a line+symbol curve whose line stops short of every symbol ("gap to symbol"),
    * a dashed curve whose dashes were outlined into separate thin polygons.
    """
    x = np.linspace(60, 270, 22)
    seg = np.column_stack([x, 90 + 20 * np.sin((x - 60) / 30)])
    gap = np.column_stack([x, 140 + 0.0015 * (x - 60) ** 2])
    dash = np.column_stack([np.linspace(60, 270, 200), 235 - 12 * np.cos(np.linspace(0, 6, 200))])
    col = (0.1, 0.2, 0.7)

    def draw(sh):
        for a, b in zip(seg[:-1], seg[1:]):
            sh.draw_line(tuple(a), tuple(b))
            sh.finish(color=col, width=0.8)
        r = 2.0
        for a, b in zip(gap[:-1], gap[1:]):
            u = (b - a) / np.hypot(*(b - a))
            sh.draw_line(tuple(a + u * (r + 0.8)), tuple(b - u * (r + 0.8)))
            sh.finish(color=col, width=0.8)
        for p in gap:
            sh.draw_rect(pymupdf.Rect(p[0] - r, p[1] - r, p[0] + r, p[1] + r))
            sh.finish(color=None, fill=col)
        # dashes: 6 vertices on, 3 off, each dash outlined as a thin polygon
        for k in range(0, len(dash) - 6, 9):
            piece = dash[k:k + 7]
            sh.draw_polyline([tuple(p) for p in _offset_polygon(piece, 0.6)])
            sh.finish(fill=col, color=None, closePath=True)
    p = tmp_path / "habits.pdf"
    _page_with(draw).save(p)
    ss = extract_series(open_page(str(p), 0), _panel((50, 60, 280, 260)))
    assert len(ss) == 3, [(s.kind, len(s.page_pts)) for s in ss]
    by_y = sorted(ss, key=lambda s: s.page_pts[:, 1].mean())
    assert by_y[0].kind == "line" and np.abs(by_y[0].page_pts - seg).max() < 0.01
    assert by_y[1].kind == "line+markers" and by_y[1].style["marker"] == "square"
    assert np.abs(by_y[1].style["marker_pts"] - gap).max() < 0.01
    assert by_y[2].kind == "line" and by_y[2].style["outlined"]
    assert _poly_dist(by_y[2].page_pts, dash).max() < 0.05


# --------------------------------------------------------------------------- synthetic corpus
def _corpus():
    path = ROOT / "tests" / "corpus" / "make_corpus.py"
    if not path.exists():
        pytest.skip("synthetic corpus generator not present")
    spec = importlib.util.spec_from_file_location("make_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = ROOT / "tests" / "corpus" / "out"
    mod.build(out)
    spec2 = importlib.util.spec_from_file_location("corpus_score", ROOT / "tests" / "corpus" / "score.py")
    sc = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(sc)
    return out, sc


def _truth_panel(tp, page_no=0) -> Panel:
    """Panel calibrated from the ground truth (so only series extraction is tested)."""
    x0, y0, x1, y1 = tp["bbox_hint"]

    def cal(orient, p0, p1, lim, scale):
        f = np.log10 if scale == "log" else (lambda v: v)
        a = (f(lim[1]) - f(lim[0])) / (p1 - p0)
        return AxisCalib(orient, scale, float(a), float(f(lim[0]) - a * p0), source="manual")
    return Panel(page_no, (x0, y0, x1, y1), cal("x", x0, x1, tp["xlim"], tp["xscale"]),
                 cal("y", y1, y0, tp["ylim"], tp["yscale"]))


CORPUS_CASES = ["linear_single", "linear_multi_legend", "markers_only", "line_markers", "log_y",
                "loglog", "negative_ticks", "inverted_y", "sci_notation", "grid", "subplots_2x2",
                "errorbars", "jv_opv", "eqe", "outlined_text", "outlined_strokes", "outlined_all",
                "multi_figure_page"]


@pytest.mark.parametrize("case", CORPUS_CASES)
def test_corpus_series(case):
    out, sc = _corpus()
    js = out / f"{case}.json"
    if not js.exists():
        pytest.skip(f"{case} not in corpus")
    truth = json.loads(js.read_text(encoding="utf-8"))
    pc = open_page(str(out / truth["pdf"]), truth["page"])
    results = []
    for tp in truth["panels"]:
        panel = _truth_panel(tp, truth["page"])
        results.append(FigureResult(panel, extract_series(pc, panel)))
    rep = sc.score(truth, results, tol=0.001)        # 0.1 % of the axis range
    assert rep["ok"], sc.format_report(rep)
    for p in rep["panels"]:
        assert p["extra_series"] == 0, sc.format_report(rep)
        for s in p["series"]:
            # names must come back whenever the legend is real text
            if s["truth_name"] and "outlined_text" not in truth["features"]:
                assert s["found_name"] == s["truth_name"], sc.format_report(rep)


# --------------------------------------------------------------------------- the real paper
@pytest.mark.skipif(not ZHENG.exists(), reason="user's Zotero PDF not available")
def test_zheng_2018_fig2d_outlined_jv():
    """Every drawing is a fill (Origin segments as thin rectangles, glyph-outlined text).

    Hand calibration from the tick marks of Fig. 2d; Table 2 on the same page gives
    Voc 0.78 / 0.87 / 0.87 V and Jsc 16.78 / 20.73 / 20.68 mA cm-2.
    """
    with pymupdf.open(ZHENG) as doc:
        pc = __import__("figrip.primitives", fromlist=["x"]).extract_page(doc[3])
    xcal = AxisCalib.fit("x", [(252.515, 0.0), (278.855, 0.2), (305.2, 0.4), (331.49, 0.6), (357.835, 0.8)])
    ycal = AxisCalib.fit("y", [(287.02, 0), (268.43, -5), (249.895, -10), (231.36, -15), (212.82, -20)])
    ss = extract_series(pc, Panel(3, (245.93, 201.68, 371.0, 294.42), xcal, ycal))
    assert len(ss) == 3
    assert sorted(s.style["marker"] for s in ss) == ["circle", "square", "triangle_up"]
    table = {"square": (16.78, 0.78), "circle": (20.73, 0.87), "triangle_up": (20.68, 0.87)}
    for s in ss:
        assert s.kind == "line+markers" and s.style["outlined"]
        v, j = s.data[:, 0], s.data[:, 1]
        jsc = -np.interp(0.0, v, j)
        k = np.where(np.diff(np.sign(j)) != 0)[0][0]
        voc = v[k] - j[k] * (v[k + 1] - v[k]) / (j[k + 1] - j[k])
        jsc_t, voc_t = table[s.style["marker"]]
        assert abs(jsc - jsc_t) < 0.1, (s.style["marker"], jsc)
        assert abs(voc - voc_t) < 0.01, (s.style["marker"], voc)
