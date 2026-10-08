"""Synthetic test corpus with exact ground truth for figrip.

    python tests/corpus/make_corpus.py            # (re)generate into tests/corpus/out/
    python tests/corpus/make_corpus.py --force    # ignore the cache

API
---
``build(outdir=None, force=False) -> list[dict]``
    Generates ``<case>.pdf`` + ``<case>.json`` for every case in ``CASE_NAMES``
    (skips regeneration when every output file is newer than this script).
    Returns the ground-truth dicts (each has an extra ``"pdf"`` key = basename).

Ground-truth JSON (per case)::

    {"case": str, "page": 0, "pdf": "<case>.pdf", "page_size": [w, h],
     "panels": [{"bbox_hint": [x0, y0, x1, y1],   # plot area, PyMuPDF page coords (pt, y down)
                 "xscale": "linear|log", "yscale": "linear|log",
                 "xlim": [left, right], "ylim": [bottom, top],   # values at the page edges
                 "series": [{"name": str|None, "kind": "line|markers|line+markers",
                             "x": [...], "y": [...],
                             "marker_idx": [...]   # optional: indices of x/y that carry a marker
                             }]}],
     "features": [...]}

Conventions
-----------
* ``xlim``/``ylim`` are the data values at (left, right) / (bottom, top) of the plot
  area, so an inverted axis has ``ylim[0] > ylim[1]``.
* ``x``/``y`` hold the exact data that was plotted (only points inside the axes limits).
* ``jv_opv``-type series additionally carry ``jsc`` (mA/cm2, positive), ``voc`` (V),
  ``ff`` (0..1), ``pce`` (%).  They are derived analytically from the single-diode
  model  J(V) = -Jph + J0 (exp(V/(n Vt)) - 1) + 1000 V / Rsh   [mA/cm2, Rsh in ohm cm2]
  as follows: Jsc = -J(0) (= Jph exactly); Voc = root of J(V)=0 by bisection (1e-14
  tolerance); on a 200001-point grid V in [0, Voc] the maximum of P = -V J(V) gives
  Pmax; FF = Pmax / (Jsc Voc); PCE = Pmax / Pin * 100 with Pin = 100 mW/cm2.
* Hard variants (``outlined_*``) are produced by the same code path as ``jv_opv`` and
  have the same data; their bbox_hint is re-measured against the produced PDF.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "out"

CASE_NAMES = [
    "linear_single", "linear_multi_legend", "markers_only", "line_markers",
    "log_y", "loglog", "negative_ticks", "inverted_y", "sci_notation", "grid",
    "subplots_2x2", "errorbars", "jv_opv", "eqe",
    "outlined_text", "outlined_strokes", "outlined_all",
    "raster_figure", "multi_figure_page",
]

FIG_W, FIG_H = 3.5, 2.6  # inches
A4 = (595.276, 841.89)
VT = 0.02585  # kT/q at 300 K, volts


# --------------------------------------------------------------------------- matplotlib plumbing
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "figure.dpi": 72,           # display units == PDF points (needed by the outliner)
        "savefig.dpi": 72,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.unicode_minus": True,
        "lines.linewidth": 1.2,
    })
    return plt


def _new_fig(w=FIG_W, h=FIG_H, **adjust):
    plt = _mpl()
    fig = plt.figure(figsize=(w, h))
    kw = dict(left=0.17, right=0.96, bottom=0.2, top=0.95)
    kw.update(adjust)
    ax = fig.add_axes([kw["left"], kw["bottom"], kw["right"] - kw["left"], kw["top"] - kw["bottom"]])
    return fig, ax


def _bbox_hint(fig, ax):
    W, H = fig.get_figwidth() * 72, fig.get_figheight() * 72
    p = ax.get_position()
    return [p.x0 * W, H * (1 - p.y1), p.x1 * W, H * (1 - p.y0)]


def _series(name, kind, x, y, **extra):
    d = {"name": name, "kind": kind,
         "x": np.asarray(x, float).tolist(), "y": np.asarray(y, float).tolist()}
    d.update(extra)
    return d


def _panel(fig, ax, series, **extra):
    fig.canvas.draw()
    d = {
        "bbox_hint": _bbox_hint(fig, ax),
        "xscale": ax.get_xscale() if ax.get_xscale() == "log" else "linear",
        "yscale": ax.get_yscale() if ax.get_yscale() == "log" else "linear",
        "xlim": [float(v) for v in ax.get_xlim()],
        "ylim": [float(v) for v in ax.get_ylim()],
        "series": series,
    }
    d.update(extra)
    return d


# --------------------------------------------------------------------------- outlining (strokes -> fills)
def _stroke_polygon(pts: np.ndarray, width: float) -> np.ndarray:
    """Outline of a stroked polyline (display space): offset +-width/2 along (mitered) normals."""
    pts = np.asarray(pts, float)
    keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-9]
    pts = pts[keep]
    if len(pts) < 2:
        raise ValueError("degenerate polyline")
    d = np.diff(pts, axis=0)
    t = d / np.hypot(d[:, 0], d[:, 1])[:, None]
    nseg = np.column_stack([-t[:, 1], t[:, 0]])
    nv = np.empty_like(pts)
    nv[0], nv[-1] = nseg[0], nseg[-1]
    for i in range(1, len(pts) - 1):
        m = nseg[i - 1] + nseg[i]
        ml = np.hypot(*m)
        if ml < 1e-9:
            nv[i] = nseg[i]
        else:
            m = m / ml
            nv[i] = m / max(float(m @ nseg[i]), 0.5)  # miter, capped
    hw = width / 2
    return np.vstack([pts + hw * nv, (pts - hw * nv)[::-1]])


def _rect_poly(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _outline_figure(fig):
    """Replace every stroked Line2D / spine / tick mark by a filled polygon (no stroke)."""
    from matplotlib.patches import Polygon
    from matplotlib.transforms import IdentityTransform
    fig.canvas.draw()

    def add(poly, color, zorder, to_ax=None):
        p = Polygon(poly, closed=True, transform=IdentityTransform(), facecolor=color,
                    edgecolor="none", linewidth=0, zorder=zorder)
        (to_ax.add_artist(p) if to_ax is not None else fig.add_artist(p))

    for ax in fig.axes:
        for ln in list(ax.lines):
            if ln.get_linestyle() not in ("-", "solid"):
                raise NotImplementedError("outlining dashed lines not supported")
            if ln.get_marker() not in ("None", None, ""):
                raise NotImplementedError("outlining markers not supported")
            xy = ln.get_transform().transform(ln.get_xydata())
            add(_stroke_polygon(xy, ln.get_linewidth()), ln.get_color(), ln.get_zorder(), to_ax=ax)
            ln.set_visible(False)
        leg = ax.get_legend()
        if leg is not None:
            for ln in leg.get_lines():
                xy = ln.get_transform().transform(ln.get_xydata())
                add(_stroke_polygon(xy, ln.get_linewidth()), ln.get_color(), 10)
                ln.set_visible(False)
        # frame + ticks as filled rectangles
        bb = ax.get_window_extent()
        hw = ax.spines["left"].get_linewidth() / 2
        for s in ax.spines.values():
            s.set_visible(False)
        x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
        for r in (_rect_poly(x0 - hw, y0 - hw, x1 + hw, y0 + hw), _rect_poly(x0 - hw, y1 - hw, x1 + hw, y1 + hw),
                  _rect_poly(x0 - hw, y0 - hw, x0 + hw, y1 + hw), _rect_poly(x1 - hw, y0 - hw, x1 + hw, y1 + hw)):
            add(r, "black", 10)
        tl, tw = 3.5, 0.8
        xl, yl = ax.get_xlim(), ax.get_ylim()
        for v in ax.xaxis.get_majorticklocs():
            if min(xl) - 1e-12 <= v <= max(xl) + 1e-12:
                px = ax.transData.transform((v, yl[0]))[0]
                add(_rect_poly(px - tw / 2, y0 - tl, px + tw / 2, y0), "black", 10)
        for v in ax.yaxis.get_majorticklocs():
            if min(yl) - 1e-12 <= v <= max(yl) + 1e-12:
                py = ax.transData.transform((xl[0], v))[1]
                add(_rect_poly(x0 - tl, py - tw / 2, x0, py + tw / 2), "black", 10)
        ax.tick_params(axis="both", which="both", length=0, pad=3.5 + tl)
    fig.canvas.draw()


# --------------------------------------------------------------------------- emit
def _frame_deviation(pdf: Path, bbox) -> float:
    """Smallest max-coordinate deviation between `bbox` and any drawing rect on page 0."""
    import pymupdf
    best = math.inf
    with pymupdf.open(str(pdf)) as doc:
        for d in doc[0].get_drawings():
            r = d["rect"]
            best = min(best, max(abs(r.x0 - bbox[0]), abs(r.y0 - bbox[1]),
                                 abs(r.x1 - bbox[2]), abs(r.y1 - bbox[3])))
    return best


def _write_case(outdir: Path, case: dict, tol_frame: float | None = 1.0):
    pdf = outdir / case["pdf"]
    if tol_frame is not None:
        for p in case["panels"]:
            dev = _frame_deviation(pdf, p["bbox_hint"])
            if dev > tol_frame:
                raise RuntimeError(f"{case['case']}: bbox_hint is {dev:.2f}pt away from the nearest "
                                   f"drawing rect in the PDF (tol {tol_frame})")
    (outdir / f"{case['case']}.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")


def _emit(outdir: Path, name: str, fig, panels, features, *, route="pdf", extra=None):
    """Save `fig` as <name>.pdf (route 'pdf': matplotlib PDF; 'svg': SVG with glyph outlines -> MuPDF)."""
    import pymupdf
    plt = _mpl()
    pdf = outdir / f"{name}.pdf"
    W, H = fig.get_figwidth() * 72, fig.get_figheight() * 72
    scale = 1.0
    if route == "pdf":
        fig.savefig(str(pdf), format="pdf", metadata={"CreationDate": None, "Creator": "figrip-corpus"})
    else:
        import matplotlib
        svg = outdir / f"{name}.svg"
        with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": "figrip"}):
            fig.savefig(str(svg), format="svg", metadata={"Date": None})
        with pymupdf.open(str(svg)) as sdoc:
            data = sdoc.convert_to_pdf()
        with pymupdf.open("pdf", data) as pdoc:
            pdoc.save(str(pdf), deflate=True)
            r = pdoc[0].rect
        sx, sy = r.width / W, r.height / H
        if abs(sx - sy) > 1e-3:
            raise RuntimeError(f"{name}: anisotropic SVG->PDF scale {sx} vs {sy}")
        scale = sx
        svg.unlink()
    plt.close(fig)
    for p in panels:
        p["bbox_hint"] = [v * scale for v in p["bbox_hint"]]
    with pymupdf.open(str(pdf)) as doc:
        rect = doc[0].rect
        text = doc[0].get_text()
    case = {"case": name, "page": 0, "pdf": f"{name}.pdf", "page_size": [rect.width, rect.height],
            "panels": panels, "features": features}
    case.update(extra or {})
    if "outlined_text" in features and re.search(r"\d", text):
        raise RuntimeError(f"{name}: expected no extractable digits, got {text!r}")
    _write_case(outdir, case, tol_frame=1.0 * max(scale, 1.0))
    return case


# --------------------------------------------------------------------------- simple cases
def case_linear_single(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 50)
    y = 2.0 * np.sin(x * 0.9) * np.exp(-x / 12) + 0.3 * x
    ax.plot(x, y, color="C0")
    ax.set(xlim=(0, 10), xlabel="Time (s)", ylabel="Signal (a.u.)")
    return _emit(outdir, "linear_single", fig,
                 [_panel(fig, ax, [_series(None, "line", x, y)])], ["real_text", "single_series"])


def case_linear_multi_legend(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 60)
    ys = [np.sin(x), 0.8 * np.cos(0.7 * x) + 0.4, 0.15 * x - 1.2]
    names = ["Sample A", "Sample B", "Baseline"]
    cols = ["C0", "C3", "C2"]
    for i, (y, nm, c) in enumerate(zip(ys, names, cols)):
        ax.plot(x, y, color=c, ls="--" if i == 2 else "-", label=nm)
    ax.set(xlim=(0, 10), ylim=(-2, 3.2), xlabel="x (a.u.)", ylabel="Response")
    ax.legend(loc="upper right", ncol=1, frameon=True)
    ser = [_series(n, "line", x, y) for n, y in zip(names, ys)]
    return _emit(outdir, "linear_multi_legend", fig, [_panel(fig, ax, ser)],
                 ["real_text", "legend_inside", "dashed", "multi_color"])


def case_markers_only(outdir):
    rng = np.random.default_rng(1)
    fig, ax = _new_fig()
    x1 = np.linspace(0.5, 9.5, 25)
    y1 = 0.8 * x1 + 1 + rng.normal(0, 0.4, x1.size)
    x2 = np.linspace(0.8, 9.2, 22)
    y2 = 0.4 * x2 + 6 + rng.normal(0, 0.4, x2.size)
    ax.scatter(x1, y1, s=14, marker="o", color="C0", label="Circles")
    ax.plot(x2, y2, "s", ms=3.5, color="C1", label="Squares")
    ax.set(xlim=(0, 10), ylim=(0, 12), xlabel="Concentration (mM)", ylabel="Response (V)")
    ax.legend(loc="upper left")
    ser = [_series("Circles", "markers", x1, y1), _series("Squares", "markers", x2, y2)]
    return _emit(outdir, "markers_only", fig, [_panel(fig, ax, ser)],
                 ["real_text", "legend_inside", "scatter_collection", "plot_markers", "no_line"])


def case_line_markers(outdir):
    fig, ax = _new_fig()
    x = np.arange(1, 16)
    y1 = 10 + 3 * np.sin(x / 2.5)
    y2 = 4 + 0.35 * x
    ax.plot(x, y1, "-o", ms=3.5, color="C0", label="Series 1")
    ax.plot(x, y2, "-s", ms=3.5, color="C3", label="Series 2")
    ax.set(xlim=(0, 16), ylim=(0, 16), xlabel="Cycle number", ylabel="Capacity (mAh)")
    ax.legend(loc="lower right")
    ser = [_series("Series 1", "line+markers", x, y1), _series("Series 2", "line+markers", x, y2)]
    return _emit(outdir, "line_markers", fig, [_panel(fig, ax, ser)],
                 ["real_text", "legend_inside", "line_with_markers"])


def case_log_y(outdir):
    fig, ax = _new_fig()
    V = np.linspace(-1, 1, 100)  # even count -> no exact zero

    def dark(j0, n, rsh):
        return np.abs(j0 * (np.exp(V / (n * VT)) - 1) + 1000 * V / rsh)
    ys = [dark(1e-9, 1.8, 5e5), dark(2e-10, 1.6, 2e6)]
    names = ["Device A", "Device B"]
    ax.set_yscale("log")
    for y, nm, c in zip(ys, names, ["C0", "C3"]):
        ax.plot(V, y, color=c, label=nm)
    ax.set(xlim=(-1, 1), ylim=(1e-6, 1e1), xlabel="Voltage (V)", ylabel="|J| (mA cm$^{-2}$)")
    ax.legend(loc="upper left")
    for y in ys:
        assert y.min() > 1e-6 and y.max() < 1e1
    ser = [_series(n, "line", V, y) for n, y in zip(names, ys)]
    return _emit(outdir, "log_y", fig, [_panel(fig, ax, ser)],
                 ["real_text", "log_y", "legend_inside", "dark_jv", "negative_ticks"])


def case_loglog(outdir):
    fig, ax = _new_fig()
    x = np.logspace(0, 3, 40)
    ys = [2.0 * x ** 1.0, 0.05 * x ** 1.8 + 1.5]
    names = ["Linear", "Superlinear"]
    ax.set_xscale("log")
    ax.set_yscale("log")
    for y, nm, c in zip(ys, names, ["C0", "C2"]):
        ax.plot(x, y, color=c, label=nm)
    ax.set(xlim=(1, 1e3), ylim=(1, 1e5), xlabel="Intensity (mW cm$^{-2}$)", ylabel="Photocurrent (nA)")
    ax.legend(loc="upper left")
    for y in ys:
        assert y.min() >= 1 and y.max() <= 1e5
    ser = [_series(n, "line", x, y) for n, y in zip(names, ys)]
    return _emit(outdir, "loglog", fig, [_panel(fig, ax, ser)],
                 ["real_text", "log_x", "log_y", "legend_inside", "minor_log_ticks"])


def case_negative_ticks(outdir):
    fig, ax = _new_fig()
    x = np.linspace(-25, 15, 60)
    y1 = 25 * np.tanh(x / 10)
    y2 = 0.8 * x - 4
    ax.plot(x, y1, color="C0")
    ax.plot(x, y2, color="C1")
    ax.set(xlim=(-25, 15), ylim=(-30, 30), xlabel="Bias (V)", ylabel="Current (nA)",
           xticks=[-20, -10, 0, 10], yticks=[-20, -10, 0, 10, 20])
    ser = [_series(None, "line", x, y1), _series(None, "line", x, y2)]
    return _emit(outdir, "negative_ticks", fig, [_panel(fig, ax, ser)],
                 ["real_text", "unicode_minus", "negative_ticks", "ticks_not_at_edges"])


def _diode(V, jph, j0, n, rsh):
    """Single-diode J(V) in mA/cm2 (Rsh in ohm cm2)."""
    return -jph + j0 * (np.exp(V / (n * VT)) - 1) + 1000.0 * V / rsh


def case_inverted_y(outdir):
    fig, ax = _new_fig()
    jph, voc, n, rsh = 20.0, 0.80, 1.8, 900.0
    j0 = (jph - 1000 * voc / rsh) / (math.exp(voc / (n * VT)) - 1)
    V = np.linspace(-0.2, 0.9, 80)
    J = _diode(V, jph, j0, n, rsh)
    m = J < 8
    V, J = V[m], J[m]
    ax.plot(V, J, color="C3")
    ax.set(xlim=(-0.2, 1.0), ylim=(-30, 10), xlabel="Voltage (V)", ylabel="J (mA cm$^{-2}$)")
    ax.invert_yaxis()
    return _emit(outdir, "inverted_y", fig, [_panel(fig, ax, [_series(None, "line", V, J)])],
                 ["real_text", "inverted_y", "negative_ticks", "unicode_minus"])


def case_sci_notation(outdir):
    fig, ax = _new_fig(top=0.88, left=0.19)
    x = np.linspace(0, 5, 50)
    y = 1e-4 * (0.5 + 3.2 * (1 - np.exp(-x / 1.5)) + 0.2 * x)
    ax.plot(x, y, color="C0")
    ax.set(xlim=(0, 5), xlabel="Time (h)", ylabel="Current (A)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.canvas.draw()
    off = ax.yaxis.get_offset_text().get_text()
    assert off, "expected an offset text"
    return _emit(outdir, "sci_notation", fig,
                 [_panel(fig, ax, [_series(None, "line", x, y)], offset_text=off, offset_axis="y")],
                 ["real_text", "sci_notation", "offset_text", "tick_labels_scaled"])


def case_grid(outdir):
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 50)
    y1 = 5 + 3 * np.sin(x / 1.7)
    y2 = 0.8 * x + 1
    ax.plot(x, y1, color="C0", label="Signal")
    ax.plot(x, y2, color="C1", label="Trend")
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="Distance (mm)", ylabel="Height (um)")
    ax.grid(True, color="0.7", lw=0.5)
    ax.legend(loc="lower right")
    ser = [_series("Signal", "line", x, y1), _series("Trend", "line", x, y2)]
    return _emit(outdir, "grid", fig, [_panel(fig, ax, ser)],
                 ["real_text", "grid", "legend_inside"])


def case_subplots_2x2(outdir):
    plt = _mpl()
    fig = plt.figure(figsize=(6.5, 4.8))
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.12, top=0.94, wspace=0.42, hspace=0.5)
    axs = fig.subplots(2, 2).ravel()
    panels = []
    x = np.linspace(0, 10, 50)
    # (a) linear line
    ya = 3 * np.sin(x / 1.6) + 4
    axs[0].plot(x, ya, color="C0")
    axs[0].set(xlim=(0, 10), ylim=(0, 8), xlabel="Time (s)", ylabel="Signal (a.u.)")
    # (b) scatter with legend
    rng = np.random.default_rng(2)
    xb = np.linspace(1, 9, 20)
    yb = 0.5 * xb ** 1.3 + rng.normal(0, 0.3, xb.size)
    axs[1].scatter(xb, yb, s=12, color="C3", label="Data")
    axs[1].set(xlim=(0, 10), ylim=(0, 14), xlabel="Dose (Gy)", ylabel="Response")
    axs[1].legend(loc="upper left")
    # (c) semilog line
    xc = np.linspace(0, 5, 40)
    yc = 1e-3 * np.exp(xc * 1.5)
    axs[2].semilogy(xc, yc, color="C2")
    axs[2].set(xlim=(0, 5), ylim=(1e-4, 1e1), xlabel="Voltage (V)", ylabel="Current (mA)")
    # (d) negative ticks, line + markers
    xd = np.linspace(-20, 20, 15)
    yd = -0.02 * xd ** 2 + 5 + 0.5 * xd
    axs[3].plot(xd, yd, "-o", ms=3, color="C1")
    axs[3].set(xlim=(-25, 25), ylim=(-25, 15), xlabel="Field (mT)", ylabel="Moment (a.u.)",
               xticks=[-20, -10, 0, 10, 20], yticks=[-20, -10, 0, 10])
    specs = [
        [_series(None, "line", x, ya)],
        [_series("Data", "markers", xb, yb)],
        [_series(None, "line", xc, yc)],
        [_series(None, "line+markers", xd, yd)],
    ]
    for ax, s, lab in zip(axs, specs, "abcd"):
        ax.text(-0.27, 1.06, f"({lab})", transform=ax.transAxes, fontsize=9, fontweight="bold")
        panels.append(_panel(fig, ax, s, label=lab))
    return _emit(outdir, "subplots_2x2", fig, panels,
                 ["real_text", "multi_panel", "panel_letters", "log_y", "negative_ticks", "mixed_kinds"])


def case_errorbars(outdir):
    fig, ax = _new_fig()
    x = np.arange(1, 13)
    y = 3 + 6 * (1 - np.exp(-x / 4.0))
    err = 0.3 + 0.08 * x
    ax.errorbar(x, y, yerr=err, fmt="-o", ms=3.5, capsize=2.5, lw=1.0, elinewidth=0.8, color="C0",
                label="Measured")
    ax.set(xlim=(0, 13), ylim=(0, 12), xlabel="Day", ylabel="Growth (mm)")
    ax.legend(loc="lower right")
    return _emit(outdir, "errorbars", fig, [_panel(fig, ax, [_series("Measured", "line+markers", x, y)])],
                 ["real_text", "error_bars", "caps", "line_with_markers", "legend_inside"])


# --------------------------------------------------------------------------- OPV cases
OPV_DEVICES = [  # name, Jph [mA/cm2], Voc target [V], ideality n, Rsh [ohm cm2], colour
    ("PM6:Y6", 25.0, 0.85, 1.75, 1500.0, "C3"),
    ("PM6:BTP-eC9", 26.0, 0.84, 1.95, 1200.0, "C0"),
    ("PM6:Y6-ref", 22.5, 0.82, 2.25, 700.0, "C2"),
]


def opv_metrics(jph, j0, n, rsh):
    """Analytic J-V figures of merit for the single-diode model (see module docstring)."""
    f = lambda v: _diode(v, jph, j0, n, rsh)
    lo, hi = 0.0, 1.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    voc = 0.5 * (lo + hi)
    jsc = -float(f(0.0))
    V = np.linspace(0, voc, 200001)
    P = -V * f(V)  # mW/cm2
    pmax = float(P.max())
    return {"jsc": jsc, "voc": float(voc), "ff": pmax / (jsc * voc), "pce": pmax / 100.0 * 100.0,
            "vmpp": float(V[P.argmax()]), "pmax": pmax}


def _jv_fig():
    fig, ax = _new_fig(left=0.18)
    ser = []
    for name, jph, voc_t, n, rsh, col in OPV_DEVICES:
        j0 = (jph - 1000 * voc_t / rsh) / (math.exp(voc_t / (n * VT)) - 1)
        m = opv_metrics(jph, j0, n, rsh)
        assert 0.65 <= m["ff"] <= 0.80 and 20 <= m["jsc"] <= 26 and 0.8 <= m["voc"] <= 0.9, (name, m)
        lo, hi = m["voc"], 1.5  # curve ends where J reaches +6 mA/cm2
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _diode(mid, jph, j0, n, rsh) < 6 else (lo, mid)
        V = np.linspace(-0.2, 0.5 * (lo + hi), 120)
        J = _diode(V, jph, j0, n, rsh)
        ax.plot(V, J, color=col, label=name)
        ser.append(_series(name, "line", V, J, jsc=m["jsc"], voc=m["voc"], ff=m["ff"], pce=m["pce"],
                           model={"jph": jph, "j0": j0, "n": n, "rsh": rsh, "vt": VT, "pin": 100.0}))
    ax.axhline(0, color="0.45", lw=0.6)
    ax.set(xlim=(-0.2, 1.0), ylim=(-30, 10), xlabel="Voltage (V)", ylabel="Current density (mA cm$^{-2}$)",
           xticks=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[-30, -20, -10, 0, 10])
    ax.legend(loc="upper left", frameon=False)
    return fig, ax, ser


_JV_FEATURES = ["opv_jv", "negative_ticks", "unicode_minus", "legend_inside", "zero_line", "mathtext_label"]


def case_jv_opv(outdir):
    fig, ax, ser = _jv_fig()
    return _emit(outdir, "jv_opv", fig, [_panel(fig, ax, ser)], ["real_text"] + _JV_FEATURES)


def case_eqe(outdir):
    fig, ax = _new_fig()
    wl = np.arange(300, 901, 5.0)

    def band(lo, hi, peak, w=25.0):
        s = 1 / (1 + np.exp(-(wl - lo) / w * 4)) * 1 / (1 + np.exp((wl - hi) / w * 4))
        return peak * s * (0.92 + 0.08 * np.sin((wl - 400) / 45))
    ys = [band(380, 880, 85.0), band(390, 820, 78.0)]
    names = ["PM6:Y6", "PM6:IT-4F"]
    mi = list(range(0, wl.size, 4))  # every 20 nm
    ax.plot(wl, ys[0], "-o", ms=3, markevery=4, color="C3", label=names[0])
    ax.plot(wl, ys[1], "-s", ms=3, markevery=4, color="C0", label=names[1])
    ax.set(xlim=(300, 900), ylim=(0, 100), xlabel="Wavelength (nm)", ylabel="EQE (%)")
    ax.legend(loc="lower center")
    ser = [_series(n, "line+markers", wl, y, marker_idx=mi) for n, y in zip(names, ys)]
    return _emit(outdir, "eqe", fig, [_panel(fig, ax, ser)],
                 ["real_text", "markevery", "line_with_markers", "legend_inside", "eqe"])


# --------------------------------------------------------------------------- hard variants
def case_outlined_text(outdir):
    fig, ax, ser = _jv_fig()
    return _emit(outdir, "outlined_text", fig, [_panel(fig, ax, ser)],
                 ["outlined_text", "svg_roundtrip"] + _JV_FEATURES, route="svg")


def case_outlined_strokes(outdir):
    fig, ax, ser = _jv_fig()
    panel = _panel(fig, ax, ser)
    _outline_figure(fig)
    return _emit(outdir, "outlined_strokes", fig, [panel],
                 ["real_text", "outlined_strokes", "filled_axes", "filled_ticks"] + _JV_FEATURES)


def case_outlined_all(outdir):
    fig, ax, ser = _jv_fig()
    panel = _panel(fig, ax, ser)
    _outline_figure(fig)
    return _emit(outdir, "outlined_all", fig, [panel],
                 ["outlined_text", "outlined_strokes", "filled_axes", "filled_ticks", "svg_roundtrip",
                  "all_fills"] + _JV_FEATURES, route="svg")


def case_raster_figure(outdir):
    import pymupdf
    fig, ax = _new_fig()
    x = np.linspace(0, 10, 50)
    ax.plot(x, np.sin(x), color="C0")
    ax.set(xlim=(0, 10), xlabel="x", ylabel="sin x")
    png = outdir / "raster_figure.png"
    fig.savefig(str(png), format="png", dpi=200)
    _mpl().close(fig)
    doc = pymupdf.open()
    page = doc.new_page(width=FIG_W * 72, height=FIG_H * 72)
    page.insert_image(page.rect, filename=str(png))
    pdf = outdir / "raster_figure.pdf"
    doc.save(str(pdf), deflate=True)
    doc.close()
    png.unlink()
    with pymupdf.open(str(pdf)) as d:
        assert not d[0].get_text().strip() and not d[0].get_drawings() and len(d[0].get_images()) == 1
    case = {"case": "raster_figure", "page": 0, "pdf": "raster_figure.pdf",
            "page_size": [FIG_W * 72, FIG_H * 72], "panels": [], "features": ["raster_only", "no_vector_data"]}
    _write_case(outdir, case, tol_frame=None)
    return case


_LOREM = (
    "The photovoltaic parameters of the devices were extracted from the current density-voltage "
    "characteristics measured under AM 1.5G illumination at 100 mW cm-2. The best device reached a "
    "power conversion efficiency of 17.2% with a short-circuit current density of 25.1 mA cm-2, an "
    "open-circuit voltage of 0.85 V and a fill factor of 0.80. The external quantum efficiency spectra "
    "confirm the integrated current within 3% of the measured value, as summarised in Table 1 and "
    "plotted in Figure 2 for 300-900 nm. "
)


def case_multi_figure_page(outdir):
    import pymupdf
    srcs = [("jv_opv", (110.0, 215.0), 1.35), ("eqe", (160.0, 590.0), 1.0)]
    doc = pymupdf.open()
    page = doc.new_page(width=A4[0], height=A4[1])
    page.insert_text((60, 62), "3. Results and discussion", fontname="hebo", fontsize=12)
    page.insert_textbox(pymupdf.Rect(60, 72, A4[0] - 60, 205), _LOREM * 2, fontname="helv", fontsize=9,
                        align=3)
    panels = []
    feats = {"real_text", "multi_panel", "body_text", "scaled_placement"}
    bottoms = []
    for k, (nm, (ox, oy), s) in enumerate(srcs, start=1):
        gt = json.loads((outdir / f"{nm}.json").read_text(encoding="utf-8"))
        feats.update(gt["features"])
        w, h = gt["page_size"]
        rect = pymupdf.Rect(ox, oy, ox + w * s, oy + h * s)
        with pymupdf.open(str(outdir / gt["pdf"])) as src:
            page.show_pdf_page(rect, src, 0)
        for p in gt["panels"]:
            q = dict(p)
            q["bbox_hint"] = [ox + p["bbox_hint"][0] * s, oy + p["bbox_hint"][1] * s,
                              ox + p["bbox_hint"][2] * s, oy + p["bbox_hint"][3] * s]
            q["source_case"] = nm
            panels.append(q)
        page.insert_textbox(pymupdf.Rect(60, rect.y1 + 6, A4[0] - 60, rect.y1 + 40),
                            f"Figure {k}. {'J-V curves' if k == 1 else 'EQE spectra'} of the devices "
                            f"(Pin = 100 mW cm-2, scan 1.2 to -0.2 V).", fontname="heit", fontsize=8)
        bottoms.append(rect.y1 + 40)
    page.insert_textbox(pymupdf.Rect(60, bottoms[0] + 5, A4[0] - 60, 585), _LOREM, fontname="helv",
                        fontsize=9, align=3)
    pdf = outdir / "multi_figure_page.pdf"
    doc.save(str(pdf), deflate=True)
    doc.close()
    case = {"case": "multi_figure_page", "page": 0, "pdf": "multi_figure_page.pdf",
            "page_size": list(A4), "panels": panels, "features": sorted(feats)}
    _write_case(outdir, case, tol_frame=1.5)
    return case


BUILDERS = {n: globals()[f"case_{n}"] for n in CASE_NAMES}


# --------------------------------------------------------------------------- build
def _outputs(outdir: Path):
    return [outdir / f"{n}.{ext}" for n in CASE_NAMES for ext in ("pdf", "json")]


def _up_to_date(outdir: Path) -> bool:
    script_m = Path(__file__).stat().st_mtime
    try:
        return all(p.stat().st_mtime >= script_m for p in _outputs(outdir))
    except FileNotFoundError:
        return False


def build(outdir=None, force: bool = False, quiet: bool = True) -> list[dict]:
    """Generate (or reuse) the corpus; returns the ground-truth dicts in CASE_NAMES order."""
    outdir = Path(outdir) if outdir else DEFAULT_OUT
    outdir.mkdir(parents=True, exist_ok=True)
    if not force and _up_to_date(outdir):
        return [json.loads((outdir / f"{n}.json").read_text(encoding="utf-8")) for n in CASE_NAMES]
    cases = []
    for n in CASE_NAMES:
        cases.append(BUILDERS[n](outdir))
        if not quiet:
            print(f"  built {n}")
    return cases


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    res = build(a.out, force=True if a.force else False, quiet=False)
    print(f"{len(res)} cases -> {a.out}")
