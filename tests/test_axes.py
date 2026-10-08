"""Tests for figrip.axes (panel detection + axis calibration) and figrip.ocr."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from figrip.axes import find_panels, fit_axis, parse_value
from figrip.primitives import extract_page

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ZHENG = Path("D:/Zotero/storage/QIZR5ARD/Zheng_et_al_2018_Adv_Mater_Fine_tuned_HTL_high_FF_OSCs.pdf")


# --------------------------------------------------------------------------- helpers
def _save(fig, path: Path) -> Path:
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def _panels(pdf: Path, page_no: int = 0, **kw):
    with pymupdf.open(str(pdf)) as doc:
        page = doc[page_no]
        return find_panels(extract_page(page), page=page, **kw)


def _ax_bbox_pt(fig, ax):
    """Plot area of `ax` in PyMuPDF page coords (pt, y down)."""
    fig.canvas.draw()
    bb = ax.get_window_extent()
    k = 72.0 / fig.dpi
    H = fig.get_figheight() * 72
    return (bb.x0 * k, H - bb.y1 * k, bb.x1 * k, H - bb.y0 * k)


def _check_limits(panel, xlim, ylim, xlog=False, ylog=False, tol=0.005):
    """Data values at the plot-area edges match the axes limits (fraction of range)."""
    b = panel.bbox
    def close(got, want, log):
        g, w = (np.log10(got), np.log10(want)) if log else (np.asarray(got), np.asarray(want))
        rng = abs(w[1] - w[0])
        assert np.all(np.abs(g - w) <= tol * rng), (got, want)
    close(panel.x.to_data([b[0], b[2]]), xlim, xlog)
    close(panel.y.to_data([b[3], b[1]]), ylim, ylog)


# --------------------------------------------------------------------------- parsing / fitting
@pytest.mark.parametrize("text,value", [
    ("0.5", 0.5), ("\u22125", -5.0), ("\u20130.2", -0.2), ("10^-3", 1e-3), ("10^\u22123", 1e-3),
    ("2\u00d710^4", 2e4), ("1e-4", 1e-4), ("50%", 50.0), ("0,5", 0.5), ("1,000", 1000.0),
    ("abc", None), ("", None), ("Voltage", None),
])
def test_parse_value(text, value):
    got = parse_value(text)
    assert (got is None and value is None) or math.isclose(got, value)


def test_fit_repairs_lost_minus_and_bad_label():
    pos = [10.0, 20.0, 30.0, 40.0, 50.0]
    c, notes = fit_axis("y", list(zip(pos, [-20, -15, 10, -5, 0])))     # one minus lost
    assert c is not None and notes and abs(c.to_data(30.0) + 10) < 1e-9
    c, notes = fit_axis("x", list(zip(pos, [0, 1, 7, 3, 4])))            # one digit misread
    assert c is not None and abs(c.to_data(30.0) - 2) < 1e-9
    c, _ = fit_axis("x", list(zip(pos, [0, 5, 1, 7, 2])))                # garbage
    assert c is None


# --------------------------------------------------------------------------- synthetic matplotlib PDFs
def test_linear(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2, 3], [0.5, 1.5, 1.0, 2.5])
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Current (mA)")
    bb = _ax_bbox_pt(fig, ax)
    ps = _panels(_save(fig, tmp_path / "lin.pdf"))
    assert len(ps) == 1
    p = ps[0]
    assert np.allclose(p.bbox, bb, atol=0.6)
    _check_limits(p, (0, 3), (0, 3))
    assert p.x.label == "Voltage (V)" and p.y.label == "Current (mA)"
    assert p.x.source == "text" and p.x.residual < 0.01


def test_log(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.semilogy([1, 2, 3], [1e-3, 1e-1, 10])
    ax.set_xlim(0, 4); ax.set_ylim(1e-4, 1e2)
    p, = _panels(_save(fig, tmp_path / "log.pdf"))
    assert p.y.scale == "log" and p.x.scale == "linear"
    _check_limits(p, (0, 4), (1e-4, 1e2), ylog=True)


def test_inverted_and_unicode_minus(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([-0.2, 0.9], [-25, 5])
    ax.set_xlim(-0.2, 1.0); ax.set_ylim(5, -25)            # J-V style: negative values up
    p, = _panels(_save(fig, tmp_path / "inv.pdf"))
    vals = [v for _, v in p.y.ticks]
    assert min(vals) <= -20 and p.y.a > 0                  # y grows down the page with value
    _check_limits(p, (-0.2, 1.0), (5, -25))


def test_subplots_and_open_spines(tmp_path):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    lims = [((0, 10), (0, 1)), ((-5, 5), (-50, 50)), ((300, 900), (0, 100))]
    for ax, (xl, yl) in zip(axs, lims):
        ax.plot(xl, yl)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
    axs[1].spines[["top", "right"]].set_visible(False)        # L-shaped frame
    axs[2].tick_params(direction="in")                        # inward ticks
    fig.tight_layout()
    ps = _panels(_save(fig, tmp_path / "sub.pdf"))
    assert len(ps) == 3
    for p, (xl, yl) in zip(sorted(ps, key=lambda p: p.bbox[0]), lims):
        _check_limits(p, xl, yl)


def test_shared_y_axis_is_borrowed(tmp_path):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True, gridspec_kw={"wspace": 0})
    for ax, xl in zip(axs, [(0, 1), (2, 3)]):
        ax.plot(xl, (0, 10)); ax.set_xlim(*xl)
    axs[0].set_ylim(0, 10)
    ps = sorted(_panels(_save(fig, tmp_path / "sharey.pdf")), key=lambda p: p.bbox[0])
    assert len(ps) == 2
    _check_limits(ps[0], (0, 1), (0, 10))
    _check_limits(ps[1], (2, 3), (0, 10))
    assert any("shared" in n for n in ps[1].notes)


def test_twin_y_is_reported(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax2 = ax.twinx(); ax2.plot([0, 1], [0, 100]); ax2.set_ylim(0, 100)
    ps = _panels(_save(fig, tmp_path / "twin.pdf"))
    assert len(ps) == 1
    _check_limits(ps[0], (0, 1), (0, 1))
    assert any("twin" in n for n in ps[0].notes)


def test_no_plot_page(tmp_path):
    fig = plt.figure(figsize=(4, 3))
    fig.text(0.1, 0.5, "Table 1: 0.1 0.2 0.3")
    fig.add_artist(matplotlib.lines.Line2D([0.1, 0.9], [0.4, 0.4]))          # table rule
    fig.add_artist(matplotlib.patches.Rectangle((0.2, 0.6), 0.3, 0.2, fill=False))  # a bare box
    assert _panels(_save(fig, tmp_path / "none.pdf")) == []


def test_hints_supply_unreadable_values(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 2)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels([]); ax.set_yticklabels([])           # nothing to read
    pdf = _save(fig, tmp_path / "hint.pdf")
    assert _panels(pdf) == []
    p, = _panels(pdf, hints={"x_ticks": [0, 0.5, 1], "y_ticks": [0, 1, 2]})
    assert p.x.source == "manual"
    _check_limits(p, (0, 1), (0, 2))


# --------------------------------------------------------------------------- synthetic corpus (other agent)
def _corpus_cases():
    out = HERE / "corpus" / "out"
    mk = HERE / "corpus" / "make_corpus.py"
    if not mk.exists():
        return []
    spec = importlib.util.spec_from_file_location("make_corpus", mk)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(out)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u


def test_corpus_ground_truth():
    cases = _corpus_cases()
    if not cases:
        pytest.skip("synthetic corpus not available")
    out = HERE / "corpus" / "out"
    for case in cases:
        ps = _panels(out / case["pdf"], case.get("page", 0))
        assert len(ps) == len(case["panels"]), case["case"]
        for gt in case["panels"]:
            p = max(ps, key=lambda p: _iou(p.bbox, gt["bbox_hint"]))
            assert _iou(p.bbox, gt["bbox_hint"]) > 0.95, case["case"]
            assert p.x.scale == gt["xscale"] and p.y.scale == gt["yscale"], case["case"]
            _check_limits(p, gt["xlim"], gt["ylim"], gt["xscale"] == "log", gt["yscale"] == "log")


# --------------------------------------------------------------------------- the real hard page
@pytest.mark.skipif(not ZHENG.exists(), reason="user's Zotero library not available")
def test_zheng_outlined_page():
    ps = {p.label: p for p in _panels(ZHENG, 3)}
    jv_x = [0.0, 0.2, 0.4, 0.6, 0.8]
    jv_y = [-20.0, -15.0, -10.0, -5.0, 0.0]
    eqe_x = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]
    eqe_y = [float(v) for v in range(0, 101, 10)]
    for label, xs, ys in (("d", jv_x, jv_y), ("f", jv_x, jv_y), ("e", eqe_x, eqe_y), ("g", eqe_x, eqe_y)):
        p = ps[label]
        for calib, want in ((p.x, xs), (p.y, ys)):
            got = sorted(v for _, v in calib.ticks)
            assert got == want, (label, calib.orient, got)
            step = want[1] - want[0]
            for pos, v in calib.ticks:
                assert abs(calib.to_data(pos) - v) < 0.01 * step
        assert p.x.scale == p.y.scale == "linear"
    # J-V: -20 is at the TOP of the plot
    d = ps["d"]
    assert d.y.to_data(d.bbox[1]) < d.y.to_data(d.bbox[3])
    assert "c" in ps                                     # the Nyquist plot is found too
    assert len(ps) == 5                                  # and nothing else on the page
