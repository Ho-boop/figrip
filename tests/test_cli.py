"""CLI, export and preview tests - all on a synthetic one-page PDF and a hand-built FigureResult.

`pipeline.extract` is monkeypatched, so these tests do not depend on the axes/series agents' code.
"""
import csv
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pymupdf
import pytest

import figrip
from figrip import cli, export, opv, pipeline, preview
from figrip.model import AxisCalib, FigureResult, Panel, Series

BBOX = (60.0, 30.0, 260.0, 190.0)            # plot area on the 300 x 220 pt test page


def _calibs():
    # x: -0.1 V .. 1.1 V left to right;  y: J = -25 (top) .. +5 (bottom) mA/cm2, i.e. an "inverted" axis
    x = AxisCalib.fit("x", [(60 + (v + 0.1) / 1.2 * 200, v) for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)],
                      label="Voltage (V)")
    y = AxisCalib.fit("y", [(30 + (j + 25) / 30 * 160, j) for j in (-20, -15, -10, -5, 0)],
                      label="Current density (mA cm-2)")
    return x, y


def _curve(shift=0.0):
    v = np.linspace(-0.1, 1.05, 120)
    j = -(20.0 - 1e-9 * (np.exp((v + shift) / 0.0388) - 1.0))
    return v, j


def make_series(name, shift=0.0, color=(0, 0, 0)):
    x, y = _calibs()
    v, j = _curve(shift)
    page_pts = np.column_stack([x.to_page(v), y.to_page(j)])
    return Series(name=name, kind="line", style={"color": color}, page_pts=page_pts,
                  data=np.column_stack([v, j]))


def make_result(page_no=3, label="d", names=("PM6:Y6", "PM6:Y6 + 5%")):
    x, y = _calibs()
    series = [make_series(n, shift=0.02 * k) for k, n in enumerate(names)]
    return FigureResult(Panel(page_no=page_no, bbox=BBOX, x=x, y=y, label=label), series)


@pytest.fixture
def pdf(tmp_path):
    """4-page PDF; page 4 (index 3) holds a plot-like drawing, page 1 a raster image."""
    doc = pymupdf.open()
    for _ in range(4):
        doc.new_page(width=300, height=220)
    pg = doc[3]
    pg.draw_rect(pymupdf.Rect(*BBOX), color=(0, 0, 0), width=0.8)
    x, y = _calibs()
    for t, val in x.ticks:
        pg.insert_text((t - 6, 205), f"{val:g}", fontsize=7)
    for t, val in y.ticks:
        pg.insert_text((30, t + 2), f"{val:g}", fontsize=7)
    s = make_series("c")
    pg.draw_polyline([pymupdf.Point(*p) for p in s.page_pts], color=(0.2, 0.2, 0.2), width=1.0)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
    pix.clear_with(200)
    doc[0].insert_image(pymupdf.Rect(50, 50, 250, 150), pixmap=pix)
    p = tmp_path / "paper.pdf"
    doc.save(str(p))
    doc.close()
    return str(p)


@pytest.fixture
def fake_extract(monkeypatch):
    """Replace pipeline.extract; records calls and returns one panel on page index 3."""
    calls = []

    def fake(pdf_path, pages=None, ocr=True, **kw):
        calls.append(dict(pdf=pdf_path, pages=pages, ocr=ocr, **kw))
        res = [make_result()]
        return [r for r in res if pages is None or r.panel.page_no in pages]

    monkeypatch.setattr(pipeline, "extract", fake)
    return calls


# ------------------------------------------------------------------ argument parsing

def test_parse_pages_is_one_based():
    assert cli.parse_pages("4", 10) == [3]
    assert cli.parse_pages("3-5", 10) == [2, 3, 4]
    assert cli.parse_pages("1,4,6-7", 10) == [0, 3, 5, 6]
    assert cli.parse_pages(None, 10) is None
    for bad in ("0", "11", "x", "5-3"):
        with pytest.raises(cli.CliError):
            cli.parse_pages(bad, 10)


def test_scan_converts_page_and_lists_panel(pdf, fake_extract, capsys):
    assert cli.main(["scan", pdf, "-p", "4"]) == 0
    assert fake_extract[0]["pages"] == [3]               # 1-based on the command line -> 0-based inside
    assert fake_extract[0]["ocr"] is True
    out = capsys.readouterr().out
    assert "Voltage (V)" in out and "Current density" in out
    row = [ln for ln in out.splitlines() if ln.startswith("4 ")][0]      # page column shows 4, not 3
    assert " d " in row and "60,30,260,190" in row


def test_no_ocr_flag(pdf, fake_extract):
    cli.main(["scan", pdf, "-p", "4", "--no-ocr"])
    assert fake_extract[0]["ocr"] is False


# ------------------------------------------------------------------ extract

def test_extract_writes_all_formats_and_preview(pdf, fake_extract, tmp_path, capsys):
    out = tmp_path / "out"
    rc = cli.main(["extract", pdf, "-p", "4", "-o", str(out), "--format", "all", "--preview", "--dpi", "80"])
    assert rc == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["p4_d.csv", "p4_d.png", "p4_d_origin.txt", "paper.json"]
    assert (out / "p4_d.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    stdout = capsys.readouterr().out
    assert "PM6:Y6" in stdout and "120" in stdout


def test_extract_default_format_is_csv_only(pdf, fake_extract, tmp_path):
    out = tmp_path / "o"
    assert cli.main(["extract", pdf, "-p", "4", "-o", str(out)]) == 0
    assert [p.name for p in out.iterdir()] == ["p4_d.csv"]


def test_panel_selection_by_letter_and_index(pdf, fake_extract, tmp_path):
    assert cli.main(["extract", pdf, "-p", "4", "--panel", "d", "-o", str(tmp_path / "a")]) == 0
    assert cli.main(["extract", pdf, "-p", "4", "--panel", "1", "-o", str(tmp_path / "b")]) == 0
    assert cli.main(["extract", pdf, "-p", "4", "--panel", "2", "-o", str(tmp_path / "c")]) == cli.EXIT_NOTHING
    assert cli.main(["extract", pdf, "-p", "4", "--panel", "z", "-o", str(tmp_path / "e")]) == cli.EXIT_NOTHING


# ------------------------------------------------------------------ jv

def test_jv_prints_table_and_writes_tool_files(pdf, fake_extract, tmp_path, capsys):
    out = tmp_path / "jv"
    assert cli.main(["jv", pdf, "-p", "4", "--panel", "d", "--pin", "100", "-o", str(out)]) == 0
    files = sorted(p.name for p in out.iterdir())
    assert files == ["PM6_Y6 + 5%.txt", "PM6_Y6.txt"]                      # ':' is illegal on Windows -> '_'
    txt = (out / "PM6_Y6.txt").read_bytes().decode()
    h, _, v = (ln.split("\t") for ln in txt.split("\r\n")[:3])
    d = dict(zip(h, v))
    assert d["Device"] == "PM6:Y6" and float(d["Jsc"]) == pytest.approx(20.0, abs=0.02)
    assert 900 < float(d["Voc"]) < 940 and 70 < float(d["FF"]) < 90
    stdout = capsys.readouterr().out
    assert "Jsc (mA/cm2)" in stdout and "PM6:Y6 + 5%" in stdout


def test_jv_requires_page(pdf, fake_extract, capsys):
    assert cli.main(["jv", pdf]) == cli.EXIT_USAGE
    assert "-p" in capsys.readouterr().err


def test_jv_reports_series_that_is_not_a_jv(pdf, monkeypatch, tmp_path, capsys):
    r = make_result(names=("good",))
    bad = Series("flat", "line", {}, np.zeros((5, 2)), np.column_stack([np.linspace(0, 1, 5), np.zeros(5)]))
    r.series.append(bad)
    monkeypatch.setattr(pipeline, "extract", lambda *a, **k: [r])
    assert cli.main(["jv", pdf, "-p", "4", "-o", str(tmp_path / "o")]) == 0
    assert "flat: not analysed" in capsys.readouterr().err


# ------------------------------------------------------------------ errors / exit codes

def test_missing_file_and_bad_page(tmp_path, pdf, fake_extract, capsys):
    assert cli.main(["scan", str(tmp_path / "nope.pdf")]) == cli.EXIT_USAGE
    assert "file not found" in capsys.readouterr().err
    assert cli.main(["scan", pdf, "-p", "9"]) == cli.EXIT_USAGE
    assert "does not exist" in capsys.readouterr().err


def test_rasterised_page_message(pdf, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "extract", lambda *a, **k: [])
    assert cli.main(["scan", pdf, "-p", "1"]) == cli.EXIT_NOTHING          # page 1 holds only an image
    err = capsys.readouterr().err
    assert "page 1 has no vector plot" in err and "WebPlotDigitizer" in err


def test_vector_page_without_calibration_message(pdf, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "extract", lambda *a, **k: [])
    # page 4 has few drawings in this synthetic file -> falls under "no figure content" or "vector graphics"
    assert cli.main(["scan", pdf, "-p", "4"]) == cli.EXIT_NOTHING
    assert "page 4" in capsys.readouterr().err


def test_crash_in_pipeline_is_a_clean_error(pdf, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(pipeline, "extract", boom)
    assert cli.main(["scan", pdf]) == cli.EXIT_FAIL
    err = capsys.readouterr().err
    assert "kaboom" in err and "Traceback" not in err


# ------------------------------------------------------------------ manual hints

def test_hints_are_forwarded_when_pipeline_accepts_them(pdf, monkeypatch):
    seen = {}

    def fake(pdf_path, pages=None, ocr=True, hints=None):
        seen["hints"] = hints
        return [make_result()]

    monkeypatch.setattr(pipeline, "extract", fake)
    cli.main(["scan", pdf, "-p", "4", "--x-ticks", "0,0.2,0.4", "--y-ticks", "-20,-10,0",
              "--bbox", "60,30,260,190", "--xscale", "log", "--yscale", "linear"])
    assert seen["hints"] == {"bbox": (60.0, 30.0, 260.0, 190.0), "x_ticks": [0, 0.2, 0.4],
                             "y_ticks": [-20, -10, 0], "xscale": "log", "yscale": "linear"}


def test_hints_reach_axes_find_panels(pdf, monkeypatch):
    """--x/y-ticks flags travel through pipeline.extract into axes.find_panels."""
    got = {}

    def find_panels(pc, page=None, hints=None):
        got["hints"] = hints
        return [make_result().panel]

    fake_axes = types.ModuleType("figrip.axes")
    fake_axes.find_panels = find_panels
    fake_series = types.ModuleType("figrip.series")
    fake_series.extract_series = lambda pc, panel: [make_series("s")]
    monkeypatch.setitem(sys.modules, "figrip.axes", fake_axes)
    monkeypatch.setitem(sys.modules, "figrip.series", fake_series)
    monkeypatch.setattr(figrip, "axes", fake_axes, raising=False)
    monkeypatch.setattr(figrip, "series", fake_series, raising=False)

    assert cli.main(["scan", pdf, "-p", "4", "--y-ticks", "-20,-10,0"]) == 0
    assert got["hints"] == {"y_ticks": [-20.0, -10.0, 0.0]}


def test_bad_hint_values(pdf, fake_extract, capsys):
    assert cli.main(["scan", pdf, "--x-ticks", "0,a"]) == cli.EXIT_USAGE
    assert cli.main(["scan", pdf, "--bbox", "1,2,3"]) == cli.EXIT_USAGE


def test_python_dash_m_entry_point(tmp_path):
    root = Path(__file__).resolve().parent.parent
    p = subprocess.run([sys.executable, "-m", "figrip", "scan", str(tmp_path / "x.pdf")],
                       cwd=root, capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == cli.EXIT_USAGE and "file not found" in p.stderr


# ------------------------------------------------------------------ export

def test_csv_wide_and_long(tmp_path):
    r = make_result()
    r.series[1].data = r.series[1].data[:50]                  # unequal lengths -> padded
    p = export.to_csv(r, tmp_path / "w.csv")
    rows = list(csv.reader(open(p, encoding="utf-8-sig")))
    assert rows[0] == ["PM6:Y6_x", "PM6:Y6_y", "PM6:Y6 + 5%_x", "PM6:Y6 + 5%_y"]
    assert len(rows) == 121 and rows[-1][2:] == ["", ""]
    assert float(rows[1][0]) == pytest.approx(-0.1)
    p = export.to_csv(r, tmp_path / "l.csv", layout="long")
    rows = list(csv.reader(open(p, encoding="utf-8-sig")))
    assert rows[0] == ["series", "x", "y"] and len(rows) == 1 + 120 + 50
    with pytest.raises(ValueError):
        export.to_csv(r, tmp_path / "bad.csv", layout="diagonal")


def test_unnamed_and_duplicate_series_names():
    r = make_result(names=(None, "a"))
    r.series.append(make_series("a"))
    assert export.series_names(r) == ["series1", "a", "a_2"]


def test_json_has_calibration_and_valid_numbers(tmp_path):
    r = make_result()
    r.series[0].data[3, 1] = np.nan                           # NaN must become null, not invalid JSON
    p = export.to_json([r], tmp_path / "r.json")
    doc = json.loads(p.read_text(encoding="utf-8"))           # strict parser would reject NaN
    pan = doc["panels"][0]
    assert pan["page"] == 4 and pan["label"] == "d"
    assert len(pan["x_axis"]["ticks"]) == 6 and pan["y_axis"]["ticks"][0]["value"] == -20
    assert pan["x_axis"]["label"] == "Voltage (V)" and pan["series"][0]["y"][3] is None
    assert len(pan["series"][0]["page_pts"]) == 120


def test_origin_txt_header_rows(tmp_path):
    r = make_result()
    p = export.to_origin_txt(r, tmp_path / "o.txt")
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].split("\t") == ["Voltage", "Current density"] * 2
    assert lines[1].split("\t") == ["V", "mA cm-2"] * 2
    assert lines[2].split("\t")[:2] == ["PM6:Y6", "PM6:Y6"]
    assert len(lines) == 3 + 120 and len(lines[3].split("\t")) == 4


def test_label_and_filename_helpers():
    assert export.split_label_unit("Voltage (V)") == ("Voltage", "V")
    assert export.split_label_unit("J [mA cm-2]") == ("J", "mA cm-2")
    assert export.split_label_unit("Wavelength") == ("Wavelength", "")
    assert export.split_label_unit(None) == ("", "")
    assert export.sanitize_filename('a:b/c*"d"') == "a_b_c__d"
    assert export.sanitize_filename("  ", "x") == "x"


# ------------------------------------------------------------------ preview

def test_overlay_png_shows_points_in_series_colour(pdf, tmp_path):
    r = make_result(names=("only",))
    out = preview.render_overlay(pdf, r, tmp_path / "ov.png", dpi=100)
    pix = pymupdf.Pixmap(str(out))
    assert pix.width > 250 and pix.height > 150
    px = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].astype(int)
    red = (px[..., 0] > 170) & (px[..., 1] < 90) & (px[..., 2] < 90)         # PALETTE[0]
    orange = (px[..., 0] > 190) & (abs(px[..., 1] - 90) < 50) & (px[..., 2] < 60)  # tick marks
    assert red.sum() > 30
    assert orange.sum() > 5
    # the source PDF must be untouched
    with pymupdf.open(pdf) as d:
        assert len(d) == 4


def test_overlay_handles_no_series_and_non_latin_names(pdf, tmp_path):
    r = make_result(names=("器件 α",))
    preview.render_overlay(pdf, r, tmp_path / "a.png", dpi=60)
    r.series.clear()
    preview.render_overlay(pdf, r, tmp_path / "b.png", dpi=60)
    assert (tmp_path / "a.png").exists() and (tmp_path / "b.png").exists()
