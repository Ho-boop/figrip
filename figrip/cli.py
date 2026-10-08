"""Command line interface:  figrip scan | extract | jv

Page numbers on the command line are 1-based (as printed on the paper's page);
internally everything is 0-based like PyMuPDF.

Exit codes
    0  success
    1  unexpected failure inside the extraction (add --debug for the traceback)
    2  bad usage / unreadable input (missing file, page out of range, malformed option)
    3  nothing to report: no (calibratable) vector plot on the selected pages / panel not found
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

import pymupdf

from . import export, opv, pipeline
from .model import FigureResult

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOTHING = 0, 1, 2, 3


class CliError(Exception):
    """A user-facing error: message goes to stderr, `code` becomes the exit status."""

    def __init__(self, message: str, code: int = EXIT_USAGE):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# argument parsing helpers
# --------------------------------------------------------------------------

def _floats(text: str, what: str, min_n: int = 1, exact: Optional[int] = None) -> list[float]:
    try:
        vals = [float(t) for t in text.replace(";", ",").split(",") if t.strip() != ""]
    except ValueError:
        raise CliError(f"{what}: cannot parse {text!r} as a comma-separated list of numbers")
    if exact is not None and len(vals) != exact:
        raise CliError(f"{what}: expected {exact} numbers, got {len(vals)}")
    if len(vals) < min_n:
        raise CliError(f"{what}: need at least {min_n} numbers, got {len(vals)}")
    return vals


def parse_pages(spec: Optional[str], n_pages: int) -> Optional[list[int]]:
    """'4' / '3-5' / '1,4,6-7' (1-based) -> sorted unique 0-based page indices; None -> None (= all pages)."""
    if spec is None:
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = (int(t) for t in part.split("-", 1))
            else:
                a = b = int(part)
        except ValueError:
            raise CliError(f"-p/--pages: cannot parse {part!r} (use e.g. 4, 3-5 or 1,4,6-7)")
        if a < 1 or b < a:
            raise CliError(f"-p/--pages: bad range {part!r}; pages are numbered from 1")
        if b > n_pages:
            raise CliError(f"page {b} does not exist: the PDF has {n_pages} page(s)")
        pages.update(range(a - 1, b))
    if not pages:
        raise CliError("-p/--pages: no page given")
    return sorted(pages)


def build_hints(args) -> Optional[dict]:
    """Manual calibration overrides. Keys follow the schema defined by axes.find_panels:
    {"bbox": (x0,y0,x1,y1), "x_ticks": [...], "y_ticks": [...], "xscale": "log", "yscale": "log"}.
    Order matters: x ticks are listed LEFT -> RIGHT, y ticks BOTTOM -> TOP (as they sit on the page)."""
    hints: dict = {}
    if getattr(args, "bbox", None):
        hints["bbox"] = tuple(_floats(args.bbox, "--bbox", exact=4))
    if getattr(args, "x_ticks", None):
        hints["x_ticks"] = _floats(args.x_ticks, "--x-ticks", min_n=2)
    if getattr(args, "y_ticks", None):
        hints["y_ticks"] = _floats(args.y_ticks, "--y-ticks", min_n=2)
    if getattr(args, "xscale", None):
        hints["xscale"] = args.xscale
    if getattr(args, "yscale", None):
        hints["yscale"] = args.yscale
    return hints or None


# --------------------------------------------------------------------------
# running the pipeline
# --------------------------------------------------------------------------

def run_pipeline(pdf: str, pages0: Optional[list[int]], ocr: bool, hints: Optional[dict]) -> list[FigureResult]:
    return pipeline.extract(pdf, pages=pages0, ocr=ocr, hints=hints)


def _diagnose(pdf: str, page0: int) -> str:
    """Explain *why* a page produced nothing, in words a user can act on."""
    with pymupdf.open(pdf) as doc:
        page = doc[page0]
        n_draw = len(page.get_drawings())
        n_img = len(page.get_images())
    n = page0 + 1
    if n_draw < 20 and n_img:
        return f"page {n} has no vector plot (figure looks rasterised - try WebPlotDigitizer)"
    if n_draw >= 20:
        return (f"page {n} has vector graphics ({n_draw} drawing operations) but no plot with readable "
                f"axes was recognised; calibrate by hand with --bbox / --x-ticks / --y-ticks")
    return f"page {n} has no figure content"


def load_results(args, need_series: bool = True) -> tuple[str, list[FigureResult]]:
    """Open the PDF, validate pages, run the pipeline; raise CliError if nothing is found."""
    pdf = args.pdf
    if not Path(pdf).is_file():
        raise CliError(f"file not found: {pdf}")
    try:
        with pymupdf.open(pdf) as doc:
            n_pages = len(doc)
    except Exception as e:                                 # corrupt / not a PDF
        raise CliError(f"cannot open {pdf} as a PDF: {e}")
    pages0 = parse_pages(args.pages, n_pages)
    hints = build_hints(args)

    try:
        results = run_pipeline(pdf, pages0, not args.no_ocr, hints)
    except CliError:
        raise
    except ImportError as e:
        raise CliError(f"a figrip component is not available yet: {e}", EXIT_FAIL)
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        raise CliError(f"extraction failed: {type(e).__name__}: {e}  (use --debug for the traceback)", EXIT_FAIL)

    if not results:
        if pages0 is not None:
            msg = "; ".join(_diagnose(pdf, p) for p in pages0)
        else:
            msg = "no plot with readable axes found in this PDF (figures look rasterised? try WebPlotDigitizer)"
        raise CliError(msg, EXIT_NOTHING)
    return pdf, results


def number_panels(results: list[FigureResult]) -> list[tuple[FigureResult, int]]:
    """Pair every result with its 1-based index among the panels of its own page."""
    counters: dict[int, int] = {}
    out = []
    for r in results:
        counters[r.panel.page_no] = counters.get(r.panel.page_no, 0) + 1
        out.append((r, counters[r.panel.page_no]))
    return out


def select_panels(results: list[FigureResult], panel_arg: Optional[str]) -> list[tuple[FigureResult, int]]:
    """--panel d  -> panels whose detected letter is 'd';  --panel 2 -> 2nd panel on each selected page."""
    numbered = number_panels(results)
    if panel_arg is None:
        return numbered
    key = panel_arg.strip().lower()
    if key.isdigit():
        sel = [(r, i) for r, i in numbered if i == int(key)]
    else:
        sel = [(r, i) for r, i in numbered if (r.panel.label or "").lower() == key]
    if not sel:
        avail = ", ".join(f"p{r.panel.page_no + 1}#{i}" + (f"({r.panel.label})" if r.panel.label else "")
                          for r, i in numbered)
        raise CliError(f"no panel {panel_arg!r} on the selected pages; found: {avail}", EXIT_NOTHING)
    return sel


# --------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]]) -> str:
    cols = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    fmt = lambda row: "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip()
    return "\n".join([fmt(headers), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows])


def _g(x: float) -> str:
    return f"{x:.4g}"


def _range(a, b) -> str:
    return f"{_g(a)}..{_g(b)}"


def _axis_text(ax, lo: float, hi: float) -> str:
    lab = ax.label or "?"
    return f"{lab} [{_range(lo, hi)}]" + (" log" if ax.scale == "log" else "")


def _panel_ranges(r: FigureResult):
    p = r.panel
    x0, y0, x1, y1 = p.bbox
    xs = p.x.to_data([x0, x1])
    ys = p.y.to_data([y1, y0])          # page y grows downward: bottom edge = lower data value (usually)
    return (min(xs), max(xs)), (min(ys), max(ys))


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_scan(args) -> int:
    pdf, results = load_results(args)
    rows = []
    for r, i in number_panels(results):
        p = r.panel
        (xl, xh), (yl, yh) = _panel_ranges(r)
        bbox = ",".join(f"{v:.0f}" for v in p.bbox)
        rows.append([str(p.page_no + 1), str(i), p.label or "-", bbox,
                     _axis_text(p.x, xl, xh), _axis_text(p.y, yl, yh), str(len(r.series)),
                     f"{p.x.source}/{p.y.source}"])
    print(_table(["page", "#", "panel", "bbox (pt)", "x axis", "y axis", "series", "calib"], rows))
    pages = {r.panel.page_no for r in results}
    print(f"\n{len(results)} panel(s) on {len(pages)} page(s).")
    for r, i in number_panels(results):
        for n in r.panel.notes:
            _warn(f"page {r.panel.page_no + 1} panel #{i}: {n}")
    return EXIT_OK


def cmd_extract(args) -> int:
    pdf, results = load_results(args)
    selected = select_panels(results, args.panel)
    outdir = Path(args.outdir or f"{Path(pdf).stem}_figrip")
    outdir.mkdir(parents=True, exist_ok=True)
    formats = {"csv", "json", "origin"} if args.format == "all" else {args.format}

    written: list[Path] = []
    rows = []
    for r, i in selected:
        stem = export.panel_stem(r, i)
        names = export.series_names(r)
        if "csv" in formats:
            written.append(export.to_csv(r, outdir / f"{stem}.csv"))
        if "origin" in formats:
            written.append(export.to_origin_txt(r, outdir / f"{stem}_origin.txt"))
        if args.preview:
            try:
                from . import preview
                written.append(preview.render_overlay(pdf, r, outdir / f"{stem}.png", dpi=args.dpi))
            except Exception as e:                       # a failed picture must not lose the data
                if args.debug:
                    traceback.print_exc()
                _warn(f"preview for {stem} failed: {type(e).__name__}: {e}")
        for n in r.panel.notes:
            _warn(f"{stem}: {n}")
        for name, s in zip(names, r.series):
            d = s.data.reshape(-1, 2)
            if len(d):
                xr, yr = _range(d[:, 0].min(), d[:, 0].max()), _range(d[:, 1].min(), d[:, 1].max())
            else:
                xr = yr = "-"
            rows.append([str(r.panel.page_no + 1), r.panel.label or str(i), name, s.kind, str(len(d)), xr, yr])
        if not r.series:
            _warn(f"{stem}: axes were calibrated but no data series was found")
            rows.append([str(r.panel.page_no + 1), r.panel.label or str(i), "(none)", "-", "0", "-", "-"])
    if "json" in formats:
        written.append(export.to_json([r for r, _ in selected], outdir / f"{Path(pdf).stem}.json"))

    print(_table(["page", "panel", "series", "kind", "points", "x range", "y range"], rows))
    print("\nwritten:")
    for w in written:
        print(f"  {w}")
    return EXIT_OK


_J_LABEL_HINTS = ("current", "mA", "A cm", "A/cm", "A m", "J ", "J(", "(J")


def _looks_like_jv(r: FigureResult) -> bool:
    lab = r.panel.y.label or ""
    return lab.strip().startswith("J") or any(h in lab for h in _J_LABEL_HINTS)


def cmd_jv(args) -> int:
    if args.pages is None:
        raise CliError("jv needs a page: -p N  (use `figrip scan` to find the J-V panel)")
    pdf, results = load_results(args)
    if args.panel is None:
        selected = [(r, i) for r, i in number_panels(results) if _looks_like_jv(r)]
        if not selected:
            _warn("no panel has a J/current-like y-axis label; analysing every panel (use --panel to choose)")
            selected = number_panels(results)
    else:
        selected = select_panels(results, args.panel)

    outdir = Path(args.outdir or f"{Path(pdf).stem}_jv")
    outdir.mkdir(parents=True, exist_ok=True)
    multi = len(selected) > 1
    used: set[str] = set()
    rows, notes, written = [], [], []
    run_id = 0

    for r, i in selected:
        pstem = export.panel_stem(r, i)
        j_unit = args.j_unit or opv.guess_j_unit(r.panel.y.label)
        v_unit = args.v_unit or opv.guess_v_unit(r.panel.x.label)
        for name, s in zip(export.series_names(r), r.series):
            d = s.data.reshape(-1, 2)
            try:
                m = opv.jv_metrics(d[:, 0], d[:, 1], pin=args.pin, j_unit=j_unit, v_unit=v_unit)
            except ValueError as e:
                rows.append([pstem, name, "-", "-", "-", "-", "-"])
                notes.append(f"{pstem}/{name}: not analysed - {e}")
                continue
            for n in m["notes"]:
                notes.append(f"{pstem}/{name}: {n}")
            for sc in m["scans"]:
                label = name if len(m["scans"]) == 1 else f"{name} ({sc['direction']})"
                base = export.sanitize_filename(label.replace(" (", "_").replace(")", ""))
                base = f"{pstem}_{base}" if multi else base
                fname, k = base, 2
                while fname.lower() in used:             # same series name twice -> don't overwrite
                    fname, k = f"{base}_{k}", k + 1
                used.add(fname.lower())
                run_id += 1
                written.append(opv.write_opv_txt(outdir / f"{fname}.txt", sc, run_id=run_id, device=name))
                rows.append([pstem, label, f"{sc['jsc']:.2f}", f"{sc['voc']:.3f}",
                             f"{sc['ff']:.1f}", f"{sc['pce']:.2f}", str(sc["n_points"])])

    print(_table(["panel", "series", "Jsc (mA/cm2)", "Voc (V)", "FF (%)", "PCE (%)", "points"], rows))
    print(f"\n(Pin = {args.pin:g} mW/cm2; Jsc by linear interpolation at V=0, Voc at J=0, MPP on a dense interpolation)")
    for n in notes:
        _warn(n)
    if written:
        print("\nwritten (ready for opv_performance_tool):")
        for w in written:
            print(f"  {w}")
        return EXIT_OK
    raise CliError("no series could be analysed as a J-V curve", EXIT_NOTHING)


# --------------------------------------------------------------------------
# parser / entry point
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("pdf", help="path of the paper (PDF)")
    common.add_argument("-p", "--pages", metavar="N", help="page(s), 1-based: 4, 3-5 or 1,4,6-7 (default: all)")
    common.add_argument("--no-ocr", action="store_true", help="never OCR tick labels (faster; needs real text labels)")
    common.add_argument("--debug", action="store_true", help="print full tracebacks")
    manual = argparse.ArgumentParser(add_help=False)
    g = manual.add_argument_group("manual calibration (when automatic detection fails)")
    g.add_argument("--x-ticks", metavar="LIST", help='labelled x tick values, LEFT to RIGHT, e.g. "0,0.2,0.4,0.6,0.8"')
    g.add_argument("--y-ticks", metavar="LIST", help='labelled y tick values, BOTTOM to TOP as on the page, e.g. "0,-5,-10,-15,-20" if -20 is drawn at the top')
    g.add_argument("--bbox", metavar="x0,y0,x1,y1", help="plot area in PDF points (see `figrip scan`)")
    g.add_argument("--xscale", choices=["linear", "log"], help="force the x axis scale")
    g.add_argument("--yscale", choices=["linear", "log"], help="force the y axis scale")

    ap = argparse.ArgumentParser(prog="figrip", description="Recover exact data from vector plots in PDF papers.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("scan", parents=[common, manual], help="list the plots (panels) found in a PDF")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("extract", parents=[common, manual], help="extract all series of the plots to files")
    sp.add_argument("--panel", metavar="N|a", help="only this panel: 1-based index on the page, or its letter")
    sp.add_argument("-o", "--outdir", metavar="DIR", help="output folder (default: <paper>_figrip)")
    sp.add_argument("--format", choices=["csv", "json", "origin", "all"], default="csv",
                    help="csv (Excel/pandas), json (with calibration), origin (OriginPro import); default csv")
    sp.add_argument("--preview", action="store_true", help="also write an overlay PNG per panel to check the result")
    sp.add_argument("--dpi", type=int, default=200, help="preview resolution (default 200)")
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("jv", parents=[common, manual], help="J-V curve -> Jsc, Voc, FF, PCE + opv_performance_tool files")
    sp.add_argument("--panel", metavar="N|a", help="panel index or letter (default: panels whose y label looks like current density)")
    sp.add_argument("--pin", type=float, default=100.0, help="incident power in mW/cm2 (default 100 = AM1.5G)")
    sp.add_argument("--j-unit", metavar="U", help="current-density unit of the y axis (default: read from its label, else mA/cm2)")
    sp.add_argument("--v-unit", metavar="U", help="voltage unit of the x axis: V or mV (default: from label)")
    sp.add_argument("-o", "--outdir", metavar="DIR", help="output folder (default: <paper>_jv)")
    sp.set_defaults(func=cmd_jv)
    return ap


_LIST_OPTIONS = ("--x-ticks", "--y-ticks", "--bbox")


def _glue_option_values(argv: list[str]) -> list[str]:
    """`--y-ticks -20,-10,0` -> `--y-ticks=-20,-10,0`.

    argparse mistakes a value that starts with '-' (but is not a plain number, like "-20,-10")
    for another option and fails with "expected one argument".  Negative tick lists are the
    normal case for J-V plots, so we glue them to their option first.
    """
    out, i = [], 0
    while i < len(argv):
        if argv[i] in _LIST_OPTIONS and i + 1 < len(argv):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv: Optional[list[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):        # the Windows console is GBK; paper titles are not
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(_glue_option_values(sys.argv[1:] if argv is None else list(argv)))
    try:
        return args.func(args)
    except CliError as e:
        print(f"figrip: error: {e}", file=sys.stderr)
        return e.code
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
