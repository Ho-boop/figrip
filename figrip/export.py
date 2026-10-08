"""Write extraction results to files that Excel / Origin / pandas open without fuss.

* ``to_csv``         one CSV per panel (``FigureResult``); wide or long layout
* ``to_json``        everything, including the axis calibration (tick pairs), for reproducibility
* ``to_origin_txt``  tab-separated with Long name / Units / Comments rows -> Origin's import wizard
                     turns them into column labels automatically

All text files are UTF-8 *with BOM* so that Excel on Windows shows Greek letters / Chinese correctly.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from .model import AxisCalib, FigureResult

ENCODING = "utf-8-sig"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def series_names(result: FigureResult) -> list[str]:
    """Display name per series; unnamed ones become 'series1', 'series2' ...
    Duplicate names get a numeric suffix so that CSV headers stay unique."""
    names, seen = [], {}
    for i, s in enumerate(result.series, 1):
        base = (s.name or "").strip() or f"series{i}"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return names


def sanitize_filename(name: str, fallback: str = "series") -> str:
    """Make `name` safe as a Windows file name (keeps unicode letters, drops <>:"/\\|?*)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" ._")
    s = re.sub(r"\s+", " ", s)
    return s[:100] or fallback


def panel_stem(result: FigureResult, index: int = 1) -> str:
    """File stem for a panel: 'p4_d' (page 4, panel letter d) or 'p4_2' (second panel, no letter).
    Page numbers are 1-based (what the reader sees)."""
    pano = result.panel.page_no + 1
    tag = sanitize_filename(result.panel.label, "") if result.panel.label else ""
    return f"p{pano}_{tag or index}"


def split_label_unit(label: Optional[str]) -> tuple[str, str]:
    """'Voltage (V)' -> ('Voltage', 'V');  'J [mA cm-2]' -> ('J', 'mA cm-2');  'Wavelength' -> ('Wavelength', '')."""
    if not label:
        return "", ""
    m = re.match(r"^(.*?)\s*[\(\[]\s*([^()\[\]]+?)\s*[\)\]]\s*$", label.strip())
    if m and m.group(1):
        return m.group(1).strip(), m.group(2).strip()
    return label.strip(), ""


def _xy(series) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(series.data, float).reshape(-1, 2)
    return d[:, 0], d[:, 1]


def _fmt(x: float) -> str:
    # repr-style shortest round-trip float: no precision lost, no trailing zeros
    return "" if not math.isfinite(x) else repr(float(x))


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def to_csv(result: FigureResult, path: Union[str, Path], layout: str = "wide") -> Path:
    """One CSV for one panel.

    layout="wide" (default): every series gets two columns ``<name>_x, <name>_y`` side by side;
        shorter series are padded with empty cells. Drag-and-drop into Origin / Excel and plot.
    layout="long": three columns ``series,x,y`` - the tidy format for pandas / R.

    Axis titles are not part of the CSV (keeps the header a single clean row); they are in the
    JSON and Origin exports.
    """
    path = Path(path)
    names = series_names(result)
    with open(path, "w", newline="", encoding=ENCODING) as f:
        w = csv.writer(f)
        if layout == "long":
            w.writerow(["series", "x", "y"])
            for name, s in zip(names, result.series):
                x, y = _xy(s)
                for a, b in zip(x, y):
                    w.writerow([name, _fmt(a), _fmt(b)])
        elif layout == "wide":
            cols, header = [], []
            for name, s in zip(names, result.series):
                x, y = _xy(s)
                header += [f"{name}_x", f"{name}_y"]
                cols += [x, y]
            w.writerow(header)
            for i in range(max((len(c) for c in cols), default=0)):
                w.writerow([_fmt(c[i]) if i < len(c) else "" for c in cols])
        else:
            raise ValueError("layout must be 'wide' or 'long'")
    return path


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------

def _axis_dict(ax: AxisCalib) -> dict:
    return {
        "scale": ax.scale,
        "label": ax.label,
        "source": ax.source,
        "a": ax.a, "b": ax.b,                       # value = a*pos + b  (log10(value) for log axes)
        "residual": ax.residual,
        "ticks": [{"pos": p, "value": v} for p, v in ax.ticks],
    }


def _jsonable(o):
    """Recursively convert numpy / tuples / NaN so that json.dump never emits invalid JSON."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def result_to_dict(result: FigureResult, panel_index: Optional[int] = None) -> dict:
    p = result.panel
    names = series_names(result)
    return {
        "page": p.page_no + 1,                      # 1-based, as printed on the paper
        "panel_index": panel_index,
        "label": p.label,
        "bbox": list(p.bbox),
        "notes": list(p.notes),
        "x_axis": _axis_dict(p.x),
        "y_axis": _axis_dict(p.y),
        "series": [
            {
                "name": name,
                "kind": s.kind,
                "style": s.style,
                "n": int(len(s.data)),
                "x": np.asarray(s.data, float).reshape(-1, 2)[:, 0],
                "y": np.asarray(s.data, float).reshape(-1, 2)[:, 1],
                "page_pts": np.asarray(s.page_pts, float).reshape(-1, 2),
            }
            for name, s in zip(names, result.series)
        ],
    }


def to_json(results: Union[FigureResult, Iterable[FigureResult]], path: Union[str, Path]) -> Path:
    """Dump panels (calibration incl. tick pairs, all series in data and page coordinates)."""
    if isinstance(results, FigureResult):
        results = [results]
    doc = {"figrip": 1,
           "coordinates": "page_pts are PDF points, origin top-left, y down; x/y are calibrated data values",
           "panels": [result_to_dict(r, i) for i, r in enumerate(results, 1)]}
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:       # JSON: plain UTF-8, no BOM
        json.dump(_jsonable(doc), f, ensure_ascii=False, indent=1)
    return path


# --------------------------------------------------------------------------
# Origin
# --------------------------------------------------------------------------

def to_origin_txt(result: FigureResult, path: Union[str, Path]) -> Path:
    """Tab-separated file with three header rows (Long name / Units / Comments) then the data.

    In Origin: File > Import > Single ASCII..., tick "Header lines: 3" and set the sub-header
    lines to Long Name / Units / Comments.  Each series has its own X column (series may be
    sampled at different x), laid out X1 Y1 X2 Y2 ... ; the Comments row carries the series name.
    """
    p = result.panel
    xname, xunit = split_label_unit(p.x.label)
    yname, yunit = split_label_unit(p.y.label)
    names = series_names(result)

    long_row, unit_row, comment_row, cols = [], [], [], []
    for name, s in zip(names, result.series):
        x, y = _xy(s)
        long_row += [xname, yname]
        unit_row += [xunit, yunit]
        comment_row += [name, name]
        cols += [x, y]

    with open(path, "w", newline="", encoding=ENCODING) as f:
        for row in (long_row, unit_row, comment_row):
            f.write("\t".join(row) + "\n")
        for i in range(max((len(c) for c in cols), default=0)):
            f.write("\t".join(_fmt(c[i]) if i < len(c) else "" for c in cols) + "\n")
    return Path(path)
