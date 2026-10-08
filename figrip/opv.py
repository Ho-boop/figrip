"""Organic-photovoltaic helpers: J-V figures of merit, EQE integration, instrument-style export.

Everything here works on plain numbers (numpy arrays), so it can be used with
or without the PDF extraction part of figrip.

Conventions used in this module
-------------------------------
* Voltage in V (``v_unit="mV"`` is accepted), current density in mA/cm^2
  (other units via ``j_unit``).
* The *reported* metrics are always positive magnitudes: ``jsc`` > 0, ``voc`` > 0,
  ``jmpp`` > 0, ``pmax`` > 0 (mW/cm^2), regardless of how the paper plotted J.
* ``pin`` is the incident power density in mW/cm^2 (100 = 1 sun, AM1.5G).
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

# --------------------------------------------------------------------------
# 1. unit helpers
# --------------------------------------------------------------------------

# factor that converts "<unit>" to mA/cm^2
_J_FACTORS = {
    "mA/cm2": 1.0,
    "A/cm2": 1e3,
    "uA/cm2": 1e-3,
    "A/m2": 0.1,       # 1 A/m^2 = 0.1 mA/cm^2
    "mA/m2": 1e-4,
}
_V_FACTORS = {"v": 1.0, "mv": 1e-3}


def _match_j_unit(text: str) -> Optional[str]:
    """Find a current-density unit inside free text ('J (mA cm-2)', 'mA/cm²', 'A m^-2' ...)."""
    s = text.lower()
    for a, b in (("µ", "u"), ("μ", "u"), ("−", "-"), ("⁻", "-"), ("²", "2"),
                 ("·", ""), ("^", ""), (" ", ""), ("*", "")):
        s = s.replace(a, b)
    # most specific patterns first: "ma/cm" also contains "a/cm"
    if re.search(r"ma/?cm", s):
        return "mA/cm2"
    if re.search(r"ua/?cm", s):
        return "uA/cm2"
    if re.search(r"ma/?m-?2", s):
        return "mA/m2"
    if re.search(r"a/?cm", s):
        return "A/cm2"
    if re.search(r"a/?m-?2", s):
        return "A/m2"
    return None


def guess_j_unit(axis_label: Optional[str]) -> str:
    """Read the current-density unit out of an axis title such as 'J (mA cm-2)'.

    Falls back to 'mA/cm2', which is what the vast majority of OPV papers use.
    """
    return (_match_j_unit(axis_label) if axis_label else None) or "mA/cm2"


def guess_v_unit(axis_label: Optional[str]) -> str:
    """'Voltage (mV)' -> 'mV'; everything else -> 'V'."""
    if axis_label and re.search(r"[\(\[]\s*mv\s*[\)\]]", axis_label.lower()):
        return "mV"
    return "V"


def _j_factor(j_unit: str) -> float:
    canon = _match_j_unit(j_unit)
    if canon is None:
        raise ValueError(f"unknown current-density unit {j_unit!r}; use mA/cm2, A/cm2, uA/cm2, A/m2 or mA/m2")
    return _J_FACTORS[canon]


# --------------------------------------------------------------------------
# 2. J-V analysis
# --------------------------------------------------------------------------

def _pchip(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Shape-preserving cubic (Fritsch-Carlson) interpolation, numpy only.

    Why not plain cubic splines? Digitised J-V curves can be sparse near the
    knee; a spline overshoots there and would fake a higher Pmax. PCHIP never
    leaves the range of neighbouring points.  x must be strictly increasing.
    """
    n = len(x)
    if n == 2:
        return np.interp(xq, x, y)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    # interior slopes: weighted harmonic mean, zero where the data change direction
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    same = delta[:-1] * delta[1:] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    d[1:-1] = np.where(same, hm, 0.0)

    def end_slope(h0, h1, d0, d1):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(s) > 3 * abs(d0):
            return 3 * d0
        return s

    d[0] = end_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = end_slope(h[-1], h[-2], delta[-1], delta[-2])

    xq = np.asarray(xq, float)
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    t = (xq - x[i]) / h[i]
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h[i] * d[i]
            + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * h[i] * d[i + 1])


def _zigzag_segments(v: np.ndarray, frac: float = 0.05):
    """Split an *ordered* V sequence at its turning points.

    A direction reversal only counts if V backs off by more than ``frac`` of the
    full V range - this ignores pixel-level jitter of digitised curves.
    Returns a list of (start, stop, direction) with direction = +1 / -1.
    """
    n = len(v)
    thr = frac * (v.max() - v.min())
    segs, start, ext, direction = [], 0, 0, 0
    for i in range(1, n):
        if direction == 0:                     # haven't committed to a direction yet
            if abs(v[i] - v[start]) > thr:
                direction = 1 if v[i] > v[start] else -1
                ext = i
            continue
        if direction * (v[i] - v[ext]) > 0:    # still moving the same way: new extreme
            ext = i
        elif direction * (v[ext] - v[i]) > thr:  # backed off far enough: turning point
            segs.append((start, ext + 1, direction))
            start, direction, ext = ext, -direction, i
    if direction == 0:
        direction = 1
    segs.append((start, n, direction))
    return segs


def _clean_sorted(v, j, notes):
    """Sort by V and average J over duplicated V values (interpolation needs strictly increasing x)."""
    order = np.argsort(v, kind="stable")
    v, j = v[order], j[order]
    uv, inv = np.unique(v, return_inverse=True)
    if len(uv) < len(v):
        notes.append(f"{len(v) - len(uv)} duplicated V value(s) averaged")
        j = np.bincount(inv, weights=j) / np.bincount(inv)
        v = uv
    return v, j


def _first_down_crossing(v, jp):
    """First V > 0 where photocurrent jp goes from >0 to <=0 (linear interpolation) or None."""
    for i in range(len(v) - 1):
        if v[i + 1] > 0 and jp[i] > 0 >= jp[i + 1]:
            if jp[i] == jp[i + 1]:
                return float(v[i + 1])
            t = jp[i] / (jp[i] - jp[i + 1])
            return float(v[i] + t * (v[i + 1] - v[i]))
    return None


def _scan_metrics(v: np.ndarray, j: np.ndarray, pin: float, direction: str) -> dict:
    """Figures of merit for ONE monotone-ish sweep (v, j already finite; any order)."""
    notes: list[str] = []
    v, j = _clean_sorted(v, j, notes)
    if len(v) < 4:
        raise ValueError(f"need at least 4 distinct voltage points, got {len(v)}")
    if v.max() <= 0:
        raise ValueError("all voltages are <= 0; this does not look like a J-V curve in the power quadrant")

    # --- Jsc: J at V = 0 -------------------------------------------------
    if v.min() <= 0 <= v.max():
        j0 = float(np.interp(0.0, v, j))                # linear interpolation, as asked
    else:
        # curve starts at V > 0: extrapolate the first few points back to V = 0
        k = min(4, len(v))
        slope, icpt = np.polyfit(v[:k], j[:k], 1)
        j0 = float(icpt)
        notes.append(f"curve starts at V={v[0]:.3g}; Jsc extrapolated to V=0")

    scale = float(np.max(np.abs(j)))
    if abs(j0) < 1e-3 * scale:
        raise ValueError("J(V=0) is ~0: looks like a dark curve, not an illuminated one")

    # --- sign convention -----------------------------------------------
    # Whatever sign the paper used, the photocurrent at V=0 defines "positive generation".
    sign = 1.0 if j0 > 0 else -1.0
    jp = sign * j                                       # photocurrent-positive current
    jsc = abs(j0)

    # --- Voc: J = 0 ------------------------------------------------------
    voc = _first_down_crossing(v, jp)
    voc_extrapolated = False
    if voc is None:
        k = min(5, len(v))
        slope, icpt = np.polyfit(v[-k:], jp[-k:], 1)
        if slope >= 0:
            raise ValueError("J never approaches zero before the curve ends; cannot estimate Voc")
        voc = float(-icpt / slope)
        voc_extrapolated = True
        notes.append(f"curve ends at V={v[-1]:.3g} before J reaches 0; Voc extrapolated linearly "
                     f"from the last {k} points (low confidence)")

    # --- MPP: maximise P = V * J on a dense PCHIP interpolation ------------
    # augment with the exact (0, Jsc) and (Voc, 0) points so the grid can cover [0, Voc]
    keep = (v > 0) & (v < voc)
    xs = np.concatenate([[0.0], v[keep], [voc]])
    ys = np.concatenate([[jsc], jp[keep], [0.0]])
    # points that coincide in V after augmentation would break PCHIP
    xs, ys = _clean_sorted(xs, ys, [])
    grid = np.linspace(0.0, voc, 20001)
    jg = _pchip(xs, ys, grid)
    pg = grid * jg
    k = int(np.argmax(pg))
    vmpp, jmpp, pmax = float(grid[k]), float(jg[k]), float(pg[k])
    if voc_extrapolated:
        notes.append("MPP/FF/PCE rely on the extrapolated tail; treat with caution")

    ff = 100.0 * pmax / (jsc * voc)
    pce = 100.0 * pmax / pin
    if not (15 <= ff <= 92):
        notes.append(f"FF={ff:.1f}% is unusual - check axis calibration and units")
    return dict(jsc=jsc, voc=float(voc), ff=float(ff), pce=float(pce), vmpp=vmpp, jmpp=jmpp,
                pmax=pmax, pin=float(pin), direction=direction, j_sign=int(sign),
                n_points=int(len(v)), notes=notes)


def jv_metrics(v, j, pin: float = 100.0, *, j_unit: str = "mA/cm2", v_unit: str = "V") -> dict:
    """Jsc, Voc, FF, PCE, MPP from a J-V curve.

    Parameters
    ----------
    v, j : array-like
        Voltage and current density, any order, any sign convention (J negative in the
        4th quadrant, or "photocurrent positive"). Duplicated V values are averaged.
        A series that contains a forward AND a reverse sweep (hysteresis loop) is split
        at the turning points and both sweeps are reported.
    pin : float
        Incident power in mW/cm^2 (100 for AM1.5G, 1 sun).
    j_unit, v_unit : str
        Units of the input (``"mA/cm2"``, ``"A/m2"``, ``"A/cm2"``, ``"uA/cm2"``; ``"V"``, ``"mV"``).

    Returns
    -------
    dict with
        jsc [mA/cm2], voc [V], ff [%], pce [%], vmpp [V], jmpp [mA/cm2], pmax [mW/cm2],
        pin, direction, j_sign (sign of J at V=0 in the *input*), n_points, notes (list of str),
        scans: list of per-sweep dicts (same keys). For a single sweep ``scans`` has one entry;
        for a loop the top level repeats the *reverse* sweep (Voc -> Jsc, the one papers quote)
        and ``hysteresis_index = (PCE_rev - PCE_fwd) / PCE_rev`` is added.
    ``direction`` is "reverse" (V decreasing), "forward" (V increasing) for loops and
    "unspecified" when the series is a single sweep (an extracted line is usually sorted by x,
    so its measurement direction is unknowable).

    Raises ValueError if the data cannot be a J-V curve (too few points, dark curve, ...).
    """
    v = np.asarray(v, float).ravel() * _V_FACTORS[v_unit.lower()]
    j = np.asarray(j, float).ravel() * _j_factor(j_unit)
    if v.shape != j.shape:
        raise ValueError("v and j must have the same length")
    ok = np.isfinite(v) & np.isfinite(j)
    v, j = v[ok], j[ok]
    if len(v) < 4:
        raise ValueError(f"need at least 4 finite points, got {len(v)}")
    if np.ptp(v) == 0:
        raise ValueError("all voltages are identical")

    top_notes: list[str] = []
    segs = _zigzag_segments(v)
    if len(segs) == 1:
        scans_in = [(v, j, "unspecified")]
    elif len(segs) <= 4:
        # group by direction: "0 -> Vmax, Vmax -> Vmin, Vmin -> 0" yields one forward
        # sweep (segments 1 and 3 together) and one reverse sweep (segment 2)
        scans_in = []
        for d, name in ((-1, "reverse"), (1, "forward")):
            idx = np.concatenate([np.arange(a, b) for a, b, dd in segs if dd == d] or [np.array([], int)])
            if len(idx) >= 4:
                scans_in.append((v[idx], j[idx], name))
            elif len(idx):
                top_notes.append(f"short {name} segment ({len(idx)} points) ignored")
        if len(scans_in) == 1:
            scans_in[0] = (*scans_in[0][:2], "unspecified")
    else:
        top_notes.append("V is not monotone and has many reversals: treated as unordered points of one sweep")
        scans_in = [(v, j, "unspecified")]

    scans, errors = [], []
    for sv, sj, name in scans_in:
        try:
            scans.append(_scan_metrics(sv, sj, pin, name))
        except ValueError as e:                     # one bad sweep must not hide the good one
            errors.append(f"{name} sweep: {e}")
    if not scans:
        raise ValueError("; ".join(errors))

    primary = next((s for s in scans if s["direction"] == "reverse"), scans[0])
    out = dict(primary)
    out["notes"] = list(primary["notes"]) + top_notes + errors
    out["scans"] = scans
    if len(scans) == 2:
        rev = next(s for s in scans if s["direction"] == "reverse")
        fwd = next(s for s in scans if s["direction"] == "forward")
        out["hysteresis_index"] = (rev["pce"] - fwd["pce"]) / rev["pce"] if rev["pce"] else math.nan
        out["notes"].append("hysteresis loop: top-level numbers are the reverse sweep; see 'scans'")
    return out


# --------------------------------------------------------------------------
# 3. instrument-style text export (for 性能表格处理台/opv_performance_tool.py)
# --------------------------------------------------------------------------

# Column layout copied from a real solar-simulator export (器件性能文件/2500p.txt).
_OPV_COLUMNS = [
    ("Run Id", ""), ("Device", ""), ("Curve", ""), ("Size", "[cm2]"), ("Material", ""),
    ("Date", ""), ("Time", ""), ("Temperature", "[C]"), ("Temperature Corrected", ""),
    ("Irradiance", "[W/m2]"), ("Irradiance Corrected", ""), ("Jsc", "[mA/cm2]"), ("Isc", "[mA]"),
    ("Voc", "[mV]"), ("Impp", "[mA]"), ("Vmpp", "[mV]"), ("FF", "[%]"), ("Eta", "[%]"),
    ("n1", ""), ("n2", ""), ("Rshunt", "[Ohm.cm2]"), ("Rseries", "[mOhm.cm2]"),
    ("Jo1", "[A/cm2]"), ("Jo2", "[A/cm2]"), ("Slope Voc", "[mOhm.cm2]"),
]


def _num(x: float) -> str:
    return f"{x:.10g}"


def write_opv_txt(path, metrics: dict, run_id: Union[int, str] = 1, device: str = "DUT_1", *,
                  size_cm2: Optional[float] = None, material: str = "", curve: str = "Illuminated IV (figrip)"):
    """Write a 3-line, tab-separated, CRLF file that mimics the solar-simulator export.

    Line 1 = column names, line 2 = units, line 3 = values. Columns we cannot know
    (temperature, shunt resistance ...) are left blank.  Voc and Vmpp are written in mV
    and Isc/Impp in mA (needs ``size_cm2``), exactly like the instrument does.

    ``run_id`` must not be empty: the consuming tool strips the line before splitting
    on tabs, so a blank first column would shift every value one column to the left.
    """
    for key in ("jsc", "voc", "ff", "pce"):
        if not math.isfinite(metrics.get(key, math.nan)):
            raise ValueError(f"metric {key!r} is missing/NaN; refusing to write an unreadable file")
    if str(run_id).strip() == "":
        raise ValueError("run_id must not be empty")

    pin = metrics.get("pin", 100.0)
    now = datetime.now()
    vals = {name: "" for name, _ in _OPV_COLUMNS}
    vals.update({
        "Run Id": str(run_id), "Device": device, "Curve": curve, "Material": material,
        "Date": f"{now.year}/{now.month}/{now.day}", "Time": f"{now.hour}:{now.minute:02d}",
        "Irradiance": _num(pin * 10.0),                    # mW/cm2 -> W/m2
        "Jsc": _num(metrics["jsc"]), "Voc": _num(metrics["voc"] * 1000.0),
        "FF": _num(metrics["ff"]), "Eta": _num(metrics["pce"]),
    })
    if "vmpp" in metrics:
        vals["Vmpp"] = _num(metrics["vmpp"] * 1000.0)
    if size_cm2:
        vals["Size"] = _num(size_cm2)
        vals["Isc"] = _num(metrics["jsc"] * size_cm2)
        if "jmpp" in metrics:
            vals["Impp"] = _num(metrics["jmpp"] * size_cm2)

    lines = ["\t".join(n for n, _ in _OPV_COLUMNS),
             "\t".join(u for _, u in _OPV_COLUMNS),
             "\t".join(vals[n] for n, _ in _OPV_COLUMNS),
             ""]                                            # the real files end with an empty line
    # newline="" so Python does not turn our \r\n into \r\r\n
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")
    return Path(path)


# --------------------------------------------------------------------------
# 4. EQE -> integrated Jsc
# --------------------------------------------------------------------------

_Q = 1.602176634e-19      # C
_H = 6.62607015e-34       # J s
_C = 299792458.0          # m/s
_DATA_DIR = Path(__file__).parent / "data"


def load_spectrum(path: Union[str, Path, None] = None) -> tuple[np.ndarray, np.ndarray]:
    """Reference spectrum as (wavelength_nm, irradiance W m^-2 nm^-1).

    Default = ASTM G173-03 AM1.5G (global tilt), 280-4000 nm, shipped as
    ``figrip/data/am15g.csv`` (NREL, see the file header for attribution).
    A user file must have two columns (wavelength nm, W/m2/nm), comma- or
    whitespace-separated, ``#`` comments allowed.
    """
    p = Path(path) if path is not None else _DATA_DIR / "am15g.csv"
    text = p.read_text(encoding="utf-8")
    data_lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    delim = "," if data_lines and "," in data_lines[0] else None
    arr = np.loadtxt(data_lines, delimiter=delim, usecols=(0, 1), ndmin=2)
    return arr[:, 0], arr[:, 1]


def _eqe_on_grid(wl_nm, eqe, eqe_unit, spectrum):
    wl = np.asarray(wl_nm, float).ravel()
    e = np.asarray(eqe, float).ravel()
    if wl.shape != e.shape or wl.size < 2:
        raise ValueError("wl_nm and eqe must be equal-length arrays with >= 2 points")
    ok = np.isfinite(wl) & np.isfinite(e)
    wl, e = wl[ok], e[ok]
    wl, e = _clean_sorted(wl, e, [])
    if eqe_unit == "percent":
        e = e / 100.0
    elif eqe_unit != "fraction":
        raise ValueError("eqe_unit must be 'percent' or 'fraction'")

    if spectrum is None or isinstance(spectrum, (str, Path)):
        swl, sirr = load_spectrum(spectrum)
    else:
        swl, sirr = (np.asarray(a, float) for a in spectrum)
    lo, hi = max(wl[0], swl[0]), min(wl[-1], swl[-1])
    if hi <= lo:
        raise ValueError("EQE wavelength range does not overlap the reference spectrum")
    # integrate on the (fine) spectrum grid; the (coarse) EQE is interpolated linearly onto it.
    # EQE is NOT extrapolated: outside its measured range it contributes nothing.
    g = np.unique(np.concatenate([[lo, hi], swl[(swl > lo) & (swl < hi)]]))
    eqe_g = np.interp(g, wl, e)
    irr_g = np.interp(g, swl, sirr)
    # photon flux density per nm: E * lambda / (h c);  J = q * EQE * flux
    # (W m-2 nm-1) * (m) / (J m) -> photons m-2 s-1 nm-1; * q -> A m-2 nm-1; * 0.1 -> mA cm-2 nm-1
    dj = eqe_g * irr_g * (g * 1e-9) / (_H * _C) * _Q * 0.1
    return g, dj


def eqe_integrated_jsc(wl_nm, eqe, *, eqe_unit: str = "percent", spectrum=None) -> float:
    """Jsc [mA/cm2] = q * integral EQE(l) * Phi_AM1.5G(l) dl, over the measured EQE range.

    ``spectrum`` = None (bundled ASTM G173-03), a path, or a ``(wl_nm, W_m2_nm)`` tuple.
    ``eqe`` in percent by default (``eqe_unit="fraction"`` for 0-1).
    """
    g, dj = _eqe_on_grid(wl_nm, eqe, eqe_unit, spectrum)
    return float(np.sum(0.5 * (dj[1:] + dj[:-1]) * np.diff(g)))


def eqe_cumulative_jsc(wl_nm, eqe, *, eqe_unit: str = "percent", spectrum=None) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative integrated Jsc vs wavelength: (wavelength_nm, mA/cm2) - the curve usually plotted on the right axis."""
    g, dj = _eqe_on_grid(wl_nm, eqe, eqe_unit, spectrum)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dj[1:] + dj[:-1]) * np.diff(g))])
    return g, cum
