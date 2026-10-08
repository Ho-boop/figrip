"""Tests for figrip.opv: J-V metrics against a single-diode model, EQE integration, instrument txt."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from figrip import opv

TOOL_DIR = Path(r"D:\实习项目\性能表格处理台")

# ---------------------------------------------------------------------------
# single-diode model with series resistance, parametrised by J so V is explicit:
#   J = Jph - J0 (exp((V + J Rs)/(n Vt)) - 1)    (Rsh = infinity)
#   =>  V = n Vt ln((Jph - J)/J0 + 1) - J Rs
# ---------------------------------------------------------------------------
JPH, J0, N, VT, RS = 20.0, 1e-9, 1.5, 0.025852, 2.0     # mA/cm2, mA/cm2, -, V, Ohm cm2


def diode_curve(n=2_000_001):
    """dense (v, jp) of the model, parametrised by eps = Jph - jp (log-spaced: the
    flat photocurrent plateau at V<0 needs eps down to 1e-12).  Then V = f(eps) is explicit."""
    eps = np.logspace(-12, math.log10(JPH + 2.0), n)        # jp from ~Jph down to -2
    jp = JPH - eps
    v = N * VT * np.log(eps / J0 + 1.0) - jp * 1e-3 * RS
    return v, jp


def truth():
    """'known' values from a very dense numerical evaluation of the model"""
    v, jp = diode_curve()
    p = v * jp
    p[(v < 0) | (jp < 0)] = -1                              # power quadrant only
    k = np.argmax(p)
    voc = float(np.interp(0.0, jp[::-1], v[::-1]))         # J = 0
    jsc = float(np.interp(0.0, v, jp))
    return dict(jsc=jsc, voc=voc, vmpp=float(v[k]), jmpp=float(jp[k]), pmax=float(p[k]),
                ff=100 * p[k] / (jsc * voc), pce=float(p[k]))   # pin=100 -> PCE = Pmax in %


T = truth()


def sample_curve(n=60, vmin=-0.2, vmax=None):
    """sparse samples on a V grid (as a digitised curve would be)"""
    vmax = vmax or T["voc"] + 0.05
    vs = np.linspace(vmin, vmax, n)
    v_grid, jp_grid = diode_curve()
    return vs, np.interp(vs, v_grid, jp_grid)               # v_grid is increasing


def check(m, tol_jsc=0.002, tol_voc=2e-3, tol_ff=0.3, tol_pce=0.05):
    assert m["jsc"] == pytest.approx(T["jsc"], rel=tol_jsc)
    assert m["voc"] == pytest.approx(T["voc"], abs=tol_voc)
    assert m["ff"] == pytest.approx(T["ff"], abs=tol_ff)
    assert m["pce"] == pytest.approx(T["pce"], abs=tol_pce)
    assert m["vmpp"] == pytest.approx(T["vmpp"], abs=0.02)


def test_truth_is_sane():
    # sanity of the reference itself (typical OPV-ish numbers)
    assert 19 < T["jsc"] < 20.1 and 0.85 < T["voc"] < 0.95 and 60 < T["ff"] < 85


@pytest.mark.parametrize("sign", [-1, +1])
def test_sign_conventions(sign):
    v, jp = sample_curve()
    m = opv.jv_metrics(v, sign * jp)               # -1: 4th quadrant, +1: photocurrent positive
    check(m)
    assert m["jsc"] > 0 and m["voc"] > 0 and m["jmpp"] > 0 and m["pmax"] > 0
    assert m["j_sign"] == sign


def test_dense_curve_is_accurate():
    # vector curves are dense: with 1500 points linear interpolation is essentially exact
    v, jp = sample_curve(1500)
    m = opv.jv_metrics(v, -jp)
    check(m, tol_jsc=1e-5, tol_voc=1e-4, tol_ff=0.01, tol_pce=0.005)
    assert m["pmax"] == pytest.approx(T["pmax"], rel=1e-4)


def test_unsorted_and_duplicated_voltage():
    v, jp = sample_curve(80)
    v = np.concatenate([v, v[10:15]])              # duplicated V
    jp = np.concatenate([jp, jp[10:15]])
    perm = np.random.default_rng(0).permutation(len(v))
    m = opv.jv_metrics(v[perm], -jp[perm])
    check(m)
    assert any("duplicated" in n for n in m["notes"])
    assert len(m["scans"]) == 1                    # shuffling must NOT be mistaken for a loop


def test_units_and_pin():
    v, jp = sample_curve()
    m1 = opv.jv_metrics(v, -jp)
    m2 = opv.jv_metrics(v * 1000, -jp * 10, j_unit="A/m2", v_unit="mV")   # 1 mA/cm2 = 10 A/m2
    for k in ("jsc", "voc", "ff", "pce"):
        assert m2[k] == pytest.approx(m1[k], rel=1e-9)
    # 50 mW/cm2 illumination: same Pmax, double the efficiency
    m3 = opv.jv_metrics(v, -jp, pin=50.0)
    assert m3["pce"] == pytest.approx(2 * m1["pce"], rel=1e-9)
    assert opv.guess_j_unit("Current density (mA cm$^{-2}$)") == "mA/cm2"
    assert opv.guess_j_unit("J (A m-2)") == "A/m2"
    assert opv.guess_j_unit("Current density (A/cm2)") == "A/cm2"
    assert opv.guess_v_unit("Voltage (mV)") == "mV" and opv.guess_v_unit("Voltage (V)") == "V"


def test_hysteresis_loop_reports_both_sweeps():
    v, jp = sample_curve(50)
    # "forward" sweep = a slightly worse device (lower Voc), "reverse" = the model curve
    jp_f = jp * 0.97
    v_f = v * 0.98
    # reverse: V decreasing from high to low, forward: low to high, all in ONE series
    vv = np.concatenate([v[::-1], v_f])
    jj = np.concatenate([-jp[::-1], -jp_f])
    m = opv.jv_metrics(vv, jj)
    assert len(m["scans"]) == 2
    rev = next(s for s in m["scans"] if s["direction"] == "reverse")
    fwd = next(s for s in m["scans"] if s["direction"] == "forward")
    assert rev["voc"] == pytest.approx(T["voc"], abs=1e-3)
    assert fwd["voc"] == pytest.approx(T["voc"] * 0.98, abs=2e-3)
    assert fwd["jsc"] == pytest.approx(T["jsc"] * 0.97, rel=2e-3)
    assert m["pce"] == pytest.approx(rev["pce"])        # top level = reverse sweep
    assert m["hysteresis_index"] > 0


def test_three_segment_loop_is_regrouped():
    # 0 -> Vmax -> Vmin -> 0 : two forward pieces must be merged into one forward sweep
    v, jp = sample_curve(60, vmin=-0.2)
    i0 = int(np.argmin(np.abs(v)))
    order = list(range(i0, len(v))) + list(range(len(v) - 1, -1, -1)) + list(range(0, i0 + 1))
    m = opv.jv_metrics(v[order], -jp[order])
    assert {s["direction"] for s in m["scans"]} == {"forward", "reverse"}
    for s in m["scans"]:
        assert s["voc"] == pytest.approx(T["voc"], abs=2e-3)


def test_curve_that_stops_before_voc_is_extrapolated_and_flagged():
    v, jp = sample_curve(60)
    keep = jp > 0.15 * JPH                          # cut the curve when J is still 15 % of Jsc
    m = opv.jv_metrics(v[keep], -jp[keep])
    assert any("extrapolated" in n for n in m["notes"])
    assert m["voc"] > v[keep][-1]
    assert m["voc"] == pytest.approx(T["voc"], rel=0.06)      # linear tail: coarse but sane


def test_curve_not_crossing_v0_is_extrapolated_and_flagged():
    v, jp = sample_curve(120, vmin=-0.2)
    keep = v >= 0.04                                  # starts at 0.04 V, never touches V=0
    m = opv.jv_metrics(v[keep], jp[keep])
    assert any("Jsc extrapolated" in n for n in m["notes"])
    assert m["jsc"] == pytest.approx(T["jsc"], rel=0.01)


def test_dark_curve_rejected():
    v = np.linspace(-0.5, 1.0, 40)
    j = 1e-6 * (np.exp(v / 0.04) - 1) + 1e-4 * v       # a dark diode: ~0 at V=0 and rising
    with pytest.raises(ValueError):
        opv.jv_metrics(v, j)


def test_too_few_points():
    with pytest.raises(ValueError):
        opv.jv_metrics([0, 1], [-1, 0])


def test_mpp_is_off_grid_when_samples_are_sparse():
    # 14 samples only: the true MPP lies between samples, dense interpolation should still land close
    v, jp = sample_curve(14, vmin=-0.05, vmax=T["voc"] + 0.02)
    m = opv.jv_metrics(v, -jp)
    assert m["pmax"] == pytest.approx(T["pmax"], rel=0.02)
    assert m["vmpp"] not in set(np.round(v, 12))      # not just a sampled point


# ------------------------------------------------------------------ EQE


def test_am15g_total_irradiance():
    wl, e = opv.load_spectrum()
    assert wl[0] == 280 and wl[-1] == 4000
    assert np.trapezoid(e, wl) == pytest.approx(1000.4, abs=1.0)   # ASTM G173: 1000.4 W/m2


def test_flat_eqe_integration_matches_independent_calc():
    wl, e = opv.load_spectrum()
    q, h, c = 1.602176634e-19, 6.62607015e-34, 299792458.0

    def ref(lo, hi):
        m = (wl >= lo) & (wl <= hi)
        return np.trapezoid(e[m] * wl[m] * 1e-9 / (h * c) * q, wl[m]) * 0.1

    full = opv.eqe_integrated_jsc([280, 4000], [100, 100])
    assert full == pytest.approx(ref(280, 4000), rel=1e-6)
    assert 68.5 < full < 69.5                                   # known: AM1.5G photon flux ~ 69 mA/cm2
    vis = opv.eqe_integrated_jsc([300, 900], [100, 100])
    assert vis == pytest.approx(ref(300, 900), rel=1e-6)
    assert 33 < vis < 34.5
    # 50 % flat EQE -> half; fraction input equivalent
    assert opv.eqe_integrated_jsc([300, 900], [50, 50]) == pytest.approx(vis / 2, rel=1e-9)
    assert opv.eqe_integrated_jsc([300, 900], [1, 1], eqe_unit="fraction") == pytest.approx(vis, rel=1e-9)


def test_eqe_coarse_sampling_and_cumulative():
    wl = np.arange(300, 901, 10.0)
    eqe = 80 * np.exp(-0.5 * ((wl - 600) / 120) ** 2)       # gaussian-ish EQE
    total = opv.eqe_integrated_jsc(wl, eqe)
    w, cum = opv.eqe_cumulative_jsc(wl, eqe)
    assert cum[0] == 0 and cum[-1] == pytest.approx(total) and np.all(np.diff(cum) >= 0)
    fine = np.arange(300, 901, 1.0)
    total_fine = opv.eqe_integrated_jsc(fine, np.interp(fine, wl, eqe))
    assert total == pytest.approx(total_fine, rel=1e-9)     # same linear EQE, grid-independent


def test_custom_spectrum_tuple():
    # monochromatic-ish check: 1 W/m2/nm flat between 500-600 nm, EQE 100 %
    wl = np.array([500.0, 600.0])
    val = opv.eqe_integrated_jsc([500, 600], [100, 100], spectrum=(wl, np.array([1.0, 1.0])))
    q, h, c = 1.602176634e-19, 6.62607015e-34, 299792458.0
    expected = q / (h * c) * 1e-9 * 0.5 * (500 + 600) * 100 * 0.1     # integral of lambda d lambda
    assert val == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------------ instrument txt


def test_write_opv_txt_format(tmp_path):
    m = opv.jv_metrics(*sample_curve())
    p = opv.write_opv_txt(tmp_path / "x.txt", m, run_id=7, device="PM6:Y6", size_cm2=0.04)
    raw = p.read_bytes()
    assert raw.count(b"\r\n") == 4 and b"\n" not in raw.replace(b"\r\n", b"")
    lines = raw.decode().split("\r\n")
    h, u, vals = (ln.split("\t") for ln in lines[:3])
    assert len(h) == len(u) == len(vals) == 25
    d = dict(zip(h, vals))
    assert d["Run Id"] == "7" and d["Device"] == "PM6:Y6" and d["Irradiance"] == "1000"
    assert float(d["Voc"]) == pytest.approx(m["voc"] * 1000)
    assert float(d["Isc"]) == pytest.approx(m["jsc"] * 0.04)


def test_write_opv_txt_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        opv.write_opv_txt(tmp_path / "x.txt", dict(jsc=1, voc=math.nan, ff=1, pce=1))


def test_roundtrip_through_existing_tool(tmp_path):
    """Our file must be readable by the user's own OPVDataProcessor.parse_file."""
    if not (TOOL_DIR / "opv_performance_tool.py").exists():
        pytest.skip("opv_performance_tool.py not found")
    sys.path.insert(0, str(TOOL_DIR))
    try:
        try:
            from opv_performance_tool import OPVDataProcessor   # imports tkinter
        except ImportError as e:
            pytest.skip(f"cannot import tool: {e}")
    finally:
        sys.path.remove(str(TOOL_DIR))
    m = opv.jv_metrics(*sample_curve())
    p = opv.write_opv_txt(tmp_path / "PM6-Y6.txt", m)          # default run_id / device
    got = OPVDataProcessor.parse_file(str(p))
    assert got["filename"] == "PM6-Y6"
    assert got["jsc"] == pytest.approx(round(m["jsc"], 2))
    assert got["voc"] == pytest.approx(round(m["voc"], 3))     # tool converts mV -> V
    assert got["ff"] == pytest.approx(round(m["ff"], 2))
    assert got["pce"] == pytest.approx(round(m["pce"], 2))


def test_against_corpus_ground_truth():
    """The corpus agent's synthetic J-V curves carry independently computed jsc/voc/ff/pce."""
    import json
    gt = Path(__file__).parent / "corpus" / "out" / "jv_opv.json"
    if not gt.exists():
        pytest.skip("corpus not generated")
    for pan in json.loads(gt.read_text(encoding="utf-8"))["panels"]:
        for s in pan["series"]:
            if "voc" not in s:
                continue
            m = opv.jv_metrics(s["x"], s["y"])
            assert m["jsc"] == pytest.approx(s["jsc"], rel=2e-3)
            assert m["voc"] == pytest.approx(s["voc"], abs=2e-3)
            assert m["ff"] == pytest.approx(100 * s["ff"], abs=0.3)
            assert m["pce"] == pytest.approx(s["pce"], abs=0.05)
