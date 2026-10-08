"""Score extraction results against the synthetic-corpus ground truth.

``score(case_json, results, tol=0.005) -> dict``

* ``case_json``: ground-truth dict, or a path to a ``<case>.json``.
* ``results``: ``list[figrip.model.FigureResult]`` (duck-typed: ``r.panel.bbox`` and
  ``r.series[i].{name,kind,data}`` are all that is used).

All errors are fractions of the axis range *in axis space* (log10 for log axes), so
0.005 means "0.5 % of the axis range".

Panel matching: greedy by bbox IoU (>= ``iou_min``).
Series matching (inside a matched panel): first by equal name (case/space-insensitive) when both
sides have one, then greedily by closeness of the data.

Per-series metrics
  err_pts    max over truth points (markers only: ``marker_idx`` if given) of the
             max-norm distance to the nearest extracted point  ("an extracted point within
             tol in both axes").  Always computed.
  err_curve  max over all truth points of the distance to the extracted polyline (extracted
             points sorted by x).  Equals the vertical error where the curve is shallow but
             stays well conditioned on near-vertical parts (e.g. J-V curves at Voc).  Only used
             when the truth kind contains "line" and the extracted series is a line.
  err_y_at_x same idea as the spec's "resampled y error" (|y_truth - interp(x_truth)|),
             reported for information only.
A series is ``ok`` when: truth "line"  -> err_curve <= tol and found has a line;
                         truth "markers" -> err_pts <= tol;
                         truth "line+markers" -> err_pts <= tol, and err_curve <= tol if the
                         extracted series is a line.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

INF = math.inf


def load_case(case_json) -> dict:
    if isinstance(case_json, (str, Path)):
        return json.loads(Path(case_json).read_text(encoding="utf-8"))
    return case_json


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _axis(v, scale):
    v = np.asarray(v, float)
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
    return v


def _norm(pts, panel):
    """(N,2) data -> normalised axis space ([0,1] over the truth axis limits); drops non-finite rows."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    u = _axis(pts[:, 0], panel["xscale"])
    v = _axis(pts[:, 1], panel["yscale"])
    xl = _axis(panel["xlim"], panel["xscale"])
    yl = _axis(panel["ylim"], panel["yscale"])
    out = np.column_stack([(u - xl[0]) / (xl[1] - xl[0]), (v - yl[0]) / (yl[1] - yl[0])])
    return out[np.isfinite(out).all(axis=1)]


def _cheb_nearest(T, F):
    """For every row of T the max-norm distance to the closest row of F."""
    if len(F) == 0:
        return np.full(len(T), INF)
    d = np.maximum(np.abs(T[:, None, 0] - F[None, :, 0]), np.abs(T[:, None, 1] - F[None, :, 1]))
    return d.min(axis=1)


def _polyline_dist(T, F):
    """L2 distance from each row of T to the polyline through F (rows of F in order)."""
    if len(F) == 0:
        return np.full(len(T), INF)
    if len(F) == 1:
        return np.hypot(*(T - F[0]).T)
    A, B = F[:-1], F[1:]
    AB = B - A
    L2 = (AB ** 2).sum(axis=1)
    L2 = np.where(L2 == 0, 1e-30, L2)
    best = np.full(len(T), INF)
    for k in range(0, len(T), 256):  # chunk to bound memory
        t = T[k:k + 256]
        w = t[:, None, :] - A[None, :, :]
        s = np.clip((w * AB[None]).sum(axis=2) / L2[None], 0, 1)
        proj = A[None] + s[..., None] * AB[None]
        best[k:k + 256] = np.hypot(*(t[:, None, :] - proj).transpose(2, 0, 1)).min(axis=1)
    return best


def _y_at_x(T, F):
    if len(F) < 2:
        return INF
    o = np.argsort(F[:, 0], kind="stable")
    Fx, Fy = F[o, 0], F[o, 1]
    m = (T[:, 0] >= Fx[0]) & (T[:, 0] <= Fx[-1])
    if not m.any():
        return INF
    return float(np.max(np.abs(np.interp(T[m, 0], Fx, Fy) - T[m, 1])))


def _truth_pts(ts, panel):
    return _norm(np.column_stack([ts["x"], ts["y"]]), panel)


def _marker_truth(ts, panel):
    x, y = np.asarray(ts["x"], float), np.asarray(ts["y"], float)
    if ts["kind"] == "line+markers" and ts.get("marker_idx") is not None:
        idx = np.asarray(ts["marker_idx"], int)
        x, y = x[idx], y[idx]
    return _norm(np.column_stack([x, y]), panel)


def _found_pts(fs, panel):
    return _norm(np.asarray(fs.data, float).reshape(-1, 2), panel)


def _is_line(kind: str) -> bool:
    return "line" in (kind or "")


def _pair_cost(ts, fs, panel):
    T = _truth_pts(ts, panel)
    F = _found_pts(fs, panel)
    if len(T) == 0 or len(F) == 0:
        return INF
    if _is_line(fs.kind):
        F = F[np.argsort(F[:, 0], kind="stable")]
        d = _polyline_dist(T, F)
    else:
        d = _cheb_nearest(T, F)
    return float(np.median(d))


def _norm_name(s):
    return "".join((s or "").split()).casefold() or None


def _score_series(ts, fs, panel, tol):
    out = {"truth_name": ts.get("name"), "truth_kind": ts["kind"], "n_truth": len(ts["x"]),
           "found_name": getattr(fs, "name", None), "found_kind": getattr(fs, "kind", None),
           "n_found": 0, "err_pts": INF, "err_curve": INF, "err_y_at_x": INF, "ok": False}
    F = _found_pts(fs, panel)
    out["n_found"] = int(len(F))
    if len(F) == 0:
        return out
    out["err_pts"] = float(_cheb_nearest(_marker_truth(ts, panel), F).max())
    Fl = F[np.argsort(F[:, 0], kind="stable")]
    T = _truth_pts(ts, panel)
    out["err_y_at_x"] = _y_at_x(T, Fl)
    if _is_line(fs.kind):
        out["err_curve"] = float(_polyline_dist(T, Fl).max())
    k = ts["kind"]
    if k == "line":
        out["ok"] = _is_line(fs.kind) and out["err_curve"] <= tol
    elif k == "markers":
        out["ok"] = out["err_pts"] <= tol
    else:  # line+markers
        out["ok"] = out["err_pts"] <= tol and (not _is_line(fs.kind) or out["err_curve"] <= tol)
    return out


def _match_series(tseries, fseries, panel):
    pairs, used_t, used_f = [], set(), set()
    for i, ts in enumerate(tseries):
        nt = _norm_name(ts.get("name"))
        if nt is None:
            continue
        for j, fs in enumerate(fseries):
            if j not in used_f and _norm_name(getattr(fs, "name", None)) == nt:
                pairs.append((i, j)); used_t.add(i); used_f.add(j)
                break
    cands = sorted((_pair_cost(ts, fs, panel), i, j)
                   for i, ts in enumerate(tseries) if i not in used_t
                   for j, fs in enumerate(fseries) if j not in used_f)
    for c, i, j in cands:
        if c <= 0.05 and i not in used_t and j not in used_f:
            pairs.append((i, j)); used_t.add(i); used_f.add(j)
    return pairs


def score(case_json, results, tol: float = 0.005, iou_min: float = 0.3) -> dict:
    case = load_case(case_json)
    tpanels = case["panels"]
    # ---- panel matching
    cand = sorted(((_iou(tp["bbox_hint"], tuple(r.panel.bbox)), i, j)
                   for i, tp in enumerate(tpanels) for j, r in enumerate(results)), reverse=True)
    used_t, used_f, pairs = set(), set(), {}
    for iou, i, j in cand:
        if iou >= iou_min and i not in used_t and j not in used_f:
            pairs[i] = (j, iou); used_t.add(i); used_f.add(j)
    panels_out = []
    for i, tp in enumerate(tpanels):
        entry = {"truth_idx": i, "found_idx": None, "iou": 0.0, "series": [], "extra_series": 0,
                 "n_truth_series": len(tp["series"]), "ok": False}
        if i in pairs:
            j, iou = pairs[i]
            res = results[j]
            entry.update(found_idx=j, iou=float(iou), n_found_series=len(res.series))
            mp = dict(_match_series(tp["series"], res.series, tp))
            for si, ts in enumerate(tp["series"]):
                if si in mp:
                    s = _score_series(ts, res.series[mp[si]], tp, tol)
                    s["found_idx"] = mp[si]
                else:
                    s = {"truth_name": ts.get("name"), "truth_kind": ts["kind"], "found_idx": None,
                         "found_name": None, "found_kind": None, "n_truth": len(ts["x"]), "n_found": 0,
                         "err_pts": INF, "err_curve": INF, "err_y_at_x": INF, "ok": False}
                s["truth_idx"] = si
                entry["series"].append(s)
            entry["extra_series"] = len(res.series) - len(mp)
            entry["ok"] = all(s["ok"] for s in entry["series"])
        panels_out.append(entry)
    worst = max([min(s["err_pts"], s["err_curve"]) if s["truth_kind"] != "line" else s["err_curve"]
                 for p in panels_out for s in p["series"]] or [0.0])
    count_ok = len(tpanels) == len(results)
    return {
        "case": case["case"], "tol": tol,
        "n_panels_truth": len(tpanels), "n_panels_found": len(results), "panel_count_ok": count_ok,
        "panels": panels_out,
        "unmatched_found_panels": [j for j in range(len(results)) if j not in used_f],
        "max_err": worst,
        "ok": count_ok and all(p["ok"] for p in panels_out),
    }


def format_report(sc: dict) -> str:
    """Human-readable multi-line summary of a ``score`` dict (for assertion messages / batch reports)."""
    lines = [f"[{sc['case']}] panels truth={sc['n_panels_truth']} found={sc['n_panels_found']} "
             f"-> {'OK' if sc['ok'] else 'FAIL'} (tol {sc['tol']:.3%})"]
    for p in sc["panels"]:
        lines.append(f"  panel {p['truth_idx']}: matched={p['found_idx']} iou={p['iou']:.2f} "
                     f"extra_series={p['extra_series']}")
        for s in p["series"]:
            lines.append(f"    {s['truth_name']!s:>14} {s['truth_kind']:<13} -> {s['found_name']!s:<12} "
                         f"{s['found_kind']!s:<13} n={s['n_found']:>4}/{s['n_truth']:<4} "
                         f"err_pts={s['err_pts']:.4f} err_curve={s['err_curve']:.4f} "
                         f"y@x={s['err_y_at_x']:.4f} {'ok' if s['ok'] else 'FAIL'}")
    return "\n".join(lines)
