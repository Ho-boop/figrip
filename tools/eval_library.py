"""Evaluate figrip on a whole PDF library (e.g. a Zotero storage folder).

    python -I tools/eval_library.py --library "D:/Zotero/storage/*/*.pdf" --max-pages 60 --out out/eval

What it does
  1. scan:    finds vector-heavy pages (sum of len(d["items"]) over page.get_drawings() > --min-items),
              caches the result in <out>/scan.json
  2. select:  up to --max-pages pages from up to --max-papers papers, <= --per-paper pages each,
              duplicates (same file size) skipped, pages that mention Figure / J-V / EQE / Voltage /
              Wavelength preferred
  3. run:     figrip.pipeline.extract() on every selected page in a *subprocess* with a hard timeout,
              then renders (a) a full-page PNG at 70 dpi, (b) an overlay PNG per detected panel
  4. writes   <out>/results.json   (one record per page: timing, panels, axes, calibration source, series,
                                    point counts, exceptions with traceback)
              <out>/png/           (pXXX_<paper>_pN_full.png, ..._panelK.png)
              <out>/index.md       (a one-line-per-panel table to tick off while looking at the images)
              <out>/sheets/        (with --sheets: one contact sheet per page = full page + all its overlays)

The point of the tool is the *human* step afterwards: open the PNGs, judge recall / precision / correctness of
each panel and write the failure taxonomy.  Nothing here modifies the PDFs or copies them anywhere.
Re-running with the same --out reuses scan.json (use --rescan to redo it) and the same page selection (--reselect).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LIBRARY = "D:/Zotero/storage/*/*.pdf"          # the author's Zotero library; override with --library
KEYWORDS = {
    "figure": re.compile(r"\bfig(ure|\.)", re.I),
    "jv": re.compile(r"J\s*[-\u2013\u2212]\s*V|current density", re.I),
    "eqe": re.compile(r"\bEQE\b|\bIPCE\b", re.I),
    "voltage": re.compile(r"voltage", re.I),
    "wavelength": re.compile(r"wavelength", re.I),
}


# ----------------------------------------------------------------------------- scan
def _scan_one(args):
    path, min_items, max_pages_per_pdf = args
    import pymupdf
    out = {"path": path, "size": os.path.getsize(path), "pages": [], "error": None}
    try:
        with pymupdf.open(path) as doc:
            out["n_pages"] = len(doc)
            for i in range(min(len(doc), max_pages_per_pdf)):
                page = doc[i]
                n = sum(len(d["items"]) for d in page.get_drawings())
                if n > min_items:
                    txt = page.get_text()
                    kw = [k for k, rx in KEYWORDS.items() if rx.search(txt)]
                    out["pages"].append({"page": i, "items": n, "kw": kw})
    except Exception as e:  # damaged / encrypted pdf
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def scan(library: str, min_items: int, workers: int, out_dir: Path, rescan: bool) -> list[dict]:
    cache = out_dir / "scan.json"
    if cache.exists() and not rescan:
        return json.loads(cache.read_text(encoding="utf-8"))
    pdfs = sorted(glob.glob(library))
    print(f"scanning {len(pdfs)} pdfs ...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_scan_one, [(p, min_items, 60) for p in pdfs], chunksize=4))
    cache.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    heavy = sum(1 for r in res if r["pages"])
    print(f"scan done in {time.time() - t0:.0f}s: {heavy}/{len(res)} pdfs have vector-heavy pages", flush=True)
    return res


# ----------------------------------------------------------------------------- select
def select(scanned: list[dict], max_pages: int, max_papers: int, per_paper: int, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    seen_sizes, cands = set(), []
    for r in scanned:
        if not r["pages"] or r["size"] in seen_sizes:
            continue
        seen_sizes.add(r["size"])
        pages = sorted(r["pages"], key=lambda p: (-len(p["kw"]), -p["items"]))[:per_paper]
        cands.append((max(len(p["kw"]) for p in pages), rnd.random(), r, pages))
    # papers whose best page hits the most keywords first, random within ties
    cands.sort(key=lambda c: (-c[0], c[1]))
    picked: list[dict] = []
    chosen = cands[:max_papers]
    # first pass: the best page of every chosen paper (so we get max_papers different papers) ...
    for rank in range(per_paper):
        # ... then fill with second-best pages while the page budget lasts
        for _, _, r, pages in chosen:
            if rank < len(pages) and len(picked) < max_pages:
                p = pages[rank]
                picked.append({"pdf": r["path"], "page": p["page"], "items": p["items"], "kw": p["kw"]})
    picked.sort(key=lambda p: (p["pdf"], p["page"]))
    return picked


# ----------------------------------------------------------------------------- worker (one page, own process)
def _worker(pdf: str, page: int, out_pkl: str) -> None:
    sys.path.insert(0, str(REPO))
    sys.stdout.reconfigure(encoding="utf-8")
    rec: dict = {"ok": False}
    t0 = time.time()
    try:
        from figrip.pipeline import extract
        results = extract(pdf, pages=[page])
        rec.update(ok=True, results=results)
    except BaseException as e:  # noqa: BLE001 - we want every failure recorded
        rec.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec["t_extract"] = time.time() - t0
    with open(out_pkl, "wb") as f:
        pickle.dump(rec, f)


def _series_summary(s) -> dict:
    return {"name": s.name, "kind": s.kind, "n": int(len(s.page_pts)),
            "color": [round(float(c), 2) for c in s.style.get("color", ())] if s.style.get("color") else None,
            "style": {k: (v if isinstance(v, (str, int, float)) or v is None else str(v)) for k, v in s.style.items()
                      if k != "color"}}


def _panel_summary(fr, idx: int) -> dict:
    p = fr.panel
    xs = [float(v) for v in p.x.to_data([p.bbox[0], p.bbox[2]])]
    ys = [float(v) for v in p.y.to_data([p.bbox[3], p.bbox[1]])]
    return {"idx": idx, "label": p.label, "bbox": [round(float(v), 1) for v in p.bbox],
            "x": {"range": xs, "scale": p.x.scale, "source": p.x.source, "label": p.x.label,
                  "n_ticks": len(p.x.ticks), "residual": round(p.x.residual, 3),
                  "ticks": [[round(float(a), 1), float(b)] for a, b in p.x.ticks]},
            "y": {"range": ys, "scale": p.y.scale, "source": p.y.source, "label": p.y.label,
                  "n_ticks": len(p.y.ticks), "residual": round(p.y.residual, 3),
                  "ticks": [[round(float(a), 1), float(b)] for a, b in p.y.ticks]},
            "notes": list(p.notes),
            "n_series": len(fr.series),
            "series": [_series_summary(s) for s in fr.series]}


# ----------------------------------------------------------------------------- run
def run(picked: list[dict], out_dir: Path, timeout: float, dpi_full: int, overlay_dpi: int) -> list[dict]:
    png_dir = out_dir / "png"
    tmp_dir = out_dir / "tmp"
    png_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))
    import pymupdf
    from figrip.preview import render_overlay

    records = []
    for n, item in enumerate(picked):
        pdf, pno = item["pdf"], item["page"]
        stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(pdf).stem)[:40]
        tag = f"p{n:03d}_{stem}_pg{pno + 1}"
        pkl = tmp_dir / f"{tag}.pkl"
        rec = {"tag": tag, "pdf": pdf, "page": pno + 1, "items": item.get("items"), "kw": item.get("kw")}
        t0 = time.time()
        try:
            subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--worker", pdf, str(pno), str(pkl)],
                           timeout=timeout, capture_output=True)
            rec["wall"] = round(time.time() - t0, 2)
            if not pkl.exists():
                rec["status"] = "crash"
                rec["error"] = "worker died without result (hard crash)"
            else:
                data = pickle.load(open(pkl, "rb"))
                rec["t_extract"] = round(data["t_extract"], 2)
                if data["ok"]:
                    rec["status"] = "ok"
                else:
                    rec["status"] = "exception"
                    rec["error"] = data["error"]
                    rec["traceback"] = data["traceback"]
        except subprocess.TimeoutExpired:
            rec["wall"] = round(time.time() - t0, 2)
            rec["status"] = "timeout"
            rec["error"] = f"> {timeout:.0f}s"
        # full-page render (so that recall can be judged by eye)
        try:
            with pymupdf.open(pdf) as doc:
                doc[pno].get_pixmap(dpi=dpi_full, alpha=False).save(str(png_dir / f"{tag}_full.png"))
        except Exception as e:  # noqa: BLE001
            rec["full_render_error"] = str(e)
        rec["panels"] = []
        if rec["status"] == "ok":
            results = data["results"]
            for k, fr in enumerate(results, 1):
                ps = _panel_summary(fr, k)
                try:
                    render_overlay(pdf, fr, png_dir / f"{tag}_panel{k}.png", dpi=overlay_dpi)
                    ps["png"] = f"{tag}_panel{k}.png"
                except Exception as e:  # noqa: BLE001
                    ps["render_error"] = f"{type(e).__name__}: {e}"
                rec["panels"].append(ps)
        records.append(rec)
        print(f"[{n + 1}/{len(picked)}] {tag}: {rec['status']} "
              f"{rec.get('t_extract', rec['wall'])}s panels={len(rec['panels'])}", flush=True)
        (out_dir / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return records


def write_index(records: list[dict], out_dir: Path) -> None:
    lines = ["| tag | status | s | panel | x range (src) | y range (src) | series | png |", "|---|---|---|---|---|---|---|---|"]
    for r in records:
        if not r["panels"]:
            lines.append(f"| {r['tag']} | {r['status']} | {r.get('t_extract', r['wall'])} | - | | | | {r['tag']}_full.png |")
        for p in r["panels"]:
            fmt = lambda a: f"[{a['range'][0]:.4g}..{a['range'][1]:.4g}] {a['scale']} ({a['source']})"  # noqa: E731
            lines.append(f"| {r['tag']} | {r['status']} | {r.get('t_extract', r['wall'])} | {p['idx']} {p['label'] or ''} | "
                         f"{fmt(p['x'])} | {fmt(p['y'])} | {p['n_series']} | {p.get('png', '')} |")
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def write_sheets(records: list[dict], out_dir: Path) -> None:
    """One contact sheet per page (full page left, panel overlays right) - this is what you look at to grade."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed: skipping --sheets (the PNGs in png/ are still there)")
        return
    sd = out_dir / "sheets"
    sd.mkdir(exist_ok=True)
    for r in records:
        full = Image.open(out_dir / "png" / f"{r['tag']}_full.png").convert("RGB")
        H = 900
        full = full.resize((int(full.width * H / full.height), H))
        tiles = []
        for i, p in enumerate((p for p in r["panels"] if "png" in p), 1):
            im = Image.open(out_dir / "png" / p["png"]).convert("RGB")
            im = im.resize((640, int(im.height * 640 / im.width)))
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 40, 18), fill=(255, 255, 0))
            d.text((4, 3), f"#{i}", fill=(0, 0, 0))
            tiles.append(im)
        cols = 2 if len(tiles) > 1 else 1
        rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
        rh = [max(t.height for t in row) for row in rows]
        sheet = Image.new("RGB", (full.width + cols * 640 + 10, max(H, sum(rh))), (255, 255, 255))
        sheet.paste(full, (0, 0))
        y = 0
        for row, h in zip(rows, rh):
            for j, t in enumerate(row):
                sheet.paste(t, (full.width + 10 + j * 640, y))
            y += h
        sheet.save(sd / f"{r['tag']}.jpg", quality=85)


def headline(records: list[dict]) -> None:
    times = [r["t_extract"] for r in records if "t_extract" in r]
    print("\n== machine-side summary (correctness needs your eyes) ==")
    print(f"pages {len(records)}, papers {len({r['pdf'] for r in records})}, "
          f"panels found {sum(len(r['panels']) for r in records)}")
    print(f"median time/page {statistics.median(times):.1f}s, max {max(times):.1f}s" if times else "no timings")
    for st in ("exception", "timeout", "crash"):
        bad = [r['tag'] for r in records if r["status"] == st]
        print(f"{st}: {len(bad)} {bad[:5]}")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=DEFAULT_LIBRARY, help="glob of PDFs (default: the author's Zotero storage)")
    ap.add_argument("--out", default=str(REPO / "out" / "eval"), help="output dir (keep it gitignored)")
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--max-papers", type=int, default=40)
    ap.add_argument("--per-paper", type=int, default=2)
    ap.add_argument("--min-items", type=int, default=800, help="'vector-heavy' threshold (sum of drawing items)")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-page extraction timeout, seconds")
    ap.add_argument("--workers", type=int, default=8, help="processes for the scan phase")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi-full", type=int, default=70)
    ap.add_argument("--overlay-dpi", type=int, default=130)
    ap.add_argument("--sheets", action="store_true", help="also write sheets/<tag>.jpg contact sheets (needs Pillow)")
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--reselect", action="store_true", help="ignore the stored selection.json")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_file = out_dir / "selection.json"
    if sel_file.exists() and not a.reselect:
        picked = json.loads(sel_file.read_text(encoding="utf-8"))
    else:
        scanned = scan(a.library, a.min_items, a.workers, out_dir, a.rescan)
        picked = select(scanned, a.max_pages, a.max_papers, a.per_paper, a.seed)
        sel_file.write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"selected {len(picked)} pages from {len({p['pdf'] for p in picked})} papers", flush=True)
    records = run(picked, out_dir, a.timeout, a.dpi_full, a.overlay_dpi)
    write_index(records, out_dir)
    if a.sheets:
        write_sheets(records, out_dir)
    headline(records)


if __name__ == "__main__":
    main()
