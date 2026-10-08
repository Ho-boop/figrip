# figrip architecture & team contract

Pipeline (one PDF page → many plots):

```
primitives.extract_page(page)            -> PageContent      (paths + text spans)   [leader, done]
axes.find_panels(pc, page=None)          -> list[Panel]      (plot area + calibrated x/y)
series.extract_series(pc, panel)         -> list[Series]     (page_pts filled; data filled by pipeline)
pipeline.extract(pdf, pages=None, ...)   -> list[FigureResult]                        [leader]
opv.jv_metrics(v, j, pin=100.0)          -> dict(jsc, voc, ff, pce, ...)
opv.write_opv_txt(path, metrics)         -> file readable by 性能表格处理台/opv_performance_tool.py
export.to_csv / to_json                  -> files
preview.render_overlay(pdf, result, png) -> PNG: page crop + extracted points drawn on top
cli.main()                               -> `figrip scan|extract|jv`
```

All types live in `figrip/model.py`. **Do not change model.py without telling the leader**;
if you need an extra field, add it with a default value and report it.

## Coordinate convention
Page coords = PyMuPDF page space, points, origin top-left, **y grows down**.
`AxisCalib.to_data(pos)` maps a page coordinate to a data value (linear or log10).

## File ownership (parallel work — only edit your own files)
| file(s) | owner |
|---|---|
| model.py, primitives.py, pipeline.py, `__init__`.py | leader |
| axes.py, ocr.py, tests/test_axes.py | axes agent |
| series.py, tests/test_series.py | series agent |
| opv.py, export.py, preview.py, cli.py, tests/test_opv.py, tests/test_cli.py, README.md | app agent |
| tests/corpus/** , tests/test_corpus_e2e.py | corpus agent |

## Real-world facts already observed (Zotero library, 173/376 OPV papers have vector-heavy pages)
`D:/Zotero/storage/QIZR5ARD/Zheng_et_al_2018_Adv_Mater_Fine_tuned_HTL_high_FF_OSCs.pdf`, page index 3
(Figure 2d/2f are J–V curves, Table 2 on the same page gives Jsc/Voc/FF/PCE = real ground truth):
- **Every** drawing on the page is a *fill* (only 25 strokes). Curves, axes and ticks are likely
  "outlined strokes" (thin filled polygons), not stroked lines.
- Tick labels are **not text** — glyphs are outlined into filled paths; `get_text` returns no tick
  numbers. Calibration must fall back to OCR of the rendered label area (or manual values).
- y axis is J with negative values upward (−20 at top).
Real PDFs are the user's own library: never commit them, never copy them into the repo;
tests that use them must `pytest.skip` when the file is missing.

Other vector-heavy OPV papers worth trying: `D:/Zotero/storage/QEZGCYHQ/Yuan2019Y6Over15.pdf`,
`D:/Zotero/storage/ZJRX9AAG/Li2012PolymerSolarCells.pdf`, and anything found with a quick
`get_drawings()` count over `D:/Zotero/storage/*/*.pdf`.

## Environment
Windows, Python 3.14, run `python -m pytest` from repo root. PyMuPDF 1.28, numpy, matplotlib, pytest installed.
Print with `sys.stdout.reconfigure(encoding="utf-8")` — the console is GBK.
Do not git commit; the leader commits.
