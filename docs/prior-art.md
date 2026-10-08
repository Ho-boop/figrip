# figrip prior art: reading plot data from PDF vector graphics

Date: 2026-10-08. Read-only research. No code from these repositories was run.
"Inferred" means read from code, not run. "WebFetch summary" means a model-written summary, not raw text.

## 0. Method and limits
- Metadata via `api.github.com`; source via `raw.githubusercontent.com` (curl into scratchpad, then read).
  Search: GitHub search API plus one WebSearch. Accuracy and test claims are the authors'; nothing reproduced.

## 1. Comparison at a glance

| Project | Stars / forks | Lang | Last push (UTC) | License | Input | PDF library | Reads vector paths? | Calibration | Test data in repo |
|---|---|---|---|---|---|---|---|---|---|
| figure-digger (Jeff-920) | 1 / 1 | Python | 2026-09-25 | MIT | PDF; raster fallback; optional VLM | pdfplumber (pdfminer) | yes, automatic | tick text + tick-mark geometry, linear, R^2 >= 0.999 | one synthetic PDF |
| paper2data (liambern) | 0 / 1 | JavaScript | 2026-04-29 | MIT | PDF rendered to SVG by `mutool` | none (CLI tools) | yes, user clicks | manual: four clicks + typed values; linear or log10 per axis | none |
| WebPlotDigitizer | 3,170 / 438 | JavaScript | 2026-07-27 | AGPL-3.0 | raster images | none | no | manual (linear/log, prior knowledge) | n/a |
| pdf-plot-digitizer (siddharth-ravichandran) | 0 | Python | 2026-08-06 | MIT | rasterised PDF | pdf2image / Poppler | no | automatic OCR + Hough; log supported | 5 sample PDFs, ground truth, benchmark |

## 2. figure-digger (Jeff-920/figure-digger)

### 2.1 Facts
- MIT (`Copyright (c) 2026 Jeff-920`). Python. Created 2026-09-17; last push 2026-09-25 03:11 UTC.
  910 KB per API; 1 star, 1 fork, 0 open issues.
- Git tree: 34 blobs. `dig/` has 16 modules; `tests/` has 9 `test_*.py`, a generator and one PDF.
- No `.github/`, no CI, no CONTRIBUTING. Deps: pdfplumber >= 0.11, pypdfium2, pdfminer.six, numpy,
  opencv-python, pillow; optional rapidocr-onnxruntime. Python 3.10+.
- README lists a `docs/` folder that is not in the tree.
- Recent commits (5 sampled, 2026-09-19 to 09-25, one author): Chinese messages, some terse ("update", "v921").
- Sizes: `run.py` 160 KB (CLI, VLM prompts, template rendering in one file); `series_seed.py` 53 KB;
  `extract_lines.py` 30 KB; `vector_extract.py` 22 KB (543 lines).

### 2.2 Files read
- In full: `dig/vector_extract.py`, `dig/axis_ranges.py`, `tests/make_test_vector_pdf.py`,
  `tests/test_vector_instances.py`, `requirements.txt`, `LICENSE`.
- In part: `README.md` (1-250, 380-420), `HANDOFF.md`, `dig/ocr_ticks.py` (140-168),
  `dig/extract_lines.py` (1-30, `_walk` 207-230), `dig/series_seed.py` (header, `unique_path`),
  `dig/split_panels.py` and `dig/legend_colors.py` (headers), `run.py` (1560-1660, 2455-2490, vector route).
- Not read: most of `run.py`, `series_seed.py`, `extract_lines.py`; `vlm_*.py`, `condition_kb.py`,
  `template_out.py`, `verify_overlay.py`.

### 2.3 Pipeline (vector route)
1. `find_vector_regions`: clusters `page.curves/lines/rects` by bbox overlap (10 pt). Keeps regions
   >= 120 x 80 pt with >= 12 objects.
2. `find_axes_box`: largest rect covering >= 45% of the region's width and height; fallback to long lines.
3. `find_tick_labels`: `page.extract_words()` numeric tokens in bands below and left of the frame.
   Unicode minus and en-dash handled; anything else that fails `float()` is dropped.
4. `find_tick_marks`: short strokes (<= 16 pt) just outside the frame give calibration positions.
   Each is paired with the nearest label within 10 pt; at least 60% of labels must pair.
5. `calibrate_from_ticks`: least-squares line, linear only, R^2 >= 0.999 or `None`.
6. `_vector_candidates`: each curve/line with >= 6 points inside the frame and x-extent >= 15% of frame
   width is one candidate; same-colour paths are NOT merged. Marker-only series are rebuilt from circles
   (`c` operators) and rects grouped by colour (`series_seed.best_marker_track`); generic glyphs are
   dropped when circles or squares exist for that colour.
7. `_color_key` prefers a chromatic fill over the stroke colour when a fill is set.
8. Optional VLM anchors (`_assign_specs`): match by anchor distance plus 0.35 colour penalty; reject above 0.32.
9. Output: one CSV per candidate, `{name}_{label}.csv`.

Raster route (summary): `triage_pdf`, `split_panels`, `legend_colors`, `extract_lines` (colour masks,
column walking), `series_seed` (occlusion, markers), `axis_ranges` (RapidOCR, same R^2 gate). VLM is
optional, but the default config expects a DeepSeek key.
Fallback (run.py ~1565-1660): a region with a frame but no readable tick text is rendered to a bitmap
and sent to the raster route. Regions covering >= 90% of the page, or without a frame, are skipped.

### 2.4 Edge-case matrix (vector route)

| Case | Handled? | Notes |
|---|---|---|
| Text tick labels | yes | `extract_words`; the only calibration source. |
| Outlined tick labels (glyphs as paths) | no in vector code | Falls back to raster when a frame exists. README says so. Not tested. |
| Curves as filled outline polygons | no | No code recognises it; both outline edges enter as points. Inferred. |
| Log axes | no | Linear fit only; log ticks fail R^2 >= 0.999, so the region goes to raster. Inferred. `ocr_ticks.py:150-165` computes `looks_log`, which nothing consumes. |
| Inverted axes | probably | Negative-slope linear fit should work. Inferred; no test. |
| Multi-panel pages | partly | Each cluster is a region; no panel splitting in the vector route. Inferred. |
| Legends | weak | Swatches excluded only by length (< 15% of frame width); a long legend line becomes a series. |
| Markers | partly | Circles and rects rebuilt; other shapes are lower-priority generic glyphs. |
| Error bars | no | Single vertical bars fail the extent filter; chains of caps are not handled. |
| Dashed lines | not verified | Dash pattern unused; dashes drawn as separate segments become separate candidates. Inferred. |
| Several series of one colour | yes | Each physical path is its own candidate (regression test). |
| Scientific-notation / multiplier labels | no, silently wrong | `x10^-4` / `x10⁻⁴` fails to parse and is dropped; values come out unscaled. Inferred. Raster route handles multipliers via VLM (HANDOFF 3.2). |
| Clip paths | no | None found. |
| Scanned / image-only figures | out of scope | Raster route. |

### 2.5 Test data
- `tests/test_vector.pdf` (2,156 bytes) from `tests/make_test_vector_pdf.py`: hand-written content
  stream, Helvetica text labels, y = 2x (6 points), y = 0.4x^2 (11 points), one legend swatch pair.
  Ground truth known.
- `test_vector_instances.py`: five same-colour blue paths stay five series; red markers form one track.
  Synthetic only; no external inputs needed (per code).
- The README's "0 error" and HANDOFF's "4.44e-16" refer to the synthetic PDF. Its labels are real text,
  so it never exercises outlined labels, log axes, multipliers or filled outlines.
- No real-paper fixtures in the repo (`papers/` and corpus are gitignored). HANDOFF says
  `test_vlm_pipeline.py` needs a local `papers/` folder (not run).
- README figures ("8 figures / 23 panels / 98% VLM check") are the author's and not reproducible from the repo.

### 2.6 Ideas worth borrowing (credit: figure-digger / Jeff-920)
1. **Positions from tick geometry, values from text.** The docstring says text bounding-box centres
   carry a font-metric bias (size not verified here).
2. **Refuse rather than guess.** Missing frame, R^2 < 0.999 or page-sized region return error dicts, not numbers.
3. **One candidate per physical path.** Earlier same-colour merging concatenated several trajectories.
4. **Marker reconstruction** from `c` operators and rects, explicit geometry preferred over generic glyphs.
5. **Chromatic fill over stroke** for marker colour.
6. **Hybrid fallback**: unreadable vector region is rendered and handed to the raster route (a pattern for outlined labels).
7. **Unique output names** via `series_seed.unique_path` (raster route only).

### 2.7 Weaknesses
- **Silent scale error** for multiplier labels (most important).
- **CSV overwrite in the vector route.** `vector_extract.py:480` writes `f"{name}_{label}.csv"` without
  `unique_path`. Names that collapse to the same string after `_safe_tag` (48 chars, ASCII only), e.g.
  "Model A" and "Model_A", overwrite each other. Inferred; not reproduced.
- **Hard-coded heuristics, no diagnostics**: 15% extent, 16 pt tick length, 10 pt tolerance, 60% match, 6 points. Drops are not reported.
- **Calibration uses one region's bbox**; no check that tick positions fall on the frame.
- **pdfplumber** is pdfminer-based; figrip uses PyMuPDF (generally faster; not measured here).
- **Monolith**: `run.py` mixes CLI, prompts, rendering and reporting.
- **Process**: no CI; tests are scripts, not pytest. HANDOFF lists 7 regression scripts; the tree has 9
  `test_*.py`. Emoji in console output breaks on GBK consoles (HANDOFF section 7).
- **Naming drift**: README says "DIG", repo is figure-digger, HANDOFF uses `fig-extract` / `WebPlotDigitizer-master`.

## 3. paper2data (liambern/paper2data)

### 3.1 Facts
- MIT (`Copyright (c) 2026 Liam Bernheimer`). JavaScript (Node 20+). 52 KB.
- Created and last pushed 2026-04-29, within about 12 minutes of the first commit. Inactive since (~5 months).
  0 stars, 1 fork, 1 open issue (not read). Five commits, all 2026-04-29.
- No npm dependencies. System tools: `mutool` (mupdf-tools), `pdfinfo` (poppler-utils).
- CI (`.github/workflows/node.js.yml`) runs `node --check` only, on Node 20.x and 22.x.

### 3.2 Files read
- In full: `README.md`, `package.json`, `server.mjs` (141 lines), `public/app.js` (657 lines),
  the workflow file, `LICENSE`.
- Not read: `public/index.html`, `styles.css` (searched for `log` / `invert` only).

### 3.3 Pipeline
1. `POST /api/pdf` stores the upload as `.paper2data/current.pdf` (100 MB cap); `pdfinfo` gives page count.
   One PDF at a time.
2. `GET /api/page/N.svg` runs `mutool draw -F svg` (argv spawn, no shell, no cache).
3. Browser inserts the SVG via `innerHTML`, strips top-level `<script>`, and accepts any node with `getTotalLength`.
4. Calibration: user clicks X1, X2, Y1, Y2 on ticks and types values; linear or log10 per axis. A click on
   an SVG element uses its `getBBox` centre.
5. Mapping (`svgToData`): each sampled point is projected onto the x1->x2 and y1->y2 vectors, interpolated in
   (log) value space and inverted.
6. Sampling: `getPointAtLength` every `step` units, mapped to page space with `getScreenCTM`.
7. Export `.dat` (`x y`, 12-digit exponent) or CSV; optional `Points` resampling by linear interpolation.

### 3.4 Edge-case matrix

| Case | Handled? | Notes |
|---|---|---|
| Vector curves | yes | User clicks; Shift-click for multi-segment curves. |
| Outlined tick labels | not a problem | Calibration uses clicked tick geometry, not text. |
| Log axes | yes | User picks log10 per axis. Non-positive calibration value invalidates calibration. Not detected. |
| Inverted axes | probably | Two-point linear with user values should work. Inferred; not tested. |
| Filled outline curves | no special handling | User must click the right element. Not tested. |
| Multi-panel pages | no | Whole page; user selects. |
| Legends, markers, error bars | no | Avoided by selection. |
| Scientific-notation offsets | no | User types full values. |
| Raster figures | out of scope | Stated in README. |

### 3.5 Test data
None. No tests, no fixtures; CI checks syntax only. No numeric accuracy claim is made.

### 3.6 Ideas worth borrowing (credit: paper2data / Liam Bernheimer)
1. **Log axes as a transform**: `scaleValue` -> linear -> `invertScale`. Clean and portable.
2. **Click-to-geometry calibration** via `getBBox` centre, the same idea as figure-digger's tick marks.
3. **Export precision**: 12-digit exponent output avoids rounding loss.
4. **Safe subprocess and static serving**: argv spawn; path check after `resolve`. Minor but correct.

### 3.7 Weaknesses
- Two clicks per axis, no redundancy, no R^2 check. One mis-click goes into every value.
- Not automatic: user selects every element and types every number.
- Single global server state (one PDF, overwritten on upload).
- Full-page SVG injected with `innerHTML`; only top-level `<script>` is stripped. Not tested whether a
  hostile PDF could exploit this. Treat uploaded PDFs as untrusted.
- Axis projection assumes roughly orthogonal axis vectors; not checked.
- Inactive since 2026-04-29; no tests; Linux-only install instructions.

## 4. Other tools found

Search results (GitHub search API, stars order):
- `pdf vector plot data extract`: 3 results (figure-digger, paper2data, larssp/x-y-data-from-pdf-with-inkscape).
- `pdf figure digitize vector`: 0. `get_drawings plot`: 0.
- `plot digitizer pdf` and `digitize plot pdf`: 11 each, mostly unrelated (land records, exam scans); relevant ones below. `svgdigitizer`: 3.
- WebSearch (`PyMuPDF get_drawings digitize curve`): PyMuPDF docs for `get_drawings` and the svgdigitizer
  package; no complete project that reads PDF vector paths and auto-calibrates.

| Name | Stars | Lang | Last push | License | What it does | Checked via |
|---|---|---|---|---|---|---|
| echemdb/svgdigitizer | 30 | Python | 2026-09-18 | GPL-3.0 | (x, y) points from SVG files | API metadata |
| larssp/x-y-data-from-pdf-with-inkscape | 0 | Jupyter | 2021-03-29 | MIT | Inkscape + Octave to extract x-y from vector PDF | API metadata |
| rossv/plotdigitizer | 2 | JavaScript | 2026-05-08 | MIT | digitise plots in PDFs or images | API metadata |
| kylemcdonald/covid-mobility-data | 27 | Python | 2020-04-17 | MIT | script digitising one publisher's PDF plots | API metadata |
| siddharth-ravichandran/pdf-plot-digitizer | 0 | Python | 2026-08-06 | MIT | raster route (pdf2image, OpenCV, Tesseract); sample PDFs and benchmark; author says archived | WebFetch summary |
| jk10001/plotlift | 2 | Python | 2026-09-04 | Apache-2.0 | vision-LLM digitiser for images and PDFs | API metadata |
| gcastellazzi/getGdata | 0 | Python | 2026-01-12 | none | desktop digitiser for images or PDFs | API metadata |
| YueQiandesedianzi/plot-digitizer | 1 | TypeScript | 2026-07-30 | MIT | desktop digitiser, images and multi-page PDFs | API metadata |

### 4.1 WebPlotDigitizer (comparison only)
- 3,170 stars, 438 forks, JavaScript, last push 2026-07-27, **AGPL-3.0**, repo size ~50 MB.
- Fetched README is short: "computer vision assisted" software for "images of data visualizations". No PDF mention.
- Features from prior knowledge, not verified here: raster input; manual linear/log calibration;
  colour-based automatic extraction with averaging; CSV export; browser-based.
- AGPL-3.0: ideas only. No code into MIT figrip. Relevant as the reference for manual calibration UX.

## 5. Could not verify
- Nothing run; all accuracy and test claims are the authors'.
- Not read: rest of figure-digger's `run.py`, `series_seed.py`, `extract_lines.py` body, `vlm_*.py`;
  paper2data `index.html` and `styles.css`; figure-digger's total commit count.
- Several metadata rows and the siddharth / WebPlotDigitizer feature lists come from WebFetch summaries.
- Behaviour marked "inferred" (log, multiplier, outlined labels, filled outlines, CSV overwrite) is from reading code only.
- Search coverage: GitHub keyword search and one web search; no PyPI or paper-corpus search.

## Implications for figrip

1. **Keep PyMuPDF; treat figure-digger's vector module as a heuristic reference, not a dependency.**
   PyMuPDF path records expose stroke/fill type, width and dash information; check the installed
   version's docs for exact field names.
2. **Calibration gate**: tick-mark geometry for positions, label text for values, >= 3 labels,
   R^2 >= 0.999. On failure return diagnostics, not numbers. Credit figure-digger for the tick-geometry idea.
3. **Log axes, explicitly**: transform to log10, fit linear, invert (paper2data's approach). Add a generated log fixture.
4. **Multiplier and scientific labels**: parse `x10^n`, `x10⁻⁴`, `1e-4`, `1 x 10^-4`; treat a standalone
   multiplier as a scale factor. This is the silent error figure-digger misses.
5. **Outlined labels**: report "frame found, no numeric labels" and stop. Do not guess. A later OCR or raster
   fallback can follow figure-digger's render-and-hand-off pattern.
6. **One series per physical path, collision-safe output names.** Do not merge by colour; keep colour, width
   and dash as features; rebuild marker-only series separately. figure-digger's vector route lacks the collision guard.
7. **Legends**: exclude swatches explicitly (adjacent text or boxes), not by length alone.
8. **Fixtures beat claims**: a generated suite with ground truth for text labels, outlined labels (text
   converted to paths), log axes, multipliers, markers, dashes, legend swatches, multi-panel pages and
   filled-outline curves. figure-digger has one synthetic fixture; this is where figrip can clearly exceed it.
9. **Diagnostics for every drop**: each rejected candidate listed with a reason. figure-digger's thresholds are silent.
10. **Do not copy the dead flag**: `looks_log` is computed but never used upstream. An unconsumed detector
    gives false confidence.

**Upstream contribution to figure-digger: realistic in a limited form, not as a strategy.**
- License: MIT to MIT. Contributions and copied functions with attribution are compatible.
- Activity: last push 2026-09-25, so the author is active; but one star, one author, no CI, no
  contribution guide. Review and merge are uncertain.
- Code fit: vector module (~540 lines) is readable; comments, prompts and commits are in Chinese.
- In order of value: (a) file issues with small generated PDFs for multiplier labels, outlined labels and
  log axes; (b) one small PR (multiplier parsing plus test) if the author responds; (c) skip any large
  rewrite, such as a PyMuPDF port, which conflicts with their architecture.
- For resume purposes, figrip with a clean test suite is a stronger signal than merges into a one-star repo.

## Sources
- https://github.com/Jeff-920/figure-digger, https://github.com/liambern/paper2data
- https://github.com/automeris-io/WebPlotDigitizer
- https://github.com/echemdb/svgdigitizer
- https://github.com/larssp/x-y-data-from-pdf-with-inkscape
- https://github.com/siddharth-ravichandran/pdf-plot-digitizer
- https://github.com/rossv/plotdigitizer
- https://github.com/jk10001/plotlift
