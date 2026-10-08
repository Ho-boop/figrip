# figrip

**Rip the exact numbers out of vector plots in PDF papers - no screenshots, no clicking, no pixel error.**

`figrip` reads the drawing instructions inside a PDF (the paths that make up the axes, ticks and curves), fits a
transform from page coordinates to data values using the tick labels, and returns every point of every curve.
It was built for the daily chore of an organic-photovoltaics (OPV) researcher - *"get the J-V curve and the EQE
spectrum of that paper so I can plot it next to my own device"* - and ships with an `opv` toolbox that turns a
recovered J-V curve straight into Jsc / Voc / FF / PCE.

> Status: early alpha (v0.1). Works on figures that are still vector inside the PDF; see [Limitations](#limitations).

![overlay preview: extracted points (coloured crosses) and calibrated ticks drawn on the original figure](docs/overlay_example.png)
<!-- TODO: replace with a real overlay produced by `figrip extract --preview` (a GIF of the 3-command workflow would be even better) -->

## The pain

You find a paper with the perfect J-V curve. The usual routine:

1. screenshot the figure (the PDF's precise vector data is now a blurry bitmap),
2. open WebPlotDigitizer, calibrate the axes by clicking four reference points,
3. click along the curve - 30 points, maybe 100 - and hope your hand is steady,
4. repeat for each of the five curves in the plot, then for the next paper.

Every step adds error and takes minutes. And it is silly: **the journal figure was vector data all along.**

## The idea

Most figures in recent journal PDFs (Adv. Mater., Nat. Energy, JACS, ...) are embedded as vector graphics:
the curve is a polyline with hundreds of exact vertices. `figrip`

1. extracts all paths and text from a page (PyMuPDF),
2. finds the plot frames and the tick marks, reads the tick labels and fits `value = a * position + b`
   (or `log10(value) = ...`) by least squares - the fit residual tells you how much to trust it,
3. picks the data curves/markers out of the other graphics (axes, grid, legend swatches),
4. maps every vertex through the fit and writes CSV / JSON / Origin files,
5. draws an **overlay preview** (your extracted points on top of the original figure, with the values it believes the
   ticks have) so you can verify the result in two seconds instead of trusting it blindly.

## Install

Python 3.10+.

```bash
git clone <this repo>
cd figrip
pip install -e .            # PyMuPDF + numpy
pip install -e ".[ocr]"     # optional: OCR for papers whose tick labels are outlined shapes, not text
pip install -e ".[dev]"     # pytest, matplotlib (tests / corpus generator)
```

(Not on PyPI yet.)  `python -m figrip ...` works like the `figrip` command.

## Quickstart

Page numbers are **1-based**, as printed in your PDF viewer.

### 1. See what is on the page

```bash
figrip scan paper.pdf -p 4
```
```
page  #  panel  bbox (pt)       x axis                    y axis                              series  calib
----  -  -----  --------------  ------------------------  ----------------------------------  ------  --------
4     1  c      60,30,260,190   Voltage (V) [-0.1..1.1]   Current density (mA cm-2) [-25..5]  3       text/text
4     2  d      ...
```
(Illustrative output.)

### 2. Extract the data

```bash
figrip extract paper.pdf -p 4 --panel d -o out --format csv --preview
```
writes `out/p4_d.csv` (wide layout: `<series>_x, <series>_y` side by side - paste into Excel or Origin),
and with `--preview` the overlay `out/p4_d.png`. Other formats: `--format json` (includes the calibration and tick
pairs, for reproducibility), `--format origin` (tab-separated with Long name / Units / Comments rows that OriginPro's
import wizard turns into column labels), `--format all`.
`--panel` takes the panel letter (`d`) or the 1-based index of the panel on the page (`2`).

### 3. J-V curve -> device metrics

```bash
figrip jv paper.pdf -p 4 --panel d --pin 100 -o jv
```
```
panel  series     Jsc (mA/cm2)  Voc (V)  FF (%)  PCE (%)  points
-----  ---------  ------------  -------  ------  -------  ------
p4_d   PM6:Y6     25.00         0.850    78.4    16.66    240
p4_d   PM6:Y6-ref 22.50         0.820    72.4    13.35    240
```
(Synthetic example: values from a model curve with known answers.)  One text file per curve is written
(`jv/PM6_Y6.txt`, ...) in the 3-line tab-separated layout of a solar-simulator export, so tools that read such
files (the author's own device-performance spreadsheet tool does) can ingest literature curves next to
your own devices.

Behind this is `figrip.opv.jv_metrics(v, j, pin=100)`, usable on its own:
- handles J negative in the 4th quadrant, "photocurrent positive", unsorted or duplicated V, mV / A m-2 units;
- Jsc = J(0) and Voc = V(J=0) by linear interpolation, the maximum power point by maximising V*J on a dense
  shape-preserving interpolation (so sparse digitised data cannot overshoot);
- a forward+reverse sweep in one series is split at the turning points and both are reported;
- curves that stop before J=0 or start after V=0 are extrapolated **and flagged** in `notes`;
- validated against a single-diode model with known analytic/numerical answers (see `tests/test_opv.py`).

Also in `figrip.opv`: `eqe_integrated_jsc(wl_nm, eqe_percent)` integrates an EQE spectrum against the bundled
ASTM G173-03 AM1.5G spectrum (NREL; `figrip/data/am15g.csv`) to give the integrated Jsc to compare with the J-V value.
It is a Python function for now (no CLI command yet).

### When automatic calibration fails

Give the tick values by hand (x left to right, y **bottom to top** as they sit on the page) and, if needed, the
plot area in PDF points (see `figrip scan`):

```bash
figrip extract paper.pdf -p 4 --x-ticks "0,0.2,0.4,0.6,0.8" --y-ticks "0,-5,-10,-15,-20" --bbox 60,30,260,190
```
(`--xscale log` / `--yscale log` force a log axis.)

## How it copes with real-world publishing quirks

| Quirk | Approach | Status |
|---|---|---|
| Tick labels are real text | read position and value directly | implemented |
| Tick labels are outlined glyphs (very common in Adobe Illustrator exports) | OCR the rendered label area (optional `[ocr]` extra), or give `--x-ticks/--y-ticks` | implemented, accuracy still being measured |
| Curves are drawn as *outlined strokes* (thin filled polygons) | recover the centreline of the polygon | in development |
| Inverted axis (e.g. J = -20 drawn at the top) | the transform is fitted per axis, sign of the slope is free | implemented |
| Log axes | fit `log10(value)` linearly; `--xscale/--yscale` to force | implemented |
| Several panels (a, b, c, d) on one page | every framed plot is calibrated on its own; select with `--panel` | implemented |
| Markers instead of lines, error bars, grids | separated from the curve by style and geometry | partly (see tests/corpus) |

The synthetic test corpus under `tests/corpus/` regenerates PDFs with exactly known data so that each quirk above
has a measurable score.

## Limitations

- **Rasterised figures** (a PNG/JPEG inside the PDF - typical for older papers and many supporting-information files)
  contain no vector data. `figrip` tells you so; use a pixel digitiser such as WebPlotDigitizer there.
- Bar charts, pie charts, polar plots, 3D plots, heat maps are not supported.
- Broken axes (an axis with a gap) and two y axes in one frame are not handled yet.
- Curves that overlap exactly are returned as one series; legend-to-curve matching is heuristic - check the overlay.
- A scanned/OCR'd copy of a paper is a bitmap again: you need the publisher's original PDF.
- Always look at the overlay. A wrong tick label read by OCR gives a systematically wrong scale; the overlay shows the
  values the tool assigned to every tick.

## Comparison

| | figrip | WebPlotDigitizer | figure-digger | paper2data |
|---|---|---|---|---|
| Input | vector figures in PDF | any image (raster) | vector figures in PDF | PDF documents |
| Precision | exact vertices (limited only by the paper's own drawing) | pixel-limited, depends on clicking / auto-extraction | exact vertices | - |
| Effort per curve | one command | calibrate + click or auto-trace | - | - |
| Maturity | early alpha | mature, widely used, active, runs in the browser | early stage | inactive |
| Domain extras | OPV J-V metrics, EQE integration | none (general purpose) | none | none |

WebPlotDigitizer is the right tool whenever the figure is a bitmap; `figrip` only complements it for the
(large) fraction of figures that are vector. Please check each project's current state yourself - this table
reflects the author's knowledge at the time of writing.

## Repository layout

```
figrip/primitives.py   PDF page -> paths + text spans (beziers flattened)
figrip/axes.py         plot-frame detection + tick calibration (text / OCR / manual)
figrip/series.py       curve and marker extraction
figrip/pipeline.py     glue: PDF -> FigureResult
figrip/opv.py          J-V metrics, EQE integration, instrument-style txt
figrip/export.py       CSV / JSON / Origin writers
figrip/preview.py      overlay PNG
figrip/cli.py          scan | extract | jv
tests/                 unit tests + synthetic corpus with exact ground truth
```
Architecture notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT - see [LICENSE](LICENSE). The AM1.5G table is the public ASTM G173-03 reference spectrum from NREL
(<https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html>).

---

## 简介（中文）

**figrip：直接从论文 PDF 的矢量图里"读"出曲线数据，不用截图、不用点点。**

做有机太阳能电池（OPV）的同学都经历过：想把别人论文里的 J-V 曲线或 EQE 谱拿来和自己的器件对比，只能截图、
放进 WebPlotDigitizer、手动标定坐标轴、再一个点一个点地点，既慢又有像素误差。但期刊里的图很多本来就是矢量的——
曲线是精确的折线。figrip 的做法是：读出 PDF 里的路径和刻度标签，拟合"页面坐标 -> 数据值"的变换，
一次性还原每条曲线的全部点，并生成叠加预览图让你一眼核对。

```bash
figrip scan  paper.pdf -p 4                       # 这一页有哪些图
figrip extract paper.pdf -p 4 --panel d --preview # 导出 CSV（Excel/Origin 可直接用）+ 校验图
figrip jv    paper.pdf -p 4 --panel d             # 直接算 Jsc / Voc / FF / PCE，并输出可被
                                                  # 性能表格处理台 (opv_performance_tool) 读取的 txt
```

页码从 1 开始。刻度标签被转成轮廓（非文字）时可装 OCR 可选依赖，或者用
`--x-ticks`、`--y-ticks`（y 从下到上）手动给出刻度值。局限：位图（光栅）图片没有矢量数据，
这种情况请用 WebPlotDigitizer；柱状图、3D 图、断轴暂不支持。开源协议 MIT。
