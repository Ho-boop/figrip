"""PDF -> FigureResults: glue primitives, axes and series together."""
from __future__ import annotations

from typing import Iterable, Optional

import pymupdf

from .model import FigureResult
from .primitives import extract_page


def extract(pdf_path: str, pages: Optional[Iterable[int]] = None, ocr: bool = True,
            hints: Optional[dict] = None) -> list[FigureResult]:
    """Find every calibratable plot on the given pages (0-based; None = all).

    `hints` is the manual-calibration dict understood by axes.find_panels.
    """
    from . import axes, series  # imported lazily so partial checkouts still import

    results: list[FigureResult] = []
    with pymupdf.open(pdf_path) as doc:
        for pno in (pages if pages is not None else range(len(doc))):
            page = doc[pno]
            pc = extract_page(page)
            for panel in axes.find_panels(pc, page=page if ocr else None, hints=hints):
                found = series.extract_series(pc, panel)
                for s in found:
                    s.data = panel.to_data(s.page_pts, s.axis)
                results.append(FigureResult(panel, found))
    return results
