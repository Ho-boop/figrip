"""Shared data model. Every module talks through these types.

Coordinate convention: all "page" coordinates are PDF points in PyMuPDF's
page space (origin top-left, y grows DOWN). Data coordinates are whatever
the figure's axes say.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

Color = tuple[float, float, float]
BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 (page coords)


@dataclass
class Path:
    """One PDF drawing operation, normalised.

    Beziers are already flattened, so every subpath is a polyline.
    """
    id: int                       # index in PageContent.paths
    subpaths: list[np.ndarray]    # each (N, 2) float array, page coords
    closed: list[bool]            # per subpath
    stroke: Optional[Color]       # None if not stroked
    fill: Optional[Color]         # None if not filled
    width: float                  # stroke width (0 if not stroked)
    dashes: Optional[str]         # PDF dash spec, None/"[] 0" = solid
    bbox: BBox
    seqno: int                    # painting order on the page
    has_curves: bool = False      # any bezier segment in the source
    n_segments: int = 0           # number of source items ('l','c','re','qu')

    @property
    def kind(self) -> Literal["stroke", "fill", "fillstroke"]:
        if self.stroke is not None and self.fill is not None:
            return "fillstroke"
        return "stroke" if self.stroke is not None else "fill"

    @property
    def w(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def points(self) -> np.ndarray:
        """All vertices of all subpaths, (N, 2)."""
        if not self.subpaths:
            return np.zeros((0, 2))
        return np.vstack(self.subpaths)


@dataclass
class TextSpan:
    text: str
    bbox: BBox
    size: float
    font: str
    dir: tuple[float, float]      # (1,0) horizontal, (0,-1) rotated 90° CCW (typical y-label)
    origin: tuple[float, float]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class PageContent:
    page_no: int                  # 0-based
    rect: BBox
    paths: list[Path]
    texts: list[TextSpan]
    images: list[BBox]            # placed raster images


@dataclass
class AxisCalib:
    """Maps one page coordinate (x or y) to data values.

    linear: value = a * pos + b
    log:    log10(value) = a * pos + b
    """
    orient: Literal["x", "y"]
    scale: Literal["linear", "log"]
    a: float
    b: float
    ticks: list[tuple[float, float]] = field(default_factory=list)  # (page_pos, value)
    residual: float = 0.0         # max |fit - tick value| in units of tick spacing
    label: Optional[str] = None   # axis title, e.g. "Voltage (V)"
    source: Literal["text", "ocr", "manual"] = "text"
    color: Optional[Color] = None  # spine/tick colour when the axis is coloured (twin-y plots)

    def to_data(self, pos):
        v = self.a * np.asarray(pos, dtype=float) + self.b
        return np.power(10.0, v) if self.scale == "log" else v

    def to_page(self, value):
        v = np.asarray(value, dtype=float)
        if self.scale == "log":
            v = np.log10(v)
        return (v - self.b) / self.a

    @staticmethod
    def fit(orient, ticks: list[tuple[float, float]], scale="linear", **kw) -> "AxisCalib":
        """Least-squares fit from (page_pos, value) pairs (>= 2)."""
        pos = np.array([t[0] for t in ticks], float)
        val = np.array([t[1] for t in ticks], float)
        y = np.log10(val) if scale == "log" else val
        a, b = np.polyfit(pos, y, 1)
        spacing = np.median(np.abs(np.diff(np.sort(y)))) if len(y) > 1 else 1.0
        res = float(np.max(np.abs(a * pos + b - y)) / spacing) if spacing else math.inf
        return AxisCalib(orient, scale, float(a), float(b), list(ticks), res, **kw)


@dataclass
class Panel:
    """One x-y plot: a plot area plus calibrated axes."""
    page_no: int
    bbox: BBox                    # plot area (inside the axes frame), page coords
    x: AxisCalib
    y: AxisCalib
    label: Optional[str] = None   # panel letter "a", "b"... if found
    notes: list[str] = field(default_factory=list)  # warnings for the user
    y2: Optional[AxisCalib] = None  # right-hand y axis of a twin-y plot

    def to_data(self, pts: np.ndarray, axis: str = "y") -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 2)
        yc = self.y2 if axis == "y2" and self.y2 is not None else self.y
        return np.column_stack([self.x.to_data(pts[:, 0]), yc.to_data(pts[:, 1])])


@dataclass
class Series:
    name: Optional[str]           # from legend if matched
    kind: Literal["line", "markers", "line+markers"]
    style: dict                   # {"color": Color, "width": float, "dashes": str, "marker": str}
    page_pts: np.ndarray          # (N, 2) page coords, ordered along x for lines
    data: np.ndarray              # (N, 2) data coords (filled by pipeline via Panel.to_data)
    path_ids: list[int] = field(default_factory=list)
    axis: Literal["y", "y2"] = "y"  # which y axis the series is read against


@dataclass
class FigureResult:
    panel: Panel
    series: list[Series]
