"""
Display objects and utilities.

These objects are returned by plotting functions
from the :mod:`~nilearn.plotting` module.
"""

from .axes import BaseAxes, CutAxes, GlassBrainAxes

from .projectors import (
    OrthoProjector, XZProjector, YZProjector,
    YXProjector, XProjector, YProjector, ZProjector,
    LZRYProjector, LYRZProjector, LYRProjector,
    LZRProjector, LRProjector, LProjector, RProjector,
    get_projector,
)

from .slicers import (
    OrthoSlicer, TiledSlicer, MosaicSlicer, BaseSlicer,
    XZSlicer, YZSlicer, YXSlicer, XSlicer, YSlicer,
    ZSlicer, get_slicer,
)

from .figures import SurfaceFigure, PlotlySurfaceFigure

__all__ = ["BaseAxes", "CutAxes", "GlassBrainAxes",
           "OrthoProjector", "XZProjector", "YZProjector",
           "YXProjector", "XProjector", "YProjector", "ZProjector",
           "LZRYProjector", "LYRZProjector", "LYRProjector",
           "LZRProjector", "LRProjector", "LProjector", "RProjector",
           "OrthoSlicer", "TiledSlicer", "MosaicSlicer", "BaseSlicer",
           "XZSlicer", "YZSlicer", "YXSlicer", "XSlicer", "YSlicer",
           "ZSlicer", "get_projector", "get_slicer", "SurfaceFigure",
           "PlotlySurfaceFigure"]
