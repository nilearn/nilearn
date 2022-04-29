"""
Display objects and utilities.

These objects are returned by plotting functions
from the :mod:`~nilearn.plotting` module.
"""

from nilearn.plotting.displays._axes import (
    BaseAxes, CutAxes, GlassBrainAxes
)
from nilearn.plotting.displays._projectors import (
    OrthoProjector, XZProjector, YZProjector,
    YXProjector, XProjector, YProjector, ZProjector,
    LZRYProjector, LYRZProjector, LYRProjector,
    LZRProjector, LRProjector, LProjector, RProjector,
    get_projector,
)
from nilearn.plotting.displays._slicers import (
    OrthoSlicer, TiledSlicer, MosaicSlicer, BaseSlicer,
    XZSlicer, YZSlicer, YXSlicer, XSlicer, YSlicer,
    ZSlicer, get_slicer,
)
from nilearn.plotting.displays._figures import (
    SurfaceFigure, PlotlySurfaceFigure
)

__all__ = ["BaseAxes", "CutAxes", "GlassBrainAxes",
           "OrthoProjector", "XZProjector", "YZProjector",
           "YXProjector", "XProjector", "YProjector", "ZProjector",
           "LZRYProjector", "LYRZProjector", "LYRProjector",
           "LZRProjector", "LRProjector", "LProjector", "RProjector",
           "OrthoSlicer", "TiledSlicer", "MosaicSlicer", "BaseSlicer",
           "XZSlicer", "YZSlicer", "YXSlicer", "XSlicer", "YSlicer",
           "ZSlicer", "get_projector", "get_slicer", "SurfaceFigure",
           "PlotlySurfaceFigure"]
