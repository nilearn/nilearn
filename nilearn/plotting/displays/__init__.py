"""
Display objects and utilities.

These objects are returned by plotting functions
from the :mod:`~nilearn.plotting` module.
"""

from nilearn.plotting.displays._axes import BaseAxes, CutAxes, GlassBrainAxes
from nilearn.plotting.displays._figures import (
    PlotlySurfaceFigure,
    SurfaceFigure,
)
from nilearn.plotting.displays._projectors import (
    LProjector,
    LRProjector,
    LYRProjector,
    LYRZProjector,
    LZRProjector,
    LZRYProjector,
    OrthoProjector,
    RProjector,
    XProjector,
    XZProjector,
    YProjector,
    YXProjector,
    YZProjector,
    ZProjector,
    get_projector,
)
from nilearn.plotting.displays._slicers import (
    BaseSlicer,
    MosaicSlicer,
    OrthoSlicer,
    TiledSlicer,
    XSlicer,
    XZSlicer,
    YSlicer,
    YXSlicer,
    YZSlicer,
    ZSlicer,
    get_slicer,
)

__all__ = [
    "BaseAxes",
    "CutAxes",
    "GlassBrainAxes",
    "OrthoProjector",
    "XZProjector",
    "YZProjector",
    "YXProjector",
    "XProjector",
    "YProjector",
    "ZProjector",
    "LZRYProjector",
    "LYRZProjector",
    "LYRProjector",
    "LZRProjector",
    "LRProjector",
    "LProjector",
    "RProjector",
    "OrthoSlicer",
    "TiledSlicer",
    "MosaicSlicer",
    "BaseSlicer",
    "XZSlicer",
    "YZSlicer",
    "YXSlicer",
    "XSlicer",
    "YSlicer",
    "ZSlicer",
    "get_projector",
    "get_slicer",
    "SurfaceFigure",
    "PlotlySurfaceFigure",
]
