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
    "BaseSlicer",
    "CutAxes",
    "GlassBrainAxes",
    "LProjector",
    "LRProjector",
    "LYRProjector",
    "LYRZProjector",
    "LZRProjector",
    "LZRYProjector",
    "MosaicSlicer",
    "OrthoProjector",
    "OrthoSlicer",
    "PlotlySurfaceFigure",
    "RProjector",
    "SurfaceFigure",
    "TiledSlicer",
    "XProjector",
    "XSlicer",
    "XZProjector",
    "XZSlicer",
    "YProjector",
    "YSlicer",
    "YXProjector",
    "YXSlicer",
    "YZProjector",
    "YZSlicer",
    "ZProjector",
    "ZSlicer",
    "get_projector",
    "get_slicer",
]
