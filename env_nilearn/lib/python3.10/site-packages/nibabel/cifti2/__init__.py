# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""CIFTI-2 format IO

.. currentmodule:: nibabel.cifti2

.. autosummary::
   :toctree: ../generated

   cifti2
   cifti2_axes
"""

from .cifti2 import (
    CIFTI_BRAIN_STRUCTURES,
    CIFTI_MODEL_TYPES,
    Cifti2BrainModel,
    Cifti2Header,
    Cifti2HeaderError,
    Cifti2Image,
    Cifti2Label,
    Cifti2LabelTable,
    Cifti2Matrix,
    Cifti2MatrixIndicesMap,
    Cifti2MetaData,
    Cifti2NamedMap,
    Cifti2Parcel,
    Cifti2Surface,
    Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ,
    Cifti2VertexIndices,
    Cifti2Vertices,
    Cifti2Volume,
    Cifti2VoxelIndicesIJK,
    load,
    save,
)
from .cifti2_axes import Axis, BrainModelAxis, LabelAxis, ParcelsAxis, ScalarAxis, SeriesAxis
from .parse_cifti2 import Cifti2Extension
