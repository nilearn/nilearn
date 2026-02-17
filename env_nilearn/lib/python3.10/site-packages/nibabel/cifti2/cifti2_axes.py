"""
Defines :class:`Axis` objects to create, read, and manipulate CIFTI-2 files

These axes provide an alternative interface to the information in the CIFTI-2 header.
Each type of CIFTI-2 axes describing the rows/columns in a CIFTI-2 matrix is given a unique class:

* :class:`BrainModelAxis`: each row/column is a voxel or vertex
* :class:`ParcelsAxis`: each row/column is a group of voxels and/or vertices
* :class:`ScalarAxis`: each row/column has a unique name (with optional meta-data)
* :class:`LabelAxis`: each row/column has a unique name and label table (with optional meta-data)
* :class:`SeriesAxis`: each row/column is a timepoint, which increases monotonically

All of these classes are derived from the Axis class.

After loading a CIFTI-2 file a tuple of axes describing the rows and columns can be obtained
from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
(e.g. ``nibabel.load(<filename>).header.get_axis()``). Inversely, a new
:class:`.cifti2.Cifti2Header` object can be created from existing Axis objects
using the :meth:`.cifti2.Cifti2Header.from_axes` factory method.

CIFTI-2 Axis objects of the same type can be concatenated using the '+'-operator.
Numpy indexing also works on axes
(except for SeriesAxis objects, which have to remain monotonically increasing or decreasing).

Creating new CIFTI-2 axes
-------------------------
New Axis objects can be constructed by providing a description for what is contained
in each row/column of the described tensor. For each Axis sub-class this descriptor is:

* :class:`BrainModelAxis`: a CIFTI-2 structure name and a voxel or vertex index
* :class:`ParcelsAxis`: a name and a sequence of voxel and vertex indices
* :class:`ScalarAxis`: a name and optionally a dict of meta-data
* :class:`LabelAxis`: a name, dict of label index to name and colour,
  and optionally a dict of meta-data
* :class:`SeriesAxis`: the time-point of each row/column is set by setting the start, stop, size,
  and unit of the time-series

Several helper functions exist to create new :class:`BrainModelAxis` axes:

* :meth:`BrainModelAxis.from_mask` creates a new BrainModelAxis volume covering the
  non-zero values of a mask
* :meth:`BrainModelAxis.from_surface` creates a new BrainModelAxis surface covering the provided
  indices of a surface

A :class:`ParcelsAxis` axis can be created from a sequence of :class:`BrainModelAxis` axes using
:meth:`ParcelsAxis.from_brain_models`.

Examples
--------
We can create brain models covering the left cortex and left thalamus using:

>>> from nibabel import cifti2
>>> import numpy as np
>>> bm_cortex = cifti2.BrainModelAxis.from_mask([True, False, True, True],
...                                             name='cortex_left')
>>> bm_thal = cifti2.BrainModelAxis.from_mask(np.ones((2, 2, 2)), affine=np.eye(4),
...                                           name='thalamus_left')

In this very simple case ``bm_cortex`` describes a left cortical surface skipping the second
out of four vertices. ``bm_thal`` contains all voxels in a 2x2x2 volume.

Brain structure names automatically get converted to valid CIFTI-2 identifiers using
:meth:`BrainModelAxis.to_cifti_brain_structure_name`.
A 1-dimensional mask will be automatically interpreted as a surface element and a 3-dimensional
mask as a volume element.

These can be concatenated in a single brain model covering the left cortex and thalamus by
simply adding them together

>>> bm_full = bm_cortex + bm_thal

Brain models covering the full HCP grayordinate space can be constructed by adding all the
volumetric and surface brain models together like this (or by reading one from an already
existing HCP file).

Getting a specific brain region from the full brain model is as simple as:

>>> assert bm_full[bm_full.name == 'CIFTI_STRUCTURE_CORTEX_LEFT'] == bm_cortex
>>> assert bm_full[bm_full.name == 'CIFTI_STRUCTURE_THALAMUS_LEFT'] == bm_thal

You can also iterate over all brain structures in a brain model:

>>> for idx, (name, slc, bm) in enumerate(bm_full.iter_structures()):
...     print((str(name), slc))
...     assert bm == bm_full[slc]
...     assert bm == bm_cortex if idx == 0 else bm_thal
('CIFTI_STRUCTURE_CORTEX_LEFT', slice(0, 3, None))
('CIFTI_STRUCTURE_THALAMUS_LEFT', slice(3, None, None))

In this case there will be two iterations, namely:
('CIFTI_STRUCTURE_CORTEX_LEFT', slice(0, <size of cortex mask>), bm_cortex)
and
('CIFTI_STRUCTURE_THALAMUS_LEFT', slice(<size of cortex mask>, None), bm_thal)

ParcelsAxis can be constructed from selections of these brain models:

>>> parcel = cifti2.ParcelsAxis.from_brain_models([
...        ('surface_parcel', bm_cortex[:2]),  # contains first 2 cortical vertices
...        ('volume_parcel', bm_thal),  # contains thalamus
...        ('combined_parcel', bm_full[[1, 8, 10]]),  # contains selected voxels/vertices
...    ])

Time series are represented by their starting time (typically 0), step size
(i.e. sampling time or TR), and number of elements:

>>> series = cifti2.SeriesAxis(start=0, step=100, size=5000)

So a header for fMRI data with a TR of 100 ms covering the left cortex and thalamus with
5000 timepoints could be created with

>>> type(cifti2.Cifti2Header.from_axes((series, bm_cortex + bm_thal)))
<class 'nibabel.cifti2.cifti2.Cifti2Header'>

Similarly the curvature and cortical thickness on the left cortex could be stored using a header
like:

>>> type(cifti2.Cifti2Header.from_axes((cifti2.ScalarAxis(['curvature', 'thickness']),
...                                     bm_cortex)))
<class 'nibabel.cifti2.cifti2.Cifti2Header'>
"""

import abc
from operator import xor

import numpy as np

from . import cifti2


def from_index_mapping(mim):
    """
    Parses the MatrixIndicesMap to find the appropriate CIFTI-2 axis describing the rows or columns

    Parameters
    ----------
    mim : :class:`.cifti2.Cifti2MatrixIndicesMap`

    Returns
    -------
    axis : subclass of :class:`Axis`
    """
    return_type = {
        'CIFTI_INDEX_TYPE_SCALARS': ScalarAxis,
        'CIFTI_INDEX_TYPE_LABELS': LabelAxis,
        'CIFTI_INDEX_TYPE_SERIES': SeriesAxis,
        'CIFTI_INDEX_TYPE_BRAIN_MODELS': BrainModelAxis,
        'CIFTI_INDEX_TYPE_PARCELS': ParcelsAxis,
    }
    return return_type[mim.indices_map_to_data_type].from_index_mapping(mim)


def to_header(axes):
    """
    Converts the axes describing the rows/columns of a CIFTI-2 vector/matrix to a Cifti2Header

    Parameters
    ----------
    axes : iterable of :py:class:`Axis` objects
        one or more axes describing each dimension in turn

    Returns
    -------
    header : :class:`.cifti2.Cifti2Header`
    """
    axes = tuple(axes)
    mims_all = []
    matrix = cifti2.Cifti2Matrix()
    for dim, ax in enumerate(axes):
        if ax in axes[:dim]:
            dim_prev = axes.index(ax)
            mims_all[dim_prev].applies_to_matrix_dimension.append(dim)
            mims_all.append(mims_all[dim_prev])
        else:
            mim = ax.to_mapping(dim)
            mims_all.append(mim)
            matrix.append(mim)
    return cifti2.Cifti2Header(matrix)


class Axis(abc.ABC):
    """
    Abstract class for any object describing the rows or columns of a CIFTI-2 vector/matrix

    Mainly used for type checking.

    Base class for the following concrete CIFTI-2 axes:

    * :class:`BrainModelAxis`: each row/column is a voxel or vertex
    * :class:`ParcelsAxis`: each row/column is a group of voxels and/or vertices
    * :class:`ScalarAxis`: each row/column has a unique name with optional meta-data
    * :class:`LabelAxis`: each row/column has a unique name and label table with optional meta-data
    * :class:`SeriesAxis`: each row/column is a timepoint, which increases monotonically
    """

    @property
    def size(self):
        return len(self)

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        """
        Compares whether two Axes are equal

        Parameters
        ----------
        other : Axis
            other axis to compare to

        Returns
        -------
        False if the axes don't have the same type or if their content differs
        """
        pass

    @abc.abstractmethod
    def __add__(self, other):
        """
        Concatenates two Axes of the same type

        Parameters
        ----------
        other : Axis
            axis to be appended to the current one

        Returns
        -------
        Axis of the same subtype as self and other
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        """
        Extracts definition of single row/column or new Axis describing a subset of the rows/columns
        """
        pass


class BrainModelAxis(Axis):
    """
    Each row/column in the CIFTI-2 vector/matrix represents a single vertex or voxel

    This Axis describes which vertex/voxel is represented by each row/column.
    """

    def __init__(
        self, name, voxel=None, vertex=None, affine=None, volume_shape=None, nvertices=None
    ):
        """
        New BrainModelAxis axes can be constructed by passing on the greyordinate brain-structure
        names and voxel/vertex indices to the constructor or by one of the
        factory methods:

        - :py:meth:`~BrainModelAxis.from_mask`: creates surface or volumetric BrainModelAxis axis
          from respectively 1D or 3D masks
        - :py:meth:`~BrainModelAxis.from_surface`: creates a surface BrainModelAxis axis

        The resulting BrainModelAxis axes can be concatenated by adding them together.

        Parameters
        ----------
        name : array_like
            brain structure name or (N, ) string array with the brain structure names
        voxel : array_like, optional
            (N, 3) array with the voxel indices (can be omitted for CIFTI-2 files only
            covering the surface)
        vertex :  array_like, optional
            (N, ) array with the vertex indices (can be omitted for volumetric CIFTI-2 files)
        affine : array_like, optional
            (4, 4) array mapping voxel indices to mm space (not needed for CIFTI-2 files only
            covering the surface)
        volume_shape : tuple of three integers, optional
            shape of the volume in which the voxels were defined (not needed for CIFTI-2 files only
            covering the surface)
        nvertices : dict from string to integer, optional
            maps names of surface elements to integers (not needed for volumetric CIFTI-2 files)
        """
        if voxel is None:
            if vertex is None:
                raise ValueError('At least one of voxel or vertex indices should be defined')
            nelements = len(vertex)
            self.voxel = np.full((nelements, 3), fill_value=-1, dtype=int)
        else:
            nelements = len(voxel)
            self.voxel = np.asanyarray(voxel, dtype=int)

        if vertex is None:
            self.vertex = np.full(nelements, fill_value=-1, dtype=int)
        else:
            self.vertex = np.asanyarray(vertex, dtype=int)

        if isinstance(name, str):
            name = [self.to_cifti_brain_structure_name(name)] * self.vertex.size
        self.name = np.asanyarray(name, dtype='U')

        if nvertices is None:
            self.nvertices = {}
        else:
            self.nvertices = {
                self.to_cifti_brain_structure_name(name): number
                for name, number in nvertices.items()
            }

        for name in list(self.nvertices.keys()):
            if name not in self.name:
                del self.nvertices[name]

        surface_mask = self.surface_mask
        if surface_mask.all():
            self.affine = None
            self.volume_shape = None
        else:
            if affine is None or volume_shape is None:
                raise ValueError(
                    'Affine and volume shape should be defined '
                    'for BrainModelAxis containing voxels'
                )
            self.affine = np.asanyarray(affine)
            self.volume_shape = volume_shape

        if np.any(self.vertex[surface_mask] < 0):
            raise ValueError('Undefined vertex indices found for surface elements')
        if np.any(self.voxel[~surface_mask] < 0):
            raise ValueError('Undefined voxel indices found for volumetric elements')

        for check_name in ('name', 'voxel', 'vertex'):
            shape = (self.size, 3) if check_name == 'voxel' else (self.size,)
            if getattr(self, check_name).shape != shape:
                raise ValueError(
                    f'Input {check_name} has incorrect shape '
                    f'({getattr(self, check_name).shape}) for BrainModelAxis axis'
                )

    @classmethod
    def from_mask(cls, mask, name='other', affine=None):
        """
        Creates a new BrainModelAxis axis describing the provided mask

        Parameters
        ----------
        mask : array_like
            all non-zero voxels will be included in the BrainModelAxis axis
            should be (Nx, Ny, Nz) array for volume mask or (Nvertex, ) array for surface mask
        name : str, optional
            Name of the brain structure (e.g. 'CortexRight', 'thalamus_left' or 'brain_stem')
        affine : array_like, optional
            (4, 4) array with the voxel to mm transformation (defaults to identity matrix)
            Argument will be ignored for surface masks

        Returns
        -------
        BrainModelAxis which covers the provided mask
        """
        if affine is None:
            affine = np.eye(4)
        else:
            affine = np.asanyarray(affine)
        if affine.shape != (4, 4):
            raise ValueError(
                f'Affine transformation should be a 4x4 array or None, not {affine!r}'
            )

        mask = np.asanyarray(mask)
        if mask.ndim == 1:
            return cls.from_surface(np.where(mask != 0)[0], mask.size, name=name)
        elif mask.ndim == 3:
            voxels = np.array(np.where(mask != 0)).T
            return cls(name, voxel=voxels, affine=affine, volume_shape=mask.shape)
        else:
            raise ValueError(
                'Mask should be either 1-dimensional (for surfaces) or '
                f'3-dimensional (for volumes), not {mask.ndim}-dimensional'
            )

    @classmethod
    def from_surface(cls, vertices, nvertex, name='Other'):
        """
        Creates a new BrainModelAxis axis describing the vertices on a surface

        Parameters
        ----------
        vertices : array_like
            indices of the vertices on the surface
        nvertex : int
            total number of vertices on the surface
        name : str
            Name of the brain structure (e.g. 'CortexLeft' or 'CortexRight')

        Returns
        -------
        BrainModelAxis which covers (part of) the surface
        """
        cifti_name = cls.to_cifti_brain_structure_name(name)
        return cls(cifti_name, vertex=vertices, nvertices={cifti_name: nvertex})

    @classmethod
    def from_index_mapping(cls, mim):
        """
        Creates a new BrainModel axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`.cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        BrainModelAxis
        """
        nbm = sum(bm.index_count for bm in mim.brain_models)
        voxel = np.full((nbm, 3), fill_value=-1, dtype=int)
        vertex = np.full(nbm, fill_value=-1, dtype=int)
        name = []

        nvertices = {}
        affine, shape = None, None
        for bm in mim.brain_models:
            index_end = bm.index_offset + bm.index_count
            is_surface = bm.model_type == 'CIFTI_MODEL_TYPE_SURFACE'
            name.extend([bm.brain_structure] * bm.index_count)
            if is_surface:
                vertex[bm.index_offset : index_end] = bm.vertex_indices
                nvertices[bm.brain_structure] = bm.surface_number_of_vertices
            else:
                voxel[bm.index_offset : index_end, :] = bm.voxel_indices_ijk
                if affine is None:
                    shape = mim.volume.volume_dimensions
                    affine = mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        return cls(name, voxel, vertex, affine, shape, nvertices)

    def to_mapping(self, dim):
        """
        Converts the brain model axis to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`.cifti2.Cifti2MatrixIndicesMap`
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
        for name, to_slice, bm in self.iter_structures():
            is_surface = name in self.nvertices.keys()
            if is_surface:
                voxels = None
                vertices = cifti2.Cifti2VertexIndices(bm.vertex)
                nvertex = self.nvertices[name]
            else:
                voxels = cifti2.Cifti2VoxelIndicesIJK(bm.voxel)
                vertices = None
                nvertex = None
                if mim.volume is None:
                    affine = cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, self.affine)
                    mim.volume = cifti2.Cifti2Volume(self.volume_shape, affine)
            cifti_bm = cifti2.Cifti2BrainModel(
                to_slice.start,
                len(bm),
                'CIFTI_MODEL_TYPE_SURFACE' if is_surface else 'CIFTI_MODEL_TYPE_VOXELS',
                name,
                nvertex,
                voxels,
                vertices,
            )
            mim.append(cifti_bm)
        return mim

    def iter_structures(self):
        """
        Iterates over all brain structures in the order that they appear along the axis

        Yields
        ------
        tuple with 3 elements:
        - CIFTI-2 brain structure name
        - slice to select the data associated with the brain structure from the tensor
        - brain model covering that specific brain structure
        """
        idx_start = 0
        start_name = self.name[idx_start]
        for idx_current, name in enumerate(self.name):
            if start_name != name:
                yield start_name, slice(idx_start, idx_current), self[idx_start:idx_current]
                idx_start = idx_current
                start_name = self.name[idx_start]
        yield start_name, slice(idx_start, None), self[idx_start:]

    @staticmethod
    def to_cifti_brain_structure_name(name):
        """
        Attempts to convert the name of an anatomical region in a format recognized by CIFTI-2

        This function returns:

        - the name if it is in the CIFTI-2 format already
        - if the name is a tuple the first element is assumed to be the structure name while
          the second is assumed to be the hemisphere (left, right or both). The latter will default
          to both.
        - names like left_cortex, cortex_left, LeftCortex, or CortexLeft will be converted to
          CIFTI_STRUCTURE_CORTEX_LEFT

        see :py:func:`nibabel.cifti2.tests.test_name` for examples of
        which conversions are possible

        Parameters
        ----------
        name: iterable of 2-element tuples of integer and string
            input name of an anatomical region

        Returns
        -------
        CIFTI-2 compatible name

        Raises
        ------
        ValueError: raised if the input name does not match a known anatomical structure in CIFTI-2
        """
        if name in cifti2.CIFTI_BRAIN_STRUCTURES:
            return cifti2.CIFTI_BRAIN_STRUCTURES.ciftiname[name]
        if not isinstance(name, str):
            if len(name) == 1:
                structure = name[0]
                orientation = 'both'
            else:
                structure, orientation = name
                if structure.lower() in ('left', 'right', 'both'):
                    orientation, structure = name
        else:
            orient_names = ('left', 'right', 'both')
            for poss_orient in orient_names:
                idx = len(poss_orient)
                if poss_orient == name.lower()[:idx]:
                    orientation = poss_orient
                    if name[idx] in '_ ':
                        structure = name[idx + 1 :]
                    else:
                        structure = name[idx:]
                    break
                if poss_orient == name.lower()[-idx:]:
                    orientation = poss_orient
                    if name[-idx - 1] in '_ ':
                        structure = name[: -idx - 1]
                    else:
                        structure = name[:-idx]
                    break
            else:
                orientation = 'both'
                structure = name
        if orientation.lower() == 'both':
            proposed_name = f'CIFTI_STRUCTURE_{structure.upper()}'
        else:
            proposed_name = f'CIFTI_STRUCTURE_{structure.upper()}_{orientation.upper()}'
        if proposed_name not in cifti2.CIFTI_BRAIN_STRUCTURES.ciftiname:
            raise ValueError(
                f'{name} was interpreted as {proposed_name}, '
                'which is not a valid CIFTI brain structure'
            )
        return proposed_name

    @property
    def surface_mask(self):
        """
        (N, ) boolean array which is true for any element on the surface
        """
        return np.vectorize(lambda name: name in self.nvertices.keys())(self.name)

    @property
    def volume_mask(self):
        """
        (N, ) boolean array which is true for any element on the surface
        """
        return np.vectorize(lambda name: name not in self.nvertices.keys())(self.name)

    _affine = None

    @property
    def affine(self):
        """
        Affine of the volumetric image in which the greyordinate voxels were defined
        """
        return self._affine

    @affine.setter
    def affine(self, value):
        if value is not None:
            value = np.asanyarray(value)
            if value.shape != (4, 4):
                raise ValueError('Affine transformation should be a 4x4 array')
        self._affine = value

    _volume_shape = None

    @property
    def volume_shape(self):
        """
        Shape of the volumetric image in which the greyordinate voxels were defined
        """
        return self._volume_shape

    @volume_shape.setter
    def volume_shape(self, value):
        if value is not None:
            value = tuple(value)
            if len(value) != 3:
                raise ValueError('Volume shape should be a tuple of length 3')
            if not all(isinstance(v, int) for v in value):
                raise ValueError('All elements of the volume shape should be integers')
        self._volume_shape = value

    _name = None

    @property
    def name(self):
        """The brain structure to which the voxel/vertices of belong"""
        return self._name

    @name.setter
    def name(self, values):
        self._name = np.array([self.to_cifti_brain_structure_name(name) for name in values])

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        if not isinstance(other, BrainModelAxis) or len(self) != len(other):
            return False
        if xor(self.affine is None, other.affine is None):
            return False
        return (
            (
                self.affine is None
                or np.allclose(self.affine, other.affine)
                and self.volume_shape == other.volume_shape
            )
            and self.nvertices == other.nvertices
            and np.array_equal(self.name, other.name)
            and np.array_equal(self.voxel[self.volume_mask], other.voxel[other.volume_mask])
            and np.array_equal(self.vertex[self.surface_mask], other.vertex[other.surface_mask])
        )

    def __add__(self, other):
        """
        Concatenates two BrainModels

        Parameters
        ----------
        other : BrainModelAxis
            brain model to be appended to the current one

        Returns
        -------
        BrainModelAxis
        """
        if not isinstance(other, BrainModelAxis):
            return NotImplemented
        if self.affine is None:
            affine, shape = other.affine, other.volume_shape
        else:
            affine, shape = self.affine, self.volume_shape
            if other.affine is not None and (
                not np.allclose(other.affine, affine) or other.volume_shape != shape
            ):
                raise ValueError(
                    'Trying to concatenate two BrainModels defined in a different brain volume'
                )

        nvertices = dict(self.nvertices)
        for name, value in other.nvertices.items():
            if name in nvertices.keys() and nvertices[name] != value:
                raise ValueError(
                    'Trying to concatenate two BrainModels with '
                    f'inconsistent number of vertices for {name}'
                )
            nvertices[name] = value
        return self.__class__(
            np.append(self.name, other.name),
            np.concatenate((self.voxel, other.voxel), 0),
            np.append(self.vertex, other.vertex),
            affine,
            shape,
            nvertices,
        )

    def __getitem__(self, item):
        """
        Extracts part of the brain structure

        Parameters
        ----------
        item : anything that can index a 1D array

        Returns
        -------
        If `item` is an integer returns a tuple with 3 elements:
        - boolean, which is True if it is a surface element
        - vertex index if it is a surface element, otherwise array with 3 voxel indices
        - structure.BrainStructure object describing the brain structure the element was taken from

        Otherwise returns a new BrainModelAxis
        """
        if isinstance(item, int):
            return self.get_element(item)
        if isinstance(item, str):
            raise IndexError('Can not index an Axis with a string (except for ParcelsAxis)')
        return self.__class__(
            self.name[item],
            self.voxel[item],
            self.vertex[item],
            self.affine,
            self.volume_shape,
            self.nvertices,
        )

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 3 elements
        - str, 'CIFTI_MODEL_TYPE_SURFACE' for vertex or 'CIFTI_MODEL_TYPE_VOXELS' for voxel
        - vertex index if it is a surface element, otherwise array with 3 voxel indices
        - structure.BrainStructure object describing the brain structure the element was taken from
        """
        element_type = 'CIFTI_MODEL_TYPE_' + (
            'SURFACE' if self.name[index] in self.nvertices.keys() else 'VOXELS'
        )
        struct = self.vertex if 'SURFACE' in element_type else self.voxel
        return element_type, struct[index], self.name[index]


class ParcelsAxis(Axis):
    """
    Each row/column in the CIFTI-2 vector/matrix represents a parcel of voxels/vertices

    This Axis describes which parcel is represented by each row/column.

    Individual parcels can be accessed based on their name, using
    ``parcel = parcel_axis[name]``
    """

    def __init__(self, name, voxels, vertices, affine=None, volume_shape=None, nvertices=None):
        """
        Use of this constructor is not recommended. New ParcelsAxis axes can be constructed more
        easily from a sequence of BrainModelAxis axes using
        :py:meth:`~ParcelsAxis.from_brain_models`

        Parameters
        ----------
        name : array_like
            (N, ) string array with the parcel names
        voxels :  array_like
            (N, ) object array each containing a sequence of voxels.
            For each parcel the voxels are represented by a (M, 3) index array
        vertices :  array_like
            (N, ) object array each containing a sequence of vertices.
            For each parcel the vertices are represented by a mapping from brain structure name to
            (M, ) index array
        affine : array_like, optional
            (4, 4) array mapping voxel indices to mm space (not needed for CIFTI-2 files only
            covering the surface)
        volume_shape : tuple of three integers, optional
            shape of the volume in which the voxels were defined (not needed for CIFTI-2 files only
            covering the surface)
        nvertices : dict from string to integer, optional
            maps names of surface elements to integers (not needed for volumetric CIFTI-2 files)
        """
        self.name = np.asanyarray(name, dtype='U')
        self.voxels = np.empty(len(voxels), dtype='object')
        for idx, vox in enumerate(voxels):
            self.voxels[idx] = vox
        self.vertices = np.asanyarray(vertices, dtype='object')
        self.affine = np.asanyarray(affine) if affine is not None else None
        self.volume_shape = volume_shape
        if nvertices is None:
            self.nvertices = {}
        else:
            self.nvertices = {
                BrainModelAxis.to_cifti_brain_structure_name(name): number
                for name, number in nvertices.items()
            }

        for check_name in ('name', 'voxels', 'vertices'):
            if getattr(self, check_name).shape != (self.size,):
                raise ValueError(
                    f'Input {check_name} has incorrect shape '
                    f'({getattr(self, check_name).shape}) for Parcel axis'
                )

    @classmethod
    def from_brain_models(cls, named_brain_models):
        """
        Creates a Parcel axis from a list of BrainModelAxis axes with names

        Parameters
        ----------
        named_brain_models : iterable of 2-element tuples of string and BrainModelAxis
            list of (parcel name, brain model representation) pairs defining each parcel

        Returns
        -------
        ParcelsAxis
        """
        nparcels = len(named_brain_models)
        affine = None
        volume_shape = None
        all_names = []
        all_voxels = np.zeros(nparcels, dtype='object')
        all_vertices = np.zeros(nparcels, dtype='object')
        nvertices = {}
        for idx_parcel, (parcel_name, bm) in enumerate(named_brain_models):
            all_names.append(parcel_name)

            voxels = bm.voxel[bm.volume_mask]
            if voxels.shape[0] != 0:
                if affine is None:
                    affine = bm.affine
                    volume_shape = bm.volume_shape
                elif not np.allclose(affine, bm.affine) or (volume_shape != bm.volume_shape):
                    raise ValueError(
                        'Can not combine brain models defined in different '
                        'volumes into a single Parcel axis'
                    )
            all_voxels[idx_parcel] = voxels

            vertices = {}
            for name, _, bm_part in bm.iter_structures():
                if name in bm.nvertices.keys():
                    if name in nvertices.keys() and nvertices[name] != bm.nvertices[name]:
                        raise ValueError(
                            'Got multiple conflicting number of '
                            f'vertices for surface structure {name}'
                        )
                    nvertices[name] = bm.nvertices[name]
                    vertices[name] = bm_part.vertex
            all_vertices[idx_parcel] = vertices
        return ParcelsAxis(all_names, all_voxels, all_vertices, affine, volume_shape, nvertices)

    @classmethod
    def from_index_mapping(cls, mim):
        """
        Creates a new Parcels axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        ParcelsAxis
        """
        nparcels = len(list(mim.parcels))
        all_names = []
        all_voxels = np.zeros(nparcels, dtype='object')
        all_vertices = np.zeros(nparcels, dtype='object')

        volume_shape = None if mim.volume is None else mim.volume.volume_dimensions
        affine = None
        if mim.volume is not None:
            affine = mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        nvertices = {}
        for surface in mim.surfaces:
            nvertices[surface.brain_structure] = surface.surface_number_of_vertices
        for idx_parcel, parcel in enumerate(mim.parcels):
            nvoxels = 0 if parcel.voxel_indices_ijk is None else len(parcel.voxel_indices_ijk)
            voxels = np.zeros((nvoxels, 3), dtype='i4')
            if nvoxels != 0:
                voxels[:] = parcel.voxel_indices_ijk
            vertices = {}
            for vertex in parcel.vertices:
                name = vertex.brain_structure
                vertices[vertex.brain_structure] = np.array(vertex)
                if name not in nvertices.keys():
                    raise ValueError(
                        f'Number of vertices for surface structure {name} not defined'
                    )
            all_voxels[idx_parcel] = voxels
            all_vertices[idx_parcel] = vertices
            all_names.append(parcel.name)
        return cls(all_names, all_voxels, all_vertices, affine, volume_shape, nvertices)

    def to_mapping(self, dim):
        """
        Converts the Parcel to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`cifti2.Cifti2MatrixIndicesMap`
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_PARCELS')
        if self.affine is not None:
            affine = cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, matrix=self.affine)
            mim.volume = cifti2.Cifti2Volume(self.volume_shape, affine)
        for name, nvertex in self.nvertices.items():
            mim.append(cifti2.Cifti2Surface(name, nvertex))
        for name, voxels, vertices in zip(self.name, self.voxels, self.vertices):
            cifti_voxels = cifti2.Cifti2VoxelIndicesIJK(voxels)
            element = cifti2.Cifti2Parcel(name, cifti_voxels)
            for name_vertex, idx_vertices in vertices.items():
                element.vertices.append(cifti2.Cifti2Vertices(name_vertex, idx_vertices))
            mim.append(element)
        return mim

    _affine = None

    @property
    def affine(self):
        """
        Affine of the volumetric image in which the greyordinate voxels were defined
        """
        return self._affine

    @affine.setter
    def affine(self, value):
        if value is not None:
            value = np.asanyarray(value)
            if value.shape != (4, 4):
                raise ValueError('Affine transformation should be a 4x4 array')
        self._affine = value

    _volume_shape = None

    @property
    def volume_shape(self):
        """
        Shape of the volumetric image in which the greyordinate voxels were defined
        """
        return self._volume_shape

    @volume_shape.setter
    def volume_shape(self, value):
        if value is not None:
            value = tuple(value)
            if len(value) != 3:
                raise ValueError('Volume shape should be a tuple of length 3')
            if not all(isinstance(v, int) for v in value):
                raise ValueError('All elements of the volume shape should be integers')
        self._volume_shape = value

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        if (
            self.__class__ != other.__class__
            or len(self) != len(other)
            or not np.array_equal(self.name, other.name)
            or self.nvertices != other.nvertices
            or any(not np.array_equal(vox1, vox2) for vox1, vox2 in zip(self.voxels, other.voxels))
        ):
            return False
        if self.affine is not None:
            if (
                other.affine is None
                or not np.allclose(self.affine, other.affine)
                or self.volume_shape != other.volume_shape
            ):
                return False
        elif other.affine is not None:
            return False
        for vert1, vert2 in zip(self.vertices, other.vertices):
            if len(vert1) != len(vert2):
                return False
            for name in vert1.keys():
                if name not in vert2 or not np.array_equal(vert1[name], vert2[name]):
                    return False
        return True

    def __add__(self, other):
        """
        Concatenates two Parcels

        Parameters
        ----------
        other : ParcelsAxis
            parcel to be appended to the current one

        Returns
        -------
        Parcel
        """
        if not isinstance(other, ParcelsAxis):
            return NotImplemented
        if self.affine is None:
            affine, shape = other.affine, other.volume_shape
        else:
            affine, shape = self.affine, self.volume_shape
            if other.affine is not None and (
                not np.allclose(other.affine, affine) or other.volume_shape != shape
            ):
                raise ValueError(
                    'Trying to concatenate two ParcelsAxis defined in a different brain volume'
                )
        nvertices = dict(self.nvertices)
        for name, value in other.nvertices.items():
            if name in nvertices.keys() and nvertices[name] != value:
                raise ValueError(
                    'Trying to concatenate two ParcelsAxis with '
                    f'inconsistent number of vertices for {name}'
                )
            nvertices[name] = value
        return self.__class__(
            np.append(self.name, other.name),
            np.append(self.voxels, other.voxels),
            np.append(self.vertices, other.vertices),
            affine,
            shape,
            nvertices,
        )

    def __getitem__(self, item):
        """
        Extracts subset of the axes based on the type of ``item``:

        - `int`: 3-element tuple of (parcel name, parcel voxels, parcel vertices)
        - `string`: 2-element tuple of (parcel voxels, parcel vertices
        - other object that can index 1D arrays: new Parcel axis
        """
        if isinstance(item, str):
            idx = np.where(self.name == item)[0]
            if len(idx) == 0:
                raise IndexError(f'Parcel {item} not found')
            if len(idx) > 1:
                raise IndexError(f'Multiple parcels with name {item} found')
            return self.voxels[idx[0]], self.vertices[idx[0]]
        if isinstance(item, int):
            return self.get_element(item)
        return self.__class__(
            self.name[item],
            self.voxels[item],
            self.vertices[item],
            self.affine,
            self.volume_shape,
            self.nvertices,
        )

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 3 elements
        - unicode name of the parcel
        - (M, 3) int array with voxel indices
        - dict from string to (K, ) int array with vertex indices
          for a specific surface brain structure
        """
        return self.name[index], self.voxels[index], self.vertices[index]


class ScalarAxis(Axis):
    """
    Along this axis of the CIFTI-2 vector/matrix each row/column has been given
    a unique name and optionally metadata
    """

    def __init__(self, name, meta=None):
        """
        Parameters
        ----------
        name : array_like
            (N, ) string array with the parcel names
        meta :  array_like
            (N, ) object array with a dictionary of metadata for each row/column.
            Defaults to empty dictionary
        """
        self.name = np.asanyarray(name, dtype='U')
        if meta is None:
            meta = [{} for _ in range(self.name.size)]
        self.meta = np.asanyarray(meta, dtype='object')

        for check_name in ('name', 'meta'):
            if getattr(self, check_name).shape != (self.size,):
                raise ValueError(
                    f'Input {check_name} has incorrect shape '
                    f'({getattr(self, check_name).shape}) for ScalarAxis axis'
                )

    @classmethod
    def from_index_mapping(cls, mim):
        """
        Creates a new Scalar axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`.cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        ScalarAxis
        """
        names = [nm.map_name for nm in mim.named_maps]
        meta = [{} if nm.metadata is None else dict(nm.metadata) for nm in mim.named_maps]
        return cls(names, meta)

    def to_mapping(self, dim):
        """
        Converts the hcp_labels to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`.cifti2.Cifti2MatrixIndicesMap`
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_SCALARS')
        for name, meta in zip(self.name, self.meta):
            named_map = cifti2.Cifti2NamedMap(name, cifti2.Cifti2MetaData(meta))
            mim.append(named_map)
        return mim

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        """
        Compares two Scalars

        Parameters
        ----------
        other : ScalarAxis
            scalar axis to be compared

        Returns
        -------
        bool : False if type, length or content do not match
        """
        if not isinstance(other, ScalarAxis) or self.size != other.size:
            return False
        return np.array_equal(self.name, other.name) and np.array_equal(self.meta, other.meta)

    def __add__(self, other):
        """
        Concatenates two Scalars

        Parameters
        ----------
        other : ScalarAxis
            scalar axis to be appended to the current one

        Returns
        -------
        ScalarAxis
        """
        if not isinstance(other, ScalarAxis):
            return NotImplemented
        return ScalarAxis(
            np.append(self.name, other.name),
            np.append(self.meta, other.meta),
        )

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_element(item)
        return self.__class__(self.name[item], self.meta[item])

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 2 elements
        - unicode name of the row/column
        - dictionary with the element metadata
        """
        return self.name[index], self.meta[index]


class LabelAxis(Axis):
    """
    Defines CIFTI-2 axis for label array.

    Along this axis of the CIFTI-2 vector/matrix each row/column has been given a unique name,
    label table, and optionally metadata
    """

    def __init__(self, name, label, meta=None):
        """
        Parameters
        ----------
        name : array_like
            (N, ) string array with the parcel names
        label : array_like
            single dictionary or (N, ) object array with dictionaries mapping
            from integers to (name, (R, G, B, A)), where name is a string and R, G, B, and A are
            floats between 0 and 1 giving the colour and alpha (i.e., transparency)
        meta :  array_like, optional
            (N, ) object array with a dictionary of metadata for each row/column
        """
        self.name = np.asanyarray(name, dtype='U')
        if isinstance(label, dict):
            label = [label.copy() for _ in range(self.name.size)]
        self.label = np.asanyarray(label, dtype='object')
        if meta is None:
            meta = [{} for _ in range(self.name.size)]
        self.meta = np.asanyarray(meta, dtype='object')

        for check_name in ('name', 'meta', 'label'):
            if getattr(self, check_name).shape != (self.size,):
                raise ValueError(
                    f'Input {check_name} has incorrect shape '
                    f'({getattr(self, check_name).shape}) for LabelAxis axis'
                )

    @classmethod
    def from_index_mapping(cls, mim):
        """
        Creates a new Label axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`.cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        LabelAxis
        """
        tables = [
            {key: (value.label, value.rgba) for key, value in nm.label_table.items()}
            for nm in mim.named_maps
        ]
        rest = ScalarAxis.from_index_mapping(mim)
        return LabelAxis(rest.name, tables, rest.meta)

    def to_mapping(self, dim):
        """
        Converts the hcp_labels to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`.cifti2.Cifti2MatrixIndicesMap`
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_LABELS')
        for name, label, meta in zip(self.name, self.label, self.meta):
            label_table = cifti2.Cifti2LabelTable()
            for key, value in label.items():
                label_table[key] = (value[0],) + tuple(value[1])
            named_map = cifti2.Cifti2NamedMap(name, cifti2.Cifti2MetaData(meta), label_table)
            mim.append(named_map)
        return mim

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        """
        Compares two Labels

        Parameters
        ----------
        other : LabelAxis
            label axis to be compared

        Returns
        -------
        bool : False if type, length or content do not match
        """
        if not isinstance(other, LabelAxis) or self.size != other.size:
            return False
        return (
            np.array_equal(self.name, other.name)
            and np.array_equal(self.meta, other.meta)
            and np.array_equal(self.label, other.label)
        )

    def __add__(self, other):
        """
        Concatenates two Labels

        Parameters
        ----------
        other : LabelAxis
            label axis to be appended to the current one

        Returns
        -------
        LabelAxis
        """
        if not isinstance(other, LabelAxis):
            return NotImplemented
        return LabelAxis(
            np.append(self.name, other.name),
            np.append(self.label, other.label),
            np.append(self.meta, other.meta),
        )

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_element(item)
        return self.__class__(self.name[item], self.label[item], self.meta[item])

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 2 elements
        - unicode name of the row/column
        - dictionary with the label table
        - dictionary with the element metadata
        """
        return self.name[index], self.label[index], self.meta[index]


class SeriesAxis(Axis):
    """
    Along this axis of the CIFTI-2 vector/matrix the rows/columns increase monotonously in time

    This Axis describes the time point of each row/column.
    """

    size = None

    def __init__(self, start, step, size, unit='SECOND'):
        """
        Creates a new SeriesAxis axis

        Parameters
        ----------
        start : float
            starting time point
        step :  float
            sampling time (TR)
        size : int
            number of time points
        unit : str
            Unit of the step size (one of 'second', 'hertz', 'meter', or 'radian')
        """
        self.unit = unit
        self.start = start
        self.step = step
        self.size = size

    @property
    def time(self):
        return np.arange(self.size) * self.step + self.start

    @classmethod
    def from_index_mapping(cls, mim):
        """
        Creates a new SeriesAxis axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`.cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        SeriesAxis
        """
        start = mim.series_start * 10**mim.series_exponent
        step = mim.series_step * 10**mim.series_exponent
        return cls(start, step, mim.number_of_series_points, mim.series_unit)

    def to_mapping(self, dim):
        """
        Converts the SeriesAxis to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`cifti2.Cifti2MatrixIndicesMap`
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_SERIES')
        mim.series_exponent = 0
        mim.series_start = self.start
        mim.series_step = self.step
        mim.number_of_series_points = self.size
        mim.series_unit = self.unit
        return mim

    _unit = None

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        if value.upper() not in ('SECOND', 'HERTZ', 'METER', 'RADIAN'):
            raise ValueError(
                'SeriesAxis unit should be one of ' + "('second', 'hertz', 'meter', or 'radian'"
            )
        self._unit = value.upper()

    def __len__(self):
        return self.size

    def __eq__(self, other):
        """
        True if start, step, size, and unit are the same.
        """
        return (
            isinstance(other, SeriesAxis)
            and self.start == other.start
            and self.step == other.step
            and self.size == other.size
            and self.unit == other.unit
        )

    def __add__(self, other):
        """
        Concatenates two SeriesAxis

        Parameters
        ----------
        other : SeriesAxis
            Time SeriesAxis to append at the end of the current time SeriesAxis.
            Note that the starting time of the other time SeriesAxis is ignored.

        Returns
        -------
        SeriesAxis
            New time SeriesAxis with the concatenation of the two

        Raises
        ------
        ValueError
            raised if the repetition time of the two time SeriesAxis is different
        """
        if isinstance(other, SeriesAxis):
            if other.step != self.step:
                raise ValueError('Can only concatenate SeriesAxis with the same step size')
            if other.unit != self.unit:
                raise ValueError('Can only concatenate SeriesAxis with the same unit')
            return SeriesAxis(self.start, self.step, self.size + other.size, self.unit)
        return NotImplemented

    def __getitem__(self, item):
        if isinstance(item, slice):
            step = 1 if item.step is None else item.step
            idx_start = (
                (self.size - 1 if step < 0 else 0)
                if item.start is None
                else (item.start if item.start >= 0 else self.size + item.start)
            )
            idx_end = (
                (-1 if step < 0 else self.size)
                if item.stop is None
                else (item.stop if item.stop >= 0 else self.size + item.stop)
            )
            if idx_start > self.size and step < 0:
                idx_start = self.size - 1
            if idx_end > self.size:
                idx_end = self.size
            nelements = (idx_end - idx_start) // step
            if nelements < 0:
                nelements = 0
            return SeriesAxis(
                idx_start * self.step + self.start, self.step * step, nelements, self.unit
            )
        elif isinstance(item, int):
            return self.get_element(item)
        raise IndexError(
            'SeriesAxis can only be indexed with integers or slices '
            'without breaking the regular structure'
        )

    def get_element(self, index):
        """
        Gives the time point of a specific row/column

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        float
        """
        original_index = index
        if index < 0:
            index = self.size + index
        if index >= self.size or index < 0:
            raise IndexError(
                f'index {original_index} is out of range for SeriesAxis with size {self.size}'
            )
        return self.start + self.step * index
