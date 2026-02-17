# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read / write access to CIFTI-2 image format

Format of the NIFTI2 container format described here:

    http://www.nitrc.org/forum/message.php?msg_id=3738

Definition of the CIFTI-2 header format and file extensions can be found at:

    http://www.nitrc.org/projects/cifti
"""

import re
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping, MutableSequence
from warnings import warn

import numpy as np

from .. import xmlutils as xml
from ..arrayproxy import reshape_dataobj
from ..caret import CaretMetaData
from ..dataobj_images import DataobjImage
from ..filebasedimages import FileBasedHeader, SerializableImage
from ..nifti1 import Nifti1Extensions
from ..nifti2 import Nifti2Header, Nifti2Image
from ..volumeutils import Recoder, make_dt_codes


def _float_01(val):
    out = float(val)
    if out < 0 or out > 1:
        raise ValueError('Float must be between 0 and 1 inclusive')
    return out


class Cifti2HeaderError(Exception):
    """Error in CIFTI-2 header"""


_dtdefs = (  # code, label, dtype definition, niistring
    (2, 'uint8', np.uint8, 'NIFTI_TYPE_UINT8'),
    (4, 'int16', np.int16, 'NIFTI_TYPE_INT16'),
    (8, 'int32', np.int32, 'NIFTI_TYPE_INT32'),
    (16, 'float32', np.float32, 'NIFTI_TYPE_FLOAT32'),
    (64, 'float64', np.float64, 'NIFTI_TYPE_FLOAT64'),
    (256, 'int8', np.int8, 'NIFTI_TYPE_INT8'),
    (512, 'uint16', np.uint16, 'NIFTI_TYPE_UINT16'),
    (768, 'uint32', np.uint32, 'NIFTI_TYPE_UINT32'),
    (1024, 'int64', np.int64, 'NIFTI_TYPE_INT64'),
    (1280, 'uint64', np.uint64, 'NIFTI_TYPE_UINT64'),
)

# Make full code alias bank, including dtype column
data_type_codes = make_dt_codes(_dtdefs)

CIFTI_MAP_TYPES = (
    'CIFTI_INDEX_TYPE_BRAIN_MODELS',
    'CIFTI_INDEX_TYPE_PARCELS',
    'CIFTI_INDEX_TYPE_SERIES',
    'CIFTI_INDEX_TYPE_SCALARS',
    'CIFTI_INDEX_TYPE_LABELS',
)

CIFTI_MODEL_TYPES = (
    'CIFTI_MODEL_TYPE_SURFACE',  # Modeled using surface vertices
    'CIFTI_MODEL_TYPE_VOXELS',  # Modeled using voxels.
)

CIFTI_SERIESUNIT_TYPES = (
    'SECOND',
    'HERTZ',
    'METER',
    'RADIAN',
)


def _full_structure(struct: str):
    """Expands STRUCT_NAME into:

    STRUCT_NAME, CIFTI_STRUCTURE_STRUCT_NAME, StructName
    """
    return (
        struct,
        f'CIFTI_STRUCTURE_{struct}',
        ''.join(word.capitalize() for word in struct.split('_')),
    )


CIFTI_BRAIN_STRUCTURES = Recoder(
    (
        # For simplicity of comparison, use the ordering from:
        # https://github.com/Washington-University/workbench/blob/b985f5d/src/Common/StructureEnum.cxx
        # (name,          ciftiname,                     guiname)
        # ('CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_LEFT', 'CortexLeft')
        _full_structure('CORTEX_LEFT'),
        _full_structure('CORTEX_RIGHT'),
        _full_structure('CEREBELLUM'),
        _full_structure('ACCUMBENS_LEFT'),
        _full_structure('ACCUMBENS_RIGHT'),
        _full_structure('ALL'),
        _full_structure('ALL_GREY_MATTER'),
        _full_structure('ALL_WHITE_MATTER'),
        _full_structure('AMYGDALA_LEFT'),
        _full_structure('AMYGDALA_RIGHT'),
        _full_structure('BRAIN_STEM'),
        _full_structure('CAUDATE_LEFT'),
        _full_structure('CAUDATE_RIGHT'),
        _full_structure('CEREBELLAR_WHITE_MATTER_LEFT'),
        _full_structure('CEREBELLAR_WHITE_MATTER_RIGHT'),
        _full_structure('CEREBELLUM_LEFT'),
        _full_structure('CEREBELLUM_RIGHT'),
        _full_structure('CEREBRAL_WHITE_MATTER_LEFT'),
        _full_structure('CEREBRAL_WHITE_MATTER_RIGHT'),
        _full_structure('CORTEX'),
        _full_structure('DIENCEPHALON_VENTRAL_LEFT'),
        _full_structure('DIENCEPHALON_VENTRAL_RIGHT'),
        _full_structure('HIPPOCAMPUS_LEFT'),
        _full_structure('HIPPOCAMPUS_RIGHT'),
        _full_structure('INVALID'),
        _full_structure('OTHER'),
        _full_structure('OTHER_GREY_MATTER'),
        _full_structure('OTHER_WHITE_MATTER'),
        _full_structure('PALLIDUM_LEFT'),
        _full_structure('PALLIDUM_RIGHT'),
        _full_structure('PUTAMEN_LEFT'),
        _full_structure('PUTAMEN_RIGHT'),
        ## Also commented out in connectome_wb; unclear if deprecated, planned, or what
        # _full_structure("SUBCORTICAL_WHITE_MATTER_LEFT")
        # _full_structure("SUBCORTICAL_WHITE_MATTER_RIGHT")
        _full_structure('THALAMUS_LEFT'),
        _full_structure('THALAMUS_RIGHT'),
    ),
    fields=('name', 'ciftiname', 'guiname'),
)


def _value_if_klass(val, klass):
    if val is None or isinstance(val, klass):
        return val
    raise ValueError(f'Not a valid {klass.__name__} instance.')


def _underscore(string):
    """Convert a string from CamelCase to underscored"""
    string = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', string)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', string).lower()


class LimitedNifti2Header(Nifti2Header):
    _data_type_codes = data_type_codes


class Cifti2MetaData(CaretMetaData):
    """A list of name-value pairs

    * Description - Provides a simple method for user-supplied metadata that
      associates names with values.
    * Attributes: [NA]
    * Child Elements

        * MD (0...N)

    * Text Content: [NA]
    * Parent Elements - Matrix, NamedMap

    MD elements are a single metadata entry consisting of a name and a value.

    Attributes
    ----------
    data : list of (name, value) tuples
    """

    @staticmethod
    def _sanitize(args, kwargs):
        """Sanitize and warn on deprecated arguments

        Accept metadata positional/keyword argument that can take
        ``None`` to indicate no initialization.

        >>> import pytest
        >>> Cifti2MetaData()
        <Cifti2MetaData {}>
        >>> Cifti2MetaData([("key", "val")])
        <Cifti2MetaData {'key': 'val'}>
        >>> Cifti2MetaData(key="val")
        <Cifti2MetaData {'key': 'val'}>
        >>> with pytest.warns(FutureWarning):
        ...     Cifti2MetaData(None)
        <Cifti2MetaData {}>
        >>> with pytest.warns(FutureWarning):
        ...     Cifti2MetaData(metadata=None)
        <Cifti2MetaData {}>
        >>> with pytest.warns(FutureWarning):
        ...     Cifti2MetaData(metadata={'key': 'val'})
        <Cifti2MetaData {'key': 'val'}>

        Note that "metadata" could be a valid key:

        >>> Cifti2MetaData(metadata='val')
        <Cifti2MetaData {'metadata': 'val'}>
        """
        if not args and list(kwargs) == ['metadata']:
            if not isinstance(kwargs['metadata'], str):
                warn(
                    'Cifti2MetaData now has a dict-like interface and will '
                    'no longer accept the ``metadata`` keyword argument in '
                    'NiBabel 6.0. See ``pydoc dict`` for initialization options.',
                    FutureWarning,
                    stacklevel=3,
                )
                md = kwargs.pop('metadata')
                if md is not None:
                    args = (md,)
        if args == (None,):
            warn(
                'Cifti2MetaData now has a dict-like interface and will no longer '
                'accept the positional argument ``None`` in NiBabel 6.0. '
                'See ``pydoc dict`` for initialization options.',
                FutureWarning,
                stacklevel=3,
            )
            args = ()
        return args, kwargs

    @property
    def data(self):
        return self._data

    def difference_update(self, metadata):
        """Remove metadata key-value pairs

        Parameters
        ----------
        metadata : dict-like datatype

        Returns
        -------
        None

        """
        if metadata is None:
            raise ValueError("The metadata parameter can't be None")
        pairs = dict(metadata)
        for k in pairs:
            del self.data[k]


class Cifti2LabelTable(xml.XmlSerializable, MutableMapping):
    r"""CIFTI-2 label table: a sequence of ``Cifti2Label``\s

    * Description - Used by NamedMap when IndicesMapToDataType is
      "CIFTI_INDEX_TYPE_LABELS" in order to associate names and display colors
      with label keys. Note that LABELS is the only mapping type that uses a
      LabelTable. Display coloring of continuous-valued data is not specified
      by CIFTI-2.
    * Attributes: [NA]
    * Child Elements

        * Label (0...N)

    * Text Content: [NA]
    * Parent Element - NamedMap
    """

    def __init__(self):
        self._labels = OrderedDict()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, key):
        return self._labels[key]

    def append(self, label):
        self[label.key] = label

    def __setitem__(self, key, value):
        if isinstance(value, Cifti2Label):
            if key != value.key:
                raise ValueError("The key and the label's key must agree")
            self._labels[key] = value
            return
        if len(value) != 5:
            raise ValueError('Value should be length 5')
        try:
            self._labels[key] = Cifti2Label(*([key] + list(value)))
        except ValueError:
            raise ValueError(
                'Key should be int, value should be sequence '
                'of str and 4 floats between 0 and 1'
            )

    def __delitem__(self, key):
        del self._labels[key]

    def __iter__(self):
        return iter(self._labels)

    def _to_xml_element(self):
        if len(self) == 0:
            raise Cifti2HeaderError('LabelTable element requires at least 1 label')
        labeltable = xml.Element('LabelTable')
        for ele in self._labels.values():
            labeltable.append(ele._to_xml_element())
        return labeltable


class Cifti2Label(xml.XmlSerializable):
    """CIFTI-2 label: association of integer key with a name and RGBA values

    For all color components, value is floating point with range 0.0 to 1.0.

    * Description - Associates a label key value with a name and a display
      color.
    * Attributes

        * Key - Integer, data value which is assigned this name and color.
        * Red - Red color component for label. Value is floating point with
          range 0.0 to 1.0.
        * Green - Green color component for label. Value is floating point with
          range 0.0 to 1.0.
        * Blue - Blue color component for label. Value is floating point with
          range 0.0 to 1.0.
        * Alpha - Alpha color component for label. Value is floating point with
          range 0.0 to 1.0.

    * Child Elements: [NA]
    * Text Content - Name of the label.
    * Parent Element - LabelTable

    Attributes
    ----------
    key : int, optional
        Integer, data value which is assigned this name and color.
    label : str, optional
        Name of the label.
    red : float, optional
        Red color component for label (between 0 and 1).
    green : float, optional
        Green color component for label (between 0 and 1).
    blue : float, optional
        Blue color component for label (between 0 and 1).
    alpha : float, optional
        Alpha color component for label (between 0 and 1).
    """

    def __init__(self, key=0, label='', red=0.0, green=0.0, blue=0.0, alpha=0.0):
        self.key = int(key)
        self.label = str(label)
        self.red = _float_01(red)
        self.green = _float_01(green)
        self.blue = _float_01(blue)
        self.alpha = _float_01(alpha)

    @property
    def rgba(self):
        """Returns RGBA as tuple"""
        return (self.red, self.green, self.blue, self.alpha)

    def _to_xml_element(self):
        if self.label == '':
            raise Cifti2HeaderError('Label needs a name')
        try:
            v = int(self.key)
        except ValueError:
            raise Cifti2HeaderError('The key must be an integer')
        for c_ in ('red', 'blue', 'green', 'alpha'):
            try:
                v = _float_01(getattr(self, c_))
            except ValueError:
                raise Cifti2HeaderError(
                    f'Label invalid {c_} needs to be a float between 0 and 1. and it is {v}'
                )

        lab = xml.Element('Label')
        lab.attrib['Key'] = str(self.key)
        lab.text = str(self.label)

        for name in ('red', 'green', 'blue', 'alpha'):
            val = getattr(self, name)
            attr = '0' if val == 0 else '1' if val == 1 else str(val)
            lab.attrib[name.capitalize()] = attr
        return lab


class Cifti2NamedMap(xml.XmlSerializable):
    """CIFTI-2 named map: association of name and optional data with a map index

    Associates a name, optional metadata, and possibly a LabelTable with an
    index in a map.

    * Description - Associates a name, optional metadata, and possibly a
      LabelTable with an index in a map.
    * Attributes: [NA]
    * Child Elements

        * MapName (1)
        * LabelTable (0...1)
        * MetaData (0...1)

    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    Attributes
    ----------
    map_name : str
        Name of map
    metadata : None or Cifti2MetaData
        Metadata associated with named map
    label_table : None or Cifti2LabelTable
        Label table associated with named map
    """

    def __init__(self, map_name=None, metadata=None, label_table=None):
        self.map_name = map_name
        self.metadata = metadata
        self.label_table = label_table

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """Set the metadata for this NamedMap

        Parameters
        ----------
        meta : Cifti2MetaData

        Returns
        -------
        None
        """
        self._metadata = _value_if_klass(metadata, Cifti2MetaData)

    @property
    def label_table(self):
        return self._label_table

    @label_table.setter
    def label_table(self, label_table):
        """Set the label_table for this NamedMap

        Parameters
        ----------
        label_table : Cifti2LabelTable

        Returns
        -------
        None
        """
        self._label_table = _value_if_klass(label_table, Cifti2LabelTable)

    def _to_xml_element(self):
        named_map = xml.Element('NamedMap')
        if self.metadata:
            named_map.append(self.metadata._to_xml_element())
        if self.label_table:
            named_map.append(self.label_table._to_xml_element())
        map_name = xml.SubElement(named_map, 'MapName')
        map_name.text = self.map_name
        return named_map


class Cifti2Surface(xml.XmlSerializable):
    """Cifti surface: association of brain structure and number of vertices

    * Description - Specifies the number of vertices for a surface, when
      IndicesMapToDataType is "CIFTI_INDEX_TYPE_PARCELS." This is separate from
      the Parcel element because there can be multiple parcels on one surface,
      and one parcel may involve multiple surfaces.
    * Attributes

        * BrainStructure - A string from the BrainStructure list to identify
          what surface structure this element refers to (usually left cortex,
          right cortex, or cerebellum).
        * SurfaceNumberOfVertices - The number of vertices that this
          structure's surface contains.

    * Child Elements: [NA]
    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    Attributes
    ----------
    brain_structure : str
        Name of brain structure
    surface_number_of_vertices : int
        Number of vertices on surface
    """

    def __init__(self, brain_structure=None, surface_number_of_vertices=None):
        self.brain_structure = brain_structure
        self.surface_number_of_vertices = surface_number_of_vertices

    def _to_xml_element(self):
        if self.brain_structure is None:
            raise Cifti2HeaderError('Surface element requires at least 1 BrainStructure')
        surf = xml.Element('Surface')
        surf.attrib['BrainStructure'] = str(self.brain_structure)
        surf.attrib['SurfaceNumberOfVertices'] = str(self.surface_number_of_vertices)
        return surf


class Cifti2VoxelIndicesIJK(xml.XmlSerializable, MutableSequence):
    """CIFTI-2 VoxelIndicesIJK: Set of voxel indices contained in a structure

    * Description - Identifies the voxels that model a brain structure, or
      participate in a parcel. Note that when this is a child of BrainModel,
      the IndexCount attribute of the BrainModel indicates the number of voxels
      contained in this element.
    * Attributes: [NA]
    * Child Elements: [NA]
    * Text Content - IJK indices (which are zero-based) of each voxel in this
      brain model or parcel, with each index separated by a whitespace
      character. There are three indices per voxel.  If the parent element is
      BrainModel, then the BrainModel element's IndexCount attribute indicates
      the number of triplets (IJK indices) in this element's content.
    * Parent Elements - BrainModel, Parcel

    Each element of this sequence is a triple of integers.
    """

    def __init__(self, indices=None):
        self._indices = []
        if indices is not None:
            self.extend(indices)

    def __len__(self):
        return len(self._indices)

    def __delitem__(self, index):
        if not isinstance(index, int) and len(index) > 1:
            raise NotImplementedError
        del self._indices[index]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._indices[index]
        elif len(index) == 2:
            if not isinstance(index[0], int):
                raise NotImplementedError
            return self._indices[index[0]][index[1]]
        else:
            raise ValueError('Only row and row,column access is allowed')

    def __setitem__(self, index, value):
        if isinstance(index, int):
            try:
                value = [int(v) for v in value]
                if len(value) != 3:
                    raise ValueError('rows are triples of ints')
                self._indices[index] = value
            except ValueError:
                raise ValueError('value must be a triple of ints')
        elif len(index) == 2:
            try:
                if not isinstance(index[0], int):
                    raise NotImplementedError
                value = int(value)
                self._indices[index[0]][index[1]] = value
            except ValueError:
                raise ValueError('value must be an int')
        else:
            raise ValueError

    def insert(self, index, value):
        if not isinstance(index, int) and len(index) != 1:
            raise ValueError('Only rows can be inserted')
        try:
            value = [int(v) for v in value]
            if len(value) != 3:
                raise ValueError
            self._indices.insert(index, value)
        except ValueError:
            raise ValueError('value must be a triple of int')

    def _to_xml_element(self):
        if len(self) == 0:
            raise Cifti2HeaderError('VoxelIndicesIJK element require an index table')

        vox_ind = xml.Element('VoxelIndicesIJK')
        vox_ind.text = '\n'.join(' '.join([str(v) for v in row]) for row in self._indices)
        return vox_ind


class Cifti2Vertices(xml.XmlSerializable, MutableSequence):
    """CIFTI-2 vertices - association of brain structure and a list of vertices

    * Description - Contains a BrainStructure type and a list of vertex indices
      within a Parcel.
    * Attributes

        * BrainStructure - A string from the BrainStructure list to identify
          what surface this vertex list is from (usually left cortex, right
          cortex, or cerebellum).

    * Child Elements: [NA]
    * Text Content - Vertex indices (which are independent for each surface,
      and zero-based) separated by whitespace characters.
    * Parent Element - Parcel

    The class behaves like a list of Vertex indices (which are independent for
    each surface, and zero-based)

    Attributes
    ----------
    brain_structure : str
        A string from the BrainStructure list to identify what surface this
        vertex list is from (usually left cortex, right cortex, or cerebellum).
    """

    def __init__(self, brain_structure=None, vertices=None):
        self._vertices = []
        if vertices is not None:
            self.extend(vertices)

        self.brain_structure = brain_structure

    def __len__(self):
        return len(self._vertices)

    def __delitem__(self, index):
        del self._vertices[index]

    def __getitem__(self, index):
        return self._vertices[index]

    def __setitem__(self, index, value):
        try:
            value = int(value)
            self._vertices[index] = value
        except ValueError:
            raise ValueError('value must be an int')

    def insert(self, index, value):
        try:
            value = int(value)
            self._vertices.insert(index, value)
        except ValueError:
            raise ValueError('value must be an int')

    def _to_xml_element(self):
        if self.brain_structure is None:
            raise Cifti2HeaderError('Vertices element require a BrainStructure')

        vertices = xml.Element('Vertices')
        vertices.attrib['BrainStructure'] = str(self.brain_structure)

        vertices.text = ' '.join([str(i) for i in self])
        return vertices


class Cifti2Parcel(xml.XmlSerializable):
    """CIFTI-2 parcel: association of a name with vertices and/or voxels

    * Description - Associates a name, plus vertices and/or voxels, with an
      index.
    * Attributes

        * Name - The name of the parcel

    * Child Elements

        * Vertices (0...N)
        * VoxelIndicesIJK (0...1)

    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    Attributes
    ----------
    name : str
        Name of parcel
    voxel_indices_ijk : None or Cifti2VoxelIndicesIJK
        Voxel indices associated with parcel
    vertices : list of Cifti2Vertices
        Vertices associated with parcel
    """

    def __init__(self, name=None, voxel_indices_ijk=None, vertices=None):
        self.name = name
        self._voxel_indices_ijk = voxel_indices_ijk
        self.vertices = vertices if vertices is not None else []
        for val in self.vertices:
            if not isinstance(val, Cifti2Vertices):
                raise ValueError('Cifti2Parcel vertices must be instances of Cifti2Vertices')

    @property
    def voxel_indices_ijk(self):
        return self._voxel_indices_ijk

    @voxel_indices_ijk.setter
    def voxel_indices_ijk(self, value):
        self._voxel_indices_ijk = _value_if_klass(value, Cifti2VoxelIndicesIJK)

    def append_cifti_vertices(self, vertices):
        """Appends a Cifti2Vertices element to the Cifti2Parcel

        Parameters
        ----------
        vertices : Cifti2Vertices
        """
        if not isinstance(vertices, Cifti2Vertices):
            raise TypeError('Not a valid Cifti2Vertices instance')
        self.vertices.append(vertices)

    def pop_cifti2_vertices(self, ith):
        """Pops the ith vertices element from the Cifti2Parcel"""
        self.vertices.pop(ith)

    def _to_xml_element(self):
        if self.name is None:
            raise Cifti2HeaderError('Parcel element requires a name')

        parcel = xml.Element('Parcel')
        parcel.attrib['Name'] = str(self.name)
        if self.voxel_indices_ijk:
            parcel.append(self.voxel_indices_ijk._to_xml_element())
        for vertex in self.vertices:
            parcel.append(vertex._to_xml_element())
        return parcel


class Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(xml.XmlSerializable):
    """Matrix that translates voxel indices to spatial coordinates

    * Description - Contains a matrix that translates Voxel IJK Indices to
      spatial XYZ coordinates (+X=>right, +Y=>anterior, +Z=> superior). The
      resulting coordinate is the center of the voxel.
    * Attributes

        * MeterExponent - Integer, specifies that the coordinate result from
          the transformation matrix should be multiplied by 10 to this power to
          get the spatial coordinates in meters (e.g., if this is "-3", then
          the transformation matrix is in millimeters).

    * Child Elements: [NA]
    * Text Content - Sixteen floating-point values, in row-major order, that
      form a 4x4 homogeneous transformation matrix.
    * Parent Element - Volume

    Attributes
    ----------
    meter_exponent : int
        See attribute description above.
    matrix : array-like shape (4, 4)
        Affine transformation matrix from voxel indices to RAS space.
    """

    # meterExponent = int
    # matrix = np.array

    def __init__(self, meter_exponent=None, matrix=None):
        self.meter_exponent = meter_exponent
        self.matrix = matrix

    def _to_xml_element(self):
        if self.matrix is None:
            raise Cifti2HeaderError(
                'TransformationMatrixVoxelIndicesIJKtoXYZ element requires a matrix'
            )
        trans = xml.Element('TransformationMatrixVoxelIndicesIJKtoXYZ')
        trans.attrib['MeterExponent'] = str(self.meter_exponent)
        trans.text = '\n'.join(' '.join(map('{:.10f}'.format, row)) for row in self.matrix)
        return trans


class Cifti2Volume(xml.XmlSerializable):
    """CIFTI-2 volume: information about a volume for mappings that use voxels

    * Description - Provides information about the volume for any mappings that
      use voxels.
    * Attributes

        * VolumeDimensions - Three integer values separated by commas, the
          lengths of the three volume file dimensions that are related to
          spatial coordinates, in number of voxels. Voxel indices (which are
          zero-based) that are used in the mapping that this element applies to
          must be within these dimensions.

    * Child Elements

        * TransformationMatrixVoxelIndicesIJKtoXYZ (1)

    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    Attributes
    ----------
    volume_dimensions : array-like shape (3,)
        See attribute description above.
    transformation_matrix_voxel_indices_ijk_to_xyz \
        : Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ
        Matrix that translates voxel indices to spatial coordinates
    """

    def __init__(self, volume_dimensions=None, transform_matrix=None):
        self.volume_dimensions = volume_dimensions
        self.transformation_matrix_voxel_indices_ijk_to_xyz = transform_matrix

    def _to_xml_element(self):
        if self.volume_dimensions is None:
            raise Cifti2HeaderError('Volume element requires dimensions')

        volume = xml.Element('Volume')
        volume.attrib['VolumeDimensions'] = ','.join([str(val) for val in self.volume_dimensions])
        volume.append(self.transformation_matrix_voxel_indices_ijk_to_xyz._to_xml_element())
        return volume


class Cifti2VertexIndices(xml.XmlSerializable, MutableSequence):
    """CIFTI-2 vertex indices: vertex indices for an associated brain model

    The vertex indices (which are independent for each surface, and
    zero-based) that are used in this brain model[.] The parent
    BrainModel's ``index_count`` indicates the number of indices.

    * Description - Contains a list of vertex indices for a BrainModel with
      ModelType equal to CIFTI_MODEL_TYPE_SURFACE.
    * Attributes: [NA]
    * Child Elements: [NA]
    * Text Content - The vertex indices (which are independent for each
      surface, and zero-based) that are used in this brain model, with each
      index separated by a whitespace character.  The parent BrainModel's
      IndexCount attribute indicates the number of indices in this element's
      content.
    * Parent Element - BrainModel
    """

    def __init__(self, indices=None):
        self._indices = []
        if indices is not None:
            self.extend(indices)

    def __len__(self):
        return len(self._indices)

    def __delitem__(self, index):
        del self._indices[index]

    def __getitem__(self, index):
        return self._indices[index]

    def __setitem__(self, index, value):
        try:
            value = int(value)
            self._indices[index] = value
        except ValueError:
            raise ValueError('value must be an int')

    def insert(self, index, value):
        try:
            value = int(value)
            self._indices.insert(index, value)
        except ValueError:
            raise ValueError('value must be an int')

    def _to_xml_element(self):
        if len(self) == 0:
            raise Cifti2HeaderError('VertexIndices element requires indices')

        vert_indices = xml.Element('VertexIndices')
        vert_indices.text = ' '.join([str(i) for i in self])
        return vert_indices


class Cifti2BrainModel(xml.XmlSerializable):
    """Element representing a mapping of the dimension to vertex or voxels.

    Mapping to vertices of voxels must be specified.

    * Description - Maps a range of indices to surface vertices or voxels when
      IndicesMapToDataType is "CIFTI_INDEX_TYPE_BRAIN_MODELS."
    * Attributes

        * IndexOffset - The matrix index of the first brainordinate of this
          BrainModel. Note that matrix indices are zero-based.
        * IndexCount - Number of surface vertices or voxels in this brain
          model, must be positive.
        * ModelType - Type of model representing the brain structure (surface
          or voxels).  Valid values are listed in the table below.
        * BrainStructure - Identifies the brain structure. Valid values for
          BrainStructure are listed in the table below. However, if the needed
          structure is not listed in the table, a message should be posted to
          the CIFTI Forum so that a standardized name can be created for the
          structure and added to the table.
        * SurfaceNumberOfVertices - When ModelType is CIFTI_MODEL_TYPE_SURFACE
          this attribute contains the actual (or true) number of vertices in
          the surface that is associated with this BrainModel. When this
          BrainModel represents all vertices in the surface, this value is the
          same as IndexCount. When this BrainModel represents only a subset of
          the surface's vertices, IndexCount will be less than this value.

    * Child Elements

        * VertexIndices (0...1)
        * VoxelIndicesIJK (0...1)

    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    For ModelType values, see CIFTI_MODEL_TYPES module attribute.

    For BrainStructure values, see CIFTI_BRAIN_STRUCTURES model attribute.

    Attributes
    ----------
    index_offset : int
        Start of the mapping
    index_count : int
        Number of elements in the array to be mapped
    model_type : str
        One of CIFTI_MODEL_TYPES
    brain_structure : str
        One of CIFTI_BRAIN_STRUCTURES
    surface_number_of_vertices : int
        Number of vertices in the surface. Use only for surface-type structure
    voxel_indices_ijk : Cifti2VoxelIndicesIJK, optional
        Indices on the image towards where the array indices are mapped
    vertex_indices : Cifti2VertexIndices, optional
        Indices of the vertices towards where the array indices are mapped
    """

    def __init__(
        self,
        index_offset=None,
        index_count=None,
        model_type=None,
        brain_structure=None,
        n_surface_vertices=None,
        voxel_indices_ijk=None,
        vertex_indices=None,
    ):
        self.index_offset = index_offset
        self.index_count = index_count
        self.model_type = model_type
        self.brain_structure = brain_structure
        self.surface_number_of_vertices = n_surface_vertices

        self.voxel_indices_ijk = voxel_indices_ijk
        self.vertex_indices = vertex_indices

    @property
    def voxel_indices_ijk(self):
        return self._voxel_indices_ijk

    @voxel_indices_ijk.setter
    def voxel_indices_ijk(self, value):
        self._voxel_indices_ijk = _value_if_klass(value, Cifti2VoxelIndicesIJK)

    @property
    def vertex_indices(self):
        return self._vertex_indices

    @vertex_indices.setter
    def vertex_indices(self, value):
        self._vertex_indices = _value_if_klass(value, Cifti2VertexIndices)

    def _to_xml_element(self):
        brain_model = xml.Element('BrainModel')

        for key in (
            'IndexOffset',
            'IndexCount',
            'ModelType',
            'BrainStructure',
            'SurfaceNumberOfVertices',
        ):
            attr = _underscore(key)
            value = getattr(self, attr)
            if value is not None:
                brain_model.attrib[key] = str(value)
        if self.voxel_indices_ijk:
            brain_model.append(self.voxel_indices_ijk._to_xml_element())
        if self.vertex_indices:
            brain_model.append(self.vertex_indices._to_xml_element())
        return brain_model


class Cifti2MatrixIndicesMap(xml.XmlSerializable, MutableSequence):
    """Class for Matrix Indices Map

    * Description - Provides a mapping between matrix indices and their
      interpretation.
    * Attributes

        * AppliesToMatrixDimension - Lists the dimension(s) of the matrix to
          which this MatrixIndicesMap applies. The dimensions of the matrix
          start at zero (dimension 0 describes the indices along the first
          dimension, dimension 1 describes the indices along the second
          dimension, etc.). If this MatrixIndicesMap applies to more than one
          matrix dimension, the values are separated by a comma.
        * IndicesMapToDataType - Type of data to which the MatrixIndicesMap
          applies.
        * NumberOfSeriesPoints - Indicates how many samples there are in a
          series mapping type. For example, this could be the number of
          timepoints in a timeseries.
        * SeriesExponent - Integer, SeriesStart and SeriesStep must be
          multiplied by 10 raised to the power of the value of this attribute
          to give the actual values assigned to indices (e.g., if SeriesStart
          is "5" and SeriesExponent is "-3", the value of the first series
          point is 0.005).
        * SeriesStart - Indicates what quantity should be assigned to the first
          series point.
        * SeriesStep - Indicates amount of change between each series point.
        * SeriesUnit - Indicates the unit of the result of multiplying
          SeriesStart and SeriesStep by 10 to the power of SeriesExponent.

    * Child Elements

        * BrainModel (0...N)
        * NamedMap (0...N)
        * Parcel (0...N)
        * Surface (0...N)
        * Volume (0...1)

    * Text Content: [NA]
    * Parent Element - Matrix

    Attributes
    ----------
    applies_to_matrix_dimension : list of ints
        Dimensions of this matrix that follow this mapping
    indices_map_to_data_type : str one of CIFTI_MAP_TYPES
        Type of mapping to the matrix indices
    number_of_series_points : int, optional
        If it is a series, number of points in the series
    series_exponent : int, optional
        If it is a series the exponent of the increment
    series_start : float, optional
        If it is a series, starting time
    series_step : float, optional
        If it is a series, step per element
    series_unit : str, optional
        If it is a series, units
    """

    _valid_type_mappings_ = {
        Cifti2BrainModel: ('CIFTI_INDEX_TYPE_BRAIN_MODELS',),
        Cifti2Parcel: ('CIFTI_INDEX_TYPE_PARCELS',),
        Cifti2NamedMap: ('CIFTI_INDEX_TYPE_LABELS',),
        Cifti2Volume: ('CIFTI_INDEX_TYPE_SCALARS', 'CIFTI_INDEX_TYPE_SERIES'),
        Cifti2Surface: ('CIFTI_INDEX_TYPE_SCALARS', 'CIFTI_INDEX_TYPE_SERIES'),
    }

    def __init__(
        self,
        applies_to_matrix_dimension,
        indices_map_to_data_type,
        number_of_series_points=None,
        series_exponent=None,
        series_start=None,
        series_step=None,
        series_unit=None,
        maps=[],
    ):
        self.applies_to_matrix_dimension = applies_to_matrix_dimension
        self.indices_map_to_data_type = indices_map_to_data_type
        self.number_of_series_points = number_of_series_points
        self.series_exponent = series_exponent
        self.series_start = series_start
        self.series_step = series_step
        self.series_unit = series_unit
        self._maps = []
        for m in maps:
            self.append(m)

    def __len__(self):
        return len(self._maps)

    def __delitem__(self, index):
        del self._maps[index]

    def __getitem__(self, index):
        return self._maps[index]

    def __setitem__(self, index, value):
        if isinstance(value, Cifti2Volume) and (
            self.volume is not None and not isinstance(self._maps[index], Cifti2Volume)
        ):
            raise Cifti2HeaderError('Only one Volume can be in a MatrixIndicesMap')
        self._maps[index] = value

    def insert(self, index, value):
        if isinstance(value, Cifti2Volume) and self.volume is not None:
            raise Cifti2HeaderError('Only one Volume can be in a MatrixIndicesMap')

        self._maps.insert(index, value)

    @property
    def named_maps(self):
        for p in self:
            if isinstance(p, Cifti2NamedMap):
                yield p

    @property
    def surfaces(self):
        for p in self:
            if isinstance(p, Cifti2Surface):
                yield p

    @property
    def parcels(self):
        for p in self:
            if isinstance(p, Cifti2Parcel):
                yield p

    @property
    def volume(self):
        for p in self:
            if isinstance(p, Cifti2Volume):
                return p
        return None

    @volume.setter
    def volume(self, volume):
        if not isinstance(volume, Cifti2Volume):
            raise ValueError('You can only set a volume with a volume')
        for i, v in enumerate(self):
            if isinstance(v, Cifti2Volume):
                break
        else:
            self.append(volume)
            return
        self[i] = volume

    @volume.deleter
    def volume(self):
        for i, v in enumerate(self):
            if isinstance(v, Cifti2Volume):
                break
        else:
            raise ValueError('No Cifti2Volume element')
        del self[i]

    @property
    def brain_models(self):
        for p in self:
            if isinstance(p, Cifti2BrainModel):
                yield p

    def _to_xml_element(self):
        if self.applies_to_matrix_dimension is None:
            raise Cifti2HeaderError(
                'MatrixIndicesMap element requires to be applied to at least 1 dimension'
            )

        mat_ind_map = xml.Element('MatrixIndicesMap')
        dims_as_strings = [str(dim) for dim in self.applies_to_matrix_dimension]
        mat_ind_map.attrib['AppliesToMatrixDimension'] = ','.join(dims_as_strings)
        for key in (
            'IndicesMapToDataType',
            'NumberOfSeriesPoints',
            'SeriesExponent',
            'SeriesStart',
            'SeriesStep',
            'SeriesUnit',
        ):
            attr = _underscore(key)
            value = getattr(self, attr)
            if value is not None:
                mat_ind_map.attrib[key] = str(value)
        for map_ in self:
            mat_ind_map.append(map_._to_xml_element())

        return mat_ind_map


class Cifti2Matrix(xml.XmlSerializable, MutableSequence):
    """CIFTI-2 Matrix object

    This is a list-like container where the elements are instances of
    :class:`Cifti2MatrixIndicesMap`.

    * Description: contains child elements that describe the meaning of the
      values in the matrix.
    * Attributes: [NA]
    * Child Elements

        * MetaData (0 .. 1)
        * MatrixIndicesMap (1 .. N)

    * Text Content: [NA]
    * Parent Element: CIFTI

    For each matrix (data) dimension, exactly one MatrixIndicesMap element must
    list it in the AppliesToMatrixDimension attribute.
    """

    def __init__(self):
        self._mims = []
        self.metadata = None

    @property
    def metadata(self):
        return self._meta

    @metadata.setter
    def metadata(self, meta):
        """Set the metadata for this Cifti2Header

        Parameters
        ----------
        meta : Cifti2MetaData

        Returns
        -------
        None
        """
        self._meta = _value_if_klass(meta, Cifti2MetaData)

    def _get_indices_from_mim(self, mim):
        applies_to_matrix_dimension = mim.applies_to_matrix_dimension
        if not isinstance(applies_to_matrix_dimension, Iterable):
            applies_to_matrix_dimension = (int(applies_to_matrix_dimension),)
        return applies_to_matrix_dimension

    @property
    def mapped_indices(self):
        """
        List of matrix indices that are mapped
        """
        mapped_indices = []
        for v in self:
            a2md = self._get_indices_from_mim(v)
            mapped_indices += a2md
        return mapped_indices

    def get_index_map(self, index):
        """
        Cifti2 Mapping class for a given index

        Parameters
        ----------
        index : int
            Index for which we want to obtain the mapping.
            Must be in the mapped_indices sequence.

        Returns
        -------
        cifti2_map : Cifti2MatrixIndicesMap
            Returns the Cifti2MatrixIndicesMap corresponding to
            the given index.
        """

        for v in self:
            a2md = self._get_indices_from_mim(v)
            if index in a2md:
                return v
        raise Cifti2HeaderError('Index not mapped')

    def _validate_new_mim(self, value):
        if value.applies_to_matrix_dimension is None:
            raise Cifti2HeaderError(
                'Cifti2MatrixIndicesMap needs to have '
                'the applies_to_matrix_dimension attribute set'
            )
        a2md = self._get_indices_from_mim(value)
        if not set(self.mapped_indices).isdisjoint(a2md):
            raise Cifti2HeaderError(
                'Indices in this Cifti2MatrixIndicesMap already mapped in this matrix'
            )

    def __setitem__(self, key, value):
        if not isinstance(value, Cifti2MatrixIndicesMap):
            raise TypeError('Not a valid Cifti2MatrixIndicesMap instance')
        self._validate_new_mim(value)
        self._mims[key] = value

    def __getitem__(self, key):
        return self._mims[key]

    def __delitem__(self, key):
        del self._mims[key]

    def __len__(self):
        return len(self._mims)

    def insert(self, index, value):
        if not isinstance(value, Cifti2MatrixIndicesMap):
            raise TypeError('Not a valid Cifti2MatrixIndicesMap instance')
        self._validate_new_mim(value)
        self._mims.insert(index, value)

    def _to_xml_element(self):
        # From the spec: "For each matrix dimension, exactly one
        # MatrixIndicesMap element must list it in the AppliesToMatrixDimension
        # attribute."
        mat = xml.Element('Matrix')
        if self.metadata:
            mat.append(self.metadata._to_xml_element())
        for mim in self._mims:
            mat.append(mim._to_xml_element())
        return mat

    def get_axis(self, index):
        """
        Generates the Cifti2 axis for a given dimension

        Parameters
        ----------
        index : int
            Dimension for which we want to obtain the mapping.

        Returns
        -------
        axis : :class:`.cifti2_axes.Axis`
        """
        from . import cifti2_axes

        return cifti2_axes.from_index_mapping(self.get_index_map(index))

    def get_data_shape(self):
        """
        Returns data shape expected based on the CIFTI-2 header

        Any dimensions omitted in the CIFTI-2 header will be given a default size of None.
        """
        from . import cifti2_axes

        if len(self.mapped_indices) == 0:
            return ()
        base_shape = [None] * (max(self.mapped_indices) + 1)
        for mim in self:
            size = len(cifti2_axes.from_index_mapping(mim))
            for idx in mim.applies_to_matrix_dimension:
                base_shape[idx] = size
        return tuple(base_shape)


class Cifti2Header(FileBasedHeader, xml.XmlSerializable):
    """Class for CIFTI-2 header extension"""

    def __init__(self, matrix=None, version='2.0'):
        FileBasedHeader.__init__(self)
        xml.XmlSerializable.__init__(self)
        if matrix is None:
            matrix = Cifti2Matrix()
        self.matrix = matrix
        self.version = version

    def _to_xml_element(self):
        cifti = xml.Element('CIFTI')
        cifti.attrib['Version'] = str(self.version)
        mat_xml = self.matrix._to_xml_element()
        if mat_xml is not None:
            cifti.append(mat_xml)
        return cifti

    def __eq__(self, other):
        return self.to_xml() == other.to_xml()

    @classmethod
    def may_contain_header(klass, binaryblock):
        from .parse_cifti2 import _Cifti2AsNiftiHeader

        return _Cifti2AsNiftiHeader.may_contain_header(binaryblock)

    @property
    def number_of_mapped_indices(self):
        """
        Number of mapped indices
        """
        return len(self.matrix)

    @property
    def mapped_indices(self):
        """
        List of matrix indices that are mapped
        """
        return self.matrix.mapped_indices

    def get_index_map(self, index):
        """
        Cifti2 Mapping class for a given index

        Parameters
        ----------
        index : int
            Index for which we want to obtain the mapping.
            Must be in the mapped_indices sequence.

        Returns
        -------
        cifti2_map : Cifti2MatrixIndicesMap
            Returns the Cifti2MatrixIndicesMap corresponding to
            the given index.
        """
        return self.matrix.get_index_map(index)

    def get_axis(self, index):
        """
        Generates the Cifti2 axis for a given dimension

        Parameters
        ----------
        index : int
            Dimension for which we want to obtain the mapping.

        Returns
        -------
        axis : :class:`.cifti2_axes.Axis`
        """
        return self.matrix.get_axis(index)

    @classmethod
    def from_axes(cls, axes):
        """
        Creates a new Cifti2 header based on the Cifti2 axes

        Parameters
        ----------
        axes : tuple of :class`.cifti2_axes.Axis`
            sequence of Cifti2 axes describing each row/column of the matrix to be stored

        Returns
        -------
        header : Cifti2Header
            new header describing the rows/columns in a format consistent with Cifti2
        """
        from . import cifti2_axes

        return cifti2_axes.to_header(axes)


class Cifti2Image(DataobjImage, SerializableImage):
    """Class for single file CIFTI-2 format image"""

    header_class = Cifti2Header
    header: Cifti2Header
    valid_exts = Nifti2Image.valid_exts
    files_types = Nifti2Image.files_types
    makeable = False
    rw = True

    def __init__(
        self,
        dataobj=None,
        header=None,
        nifti_header=None,
        extra=None,
        file_map=None,
        dtype=None,
    ):
        """Initialize image

        The image is a combination of (dataobj, header), with optional metadata
        in `nifti_header` (a NIfTI2 header).  There may be more metadata in the
        mapping `extra`. Filename / file-like objects can also go in the
        `file_map` mapping.

        Parameters
        ----------
        dataobj : object
            Object containing image data.  It should be some object that
            returns an array from ``np.asanyarray``.  It should have a
            ``shape`` attribute or property.
        header : Cifti2Header instance or sequence of :class:`cifti2_axes.Axis`
            Header with data for / from XML part of CIFTI-2 format.
            Alternatively a sequence of cifti2_axes.Axis objects can be provided
            describing each dimension of the array.
        nifti_header : None or mapping or NIfTI2 header instance, optional
            Metadata for NIfTI2 component of this format.
        extra : None or mapping
            Extra metadata not captured by `header` or `nifti_header`.
        file_map : mapping, optional
            Mapping giving file information for this image format.
        """
        if not isinstance(header, Cifti2Header) and header:
            header = Cifti2Header.from_axes(header)
        super().__init__(dataobj, header=header, extra=extra, file_map=file_map)
        self._nifti_header = LimitedNifti2Header.from_header(nifti_header)

        # if NIfTI header not specified, get data type from input array
        if dtype is not None:
            self.set_data_dtype(dtype)
        elif nifti_header is None and hasattr(dataobj, 'dtype'):
            self.set_data_dtype(dataobj.dtype)
        self.update_headers()

        if self._dataobj.shape != self.header.matrix.get_data_shape():
            warn(
                f'Dataobj shape {self._dataobj.shape} does not match shape '
                f'expected from CIFTI-2 header {self.header.matrix.get_data_shape()}'
            )

    @property
    def nifti_header(self):
        return self._nifti_header

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """Load a CIFTI-2 image from a file_map

        Parameters
        ----------
        file_map : file_map

        Returns
        -------
        img : Cifti2Image
            Returns a Cifti2Image
        """
        from .parse_cifti2 import Cifti2Extension, _Cifti2AsNiftiImage

        nifti_img = _Cifti2AsNiftiImage.from_file_map(
            file_map, mmap=mmap, keep_file_open=keep_file_open
        )

        # Get cifti2 header
        for item in nifti_img.header.extensions:
            if isinstance(item, Cifti2Extension):
                cifti_header = item.get_content()
                break
        else:
            raise ValueError('NIfTI2 header does not contain a CIFTI-2 extension')

        # Construct cifti image.
        # Use array proxy object where possible
        dataobj = nifti_img.dataobj
        return Cifti2Image(
            reshape_dataobj(dataobj, dataobj.shape[4:]),
            header=cifti_header,
            nifti_header=nifti_img.header,
            file_map=file_map,
        )

    @classmethod
    def from_image(klass, img):
        """Class method to create new instance of own class from `img`

        Parameters
        ----------
        img : instance
            In fact, an object with the API of :class:`DataobjImage`.

        Returns
        -------
        cimg : instance
            Image, of our own class
        """
        if isinstance(img, klass):
            return img
        raise NotImplementedError

    def to_file_map(self, file_map=None, dtype=None):
        """Write image to `file_map` or contained ``self.file_map``

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead.

        Returns
        -------
        None
        """
        from .parse_cifti2 import Cifti2Extension

        self.update_headers()
        header = self._nifti_header
        extension = Cifti2Extension.from_bytes(self.header.to_xml())
        header.extensions = Nifti1Extensions(
            ext for ext in header.extensions if not isinstance(ext, Cifti2Extension)
        )
        header.extensions.append(extension)
        if self._dataobj.shape != self.header.matrix.get_data_shape():
            raise ValueError(
                f'Dataobj shape {self._dataobj.shape} does not match shape '
                f'expected from CIFTI-2 header {self.header.matrix.get_data_shape()}'
            )
        # if intent code is not set, default to unknown CIFTI
        if header.get_intent()[0] == 'none':
            header.set_intent('NIFTI_INTENT_CONNECTIVITY_UNKNOWN')
        data = reshape_dataobj(self.dataobj, (1, 1, 1, 1) + self.dataobj.shape)
        # If qform not set, reset pixdim values so Nifti2 does not complain
        if header['qform_code'] == 0:
            header['pixdim'][:4] = 1
        img = Nifti2Image(data, None, header, dtype=dtype)
        img.to_file_map(file_map or self.file_map)

    def update_headers(self):
        """Harmonize NIfTI headers with image data

        Ensures that the NIfTI-2 header records the data shape in the last three
        ``dim`` fields. Per the spec:

            Because the first four dimensions in NIfTI are reserved for space and time, the CIFTI
            dimensions are stored in the NIfTI header in dim[5] and up, where dim[5] is the length
            of the first CIFTI dimension (number of values in a row), dim[6] is the length of the
            second CIFTI dimension, and dim[7] is the length of the third CIFTI dimension, if
            applicable. The fields dim[1] through dim[4] will be 1; dim[0] will be 6 or 7,
            depending on whether a third matrix dimension exists.

        >>> import numpy as np
        >>> data = np.zeros((2,3,4))
        >>> img = Cifti2Image(data)  # doctest: +IGNORE_WARNINGS
        >>> img.shape == (2, 3, 4)
        True
        >>> img.update_headers()
        >>> img.nifti_header.get_data_shape() == (1, 1, 1, 1, 2, 3, 4)
        True
        >>> img.shape == (2, 3, 4)
        True
        """
        self._nifti_header.set_data_shape((1, 1, 1, 1) + self._dataobj.shape)

    def get_data_dtype(self):
        return self._nifti_header.get_data_dtype()

    def set_data_dtype(self, dtype):
        self._nifti_header.set_data_dtype(dtype)


load = Cifti2Image.from_filename
save = Cifti2Image.instance_to_filename
