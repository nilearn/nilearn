# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Classes defining Gifti objects

The Gifti specification was (at time of writing) available as a PDF download
from http://www.nitrc.org/projects/gifti/
"""

from __future__ import annotations

import base64
import sys
import warnings
from copy import copy
from typing import cast

import numpy as np

from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes

GIFTI_DTYPES = (
    data_type_codes['NIFTI_TYPE_UINT8'],
    data_type_codes['NIFTI_TYPE_INT32'],
    data_type_codes['NIFTI_TYPE_FLOAT32'],
)


class _GiftiMDList(list):
    """List view of GiftiMetaData object that will translate most operations"""

    def __init__(self, metadata):
        self._md = metadata
        super().__init__(GiftiNVPairs._private_init(k, v, metadata) for k, v in metadata.items())

    def append(self, nvpair):
        self._md[nvpair.name] = nvpair.value
        super().append(nvpair)

    def clear(self):
        super().clear()
        self._md.clear()

    def extend(self, iterable):
        for nvpair in iterable:
            self.append(nvpair)

    def insert(self, index, nvpair):
        self._md[nvpair.name] = nvpair.value
        super().insert(index, nvpair)

    def pop(self, index=-1):
        nvpair = super().pop(index)
        nvpair._container = None
        del self._md[nvpair.name]
        return nvpair

    def remove(self, nvpair):
        super().remove(nvpair)
        del self._md[nvpair.name]


class GiftiMetaData(CaretMetaData):
    """A sequence of GiftiNVPairs containing metadata for a gifti data array"""

    @staticmethod
    def _sanitize(args, kwargs):
        """Sanitize and warn on deprecated arguments

        Accept nvpair positional/keyword argument that is a single
        ``GiftiNVPairs`` object.

        >>> import pytest
        >>> GiftiMetaData()
        <GiftiMetaData {}>
        >>> GiftiMetaData([("key", "val")])
        <GiftiMetaData {'key': 'val'}>
        >>> GiftiMetaData(key="val")
        <GiftiMetaData {'key': 'val'}>
        >>> GiftiMetaData({"key": "val"})
        <GiftiMetaData {'key': 'val'}>
        >>> with pytest.deprecated_call():
        ...     nvpairs = GiftiNVPairs(name='key', value='val')
        >>> with pytest.warns(FutureWarning):
        ...     GiftiMetaData(nvpairs)
        <GiftiMetaData {'key': 'val'}>
        >>> with pytest.warns(FutureWarning):
        ...     GiftiMetaData(nvpair=nvpairs)
        <GiftiMetaData {'key': 'val'}>
        """
        dep_init = False
        # Positional arg
        dep_init |= not kwargs and len(args) == 1 and isinstance(args[0], GiftiNVPairs)
        # Keyword arg
        dep_init |= not args and list(kwargs) == ['nvpair']
        if not dep_init:
            return args, kwargs

        warnings.warn(
            'GiftiMetaData now has a dict-like interface. '
            'See ``pydoc dict`` for initialization options. '
            'Passing ``GiftiNVPairs()`` or using the ``nvpair`` '
            'keyword will fail or behave unexpectedly in NiBabel 6.0.',
            FutureWarning,
            stacklevel=3,
        )
        pair = args[0] if args else kwargs.get('nvpair')
        return (), {pair.name: pair.value}

    @property
    @deprecate_with_version(
        'The data attribute is deprecated. Use GiftiMetaData object directly as a dict.',
        '4.0',
        '6.0',
    )
    def data(self):
        return _GiftiMDList(self)

    @classmethod
    @deprecate_with_version(
        'from_dict class method deprecated. Use GiftiMetaData directly.', '4.0', '6.0'
    )
    def from_dict(klass, data_dict):
        return klass(data_dict)

    @property
    @deprecate_with_version(
        'metadata property deprecated. Use GiftiMetaData object '
        'as dict or pass to dict() for a standard dictionary.',
        '4.0',
        '6.0',
    )
    def metadata(self):
        """Returns metadata as dictionary"""
        return dict(self)

    def print_summary(self):
        print(dict(self))


class GiftiNVPairs:
    """Gifti name / value pairs

    Attributes
    ----------
    name : str
    value : str
    """

    @deprecate_with_version(
        'GiftiNVPairs objects are deprecated. Use the GiftiMetaData object as a dict, instead.',
        '4.0',
        '6.0',
    )
    def __init__(self, name='', value=''):
        self._name = name
        self._value = value
        self._container = None

    @classmethod
    def _private_init(cls, name, value, md):
        """Private init method to provide warning-free experience"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self = cls(name, value)
        self._container = md
        return self

    def __eq__(self, other):
        if not isinstance(other, GiftiNVPairs):
            return NotImplemented
        return self.name == other.name and self.value == other.value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, key):
        if self._container:
            self._container[key] = self._container.pop(self._name)
        self._name = key

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if self._container:
            self._container[self._name] = val
        self._value = val


class GiftiLabelTable(xml.XmlSerializable):
    """Gifti label table: a sequence of key, label pairs

    From the gifti spec dated 2011-01-14:
        The label table is used by DataArrays whose values are an key into the
        LabelTable's labels. A file should contain at most one LabelTable and
        it must be located in the file prior to any DataArray elements.
    """

    def __init__(self):
        self.labels = []

    def __repr__(self):
        return f'<GiftiLabelTable {self.labels!r}>'

    def get_labels_as_dict(self):
        self.labels_as_dict = {}
        for ele in self.labels:
            self.labels_as_dict[ele.key] = ele.label
        return self.labels_as_dict

    def _to_xml_element(self):
        labeltable = xml.Element('LabelTable')
        for ele in self.labels:
            label = xml.SubElement(labeltable, 'Label')
            label.attrib['Key'] = str(ele.key)
            label.text = ele.label
            for attr in ('Red', 'Green', 'Blue', 'Alpha'):
                if getattr(ele, attr.lower(), None) is not None:
                    label.attrib[attr] = str(getattr(ele, attr.lower()))
        return labeltable

    def print_summary(self):
        print(self.get_labels_as_dict())


class GiftiLabel(xml.XmlSerializable):
    """Gifti label: association of integer key with optional RGBA values

    Quotes are from the gifti spec dated 2011-01-14.

    Attributes
    ----------
    key : int
        (From the spec): "This required attribute contains a non-negative
        integer value. If a DataArray's Intent is NIFTI_INTENT_LABEL and a
        value in the DataArray is 'X', its corresponding label is the label
        with the Key attribute containing the value 'X'. In early versions of
        the GIFTI file format, the attribute Index was used instead of Key. If
        an Index attribute is encountered, it should be processed like the Key
        attribute."
    red : None or float
        Optional value for red.
    green : None or float
        Optional value for green.
    blue : None or float
        Optional value for blue.
    alpha : None or float
        Optional value for alpha.

    Notes
    -----
    freesurfer examples seem not to conform to datatype "NIFTI_TYPE_RGBA32"
    because they are floats, not 4 8-bit integers.
    """

    def __init__(self, key=0, red=None, green=None, blue=None, alpha=None):
        self.key = key
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def __repr__(self):
        chars = 255 * np.array([self.red or 0, self.green or 0, self.blue or 0, self.alpha or 0])
        r, g, b, a = chars.astype('u1')
        return f'<GiftiLabel {self.key}="#{r:02x}{g:02x}{b:02x}{a:02x}">'

    @property
    def rgba(self):
        """Returns RGBA as tuple"""
        return (self.red, self.green, self.blue, self.alpha)

    @rgba.setter
    def rgba(self, rgba):
        """Set RGBA via sequence

        Parameters
        ----------
        rgba : length 4 sequence
            Sequence containing values for red, green, blue, alpha.
        """
        if len(rgba) != 4:
            raise ValueError('rgba must be length 4.')
        self.red, self.green, self.blue, self.alpha = rgba


def _arr2txt(arr, elem_fmt):
    arr = np.asarray(arr)
    assert arr.dtype.names is None
    if arr.ndim == 1:
        arr = arr[:, None]
    fmt = ' '.join([elem_fmt] * arr.shape[1])
    return '\n'.join(fmt % tuple(row) for row in arr)


class GiftiCoordSystem(xml.XmlSerializable):
    """Gifti coordinate system transform matrix

    Quotes are from the gifti spec dated 2011-01-14.

        "For a DataArray with an Intent NIFTI_INTENT_POINTSET, this element
        describes the stereotaxic space of the data before and after the
        application of a transformation matrix. The most common stereotaxic
        space is the Talairach Space that places the origin at the anterior
        commissure and the negative X, Y, and Z axes correspond to left,
        posterior, and inferior respectively.  At least one
        CoordinateSystemTransformMatrix is required in a DataArray with an
        intent of NIFTI_INTENT_POINTSET. Multiple
        CoordinateSystemTransformMatrix elements may be used to describe the
        transformation to multiple spaces."

    Attributes
    ----------
    dataspace : int
        From the spec: Contains the stereotaxic space of a DataArray's data
        prior to application of the transformation matrix. The stereotaxic
        space should be one of:

          - NIFTI_XFORM_UNKNOWN
          - NIFTI_XFORM_SCANNER_ANAT
          - NIFTI_XFORM_ALIGNED_ANAT
          - NIFTI_XFORM_TALAIRACH
          - NIFTI_XFORM_MNI_152

    xformspace : int
        Spec: "Contains the stereotaxic space of a DataArray's data after
        application of the transformation matrix. See the DataSpace element for
        a list of stereotaxic spaces."

    xform : array-like shape (4, 4)
        Affine transformation matrix
    """

    def __init__(self, dataspace=0, xformspace=0, xform=None):
        self.dataspace = dataspace
        self.xformspace = xformspace
        if xform is None:
            # create identity matrix
            self.xform = np.identity(4)
        else:
            self.xform = xform

    def __repr__(self):
        src = xform_codes.label[self.dataspace]
        dst = xform_codes.label[self.xformspace]
        return f'<GiftiCoordSystem {src}-to-{dst}>'

    def _to_xml_element(self):
        coord_xform = xml.Element('CoordinateSystemTransformMatrix')
        if self.xform is not None:
            dataspace = xml.SubElement(coord_xform, 'DataSpace')
            dataspace.text = xform_codes.niistring[self.dataspace]
            xformed_space = xml.SubElement(coord_xform, 'TransformedSpace')
            xformed_space.text = xform_codes.niistring[self.xformspace]
            matrix_data = xml.SubElement(coord_xform, 'MatrixData')
            matrix_data.text = _arr2txt(self.xform, '%10.6f')
        return coord_xform

    def print_summary(self):
        print('Dataspace: ', xform_codes.niistring[self.dataspace])
        print('XFormSpace: ', xform_codes.niistring[self.xformspace])
        print('Affine Transformation Matrix:\n', self.xform)


def _data_tag_element(dataarray, encoding, dtype, ordering):
    """Creates data tag with given `encoding`, returns as XML element"""
    import zlib

    order = array_index_order_codes.npcode[ordering]
    enclabel = gifti_encoding_codes.label[encoding]
    if enclabel == 'ASCII':
        da = _arr2txt(dataarray, KIND2FMT[dtype.kind])
    elif enclabel in ('B64BIN', 'B64GZ'):
        out = np.asanyarray(dataarray, dtype).tobytes(order)
        if enclabel == 'B64GZ':
            out = zlib.compress(out)
        da = base64.b64encode(out).decode()
    elif enclabel == 'External':
        raise NotImplementedError('In what format are the external files?')
    else:
        da = ''

    data = xml.Element('Data')
    data.text = da
    return data


class GiftiDataArray(xml.XmlSerializable):
    """Container for Gifti numerical data array and associated metadata

    Quotes are from the gifti spec dated 2011-01-14.

    Description of DataArray in spec:
        "This element contains the numeric data and its related metadata. The
        CoordinateSystemTransformMatrix child is only used when the DataArray's
        Intent is NIFTI_INTENT_POINTSET.  FileName and FileOffset are required
        if the data is stored in an external file."

    Attributes
    ----------
    darray : None or ndarray
        Data array
    intent : int
        NIFTI intent code, see nifti1.intent_codes
    datatype : int
        NIFTI data type codes, see nifti1.data_type_codes.  From the spec:
        "This required attribute describes the numeric type of the data
        contained in a Data Array and are limited to the types displayed in the
        table:

        NIFTI_TYPE_UINT8 : Unsigned, 8-bit bytes.
        NIFTI_TYPE_INT32 : Signed, 32-bit integers.
        NIFTI_TYPE_FLOAT32 : 32-bit single precision floating point."

        At the moment, we do not enforce that the datatype is one of these
        three.
    encoding : string
        Encoding of the data, see util.gifti_encoding_codes; default is
        GIFTI_ENCODING_B64GZ.
    endian : string
        The Endianness to store the data array.  Should correspond to the
        machine endianness.  Default is system byteorder.
    coordsys : :class:`GiftiCoordSystem` instance
        Input and output coordinate system with transformation matrix between
        the two.
    ind_ord : int
        The ordering of the array. see util.array_index_order_codes.  Default
        is RowMajorOrder - C ordering
    meta : :class:`GiftiMetaData` instance
        An instance equivalent to a dictionary for metadata information.
    ext_fname : str
        Filename in which data is stored, or empty string if no corresponding
        filename.
    ext_offset : int
        Position in bytes within `ext_fname` at which to start reading data.
    """

    def __init__(
        self,
        data=None,
        intent='NIFTI_INTENT_NONE',
        datatype=None,
        encoding='GIFTI_ENCODING_B64GZ',
        endian=sys.byteorder,
        coordsys=None,
        ordering='C',
        meta=None,
        ext_fname='',
        ext_offset=0,
    ):
        """
        Returns a shell object that cannot be saved.
        """
        self.data = None if data is None else np.asarray(data)
        self.intent = intent_codes.code[intent]
        if datatype is None:
            if self.data is None:
                datatype = 'none'
            elif data_type_codes[self.data.dtype] in GIFTI_DTYPES:
                datatype = self.data.dtype
            else:
                raise ValueError(
                    f'Data array has type {self.data.dtype}. '
                    'The GIFTI standard only supports uint8, int32 and float32 arrays.\n'
                    'Explicitly cast the data array to a supported dtype or pass an '
                    'explicit "datatype" parameter to GiftiDataArray().'
                )
        self.datatype = data_type_codes.code[datatype]
        self.encoding = gifti_encoding_codes.code[encoding]
        self.endian = gifti_endian_codes.code[endian]
        self.coordsys = coordsys or GiftiCoordSystem()
        self.ind_ord = array_index_order_codes.code[ordering]
        self.meta = (
            GiftiMetaData()
            if meta is None
            else meta
            if isinstance(meta, GiftiMetaData)
            else GiftiMetaData(meta)
        )
        self.ext_fname = ext_fname
        self.ext_offset = ext_offset
        self.dims = [] if self.data is None else list(self.data.shape)

    def __repr__(self):
        return f'<GiftiDataArray {intent_codes.label[self.intent]}{self.dims}>'

    @property
    def num_dim(self):
        return len(self.dims)

    def _to_xml_element(self):
        # fix endianness to machine endianness
        self.endian = gifti_endian_codes.code[sys.byteorder]

        # All attribute values must be strings
        data_array = xml.Element(
            'DataArray',
            attrib={
                'Intent': intent_codes.niistring[self.intent],
                'DataType': data_type_codes.niistring[self.datatype],
                'ArrayIndexingOrder': array_index_order_codes.label[self.ind_ord],
                'Dimensionality': str(self.num_dim),
                'Encoding': gifti_encoding_codes.specs[self.encoding],
                'Endian': gifti_endian_codes.specs[self.endian],
                'ExternalFileName': self.ext_fname,
                'ExternalFileOffset': str(self.ext_offset),
            },
        )
        for di, dn in enumerate(self.dims):
            data_array.attrib[f'Dim{di}'] = str(dn)

        if self.meta is not None:
            data_array.append(self.meta._to_xml_element())
        if self.coordsys is not None:
            data_array.append(self.coordsys._to_xml_element())
        # write data array depending on the encoding
        data_array.append(
            _data_tag_element(
                self.data,
                gifti_encoding_codes.specs[self.encoding],
                data_type_codes.dtype[self.datatype],
                self.ind_ord,
            )
        )

        return data_array

    def print_summary(self):
        print('Intent: ', intent_codes.niistring[self.intent])
        print('DataType: ', data_type_codes.niistring[self.datatype])
        print('ArrayIndexingOrder: ', array_index_order_codes.label[self.ind_ord])
        print('Dimensionality: ', self.num_dim)
        print('Dimensions: ', self.dims)
        print('Encoding: ', gifti_encoding_codes.specs[self.encoding])
        print('Endian: ', gifti_endian_codes.specs[self.endian])
        print('ExternalFileName: ', self.ext_fname)
        print('ExternalFileOffset: ', self.ext_offset)
        if self.coordsys is not None:
            print('----')
            print('Coordinate System:')
            print(self.coordsys.print_summary())

    @property
    def metadata(self):
        """Returns metadata as dictionary"""
        return dict(self.meta)


class GiftiImage(xml.XmlSerializable, SerializableImage):
    """GIFTI image object

    The Gifti spec suggests using the following suffixes to your
    filename when saving each specific type of data:

    .gii
        Generic GIFTI File
    .coord.gii
        Coordinates
    .func.gii
        Functional
    .label.gii
        Labels
    .rgba.gii
        RGB or RGBA
    .shape.gii
        Shape
    .surf.gii
        Surface
    .tensor.gii
        Tensors
    .time.gii
        Time Series
    .topo.gii
        Topology

    The Gifti file is stored in endian convention of the current machine.
    """

    valid_exts = ('.gii',)
    files_types = (('image', '.gii'),)
    _compressed_suffixes = ('.gz', '.bz2')

    # The parser will in due course be a GiftiImageParser, but we can't set
    # that now, because it would result in a circular import.  We set it after
    # the class has been defined, at the end of the class definition.
    parser: type[xml.XmlParser]

    def __init__(
        self,
        header=None,
        extra=None,
        file_map=None,
        meta=None,
        labeltable=None,
        darrays=None,
        version='1.0',
    ):
        super().__init__(header=header, extra=extra, file_map=file_map)
        if darrays is None:
            darrays = []
        if meta is None:
            meta = GiftiMetaData()
        if labeltable is None:
            labeltable = GiftiLabelTable()

        self._labeltable = labeltable
        self._meta = meta

        self.darrays = darrays
        self.version = version

    @property
    def numDA(self):
        return len(self.darrays)

    @property
    def labeltable(self):
        return self._labeltable

    @labeltable.setter
    def labeltable(self, labeltable):
        """Set the labeltable for this GiftiImage

        Parameters
        ----------
        labeltable : :class:`GiftiLabelTable` instance
        """
        if not isinstance(labeltable, GiftiLabelTable):
            raise TypeError('Not a valid GiftiLabelTable instance')
        self._labeltable = labeltable

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, meta):
        """Set the metadata for this GiftiImage

        Parameters
        ----------
        meta : :class:`GiftiMetaData` instance
        """
        if not isinstance(meta, GiftiMetaData):
            raise TypeError('Not a valid GiftiMetaData instance')
        self._meta = meta

    def add_gifti_data_array(self, dataarr):
        """Adds a data array to the GiftiImage

        Parameters
        ----------
        dataarr : :class:`GiftiDataArray` instance
        """
        if not isinstance(dataarr, GiftiDataArray):
            raise TypeError('Not a valid GiftiDataArray instance')
        self.darrays.append(dataarr)

    def remove_gifti_data_array(self, ith):
        """Removes the ith data array element from the GiftiImage"""
        self.darrays.pop(ith)

    def remove_gifti_data_array_by_intent(self, intent):
        """Removes all the data arrays with the given intent type"""
        intent2remove = intent_codes.code[intent]
        for dele in self.darrays:
            if dele.intent == intent2remove:
                self.darrays.remove(dele)

    def get_arrays_from_intent(self, intent):
        """Return list of GiftiDataArray elements matching given intent"""
        it = intent_codes.code[intent]
        return [x for x in self.darrays if x.intent == it]

    def agg_data(self, intent_code=None):
        """
        Aggregate GIFTI data arrays into an ndarray or tuple of ndarray

        In the general case, the numpy data array is extracted from each ``GiftiDataArray``
        object and returned in a ``tuple``, in the order they are found in the GIFTI image.

        If all ``GiftiDataArray`` s have ``intent`` of 2001 (``NIFTI_INTENT_TIME_SERIES``),
        then the data arrays are concatenated as columns, producing a vertex-by-time array.
        If an ``intent_code`` is passed, data arrays are filtered by the selected intents,
        before being aggregated.
        This may be useful for images containing several intents, or ensuring an expected
        data type in an image of uncertain provenance.
        If ``intent_code`` is a ``tuple``, then a ``tuple`` will be returned with the result of
        ``agg_data`` for each element, in order.
        This may be useful for ensuring that expected data arrives in a consistent order.

        Parameters
        ----------
        intent_code : None, string, integer or tuple of strings or integers, optional
            code(s) specifying nifti intent

        Returns
        -------
        tuple of ndarrays or ndarray
            If the input is a tuple, the returned tuple will match the order.

        Examples
        --------

        Consider a surface GIFTI file:

        >>> import nibabel as nib
        >>> from nibabel.testing import get_test_data
        >>> surf_img = nib.load(get_test_data('gifti', 'ascii.gii'))

        The coordinate data, which is indicated by the ``NIFTI_INTENT_POINTSET``
        intent code, may be retrieved using any of the following equivalent
        calls:

        >>> coords = surf_img.agg_data('NIFTI_INTENT_POINTSET')
        >>> coords_2 = surf_img.agg_data('pointset')
        >>> coords_3 = surf_img.agg_data(1008)  # Numeric code for pointset
        >>> print(np.array2string(coords, precision=3))
        [[-16.072 -66.188  21.267]
         [-16.706 -66.054  21.233]
         [-17.614 -65.402  21.071]]
        >>> np.array_equal(coords, coords_2)
        True
        >>> np.array_equal(coords, coords_3)
        True

        Similarly, the triangle mesh can be retrieved using various intent
        specifiers:

        >>> triangles = surf_img.agg_data('NIFTI_INTENT_TRIANGLE')
        >>> triangles_2 = surf_img.agg_data('triangle')
        >>> triangles_3 = surf_img.agg_data(1009)  # Numeric code for pointset
        >>> print(np.array2string(triangles))
        [[0 1 2]]
        >>> np.array_equal(triangles, triangles_2)
        True
        >>> np.array_equal(triangles, triangles_3)
        True

        All arrays can be retrieved as a ``tuple`` by omitting the intent
        code:

        >>> coords_4, triangles_4 = surf_img.agg_data()
        >>> np.array_equal(coords, coords_4)
        True
        >>> np.array_equal(triangles, triangles_4)
        True

        Finally, a tuple of intent codes may be passed in order to select
        the arrays in a specific order:

        >>> triangles_5, coords_5 = surf_img.agg_data(('triangle', 'pointset'))
        >>> np.array_equal(triangles, triangles_5)
        True
        >>> np.array_equal(coords, coords_5)
        True

        The following image is a GIFTI file with ten (10) data arrays of the same
        size, and with intent code 2001 (``NIFTI_INTENT_TIME_SERIES``):

        >>> func_img = nib.load(get_test_data('gifti', 'task.func.gii'))

        When aggregating time series data, these arrays are concatenated into
        a single, vertex-by-timestep array:

        >>> series = func_img.agg_data()
        >>> series.shape
        (642, 10)

        In the case of a GIFTI file with unknown data arrays, it may be preferable
        to specify the intent code, so that a time series array is always returned:

        >>> series_2 = func_img.agg_data('NIFTI_INTENT_TIME_SERIES')
        >>> series_3 = func_img.agg_data('time series')
        >>> series_4 = func_img.agg_data(2001)
        >>> np.array_equal(series, series_2)
        True
        >>> np.array_equal(series, series_3)
        True
        >>> np.array_equal(series, series_4)
        True

        Requesting a data array from a GIFTI file with no matching intent codes
        will result in an empty tuple:

        >>> surf_img.agg_data('time series')
        ()
        >>> func_img.agg_data('triangle')
        ()
        """

        # Allow multiple intents to specify the order
        # e.g., agg_data(('pointset', 'triangle')) ensures consistent order

        if isinstance(intent_code, tuple):
            return tuple(self.agg_data(intent_code=code) for code in intent_code)

        darrays = self.darrays if intent_code is None else self.get_arrays_from_intent(intent_code)
        all_data = tuple(da.data for da in darrays)
        all_intent = {intent_codes.niistring[da.intent] for da in darrays}

        if all_intent == {'NIFTI_INTENT_TIME_SERIES'}:  # stack when the gifti is a timeseries
            return np.column_stack(all_data)

        if len(all_data) == 1:
            all_data = all_data[0]

        return all_data

    def print_summary(self):
        print('----start----')
        print('Source filename: ', self.get_filename())
        print('Number of data arrays: ', self.numDA)
        print('Version: ', self.version)
        if self.meta is not None:
            print('----')
            print('Metadata:')
            print(self.meta.print_summary())
        if self.labeltable is not None:
            print('----')
            print('Labeltable:')
            print(self.labeltable.print_summary())
        for i, da in enumerate(self.darrays):
            print('----')
            print(f'DataArray {i}:')
            print(da.print_summary())
        print('----end----')

    def _to_xml_element(self):
        GIFTI = xml.Element(
            'GIFTI', attrib={'Version': self.version, 'NumberOfDataArrays': str(self.numDA)}
        )
        if self.meta is not None:
            GIFTI.append(self.meta._to_xml_element())
        if self.labeltable is not None:
            GIFTI.append(self.labeltable._to_xml_element())
        for dar in self.darrays:
            GIFTI.append(dar._to_xml_element())
        return GIFTI

    def to_xml(self, enc='utf-8', *, mode='strict', **kwargs) -> bytes:
        """Return XML corresponding to image content"""
        if mode == 'strict':
            if any(arr.datatype not in GIFTI_DTYPES for arr in self.darrays):
                raise ValueError(
                    'GiftiImage contains data arrays with invalid data types; '
                    'use mode="compat" to automatically cast to conforming types'
                )
        elif mode == 'compat':
            darrays = []
            for arr in self.darrays:
                if arr.datatype not in GIFTI_DTYPES:
                    arr = copy(arr)
                    # TODO: Better typing for recoders
                    dtype = cast(np.dtype, data_type_codes.dtype[arr.datatype])
                    if np.issubdtype(dtype, np.floating):
                        arr.datatype = data_type_codes['float32']
                    elif np.issubdtype(dtype, np.integer):
                        arr.datatype = data_type_codes['int32']
                    else:
                        raise ValueError(f'Cannot convert {dtype} to float32/int32')
                darrays.append(arr)
            gii = copy(self)
            gii.darrays = darrays
            return gii.to_xml(enc=enc, mode='strict')
        elif mode != 'force':
            raise TypeError(f'Unknown mode {mode}')
        header = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/115/gifti.dtd">
"""
        return header + super().to_xml(enc, **kwargs)

    # Avoid the indirection of going through to_file_map
    def to_bytes(self, enc='utf-8', *, mode='strict'):
        return self.to_xml(enc=enc, mode=mode)

    to_bytes.__doc__ = SerializableImage.to_bytes.__doc__

    def to_file_map(self, file_map=None, enc='utf-8', *, mode='strict'):
        """Save the current image to the specified file_map

        Parameters
        ----------
        file_map : dict
            Dictionary with single key ``image`` with associated value which is
            a :class:`FileHolder` instance pointing to the image file.

        Returns
        -------
        None
        """
        if file_map is None:
            file_map = self.file_map
        with file_map['image'].get_prepare_fileobj('wb') as f:
            f.write(self.to_xml(enc=enc, mode=mode))

    @classmethod
    def from_file_map(klass, file_map, buffer_size=35000000, mmap=True):
        """Load a Gifti image from a file_map

        Parameters
        ----------
        file_map : dict
            Dictionary with single key ``image`` with associated value which is
            a :class:`FileHolder` instance pointing to the image file.

        buffer_size: None or int, optional
            size of read buffer. None uses default buffer_size
            from xml.parsers.expat.

        mmap : {True, False, 'c', 'r', 'r+'}
            Controls the use of numpy memory mapping for reading data.  Only
            has an effect when loading GIFTI images with data stored in
            external files (``DataArray`` elements with an ``Encoding`` equal
            to ``ExternalFileBinary``).  If ``False``, do not try numpy
            ``memmap`` for data array.  If one of ``{'c', 'r', 'r+'}``, try
            numpy ``memmap`` with ``mode=mmap``.  A `mmap` value of ``True``
            gives the same behavior as ``mmap='c'``.  If the file cannot be
            memory-mapped, ignore `mmap` value and read array from file.

        Returns
        -------
        img : GiftiImage
        """
        parser = klass.parser(buffer_size=buffer_size, mmap=mmap)
        with file_map['image'].get_prepare_fileobj('rb') as fptr:
            parser.parse(fptr=fptr)
        return parser.img

    @classmethod
    def from_filename(klass, filename, buffer_size=35000000, mmap=True):
        file_map = klass.filespec_to_file_map(filename)
        img = klass.from_file_map(file_map, buffer_size=buffer_size, mmap=mmap)
        return img


# Now GiftiImage is defined, we can import the parser module and set the parser
from .parse_gifti_fast import GiftiImageParser

GiftiImage.parser = GiftiImageParser
