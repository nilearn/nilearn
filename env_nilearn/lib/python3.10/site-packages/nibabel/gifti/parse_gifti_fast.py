# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import base64
import os.path as op
import sys
import warnings
import zlib
from io import StringIO
from xml.parsers.expat import ExpatError

import numpy as np

from ..nifti1 import data_type_codes, intent_codes, xform_codes
from ..xmlutils import XmlParser
from .gifti import (
    GiftiCoordSystem,
    GiftiDataArray,
    GiftiImage,
    GiftiLabel,
    GiftiLabelTable,
    GiftiMetaData,
)
from .util import array_index_order_codes, gifti_encoding_codes, gifti_endian_codes


class GiftiParseError(ExpatError):
    """Gifti-specific parsing error"""


def read_data_block(darray, fname, data, mmap):
    """Parses data from a <Data> element, or loads from an external file.

    Parameters
    ----------
    darray : GiftiDataArray
         GiftiDataArray object representing the parent <DataArray> of this
         <Data> element

    fname : str or None
         Name of GIFTI file being loaded, or None if in-memory

    data : str or None
         Data to parse, or None if data is in an external file

    mmap : {True, False, 'c', 'r', 'r+'}
        Controls the use of numpy memory mapping for reading data.  Only has
        an effect when loading GIFTI images with data stored in external files
        (``DataArray`` elements with an ``Encoding`` equal to
        ``ExternalFileBinary``).  If ``False``, do not try numpy ``memmap``
        for data array.  If one of ``{'c', 'r', 'r+'}``, try numpy ``memmap``
        with ``mode=mmap``.  A `mmap` value of ``True`` gives the same
        behavior as ``mmap='c'``.  If the file cannot be memory-mapped, ignore
        `mmap` value and read array from file.

    Returns
    -------
    ``numpy.ndarray`` or ``numpy.memmap`` containing the parsed data
    """
    if mmap not in (True, False, 'c', 'r', 'r+'):
        raise ValueError("mmap value should be one of True, False, 'c', 'r', 'r+'")
    if mmap is True:
        mmap = 'c'
    enclabel = gifti_encoding_codes.label[darray.encoding]

    if enclabel not in ('ASCII', 'B64BIN', 'B64GZ', 'External'):
        raise GiftiParseError(f'Unknown encoding {darray.encoding}')

    # Encode the endianness in the dtype
    byteorder = gifti_endian_codes.byteorder[darray.endian]
    dtype = data_type_codes.dtype[darray.datatype].newbyteorder(byteorder)

    shape = tuple(darray.dims)
    order = array_index_order_codes.npcode[darray.ind_ord]

    # GIFTI_ENCODING_ASCII
    if enclabel == 'ASCII':
        return np.loadtxt(StringIO(data), dtype=dtype, ndmin=1).reshape(shape, order=order)

    # We assume that the external data file is raw uncompressed binary, with
    # the data type/endianness/ordering specified by the other DataArray
    # attributes
    if enclabel == 'External':
        if fname is None:
            raise GiftiParseError(
                'ExternalFileBinary is not supported when loading from in-memory XML'
            )
        ext_fname = op.join(op.dirname(fname), darray.ext_fname)
        if not op.exists(ext_fname):
            raise GiftiParseError('Cannot locate external file ' + ext_fname)
        # We either create a memmap, or load into memory
        newarr = None
        if mmap:
            try:
                return np.memmap(
                    ext_fname,
                    dtype=dtype,
                    mode=mmap,
                    offset=darray.ext_offset,
                    shape=shape,
                    order=order,
                )
            # If the memmap fails, we ignore the error and load the data into
            # memory below
            except (AttributeError, TypeError, ValueError):
                pass
        # mmap=False or np.memmap failed
        if newarr is None:
            return np.fromfile(
                ext_fname,
                dtype=dtype,
                count=np.prod(darray.dims),
                offset=darray.ext_offset,
            ).reshape(shape, order=order)

    # Numpy arrays created from bytes objects are read-only.
    # Neither b64decode nor decompress will return bytearrays, and there
    # are not equivalents to fobj.readinto to allow us to pass them, so
    # there is not a simple way to avoid making copies.
    # If this becomes a problem, we should write a decoding interface with
    # a tunable chunk size.
    dec = base64.b64decode(data.encode('ascii'))
    if enclabel == 'B64BIN':
        buff = bytearray(dec)
    else:
        # GIFTI_ENCODING_B64GZ
        buff = bytearray(zlib.decompress(dec))
    del dec
    return np.frombuffer(buff, dtype=dtype).reshape(shape, order=order)


def _str2int(in_str):
    # Convert string to integer, where empty string gives 0
    return int(in_str) if in_str else 0


class GiftiImageParser(XmlParser):
    def __init__(self, encoding=None, buffer_size=35000000, verbose=0, mmap=True):
        super().__init__(encoding=encoding, buffer_size=buffer_size, verbose=verbose)
        # output
        self.img = None

        # Queried when loading data from <Data> elements - see read_data_block
        self.mmap = mmap

        # finite state machine stack
        self.fsm_state = []

        # temporary constructs
        self.nvpair = None
        self.da = None
        self.coordsys = None
        self.lata = None
        self.label = None

        self.meta_global = None
        self.meta_da = None
        self.count_da = True

        # where to write CDATA:
        self.write_to = None

        # Collecting char buffer fragments
        self._char_blocks = None

    def StartElementHandler(self, name, attrs):
        self.flush_chardata()
        if self.verbose > 0:
            print('Start element:\n\t', repr(name), attrs)

        if name == 'GIFTI':
            # create gifti image
            self.img = GiftiImage()
            if 'Version' in attrs:
                self.img.version = attrs['Version']
            if 'NumberOfDataArrays' in attrs:
                self.expected_numDA = int(attrs['NumberOfDataArrays'])
            self.fsm_state.append('GIFTI')

        elif name == 'MetaData':
            self.fsm_state.append('MetaData')
            # if this metadata tag is first, create self.img.meta
            if len(self.fsm_state) == 2:
                self.meta_global = GiftiMetaData()
            else:
                # otherwise, create darray.meta
                self.meta_da = GiftiMetaData()

        elif name == 'MD':
            self.nvpair = ['', '']
            self.fsm_state.append('MD')

        elif name == 'Name':
            if self.nvpair is None:
                raise GiftiParseError
            self.write_to = 'Name'

        elif name == 'Value':
            if self.nvpair is None:
                raise GiftiParseError
            self.write_to = 'Value'

        elif name == 'LabelTable':
            self.lata = GiftiLabelTable()
            self.fsm_state.append('LabelTable')

        elif name == 'Label':
            self.label = GiftiLabel()
            if 'Index' in attrs:
                self.label.key = int(attrs['Index'])
            if 'Key' in attrs:
                self.label.key = int(attrs['Key'])
            if 'Red' in attrs:
                self.label.red = float(attrs['Red'])
            if 'Green' in attrs:
                self.label.green = float(attrs['Green'])
            if 'Blue' in attrs:
                self.label.blue = float(attrs['Blue'])
            if 'Alpha' in attrs:
                self.label.alpha = float(attrs['Alpha'])
            self.write_to = 'Label'

        elif name == 'DataArray':
            self.da = GiftiDataArray()
            if 'Intent' in attrs:
                self.da.intent = intent_codes.code[attrs['Intent']]
            if 'DataType' in attrs:
                self.da.datatype = data_type_codes.code[attrs['DataType']]
            if 'ArrayIndexingOrder' in attrs:
                self.da.ind_ord = array_index_order_codes.code[attrs['ArrayIndexingOrder']]
            num_dim = int(attrs.get('Dimensionality', 0))
            for i in range(num_dim):
                di = f'Dim{i}'
                if di in attrs:
                    self.da.dims.append(int(attrs[di]))
            # dimensionality has to correspond to the number of DimX given
            # TODO (bcipolli): don't assert; raise parse warning, and recover.
            assert len(self.da.dims) == num_dim
            if 'Encoding' in attrs:
                self.da.encoding = gifti_encoding_codes.code[attrs['Encoding']]
            if 'Endian' in attrs:
                self.da.endian = gifti_endian_codes.code[attrs['Endian']]
            if 'ExternalFileName' in attrs:
                self.da.ext_fname = attrs['ExternalFileName']
            if 'ExternalFileOffset' in attrs:
                self.da.ext_offset = _str2int(attrs['ExternalFileOffset'])
            self.img.darrays.append(self.da)
            self.fsm_state.append('DataArray')

        elif name == 'CoordinateSystemTransformMatrix':
            self.coordsys = GiftiCoordSystem()
            self.img.darrays[-1].coordsys = self.coordsys
            self.fsm_state.append('CoordinateSystemTransformMatrix')

        elif name == 'DataSpace':
            if self.coordsys is None:
                raise GiftiParseError
            self.write_to = 'DataSpace'

        elif name == 'TransformedSpace':
            if self.coordsys is None:
                raise GiftiParseError
            self.write_to = 'TransformedSpace'

        elif name == 'MatrixData':
            if self.coordsys is None:
                raise GiftiParseError
            self.write_to = 'MatrixData'

        elif name == 'Data':
            self.write_to = 'Data'

    def EndElementHandler(self, name):
        self.flush_chardata()
        if self.verbose > 0:
            print('End element:\n\t', repr(name))

        if name == 'GIFTI':
            if hasattr(self, 'expected_numDA') and self.expected_numDA != self.img.numDA:
                warnings.warn(
                    'Actual # of data arrays does not match # expected: '
                    f'{self.expected_numDA} != {self.img.numDA}.'
                )
            # remove last element of the list
            self.fsm_state.pop()
            # assert len(self.fsm_state) == 0

        elif name == 'MetaData':
            self.fsm_state.pop()
            if len(self.fsm_state) == 1:
                # only Gifti there, so this was a closing global
                # metadata tag
                self.img.meta = self.meta_global
                self.meta_global = None
            else:
                self.img.darrays[-1].meta = self.meta_da
                self.meta_da = None

        elif name == 'MD':
            self.fsm_state.pop()
            key, val = self.nvpair
            if self.meta_global is not None and self.meta_da is None:
                self.meta_global[key] = val
            elif self.meta_da is not None and self.meta_global is None:
                self.meta_da[key] = val
            # remove reference
            self.nvpair = None

        elif name == 'LabelTable':
            self.fsm_state.pop()
            # add labeltable
            self.img.labeltable = self.lata
            self.lata = None

        elif name == 'DataArray':
            self.fsm_state.pop()

        elif name == 'CoordinateSystemTransformMatrix':
            self.fsm_state.pop()
            self.coordsys = None

        elif name in ('DataSpace', 'TransformedSpace', 'MatrixData', 'Name', 'Value', 'Data'):
            self.write_to = None

        elif name == 'Label':
            self.lata.labels.append(self.label)
            self.label = None
            self.write_to = None

    def CharacterDataHandler(self, data):
        """Collect character data chunks pending collation

        The parser breaks the data up into chunks of size depending on the
        buffer_size of the parser.  A large bit of character data, with
        standard parser buffer_size (such as 8K) can easily span many calls to
        this function.  We thus collect the chunks and process them when we
        hit start or end tags.
        """
        if self._char_blocks is None:
            self._char_blocks = []
        self._char_blocks.append(data)

    def flush_chardata(self):
        """Collate and process collected character data"""
        # Nothing to do for empty elements, except for Data elements which
        # are within a DataArray with an external file
        if self.write_to != 'Data' and self._char_blocks is None:
            return
        # Just join the strings to get the data.  Maybe there are some memory
        # optimizations we could do by passing the list of strings to the
        # read_data_block function.
        if self._char_blocks is not None:
            data = ''.join(self._char_blocks)
        else:
            data = None
        # Reset the char collector
        self._char_blocks = None

        # Process data
        if self.write_to == 'Name':
            data = data.strip()
            self.nvpair[0] = data

        elif self.write_to == 'Value':
            data = data.strip()
            self.nvpair[1] = data

        elif self.write_to == 'DataSpace':
            data = data.strip()
            self.coordsys.dataspace = xform_codes.code[data]

        elif self.write_to == 'TransformedSpace':
            data = data.strip()
            self.coordsys.xformspace = xform_codes.code[data]

        elif self.write_to == 'MatrixData':
            # conversion to numpy array
            c = StringIO(data)
            self.coordsys.xform = np.loadtxt(c)
            c.close()

        elif self.write_to == 'Data':
            self.da.data = read_data_block(self.da, self.fname, data, self.mmap)
            # update the endianness according to the
            # current machine setting
            self.endian = gifti_endian_codes.code[sys.byteorder]

        elif self.write_to == 'Label':
            self.label.label = data.strip()

    @property
    def pending_data(self):
        """True if there is character data pending for processing"""
        return self._char_blocks is not None
