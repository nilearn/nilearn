# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Header and image reading / writing functions for MGH image format

Author: Krish Subramaniam
"""

from os.path import splitext

import numpy as np

from ..affines import from_matvec, voxel_sizes
from ..arrayproxy import ArrayProxy, reshape_dataobj
from ..batteryrunners import BatteryRunner, Report
from ..filebasedimages import SerializableImage
from ..fileholders import FileHolder
from ..filename_parser import _stringify_path
from ..openers import ImageOpener
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..volumeutils import Recoder, array_from_file, array_to_file, endian_codes
from ..wrapstruct import LabeledWrapStruct

# mgh header
# See https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
DATA_OFFSET = 284
# Note that mgh data is strictly big endian ( hence the > sign )
# fmt: off
header_dtd = [
    ('version', '>i4'),             # 0; must be 1
    ('dims', '>i4', (4,)),          # 4; width, height, depth, nframes
    ('type', '>i4'),                # 20; data type
    ('dof', '>i4'),                 # 24; degrees of freedom
    ('goodRASFlag', '>i2'),         # 28; Mdc, Pxyz_c fields valid
    ('delta', '>f4', (3,)),         # 30; zooms (X, Y, Z)
    ('Mdc', '>f4', (3, 3)),         # 42; TRANSPOSE of direction cosine matrix
    ('Pxyz_c', '>f4', (3,)),        # 78; mm from (0, 0, 0) RAS to vol center
]
# Optional footer. Also has more stuff after this, optionally
footer_dtd = [
    ('tr', '>f4'),                  # 0; repetition time
    ('flip_angle', '>f4'),          # 4; flip angle
    ('te', '>f4'),                  # 8; echo time
    ('ti', '>f4'),                  # 12; inversion time
    ('fov', '>f4'),                 # 16; field of view (unused)
]
# fmt: on

header_dtype = np.dtype(header_dtd)
footer_dtype = np.dtype(footer_dtd)
hf_dtype = np.dtype(header_dtd + footer_dtd)

# caveat: Note that it's ambiguous to get the code given the bytespervoxel
# caveat 2: Note that the bytespervox you get is in str ( not an int)
_dtdefs = (  # code, conversion function, dtype, bytes per voxel
    (0, 'uint8', '>u1', '1', 'MRI_UCHAR', np.uint8, np.dtype('u1'), np.dtype('>u1')),
    (4, 'int16', '>i2', '2', 'MRI_SHORT', np.int16, np.dtype('i2'), np.dtype('>i2')),
    (1, 'int32', '>i4', '4', 'MRI_INT', np.int32, np.dtype('i4'), np.dtype('>i4')),
    (3, 'float', '>f4', '4', 'MRI_FLOAT', np.float32, np.dtype('f4'), np.dtype('>f4')),
)

# make full code alias bank, including dtype column
data_type_codes = Recoder(
    _dtdefs,
    fields=(
        'code',
        'label',
        'dtype',
        'bytespervox',
        'mritype',
        'np_dtype1',
        'np_dtype2',
        'numpy_dtype',
    ),
)


class MGHError(Exception):
    """Exception for MGH format related problems.

    To be raised whenever MGH is not happy, or we are not happy with
    MGH.
    """


class MGHHeader(LabeledWrapStruct, SpatialHeader):
    """Class for MGH format header

    The header also consists of the footer data which MGH places after the data
    chunk.
    """

    # Copies of module-level definitions
    template_dtype = hf_dtype
    _hdrdtype = header_dtype
    _ftrdtype = footer_dtype
    _data_type_codes = data_type_codes

    def __init__(self, binaryblock=None, check=True):
        """Initialize header from binary data block

        Parameters
        ----------
        binaryblock : {None, string} optional
            binary block to set into header.  By default, None, in
            which case we insert the default empty header block
        check : bool, optional
            Whether to check content of header in initialization.
            Default is True.
        """
        min_size = self._hdrdtype.itemsize
        full_size = self.template_dtype.itemsize
        if binaryblock is not None and len(binaryblock) >= min_size:
            # Right zero-pad or truncate binaryblock to appropriate size
            # Footer is optional and may contain variable-length text fields,
            # so limit to fixed fields
            binaryblock = binaryblock[:full_size] + b'\x00' * (full_size - len(binaryblock))
        super().__init__(binaryblock=binaryblock, endianness='big', check=False)
        if not self._structarr['goodRASFlag']:
            self._set_affine_default()
        if check:
            self.check_fix()

    @staticmethod
    def chk_version(hdr, fix=False):
        rep = Report()
        if hdr['version'] != 1:
            rep = Report(HeaderDataError, 40)
            rep.problem_msg = 'Unknown MGH format version'
            if fix:
                hdr['version'] = 1
        return hdr, rep

    @classmethod
    def _get_checks(klass):
        return (klass.chk_version,)

    @classmethod
    def from_header(klass, header=None, check=True):
        """Class method to create MGH header from another MGH header"""
        # own type, return copy
        if type(header) == klass:
            obj = header.copy()
            if check:
                obj.check_fix()
            return obj
        # not own type, make fresh header instance
        obj = klass(check=check)
        return obj

    @classmethod
    def from_fileobj(klass, fileobj, check=True):
        """
        classmethod for loading a MGH fileobject
        """
        # We need the following hack because MGH data stores header information
        # after the data chunk too. We read the header initially, deduce the
        # dimensions from the header, skip over and then read the footer
        # information
        hdr_str = fileobj.read(klass._hdrdtype.itemsize)
        hdr_str_to_np = np.ndarray(shape=(), dtype=klass._hdrdtype, buffer=hdr_str)
        if not np.all(hdr_str_to_np['dims']):
            raise MGHError('Dimensions of the data should be non-zero')
        tp = int(hdr_str_to_np['type'])
        fileobj.seek(
            DATA_OFFSET
            + int(klass._data_type_codes.bytespervox[tp]) * np.prod(hdr_str_to_np['dims'])
        )
        ftr_str = fileobj.read(klass._ftrdtype.itemsize)
        return klass(hdr_str + ftr_str, check=check)

    def get_affine(self):
        """Get the affine transform from the header information.

        MGH format doesn't store the transform directly. Instead it's gleaned
        from the zooms ( delta ), direction cosines ( Mdc ), RAS centers (
        Pxyz_c ) and the dimensions.
        """
        hdr = self._structarr
        MdcD = hdr['Mdc'].T * hdr['delta']
        vol_center = MdcD.dot(hdr['dims'][:3]) / 2
        return from_matvec(MdcD, hdr['Pxyz_c'] - vol_center)

    # For compatibility with nifti (multiple affines)
    get_best_affine = get_affine

    def get_vox2ras(self):
        """return the get_affine()"""
        return self.get_affine()

    def get_vox2ras_tkr(self):
        """Get the vox2ras-tkr transform. See "Torig" here:
        https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        """
        ds = self._structarr['delta']
        ns = self._structarr['dims'][:3] * ds / 2.0
        v2rtkr = np.array(
            [
                [-ds[0], 0, 0, ns[0]],
                [0, 0, ds[2], -ns[2]],
                [0, -ds[1], 0, ns[1]],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        return v2rtkr

    def get_ras2vox(self):
        """return the inverse get_affine()"""
        return np.linalg.inv(self.get_affine())

    def get_data_dtype(self):
        """Get numpy dtype for MGH data

        For examples see ``set_data_dtype``
        """
        code = int(self._structarr['type'])
        dtype = self._data_type_codes.numpy_dtype[code]
        return dtype

    def set_data_dtype(self, datatype):
        """Set numpy dtype for data from code or dtype or type"""
        try:
            code = self._data_type_codes[datatype]
        except KeyError:
            raise MGHError(f'datatype dtype "{datatype}" not recognized')
        self._structarr['type'] = code

    def _ndims(self):
        """Get dimensionality of data

        MGH does not encode dimensionality explicitly, so an image where the
        fourth dimension is 1 is treated as three-dimensional.

        Returns
        -------
        ndims : 3 or 4
        """
        return 3 + (self._structarr['dims'][3] > 1)

    def get_zooms(self):
        """Get zooms from header

        Returns the spacing of voxels in the x, y, and z dimensions.
        For four-dimensional files, a fourth zoom is included, equal to the
        repetition time (TR) in ms (see `The MGH/MGZ Volume Format
        <mghformat>`_).

        To access only the spatial zooms, use `hdr['delta']`.

        Returns
        -------
        z : tuple
           tuple of header zoom values

        .. _mghformat: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat#line-82
        """
        # Do not return time zoom (TR) if 3D image
        tzoom = (self['tr'],) if self._ndims() > 3 else ()
        return tuple(self._structarr['delta']) + tzoom

    def set_zooms(self, zooms):
        """Set zooms into header fields

        Sets the spacing of voxels in the x, y, and z dimensions.
        For four-dimensional files, a temporal zoom (repetition time, or TR, in
        ms) may be provided as a fourth sequence element.

        Parameters
        ----------
        zooms : sequence
            sequence of floats specifying spatial and (optionally) temporal
            zooms
        """
        hdr = self._structarr
        zooms = np.asarray(zooms)
        ndims = self._ndims()
        if len(zooms) > ndims:
            raise HeaderDataError(f'Expecting {ndims} zoom values')
        if np.any(zooms[:3] <= 0):
            raise HeaderDataError(
                f'Spatial (first three) zooms must be positive; got {tuple(zooms[:3])}'
            )
        hdr['delta'] = zooms[:3]
        if len(zooms) == 4:
            if zooms[3] < 0:
                raise HeaderDataError(f'TR must be non-negative; got {zooms[3]}')
            hdr['tr'] = zooms[3]

    def get_data_shape(self):
        """Get shape of data"""
        shape = tuple(self._structarr['dims'])
        # If last dimension (nframes) is 1, remove it because
        # we want to maintain 3D and it's redundant
        if shape[3] == 1:
            shape = shape[:3]
        return shape

    def set_data_shape(self, shape):
        """Set shape of data

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        """
        shape = tuple(shape)
        if len(shape) > 4:
            raise ValueError('Shape may be at most 4 dimensional')
        self._structarr['dims'] = shape + (1,) * (4 - len(shape))
        self._structarr['delta'] = 1

    def get_data_bytespervox(self):
        """Get the number of bytes per voxel of the data"""
        return int(self._data_type_codes.bytespervox[int(self._structarr['type'])])

    def get_data_size(self):
        """Get the number of bytes the data chunk occupies."""
        return self.get_data_bytespervox() * np.prod(self._structarr['dims'])

    def get_data_offset(self):
        """Return offset into data file to read data"""
        return DATA_OFFSET

    def get_footer_offset(self):
        """Return offset where the footer resides.
        Occurs immediately after the data chunk.
        """
        return self.get_data_offset() + self.get_data_size()

    def data_from_fileobj(self, fileobj):
        """Read data array from `fileobj`

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           data array
        """
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        offset = self.get_data_offset()
        return array_from_file(shape, dtype, fileobj, offset)

    def get_slope_inter(self):
        """MGH format does not do scaling?"""
        return None, None

    @classmethod
    def guessed_endian(klass, mapping):
        """MGHHeader data must be big-endian"""
        return '>'

    @classmethod
    def default_structarr(klass, endianness=None):
        """Return header data for empty header

        Ignores byte order; always big endian
        """
        if endianness is not None and endian_codes[endianness] != '>':
            raise ValueError('MGHHeader must always be big endian')
        structarr = super().default_structarr(endianness=endianness)
        structarr['version'] = 1
        structarr['dims'] = 1
        structarr['type'] = 3
        structarr['goodRASFlag'] = 1
        structarr['delta'] = 1
        structarr['Mdc'] = [[-1, 0, 0], [0, 0, 1], [0, -1, 0]]
        return structarr

    def _set_affine_default(self):
        """If goodRASFlag is 0, set the default affine"""
        self._structarr['goodRASFlag'] = 1
        self._structarr['delta'] = 1
        self._structarr['Mdc'] = [[-1, 0, 0], [0, 0, 1], [0, -1, 0]]
        self._structarr['Pxyz_c'] = 0

    def writehdr_to(self, fileobj):
        """Write header to fileobj

        Write starts at the beginning.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` and ``seek`` method

        Returns
        -------
        None
        """
        hdr_nofooter = np.ndarray((), dtype=self._hdrdtype, buffer=self.binaryblock)
        # goto the very beginning of the file-like obj
        fileobj.seek(0)
        fileobj.write(hdr_nofooter.tobytes())

    def writeftr_to(self, fileobj):
        """Write footer to fileobj

        Footer data is located after the data chunk. So move there and write.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` and ``seek`` method

        Returns
        -------
        None
        """
        ftr_loc_in_hdr = len(self.binaryblock) - self._ftrdtype.itemsize
        ftr_nd = np.ndarray(
            (), dtype=self._ftrdtype, buffer=self.binaryblock, offset=ftr_loc_in_hdr
        )
        fileobj.seek(self.get_footer_offset())
        fileobj.write(ftr_nd.tobytes())

    def copy(self):
        """Return copy of structure"""
        return self.__class__(self.binaryblock, check=False)

    def as_byteswapped(self, endianness=None):
        """Return new object with given ``endianness``

        If big endian, returns a copy of the object. Otherwise raises ValueError.

        Parameters
        ----------
        endianness : None or string, optional
           endian code to which to swap.  None means swap from current
           endianness, and is the default

        Returns
        -------
        wstr : ``MGHHeader``
           ``MGHHeader`` object

        """
        if endianness is None or endian_codes[endianness] != '>':
            raise ValueError('Cannot byteswap MGHHeader - must always be big endian')
        return self.copy()

    @classmethod
    def diagnose_binaryblock(klass, binaryblock, endianness=None):
        if endianness is not None and endian_codes[endianness] != '>':
            raise ValueError('MGHHeader must always be big endian')
        wstr = klass(binaryblock, check=False)
        battrun = BatteryRunner(klass._get_checks())
        reports = battrun.check_only(wstr)
        return '\n'.join([report.message for report in reports if report.message])


class MGHImage(SpatialImage, SerializableImage):
    """Class for MGH format image"""

    header_class = MGHHeader
    header: MGHHeader
    valid_exts = ('.mgh', '.mgz')
    # Register that .mgz extension signals gzip compression
    ImageOpener.compress_ext_map['.mgz'] = ImageOpener.gz_def
    files_types = (('image', '.mgh'),)
    _compressed_suffixes = ()

    makeable = True
    rw = True

    ImageArrayProxy = ArrayProxy

    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None):
        shape = dataobj.shape
        if len(shape) < 3:
            dataobj = reshape_dataobj(dataobj, shape + (1,) * (3 - len(shape)))
        super().__init__(dataobj, affine, header=header, extra=extra, file_map=file_map)

    @classmethod
    def filespec_to_file_map(klass, filespec):
        filespec = _stringify_path(filespec)
        """ Check for compressed .mgz format, then .mgh format """
        if splitext(filespec)[1].lower() == '.mgz':
            return dict(image=FileHolder(filename=filespec))
        return super().filespec_to_file_map(filespec)

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """Class method to create image from mapping in ``file_map``

        Parameters
        ----------
        file_map : dict
            Mapping with (key, value) pairs of (``file_type``, FileHolder
            instance giving file-likes for each file needed for this image
            type.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_map`` refers to an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.

        Returns
        -------
        img : MGHImage instance
        """
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        img_fh = file_map['image']
        mghf = img_fh.get_prepare_fileobj('rb')
        header = klass.header_class.from_fileobj(mghf)
        affine = header.get_affine()
        hdr_copy = header.copy()
        # Pass original image fileobj / filename to array proxy
        data = klass.ImageArrayProxy(
            img_fh.file_like, hdr_copy, mmap=mmap, keep_file_open=keep_file_open
        )
        img = klass(data, affine, header, file_map=file_map)
        return img

    def to_file_map(self, file_map=None):
        """Write image to `file_map` or contained ``self.file_map``

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        """
        if file_map is None:
            file_map = self.file_map
        data = np.asanyarray(self.dataobj)
        self.update_header()
        hdr = self.header
        with file_map['image'].get_prepare_fileobj('wb') as mghf:
            hdr.writehdr_to(mghf)
            self._write_data(mghf, data, hdr)
            hdr.writeftr_to(mghf)
        self._header = hdr
        self.file_map = file_map

    def _write_data(self, mghfile, data, header):
        """Utility routine to write image

        Parameters
        ----------
        mghfile : file-like
           file-like object implementing ``seek`` or ``tell``, and
           ``write``
        data : array-like
           array to write
        header : analyze-type header object
           header
        """
        shape = header.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError(
                'Data should be shape ({})'.format(', '.join(str(s) for s in shape))
            )
        offset = header.get_data_offset()
        out_dtype = header.get_data_dtype()
        array_to_file(data, mghfile, out_dtype, offset)

    def _affine2header(self):
        """Unconditionally set affine into the header"""
        hdr = self._header
        shape = np.array(self._dataobj.shape[:3])

        # for more information, go through save_mgh.m in FreeSurfer dist
        voxelsize = voxel_sizes(self._affine)
        Mdc = self._affine[:3, :3] / voxelsize
        c_ras = self._affine.dot(np.hstack((shape / 2.0, [1])))[:3]

        # Assign after we've had a chance to raise exceptions
        hdr['delta'] = voxelsize
        hdr['Mdc'] = Mdc.T
        hdr['Pxyz_c'] = c_ras


load = MGHImage.from_filename
save = MGHImage.instance_to_filename
