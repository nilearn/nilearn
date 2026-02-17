# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Class for reading AFNI BRIK/HEAD datasets

See https://afni.nimh.nih.gov/pub/dist/doc/program_help/README.attributes.html
for information on what is required to have a valid BRIK/HEAD dataset.

Unless otherwise noted, descriptions AFNI attributes in the code refer to this
document.

Notes
-----

In the AFNI HEAD file, the first two values of the attribute DATASET_RANK
determine the shape of the data array stored in the corresponding BRIK file.
The first value, DATASET_RANK[0], must be set to 3 denoting a 3D image. The
second value, DATASET_RANK[1], determines how many "sub-bricks" (in AFNI
parlance) / volumes there are along the fourth (traditionally, but not
exclusively) time axis. Thus, DATASET_RANK[1] will (at least as far as I (RM)
am aware) always be >= 1. This permits sub-brick indexing common in AFNI
programs (e.g., example4d+orig'[0]').
"""

import os
import re
from copy import deepcopy

import numpy as np

from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .spatialimages import HeaderDataError, ImageDataError, SpatialHeader, SpatialImage
from .volumeutils import Recoder

# used for doc-tests
filepath = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.realpath(os.path.join(filepath, 'tests/data'))

_attr_dic = {'string': str, 'integer': int, 'float': float}

_endian_dict = {
    'LSB_FIRST': '<',
    'MSB_FIRST': '>',
}

_dtype_dict = {
    0: 'B',
    1: 'h',
    3: 'f',
    5: 'D',
}

space_codes = Recoder(
    (
        (0, 'unknown', ''),
        (1, 'scanner', 'ORIG'),
        (3, 'talairach', 'TLRC'),
        (4, 'mni', 'MNI'),
    ),
    fields=('code', 'label', 'space'),
)


class AFNIImageError(ImageDataError):
    """Error when reading AFNI BRIK files"""


class AFNIHeaderError(HeaderDataError):
    """Error when reading AFNI HEAD file"""


DATA_OFFSET = 0
TYPE_RE = re.compile(r'type\s*=\s*(string|integer|float)-attribute\s*\n')
NAME_RE = re.compile(r'name\s*=\s*(\w+)\s*\n')


def _unpack_var(var):
    """
    Parses key : value pair from `var`

    Parameters
    ----------
    var : str
        Entry from HEAD file

    Returns
    -------
    name : str
        Name of attribute
    value : object
        Value of attribute

    Examples
    --------
    >>> var = "type = integer-attribute\\nname = BRICK_TYPES\\ncount = 1\\n1\\n"
    >>> name, attr = _unpack_var(var)
    >>> print(name, attr)
    BRICK_TYPES 1
    >>> var = "type = string-attribute\\nname = TEMPLATE_SPACE\\ncount = 5\\n'ORIG~"
    >>> name, attr = _unpack_var(var)
    >>> print(name, attr)
    TEMPLATE_SPACE ORIG
    """

    err_msg = f'Please check HEAD file to ensure it is AFNI compliant. Offending attribute:\n{var}'
    atype, aname = TYPE_RE.findall(var), NAME_RE.findall(var)
    if len(atype) != 1:
        raise AFNIHeaderError(f'Invalid attribute type entry in HEAD file. {err_msg}')
    if len(aname) != 1:
        raise AFNIHeaderError(f'Invalid attribute name entry in HEAD file. {err_msg}')
    atype = _attr_dic.get(atype[0], str)
    attr = ' '.join(var.strip().splitlines()[3:])
    if atype is not str:
        try:
            attr = [atype(f) for f in attr.split()]
        except ValueError:
            raise AFNIHeaderError(
                f'Failed to read variable from HEAD file due to improper type casting. {err_msg}'
            )
    else:
        # AFNI string attributes will always start with open single quote and
        # end with a tilde (NUL). These attributes CANNOT contain tildes (so
        # stripping is safe), but can contain single quotes (so we replace)
        attr = attr.replace("'", '', 1).rstrip('~')

    return aname[0], attr[0] if len(attr) == 1 else attr


def _get_datatype(info):
    """
    Gets datatype of BRIK file associated with HEAD file yielding `info`

    Parameters
    ----------
    info : dict
        As obtained by :func:`parse_AFNI_header`

    Returns
    -------
    dt : np.dtype
        Datatype of BRIK file associated with HEAD

    Notes
    -----
    ``BYTEORDER_STRING`` may be absent, signifying platform native byte order,
    or contain one of "LSB_FIRST" or "MSB_FIRST".

    ``BRICK_TYPES`` gives the storage data type for each sub-brick, with
    0=uint, 1=int16, 3=float32, 5=complex64 (see ``_dtype_dict``).  This should
    generally be the same value for each sub-brick in the dataset.
    """
    bo = info['BYTEORDER_STRING']
    bt = info['BRICK_TYPES']
    if isinstance(bt, list):
        if np.unique(bt).size > 1:
            raise AFNIImageError("Can't load file with multiple data types.")
        bt = bt[0]
    bo = _endian_dict.get(bo, '=')
    bt = _dtype_dict.get(bt, None)
    if bt is None:
        raise AFNIImageError("Can't deduce image data type.")
    return np.dtype(bo + bt)


def parse_AFNI_header(fobj):
    """
    Parses `fobj` to extract information from HEAD file

    Parameters
    ----------
    fobj : file-like object
        AFNI HEAD file object or filename. If file object, should
        implement at least ``read``

    Returns
    -------
    info : dict
        Dictionary containing AFNI-style key:value pairs from HEAD file

    Examples
    --------
    >>> fname = os.path.join(datadir, 'example4d+orig.HEAD')
    >>> info = parse_AFNI_header(fname)
    >>> print(info['BYTEORDER_STRING'])
    LSB_FIRST
    >>> print(info['BRICK_TYPES'])
    [1, 1, 1]
    """
    # edge case for being fed a filename instead of a file object
    if isinstance(fobj, str):
        with open(fobj) as src:
            return parse_AFNI_header(src)
    # unpack variables in HEAD file
    head = fobj.read().split('\n\n')
    return dict(map(_unpack_var, head))


class AFNIArrayProxy(ArrayProxy):
    """Proxy object for AFNI image array.

    Attributes
    ----------
    scaling : np.ndarray
        Scaling factor (one factor per volume/sub-brick) for data. Default is
        None
    """

    def __init__(self, file_like, header, *, mmap=True, keep_file_open=None):
        """
        Initialize AFNI array proxy

        Parameters
        ----------
        file_like : file-like object
            File-like object or filename. If file-like object, should implement
            at least ``read`` and ``seek``.
        header : ``AFNIHeader`` object
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading data.
            If False, do not try numpy ``memmap`` for data array.  If one of
            {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A `mmap` value of
            True gives the same behavior as ``mmap='c'``.  If `file_like`
            cannot be memory-mapped, ignore `mmap` value and read array from
            file.
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_like`` refers to an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.
        """
        super().__init__(file_like, header, mmap=mmap, keep_file_open=keep_file_open)
        self._scaling = header.get_data_scaling()

    @property
    def scaling(self):
        return self._scaling

    def _get_scaled(self, dtype, slicer):
        raw_data = self._get_unscaled(slicer=slicer)
        if self.scaling is None:
            if dtype is None:
                return raw_data
            final_type = np.promote_types(raw_data.dtype, dtype)
            return raw_data.astype(final_type, copy=False)

        # Broadcast scaling to shape of original data
        fake_data = strided_scalar(self._shape)
        _, scaling = np.broadcast_arrays(fake_data, self.scaling)

        final_type = np.result_type(raw_data, scaling)
        if dtype is not None:
            final_type = np.promote_types(final_type, dtype)

        # Slice scaling to give output shape
        return raw_data * scaling[slicer].astype(final_type)


class AFNIHeader(SpatialHeader):
    """Class for AFNI header"""

    def __init__(self, info):
        """
        Initialize AFNI header object

        Parameters
        ----------
        info : dict
            Information from HEAD file as obtained by :func:`parse_AFNI_header`

        Examples
        --------
        >>> fname = os.path.join(datadir, 'example4d+orig.HEAD')
        >>> header = AFNIHeader(parse_AFNI_header(fname))
        >>> header.get_data_dtype().str
        '<i2'
        >>> header.get_zooms()
        (3.0, 3.0, 3.0, 3.0)
        >>> header.get_data_shape()
        (33, 41, 25, 3)
        """
        self.info = info
        dt = _get_datatype(self.info)
        super().__init__(data_dtype=dt, shape=self._calc_data_shape(), zooms=self._calc_zooms())

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            raise AFNIHeaderError('Cannot create AFNIHeader from nothing.')
        if type(header) == klass:
            return header.copy()
        raise AFNIHeaderError('Cannot create AFNIHeader from non-AFNIHeader.')

    @classmethod
    def from_fileobj(klass, fileobj):
        info = parse_AFNI_header(fileobj)
        return klass(info)

    def copy(self):
        return AFNIHeader(deepcopy(self.info))

    def _calc_data_shape(self):
        """
        Calculate the output shape of the image data

        Returns length 3 tuple for 3D image, length 4 tuple for 4D.

        Returns
        -------
        (x, y, z, t) : tuple of int

        Notes
        -----
        ``DATASET_RANK[0]`` gives number of spatial dimensions (and apparently
        must be 3). ``DATASET_RANK[1]`` gives the number of sub-bricks.
        ``DATASET_DIMENSIONS`` is length 3, giving the number of voxels in i,
        j, k.
        """
        dset_rank = self.info['DATASET_RANK']
        shape = tuple(self.info['DATASET_DIMENSIONS'][: dset_rank[0]])
        n_vols = dset_rank[1]
        return shape + (n_vols,)

    def _calc_zooms(self):
        """
        Get image zooms from header data

        Spatial axes are first three indices, time axis is last index. If
        dataset is not a time series the last value will be zero.

        Returns
        -------
        zooms : tuple

        Notes
        -----
        Gets zooms from attributes ``DELTA`` and ``TAXIS_FLOATS``.

        ``DELTA`` gives (x,y,z) voxel sizes.

        ``TAXIS_FLOATS`` should be length 5, with first entry giving "Time
        origin", and second giving "Time step (TR)".
        """
        xyz_step = tuple(np.abs(self.info['DELTA']))
        t_step = self.info.get('TAXIS_FLOATS', (0, 0))
        if len(t_step) > 0:
            t_step = (t_step[1],)
        return xyz_step + t_step

    def get_space(self):
        """
        Return label for anatomical space to which this dataset is aligned.

        Returns
        -------
        space : str
            AFNI "space" designation; one of [ORIG, ANAT, TLRC, MNI]

        Notes
        -----
        There appears to be documentation for these spaces at
        https://afni.nimh.nih.gov/pub/dist/atlases/elsedemo/AFNI_atlas_spaces.niml
        """
        listed_space = self.info.get('TEMPLATE_SPACE', 0)
        space = space_codes.space[listed_space]
        return space

    def get_affine(self):
        """
        Returns affine of dataset

        Examples
        --------
        >>> fname = os.path.join(datadir, 'example4d+orig.HEAD')
        >>> header = AFNIHeader(parse_AFNI_header(fname))
        >>> header.get_affine()
        array([[ -3.    ,  -0.    ,  -0.    ,  49.5   ],
               [ -0.    ,  -3.    ,  -0.    ,  82.312 ],
               [  0.    ,   0.    ,   3.    , -52.3511],
               [  0.    ,   0.    ,   0.    ,   1.    ]])
        """
        # AFNI default is RAI- == LPS+ == DICOM order.  We need to flip RA sign
        # to align with nibabel RAS+ system
        affine = np.asarray(self.info['IJK_TO_DICOM_REAL']).reshape(3, 4)
        affine = np.vstack((affine * [[-1], [-1], [1]], [0, 0, 0, 1]))
        return affine

    def get_data_scaling(self):
        """
        AFNI applies volume-specific data scaling

        Examples
        --------
        >>> fname = os.path.join(datadir, 'scaled+tlrc.HEAD')
        >>> header = AFNIHeader(parse_AFNI_header(fname))
        >>> header.get_data_scaling()
        array([3.883363e-08])
        """
        # BRICK_FLOAT_FACS has one value per sub-brick, such that the scaled
        # values for sub-brick array [n] are the values read from disk *
        # BRICK_FLOAT_FACS[n]
        floatfacs = self.info.get('BRICK_FLOAT_FACS', None)
        if floatfacs is None or not np.any(floatfacs):
            return None
        scale = np.ones(self.info['DATASET_RANK'][1])
        floatfacs = np.atleast_1d(floatfacs)
        scale[floatfacs.nonzero()] = floatfacs[floatfacs.nonzero()]
        return scale

    def get_slope_inter(self):
        """
        Use `self.get_data_scaling()` instead

        Holdover because ``AFNIArrayProxy`` (inheriting from ``ArrayProxy``)
        requires this functionality so as to not error.
        """
        return None, None

    def get_data_offset(self):
        """Data offset in BRIK file

        Offset is always 0.
        """
        return DATA_OFFSET

    def get_volume_labels(self):
        """
        Returns volume labels

        Returns
        -------
        labels : list of str
            Labels for volumes along fourth dimension

        Examples
        --------
        >>> header = AFNIHeader(parse_AFNI_header(os.path.join(datadir, 'example4d+orig.HEAD')))
        >>> header.get_volume_labels()
        ['#0', '#1', '#2']
        """
        labels = self.info.get('BRICK_LABS', None)
        if labels is not None:
            labels = labels.split('~')
        return labels


class AFNIImage(SpatialImage):
    """
    AFNI Image file

    Can be loaded from either the BRIK or HEAD file (but MUST specify one!)

    Examples
    --------
    >>> import nibabel as nib
    >>> brik = nib.load(os.path.join(datadir, 'example4d+orig.BRIK.gz'))
    >>> brik.shape
    (33, 41, 25, 3)
    >>> brik.affine
    array([[ -3.    ,  -0.    ,  -0.    ,  49.5   ],
           [ -0.    ,  -3.    ,  -0.    ,  82.312 ],
           [  0.    ,   0.    ,   3.    , -52.3511],
           [  0.    ,   0.    ,   0.    ,   1.    ]])
    >>> head = load(os.path.join(datadir, 'example4d+orig.HEAD'))
    >>> np.array_equal(head.get_fdata(), brik.get_fdata())
    True
    """

    header_class = AFNIHeader
    header: AFNIHeader
    valid_exts = ('.brik', '.head')
    files_types = (('image', '.brik'), ('header', '.head'))
    _compressed_suffixes = ('.gz', '.bz2', '.Z', '.zst')
    makeable = False
    rw = False
    ImageArrayProxy = AFNIArrayProxy

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """
        Creates an AFNIImage instance from `file_map`

        Parameters
        ----------
        file_map : dict
            dict with keys ``image, header`` and values being fileholder
            objects for the respective BRIK and HEAD files
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        keep_file_open : {None, True, False}, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_like`` refers to an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.
        """
        with file_map['header'].get_prepare_fileobj('rt') as hdr_fobj:
            hdr = klass.header_class.from_fileobj(hdr_fobj)
        imgf = file_map['image'].fileobj
        imgf = file_map['image'].filename if imgf is None else imgf
        data = klass.ImageArrayProxy(imgf, hdr.copy(), mmap=mmap, keep_file_open=keep_file_open)
        return klass(data, hdr.get_affine(), header=hdr, extra=None, file_map=file_map)

    @classmethod
    def filespec_to_file_map(klass, filespec):
        """
        Make `file_map` from filename `filespec`

        AFNI BRIK files can be compressed, but HEAD files cannot - see
        afni.nimh.nih.gov/pub/dist/doc/program_help/README.compression.html.
        Thus, if you have AFNI files my_image.HEAD and my_image.BRIK.gz and you
        want to load the AFNI BRIK / HEAD pair, you can specify:

            * The HEAD filename - e.g., my_image.HEAD
            * The BRIK filename w/o compressed extension - e.g., my_image.BRIK
            * The full BRIK filename - e.g., my_image.BRIK.gz

        Parameters
        ----------
        filespec : str
            Filename that might be for this image file type.

        Returns
        -------
        file_map : dict
            dict with keys ``image`` and ``header`` where values are fileholder
            objects for the respective BRIK and HEAD files

        Raises
        ------
        ImageFileError
            If `filespec` is not recognizable as being a filename for this
            image type.
        """
        file_map = super().filespec_to_file_map(filespec)
        # check for AFNI-specific BRIK/HEAD compression idiosyncrasies
        for key, fholder in file_map.items():
            fname = fholder.filename
            if key == 'header' and not os.path.exists(fname):
                for ext in klass._compressed_suffixes:
                    fname = fname[: -len(ext)] if fname.endswith(ext) else fname
            elif key == 'image' and not os.path.exists(fname):
                for ext in klass._compressed_suffixes:
                    if os.path.exists(fname + ext):
                        fname += ext
                        break
            file_map[key].filename = fname
        return file_map


load = AFNIImage.from_filename
