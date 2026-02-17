# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read ECAT format images

An ECAT format image consists of:

* a *main header*;
* at least one *matrix list* (mlist);

ECAT thinks of memory locations in terms of *blocks*.  One block is 512
bytes.  Thus block 1 starts at 0 bytes, block 2 at 512 bytes, and so on.

The matrix list is an array with one row per frame in the data.

Columns in the matrix list are:

* 0: Matrix identifier (frame number)
* 1: matrix data start block number (subheader followed by image data)
* 2: Last block number of matrix (image) data
* 3: Matrix status

    * 1: hxists - rw
    * 2: exists - ro
    * 3: matrix deleted

There is one sub-header for each image frame (or matrix in the terminology
above).  A sub-header can also be called an *image header*.  The sub-header is
one block (512 bytes), and the frame (image) data follows.

There is very little documentation of the ECAT format, and many of the comments
in this code come from a combination of trial and error and wild speculation.

XMedcon can read and write ECAT 6 format, and read ECAT 7 format: see
http://xmedcon.sourceforge.net and the ECAT files in the source of XMedCon,
currently ``libs/tpc/*ecat*`` and ``source/m-ecat*``.  Unfortunately XMedCon is
GPL and some of the header files are adapted from CTI files (called CTI code
below).  It's not clear what the licenses are for these files.
"""

import warnings
from numbers import Integral

import numpy as np

from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct

BLOCK_SIZE = 512

main_header_dtd = [
    ('magic_number', '14S'),
    ('original_filename', '32S'),
    ('sw_version', np.uint16),
    ('system_type', np.uint16),
    ('file_type', np.uint16),
    ('serial_number', '10S'),
    ('scan_start_time', np.uint32),
    ('isotope_name', '8S'),
    ('isotope_halflife', np.float32),
    ('radiopharmaceutical', '32S'),
    ('gantry_tilt', np.float32),
    ('gantry_rotation', np.float32),
    ('bed_elevation', np.float32),
    ('intrinsic_tilt', np.float32),
    ('wobble_speed', np.uint16),
    ('transm_source_type', np.uint16),
    ('distance_scanned', np.float32),
    ('transaxial_fov', np.float32),
    ('angular_compression', np.uint16),
    ('coin_samp_mode', np.uint16),
    ('axial_samp_mode', np.uint16),
    ('ecat_calibration_factor', np.float32),
    ('calibration_unitS', np.uint16),
    ('calibration_units_type', np.uint16),
    ('compression_code', np.uint16),
    ('study_type', '12S'),
    ('patient_id', '16S'),
    ('patient_name', '32S'),
    ('patient_sex', '1S'),
    ('patient_dexterity', '1S'),
    ('patient_age', np.float32),
    ('patient_height', np.float32),
    ('patient_weight', np.float32),
    ('patient_birth_date', np.uint32),
    ('physician_name', '32S'),
    ('operator_name', '32S'),
    ('study_description', '32S'),
    ('acquisition_type', np.uint16),
    ('patient_orientation', np.uint16),
    ('facility_name', '20S'),
    ('num_planes', np.uint16),
    ('num_frames', np.uint16),
    ('num_gates', np.uint16),
    ('num_bed_pos', np.uint16),
    ('init_bed_position', np.float32),
    ('bed_position', '15f'),
    ('plane_separation', np.float32),
    ('lwr_sctr_thres', np.uint16),
    ('lwr_true_thres', np.uint16),
    ('upr_true_thres', np.uint16),
    ('user_process_code', '10S'),
    ('acquisition_mode', np.uint16),
    ('bin_size', np.float32),
    ('branching_fraction', np.float32),
    ('dose_start_time', np.uint32),
    ('dosage', np.float32),
    ('well_counter_corr_factor', np.float32),
    ('data_units', '32S'),
    ('septa_state', np.uint16),
    ('fill', '12S'),
]
hdr_dtype = np.dtype(main_header_dtd)


subheader_dtd = [
    ('data_type', np.uint16),
    ('num_dimensions', np.uint16),
    ('x_dimension', np.uint16),
    ('y_dimension', np.uint16),
    ('z_dimension', np.uint16),
    ('x_offset', np.float32),
    ('y_offset', np.float32),
    ('z_offset', np.float32),
    ('recon_zoom', np.float32),
    ('scale_factor', np.float32),
    ('image_min', np.int16),
    ('image_max', np.int16),
    ('x_pixel_size', np.float32),
    ('y_pixel_size', np.float32),
    ('z_pixel_size', np.float32),
    ('frame_duration', np.uint32),
    ('frame_start_time', np.uint32),
    ('filter_code', np.uint16),
    ('x_resolution', np.float32),
    ('y_resolution', np.float32),
    ('z_resolution', np.float32),
    ('num_r_elements', np.float32),
    ('num_angles', np.float32),
    ('z_rotation_angle', np.float32),
    ('decay_corr_fctr', np.float32),
    ('corrections_applied', np.uint32),
    ('gate_duration', np.uint32),
    ('r_wave_offset', np.uint32),
    ('num_accepted_beats', np.uint32),
    ('filter_cutoff_frequency', np.float32),
    ('filter_resolution', np.float32),
    ('filter_ramp_slope', np.float32),
    ('filter_order', np.uint16),
    ('filter_scatter_fraction', np.float32),
    ('filter_scatter_slope', np.float32),
    ('annotation', '40S'),
    ('mt_1_1', np.float32),
    ('mt_1_2', np.float32),
    ('mt_1_3', np.float32),
    ('mt_2_1', np.float32),
    ('mt_2_2', np.float32),
    ('mt_2_3', np.float32),
    ('mt_3_1', np.float32),
    ('mt_3_2', np.float32),
    ('mt_3_3', np.float32),
    ('rfilter_cutoff', np.float32),
    ('rfilter_resolution', np.float32),
    ('rfilter_code', np.uint16),
    ('rfilter_order', np.uint16),
    ('zfilter_cutoff', np.float32),
    ('zfilter_resolution', np.float32),
    ('zfilter_code', np.uint16),
    ('zfilter_order', np.uint16),
    ('mt_4_1', np.float32),
    ('mt_4_2', np.float32),
    ('mt_4_3', np.float32),
    ('scatter_type', np.uint16),
    ('recon_type', np.uint16),
    ('recon_views', np.uint16),
    ('fill', '174S'),
    ('fill2', '96S'),
]
subhdr_dtype = np.dtype(subheader_dtd)

# Ecat Data Types
# See:
# http://www.turkupetcentre.net/software/libdoc/libtpcimgio/ecat7_8h_source.html#l00060
# and:
# http://www.turkupetcentre.net/software/libdoc/libtpcimgio/ecat7r_8c_source.html#l00717
_dtdefs = (  # code, name, equivalent dtype
    (1, 'ECAT7_BYTE', np.uint8),
    # Byte signed? https://github.com/nipy/nibabel/pull/302/files#r28275780
    (2, 'ECAT7_VAXI2', np.int16),
    (3, 'ECAT7_VAXI4', np.int32),
    (4, 'ECAT7_VAXR4', np.float32),
    (5, 'ECAT7_IEEER4', np.float32),
    (6, 'ECAT7_SUNI2', np.int16),
    (7, 'ECAT7_SUNI4', np.int32),
)
data_type_codes = make_dt_codes(_dtdefs)


# Matrix File Types
ft_defs = (  # code, name
    (0, 'ECAT7_UNKNOWN'),
    (1, 'ECAT7_2DSCAN'),
    (2, 'ECAT7_IMAGE16'),
    (3, 'ECAT7_ATTEN'),
    (4, 'ECAT7_2DNORM'),
    (5, 'ECAT7_POLARMAP'),
    (6, 'ECAT7_VOLUME8'),
    (7, 'ECAT7_VOLUME16'),
    (8, 'ECAT7_PROJ'),
    (9, 'ECAT7_PROJ16'),
    (10, 'ECAT7_IMAGE8'),
    (11, 'ECAT7_3DSCAN'),
    (12, 'ECAT7_3DSCAN8'),
    (13, 'ECAT7_3DNORM'),
    (14, 'ECAT7_3DSCANFIT'),
)
file_type_codes = dict(ft_defs)

patient_orient_defs = (  # code, description
    (0, 'ECAT7_Feet_First_Prone'),
    (1, 'ECAT7_Head_First_Prone'),
    (2, 'ECAT7_Feet_First_Supine'),
    (3, 'ECAT7_Head_First_Supine'),
    (4, 'ECAT7_Feet_First_Decubitus_Right'),
    (5, 'ECAT7_Head_First_Decubitus_Right'),
    (6, 'ECAT7_Feet_First_Decubitus_Left'),
    (7, 'ECAT7_Head_First_Decubitus_Left'),
    (8, 'ECAT7_Unknown_Orientation'),
)
patient_orient_codes = dict(patient_orient_defs)

# Indexes from the patient_orient_defs structure defined above for the
# neurological and radiological viewing conventions
patient_orient_radiological = [0, 2, 4, 6]
patient_orient_neurological = [1, 3, 5, 7]


class EcatHeader(WrapStruct, SpatialHeader):
    """Class for basic Ecat PET header

    Sub-parts of standard Ecat File

    * main header
    * matrix list
      which lists the information for each frame collected (can have 1 to many
      frames)
    * subheaders specific to each frame with possibly-variable sized data
      blocks

    This just reads the main Ecat Header, it does not load the data or read the
    mlist or any sub headers
    """

    template_dtype = hdr_dtype
    _ft_codes = file_type_codes
    _patient_orient_codes = patient_orient_codes

    def __init__(self, binaryblock=None, endianness=None, check=True):
        """Initialize Ecat header from bytes object

        Parameters
        ----------
        binaryblock : {None, bytes} optional
            binary block to set into header, By default, None in which case we
            insert default empty header block
        endianness : {None, '<', '>', other endian code}, optional
            endian code of binary block, If None, guess endianness
            from the data
        check : {True, False}, optional
            Whether to check and fix header for errors.  No checks currently
            implemented, so value has no effect.
        """
        super().__init__(binaryblock, endianness, check)

    @classmethod
    def guessed_endian(klass, hdr):
        """Guess endian from MAGIC NUMBER value of header data"""
        if not hdr['sw_version'] == 74:
            return swapped_code
        else:
            return native_code

    @classmethod
    def default_structarr(klass, endianness=None):
        """Return header data for empty header with given endianness"""
        hdr_data = super().default_structarr(endianness)
        hdr_data['magic_number'] = 'MATRIX72'
        hdr_data['sw_version'] = 74
        hdr_data['num_frames'] = 0
        hdr_data['file_type'] = 0  # Unknown
        hdr_data['ecat_calibration_factor'] = 1.0  # scale factor
        return hdr_data

    def get_data_dtype(self):
        """Get numpy dtype for data from header"""
        raise NotImplementedError('dtype is only valid from subheaders')

    def get_patient_orient(self):
        """gets orientation of patient based on code stored
        in header, not always reliable
        """
        code = self._structarr['patient_orientation'].item()
        if code not in self._patient_orient_codes:
            raise KeyError(f'Ecat Orientation CODE {code} not recognized')
        return self._patient_orient_codes[code]

    def get_filetype(self):
        """Type of ECAT Matrix File from code stored in header"""
        code = self._structarr['file_type'].item()
        if code not in self._ft_codes:
            raise KeyError(f'Ecat Filetype CODE {code} not recognized')
        return self._ft_codes[code]

    @classmethod
    def _get_checks(klass):
        """Return sequence of check functions for this class"""
        return ()


def read_mlist(fileobj, endianness):
    """read (nframes, 4) matrix list array from `fileobj`

    Parameters
    ----------
    fileobj : file-like
        an open file-like object implementing ``seek`` and ``read``

    Returns
    -------
    mlist : (nframes, 4) ndarray
        matrix list is an array with ``nframes`` rows and columns:

        * 0: Matrix identifier (frame number)
        * 1: matrix data start block number (subheader followed by image data)
        * 2: Last block number of matrix (image) data
        * 3: Matrix status

            * 1: hxists - rw
            * 2: exists - ro
            * 3: matrix deleted

    Notes
    -----
    A block is 512 bytes.

    ``block_no`` in the code below is 1-based.  block 1 is the main header,
    and the mlist blocks start at block number 2.

    The 512 bytes in an mlist block contain 32 rows of the int32 (nframes,
    4) mlist matrix.

    The first row of these 32 looks like a special row.  The 4 values appear
    to be (respectively):

    * not sure - maybe negative number of mlist rows (out of 31) that are
      blank and not used in this block.  Called `nfree` but unused in CTI
      code;
    * block_no - of next set of mlist entries or 2 if no more entries. We also
      allow 1 or 0 to signal no more entries;
    * <no idea>.  Called `prvblk` in CTI code, so maybe previous block no;
    * n_rows - number of mlist rows in this block (between ?0 and 31) (called
      `nused` in CTI code).
    """
    dt = np.dtype(np.int32)
    if endianness is not native_code:
        dt = dt.newbyteorder(endianness)
    mlists = []
    mlist_index = 0
    mlist_block_no = 2  # 1-based indexing, block with first mlist
    while True:
        # Read block containing mlist entries
        fileobj.seek((mlist_block_no - 1) * BLOCK_SIZE)  # fix 1-based indexing
        dat = fileobj.read(BLOCK_SIZE)
        rows = np.ndarray(shape=(32, 4), dtype=dt, buffer=dat)
        # First row special, points to next mlist entries if present
        n_unused, mlist_block_no, _, n_rows = rows[0]
        if not (n_unused + n_rows) == 31:  # Some error condition here?
            mlist = []
            return mlist
        # Use all but first housekeeping row
        mlists.append(rows[1 : n_rows + 1])
        mlist_index += n_rows
        if mlist_block_no <= 2:  # should block_no in (1, 2) be an error?
            break
    return np.vstack(mlists)


def get_frame_order(mlist):
    """Returns the order of the frames stored in the file
    Sometimes Frames are not stored in the file in
    chronological order, this can be used to extract frames
    in correct order

    Returns
    -------
    id_dict: dict mapping frame number -> [mlist_row, mlist_id]

    (where mlist id is value in the first column of the mlist matrix )

    Examples
    --------
    >>> import os
    >>> import nibabel as nib
    >>> nibabel_dir = os.path.dirname(nib.__file__)
    >>> from nibabel import ecat
    >>> ecat_file = os.path.join(nibabel_dir,'tests','data','tinypet.v')
    >>> img = ecat.load(ecat_file)
    >>> mlist = img.get_mlist()
    >>> get_frame_order(mlist)
    {0: [0, 16842758]}
    """
    ids = mlist[:, 0].copy()
    n_valid = np.sum(ids > 0)
    ids[ids <= 0] = ids.max() + 1  # put invalid frames at end after sort
    valid_order = np.argsort(ids)
    if not all(valid_order == sorted(valid_order)):
        # raise UserWarning if Frames stored out of order
        warnings.warn_explicit(
            f'Frames stored out of order; true order = {valid_order}\n'
            'frames will be accessed in order STORED, NOT true order',
            UserWarning,
            'ecat',
            0,
        )
    id_dict = {}
    for i in range(n_valid):
        id_dict[i] = [valid_order[i], ids[valid_order[i]]]
    return id_dict


def get_series_framenumbers(mlist):
    """Returns framenumber of data as it was collected,
    as part of a series; not just the order of how it was
    stored in this or across other files

    For example, if the data is split between multiple files
    this should give you the true location of this frame as
    collected in the series
    (Frames are numbered starting at ONE (1) not Zero)

    Returns
    -------
    frame_dict: dict mapping order_stored -> frame in series
            where frame in series counts from 1; [1,2,3,4...]

    Examples
    --------
    >>> import os
    >>> import nibabel as nib
    >>> nibabel_dir = os.path.dirname(nib.__file__)
    >>> from nibabel import ecat
    >>> ecat_file = os.path.join(nibabel_dir,'tests','data','tinypet.v')
    >>> img = ecat.load(ecat_file)
    >>> mlist = img.get_mlist()
    >>> get_series_framenumbers(mlist)
    {0: 1}
    """
    nframes = len(mlist)
    frames_order = get_frame_order(mlist)
    mlist_nframes = len(frames_order)
    trueframenumbers = np.arange(nframes - mlist_nframes, nframes)
    frame_dict = {}
    for frame_stored, (true_order, _) in frames_order.items():
        # frame as stored in file -> true number in series
        try:
            frame_dict[frame_stored] = trueframenumbers[true_order] + 1
        except IndexError:
            raise OSError('Error in header or mlist order unknown')
    return frame_dict


def read_subheaders(fileobj, mlist, endianness):
    """Retrieve all subheaders and return list of subheader recarrays

    Parameters
    ----------
    fileobj : file-like
        implementing ``read`` and ``seek``
    mlist : (nframes, 4) ndarray
        Columns are:
        * 0 - Matrix identifier.
        * 1 - subheader block number
        * 2 - Last block number of matrix data block.
        * 3 - Matrix status
    endianness : {'<', '>'}
        little / big endian code

    Returns
    -------
    subheaders : list
        List of subheader structured arrays
    """
    subheaders = []
    dt = subhdr_dtype
    if endianness is not native_code:
        dt = dt.newbyteorder(endianness)
    for mat_id, sh_blkno, sh_last_blkno, mat_stat in mlist:
        if sh_blkno == 0:
            break
        offset = (sh_blkno - 1) * BLOCK_SIZE
        fileobj.seek(offset)
        tmpdat = fileobj.read(BLOCK_SIZE)
        sh = np.ndarray(shape=(), dtype=dt, buffer=tmpdat)
        subheaders.append(sh)
    return subheaders


class EcatSubHeader:
    _subhdrdtype = subhdr_dtype
    _data_type_codes = data_type_codes

    def __init__(self, hdr, mlist, fileobj):
        """parses the subheaders in the ecat (.v) file
        there is one subheader for each frame in the ecat file

        Parameters
        ----------
        hdr : EcatHeader
            ECAT main header
        mlist : array shape (N, 4)
            Matrix list
        fileobj : ECAT file <filename>.v  fileholder or file object
                  with read, seek methods
        """
        self._header = hdr
        self.endianness = hdr.endianness
        self._mlist = mlist
        self.fileobj = fileobj
        self.subheaders = read_subheaders(fileobj, mlist, hdr.endianness)

    def get_shape(self, frame=0):
        """returns shape of given frame"""
        subhdr = self.subheaders[frame]
        x = subhdr['x_dimension'].item()
        y = subhdr['y_dimension'].item()
        z = subhdr['z_dimension'].item()
        return x, y, z

    def get_nframes(self):
        """returns number of frames"""
        framed = get_frame_order(self._mlist)
        return len(framed)

    def _check_affines(self):
        """checks if all affines are equal across frames"""
        nframes = self.get_nframes()
        if nframes == 1:
            return True
        affs = [self.get_frame_affine(i) for i in range(nframes)]
        if affs:
            i = iter(affs)
            first = next(i)
            for item in i:
                if not np.allclose(first, item):
                    return False
        return True

    def get_frame_affine(self, frame=0):
        """returns best affine for given frame of data"""
        subhdr = self.subheaders[frame]
        x_off = subhdr['x_offset']
        y_off = subhdr['y_offset']
        z_off = subhdr['z_offset']

        zooms = self.get_zooms(frame=frame)

        dims = self.get_shape(frame)
        # get translations from center of image
        origin_offset = (np.array(dims) - 1) / 2.0
        aff = np.diag(zooms)
        aff[:3, -1] = -origin_offset * zooms[:-1] + np.array([x_off, y_off, z_off])
        return aff

    def get_zooms(self, frame=0):
        """returns zooms  ...pixdims"""
        subhdr = self.subheaders[frame]
        x_zoom = subhdr['x_pixel_size'] * 10
        y_zoom = subhdr['y_pixel_size'] * 10
        z_zoom = subhdr['z_pixel_size'] * 10
        return (x_zoom, y_zoom, z_zoom, 1)

    def _get_data_dtype(self, frame):
        dtcode = self.subheaders[frame]['data_type'].item()
        return self._data_type_codes.dtype[dtcode]

    def _get_frame_offset(self, frame=0):
        return int(self._mlist[frame][1] * BLOCK_SIZE)

    def _get_oriented_data(self, raw_data, orientation=None):
        """
        Get data oriented following ``patient_orientation`` header field. If
        the ``orientation`` parameter is given, return data according to this
        orientation.

        :param raw_data: Numpy array containing the raw data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing the oriented data
        """
        if orientation is None:
            orientation = self._header['patient_orientation']
        elif orientation == 'neurological':
            orientation = patient_orient_neurological[0]
        elif orientation == 'radiological':
            orientation = patient_orient_radiological[0]
        else:
            raise ValueError('orientation should be None, neurological or radiological')

        if orientation in patient_orient_neurological:
            raw_data = raw_data[::-1, ::-1, ::-1]
        elif orientation in patient_orient_radiological:
            raw_data = raw_data[::, ::-1, ::-1]

        return raw_data

    def raw_data_from_fileobj(self, frame=0, orientation=None):
        """
        Get raw data from file object.

        :param frame: Time frame index from where to fetch data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing (possibly oriented) raw data

        .. seealso:: data_from_fileobj
        """
        dtype = self._get_data_dtype(frame)
        if self._header.endianness is not native_code:
            dtype = dtype.newbyteorder(self._header.endianness)
        shape = self.get_shape(frame)
        offset = self._get_frame_offset(frame)
        fid_obj = self.fileobj
        raw_data = array_from_file(shape, dtype, fid_obj, offset=offset)
        raw_data = self._get_oriented_data(raw_data, orientation)
        return raw_data

    def data_from_fileobj(self, frame=0, orientation=None):
        """
        Read scaled data from file for a given frame

        :param frame: Time frame index from where to fetch data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing (possibly oriented) raw data

        .. seealso:: raw_data_from_fileobj
        """
        header = self._header
        subhdr = self.subheaders[frame]
        raw_data = self.raw_data_from_fileobj(frame, orientation)
        # Scale factors have to be set to scalars to force scalar upcasting
        data = raw_data * header['ecat_calibration_factor'].item()
        data = data * subhdr['scale_factor'].item()
        return data


class EcatImageArrayProxy:
    """Ecat implementation of array proxy protocol

    The array proxy allows us to freeze the passed fileobj and
    header such that it returns the expected data array.
    """

    def __init__(self, subheader):
        self._subheader = subheader
        self._data = None
        x, y, z = subheader.get_shape()
        nframes = subheader.get_nframes()
        self._shape = (x, y, z, nframes)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_proxy(self):
        return True

    def __array__(self, dtype=None):
        """Read of data from file

        This reads ALL FRAMES into one array, can be memory expensive.

        If you want to read only some slices, use the slicing syntax
        (``__getitem__``) below, or ``subheader.data_from_fileobj(frame)``

        Parameters
        ----------
        dtype : numpy dtype specifier, optional
            A numpy dtype specifier specifying the type of the returned array.

        Returns
        -------
        array
            Scaled image data with type `dtype`.
        """
        # dtype=None is interpreted as float64
        data = np.empty(self.shape)
        frame_mapping = get_frame_order(self._subheader._mlist)
        for i in sorted(frame_mapping):
            data[:, :, :, i] = self._subheader.data_from_fileobj(frame_mapping[i][0])
        if dtype is not None:
            data = data.astype(dtype, copy=False)
        return data

    def __getitem__(self, sliceobj):
        """Return slice `sliceobj` from ECAT data, optimizing if possible"""
        sliceobj = canonical_slicers(sliceobj, self.shape)
        # Indices into sliceobj referring to image axes
        ax_inds = [i for i, obj in enumerate(sliceobj) if obj is not None]
        assert len(ax_inds) == len(self.shape)
        frame_mapping = get_frame_order(self._subheader._mlist)
        # Analyze index for 4th axis
        slice3 = sliceobj[ax_inds[3]]
        # We will load volume by volume.  Make slicer into volume by dropping
        # index over the volume axis
        in_slicer = sliceobj[: ax_inds[3]] + sliceobj[ax_inds[3] + 1 :]
        # int index for 4th axis, load one slice
        if isinstance(slice3, Integral):
            data = self._subheader.data_from_fileobj(frame_mapping[slice3][0])
            return data[in_slicer]
        # slice axis for 4th axis, we will iterate over slices
        out_shape = predict_shape(sliceobj, self.shape)
        out_data = np.empty(out_shape)
        # Slice into output data with out_slicer
        out_slicer = [slice(None)] * len(out_shape)
        # Work out axis corresponding to volume in output
        in2out_ind = slice2outax(len(self.shape), sliceobj)[3]
        # Iterate over specified 4th axis indices
        for i in list(range(self.shape[3]))[slice3]:
            data = self._subheader.data_from_fileobj(frame_mapping[i][0])
            out_slicer[in2out_ind] = i
            out_data[tuple(out_slicer)] = data[in_slicer]
        return out_data


class EcatImage(SpatialImage):
    """Class returns a list of Ecat images, with one image(hdr/data) per frame"""

    header_class = EcatHeader
    subheader_class = EcatSubHeader
    valid_exts = ('.v',)
    files_types = (('image', '.v'), ('header', '.v'))

    header: EcatHeader
    _subheader: EcatSubHeader

    ImageArrayProxy = EcatImageArrayProxy

    def __init__(self, dataobj, affine, header, subheader, mlist, extra=None, file_map=None):
        """Initialize Image

        The image is a combination of
        (array, affine matrix, header, subheader, mlist)
        with optional meta data in `extra`, and filename / file-like objects
        contained in the `file_map`.

        Parameters
        ----------
        dataobj : array-like
            image data
        affine : None or (4,4) array-like
            homogeneous affine giving relationship between voxel coords and
            world coords.
        header : None or header instance
            meta data for this image format
        subheader : None or subheader instance
            meta data for each sub-image for frame in the image
        mlist : None or array
            Matrix list array giving offset and order of data in file
        extra : None or mapping, optional
            metadata associated with this image that cannot be
            stored in header or subheader
        file_map : mapping, optional
            mapping giving file information for this image format

        Examples
        --------
        >>> import os
        >>> import nibabel as nib
        >>> nibabel_dir = os.path.dirname(nib.__file__)
        >>> from nibabel import ecat
        >>> ecat_file = os.path.join(nibabel_dir,'tests','data','tinypet.v')
        >>> img = ecat.load(ecat_file)
        >>> frame0 = img.get_frame(0)
        >>> frame0.shape == (10, 10, 3)
        True
        >>> data4d = img.get_fdata()
        >>> data4d.shape == (10, 10, 3, 1)
        True
        """
        self._subheader = subheader
        self._mlist = mlist
        self._dataobj = dataobj
        if affine is not None:
            # Check that affine is array-like 4,4.  Maybe this is too strict at
            # this abstract level, but so far I think all image formats we know
            # do need 4,4.
            affine = np.array(affine, dtype=np.float64, copy=True)
            if not affine.shape == (4, 4):
                raise ValueError('Affine should be shape 4,4')
        self._affine = affine
        if extra is None:
            extra = {}
        self.extra = extra
        self._header = header
        if file_map is None:
            file_map = self.__class__.make_file_map()
        self.file_map = file_map
        self._data_cache = None
        self._fdata_cache = None

    @property
    def affine(self):
        if not self._subheader._check_affines():
            warnings.warn(
                'Affines different across frames, loading affine from FIRST frame', UserWarning
            )
        return self._affine

    def get_frame_affine(self, frame):
        """returns 4X4 affine"""
        return self._subheader.get_frame_affine(frame=frame)

    def get_frame(self, frame, orientation=None):
        """
        Get full volume for a time frame

        :param frame: Time frame index from where to fetch data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing (possibly oriented) raw data
        """
        return self._subheader.data_from_fileobj(frame, orientation)

    def get_data_dtype(self, frame):
        subhdr = self._subheader
        dt = subhdr._get_data_dtype(frame)
        return dt

    @property
    def shape(self):
        x, y, z = self._subheader.get_shape()
        nframes = self._subheader.get_nframes()
        return (x, y, z, nframes)

    def get_mlist(self):
        """get access to the mlist"""
        return self._mlist

    def get_subheaders(self):
        """get access to subheaders"""
        return self._subheader

    @staticmethod
    def _get_fileholders(file_map):
        """returns files specific to header and image of the image
        for ecat .v this is the same image file

        Returns
        -------
        header : file holding header data
        image : file holding image data
        """
        return file_map['header'], file_map['image']

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """class method to create image from mapping
        specified in file_map
        """
        hdr_file, img_file = klass._get_fileholders(file_map)
        # note header and image are in same file
        hdr_fid = hdr_file.get_prepare_fileobj(mode='rb')
        header = klass.header_class.from_fileobj(hdr_fid)
        hdr_copy = header.copy()
        # LOAD MLIST
        mlist = np.zeros((header['num_frames'], 4), dtype=np.int32)
        mlist_data = read_mlist(hdr_fid, hdr_copy.endianness)
        mlist[: len(mlist_data)] = mlist_data
        # LOAD SUBHEADERS
        subheaders = klass.subheader_class(hdr_copy, mlist, hdr_fid)
        # LOAD DATA
        # Class level ImageArrayProxy
        data = klass.ImageArrayProxy(subheaders)
        # Get affine
        if not subheaders._check_affines():
            warnings.warn(
                'Affines different across frames, loading affine from FIRST frame', UserWarning
            )
        aff = subheaders.get_frame_affine()
        img = klass(data, aff, header, subheaders, mlist, extra=None, file_map=file_map)
        return img

    def _get_empty_dir(self):
        """
        Get empty directory entry of the form
        [numAvail, nextDir, previousDir, numUsed]
        """
        return np.array([31, 2, 0, 0], dtype=np.int32)

    def _write_data(self, data, stream, pos, dtype=None, endianness=None):
        """
        Write data to ``stream`` using an array_writer

        :param data: Numpy array containing the dat
        :param stream: The file-like object to write the data to
        :param pos: The position in the stream to write the data to
        :param endianness: Endianness code of the data to write
        """
        if dtype is None:
            dtype = data.dtype

        if endianness is None:
            endianness = native_code

        stream.seek(pos)
        make_array_writer(data.view(data.dtype.newbyteorder(endianness)), dtype).to_fileobj(stream)

    def to_file_map(self, file_map=None):
        """Write ECAT7 image to `file_map` or contained ``self.file_map``

        The format consist of:

        - A main header (512L) with dictionary entries in the form
            [numAvail, nextDir, previousDir, numUsed]
        - For every frame (3D volume in 4D data)
          - A subheader (size = frame_offset)
          - Frame data (3D volume)
        """
        if file_map is None:
            file_map = self.file_map

        # It appears to be necessary to load the data before saving even if the
        # data itself is not used.
        self.get_fdata()
        hdr = self.header
        mlist = self._mlist
        subheaders = self.get_subheaders()
        dir_pos = 512
        entry_pos = dir_pos + 16  # 528
        current_dir = self._get_empty_dir()

        hdr_fh, img_fh = self._get_fileholders(file_map)
        hdrf = hdr_fh.get_prepare_fileobj(mode='wb')
        imgf = hdrf

        # Write main header
        hdr.write_to(hdrf)

        # Write every frames
        for index in range(self.header['num_frames']):
            # Move to subheader offset
            frame_offset = subheaders._get_frame_offset(index) - 512
            imgf.seek(frame_offset)

            # Write subheader
            subhdr = subheaders.subheaders[index]
            imgf.write(subhdr.tobytes())

            # Seek to the next image block
            pos = imgf.tell()
            imgf.seek(pos + 2)

            # Get frame
            image = self._subheader.raw_data_from_fileobj(index)

            # Write frame images
            self._write_data(image, imgf, pos + 2, endianness='>')

            # Move to dictionary offset and write dictionary entry
            self._write_data(mlist[index], imgf, entry_pos, endianness='>')

            entry_pos = entry_pos + 16

            current_dir[0] = current_dir[0] - 1
            current_dir[3] = current_dir[3] + 1

            # Create a new directory is previous one is full
            if current_dir[0] == 0:
                # self._write_dir(current_dir, imgf, dir_pos)
                self._write_data(current_dir, imgf, dir_pos)
                current_dir = self._get_empty_dir()
                current_dir[3] = dir_pos / 512
                dir_pos = mlist[index][2] + 1
                entry_pos = dir_pos + 16

        tmp_avail = current_dir[0]
        tmp_used = current_dir[3]

        # Fill directory with empty data until directory is full
        while current_dir[0] > 0:
            entry_pos = dir_pos + 16 + (16 * current_dir[3])
            self._write_data(np.zeros(4, dtype=np.int32), imgf, entry_pos)
            current_dir[0] = current_dir[0] - 1
            current_dir[3] = current_dir[3] + 1

        current_dir[0] = tmp_avail
        current_dir[3] = tmp_used

        # Write directory index
        self._write_data(current_dir, imgf, dir_pos, endianness='>')

    @classmethod
    def from_image(klass, img):
        raise NotImplementedError('Ecat images can only be generated from file objects')

    @classmethod
    def load(klass, filespec):
        return klass.from_filename(filespec)


load = EcatImage.load
