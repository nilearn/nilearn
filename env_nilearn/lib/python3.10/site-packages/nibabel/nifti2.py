# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read / write access to NIfTI2 image format

Format described here:

    https://www.nitrc.org/forum/message.php?msg_id=3738
"""

import numpy as np

from .analyze import AnalyzeHeader
from .batteryrunners import Report
from .filebasedimages import ImageFileError
from .nifti1 import Nifti1Header, Nifti1Image, Nifti1Pair
from .spatialimages import HeaderDataError

r"""
Header struct from : https://www.nitrc.org/forum/message.php?msg_id=3738

/*! \struct nifti_2_header
    \brief Data structure defining the fields in the nifti2 header.
           This binary header should be found at the beginning of a valid
           NIFTI-2 header file.
 */
                        /*************************/ /************/
struct nifti_2_header { /* NIFTI-2 usage         */ /*  offset
                        /*************************/ /************/
int   sizeof_hdr;     /*!< MUST be 540           */   /*   0 */
char  magic[8] ;      /*!< MUST be valid signature.   /*   4 */
short datatype;       /*!< Defines data type!    */   /*  12 */
short bitpix;         /*!< Number bits/voxel.    */   /*  14 */
int64_t dim[8];       /*!< Data array dimensions.*/   /*  16 */
double intent_p1 ;    /*!< 1st intent parameter. */   /*  80 */
double intent_p2 ;    /*!< 2nd intent parameter. */   /*  88 */
double intent_p3 ;    /*!< 3rd intent parameter. */   /*  96 */
double pixdim[8];     /*!< Grid spacings.        */   /* 104 */
int64_t vox_offset;   /*!< Offset into .nii file */   /* 168 */
double scl_slope ;    /*!< Data scaling: slope.  */   /* 176 */
double scl_inter ;    /*!< Data scaling: offset. */   /* 184 */
double cal_max;       /*!< Max display intensity */   /* 192 */
double cal_min;       /*!< Min display intensity */   /* 200 */
double slice_duration;/*!< Time for 1 slice.     */   /* 208 */
double toffset;       /*!< Time axis shift.      */   /* 216 */
int64_t slice_start;  /*!< First slice index.    */   /* 224 */
int64_t slice_end;    /*!< Last slice index.     */   /* 232 */
char  descrip[80];    /*!< any text you like.    */   /* 240 */
char  aux_file[24];   /*!< auxiliary filename.   */   /* 320 */
int qform_code ;      /*!< NIFTI_XFORM_* code.   */   /* 344 */
int sform_code ;      /*!< NIFTI_XFORM_* code.   */   /* 348 */
double quatern_b ;    /*!< Quaternion b param.   */   /* 352 */
double quatern_c ;    /*!< Quaternion c param.   */   /* 360 */
double quatern_d ;    /*!< Quaternion d param.   */   /* 368 */
double qoffset_x ;    /*!< Quaternion x shift.   */   /* 376 */
double qoffset_y ;    /*!< Quaternion y shift.   */   /* 384 */
double qoffset_z ;    /*!< Quaternion z shift.   */   /* 392 */
double srow_x[4] ;    /*!< 1st row affine transform. */  /* 400 */
double srow_y[4] ;    /*!< 2nd row affine transform. */  /* 432 */
double srow_z[4] ;    /*!< 3rd row affine transform. */  /* 464 */
int slice_code ;      /*!< Slice timing order.   */ /* 496 */
int xyzt_units ;      /*!< Units of pixdim[1..4] */ /* 500 */
int intent_code ;     /*!< NIFTI_INTENT_* code.  */ /* 504 */
char intent_name[16]; /*!< 'name' or meaning of data. */ /* 508 */
char dim_info;        /*!< MRI slice ordering.   */      /* 524 */
char unused_str[15];  /*!< unused, filled with \0 */     /* 525 */
} ;                   /**** 540 bytes total ****/
typedef struct nifti_2_header nifti_2_header ;
"""

# nifti2 flat header definition for first 540 bytes
# First number in comments indicates offset in file header in bytes
# fmt: off
header_dtd = [
    ('sizeof_hdr', 'i4'),       # 0; must be 540
    ('magic', 'S4'),            # 4; must be 'ni2\0' or 'n+2\0'
    ('eol_check', 'i1', (4,)),  # 8; must be 0D 0A 1A 0A
    ('datatype', 'i2'),         # 12; it's the datatype
    ('bitpix', 'i2'),           # 14; number of bits per voxel
    ('dim', 'i8', (8,)),        # 16; data array dimensions
    ('intent_p1', 'f8'),        # 80; first intent parameter
    ('intent_p2', 'f8'),        # 88; second intent parameter
    ('intent_p3', 'f8'),        # 96; third intent parameter
    ('pixdim', 'f8', (8,)),     # 104; grid spacings (units below)
    ('vox_offset', 'i8'),       # 168; offset to data in image file
    ('scl_slope', 'f8'),        # 176; data scaling slope
    ('scl_inter', 'f8'),        # 184; data scaling intercept
    ('cal_max', 'f8'),          # 192; max display intensity
    ('cal_min', 'f8'),          # 200; min display intensity
    ('slice_duration', 'f8'),   # 208; time for 1 slice
    ('toffset', 'f8'),          # 216; time axis shift
    ('slice_start', 'i8'),      # 224; first slice index
    ('slice_end', 'i8'),        # 232; last slice index
    ('descrip', 'S80'),         # 240; any text
    ('aux_file', 'S24'),        # 320; auxiliary filename
    ('qform_code', 'i4'),       # 344; xform code
    ('sform_code', 'i4'),       # 348; xform code
    ('quatern_b', 'f8'),        # 352; quaternion b param
    ('quatern_c', 'f8'),        # 360; quaternion c param
    ('quatern_d', 'f8'),        # 368; quaternion d param
    ('qoffset_x', 'f8'),        # 376; quaternion x shift
    ('qoffset_y', 'f8'),        # 384; quaternion y shift
    ('qoffset_z', 'f8'),        # 392; quaternion z shift
    ('srow_x', 'f8', (4,)),     # 400; 1st row affine transform
    ('srow_y', 'f8', (4,)),     # 432; 2nd row affine transform
    ('srow_z', 'f8', (4,)),     # 464; 3rd row affine transform
    ('slice_code', 'i4'),       # 496; slice timing order
    ('xyzt_units', 'i4'),       # 500; inits of pixdim[1..4]
    ('intent_code', 'i4'),      # 504; NIFTI intent code
    ('intent_name', 'S16'),     # 508; name or meaning of data
    ('dim_info', 'u1'),         # 524; MRI slice ordering code
    ('unused_str', 'S15'),      # 525; unused, filled with \0
]  # total 540
# fmt: on

# Full header numpy dtype
header_dtype = np.dtype(header_dtd)


class Nifti2Header(Nifti1Header):
    """Class for NIfTI2 header

    NIfTI2 is a slightly simplified variant of NIfTI1 which replaces 32-bit
    floats with 64-bit floats, and increases some integer widths to 32 or 64
    bits.
    """

    template_dtype = header_dtype
    pair_vox_offset = 0
    single_vox_offset = 544

    # Magics for single and pair
    pair_magic = b'ni2'
    single_magic = b'n+2'

    # Size of header in sizeof_hdr field
    sizeof_hdr = 540

    # Quaternion threshold near 0, based on float64 preicision
    quaternion_threshold = -np.finfo(np.float64).eps * 3

    def get_data_shape(self):
        """Get shape of data

        Examples
        --------
        >>> hdr = Nifti2Header()
        >>> hdr.get_data_shape()
        (0,)
        >>> hdr.set_data_shape((1,2,3))
        >>> hdr.get_data_shape()
        (1, 2, 3)

        Expanding number of dimensions gets default zooms

        >>> hdr.get_zooms()
        (1.0, 1.0, 1.0)

        Notes
        -----
        Does not use Nifti1 freesurfer hack for large vectors described in
        :meth:`Nifti1Header.set_data_shape`
        """
        return AnalyzeHeader.get_data_shape(self)

    def set_data_shape(self, shape):
        """Set shape of data

        If ``ndims == len(shape)`` then we set zooms for dimensions higher than
        ``ndims`` to 1.0

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape

        Notes
        -----
        Does not apply nifti1 Freesurfer hack for long vectors (see
        :meth:`Nifti1Header.set_data_shape`)
        """
        AnalyzeHeader.set_data_shape(self, shape)

    @classmethod
    def default_structarr(klass, endianness=None):
        """Create empty header binary block with given endianness"""
        hdr_data = super().default_structarr(endianness)
        hdr_data['eol_check'] = (13, 10, 26, 10)
        return hdr_data

    """ Checks only below here """

    @classmethod
    def _get_checks(klass):
        # Add our own checks
        return super()._get_checks() + (klass._chk_eol_check,)

    @staticmethod
    def _chk_eol_check(hdr, fix=False):
        rep = Report(HeaderDataError)
        if np.all(hdr['eol_check'] == (13, 10, 26, 10)):
            return hdr, rep
        if np.all(hdr['eol_check'] == 0):
            rep.problem_level = 20
            rep.problem_msg = 'EOL check all 0'
            if fix:
                hdr['eol_check'] = (13, 10, 26, 10)
                rep.fix_msg = 'setting EOL check to 13, 10, 26, 10'
            return hdr, rep
        rep.problem_level = 40
        rep.problem_msg = (
            'EOL check not 0 or 13, 10, 26, 10; data may be corrupted by EOL conversion'
        )
        if fix:
            hdr['eol_check'] = (13, 10, 26, 10)
            rep.fix_msg = 'setting EOL check to 13, 10, 26, 10'
        return hdr, rep

    @classmethod
    def may_contain_header(klass, binaryblock):
        if len(binaryblock) < klass.sizeof_hdr:
            return False

        hdr_struct = np.ndarray(
            shape=(), dtype=header_dtype, buffer=binaryblock[: klass.sizeof_hdr]
        )
        bs_hdr_struct = hdr_struct.byteswap()
        return 540 in (hdr_struct['sizeof_hdr'], bs_hdr_struct['sizeof_hdr'])


class Nifti2PairHeader(Nifti2Header):
    """Class for NIfTI2 pair header"""

    # Signal whether this is single (header + data) file
    is_single = False


class Nifti2Pair(Nifti1Pair):
    """Class for NIfTI2 format image, header pair"""

    header_class = Nifti2PairHeader
    _meta_sniff_len = header_class.sizeof_hdr


class Nifti2Image(Nifti1Image):
    """Class for single file NIfTI2 format image"""

    header_class = Nifti2Header
    _meta_sniff_len = header_class.sizeof_hdr


def load(filename):
    """Load NIfTI2 single or pair image from `filename`

    Parameters
    ----------
    filename : str
        filename of image to be loaded

    Returns
    -------
    img : Nifti2Image or Nifti2Pair
        nifti2 single or pair image instance

    Raises
    ------
    ImageFileError
        if `filename` doesn't look like nifti2;
    OSError
        if `filename` does not exist.
    """
    try:
        img = Nifti2Image.load(filename)
    except ImageFileError:
        return Nifti2Pair.load(filename)
    return img


def save(img, filename):
    """Save NIfTI2 single or pair to `filename`

    Parameters
    ----------
    filename : str
        filename to which to save image
    """
    try:
        Nifti2Image.instance_to_filename(img, filename)
    except ImageFileError:
        Nifti2Pair.instance_to_filename(img, filename)
