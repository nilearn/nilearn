# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read / write access to NIfTI1 image format

NIfTI1 format defined at http://nifti.nimh.nih.gov/nifti-1/
"""

from __future__ import annotations

import json
import sys
import typing as ty
import warnings
from io import BytesIO

import numpy as np
import numpy.linalg as npl

if sys.version_info < (3, 13):
    from typing_extensions import Self, TypeVar  # PY312
else:
    from typing import Self, TypeVar

from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes

if ty.TYPE_CHECKING:
    import pydicom as pdcm

    have_dicom = True
    DicomDataset = pdcm.Dataset
else:
    pdcm, have_dicom, _ = optional_package('pydicom')
    if have_dicom:
        DicomDataset = pdcm.Dataset
    else:
        DicomDataset = ty.Any

T = TypeVar('T', default=bytes)

# nifti1 flat header definition for Analyze-like first 348 bytes
# first number in comments indicates offset in file header in bytes
# fmt: off
header_dtd = [
    ('sizeof_hdr', 'i4'),      # 0; must be 348
    ('data_type', 'S10'),      # 4; unused
    ('db_name', 'S18'),        # 14; unused
    ('extents', 'i4'),         # 32; unused
    ('session_error', 'i2'),   # 36; unused
    ('regular', 'S1'),         # 38; unused
    ('dim_info', 'u1'),        # 39; MRI slice ordering code
    ('dim', 'i2', (8,)),       # 40; data array dimensions
    ('intent_p1', 'f4'),       # 56; first intent parameter
    ('intent_p2', 'f4'),       # 60; second intent parameter
    ('intent_p3', 'f4'),       # 64; third intent parameter
    ('intent_code', 'i2'),     # 68; NIFTI intent code
    ('datatype', 'i2'),        # 70; it's the datatype
    ('bitpix', 'i2'),          # 72; number of bits per voxel
    ('slice_start', 'i2'),     # 74; first slice index
    ('pixdim', 'f4', (8,)),    # 76; grid spacings (units below)
    ('vox_offset', 'f4'),      # 108; offset to data in image file
    ('scl_slope', 'f4'),       # 112; data scaling slope
    ('scl_inter', 'f4'),       # 116; data scaling intercept
    ('slice_end', 'i2'),       # 120; last slice index
    ('slice_code', 'u1'),      # 122; slice timing order
    ('xyzt_units', 'u1'),      # 123; units of pixdim[1..4]
    ('cal_max', 'f4'),         # 124; max display intensity
    ('cal_min', 'f4'),         # 128; min display intensity
    ('slice_duration', 'f4'),  # 132; time for 1 slice
    ('toffset', 'f4'),         # 136; time axis shift
    ('glmax', 'i4'),           # 140; unused
    ('glmin', 'i4'),           # 144; unused
    ('descrip', 'S80'),        # 148; any text
    ('aux_file', 'S24'),       # 228; auxiliary filename
    ('qform_code', 'i2'),      # 252; xform code
    ('sform_code', 'i2'),      # 254; xform code
    ('quatern_b', 'f4'),       # 256; quaternion b param
    ('quatern_c', 'f4'),       # 260; quaternion c param
    ('quatern_d', 'f4'),       # 264; quaternion d param
    ('qoffset_x', 'f4'),       # 268; quaternion x shift
    ('qoffset_y', 'f4'),       # 272; quaternion y shift
    ('qoffset_z', 'f4'),       # 276; quaternion z shift
    ('srow_x', 'f4', (4,)),    # 280; 1st row affine transform
    ('srow_y', 'f4', (4,)),    # 296; 2nd row affine transform
    ('srow_z', 'f4', (4,)),    # 312; 3rd row affine transform
    ('intent_name', 'S16'),    # 328; name or meaning of data
    ('magic', 'S4'),           # 344; must be 'ni1\0' or 'n+1\0'
]
# fmt: on

# Full header numpy dtype
header_dtype = np.dtype(header_dtd)

# datatypes not in analyze format, with codes
if have_binary128():
    # Only enable 128 bit floats if we really have IEEE binary 128 longdoubles
    _float128t: type[np.generic] = np.longdouble
    _complex256t: type[np.generic] = np.clongdouble
else:
    _float128t = np.void
    _complex256t = np.void

_dtdefs = (  # code, label, dtype definition, niistring
    (0, 'none', np.void, ''),
    (1, 'binary', np.void, ''),
    (2, 'uint8', np.uint8, 'NIFTI_TYPE_UINT8'),
    (4, 'int16', np.int16, 'NIFTI_TYPE_INT16'),
    (8, 'int32', np.int32, 'NIFTI_TYPE_INT32'),
    (16, 'float32', np.float32, 'NIFTI_TYPE_FLOAT32'),
    (32, 'complex64', np.complex64, 'NIFTI_TYPE_COMPLEX64'),
    (64, 'float64', np.float64, 'NIFTI_TYPE_FLOAT64'),
    (128, 'RGB', np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]), 'NIFTI_TYPE_RGB24'),
    (255, 'all', np.void, ''),
    (256, 'int8', np.int8, 'NIFTI_TYPE_INT8'),
    (512, 'uint16', np.uint16, 'NIFTI_TYPE_UINT16'),
    (768, 'uint32', np.uint32, 'NIFTI_TYPE_UINT32'),
    (1024, 'int64', np.int64, 'NIFTI_TYPE_INT64'),
    (1280, 'uint64', np.uint64, 'NIFTI_TYPE_UINT64'),
    (1536, 'float128', _float128t, 'NIFTI_TYPE_FLOAT128'),
    (1792, 'complex128', np.complex128, 'NIFTI_TYPE_COMPLEX128'),
    (2048, 'complex256', _complex256t, 'NIFTI_TYPE_COMPLEX256'),
    (
        2304,
        'RGBA',
        np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')]),
        'NIFTI_TYPE_RGBA32',
    ),
)

# Make full code alias bank, including dtype column
data_type_codes = make_dt_codes(_dtdefs)

# Transform (qform, sform) codes
xform_codes = Recoder(
    (  # code, label, niistring
        (0, 'unknown', 'NIFTI_XFORM_UNKNOWN'),
        (1, 'scanner', 'NIFTI_XFORM_SCANNER_ANAT'),
        (2, 'aligned', 'NIFTI_XFORM_ALIGNED_ANAT'),
        (3, 'talairach', 'NIFTI_XFORM_TALAIRACH'),
        (4, 'mni', 'NIFTI_XFORM_MNI_152'),
        (5, 'template', 'NIFTI_XFORM_TEMPLATE_OTHER'),
    ),
    fields=('code', 'label', 'niistring'),
)

# unit codes
unit_codes = Recoder(
    (  # code, label
        (0, 'unknown'),
        (1, 'meter'),
        (2, 'mm'),
        (3, 'micron'),
        (8, 'sec'),
        (16, 'msec'),
        (24, 'usec'),
        (32, 'hz'),
        (40, 'ppm'),
        (48, 'rads'),
    ),
    fields=('code', 'label'),
)

slice_order_codes = Recoder(
    (  # code, label
        (0, 'unknown'),
        (1, 'sequential increasing', 'seq inc'),
        (2, 'sequential decreasing', 'seq dec'),
        (3, 'alternating increasing', 'alt inc'),
        (4, 'alternating decreasing', 'alt dec'),
        (5, 'alternating increasing 2', 'alt inc 2'),
        (6, 'alternating decreasing 2', 'alt dec 2'),
    ),
    fields=('code', 'label'),
)

intent_codes = Recoder(
    (
        # code, label, parameters description tuple
        (0, 'none', (), 'NIFTI_INTENT_NONE'),
        (2, 'correlation', ('p1 = DOF',), 'NIFTI_INTENT_CORREL'),
        (3, 't test', ('p1 = DOF',), 'NIFTI_INTENT_TTEST'),
        (4, 'f test', ('p1 = numerator DOF', 'p2 = denominator DOF'), 'NIFTI_INTENT_FTEST'),
        (5, 'z score', (), 'NIFTI_INTENT_ZSCORE'),
        (6, 'chi2', ('p1 = DOF',), 'NIFTI_INTENT_CHISQ'),
        # two parameter beta distribution
        (7, 'beta', ('p1=a', 'p2=b'), 'NIFTI_INTENT_BETA'),
        # Prob(x) = (p1 choose x) * p2^x * (1-p2)^(p1-x), for x=0,1,...,p1
        (
            8,
            'binomial',
            ('p1 = number of trials', 'p2 = probability per trial'),
            'NIFTI_INTENT_BINOM',
        ),
        # 2 parameter gamma
        # Density(x) proportional to  # x^(p1-1) * exp(-p2*x)
        (9, 'gamma', ('p1 = shape, p2 = scale', 2), 'NIFTI_INTENT_GAMMA'),
        (10, 'poisson', ('p1 = mean',), 'NIFTI_INTENT_POISSON'),
        (
            11,
            'normal',
            (
                'p1 = mean',
                'p2 = standard deviation',
            ),
            'NIFTI_INTENT_NORMAL',
        ),
        (
            12,
            'non central f test',
            (
                'p1 = numerator DOF',
                'p2 = denominator DOF',
                'p3 = numerator noncentrality parameter',
            ),
            'NIFTI_INTENT_FTEST_NONC',
        ),
        (
            13,
            'non central chi2',
            (
                'p1 = DOF',
                'p2 = noncentrality parameter',
            ),
            'NIFTI_INTENT_CHISQ_NONC',
        ),
        (
            14,
            'logistic',
            (
                'p1 = location',
                'p2 = scale',
            ),
            'NIFTI_INTENT_LOGISTIC',
        ),
        (15, 'laplace', ('p1 = location', 'p2 = scale'), 'NIFTI_INTENT_LAPLACE'),
        (16, 'uniform', ('p1 = lower end', 'p2 = upper end'), 'NIFTI_INTENT_UNIFORM'),
        (
            17,
            'non central t test',
            ('p1 = DOF', 'p2 = noncentrality parameter'),
            'NIFTI_INTENT_TTEST_NONC',
        ),
        (18, 'weibull', ('p1 = location', 'p2 = scale, p3 = power'), 'NIFTI_INTENT_WEIBULL'),
        # p1 = 1 = 'half normal' distribution
        # p1 = 2 = Rayleigh distribution
        # p1 = 3 = Maxwell-Boltzmann distribution.
        (19, 'chi', ('p1 = DOF',), 'NIFTI_INTENT_CHI'),
        (20, 'inverse gaussian', ('pi = mu', 'p2 = lambda'), 'NIFTI_INTENT_INVGAUSS'),
        (21, 'extreme value 1', ('p1 = location', 'p2 = scale'), 'NIFTI_INTENT_EXTVAL'),
        (22, 'p value', (), 'NIFTI_INTENT_PVAL'),
        (23, 'log p value', (), 'NIFTI_INTENT_LOGPVAL'),
        (24, 'log10 p value', (), 'NIFTI_INTENT_LOG10PVAL'),
        (1001, 'estimate', (), 'NIFTI_INTENT_ESTIMATE'),
        (1002, 'label', (), 'NIFTI_INTENT_LABEL'),
        (1003, 'neuroname', (), 'NIFTI_INTENT_NEURONAME'),
        (1004, 'general matrix', ('p1 = M', 'p2 = N'), 'NIFTI_INTENT_GENMATRIX'),
        (1005, 'symmetric matrix', ('p1 = M',), 'NIFTI_INTENT_SYMMATRIX'),
        (1006, 'displacement vector', (), 'NIFTI_INTENT_DISPVECT'),
        (1007, 'vector', (), 'NIFTI_INTENT_VECTOR'),
        (1008, 'pointset', (), 'NIFTI_INTENT_POINTSET'),
        (1009, 'triangle', (), 'NIFTI_INTENT_TRIANGLE'),
        (1010, 'quaternion', (), 'NIFTI_INTENT_QUATERNION'),
        (1011, 'dimensionless', (), 'NIFTI_INTENT_DIMLESS'),
        (
            2001,
            'time series',
            (),
            'NIFTI_INTENT_TIME_SERIES',
            'NIFTI_INTENT_TIMESERIES',
        ),  # this mis-spell occurs in the wild
        (2002, 'node index', (), 'NIFTI_INTENT_NODE_INDEX'),
        (2003, 'rgb vector', (), 'NIFTI_INTENT_RGB_VECTOR'),
        (2004, 'rgba vector', (), 'NIFTI_INTENT_RGBA_VECTOR'),
        (2005, 'shape', (), 'NIFTI_INTENT_SHAPE'),
        # FSL-specific intent codes - codes used by FNIRT
        # ($FSLDIR/warpfns/fnirt_file_reader.h:104)
        (2006, 'fnirt disp field', (), 'FSL_FNIRT_DISPLACEMENT_FIELD'),
        (2007, 'fnirt cubic spline coef', (), 'FSL_CUBIC_SPLINE_COEFFICIENTS'),
        (2008, 'fnirt dct coef', (), 'FSL_DCT_COEFFICIENTS'),
        (2009, 'fnirt quad spline coef', (), 'FSL_QUADRATIC_SPLINE_COEFFICIENTS'),
        # FSL-specific intent codes - codes used by TOPUP
        # ($FSLDIR/topup/topup_file_io.h:104)
        (2016, 'topup cubic spline coef ', (), 'FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS'),
        (2017, 'topup quad spline coef', (), 'FSL_TOPUP_QUADRATIC_SPLINE_COEFFICIENTS'),
        (2018, 'topup field', (), 'FSL_TOPUP_FIELD'),
    ),
    fields=('code', 'label', 'parameters', 'niistring'),
)


class NiftiExtension(ty.Generic[T]):
    """Base class for NIfTI header extensions.

    This class provides access to the extension content in various forms.
    For simple extensions that expose data as bytes, text or JSON, this class
    is sufficient. More complex extensions should be implemented as subclasses
    that provide custom serialization/deserialization methods.

    Efficiency note:

    This class assumes that the runtime representation of the extension content
    is mutable. Once a runtime representation is set, it is cached and will be
    serialized on any attempt to access the extension content as bytes, including
    determining the size of the extension in the NIfTI file.

    If the runtime representation is never accessed, the raw bytes will be used
    without modification. While avoiding unnecessary deserialization, if there
    are bytestrings that do not produce a valid runtime representation, they will
    be written as-is, and may cause errors downstream.
    """

    code: int
    encoding: str | None = None
    _raw: bytes
    _object: T | None = None

    def __init__(
        self,
        code: int | str,
        content: bytes = b'',
        object: T | None = None,
    ) -> None:
        """
        Parameters
        ----------
        code : int or str
          Canonical extension code as defined in the NIfTI standard, given
          either as integer or corresponding label
          (see :data:`~nibabel.nifti1.extension_codes`)
        content : bytes, optional
          Extension content as read from the NIfTI file header.
        object : optional
          Extension content in runtime form.
        """
        try:
            self.code = extension_codes.code[code]  # type: ignore[assignment]
        except KeyError:
            self.code = code  # type: ignore[assignment]
        self._raw = content
        if object is not None:
            self._object = object

    @property
    def _content(self):
        return self.get_object()

    @classmethod
    def from_bytes(cls, content: bytes) -> Self:
        """Create an extension from raw bytes.

        This constructor may only be used in extension classes with a class
        attribute `code` to indicate the extension type.
        """
        if not hasattr(cls, 'code'):
            raise NotImplementedError('from_bytes() requires a class attribute `code`')
        return cls(cls.code, content=content)

    @classmethod
    def from_object(cls, obj: T) -> Self:
        """Create an extension from a runtime object.

        This constructor may only be used in extension classes with a class
        attribute `code` to indicate the extension type.
        """
        if not hasattr(cls, 'code'):
            raise NotImplementedError('from_object() requires a class attribute `code`')
        return cls(cls.code, object=obj)

    # Handle (de)serialization of extension content
    # Subclasses may implement these methods to provide an alternative
    # view of the extension content. If left unimplemented, the content
    # must be bytes and is not modified.
    def _mangle(self, obj: T) -> bytes:
        raise NotImplementedError

    def _unmangle(self, content: bytes) -> T:
        raise NotImplementedError

    def _sync(self) -> None:
        """Synchronize content with object.

        This permits the runtime representation to be modified in-place
        and updates the bytes representation accordingly.
        """
        if self._object is not None:
            self._raw = self._mangle(self._object)

    def __repr__(self) -> str:
        try:
            code = extension_codes.label[self.code]
        except KeyError:
            # deal with unknown codes
            code = self.code
        return f'{self.__class__.__name__}({code}, {self._raw!r})'

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.code == other.code
            and self.content == other.content
        )

    def __ne__(self, other):
        return not self == other

    def get_code(self):
        """Return the canonical extension type code."""
        return self.code

    # Canonical access to extension content
    # Follows the lead of httpx.Response .content, .text and .json()
    # properties/methods
    @property
    def content(self) -> bytes:
        """Return the extension content as raw bytes."""
        self._sync()
        return self._raw

    @property
    def text(self) -> str:
        """Attempt to decode the extension content as text.

        The encoding is determined by the `encoding` attribute, which may be
        set by the user or subclass. If not set, the default encoding is 'utf-8'.
        """
        return self.content.decode(self.encoding or 'utf-8')

    def json(self) -> ty.Any:
        """Attempt to decode the extension content as JSON.

        If the content is not valid JSON, a JSONDecodeError or UnicodeDecodeError
        will be raised.
        """
        return json.loads(self.content)

    def get_object(self) -> T:
        """Return the extension content in its runtime representation.

        This method may return a different type for each extension type.
        For simple use cases, consider using ``.content``, ``.text`` or ``.json()``
        instead.
        """
        if self._object is None:
            self._object = self._unmangle(self._raw)
        return self._object

    # Backwards compatibility
    get_content = get_object

    def get_sizeondisk(self) -> int:
        """Return the size of the extension in the NIfTI file."""
        # need raw value size plus 8 bytes for esize and ecode, rounded up to next 16 bytes
        # Rounding C+8 up to M is done by (C+8 + (M-1)) // M * M
        return (len(self.content) + 23) // 16 * 16

    def write_to(self, fileobj: ty.BinaryIO, byteswap: bool = False) -> None:
        """Write header extensions to fileobj

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method
        byteswap : boolean
          Flag if byteswapping the data is required.

        Returns
        -------
        None
        """
        extstart = fileobj.tell()
        rawsize = self.get_sizeondisk()  # Calls _sync()
        # write esize and ecode first
        extinfo = np.array((rawsize, self.code), dtype=np.int32)
        if byteswap:
            extinfo = extinfo.byteswap()
        fileobj.write(extinfo.tobytes())
        # followed by the actual extension content, synced above
        fileobj.write(self._raw)
        # be nice and zero out remaining part of the extension till the
        # next 16 byte border
        pad = extstart + rawsize - fileobj.tell()
        if pad:
            fileobj.write(bytes(pad))


class Nifti1Extension(NiftiExtension[T]):
    """Baseclass for NIfTI1 header extensions.

    This class is sufficient to handle very simple text-based extensions, such
    as `comment`. More sophisticated extensions should/will be supported by
    dedicated subclasses.
    """

    code = 0  # Default to unknown extension

    def _unmangle(self, value: bytes) -> T:
        """Convert the extension content into its runtime representation.

        The default implementation does nothing at all.

        Parameters
        ----------
        value : str
          Extension content as read from file.

        Returns
        -------
        The same object that was passed as `value`.

        Notes
        -----
        Subclasses should reimplement this method to provide the desired
        unmangling procedure and may return any type of object.
        """
        return value  # type: ignore[return-value]

    def _mangle(self, value: T) -> bytes:
        """Convert the extension content into NIfTI file header representation.

        The default implementation does nothing at all.

        Parameters
        ----------
        value : str
          Extension content in runtime form.

        Returns
        -------
        str

        Notes
        -----
        Subclasses should reimplement this method to provide the desired
        mangling procedure.
        """
        return value  # type: ignore[return-value]


class Nifti1DicomExtension(Nifti1Extension[DicomDataset]):
    """NIfTI1 DICOM header extension

    This class is a thin wrapper around pydicom to read a binary DICOM
    byte string. If pydicom is available, content is exposed as a Dicom Dataset.
    Otherwise, this silently falls back to the standard NiftiExtension class
    and content is the raw bytestring loaded directly from the nifti file
    header.
    """

    code = 2
    _is_implicit_VR: bool = False
    _is_little_endian: bool = True

    def __init__(
        self,
        code: int | str,
        content: bytes | DicomDataset | None = None,
        parent_hdr: Nifti1Header | None = None,
    ) -> None:
        """
        Parameters
        ----------
        code : int or str
          Canonical extension code as defined in the NIfTI standard, given
          either as integer or corresponding label
          (see :data:`~nibabel.nifti1.extension_codes`)
        content : bytes or pydicom Dataset or None
          Extension content - either a bytestring as read from the NIfTI file
          header or an existing pydicom Dataset. If a bystestring, the content
          is converted into a Dataset on initialization. If None, a new empty
          Dataset is created.
        parent_hdr : :class:`~nibabel.nifti1.Nifti1Header`, optional
          If a dicom extension belongs to an existing
          :class:`~nibabel.nifti1.Nifti1Header`, it may be provided here to
          ensure that the DICOM dataset is written with correctly corresponding
          endianness; otherwise it is assumed the dataset is little endian.

        Notes
        -----

        code should always be 2 for DICOM.
        """

        if code != 2:
            raise ValueError(f'code must be 2 for DICOM. Got {code}.')

        if content is None:
            content = pdcm.Dataset()

        if parent_hdr is not None:
            self._is_little_endian = parent_hdr.endianness == '<'

        if isinstance(content, pdcm.dataset.Dataset):
            super().__init__(code, object=content)
        elif isinstance(content, bytes):  # Got a byte string - unmangle it
            self._is_implicit_VR = self._guess_implicit_VR(content)
            super().__init__(code, content=content)
        else:
            raise TypeError(
                f'content must be either a bytestring or a pydicom Dataset. '
                f'Got {content.__class__}'
            )

    @staticmethod
    def _guess_implicit_VR(content) -> bool:
        """Try to guess DICOM syntax by checking for valid VRs.

        Without a DICOM Transfer Syntax, it's difficult to tell if Value
        Representations (VRs) are included in the DICOM encoding or not.
        This reads where the first VR would be and checks it against a list of
        valid VRs
        """
        potential_vr = content[4:6].decode()
        return potential_vr not in pdcm.values.converters.keys()

    def _unmangle(self, obj: bytes) -> DicomDataset:
        return pdcm.filereader.read_dataset(
            BytesIO(obj),
            self._is_implicit_VR,
            self._is_little_endian,
        )

    def _mangle(self, dataset: DicomDataset) -> bytes:
        bio = BytesIO()
        dio = pdcm.filebase.DicomFileLike(bio)
        dio.is_implicit_VR = self._is_implicit_VR
        dio.is_little_endian = self._is_little_endian
        ds_len = pdcm.filewriter.write_dataset(dio, dataset)
        dio.seek(0)
        return dio.read(ds_len)


# NIfTI header extension type codes (ECODE)
# see nifti1_io.h for a complete list of all known extensions and
# references to their description or contacts of the respective
# initiators
extension_codes = Recoder(
    (
        (0, 'ignore', Nifti1Extension),
        (2, 'dicom', Nifti1DicomExtension if have_dicom else Nifti1Extension),
        (4, 'afni', Nifti1Extension),
        (6, 'comment', Nifti1Extension),
        (8, 'xcede', Nifti1Extension),
        (10, 'jimdiminfo', Nifti1Extension),
        (12, 'workflow_fwds', Nifti1Extension),
        (14, 'freesurfer', Nifti1Extension),
        (16, 'pypickle', Nifti1Extension),
        (18, 'mind_ident', NiftiExtension),
        (20, 'b_value', NiftiExtension),
        (22, 'spherical_direction', NiftiExtension),
        (24, 'dt_component', NiftiExtension),
        (26, 'shc_degreeorder', NiftiExtension),
        (28, 'voxbo', NiftiExtension),
        (30, 'caret', NiftiExtension),
        ## Defined in nibabel.cifti2.parse_cifti2
        # (32, 'cifti', Cifti2Extension),
        (34, 'variable_frame_timing', NiftiExtension),
        (36, 'unassigned', NiftiExtension),
        (38, 'eval', NiftiExtension),
        (40, 'matlab', NiftiExtension),
        (42, 'quantiphyse', NiftiExtension),
        (44, 'mrs', Nifti1Extension),
    ),
    fields=('code', 'label', 'handler'),
)


class Nifti1Extensions(list):
    """Simple extension collection, implemented as a list-subclass."""

    def count(self, ecode):
        """Returns the number of extensions matching a given *ecode*.

        Parameters
        ----------
        code : int | str
            The ecode can be specified either literal or as numerical value.
        """
        count = 0
        code = extension_codes.code[ecode]
        for e in self:
            if e.get_code() == code:
                count += 1
        return count

    def get_codes(self):
        """Return a list of the extension code of all available extensions"""
        return [e.get_code() for e in self]

    def get_sizeondisk(self):
        """Return the size of the complete header extensions in the NIfTI file."""
        return np.sum([e.get_sizeondisk() for e in self])

    def __repr__(self):
        return 'Nifti1Extensions({})'.format(', '.join(str(e) for e in self))

    def write_to(self, fileobj, byteswap):
        """Write header extensions to fileobj

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method
        byteswap : boolean
          Flag if byteswapping the data is required.

        Returns
        -------
        None
        """
        for e in self:
            e.write_to(fileobj, byteswap)

    @classmethod
    def from_fileobj(klass, fileobj, size, byteswap):
        """Read header extensions from a fileobj

        Parameters
        ----------
        fileobj : file-like object
            We begin reading the extensions at the current file position
        size : int
            Number of bytes to read. If negative, fileobj will be read till its
            end.
        byteswap : boolean
            Flag if byteswapping the read data is required.

        Returns
        -------
        An extension list. This list might be empty in case not extensions
        were present in fileobj.
        """
        # make empty extension list
        extensions = klass()
        # assume the file pointer is at the beginning of any extensions.
        # read until the whole header is parsed (each extension is a multiple
        # of 16 bytes) or in case of a separate header file till the end
        # (break inside the body)
        while size >= 16 or size < 0:
            # the next 8 bytes should have esize and ecode
            ext_def = fileobj.read(8)
            # nothing was read and instructed to read till the end
            # -> assume all extensions where parsed and break
            if not len(ext_def) and size < 0:
                break
            # otherwise there should be a full extension header
            if not len(ext_def) == 8:
                raise HeaderDataError('failed to read extension header')
            ext_def = np.frombuffer(ext_def, dtype=np.int32)
            if byteswap:
                ext_def = ext_def.byteswap()
            # be extra verbose
            ecode = ext_def[1]
            esize = ext_def[0]
            if esize % 16:
                warnings.warn(
                    'Extension size is not a multiple of 16 bytes; '
                    'Assuming size is correct and hoping for the best',
                    UserWarning,
                )
            # read extension itself; esize includes the 8 bytes already read
            evalue = fileobj.read(int(esize - 8))
            if not len(evalue) == esize - 8:
                raise HeaderDataError('failed to read extension content')
            # note that we read a full extension
            size -= esize
            # store raw extension content, but strip trailing NULL chars
            evalue = evalue.rstrip(b'\x00')
            # 'extension_codes' also knows the best implementation to handle
            # a particular extension type
            try:
                ext = extension_codes.handler[ecode](ecode, evalue)
            except KeyError:
                # unknown extension type
                # XXX complain or fail or go with a generic extension
                ext = Nifti1Extension(ecode, evalue)
            extensions.append(ext)
        return extensions


class Nifti1Header(SpmAnalyzeHeader):
    """Class for NIfTI1 header

    The NIfTI1 header has many more coded fields than the simpler Analyze
    variants.  NIfTI1 headers also have extensions.

    Nifti allows the header to be a separate file, as part of a nifti image /
    header pair, or to precede the data in a single file.  The object needs to
    know which type it is, in order to manage the voxel offset pointing to the
    data, extension reading, and writing the correct magic string.

    This class handles the header-preceding-data case.
    """

    # Copies of module level definitions
    template_dtype = header_dtype
    _data_type_codes = data_type_codes

    # fields with recoders for their values
    _field_recoders = {
        'datatype': data_type_codes,
        'qform_code': xform_codes,
        'sform_code': xform_codes,
        'intent_code': intent_codes,
        'slice_code': slice_order_codes,
    }

    # data scaling capabilities
    has_data_slope = True
    has_data_intercept = True

    # Extension class; should implement __call__ for construction, and
    # ``from_fileobj`` for reading from file
    exts_klass = Nifti1Extensions

    # Signal whether this is single (header + data) file
    is_single = True

    # Default voxel data offsets for single and pair
    pair_vox_offset = 0
    single_vox_offset = 352

    # Magics for single and pair
    pair_magic = b'ni1'
    single_magic = b'n+1'

    # Quaternion threshold near 0, based on float32 precision
    quaternion_threshold = np.finfo(np.float32).eps * 3

    def __init__(self, binaryblock=None, endianness=None, check=True, extensions=()):
        """Initialize header from binary data block and extensions"""
        super().__init__(binaryblock, endianness, check)
        self.extensions = self.exts_klass(extensions)

    def copy(self):
        """Return copy of header

        Take reference to extensions as well as copy of header contents
        """
        return self.__class__(self.binaryblock, self.endianness, False, self.extensions)

    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        raw_str = fileobj.read(klass.template_dtype.itemsize)
        hdr = klass(raw_str, endianness, check)
        # Read next 4 bytes to see if we have extensions.  The nifti standard
        # has this as a 4 byte string; if the first value is not zero, then we
        # have extensions.
        extension_status = fileobj.read(4)
        # Need to test *slice* of extension_status to preserve byte string type
        # on Python 3
        if len(extension_status) < 4 or extension_status[0:1] == b'\x00':
            return hdr
        # If this is a detached header file read to end
        if not klass.is_single:
            extsize = -1
        else:  # otherwise read until the beginning of the data
            extsize = hdr._structarr['vox_offset'] - fileobj.tell()
        byteswap = endian_codes['native'] != hdr.endianness
        hdr.extensions = klass.exts_klass.from_fileobj(fileobj, extsize, byteswap)
        return hdr

    def write_to(self, fileobj):
        # First check that vox offset is large enough; set if necessary
        if self.is_single:
            vox_offset = self._structarr['vox_offset']
            min_vox_offset = self.single_vox_offset + self.extensions.get_sizeondisk()
            if vox_offset == 0:  # vox offset unset; set as necessary
                self._structarr['vox_offset'] = min_vox_offset
            elif vox_offset < min_vox_offset:
                raise HeaderDataError(
                    f'vox offset set to {vox_offset}, but need at least {min_vox_offset}'
                )
        super().write_to(fileobj)
        # Write extensions
        if len(self.extensions) == 0:
            # If single file, write required 0 stream to signal no extensions
            if self.is_single:
                fileobj.write(b'\x00' * 4)
            return
        # Signal there are extensions that follow
        fileobj.write(b'\x01\x00\x00\x00')
        byteswap = endian_codes['native'] != self.endianness
        self.extensions.write_to(fileobj, byteswap)

    def get_best_affine(self):
        """Select best of available transforms"""
        hdr = self._structarr
        if hdr['sform_code'] != 0:
            return self.get_sform()
        if hdr['qform_code'] != 0:
            return self.get_qform()
        return self.get_base_affine()

    @classmethod
    def default_structarr(klass, endianness=None):
        """Create empty header binary block with given endianness"""
        hdr_data = super().default_structarr(endianness)
        if klass.is_single:
            hdr_data['magic'] = klass.single_magic
        else:
            hdr_data['magic'] = klass.pair_magic
        return hdr_data

    @classmethod
    def from_header(klass, header=None, check=True):
        """Class method to create header from another header

        Extend Analyze header copy by copying extensions from other Nifti
        types.

        Parameters
        ----------
        header : ``Header`` instance or mapping
           a header of this class, or another class of header for
           conversion to this type
        check : {True, False}
           whether to check header for integrity

        Returns
        -------
        hdr : header instance
           fresh header instance of our own class
        """
        new_hdr = super().from_header(header, check)
        if isinstance(header, Nifti1Header):
            new_hdr.extensions[:] = header.extensions[:]
        return new_hdr

    def get_data_shape(self):
        """Get shape of data

        Examples
        --------
        >>> hdr = Nifti1Header()
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
        Applies freesurfer hack for large vectors described in `issue 100`_ and
        `save_nifti.m <save77_>`_.

        Allows for freesurfer hack for 7th order icosahedron surface described
        in `issue 309`_, load_nifti.m_, and `save_nifti.m <save50_>`_.
        """
        shape = super().get_data_shape()
        # Apply freesurfer hack for large vectors
        if shape[:3] == (-1, 1, 1):
            vec_len = int(self._structarr['glmin'])
            if vec_len == 0:
                raise HeaderDataError(
                    '-1 in dim[1] but 0 in glmin; inconsistent freesurfer type header?'
                )
            return (vec_len, 1, 1) + shape[3:]
        # Apply freesurfer hack for ico7 surface
        elif shape[:3] == (27307, 1, 6):
            return (163842, 1, 1) + shape[3:]
        else:  # Normal case
            return shape

    def set_data_shape(self, shape):
        """Set shape of data  # noqa

        If ``ndims == len(shape)`` then we set zooms for dimensions higher than
        ``ndims`` to 1.0

        Nifti1 images can have up to seven dimensions. For FreeSurfer-variant
        Nifti surface files, the first dimension is assumed to correspond to
        vertices/nodes on a surface, and dimensions two and three are
        constrained to have depth of 1. Dimensions 4-7 are constrained only by
        type bounds.

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape

        Notes
        -----
        Applies freesurfer hack for large vectors described in `issue 100`_ and
        `save_nifti.m <save77_>`_.

        Allows for freesurfer hack for 7th order icosahedron surface described
        in `issue 309`_, load_nifti.m_, and `save_nifti.m <save50_>`_.

        The Nifti1 `standard header`_ allows for the following "point set"
        definition of a surface, not currently implemented in nibabel.

        ::

          To signify that the vector value at each voxel is really a
          spatial coordinate (e.g., the vertices or nodes of a surface mesh):
            - dataset must have a 5th dimension
            - intent_code must be NIFTI_INTENT_POINTSET
            - dim[0] = 5
            - dim[1] = number of points
            - dim[2] = dim[3] = dim[4] = 1
            - dim[5] must be the dimensionality of space (e.g., 3 => 3D space).
            - intent_name may describe the object these points come from
              (e.g., "pial", "gray/white" , "EEG", "MEG").

        .. _issue 100: https://github.com/nipy/nibabel/issues/100
        .. _issue 309: https://github.com/nipy/nibabel/issues/309
        .. _save77:
            https://github.com/fieldtrip/fieldtrip/blob/428798b/external/freesurfer/save_nifti.m#L77-L82
        .. _save50:
            https://github.com/fieldtrip/fieldtrip/blob/428798b/external/freesurfer/save_nifti.m#L50-L56
        .. _load_nifti.m:
            https://github.com/fieldtrip/fieldtrip/blob/428798b/external/freesurfer/load_nifti.m#L86-L89
        .. _standard header: http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
        """
        hdr = self._structarr
        shape = tuple(shape)

        # Apply freesurfer hack for ico7 surface
        if shape[:3] == (163842, 1, 1):
            shape = (27307, 1, 6) + shape[3:]
        # Apply freesurfer hack for large vectors
        elif (
            len(shape) >= 3
            and shape[1:3] == (1, 1)
            and shape[0] > np.iinfo(hdr['dim'].dtype.base).max
        ):
            try:
                hdr['glmin'] = shape[0]
            except OverflowError:
                overflow = True
            else:
                overflow = hdr['glmin'] != shape[0]
            if overflow:
                raise HeaderDataError(f'shape[0] {shape[0]} does not fit in glmax datatype')
            warnings.warn(
                'Using large vector Freesurfer hack; header will '
                'not be compatible with SPM or FSL',
                stacklevel=2,
            )
            shape = (-1, 1, 1) + shape[3:]
        super().set_data_shape(shape)

    def set_data_dtype(self, datatype):
        """Set numpy dtype for data from code or dtype or type

        Using :py:class:`int` or ``"int"`` is disallowed, as these types
        will be interpreted as ``np.int64``, which is almost never desired.
        ``np.int64`` is permitted for those intent on making poor choices.

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_data_dtype(np.uint8)
        >>> hdr.get_data_dtype()
        dtype('uint8')
        >>> hdr.set_data_dtype(np.dtype(np.uint8))
        >>> hdr.get_data_dtype()
        dtype('uint8')
        >>> hdr.set_data_dtype('implausible')
        Traceback (most recent call last):
           ...
        nibabel.spatialimages.HeaderDataError: data dtype "implausible" not recognized
        >>> hdr.set_data_dtype('none')
        Traceback (most recent call last):
           ...
        nibabel.spatialimages.HeaderDataError: data dtype "none" known but not supported
        >>> hdr.set_data_dtype(np.void)
        Traceback (most recent call last):
           ...
        nibabel.spatialimages.HeaderDataError: data dtype "<class 'numpy.void'>" known
        but not supported
        >>> hdr.set_data_dtype('int')
        Traceback (most recent call last):
           ...
        ValueError: Invalid data type 'int'. Specify a sized integer, e.g., 'uint8' or numpy.int16.
        >>> hdr.set_data_dtype(int)
        Traceback (most recent call last):
           ...
        ValueError: Invalid data type <class 'int'>. Specify a sized integer, e.g., 'uint8' or
        numpy.int16.
        >>> hdr.set_data_dtype('int64')
        >>> hdr.get_data_dtype() == np.dtype('int64')
        True
        """
        if not isinstance(datatype, np.dtype) and datatype in (int, 'int'):
            raise ValueError(
                f'Invalid data type {datatype!r}. Specify a sized integer, '
                "e.g., 'uint8' or numpy.int16."
            )
        super().set_data_dtype(datatype)

    def get_qform_quaternion(self):
        """Compute quaternion from b, c, d of quaternion

        Fills a value by assuming this is a unit quaternion
        """
        hdr = self._structarr
        bcd = [hdr['quatern_b'], hdr['quatern_c'], hdr['quatern_d']]
        # Adjust threshold to precision of stored values in header
        return fillpositive(bcd, self.quaternion_threshold)

    def get_qform(self, coded=False):
        """Return 4x4 affine matrix from qform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and qform code.  If False, just
            return affine.  {affine or None} means, return None if qform code
            == 0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine reconstructed from qform
            quaternion.  If `coded` is True, return None if qform code is 0,
            else return the affine.
        code : int
            Qform code. Only returned if `coded` is True.
        """
        hdr = self._structarr
        code = int(hdr['qform_code'])
        if code == 0 and coded:
            return None, 0
        quat = self.get_qform_quaternion()
        R = quat2mat(quat)
        vox = hdr['pixdim'][1:4].copy()
        if np.any(vox < 0):
            raise HeaderDataError('pixdims[1,2,3] should be positive')
        qfac = hdr['pixdim'][0]
        if qfac not in (-1, 1):
            raise HeaderDataError('qfac (pixdim[0]) should be 1 or -1')
        vox[-1] *= qfac
        S = np.diag(vox)
        M = np.dot(R, S)
        out = np.eye(4)
        out[0:3, 0:3] = M
        out[0:3, 3] = [hdr['qoffset_x'], hdr['qoffset_y'], hdr['qoffset_z']]
        if coded:
            return out, code
        return out

    def set_qform(self, affine, code=None, strip_shears=True):
        """Set qform header values from 4x4 affine

        Parameters
        ----------
        affine : None or 4x4 array
            affine transform to write into sform. If None, only set code.
        code : None, string or integer, optional
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:

            * If affine is None, `code`-> 0
            * If affine not None and existing qform code in header == 0,
              `code`-> 2 (aligned)
            * If affine not None and existing qform code in header != 0,
              `code`-> existing qform code in header

        strip_shears : bool, optional
            Whether to strip shears in `affine`.  If True, shears will be
            silently stripped. If False, the presence of shears will raise a
            ``HeaderDataError``

        Notes
        -----
        The qform transform only encodes translations, rotations and
        zooms. If there are shear components to the `affine` transform, and
        `strip_shears` is True (the default), the written qform gives the
        closest approximation where the rotation matrix is orthogonal. This is
        to allow quaternion representation. The orthogonal representation
        enforces orthogonal axes.

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> int(hdr['qform_code'])  # gives 0 - unknown
        0
        >>> affine = np.diag([1,2,3,1])
        >>> np.all(hdr.get_qform() == affine)
        False
        >>> hdr.set_qform(affine)
        >>> np.all(hdr.get_qform() == affine)
        True
        >>> int(hdr['qform_code'])  # gives 2 - aligned
        2
        >>> hdr.set_qform(affine, code='talairach')
        >>> int(hdr['qform_code'])
        3
        >>> hdr.set_qform(affine, code=None)
        >>> int(hdr['qform_code'])
        3
        >>> hdr.set_qform(affine, code='scanner')
        >>> int(hdr['qform_code'])
        1
        >>> hdr.set_qform(None)
        >>> int(hdr['qform_code'])
        0
        """
        hdr = self._structarr
        old_code = hdr['qform_code']
        if code is None:
            if affine is None:
                code = 0
            elif old_code == 0:
                code = 2  # aligned
            else:
                code = old_code
        else:  # code set
            code = self._field_recoders['qform_code'][code]
        hdr['qform_code'] = code
        if affine is None:
            return
        affine = np.asarray(affine)
        if not affine.shape == (4, 4):
            raise TypeError('Need 4x4 affine as input')
        trans = affine[:3, 3]
        RZS = affine[:3, :3]
        zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
        R = RZS / zooms
        # Set qfac to make R determinant positive
        if npl.det(R) > 0:
            qfac = 1
        else:
            qfac = -1
            R[:, -1] *= -1
        # Make R orthogonal (to allow quaternion representation)
        # The orthogonal representation enforces orthogonal axes
        # (a subtle requirement of the NIFTI format qform transform)
        # Transform below is polar decomposition, returning the closest
        # orthogonal matrix PR, to input R
        try:
            P, S, Qs = npl.svd(R)
        except np.linalg.LinAlgError as e:
            raise HeaderDataError(f'Could not decompose affine:\n{affine}') from e
        PR = np.dot(P, Qs)
        if not strip_shears and not np.allclose(PR, R):
            raise HeaderDataError('Shears in affine and `strip_shears` is False')
        # Convert to quaternion
        quat = mat2quat(PR)
        # Set into header
        hdr['qoffset_x'], hdr['qoffset_y'], hdr['qoffset_z'] = trans
        hdr['pixdim'][0] = qfac
        hdr['pixdim'][1:4] = zooms
        hdr['quatern_b'], hdr['quatern_c'], hdr['quatern_d'] = quat[1:]

    def get_sform(self, coded=False):
        """Return 4x4 affine matrix from sform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and sform code.  If False, just
            return affine.  {affine or None} means, return None if sform code
            == 0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine from sform fields. If
            `coded` is True, return None if sform code is 0, else return the
            affine.
        code : int
            Sform code. Only returned if `coded` is True.
        """
        hdr = self._structarr
        code = int(hdr['sform_code'])
        if code == 0 and coded:
            return None, 0
        out = np.eye(4)
        out[0, :] = hdr['srow_x'][:]
        out[1, :] = hdr['srow_y'][:]
        out[2, :] = hdr['srow_z'][:]
        if coded:
            return out, code
        return out

    def set_sform(self, affine, code=None):
        """Set sform transform from 4x4 affine

        Parameters
        ----------
        affine : None or 4x4 array
            affine transform to write into sform.  If None, only set `code`
        code : None, string or integer, optional
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:

            * If affine is None, `code`-> 0
            * If affine not None and existing sform code in header == 0,
              `code`-> 2 (aligned)
            * If affine not None and existing sform code in header != 0,
              `code`-> existing sform code in header

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> int(hdr['sform_code'])  # gives 0 - unknown
        0
        >>> affine = np.diag([1,2,3,1])
        >>> np.all(hdr.get_sform() == affine)
        False
        >>> hdr.set_sform(affine)
        >>> np.all(hdr.get_sform() == affine)
        True
        >>> int(hdr['sform_code'])  # gives 2 - aligned
        2
        >>> hdr.set_sform(affine, code='talairach')
        >>> int(hdr['sform_code'])
        3
        >>> hdr.set_sform(affine, code=None)
        >>> int(hdr['sform_code'])
        3
        >>> hdr.set_sform(affine, code='scanner')
        >>> int(hdr['sform_code'])
        1
        >>> hdr.set_sform(None)
        >>> int(hdr['sform_code'])
        0
        """
        hdr = self._structarr
        old_code = hdr['sform_code']
        if code is None:
            if affine is None:
                code = 0
            elif old_code == 0:
                code = 2  # aligned
            else:
                code = old_code
        else:  # code set
            code = self._field_recoders['sform_code'][code]
        hdr['sform_code'] = code
        if affine is None:
            return
        affine = np.asarray(affine)
        hdr['srow_x'][:] = affine[0, :]
        hdr['srow_y'][:] = affine[1, :]
        hdr['srow_z'][:] = affine[2, :]

    def get_slope_inter(self):
        """Get data scaling (slope) and DC offset (intercept) from header data

        Returns
        -------
        slope : None or float
           scaling (slope).  None if there is no valid scaling from these
           fields
        inter : None or float
           offset (intercept). None if there is no valid scaling or if offset
           is not finite.

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.get_slope_inter()
        (1.0, 0.0)
        >>> hdr['scl_slope'] = 0
        >>> hdr.get_slope_inter()
        (None, None)
        >>> hdr['scl_slope'] = np.nan
        >>> hdr.get_slope_inter()
        (None, None)
        >>> hdr['scl_slope'] = 1
        >>> hdr['scl_inter'] = 1
        >>> hdr.get_slope_inter()
        (1.0, 1.0)
        >>> hdr['scl_inter'] = np.inf
        >>> hdr.get_slope_inter() #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        HeaderDataError: Valid slope but invalid intercept inf
        """
        # Note that we are returning float (float64) scalefactors and
        # intercepts, although they are stored as in nifti1 as float32.
        slope = float(self['scl_slope'])
        inter = float(self['scl_inter'])
        if slope == 0 or not np.isfinite(slope):
            return None, None
        if not np.isfinite(inter):
            raise HeaderDataError(f'Valid slope but invalid intercept {inter}')
        return slope, inter

    def set_slope_inter(self, slope, inter=None):
        """Set slope and / or intercept into header

        Set slope and intercept for image data, such that, if the image
        data is ``arr``, then the scaled image data will be ``(arr *
        slope) + inter``

        (`slope`, `inter`) of (NaN, NaN) is a signal to a containing image to
        set `slope`, `inter` automatically on write.

        Parameters
        ----------
        slope : None or float
           If None, implies `slope`  of NaN. If `slope` is None or NaN then
           `inter` should be None or NaN.  Values of 0, Inf or -Inf raise
           HeaderDataError
        inter : None or float, optional
           Intercept. If None, implies `inter` of NaN. If `slope` is None or
           NaN then `inter` should be None or NaN.  Values of Inf or -Inf raise
           HeaderDataError
        """
        if slope is None:
            slope = np.nan
        if inter is None:
            inter = np.nan
        if slope in (0, np.inf, -np.inf):
            raise HeaderDataError('Slope cannot be 0 or infinite')
        if inter in (np.inf, -np.inf):
            raise HeaderDataError('Intercept cannot be infinite')
        if np.isnan(slope) ^ np.isnan(inter):
            raise HeaderDataError('None or both of slope, inter should be nan')
        self._structarr['scl_slope'] = slope
        self._structarr['scl_inter'] = inter

    def get_dim_info(self):
        """Gets NIfTI MRI slice etc dimension information

        Returns
        -------
        freq : {None,0,1,2}
           Which data array axis is frequency encode direction
        phase : {None,0,1,2}
           Which data array axis is phase encode direction
        slice : {None,0,1,2}
           Which data array axis is slice encode direction

        where ``data array`` is the array returned by ``get_data``

        Because NIfTI1 files are natively Fortran indexed:
          0 is fastest changing in file
          1 is medium changing in file
          2 is slowest changing in file

        ``None`` means the axis appears not to be specified.

        Examples
        --------
        See set_dim_info function

        """
        hdr = self._structarr
        info = int(hdr['dim_info'])
        freq = info & 3
        phase = (info >> 2) & 3
        slice = (info >> 4) & 3
        return (
            freq - 1 if freq else None,
            phase - 1 if phase else None,
            slice - 1 if slice else None,
        )

    def set_dim_info(self, freq=None, phase=None, slice=None):
        """Sets nifti MRI slice etc dimension information

        Parameters
        ----------
        freq : {None, 0, 1, 2}
            axis of data array referring to frequency encoding
        phase : {None, 0, 1, 2}
            axis of data array referring to phase encoding
        slice : {None, 0, 1, 2}
            axis of data array referring to slice encoding

        ``None`` means the axis is not specified.

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(1, 2, 0)
        >>> hdr.get_dim_info()
        (1, 2, 0)
        >>> hdr.set_dim_info(freq=1, phase=2, slice=0)
        >>> hdr.get_dim_info()
        (1, 2, 0)
        >>> hdr.set_dim_info()
        >>> hdr.get_dim_info()
        (None, None, None)
        >>> hdr.set_dim_info(freq=1, phase=None, slice=0)
        >>> hdr.get_dim_info()
        (1, None, 0)

        Notes
        -----
        This is stored in one byte in the header
        """
        for inp in (freq, phase, slice):
            # Don't use == on None to avoid a FutureWarning in python3
            if inp is not None and inp not in (0, 1, 2):
                raise HeaderDataError('Inputs must be in [None, 0, 1, 2]')
        info = 0
        if freq is not None:
            info = info | ((freq + 1) & 3)
        if phase is not None:
            info = info | (((phase + 1) & 3) << 2)
        if slice is not None:
            info = info | (((slice + 1) & 3) << 4)
        self._structarr['dim_info'] = info

    def get_intent(self, code_repr='label'):
        """Get intent code, parameters and name

        Parameters
        ----------
        code_repr : string
           string giving output form of intent code representation.
           Default is 'label'; use 'code' for integer representation.

        Returns
        -------
        code : string or integer
            intent code, or string describing code
        parameters : tuple
            parameters for the intent
        name : string
            intent name

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_intent('t test', (10,), name='some score')
        >>> hdr.get_intent()
        ('t test', (10.0,), 'some score')
        >>> hdr.get_intent('code')
        (3, (10.0,), 'some score')
        """
        hdr = self._structarr
        recoder = self._field_recoders['intent_code']
        code = int(hdr['intent_code'])
        known_intent = code in recoder
        if code_repr == 'code':
            label = code
        elif code_repr == 'label':
            if known_intent:
                label = recoder.label[code]
            else:
                label = 'unknown code ' + str(code)
        else:
            raise TypeError('repr can be "label" or "code"')
        n_params = len(recoder.parameters[code]) if known_intent else 0
        params = (float(hdr[f'intent_p{i}']) for i in range(1, n_params + 1))
        name = hdr['intent_name'].item().decode('latin-1')
        return label, tuple(params), name

    def set_intent(self, code, params=(), name='', allow_unknown=False):
        """Set the intent code, parameters and name

        If parameters are not specified, assumed to be all zero. Each
        intent code has a set number of parameters associated. If you
        specify any parameters, then it will need to be the correct number
        (e.g the "f test" intent requires 2).  However, parameters can
        also be set in the file data, so we also allow not setting any
        parameters (empty parameter tuple).

        Parameters
        ----------
        code : integer or string
            code specifying nifti intent
        params : list, tuple of scalars
            parameters relating to intent (see intent_codes)
            defaults to ().  Unspecified parameters are set to 0.0
        name : string
            intent name (description). Defaults to ''
        allow_unknown : {False, True}, optional
            Allow unknown integer intent codes. If False (the default),
            a KeyError is raised on attempts to set the intent
            to an unknown code.

        Returns
        -------
        None

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_intent(0)  # no intent
        >>> hdr.set_intent('z score')
        >>> hdr.get_intent()
        ('z score', (), '')
        >>> hdr.get_intent('code')
        (5, (), '')
        >>> hdr.set_intent('t test', (10,), name='some score')
        >>> hdr.get_intent()
        ('t test', (10.0,), 'some score')
        >>> hdr.set_intent('f test', (2, 10), name='another score')
        >>> hdr.get_intent()
        ('f test', (2.0, 10.0), 'another score')
        >>> hdr.set_intent('f test')
        >>> hdr.get_intent()
        ('f test', (0.0, 0.0), '')
        >>> hdr.set_intent(9999, allow_unknown=True) # unknown code
        >>> hdr.get_intent()
        ('unknown code 9999', (), '')
        """
        hdr = self._structarr
        known_intent = code in intent_codes
        if not known_intent:
            # We can set intent via an unknown integer code, but can't via an
            # unknown string label
            if not allow_unknown or isinstance(code, str):
                raise KeyError('Unknown intent code: ' + str(code))
        if known_intent:
            icode = intent_codes.code[code]
            p_descr = intent_codes.parameters[code]
        else:
            icode = code
            p_descr = ('p1', 'p2', 'p3')
        if len(params) and len(params) != len(p_descr):
            raise HeaderDataError(f'Need params of form {p_descr}, or empty')
        hdr['intent_code'] = icode
        hdr['intent_name'] = name
        all_params = [0] * 3
        all_params[: len(params)] = params[:]
        for i, param in enumerate(all_params, start=1):
            hdr[f'intent_p{i}'] = param

    def get_slice_duration(self):
        """Get slice duration

        Returns
        -------
        slice_duration : float
            time to acquire one slice

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(slice=2)
        >>> hdr.set_slice_duration(0.3)
        >>> print("%0.1f" % hdr.get_slice_duration())
        0.3

        Notes
        -----
        The NIfTI1 spec appears to require the slice dimension to be
        defined for slice_duration to have meaning.
        """
        _, _, slice_dim = self.get_dim_info()
        if slice_dim is None:
            raise HeaderDataError('Slice dimension must be set for duration to be valid')
        return float(self._structarr['slice_duration'])

    def set_slice_duration(self, duration):
        """Set slice duration

        Parameters
        ----------
        duration : scalar
            time to acquire one slice

        Examples
        --------
        See ``get_slice_duration``
        """
        _, _, slice_dim = self.get_dim_info()
        if slice_dim is None:
            raise HeaderDataError('Slice dimension must be set for duration to be valid')
        self._structarr['slice_duration'] = duration

    def get_n_slices(self):
        """Return the number of slices"""
        _, _, slice_dim = self.get_dim_info()
        if slice_dim is None:
            raise HeaderDataError('Slice dimension not set in header dim_info')
        shape = self.get_data_shape()
        try:
            slice_len = shape[slice_dim]
        except IndexError:
            raise HeaderDataError(
                f'Slice dimension index ({slice_dim}) outside shape tuple ({shape})'
            )
        return slice_len

    def get_slice_times(self):
        """Get slice times from slice timing information

        Returns
        -------
        slice_times : tuple
            Times of acquisition of slices, where 0 is the beginning of
            the acquisition, ordered by position in file.  nifti allows
            slices at the top and bottom of the volume to be excluded from
            the standard slice timing specification, and calls these
            "padding slices".  We give padding slices ``None`` as a time
            of acquisition

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(slice=2)
        >>> hdr.set_data_shape((1, 1, 7))
        >>> hdr.set_slice_duration(0.1)
        >>> hdr['slice_code'] = slice_order_codes['sequential increasing']
        >>> slice_times = hdr.get_slice_times()
        >>> np.allclose(slice_times, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        True
        """
        hdr = self._structarr
        slice_len = self.get_n_slices()
        duration = self.get_slice_duration()
        slabel = self.get_value_label('slice_code')
        if slabel == 'unknown':
            raise HeaderDataError('Cannot get slice times when slice code is "unknown"')
        slice_start, slice_end = (int(hdr['slice_start']), int(hdr['slice_end']))
        if slice_start < 0:
            raise HeaderDataError('slice_start should be >= 0')
        if slice_end == 0:
            slice_end = slice_len - 1
        n_timed = slice_end - slice_start + 1
        if n_timed < 1:
            raise HeaderDataError('slice_end should be > slice_start')
        st_order = self._slice_time_order(slabel, n_timed)
        times = st_order * duration
        return (None,) * slice_start + tuple(times) + (None,) * (slice_len - slice_end - 1)

    def set_slice_times(self, slice_times):
        """Set slice times into *hdr*

        Parameters
        ----------
        slice_times : tuple
            tuple of slice times, one value per slice
            tuple can include None to indicate no slice time for that slice

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(slice=2)
        >>> hdr.set_data_shape([1, 1, 7])
        >>> hdr.set_slice_duration(0.1)
        >>> times = [None, 0.2, 0.4, 0.1, 0.3, 0.0, None]
        >>> hdr.set_slice_times(times)
        >>> hdr.get_value_label('slice_code')
        'alternating decreasing'
        >>> int(hdr['slice_start'])
        1
        >>> int(hdr['slice_end'])
        5
        """
        # Check if number of slices matches header
        hdr = self._structarr
        slice_len = self.get_n_slices()
        if slice_len != len(slice_times):
            raise HeaderDataError('Number of slice times does not match number of slices')
        # Extract Nones at beginning and end.  Check for others
        for ind, time in enumerate(slice_times):
            if time is not None:
                slice_start = ind
                break
        else:
            raise HeaderDataError('Not all slice times can be None')
        for ind, time in enumerate(slice_times[::-1]):
            if time is not None:
                slice_end = slice_len - ind - 1
                break
        timed = slice_times[slice_start : slice_end + 1]
        for time in timed:
            if time is None:
                raise HeaderDataError('Cannot have None in middle of slice time vector')
        # Find slice duration, check times are compatible with single
        # duration
        tdiffs = np.diff(np.sort(timed))
        if not np.allclose(np.diff(tdiffs), 0):
            raise HeaderDataError('Slice times not compatible with single slice duration')
        duration = np.mean(tdiffs)
        # To slice time order
        st_order = np.round(np.array(timed) / duration)
        # Check if slice times fit known schemes
        n_timed = len(timed)
        so_recoder = self._field_recoders['slice_code']
        labels = so_recoder.value_set('label')
        labels.remove('unknown')

        matching_labels = [
            label for label in labels if np.all(st_order == self._slice_time_order(label, n_timed))
        ]

        if not matching_labels:
            raise HeaderDataError(f'slice ordering of {st_order} fits with no known scheme')
        if len(matching_labels) > 1:
            warnings.warn(
                f"Multiple slice orders satisfy: {', '.join(matching_labels)}. "
                'Choosing the first one'
            )
        label = matching_labels[0]
        # Set values into header
        hdr['slice_start'] = slice_start
        hdr['slice_end'] = slice_end
        hdr['slice_duration'] = duration
        hdr['slice_code'] = slice_order_codes.code[label]

    def _slice_time_order(self, slabel, n_slices):
        """Supporting function to give time order of slices from label"""
        if slabel == 'sequential increasing':
            sp_ind_time_order = list(range(n_slices))
        elif slabel == 'sequential decreasing':
            sp_ind_time_order = list(range(n_slices)[::-1])
        elif slabel == 'alternating increasing':
            sp_ind_time_order = list(range(0, n_slices, 2)) + list(range(1, n_slices, 2))
        elif slabel == 'alternating decreasing':
            sp_ind_time_order = list(range(n_slices - 1, -1, -2)) + list(
                range(n_slices - 2, -1, -2)
            )
        elif slabel == 'alternating increasing 2':
            sp_ind_time_order = list(range(1, n_slices, 2)) + list(range(0, n_slices, 2))
        elif slabel == 'alternating decreasing 2':
            sp_ind_time_order = list(range(n_slices - 2, -1, -2)) + list(
                range(n_slices - 1, -1, -2)
            )
        else:
            raise HeaderDataError(f'We do not handle slice ordering "{slabel}"')
        return np.argsort(sp_ind_time_order)

    def get_xyzt_units(self):
        xyz_code = self.structarr['xyzt_units'] % 8
        t_code = self.structarr['xyzt_units'] - xyz_code
        return (unit_codes.label[xyz_code], unit_codes.label[t_code])

    def set_xyzt_units(self, xyz=None, t=None):
        if xyz is None:
            xyz = 0
        if t is None:
            t = 0
        xyz_code = self.structarr['xyzt_units'] % 8
        t_code = self.structarr['xyzt_units'] - xyz_code
        xyz_code = unit_codes[xyz]
        t_code = unit_codes[t]
        self.structarr['xyzt_units'] = xyz_code + t_code

    def _clean_after_mapping(self):
        """Set format-specific stuff after converting header from mapping

        Clean up header after it has been initialized from an
        ``as_analyze_map`` method of another header type

        See :meth:`nibabel.analyze.AnalyzeHeader._clean_after_mapping` for a
        more detailed description.
        """
        self._structarr['magic'] = self.single_magic if self.is_single else self.pair_magic

    """ Checks only below here """

    @classmethod
    def _get_checks(klass):
        # We need to return our own versions of - e.g. chk_datatype, to
        # pick up the Nifti datatypes from our class
        return (
            klass._chk_sizeof_hdr,
            klass._chk_datatype,
            klass._chk_bitpix,
            klass._chk_pixdims,
            klass._chk_qfac,
            klass._chk_magic,
            klass._chk_offset,
            klass._chk_qform_code,
            klass._chk_sform_code,
        )

    @staticmethod
    def _chk_qfac(hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['pixdim'][0] in (-1, 1):
            return hdr, rep
        rep.problem_level = 20
        rep.problem_msg = 'pixdim[0] (qfac) should be 1 (default) or -1'
        if fix:
            hdr['pixdim'][0] = 1
            rep.fix_msg = 'setting qfac to 1'
        return hdr, rep

    @staticmethod
    def _chk_magic(hdr, fix=False):
        rep = Report(HeaderDataError)
        magic = hdr['magic'].item()
        if magic in (hdr.pair_magic, hdr.single_magic):
            return hdr, rep
        rep.problem_msg = f'magic string {magic.decode("latin1")!r} is not valid'
        rep.problem_level = 45
        if fix:
            rep.fix_msg = 'leaving as is, but future errors are likely'
        return hdr, rep

    @staticmethod
    def _chk_offset(hdr, fix=False):
        rep = Report(HeaderDataError)
        # for ease of later string formatting, use scalar of byte string
        magic = hdr['magic'].item()
        offset = hdr['vox_offset'].item()
        if offset == 0:
            return hdr, rep
        if magic == hdr.single_magic and offset < hdr.single_vox_offset:
            rep.problem_level = 40
            rep.problem_msg = f'vox offset {int(offset)} too low for single file nifti1'
            if fix:
                hdr['vox_offset'] = hdr.single_vox_offset
                rep.fix_msg = f'setting to minimum value of {hdr.single_vox_offset}'
            return hdr, rep
        if not offset % 16:
            return hdr, rep
        # SPM uses memory mapping to read the data, and
        # apparently this has to start on 16 byte boundaries
        rep.problem_msg = f'vox offset (={offset:g}) not divisible by 16, not SPM compatible'
        rep.problem_level = 30
        if fix:
            rep.fix_msg = 'leaving at current value'
        return hdr, rep

    @classmethod
    def _chk_qform_code(klass, hdr, fix=False):
        return klass._chk_xform_code('qform_code', hdr, fix)

    @classmethod
    def _chk_sform_code(klass, hdr, fix=False):
        return klass._chk_xform_code('sform_code', hdr, fix)

    @classmethod
    def _chk_xform_code(klass, code_type, hdr, fix):
        # utility method for sform and qform codes
        rep = Report(HeaderDataError)
        code = int(hdr[code_type])
        recoder = klass._field_recoders[code_type]
        if code in recoder.value_set():
            return hdr, rep
        rep.problem_level = 30
        rep.problem_msg = f'{code_type} {code} not valid'
        if fix:
            hdr[code_type] = 0
            rep.fix_msg = 'setting to 0'
        return hdr, rep

    @classmethod
    def may_contain_header(klass, binaryblock):
        if len(binaryblock) < klass.sizeof_hdr:
            return False

        hdr_struct = np.ndarray(
            shape=(), dtype=header_dtype, buffer=binaryblock[: klass.sizeof_hdr]
        )
        return hdr_struct['magic'] in (b'ni1', b'n+1')


class Nifti1PairHeader(Nifti1Header):
    """Class for NIfTI1 pair header"""

    # Signal whether this is single (header + data) file
    is_single = False


class Nifti1Pair(analyze.AnalyzeImage):
    """Class for NIfTI1 format image, header pair"""

    header_class: type[Nifti1Header] = Nifti1PairHeader
    header: Nifti1Header
    _meta_sniff_len = header_class.sizeof_hdr
    rw = True

    # If a _dtype_alias has been set, it can only be resolved by inspecting
    # the data at serialization time
    _dtype_alias = None

    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None, dtype=None):
        # Special carve-out for 64 bit integers
        # See GitHub issues
        #  * https://github.com/nipy/nibabel/issues/1046
        #  * https://github.com/nipy/nibabel/issues/1089
        # This only applies to NIfTI because the parent Analyze formats did
        # not support 64-bit integer data, so `set_data_dtype(int64)` would
        # already fail.
        danger_dts = (np.dtype('int64'), np.dtype('uint64'))
        if header is None and dtype is None and get_obj_dtype(dataobj) in danger_dts:
            alert_future_error(
                f'Image data has type {dataobj.dtype}, which may cause '
                'incompatibilities with other tools.',
                '5.0',
                warning_rec='This warning can be silenced by passing the dtype argument'
                f' to {self.__class__.__name__}().',
                error_rec='To use this type, pass an explicit header or dtype argument'
                f' to {self.__class__.__name__}().',
                error_class=ValueError,
            )
        super().__init__(dataobj, affine, header, extra, file_map, dtype)
        # Force set of s/q form when header is None unless affine is also None
        if header is None and affine is not None:
            self._affine2header()

    # Copy docstring
    __init__.__doc__ = f"""{analyze.AnalyzeImage.__init__.__doc__}
        Notes
        -----

        If both a `header` and an `affine` are specified, and the `affine` does
        not match the affine that is in the `header`, the `affine` will be used,
        but the ``sform_code`` and ``qform_code`` fields in the header will be
        re-initialised to their default values. This is performed on the basis
        that, if you are changing the affine, you are likely to be changing the
        space to which the affine is pointing.  The :meth:`set_sform` and
        :meth:`set_qform` methods can be used to update the codes after an image
        has been created - see those methods, and the :ref:`manual
        <default-sform-qform-codes>` for more details.  """

    def update_header(self):
        """Harmonize header with image data and affine

        See AnalyzeImage.update_header for more examples

        Examples
        --------
        >>> data = np.zeros((2,3,4))
        >>> affine = np.diag([1.0,2.0,3.0,1.0])
        >>> img = Nifti1Image(data, affine)
        >>> hdr = img.header
        >>> np.all(hdr.get_qform() == affine)
        True
        >>> np.all(hdr.get_sform() == affine)
        True
        """
        super().update_header()
        hdr = self._header
        hdr['magic'] = hdr.pair_magic

    def _affine2header(self):
        """Unconditionally set affine into the header"""
        hdr = self._header
        # Set affine into sform with default code
        hdr.set_sform(self._affine, code='aligned')
        # Make qform 'unknown'
        hdr.set_qform(self._affine, code='unknown')

    def get_qform(self, coded=False):
        """Return 4x4 affine matrix from qform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and qform code.  If False, just
            return affine.  {affine or None} means, return None if qform code
            == 0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine reconstructed from qform
            quaternion.  If `coded` is True, return None if qform code is 0,
            else return the affine.
        code : int
            Qform code. Only returned if `coded` is True.

        See also
        --------
        set_qform
        get_sform
        """
        return self._header.get_qform(coded)

    def set_qform(self, affine, code=None, strip_shears=True, **kwargs):
        """Set qform header values from 4x4 affine

        Parameters
        ----------
        affine : None or 4x4 array
            affine transform to write into sform. If None, only set code.
        code : None, string or integer
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:

            * If affine is None, `code`-> 0
            * If affine not None and existing qform code in header == 0,
              `code`-> 2 (aligned)
            * If affine not None and existing qform code in header != 0,
              `code`-> existing qform code in header

        strip_shears : bool, optional
            Whether to strip shears in `affine`.  If True, shears will be
            silently stripped. If False, the presence of shears will raise a
            ``HeaderDataError``
        update_affine : bool, optional
            Whether to update the image affine from the header best affine
            after setting the qform. Must be keyword argument (because of
            different position in `set_qform`). Default is True

        See also
        --------
        get_qform
        set_sform

        Examples
        --------
        >>> data = np.arange(24, dtype='f4').reshape((2,3,4))
        >>> aff = np.diag([2, 3, 4, 1])
        >>> img = Nifti1Pair(data, aff)
        >>> img.get_qform()
        array([[2., 0., 0., 0.],
               [0., 3., 0., 0.],
               [0., 0., 4., 0.],
               [0., 0., 0., 1.]])
        >>> img.get_qform(coded=True)
        (None, 0)
        >>> aff2 = np.diag([3, 4, 5, 1])
        >>> img.set_qform(aff2, 'talairach')
        >>> qaff, code = img.get_qform(coded=True)
        >>> np.all(qaff == aff2)
        True
        >>> int(code)
        3
        """
        update_affine = kwargs.pop('update_affine', True)
        if kwargs:
            raise TypeError(f'Unexpected keyword argument(s) {kwargs}')
        self._header.set_qform(affine, code, strip_shears)
        if update_affine:
            if self._affine is None:
                self._affine = self._header.get_best_affine()
            else:
                self._affine[:] = self._header.get_best_affine()

    def get_sform(self, coded=False):
        """Return 4x4 affine matrix from sform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and sform code.  If False, just
            return affine.  {affine or None} means, return None if sform code
            == 0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine from sform fields. If
            `coded` is True, return None if sform code is 0, else return the
            affine.
        code : int
            Sform code. Only returned if `coded` is True.

        See also
        --------
        set_sform
        get_qform
        """
        return self._header.get_sform(coded)

    def set_sform(self, affine, code=None, **kwargs):
        """Set sform transform from 4x4 affine

        Parameters
        ----------
        affine : None or 4x4 array
            affine transform to write into sform.  If None, only set `code`
        code : None, string or integer
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:

            * If affine is None, `code`-> 0
            * If affine not None and existing sform code in header == 0,
              `code`-> 2 (aligned)
            * If affine not None and existing sform code in header != 0,
              `code`-> existing sform code in header

        update_affine : bool, optional
            Whether to update the image affine from the header best affine
            after setting the qform.  Must be keyword argument (because of
            different position in `set_qform`). Default is True

        See also
        --------
        get_sform
        set_qform

        Examples
        --------
        >>> data = np.arange(24, dtype='f4').reshape((2,3,4))
        >>> aff = np.diag([2, 3, 4, 1])
        >>> img = Nifti1Pair(data, aff)
        >>> img.get_sform()
        array([[2., 0., 0., 0.],
               [0., 3., 0., 0.],
               [0., 0., 4., 0.],
               [0., 0., 0., 1.]])
        >>> saff, code = img.get_sform(coded=True)
        >>> saff
        array([[2., 0., 0., 0.],
               [0., 3., 0., 0.],
               [0., 0., 4., 0.],
               [0., 0., 0., 1.]])
        >>> int(code)
        2
        >>> aff2 = np.diag([3, 4, 5, 1])
        >>> img.set_sform(aff2, 'talairach')
        >>> saff, code = img.get_sform(coded=True)
        >>> np.all(saff == aff2)
        True
        >>> int(code)
        3
        """
        update_affine = kwargs.pop('update_affine', True)
        if kwargs:
            raise TypeError(f'Unexpected keyword argument(s) {kwargs}')
        self._header.set_sform(affine, code)
        if update_affine:
            if self._affine is None:
                self._affine = self._header.get_best_affine()
            else:
                self._affine[:] = self._header.get_best_affine()

    def set_data_dtype(self, datatype):
        """Set numpy dtype for data from code, dtype, type or alias

        Using :py:class:`int` or ``"int"`` is disallowed, as these types
        will be interpreted as ``np.int64``, which is almost never desired.
        ``np.int64`` is permitted for those intent on making poor choices.

        The following aliases are defined to allow for flexible specification:

          * ``'mask'`` - Alias for ``uint8``
          * ``'compat'`` - The nearest Analyze-compatible datatype
            (``uint8``, ``int16``, ``int32``, ``float32``)
          * ``'smallest'`` - The smallest Analyze-compatible integer
            (``uint8``, ``int16``, ``int32``)

        Dynamic aliases are resolved when ``get_data_dtype()`` is called
        with a ``finalize=True`` flag. Until then, these aliases are not
        written to the header and will not persist to new images.

        Examples
        --------
        >>> ints = np.arange(24, dtype='i4').reshape((2,3,4))

        >>> img = Nifti1Image(ints, np.eye(4))
        >>> img.set_data_dtype(np.uint8)
        >>> img.get_data_dtype()
        dtype('uint8')
        >>> img.set_data_dtype('mask')
        >>> img.get_data_dtype()
        dtype('uint8')
        >>> img.set_data_dtype('compat')
        >>> img.get_data_dtype()
        'compat'
        >>> img.get_data_dtype(finalize=True)
        dtype('<i4')
        >>> img.get_data_dtype()
        dtype('<i4')
        >>> img.set_data_dtype('smallest')
        >>> img.get_data_dtype()
        'smallest'
        >>> img.get_data_dtype(finalize=True)
        dtype('uint8')
        >>> img.get_data_dtype()
        dtype('uint8')

        Note that floating point values will not be coerced to ``int``

        >>> floats = np.arange(24, dtype='f4').reshape((2,3,4))
        >>> img = Nifti1Image(floats, np.eye(4))
        >>> img.set_data_dtype('smallest')
        >>> img.get_data_dtype(finalize=True)
        Traceback (most recent call last):
           ...
        ValueError: Cannot automatically cast array (of type float32) to an integer
        type with fewer than 64 bits. Please set_data_dtype() to an explicit data type.

        >>> arr = np.arange(1000, 1024, dtype='i4').reshape((2,3,4))
        >>> img = Nifti1Image(arr, np.eye(4))
        >>> img.set_data_dtype('smallest')
        >>> img.set_data_dtype('implausible')
        Traceback (most recent call last):
           ...
        nibabel.spatialimages.HeaderDataError: data dtype "implausible" not recognized
        >>> img.set_data_dtype('none')
        Traceback (most recent call last):
           ...
        nibabel.spatialimages.HeaderDataError: data dtype "none" known but not supported
        >>> img.set_data_dtype(np.void)
        Traceback (most recent call last):
           ...
        nibabel.spatialimages.HeaderDataError: data dtype "<class 'numpy.void'>" known
        but not supported
        >>> img.set_data_dtype('int')
        Traceback (most recent call last):
           ...
        ValueError: Invalid data type 'int'. Specify a sized integer, e.g., 'uint8' or numpy.int16.
        >>> img.set_data_dtype(int)
        Traceback (most recent call last):
           ...
        ValueError: Invalid data type <class 'int'>. Specify a sized integer, e.g., 'uint8' or
        numpy.int16.
        >>> img.set_data_dtype('int64')
        >>> img.get_data_dtype() == np.dtype('int64')
        True
        """
        # Comparing dtypes to strings, numpy will attempt to call, e.g., dtype('mask'),
        # so only check for aliases if the type is a string
        # See https://github.com/numpy/numpy/issues/7242
        if isinstance(datatype, str):
            # Static aliases
            if datatype == 'mask':
                datatype = 'u1'
            # Dynamic aliases
            elif datatype in ('compat', 'smallest'):
                self._dtype_alias = datatype
                return

        self._dtype_alias = None
        super().set_data_dtype(datatype)

    def get_data_dtype(self, finalize=False):
        """Get numpy dtype for data

        If ``set_data_dtype()`` has been called with an alias
        and ``finalize`` is ``False``, return the alias.
        If ``finalize`` is ``True``, determine the appropriate dtype
        from the image data object and set the final dtype in the
        header before returning it.
        """
        if self._dtype_alias is None:
            return super().get_data_dtype()
        if not finalize:
            return self._dtype_alias

        datatype = None
        if self._dtype_alias == 'compat':
            datatype = _get_analyze_compat_dtype(self._dataobj)
            descrip = 'an Analyze-compatible dtype'
        elif self._dtype_alias == 'smallest':
            datatype = _get_smallest_dtype(self._dataobj)
            descrip = 'an integer type with fewer than 64 bits'
        else:
            raise ValueError(f'Unknown dtype alias {self._dtype_alias}.')
        if datatype is None:
            dt = get_obj_dtype(self._dataobj)
            raise ValueError(
                f'Cannot automatically cast array (of type {dt}) to {descrip}.'
                ' Please set_data_dtype() to an explicit data type.'
            )

        self.set_data_dtype(datatype)  # Clears the alias
        return super().get_data_dtype()

    def to_file_map(self, file_map=None, dtype=None):
        """Write image to `file_map` or contained ``self.file_map``

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        dtype : dtype-like, optional
           The on-disk data type to coerce the data array.
        """
        img_dtype = self.get_data_dtype()
        self.get_data_dtype(finalize=True)
        try:
            super().to_file_map(file_map, dtype)
        finally:
            self.set_data_dtype(img_dtype)

    def as_reoriented(self, ornt):
        """Apply an orientation change and return a new image

        If ornt is identity transform, return the original image, unchanged

        Parameters
        ----------
        ornt : (n,2) orientation array
           orientation transform. ``ornt[N,1]` is flip of axis N of the
           array implied by `shape`, where 1 means no flip and -1 means
           flip.  For example, if ``N==0`` and ``ornt[0,1] == -1``, and
           there's an array ``arr`` of shape `shape`, the flip would
           correspond to the effect of ``np.flipud(arr)``.  ``ornt[:,0]`` is
           the transpose that needs to be done to the implied array, as in
           ``arr.transpose(ornt[:,0])``
        """
        img = super().as_reoriented(ornt)

        if img is self:
            return img

        # Also apply the transform to the dim_info fields
        new_dim = [
            None if orig_dim is None else int(ornt[orig_dim, 0])
            for orig_dim in img.header.get_dim_info()
        ]

        img.header.set_dim_info(*new_dim)

        return img


class Nifti1Image(Nifti1Pair, SerializableImage):
    """Class for single file NIfTI1 format image"""

    header_class = Nifti1Header
    valid_exts = ('.nii',)
    files_types = (('image', '.nii'),)

    @staticmethod
    def _get_fileholders(file_map):
        """Return fileholder for header and image

        For single-file niftis, the fileholder for the header and the image
        will be the same
        """
        return file_map['image'], file_map['image']

    def update_header(self):
        """Harmonize header with image data and affine"""
        super().update_header()
        hdr = self._header
        hdr['magic'] = hdr.single_magic


def load(filename):
    """Load NIfTI1 single or pair from `filename`

    Parameters
    ----------
    filename : str
        filename of image to be loaded

    Returns
    -------
    img : Nifti1Image or Nifti1Pair
        NIfTI1 single or pair image instance

    Raises
    ------
    ImageFileError
        if `filename` doesn't look like NIfTI1;
    OSError
        if `filename` does not exist.
    """
    try:
        img = Nifti1Image.load(filename)
    except ImageFileError:
        return Nifti1Pair.load(filename)
    return img


def save(img, filename):
    """Save NIfTI1 single or pair to `filename`

    Parameters
    ----------
    filename : str
        filename to which to save image
    """
    try:
        Nifti1Image.instance_to_filename(img, filename)
    except ImageFileError:
        Nifti1Pair.instance_to_filename(img, filename)


def _get_smallest_dtype(
    arr,
    itypes=(np.uint8, np.int16, np.int32),
    ftypes=(),
):
    """Return the smallest "sensible" dtype that will hold the array data

    The purpose of this function is to support automatic type selection
    for serialization, so "sensible" here means well-supported in the NIfTI-1 world.

    For floating point data, select between single- and double-precision.
    For integer data, select among uint8, int16 and int32.

    The test is for min/max range, so float64 is pretty unlikely to be hit.

    Returns ``None`` if these dtypes do not suffice.

    >>> _get_smallest_dtype(np.array([0, 1]))
    dtype('uint8')
    >>> _get_smallest_dtype(np.array([-1, 1]))
    dtype('int16')
    >>> _get_smallest_dtype(np.array([0, 256]))
    dtype('int16')
    >>> _get_smallest_dtype(np.array([-65536, 65536]))
    dtype('int32')
    >>> _get_smallest_dtype(np.array([-2147483648, 2147483648]))

    By default floating point types are not searched:

    >>> _get_smallest_dtype(np.array([1.]))
    >>> _get_smallest_dtype(np.array([2. ** 1000]))
    >>> _get_smallest_dtype(np.longdouble(2) ** 2000)
    >>> _get_smallest_dtype(np.array([1+0j]))

    However, this function can be passed "legal" floating point types, and
    the logic works the same.

    >>> _get_smallest_dtype(np.array([1.]), ftypes=('float32',))
    dtype('float32')
    >>> _get_smallest_dtype(np.array([2. ** 1000]), ftypes=('float32',))
    >>> _get_smallest_dtype(np.longdouble(2) ** 2000, ftypes=('float32',))
    >>> _get_smallest_dtype(np.array([1+0j]), ftypes=('float32',))
    """
    arr = np.asanyarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        test_dts = ftypes
        info = np.finfo
    elif np.issubdtype(arr.dtype, np.integer):
        test_dts = itypes
        info = np.iinfo
    else:
        return None

    mn, mx = np.min(arr), np.max(arr)
    for dt in test_dts:
        dtinfo = info(dt)
        if dtinfo.min <= mn and mx <= dtinfo.max:
            return np.dtype(dt)


def _get_analyze_compat_dtype(arr):
    """Return an Analyze-compatible dtype that ``arr`` can be safely cast to

    Analyze-compatible types are returned without inspection:

    >>> _get_analyze_compat_dtype(np.uint8([0, 1]))
    dtype('uint8')
    >>> _get_analyze_compat_dtype(np.int16([0, 1]))
    dtype('int16')
    >>> _get_analyze_compat_dtype(np.int32([0, 1]))
    dtype('int32')
    >>> _get_analyze_compat_dtype(np.float32([0, 1]))
    dtype('float32')

    Signed ``int8`` are cast to ``uint8`` or ``int16`` based on value ranges:

    >>> _get_analyze_compat_dtype(np.int8([0, 1]))
    dtype('uint8')
    >>> _get_analyze_compat_dtype(np.int8([-1, 1]))
    dtype('int16')

    Unsigned ``uint16`` are cast to ``int16`` or ``int32`` based on value ranges:

    >>> _get_analyze_compat_dtype(np.uint16([32767]))
    dtype('int16')
    >>> _get_analyze_compat_dtype(np.uint16([65535]))
    dtype('int32')

    ``int32`` is returned for integer types and ``float32`` for floating point types:

    >>> _get_analyze_compat_dtype(np.array([-1, 1]))
    dtype('int32')
    >>> _get_analyze_compat_dtype(np.array([-1., 1.]))
    dtype('float32')

    If the value ranges exceed 4 bytes or cannot be cast, then a ``ValueError`` is raised:

    >>> _get_analyze_compat_dtype(np.array([0, 4294967295]))
    Traceback (most recent call last):
       ...
    ValueError: Cannot find analyze-compatible dtype for array with dtype=int64
        (min=0, max=4294967295)

    >>> _get_analyze_compat_dtype([0., 2.e40])
    Traceback (most recent call last):
       ...
    ValueError: Cannot find analyze-compatible dtype for array with dtype=float64
        (min=0.0, max=2e+40)

    Note that real-valued complex arrays cannot be safely cast.

    >>> _get_analyze_compat_dtype(np.array([1+0j]))
    Traceback (most recent call last):
       ...
    ValueError: Cannot find analyze-compatible dtype for array with dtype=complex128
        (min=(1+0j), max=(1+0j))
    """
    arr = np.asanyarray(arr)
    dtype = arr.dtype
    if dtype in (np.uint8, np.int16, np.int32, np.float32):
        return dtype

    if dtype == np.int8:
        return np.dtype('uint8' if arr.min() >= 0 else 'int16')
    elif dtype == np.uint16:
        return np.dtype('int16' if arr.max() <= np.iinfo(np.int16).max else 'int32')

    mn, mx = arr.min(), arr.max()
    if arr.dtype.kind in 'iu':
        info = np.iinfo('int32')
        if mn >= info.min and mx <= info.max:
            return np.dtype('int32')
    elif arr.dtype.kind == 'f':
        info = np.finfo('float32')
        if mn >= info.min and mx <= info.max:
            return np.dtype('float32')

    raise ValueError(
        f'Cannot find analyze-compatible dtype for array with dtype={dtype} (min={mn}, max={mx})'
    )
