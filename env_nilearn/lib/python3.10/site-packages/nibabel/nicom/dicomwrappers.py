"""Classes to wrap DICOM objects and files

The wrappers encapsulate the capabilities of the different DICOM
formats.

They also allow dictionary-like access to named fields.

For calculated attributes, we return None where needed data is missing.
It seemed strange to raise an error during attribute processing, other
than an AttributeError - breaking the 'properties manifesto'.   So, any
processing that needs to raise an error, should be in a method, rather
than in a property, or property-like thing.
"""

import operator
import re
import warnings
from functools import cached_property

import numpy as np

from nibabel.optpkg import optional_package

from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
from .utils import Vendor, find_private_section, vendor_from_private

pydicom = optional_package('pydicom')[0]


class WrapperError(Exception):
    pass


class WrapperPrecisionError(WrapperError):
    pass


def wrapper_from_file(file_like, *args, **kwargs):
    r"""Create DICOM wrapper from `file_like` object

    Parameters
    ----------
    file_like : object
       filename string or file-like object, pointing to a valid DICOM
       file readable by ``pydicom``
    \*args : positional
        args to ``dicom.dcmread`` command.
    \*\*kwargs : keyword
        args to ``dicom.dcmread`` command.  ``force=True`` might be a
        likely keyword argument.

    Returns
    -------
    dcm_w : ``dicomwrappers.Wrapper`` or subclass
       DICOM wrapper corresponding to DICOM data type
    """
    with ImageOpener(file_like) as fobj:
        dcm_data = pydicom.dcmread(fobj, *args, **kwargs)
    return wrapper_from_data(dcm_data)


def wrapper_from_data(dcm_data, frame_filters=None):
    """Create DICOM wrapper from DICOM data object

    Parameters
    ----------
    dcm_data : ``dicom.dataset.Dataset`` instance or similar
       Object allowing attribute access, with DICOM attributes.
       Probably a dataset as read by ``pydicom``.

    frame_filters
        Optionally override the `frame_filters` used to create a `MultiFrameWrapper`

    Returns
    -------
    dcm_w : ``dicomwrappers.Wrapper`` or subclass
       DICOM wrapper corresponding to DICOM data type
    """
    sop_class = dcm_data.get('SOPClassUID')
    # try to detect what type of dicom object to wrap
    if sop_class == '1.2.840.10008.5.1.4.1.1.4.1':  # Enhanced MR Image Storage
        return MultiframeWrapper(dcm_data, frame_filters)
    # Check for non-enhanced (legacy) Siemens DICOM format types
    # Only Siemens will have data for the CSA header
    try:
        csa = csar.get_csa_header(dcm_data)
    except csar.CSAReadError as e:
        warnings.warn(
            f'Error while attempting to read CSA header: {e.args}\n'
            'Ignoring Siemens private (CSA) header info.'
        )
        csa = None
    if csa is None:
        return Wrapper(dcm_data)
    if csar.is_mosaic(csa):
        # Mosaic is a "tiled" image
        return MosaicWrapper(dcm_data, csa)
    # Assume data is in a single slice format per file
    return SiemensWrapper(dcm_data, csa)


class Wrapper:
    """Class to wrap general DICOM files

    Methods:

    * get_data()
    * get_unscaled_data()
    * get_pixel_array()
    * is_same_series(other)
    * __getitem__ : return attributes from `dcm_data`
    * get(key[, default]) - as usual given __getitem__ above

    Attributes and things that look like attributes:

    * affine : (4, 4) array
    * dcm_data : object
    * image_shape : tuple
    * image_orient_patient : (3,2) array
    * slice_normal : (3,) array
    * rotation_matrix : (3,3) array
    * voxel_sizes : tuple length 3
    * image_position : sequence length 3
    * slice_indicator : float
    * series_signature : tuple
    * scale_factors : (N, 2) array
    * vendor : Vendor
    """

    is_csa = False
    is_mosaic = False
    is_multiframe = False
    b_matrix = None
    q_vector = None

    def __init__(self, dcm_data):
        """Initialize wrapper

        Parameters
        ----------
        dcm_data : object
           object should allow 'get' and '__getitem__' access.  Usually this
           will be a ``dicom.dataset.Dataset`` object resulting from reading a
           DICOM file.
        """
        self.dcm_data = dcm_data

    @cached_property
    def vendor(self):
        """The vendor of the instrument that produced the DICOM"""
        # Look at manufacturer tag first
        mfgr = self.get('Manufacturer')
        if mfgr:
            if re.search('Siemens', mfgr, re.IGNORECASE):
                return Vendor.SIEMENS
            if re.search('Philips', mfgr, re.IGNORECASE):
                return Vendor.PHILIPS
            if re.search('GE Medical', mfgr, re.IGNORECASE):
                return Vendor.GE
        # Next look at UID prefixes
        for uid_src in ('StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'):
            uid = str(self.get(uid_src))
            if uid.startswith(('1.3.12.2.1007.', '1.3.12.2.1107.')):
                return Vendor.SIEMENS
            if uid.startswith(('1.3.46', '1.3.12.2.1017')):
                return Vendor.PHILIPS
            if uid.startswith('1.2.840.113619'):
                return Vendor.GE
        # Finally look for vendor specific private blocks
        return vendor_from_private(self.dcm_data)

    @cached_property
    def image_shape(self):
        """The array shape as it will be returned by ``get_data()``"""
        shape = (self.get('Rows'), self.get('Columns'))
        if None in shape:
            return None
        return shape

    @cached_property
    def image_orient_patient(self):
        """Note that this is _not_ LR flipped"""
        iop = self.get('ImageOrientationPatient')
        if iop is None:
            return None
        # Values are python Decimals in pydicom 0.9.7
        iop = np.array(list(map(float, iop)))
        return np.array(iop).reshape(2, 3).T

    @cached_property
    def slice_normal(self):
        iop = self.image_orient_patient
        if iop is None:
            return None
        # iop[:, 0] is column index cosine, iop[:, 1] is row index cosine
        return np.cross(iop[:, 1], iop[:, 0])

    @cached_property
    def rotation_matrix(self):
        """Return rotation matrix between array indices and mm

        Note that we swap the two columns of the 'ImageOrientPatient'
        when we create the rotation matrix.  This is takes into account
        the slightly odd ij transpose construction of the DICOM
        orientation fields - see doc/theory/dicom_orientaiton.rst.
        """
        iop = self.image_orient_patient
        s_norm = self.slice_normal
        if iop is None or s_norm is None:
            return None
        R = np.eye(3)
        # np.fliplr(iop) gives matrix F in
        # doc/theory/dicom_orientation.rst The fliplr accounts for the
        # fact that the first column in ``iop`` refers to changes in
        # column index, and the second to changes in row index.
        R[:, :2] = np.fliplr(iop)
        R[:, 2] = s_norm
        # check this is in fact a rotation matrix. Error comes from compromise
        # motivated in ``doc/source/notebooks/ata_error.ipynb``, and from
        # discussion at https://github.com/nipy/nibabel/pull/156
        if not np.allclose(np.eye(3), np.dot(R, R.T), atol=5e-5):
            raise WrapperPrecisionError('Rotation matrix not nearly orthogonal')
        return R

    @cached_property
    def voxel_sizes(self):
        """voxel sizes for array as returned by ``get_data()``"""
        # pix space gives (row_spacing, column_spacing).  That is, the
        # mm you move when moving from one row to the next, and the mm
        # you move when moving from one column to the next
        pix_space = self.get('PixelSpacing')
        if pix_space is None:
            return None
        zs = self.get('SpacingBetweenSlices')
        if zs is None:
            zs = self.get('SliceThickness')
            if zs is None or zs == '':
                zs = 1
        # Protect from python decimals in pydicom 0.9.7
        zs = float(zs)
        pix_space = list(map(float, pix_space))
        return tuple(pix_space + [zs])

    @cached_property
    def image_position(self):
        """Return position of first voxel in data block

        Parameters
        ----------
        None

        Returns
        -------
        img_pos : (3,) array
           position in mm of voxel (0,0) in image array
        """
        ipp = self.get('ImagePositionPatient')
        if ipp is None:
            return None
            # Values are python Decimals in pydicom 0.9.7
        return np.array(list(map(float, ipp)))

    @cached_property
    def slice_indicator(self):
        """A number that is higher for higher slices in Z

        Comparing this number between two adjacent slices should give a
        difference equal to the voxel size in Z.

        See doc/theory/dicom_orientation for description
        """
        ipp = self.image_position
        s_norm = self.slice_normal
        if ipp is None or s_norm is None:
            return None
        return np.inner(ipp, s_norm)

    @cached_property
    def instance_number(self):
        """Just because we use this a lot for sorting"""
        return self.get('InstanceNumber')

    @cached_property
    def series_signature(self):
        """Signature for matching slices into series

        We use `signature` in ``self.is_same_series(other)``.

        Returns
        -------
        signature : dict
           with values of 2-element sequences, where first element is
           value, and second element is function to compare this value
           with another.  This allows us to pass things like arrays,
           that might need to be ``allclose`` instead of equal
        """
        # dictionary with value, comparison func tuple
        signature = {}
        eq = operator.eq
        for key in (
            'SeriesInstanceUID',
            'SeriesNumber',
            'ImageType',
            'SequenceName',
            'EchoNumbers',
        ):
            signature[key] = (self.get(key), eq)
        signature['image_shape'] = (self.image_shape, eq)
        signature['iop'] = (self.image_orient_patient, none_or_close)
        signature['vox'] = (self.voxel_sizes, none_or_close)
        return signature

    def __getitem__(self, key):
        """Return values from DICOM object"""
        if key not in self.dcm_data:
            raise KeyError(f'"{key}" not in self.dcm_data')
        return self.dcm_data.get(key)

    def get(self, key, default=None):
        """Get values from underlying dicom data"""
        return self.dcm_data.get(key, default)

    @property
    def affine(self):
        """Mapping between voxel and DICOM coordinate system

        (4, 4) affine matrix giving transformation between voxels in data array
        and mm in the DICOM patient coordinate system.
        """
        # rotation matrix already accounts for the ij transpose in the
        # DICOM image orientation patient transform.  So. column 0 is
        # direction cosine for changes in row index, column 1 is
        # direction cosine for changes in column index
        orient = self.rotation_matrix
        # therefore, these voxel sizes are in the right order (row,
        # column, slice)
        vox = self.voxel_sizes
        ipp = self.image_position
        if any(p is None for p in (orient, vox, ipp)):
            raise WrapperError('Not enough information for affine')
        aff = np.eye(4)
        aff[:3, :3] = orient * np.array(vox)
        aff[:3, 3] = ipp
        return aff

    def get_pixel_array(self):
        """Return raw pixel array without reshaping or scaling

        Returns
        -------
        data : array
           array with raw pixel data from DICOM
        """
        data = self.dcm_data.get('pixel_array')
        if data is None:
            raise WrapperError('Cannot find data in DICOM')
        return data

    def get_unscaled_data(self):
        """Return pixel array that is potentially reshaped, but without any scaling

        Returns
        -------
        data : array
           array with raw pixel data from DICOM
        """
        return self.get_pixel_array()

    def get_data(self):
        """Get potentially scaled and reshaped image data from DICOMs

        We return the data as DICOM understands it, first dimension is
        rows, second dimension is columns

        Returns
        -------
        data : array
           array with data as scaled from any scaling in the DICOM
           fields.
        """
        return self._scale_data(self.get_unscaled_data())

    def is_same_series(self, other):
        """Return True if `other` appears to be in same series

        Parameters
        ----------
        other : object
           object with ``series_signature`` attribute that is a
           mapping.  Usually it's a ``Wrapper`` or sub-class instance.

        Returns
        -------
        tf : bool
           True if `other` might be in the same series as `self`, False
           otherwise.
        """
        # compare signature dictionaries.  The dictionaries each contain
        # comparison rules, we prefer our own when we have them.  If a
        # key is not present in either dictionary, assume the value is
        # None.
        my_sig = self.series_signature
        your_sig = other.series_signature
        my_keys = set(my_sig)
        your_keys = set(your_sig)
        # we have values in both signatures
        for key in my_keys.intersection(your_keys):
            v1, func = my_sig[key]
            v2, _ = your_sig[key]
            if not func(v1, v2):
                return False
        # values present in one or the other but not both
        for keys, sig in ((my_keys - your_keys, my_sig), (your_keys - my_keys, your_sig)):
            for key in keys:
                v1, func = sig[key]
                if not func(v1, None):
                    return False
        return True

    @cached_property
    def scale_factors(self):
        """Return (2, N) array of slope/intercept pairs"""
        scaling = self._get_best_scale_factor(self.dcm_data)
        if scaling is None:
            if self.vendor == Vendor.PHILIPS:
                warnings.warn(
                    'Unable to find Philips private scale factor, cross-series comparisons may be invalid'
                )
            scaling = (1, 0)
        return np.array((scaling,))

    def _get_rwv_scale_factor(self, dcm_data):
        """Return the first set of 'real world' scale factors with defined units"""
        rw_seq = dcm_data.get('RealWorldValueMappingSequence')
        if rw_seq:
            for rw_map in rw_seq:
                try:
                    units = rw_map.MeasurementUnitsCodeSequence[0].CodeMeaning
                except (AttributeError, IndexError):
                    continue
                if units not in ('', 'no units', 'UNDEFINED'):
                    return (
                        rw_map.get('RealWorldValueSlope', 1),
                        rw_map.get('RealWorldValueIntercept', 0),
                    )

    def _get_legacy_scale_factor(self, dcm_data):
        """Return scale factors from older 'Modality LUT' macro

        For Philips data we require RescaleType is defined and not set to 'normalized'
        """
        pix_trans_seq = dcm_data.get('PixelValueTransformationSequence')
        if pix_trans_seq is not None:
            pix_trans = pix_trans_seq[0]
            if self.vendor != Vendor.PHILIPS or pix_trans.get('RescaleType', 'US') not in (
                '',
                'US',
                'normalized',
            ):
                return (pix_trans.get('RescaleSlope', 1), pix_trans.get('RescaleIntercept', 0))
        if (
            dcm_data.get('RescaleSlope') is not None
            or dcm_data.get('RescaleIntercept') is not None
        ):
            if self.vendor != Vendor.PHILIPS or dcm_data.get('RescaleType', 'US') not in (
                '',
                'US',
                'normalized',
            ):
                return (dcm_data.get('RescaleSlope', 1), dcm_data.get('RescaleIntercept', 0))

    def _get_philips_scale_factor(self, dcm_data):
        """Return scale factors from Philips private element

        If we don't have any other scale factors that are tied to real world units, then
        this is the best scaling to use to enable cross-series comparisons
        """
        offset = find_private_section(dcm_data, 0x2005, 'Philips MR Imaging DD 001')
        priv_scale = None if offset is None else dcm_data.get((0x2005, offset + 0xE))
        if priv_scale is not None:
            return (priv_scale.value, 0.0)

    def _get_best_scale_factor(self, dcm_data):
        """Return the most appropriate scale factor found or None"""
        scaling = self._get_rwv_scale_factor(dcm_data)
        if scaling is not None:
            return scaling
        scaling = self._get_legacy_scale_factor(dcm_data)
        if scaling is not None:
            return scaling
        if self.vendor == Vendor.PHILIPS:
            scaling = self._get_philips_scale_factor(dcm_data)
            if scaling is not None:
                return scaling

    def _scale_data(self, data):
        # depending on pydicom and dicom files, values might need casting from
        # Decimal to float
        scale, offset = self.scale_factors[0]
        return self._apply_scale_offset(data, scale, offset)

    def _apply_scale_offset(self, data, scale, offset):
        # a little optimization.  If we are applying either the scale or
        # the offset, we need to allow upcasting to float.
        if scale != 1:
            if offset == 0:
                return data * scale
            return data * scale + offset
        if offset != 0:
            return data + offset
        return data

    @cached_property
    def b_value(self):
        """Return b value for diffusion or None if not available"""
        q_vec = self.q_vector
        if q_vec is None:
            return None
        return q2bg(q_vec)[0]

    @cached_property
    def b_vector(self):
        """Return b vector for diffusion or None if not available"""
        q_vec = self.q_vector
        if q_vec is None:
            return None
        return q2bg(q_vec)[1]


class FrameFilter:
    """Base class for defining how to filter out (ignore) frames from a multiframe file

    It is guaranteed that the `applies` method will called on a dataset before the `keep`
    method is called on any of the frames inside.
    """

    def applies(self, dcm_wrp) -> bool:
        """Returns true if the filter should be applied to a dataset"""
        return True

    def keep(self, frame_data) -> bool:
        """Return true if the frame should be kept"""
        raise NotImplementedError


class FilterMultiStack(FrameFilter):
    """Filter out all but one `StackID`"""

    def __init__(self, keep_id=None):
        self._keep_id = str(keep_id) if keep_id is not None else None

    def applies(self, dcm_wrp) -> bool:
        first_fcs = dcm_wrp.frames[0].get('FrameContentSequence', (None,))[0]
        if first_fcs is None or not hasattr(first_fcs, 'StackID'):
            return False
        stack_ids = {frame.FrameContentSequence[0].StackID for frame in dcm_wrp.frames}
        if self._keep_id is not None:
            if self._keep_id not in stack_ids:
                raise WrapperError('Explicitly requested StackID not found')
            self._selected = self._keep_id
        if len(stack_ids) > 1:
            if self._keep_id is None:
                try:
                    sids = [int(x) for x in stack_ids]
                except:
                    self._selected = dcm_wrp.frames[0].FrameContentSequence[0].StackID
                else:
                    self._selected = str(min(sids))
                warnings.warn(
                    'A multi-stack file was passed without an explicit filter, '
                    f'using StackID = {self._selected}'
                )
            return True
        return False

    def keep(self, frame) -> bool:
        return frame.FrameContentSequence[0].StackID == self._selected


class FilterDwiIso(FrameFilter):
    """Filter out derived ISOTROPIC frames from DWI series"""

    def applies(self, dcm_wrp) -> bool:
        if not hasattr(dcm_wrp.frames[0], 'MRDiffusionSequence'):
            return False
        diff_dirs = {
            f.MRDiffusionSequence[0].get('DiffusionDirectionality') for f in dcm_wrp.frames
        }
        if len(diff_dirs) > 1 and 'ISOTROPIC' in diff_dirs:
            warnings.warn('Derived images found and removed')
            return True
        return False

    def keep(self, frame) -> bool:
        return frame.MRDiffusionSequence[0].DiffusionDirectionality != 'ISOTROPIC'


DEFUALT_FRAME_FILTERS = (FilterMultiStack(), FilterDwiIso())


class MultiframeWrapper(Wrapper):
    """Wrapper for Enhanced MR Storage SOP Class

    Tested with Philips' Enhanced DICOM implementation.

    The specification for the Enhanced MR image IOP / SOP began life as `DICOM
    supplement 49 <ftp://medical.nema.org/medical/dicom/final/sup49_ft.pdf>`_,
    but as of 2016 it is part of the standard. In particular see:

    * `A.36 Enhanced MR Information Object Definitions
      <http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_A.36>`_;
    * `C.7.6.16 Multi-Frame Functional Groups Module
      <http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_C.7.6.16>`_;
    * `C.7.6.17 Multi-Frame Dimension Module
      <http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_C.7.6.17>`_.

    Attributes
    ----------
    is_multiframe : boolean
        Identifies `dcmdata` as multi-frame
    frames : sequence
        A sequence of ``dicom.dataset.Dataset`` objects populated by the
        ``dicom.dataset.Dataset.PerFrameFunctionalGroupsSequence`` attribute
    shared : object
        The first (and only) ``dicom.dataset.Dataset`` object from a
        ``dicom.dataset.Dataset.SharedFunctionalgroupSequence``.

    Methods
    -------
    vendor(self)
    frame_order(self)
    image_shape(self)
    image_orient_patient(self)
    voxel_sizes(self)
    image_position(self)
    series_signature(self)
    scale_factors(self)
    get_data(self)
    """

    is_multiframe = True

    def __init__(self, dcm_data, frame_filters=None):
        """Initializes MultiframeWrapper

        Parameters
        ----------
        dcm_data : object
           object should allow 'get' and '__getitem__' access.  Usually this
           will be a ``dicom.dataset.Dataset`` object resulting from reading a
           DICOM file.

        frame_filters : Iterable of FrameFilter
            defines which frames inside the dataset should be ignored. If None then
            `dicomwrappers.DEFAULT_FRAME_FILTERS` will be used.
        """
        Wrapper.__init__(self, dcm_data)
        self.frames = dcm_data.get('PerFrameFunctionalGroupsSequence')
        try:
            self.frames[0]
        except TypeError:
            raise WrapperError('PerFrameFunctionalGroupsSequence is empty.')
        try:
            self.shared = dcm_data.get('SharedFunctionalGroupsSequence')[0]
        except TypeError:
            raise WrapperError('SharedFunctionalGroupsSequence is empty.')
        # Apply frame filters one at a time in the order provided
        if frame_filters is None:
            frame_filters = DEFUALT_FRAME_FILTERS
        frame_filters = [filt for filt in frame_filters if filt.applies(self)]
        for filt in frame_filters:
            self.frames = [f for f in self.frames if filt.keep(f)]
        # Make sure there is only one StackID remaining
        first_fcs = self.frames[0].get('FrameContentSequence', (None,))[0]
        if first_fcs is not None and hasattr(first_fcs, 'StackID'):
            if len({frame.FrameContentSequence[0].StackID for frame in self.frames}) > 1:
                raise WrapperError('More than one StackID remains after filtering')
        # Try to determine slice order and minimal image position patient
        self._frame_slc_ord = self._ipp = self._slice_spacing = None
        try:
            frame_ipps = [f.PlanePositionSequence[0].ImagePositionPatient for f in self.frames]
        except AttributeError:
            try:
                frame_ipps = [self.shared.PlanePositionSequence[0].ImagePositionPatient]
            except AttributeError:
                frame_ipps = None
        if frame_ipps is not None and all(ipp is not None for ipp in frame_ipps):
            frame_ipps = [np.array(list(map(float, ipp))) for ipp in frame_ipps]
            frame_slc_pos = [np.inner(ipp, self.slice_normal) for ipp in frame_ipps]
            rnd_slc_pos = np.round(frame_slc_pos, 4)
            uniq_slc_pos = np.unique(rnd_slc_pos)
            pos_ord_map = dict(zip(uniq_slc_pos, np.argsort(uniq_slc_pos)))
            self._frame_slc_ord = [pos_ord_map[pos] for pos in rnd_slc_pos]
            if len(self._frame_slc_ord) > 1:
                self._slice_spacing = (
                    frame_slc_pos[self._frame_slc_ord[1]] - frame_slc_pos[self._frame_slc_ord[0]]
                )
            self._ipp = frame_ipps[np.argmin(frame_slc_pos)]
        self._frame_indices = None

    @cached_property
    def vendor(self):
        """The vendor of the instrument that produced the DICOM"""
        vendor = super().vendor
        if vendor is not None:
            return vendor
        vendor = vendor_from_private(self.shared)
        if vendor is not None:
            return vendor
        return vendor_from_private(self.frames[0])

    @cached_property
    def frame_order(self):
        """The ordering of frames to make nD array"""
        if self._frame_indices is None:
            _ = self.image_shape
        return np.lexsort(self._frame_indices.T)

    @cached_property
    def image_shape(self):
        """The array shape as it will be returned by ``get_data()``

        The shape is determined by the *Rows* DICOM attribute, *Columns*
        DICOM attribute, and the set of frame indices given by the
        *FrameContentSequence[0].DimensionIndexValues* DICOM attribute of each
        element in the *PerFrameFunctionalGroupsSequence*.  The first two
        axes of the returned shape correspond to the rows, and columns
        respectively. The remaining axes correspond to those of the frame
        indices with order preserved.

        What each axis in the frame indices refers to is given by the
        corresponding entry in the *DimensionIndexSequence* DICOM attribute.
        **WARNING**: Any axis referring to the *StackID* DICOM attribute will
        have been removed from the frame indices in determining the shape. This
        is because only a file containing a single stack is currently allowed by
        this wrapper.

        References
        ----------
        * C.7.6.16 Multi-Frame Functional Groups Module:
          http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_C.7.6.16
        * C.7.6.17 Multi-Frame Dimension Module:
          http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_C.7.6.17
        * Diagram of DimensionIndexSequence and DimensionIndexValues:
          http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#figure_C.7.6.17-1
        """
        rows, cols = self.get('Rows'), self.get('Columns')
        if None in (rows, cols):
            raise WrapperError('Rows and/or Columns are empty.')
        # Check number of frames and handle single frame files
        n_frames = len(self.frames)
        if n_frames == 1:
            self._frame_indices = np.array([[0]], dtype=np.int64)
            return (rows, cols)
        # Initialize array of frame indices
        try:
            frame_indices = np.array(
                [frame.FrameContentSequence[0].DimensionIndexValues for frame in self.frames]
            )
        except AttributeError:
            raise WrapperError("Can't find frame 'DimensionIndexValues'")
        if len(frame_indices.shape) == 1:
            frame_indices = frame_indices.reshape(frame_indices.shape + (1,))
        # Determine the shape and which indices to use
        shape = [rows, cols]
        curr_parts = n_frames
        frames_per_part = 1
        del_indices = {}
        dim_seq = [dim.DimensionIndexPointer for dim in self.get('DimensionIndexSequence')]
        stackpos_tag = pydicom.datadict.tag_for_keyword('InStackPositionNumber')
        slice_dim_idx = dim_seq.index(stackpos_tag)
        for row_idx, row in enumerate(frame_indices.T):
            unique = np.unique(row)
            count = len(unique)
            if curr_parts == 1 or (count == 1 and row_idx != slice_dim_idx):
                del_indices[row_idx] = count
                continue
            # Replace slice indices with order determined from slice positions along normal
            if row_idx == slice_dim_idx:
                if len(shape) > 2:
                    raise WrapperError('Non-singular index precedes the slice index')
                row = self._frame_slc_ord
                frame_indices.T[row_idx, :] = row
                unique = np.unique(row)
                if len(unique) != count:
                    raise WrapperError("Number of slice indices and positions don't match")
            elif count == n_frames:
                if shape[-1] == 'remaining':
                    raise WrapperError('At most one index have ambiguous size')
                shape.append('remaining')
                continue
            new_parts, leftover = divmod(curr_parts, count)
            expected = new_parts * frames_per_part
            if leftover != 0 or any(np.count_nonzero(row == val) != expected for val in unique):
                if row_idx == slice_dim_idx:
                    raise WrapperError('Missing slices from multiframe')
                del_indices[row_idx] = count
                continue
            if shape[-1] == 'remaining':
                shape[-1] = new_parts
                frames_per_part *= shape[-1]
                new_parts = 1
            frames_per_part *= count
            shape.append(count)
            curr_parts = new_parts
        if shape[-1] == 'remaining':
            if curr_parts > 1:
                shape[-1] = curr_parts
                curr_parts = 1
            else:
                del_indices[len(shape)] = 1
                shape = shape[:-1]
        if del_indices:
            if curr_parts > 1:
                ns_failed = [k for k, v in del_indices.items() if v != 1]
                if len(ns_failed) > 1:
                    # If some indices weren't used yet but we still have unaccounted for
                    # partitions, try combining indices into single tuple and using that
                    tup_dtype = np.dtype(','.join(['I'] * len(ns_failed)))
                    row = [tuple(x for x in vals) for vals in frame_indices[:, ns_failed]]
                    row = np.array(row, dtype=tup_dtype)
            frame_indices = np.delete(frame_indices, np.array(list(del_indices.keys())), axis=1)
            if curr_parts > 1 and len(ns_failed) > 1:
                unique = np.unique(row, axis=0)
                count = len(unique)
                new_parts, rem = divmod(curr_parts, count)
                allowed_val_counts = [new_parts * frames_per_part, n_frames]
                if rem == 0 and all(
                    np.count_nonzero(row == val) in allowed_val_counts for val in unique
                ):
                    shape.append(count)
                    curr_parts = new_parts
                    ord_vals = np.argsort(unique)
                    order = {tuple(unique[i]): ord_vals[i] for i in range(count)}
                    ord_row = np.array([order[tuple(v)] for v in row])
                    frame_indices = np.hstack(
                        [frame_indices, np.array(ord_row).reshape((n_frames, 1))]
                    )
        if curr_parts > 1:
            raise WrapperError('Unable to determine sorting of final dimension(s)')
        # Store frame indices
        self._frame_indices = frame_indices
        return tuple(shape)

    @cached_property
    def image_orient_patient(self):
        """
        Note that this is _not_ LR flipped
        """
        try:
            iop = self.shared.PlaneOrientationSequence[0].ImageOrientationPatient
        except AttributeError:
            try:
                iop = self.frames[0].PlaneOrientationSequence[0].ImageOrientationPatient
            except AttributeError:
                raise WrapperError('Not enough information for image_orient_patient')
        if iop is None:
            return None
        iop = np.array(list(map(float, iop)))
        return np.array(iop).reshape(2, 3).T

    @cached_property
    def voxel_sizes(self):
        """Get i, j, k voxel sizes"""
        try:
            pix_measures = self.shared.PixelMeasuresSequence[0]
        except AttributeError:
            try:
                pix_measures = self.frames[0].PixelMeasuresSequence[0]
            except AttributeError:
                raise WrapperError('Not enough data for pixel spacing')
        pix_space = pix_measures.PixelSpacing
        if self._slice_spacing is not None:
            zs = self._slice_spacing
        else:
            try:
                zs = pix_measures.SliceThickness
            except AttributeError:
                zs = self.get('SpacingBetweenSlices')
                if zs is None:
                    raise WrapperError('Not enough data for slice thickness')
        # Ensure values are float rather than Decimal
        return tuple(map(float, list(pix_space) + [zs]))

    @property
    def image_position(self):
        if self._ipp is None:
            raise WrapperError('Not enough information for image_position_patient')
        return self._ipp

    @cached_property
    def series_signature(self):
        signature = {}
        eq = operator.eq
        for key in ('SeriesInstanceUID', 'SeriesNumber', 'ImageType'):
            signature[key] = (self.get(key), eq)
        signature['image_shape'] = (self.image_shape, eq)
        signature['iop'] = (self.image_orient_patient, none_or_close)
        signature['vox'] = (self.voxel_sizes, none_or_close)
        return signature

    @cached_property
    def scale_factors(self):
        """Return `(2, N)` array of slope/intercept pairs

        If there is a single global scale factor then `N` will be one, otherwise it will
        be the number of frames
        """
        # Look for shared / global RWV scale factor first
        shared_scale = self._get_rwv_scale_factor(self.shared)
        if shared_scale is not None:
            return np.array([shared_scale])
        shared_scale = self._get_rwv_scale_factor(self.dcm_data)
        if shared_scale is not None:
            return np.array([shared_scale])
        # Try pulling out best scale factors from each individual frame
        frame_scales = [self._get_best_scale_factor(f) for f in self.frames]
        if any(s is not None for s in frame_scales):
            if any(s is None for s in frame_scales):
                if self.vendor == Vendor.PHILIPS:
                    warnings.warn(
                        'Unable to find Philips private scale factor, cross-series comparisons may be invalid'
                    )
                frame_scales = [s if s is not None else (1, 0) for s in frame_scales]
            if all(s == frame_scales[0] for s in frame_scales[1:]):
                return np.array([frame_scales[0]])
            return np.array(frame_scales)[self.frame_order]
        # Finally look for shared non-RWV scale factors
        shared_scale = self._get_best_scale_factor(self.shared)
        if shared_scale is not None:
            return np.array([shared_scale])
        shared_scale = self._get_best_scale_factor(self.dcm_data)
        if shared_scale is None:
            if self.vendor == Vendor.PHILIPS:
                warnings.warn(
                    'Unable to find Philips private scale factor, cross-series comparisons may be invalid'
                )
            shared_scale = (1, 0)
        return np.array([shared_scale])

    def get_unscaled_data(self):
        shape = self.image_shape
        if shape is None:
            raise WrapperError('No valid information for image shape')
        data = self.get_pixel_array()
        # Roll frames axis to last and reorder
        if len(data.shape) > 2:
            data = data.transpose((1, 2, 0))[..., self.frame_order]
        return data.reshape(shape, order='F')

    def _scale_data(self, data):
        scale_factors = self.scale_factors
        if scale_factors.shape[0] == 1:
            scale, offset = scale_factors[0]
            return self._apply_scale_offset(data, scale, offset)
        orig_shape = data.shape
        data = data.reshape(data.shape[:2] + (len(self.frames),))
        return (data * scale_factors[:, 0] + scale_factors[:, 1]).reshape(orig_shape)


class SiemensWrapper(Wrapper):
    """Wrapper for Siemens format DICOMs

    Adds attributes:

    * csa_header : mapping
    * b_matrix : (3,3) array
    * q_vector : (3,) array
    """

    is_csa = True

    def __init__(self, dcm_data, csa_header=None):
        """Initialize Siemens wrapper

        The Siemens-specific information is in the `csa_header`, either
        passed in here, or read from the input `dcm_data`.

        Parameters
        ----------
        dcm_data : object
           object should allow 'get' and '__getitem__' access.  If `csa_header`
           is None, it should also be possible to extract a CSA header from
           `dcm_data`. Usually this will be a ``dicom.dataset.Dataset`` object
           resulting from reading a DICOM file.
        csa_header : None or mapping, optional
           mapping giving values for Siemens CSA image sub-header.  If
           None, we try and read the CSA information from `dcm_data`.
           If this fails, we fall back to an empty dict.
        """
        super().__init__(dcm_data)
        if dcm_data is None:
            dcm_data = {}
        self.dcm_data = dcm_data
        if csa_header is None:
            csa_header = csar.get_csa_header(dcm_data)
            if csa_header is None:
                csa_header = {}
        self.csa_header = csa_header

    @cached_property
    def vendor(self):
        """The vendor of the instrument that produced the DICOM"""
        return Vendor.SIEMENS

    @cached_property
    def slice_normal(self):
        # The std_slice_normal comes from the cross product of the directions
        # in the ImageOrientationPatient
        std_slice_normal = super().slice_normal
        csa_slice_normal = csar.get_slice_normal(self.csa_header)
        if std_slice_normal is None and csa_slice_normal is None:
            return None
        elif std_slice_normal is None:
            return np.array(csa_slice_normal)
        elif csa_slice_normal is None:
            return std_slice_normal
        else:
            # Make sure the two normals are very close to parallel unit vectors
            dot_prod = np.dot(csa_slice_normal, std_slice_normal)
            assert np.allclose(np.fabs(dot_prod), 1.0, atol=1e-5)
            # Use the slice normal computed with the cross product as it will
            # always be the most orthogonal, but take the sign from the CSA
            # slice normal
            if dot_prod < 0:
                return -std_slice_normal
            else:
                return std_slice_normal

    @cached_property
    def series_signature(self):
        """Add ICE dims from CSA header to signature"""
        signature = super().series_signature
        ice = csar.get_ice_dims(self.csa_header)
        if ice is not None:
            ice = ice[:6] + ice[8:9]
        signature['ICE_Dims'] = (ice, operator.eq)
        return signature

    @cached_property
    def b_matrix(self):
        """Get DWI B matrix referring to voxel space

        Parameters
        ----------
        None

        Returns
        -------
        B : (3,3) array or None
           B matrix in *voxel* orientation space.  Returns None if this is
           not a Siemens header with the required information.  We return
           None if this is a b0 acquisition
        """
        hdr = self.csa_header
        # read B matrix as recorded in CSA header.  This matrix refers to
        # the space of the DICOM patient coordinate space.
        B = csar.get_b_matrix(hdr)
        if B is None:  # may be not diffusion or B0 image
            bval_requested = csar.get_b_value(hdr)
            if bval_requested is None:
                return None
            if bval_requested != 0:
                raise csar.CSAError('No B matrix and b value != 0')
            return np.zeros((3, 3))
        # rotation from voxels to DICOM PCS, inverted to give the rotation
        # from DPCS to voxels.  Because this is an orthonormal matrix, its
        # transpose is its inverse
        R = self.rotation_matrix.T
        # because B results from V dot V.T, the rotation B is given by R dot
        # V dot V.T dot R.T == R dot B dot R.T
        B_vox = np.dot(R, np.dot(B, R.T))
        # fix presumed rounding errors in the B matrix by making it positive
        # semi-definite.
        return nearest_pos_semi_def(B_vox)

    @cached_property
    def q_vector(self):
        """Get DWI q vector referring to voxel space

        Parameters
        ----------
        None

        Returns
        -------
        q: (3,) array
           Estimated DWI q vector in *voxel* orientation space.  Returns
           None if this is not (detectably) a DWI
        """
        B = self.b_matrix
        if B is None:
            return None
        # We've enforced more or less positive semi definite with the
        # b_matrix routine
        return B2q(B, tol=1e-8)


class MosaicWrapper(SiemensWrapper):
    """Class for Siemens mosaic format data

    Mosaic format is a way of storing a 3D image in a 2D slice - and
    it's as simple as you'd imagine it would be - just storing the slices
    in a mosaic similar to a light-box print.

    We need to allow for this when getting the data and (because of an
    idiosyncrasy in the way Siemens stores the images) calculating the
    position of the first voxel.

    Adds attributes:

    * n_mosaic : int
    * mosaic_size : int
    """

    is_mosaic = True

    def __init__(self, dcm_data, csa_header=None, n_mosaic=None):
        """Initialize Siemens Mosaic wrapper

        The Siemens-specific information is in the `csa_header`, either
        passed in here, or read from the input `dcm_data`.

        Parameters
        ----------
        dcm_data : object
           object should allow 'get' and '__getitem__' access.  If `csa_header`
           is None, it should also be possible for to extract a CSA header from
           `dcm_data`. Usually this will be a ``dicom.dataset.Dataset`` object
           resulting from reading a DICOM file.  A dict should also work.
        csa_header : None or mapping, optional
           mapping giving values for Siemens CSA image sub-header.
        n_mosaic : None or int, optional
           number of images in mosaic.  If None, try to get this number
           from `csa_header`.  If this fails, raise an error
        """
        SiemensWrapper.__init__(self, dcm_data, csa_header)
        if n_mosaic is None:
            try:
                n_mosaic = csar.get_n_mosaic(self.csa_header)
            except KeyError:
                pass
            if n_mosaic is None or n_mosaic == 0:
                raise WrapperError(
                    'No valid mosaic number in CSA header; is this really Siemens mosiac data?'
                )
        self.n_mosaic = n_mosaic
        self.mosaic_size = int(np.ceil(np.sqrt(n_mosaic)))

    @cached_property
    def image_shape(self):
        """Return image shape as returned by ``get_data()``"""
        # reshape pixel slice array back from mosaic
        rows = self.get('Rows')
        cols = self.get('Columns')
        if None in (rows, cols):
            return None
        return (rows // self.mosaic_size, cols // self.mosaic_size, self.n_mosaic)

    @cached_property
    def image_position(self):
        """Return position of first voxel in data block

        Adjusts Siemens mosaic position vector for bug in mosaic format
        position.  See ``dicom_mosaic`` in doc/theory for details.

        Parameters
        ----------
        None

        Returns
        -------
        img_pos : (3,) array
           position in mm of voxel (0,0,0) in Mosaic array
        """
        ipp = super().image_position
        # mosaic image size
        md_rows, md_cols = (self.get('Rows'), self.get('Columns'))
        iop = self.image_orient_patient
        pix_spacing = self.get('PixelSpacing')
        if any(x is None for x in (ipp, md_rows, md_cols, iop, pix_spacing)):
            return None
        # PixelSpacing values are python Decimal in pydicom 0.9.7
        pix_spacing = np.array(list(map(float, pix_spacing)))
        # size of mosaic array before rearranging to 3D.
        md_rc = np.array([md_rows, md_cols])
        # size of slice array after reshaping to 3D
        rd_rc = md_rc / self.mosaic_size
        # apply algorithm for undoing mosaic translation error - see
        # ``dicom_mosaic`` doc
        vox_trans_fixes = (md_rc - rd_rc) / 2
        # flip IOP field to refer to rows then columns index change -
        # see dicom_orientation doc
        Q = np.fliplr(iop) * pix_spacing
        return ipp + np.dot(Q, vox_trans_fixes[:, None]).ravel()

    def get_unscaled_data(self):
        """Get scaled image data from DICOMs

        Resorts data block from mosaic to 3D

        Returns
        -------
        data : array
           array with data as scaled from any scaling in the DICOM
           fields.

        Notes
        -----
        The apparent image in the DICOM file is a 2D array that consists of
        blocks, that are the output 2D slices.  Let's call the original array
        the *slab*, and the contained slices *slices*.   The slices are of
        pixel dimension ``n_slice_rows`` x ``n_slice_cols``.  The slab is of
        pixel dimension ``n_slab_rows`` x ``n_slab_cols``.  Because the
        arrangement of blocks in the slab is defined as being square, the
        number of blocks per slab row and slab column is the same.  Let
        ``n_blocks`` be the number of blocks contained in the slab.  There is
        also ``n_slices`` - the number of slices actually collected, some
        number <= ``n_blocks``.  We have the value ``n_slices`` from the
        'NumberOfImagesInMosaic' field of the Siemens private (CSA) header.
        ``n_row_blocks`` and ``n_col_blocks`` are therefore given by
        ``ceil(sqrt(n_slices))``, and ``n_blocks`` is ``n_row_blocks ** 2``.
        Also ``n_slice_rows == n_slab_rows / n_row_blocks``, etc.  Using these
        numbers we can therefore reconstruct the slices from the 2D DICOM pixel
        array.
        """
        shape = self.image_shape
        if shape is None:
            raise WrapperError('No valid information for image shape')
        n_slice_rows, n_slice_cols, n_mosaic = shape
        n_slab_rows = self.mosaic_size
        n_blocks = n_slab_rows**2
        data = self.get_pixel_array()
        v4 = data.reshape(n_slab_rows, n_slice_rows, n_slab_rows, n_slice_cols)
        # move the mosaic dims to the end
        v4 = v4.transpose((1, 3, 0, 2))
        # pool mosaic-generated dims
        v3 = v4.reshape((n_slice_rows, n_slice_cols, n_blocks))
        # delete any padding slices
        return v3[..., :n_mosaic]


def none_or_close(val1, val2, rtol=1e-5, atol=1e-6):
    """Match if `val1` and `val2` are both None, or are close

    Parameters
    ----------
    val1 : None or array-like
    val2 : None or array-like
    rtol : float, optional
       Relative tolerance; see ``np.allclose``
    atol : float, optional
       Absolute tolerance; see ``np.allclose``

    Returns
    -------
    tf : bool
       True iff (both `val1` and `val2` are None) or (`val1` and `val2`
       are close arrays, as detected by ``np.allclose`` with parameters
       `rtol` and `atal`).

    Examples
    --------
    >>> none_or_close(None, None)
    True
    >>> none_or_close(1, None)
    False
    >>> none_or_close(None, 1)
    False
    >>> none_or_close([1,2], [1,2])
    True
    >>> none_or_close([0,1], [0,2])
    False
    """
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False
    return np.allclose(val1, val2, rtol, atol)
