# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read / write access to SPM99 version of analyze image format"""

import warnings
from io import BytesIO

import numpy as np

from . import analyze  # module import
from .batteryrunners import Report
from .optpkg import optional_package
from .spatialimages import HeaderDataError, HeaderTypeError

have_scipy = optional_package('scipy')[1]

""" Support subtle variations of SPM version of Analyze """
header_key_dtd = analyze.header_key_dtd
# funused1 in dime subfield is scalefactor
image_dimension_dtd = analyze.image_dimension_dtd.copy()
image_dimension_dtd[image_dimension_dtd.index(('funused1', 'f4'))] = ('scl_slope', 'f4')
# originator text field used as image origin (translations)
data_history_dtd = analyze.data_history_dtd.copy()
data_history_dtd[data_history_dtd.index(('originator', 'S10'))] = ('origin', 'i2', (5,))

# Full header numpy dtype combined across sub-fields
header_dtype = np.dtype(header_key_dtd + image_dimension_dtd + data_history_dtd)


class SpmAnalyzeHeader(analyze.AnalyzeHeader):
    """Basic scaling Spm Analyze header"""

    # Copies of module level definitions
    template_dtype = header_dtype

    # data scaling capabilities
    has_data_slope = True
    has_data_intercept = False

    @classmethod
    def default_structarr(klass, endianness=None):
        """Create empty header binary block with given endianness"""
        hdr_data = super().default_structarr(endianness)
        hdr_data['scl_slope'] = 1
        return hdr_data

    def get_slope_inter(self):
        """Get scalefactor and intercept

        If scalefactor is 0.0 return None to indicate no scalefactor.
        Intercept is always None because SPM99 analyze cannot store intercepts.
        """
        slope = self._structarr['scl_slope']
        # Return invalid slopes as None
        if np.isnan(slope) or slope in (0, -np.inf, np.inf):
            return None, None
        return slope, None

    def set_slope_inter(self, slope, inter=None):
        """Set slope and / or intercept into header

        Set slope and intercept for image data, such that, if the image
        data is ``arr``, then the scaled image data will be ``(arr *
        slope) + inter``

        The SPM Analyze header can't save an intercept value, and we raise an
        error unless `inter` is None, NaN or 0

        Parameters
        ----------
        slope : None or float
           If None, implies `slope` of NaN.  NaN is a signal to the image
           writing routines to rescale on save.  0, Inf, -Inf are invalid and
           cause a HeaderDataError
        inter : None or float, optional
           intercept. Must be None, NaN or 0, because SPM99 cannot store
           intercepts.
        """
        if slope is None:
            slope = np.nan
        if slope in (0, -np.inf, np.inf):
            raise HeaderDataError('Slope cannot be 0 or infinite')
        self._structarr['scl_slope'] = slope
        if inter in (None, 0) or np.isnan(inter):
            return
        raise HeaderTypeError('Cannot set non-zero intercept for SPM headers')


class Spm99AnalyzeHeader(SpmAnalyzeHeader):
    """Class for SPM99 variant of basic Analyze header

    SPM99 variant adds the following to basic Analyze format:

    * voxel origin;
    * slope scaling of data.
    """

    def get_origin_affine(self):
        """Get affine from header, using SPM origin field if sensible

        The default translations are got from the ``origin``
        field, if set, or from the center of the image otherwise.

        Examples
        --------
        >>> hdr = Spm99AnalyzeHeader()
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.set_zooms((3, 2, 1))
        >>> hdr.default_x_flip
        True
        >>> hdr.get_origin_affine() # from center of image
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        >>> hdr['origin'][:3] = [3,4,5]
        >>> hdr.get_origin_affine() # using origin
        array([[-3.,  0.,  0.,  6.],
               [ 0.,  2.,  0., -6.],
               [ 0.,  0.,  1., -4.],
               [ 0.,  0.,  0.,  1.]])
        >>> hdr['origin'] = 0 # unset origin
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.get_origin_affine() # from center of image
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        """
        hdr = self._structarr
        zooms = hdr['pixdim'][1:4].copy()
        if self.default_x_flip:
            zooms[0] *= -1
        # Get translations from origin, or center of image
        # Remember that the origin is for matlab (1-based indexing)
        origin = hdr['origin'][:3]
        dims = hdr['dim'][1:4]
        if np.any(origin) and np.all(origin > -dims) and np.all(origin < dims * 2):
            origin = origin - 1
        else:
            origin = (dims - 1) / 2.0
        aff = np.eye(4)
        aff[:3, :3] = np.diag(zooms)
        aff[:3, -1] = -origin * zooms
        return aff

    get_best_affine = get_origin_affine

    def set_origin_from_affine(self, affine):
        """Set SPM origin to header from affine matrix.

        The ``origin`` field was read but not written by SPM99 and 2.  It was
        used for storing a central voxel coordinate, that could be used in
        aligning the image to some standard position - a proxy for a full
        translation vector that was usually stored in a separate matlab .mat
        file.

        Nifti uses the space occupied by the SPM ``origin`` field for important
        other information (the transform codes), so writing the origin will
        make the header a confusing Nifti file.  If you work with both Analyze
        and Nifti, you should probably avoid doing this.

        Parameters
        ----------
        affine : array-like, shape (4,4)
           Affine matrix to set

        Returns
        -------
        None

        Examples
        --------
        >>> hdr = Spm99AnalyzeHeader()
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.set_zooms((3,2,1))
        >>> hdr.get_origin_affine()
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        >>> affine = np.diag([3,2,1,1])
        >>> affine[:3,3] = [-6, -6, -4]
        >>> hdr.set_origin_from_affine(affine)
        >>> np.all(hdr['origin'][:3] == [3,4,5])
        True
        >>> hdr.get_origin_affine()
        array([[-3.,  0.,  0.,  6.],
               [ 0.,  2.,  0., -6.],
               [ 0.,  0.,  1., -4.],
               [ 0.,  0.,  0.,  1.]])
        """
        if affine.shape != (4, 4):
            raise ValueError('Need 4x4 affine to set')
        hdr = self._structarr
        RZS = affine[:3, :3]
        Z = np.sqrt(np.sum(RZS * RZS, axis=0))
        T = affine[:3, 3]
        # Remember that the origin is for matlab (1-based) indexing
        hdr['origin'][:3] = -T / Z + 1

    @classmethod
    def _get_checks(klass):
        checks = super()._get_checks()
        return checks + (klass._chk_origin,)

    @staticmethod
    def _chk_origin(hdr, fix=False):
        rep = Report(HeaderDataError)
        origin = hdr['origin'][0:3]
        dims = hdr['dim'][1:4]
        if not np.any(origin) or (np.all(origin > -dims) and np.all(origin < dims * 2)):
            return hdr, rep
        rep.problem_level = 20
        rep.problem_msg = 'very large origin values relative to dims'
        if fix:
            rep.fix_msg = 'leaving as set, ignoring for affine'
        return hdr, rep


class Spm99AnalyzeImage(analyze.AnalyzeImage):
    """Class for SPM99 variant of basic Analyze image"""

    header_class = Spm99AnalyzeHeader
    header: Spm99AnalyzeHeader
    files_types = (('image', '.img'), ('header', '.hdr'), ('mat', '.mat'))
    has_affine = True
    makeable = True
    rw = have_scipy

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
        img : Spm99AnalyzeImage instance

        """
        ret = super().from_file_map(file_map, mmap=mmap, keep_file_open=keep_file_open)
        try:
            matf = file_map['mat'].get_prepare_fileobj()
        except OSError:
            return ret
        # Allow for possibility of empty file -> no update to affine
        with matf:
            contents = matf.read()
        if len(contents) == 0:
            return ret
        import scipy.io as sio  # type: ignore[import]

        mats = sio.loadmat(BytesIO(contents))
        if 'mat' in mats:  # this overrides a 'M', and includes any flip
            mat = mats['mat']
            if mat.ndim > 2:
                warnings.warn('More than one affine in "mat" matrix, using first')
                mat = mat[:, :, 0]
            ret._affine = mat
        elif 'M' in mats:  # the 'M' matrix does not include flips
            hdr = ret._header
            if hdr.default_x_flip:
                ret._affine = np.dot(np.diag([-1, 1, 1, 1]), mats['M'])
            else:
                ret._affine = mats['M']
        else:
            raise ValueError('mat file found but no "mat" or "M" in it')
        # Adjust for matlab 1,1,1 voxel origin
        to_111 = np.eye(4)
        to_111[:3, 3] = 1
        ret._affine = np.dot(ret._affine, to_111)
        return ret

    def to_file_map(self, file_map=None, dtype=None):
        """Write image to `file_map` or contained ``self.file_map``

        Extends Analyze ``to_file_map`` method by writing ``mat`` file

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        """
        if file_map is None:
            file_map = self.file_map
        super().to_file_map(file_map, dtype=dtype)
        mat = self._affine
        if mat is None:
            return
        import scipy.io as sio

        hdr = self._header
        if hdr.default_x_flip:
            M = np.dot(np.diag([-1, 1, 1, 1]), mat)
        else:
            M = mat
        # Adjust for matlab 1,1,1 voxel origin
        from_111 = np.eye(4)
        from_111[:3, 3] = -1
        M = np.dot(M, from_111)
        mat = np.dot(mat, from_111)
        # use matlab 4 format to allow gzipped write without error
        with file_map['mat'].get_prepare_fileobj(mode='wb') as mfobj:
            sio.savemat(mfobj, {'M': M, 'mat': mat}, format='4')


load = Spm99AnalyzeImage.from_filename
save = Spm99AnalyzeImage.instance_to_filename
