"""
Transformer used to apply basic transformations on multi subject MRI data.
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import warnings

import numpy as np
from sklearn.externals.joblib import Memory

from nibabel import Nifti1Image

from .. import masking
from .. import resampling
from .. import utils
from .base_masker import BaseMasker


class NiftiMultiMasker(BaseMasker):
    """Nifti data loader with preprocessing for multiple subjects

    Parameters
    ----------
    mask: filename or NiImage, optional
        Mask of the data. If not given, a mask is computed in the fit step.
        Optional parameters detailed below (mask_connected...) can be set to
        fine tune the mask extraction.

    smooth: False or float, optional
        If smooth is not False, it gives the size, in voxel of the
        spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1.

    detrend: boolean, optional
        This parameter is passed to signals.clean. Please see the related
        documentation for details

    low_pass: False or float, optional
        This parameter is passed to signals.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signals.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signals.clean. Please see the related
        documentation for details

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    transform_memory: instance of joblib.Memory or string
        Used to cache the perprocessing step.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose: interger, optional
        Indicate the level of verbosity. By default, nothing is printed

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to resampling.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to resampling.resample_img. Please see the
        related documentation for details.

    mask_connected: boolean, optional
        If mask is None, this parameter is passed to masking.compute_epi_mask
        for mask computation. Please see the related documentation for details.

    mask_opening: boolean, optional
        If mask is None, this parameter is passed to masking.compute_epi_mask
        for mask computation. Please see the related documentation for details.

    mask_lower_cutoff: float, optional
        If mask is None, this parameter is passed to masking.compute_epi_mask
        for mask computation. Please see the related documentation for details.

    mask_upper_cutoff: float, optional
        If mask is None, this parameter is passed to masking.compute_epi_mask
        for mask computation. Please see the related documentation for details.

    transpose: boolean, optional
        If True, data is transposed after preprocessing step.

    Attributes
    ----------
    `mask_img_`: Nifti like image
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `affine_`: 4x4 numpy array
        Affine of the transformed NiImages. If affine is different across
        subjects, contains the affine of the first subject on which other
        subject data have been resampled.

    See also
    --------
    nisl.masking.compute_epi_mask
    nisl.resampling.resample_img
    nisl.masking.apply_mask
    nisl.signals.clean
    """

    def __init__(self, mask=None, mask_connected=True,
                 mask_opening=False, mask_lower_cutoff=0.2,
                 mask_upper_cutoff=0.9,
                 smooth=False, standardize=False, detrend=False,
                 target_affine=None, target_shape=None, low_pass=None,
                 high_pass=None, t_r=None, transpose=False, n_jobs=1,
                 memory=Memory(cachedir=None, verbose=0),
                 transform_memory=Memory(cachedir=None, verbose=0), verbose=0):
        # Mask is compulsory or computed
        self.mask = mask
        self.mask_connected = mask_connected
        self.mask_opening = mask_opening
        self.mask_lower_cutoff = mask_lower_cutoff
        self.mask_upper_cutoff = mask_upper_cutoff
        self.smooth = smooth
        self.standardize = standardize
        self.detrend = detrend
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory
        self.transform_memory = transform_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.transpose = transpose

    def fit(self, niimgs, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.
        """

        memory = self.memory
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory)

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.fit] Loading data from %s" % (
                self.__class__.__name__,
                utils._repr_niimgs(niimgs)[:200])
        data = []
        for niimg in niimgs:
            # Note that data is not loaded into memory at this stage
            # if niimg is a string
            data.append(utils.check_niimgs(niimg, accept_3d=True))

        # Compute the mask if not given by the user
        if self.mask is None:
            if self.verbose > 0:
                print "[%s.fit] Computing the mask" % self.__class__.__name__
            mask = memory.cache(masking.compute_multi_epi_mask,
                                ignore=['verbose'])(
                                    niimgs,
                                    connected=self.mask_connected,
                                    opening=self.mask_opening,
                                    lower_cutoff=self.mask_lower_cutoff,
                                    upper_cutoff=self.mask_upper_cutoff,
                                    n_jobs=self.n_jobs,
                                    verbose=(self.verbose - 1))
            self.mask_img_ = Nifti1Image(mask.astype(np.int), data[0].get_affine())
        else:
            self.mask_img_ = utils.check_niimg(self.mask)

        # If resampling is requested, resample also the mask
        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[%s.transform] Resampling mask" % self.__class__.__name__
        self.mask_img_ = memory.cache(resampling.resample_img)(
            self.mask_img_,
            target_affine=self.target_affine,
            target_shape=self.target_shape,
            copy=(self.target_affine is not None and
                  self.target_shape is not None))

        return self

    def transform(self, niimgs, confounds=None):
        """ Apply mask, spatial and temporal preprocessing

        Parameters
        ----------
        niimgs: nifti like images
            Data to be preprocessed

        confounds: CSV file path or 2D matrix
            This parameter is passed to signals.clean. Please see the related
            documentation for details
        """
        data = []
        affine = None
        for index, niimg in enumerate(niimgs):
            # If we have a string (filename), we won't need to copy, as
            # there will be no side effect
            copy = not isinstance(niimg, basestring)
            niimg = utils.check_niimgs(niimg)

            if affine is not None and np.all(niimg.get_affine() != affine):
                warnings.warn('Affine is different across subjects.'
                              ' Realignement on first subject affine forced')
                self.target_affine = affine
            if confounds is not None:
                data.append(self.transform_single_niimgs(
                    niimg, confounds=confounds[index],
                    copy=copy))
            else:
                data.append(self.transform_single_niimgs(niimg,
                                                         copy=copy))
            if affine is None:
                affine = self.affine_
        return data
