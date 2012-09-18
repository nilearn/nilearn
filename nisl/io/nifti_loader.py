"""
Transformer used to apply basic tranisformations on MRI data.
"""

import types
import collections

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

import nibabel
from nibabel import Nifti1Image

from .. import masking
from .. import resampling
from .. import signals
from .. import utils


class NiftiLoader(BaseEstimator, TransformerMixin):
    """Nifti data loader with preprocessing

    Parameters
    ----------
    mask: 3D numpy matrix, optional
        Mask of the data. If not given, a mask is computed in the fit step.

    smooth: False or float, optional
        If smooth is not False, it gives the size, in voxel of the
        spatial smoothing to apply to the signal.

    confounds: CSV file path or 2D matrix

    detrend: boolean, optional

    verbose: interger, optional
        Indicate the level of verbosity. By default, nothing is printed
    """

    def __init__(self, sessions=None, mask=None, mask_connected=False,
            mask_opening=False, mask_lower_cutoff=0.2, mask_upper_cutoff=0.9,
            smooth=False, confounds=None, detrend=False,
            target_affine=None, target_shape=None, low_pass=None, high_pass=None,
            t_r=None, copy=False, memory=Memory(cachedir=None, verbose=0),
            transform_memory=Memory(cachedir=None, verbose=0), verbose=0):
        # Mask is compulsory or computed
        self.mask = mask
        self.mask_connected = mask_connected
        self.mask_opening = mask_opening
        self.mask_lower_cutoff = mask_lower_cutoff
        self.mask_upper_cutoff = mask_upper_cutoff
        self.smooth = smooth
        self.confounds = confounds
        self.detrend = detrend
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.copy = copy
        self.memory = memory
        self.transform_memory = transform_memory
        self.verbose = verbose
        self.sessions_ = sessions

    def load_imgs(self, imgs):
        # Initialization: 
        # If given data is a list, we count the number of levels to check
        # dimensionality and make a consistent error message.
        depth = 0
        first_img = imgs
        while isinstance(first_img, collections.Iterable) \
                and not isinstance(first_img, types.StringTypes):
            first_img = iter(first_img).next()
            depth += 1

        # First Image is supposed to be a path or a Nifti like element
        first_img = utils.check_nifti_image(first_img)

        # Check dimension and depth
        dim = len(first_img.get_data().shape)

        if (dim + depth) != 4:
            # This error message is poor but givin details about each case
            # would be too complicated
            raise ValueError("[%s] Data must be either a 4D Nifti image or a"
                    " list of 3D Nifti images. You provided a %s %dD image(s)."
                    % (self.__class__.__name__, 'list of ' * depth, dim))

        # Now, we that data is in a known format, load it
        if dim == 4:
            data = utils.check_nifti_image(imgs)
            affine = data.get_affine()
            data = data.get_data()
        else:
            data, affine = utils.collapse_nifti_images(imgs)
        self.affine = affine
        return data

    def fit(self, X, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        X: list of filenames or NiImages
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.
        """

        memory = self.memory
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory)

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.fit] Loading data" % self.__class__.__name__
        data = self.load_imgs(X)

        # Compute the mask if not given by the user
        if self.mask is None:
            if self.verbose > 0:
                print "[%s.fit] Computing the mask" % self.__class__.__name__
            self.mask_ = memory.cache(masking.compute_epi_mask)(np.mean(data, axis=-1),
                    connected=self.mask_connected, opening=self.mask_opening,
                    lower_cutoff=self.mask_lower_cutoff,
                    upper_cutoff=self.mask_upper_cutoff, verbose=(self.verbose -1))
        else:
            if isinstance(self.mask, types.StringTypes):
                self.mask_ = nibabel.load(self.mask).get_data() \
                    .astype(np.bool)
            else:
                self.mask_ = self.mask

        # If resampling is requested, resample also the mask
        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[%s.transform] Resampling mask" % self.__class__.__name__
        self.mask_, _ = memory.cache(resampling.resample)(self.mask_,
                self.affine, target_affine=self.target_affine,
                target_shape=self.target_shape)

        return self

    def transform(self, X):
        memory = self.transform_memory
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory)

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.fit] Loading data" % self.__class__.__name__
        data = self.load_imgs(X)
        affine = self.affine

        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[%s.transform] Resampling" % self.__class__.__name__
        data, affine = memory.cache(resampling.resample)(data, affine,
                target_affine=self.target_affine,
                target_shape=self.target_shape, copy=self.copy)

        # Get series from data with optional smoothing
        if self.verbose > 0:
            print "[%s.transform] Masking and smoothing" \
                % self.__class__.__name__
        data = masking.extract_time_series(data, affine,
                self.mask_, smooth=self.smooth)

        # Temporal
        # ========
        # Detrending (optional)
        # Filtering (grab TR from header)
        # Confounds (from csv file or numpy array)
        # Normalizing

        if self.verbose > 0:
            print "[%s.transform] Cleaning signal" % self.__class__.__name__
        if self.sessions_ is None:
            data = memory.cache(signals.clean)(data,
                    confounds=self.confounds, low_pass=self.low_pass,
                    high_pass=self.high_pass, t_r=self.t_r,
                    detrend=self.detrend, normalize=False)
        else:
            for s in np.unique(self.sessions_):
                if self.confounds is not None:
                    session_confounds = self.confounds[self.sessions_ == s]
                    data[self.sessions_ == s] = \
                        memory.cache(signals.clean)(
                                data=data[self.sessions_ == s],
                                confounds=session_confounds,
                                low_pass=self.low_pass,
                                high_pass=self.high_pass, t_r=self.t_r,
                                detrend=self.detrend, normalize=False)

        # For _later_: missing value removal or imputing of missing data
        # (i.e. we want to get rid of NaNs, if smoothing must be done
        # earlier)
        # Optionally: 'doctor_nan', remove voxels with NaNs, other option
        # for later: some form of imputation

        # data is in format voxel x time_series. We inverse it
        data = np.rollaxis(data, -1)

        self.affine_ = affine
        return data

    def inverse_transform(self, X):
        null = 0
        if len(X.shape) > 1:
            # we build the data iteratively to avoid MemoryError
            data = []
            for x in X:
                img = np.empty(self.mask_.shape)
                img.fill(null)
                img[self.mask_] = x
                data.append(img)
            data = np.asarray(data)
        else:
            data = np.empty(self.mask_.shape)
            data.fill(null)
            data[self.mask_] = X
        return Nifti1Image(data, self.affine)
