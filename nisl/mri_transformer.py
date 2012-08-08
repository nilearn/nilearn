"""
Transformer used to apply basic tranisformations on MRI data.
"""

import types
import collections

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

import nibabel

from . import masking
from . import resampling
from . import preprocessing


def _check_nifti_methods(object):
    try:
        get_data = getattr(object, "get_data")
        get_affine = getattr(object, "get_affine")
        return callable(get_data) and callable(get_affine)
    except Exception:
        return False


def check_niimg(data):
    if isinstance(data, types.StringTypes):
        # data is a filename, we load it
        result = nibabel.load(data)
    else:
        # it is an object, it should have get_data and get_affine methods
        if not _check_nifti_methods(data):
            raise AttributeError("Given data does not expose"
                "data of affine getter")
        result = data
    return result


def load_data(data):
    """Check and load data.

    Parameters
    ----------
    data: string, list of strings or iterable
        If given filename(s), load it. If given data, check that it
        is in the right format.
    """
    # If it is a string, it should a path to a Nifti file
    if (isinstance(data, types.StringTypes)):
        return nibabel.load(data)
    # Check if it is a Nifti file
    if _check_nifti_methods(data):
        return data
    # If it's not a string it must be iterable
    if (not isinstance(data, collections.Iterable)):
        raise ValueError("data must be a filename or iterable")
    for index, item in enumerate(data):
        if (isinstance(item, types.StringTypes)):
            img = nibabel.load(item)
        else:
            # TODO: do we uncomment this part ? It would lead to data copy
            # which may provoke memory issues...
            #
            # if _check_nifti_methods(data):
            #     img = data
            # else:
                raise ValueError
        if index == 0:
            # Initialize the final array
            result = np.empty(img.get_shape() + (len(data),))
            affine = img.get_affine()
        else:
            if not np.array_equal(affine, img.get_affine()):
                raise ValueError("Affine must the same in all images")
        result[..., index] = img.get_data()
    return result


# Rmk: needs a similar object for ROIs or not

class MRITransformer(BaseEstimator, TransformerMixin):
    """MRI data loader with preprocessing

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

    def __init__(self, mask=None, mask_connected=False, mask_opening=False,
            mask_lower_cutoff=0.2, mask_upper_cutoff=0.9,
            smooth=False, confounds=None, detrend=False,
            affine=None, low_pass=None, high_pass=None, t_r=None,
            memory=Memory(cachedir=None, verbose=0), verbose=0):
        # Mask is compulsory or computed
        # Must integrate memory, as in WardAgglomeration
        self.mask = mask
        self.mask_connected = mask_connected
        self.mask_opening = mask_opening
        self.mask_lower_cutoff = mask_lower_cutoff
        self.mask_upper_cutoff = mask_upper_cutoff
        self.smooth = smooth
        self.confounds = confounds
        self.detrend = detrend
        self.affine = affine
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory
        self.verbose = verbose

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
            print "[MRITransformer.fit] Loading data"
        if (isinstance(X, types.StringTypes)):
            data = []
            affine = None
            # Roll axis to index by scan
            scans = np.roll_axis(X, -1)
            for scan in scans:
                img = check_niimg(scan)
                if affine is None:
                    affine = img.get_affine()
                else:
                    if not np.array_equal(affine, img.get_affine()):
                        raise ValueError("affine is not the same"
                                "for all images")
                data.append(img.get_data())
                del img
            data = np.asarray(data)
            np.rollaxis(data, 0, start=4)
        else:
            img = check_niimg(X)
            affine = img.get_affine()
            data = img.get_data()

        # Compute the mask if not given by the user
        if self.mask is None:
            if self.verbose > 0:
                print "[MRITransformer.fit] Computing the mask"
            self.mask_ = masking.compute_epi_mask(np.mean(data, axis=-1),
                    connected=self.mask_connected, opening=self.mask_opening,
                    lower_cutoff=self.mask_lower_cutoff,
                    upper_cutoff=self.mask_upper_cutoff)
        else:
            if isinstance(self.mask, types.StringTypes):
                self.mask_ = nibabel.load(self.mask).get_data() \
                    .astype(np.bool)
            else:
                self.mask_ = self.mask

        return self

    def transform(self, X):
        memory = self.memory

        # Load data (if filenames are given, load them)
        # TODO paste of the code of fit, make a function
        if self.verbose > 0:
            print "[MRITransformer.fit] Loading data"
        if (isinstance(X, types.StringTypes)):
            data = []
            affine = None
            # Roll axis to index by scan
            scans = np.roll_axis(X, -1)
            for scan in scans:
                img = check_niimg(scan)
                if affine is None:
                    affine = img.get_affine()
                else:
                    if not np.array_equal(affine, img.get_affine()):
                        raise ValueError("affine is not the same"
                                "for all images")
                data.append(img.get_data())
                del img
            data = np.asarray(data)
            np.rollaxis(data, 0, start=4)
        else:
            img = check_niimg(X)
            affine = img.get_affine()
            data = img.get_data()

        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[MRITransformer.transform] Resampling"
        data, affine = memory.cache(resampling.as_volume_img)(data, affine,
                new_affine=self.affine, copy=False)

        # Function that does that exposes interpolation order, but not
        # this object
        # XXX -> ?

        # Get series from data with optional smoothing
        if self.verbose > 0:
            print "[MRITransformer.transform] Masking and smoothing"
        data = masking.series_from_mask(data, affine,
                self.mask_, smooth=self.smooth)

        # Temporal
        # ========
        # Detrending (optional)
        # Filtering (grab TR from header)
        # Confounds (from csv file or numpy array)
        # Normalizing

        if self.verbose > 0:
            print "[MRITransformer.transform] Cleaning signal"
        data = memory.cache(preprocessing.clean_signals)(data,
                confounds=self.confounds, low_pass=self.low_pass,
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
        if len(X.shape) > self.mask_.shape:
            shape = self.mask_.shape + (X.shape[-1],)
        else:
            shape = self.mask_.shape
        data = np.ma.masked_all(shape)
        # As values are assigned to voxel, they will be automatically unmasked
        data[self.mask_] = np.rollaxis(X, -1)
        return data
