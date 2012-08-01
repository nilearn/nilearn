"""
Transformer used to apply basic tranformations on MRI data.
"""

from sklearn.base import BaseEstimator, TransformerMixin

# Note: every input here is either a filename or a NiImage

# check_niimg: utility function that loads if given filename, else
# check 'get_data', 'get_affine'

import types
import nibabel
import masking
import resampling
import preprocessing
import numpy as np


def check_niimg(data):
    if isinstance(data, types.StringTypes):
        # data is a filename, we load it
        result = nibabel.load(data)
    else:
        # it is an object, it should have get_data and get_affine methods
        # list the methods of the object
        methods = [method for method in dir(data)
                if callable(getattr(data, method))]
        if not ('get_data' in methods and 'get_affine' in methods):
            raise AttributeError('missing get_data or get_affine method')
        result = data
    return result


# Rmk: needs a similar object for ROIs or not

class NiftiLoader(BaseEstimator, TransformerMixin):

    def __init__(self, mask=None, smooth=False, confounds=None, detrend=False):
        # Mask is compulsory or computed
        # Must integrate memory, as in WardAgglomeration
        self.mask = mask
        self.smooth = smooth
        self.confounds = confounds
        self.detrend = detrend

    def fit(self, X, y=None):
        """
        X: list of filenames or NiImages
        """
        # Spatial
        # =======
        #
        # optional compute_mask (nipy.labs.mask.compute_mask_files)

        mask = self.mask
        if mask is None:
            mask = masking.compute_mask(np.mean(X, axis=3))

        # resampling:
        # nipy.labs.datasets.volumes.VolumeImg.as_volume_img
        # affine is either 4x4, 3x3

        resampled, affine = resampling.as_volume_img(X.get_data(),
                affine=X.get_affine(), copy=False)

        # Function that does that exposes interpolation order, but not
        # this object
        # optional extract series (apply mask: nipy.labs.mask.series_from_mask)
        #           -> smoothing

        series, header = masking.series_from_mask(resampled,
                mask, smooth=self.smooth)

        # Temporal
        # ========
        # (parietal-python) parietal.time_series.preprocessing.clean_signals
        # Detrending (optional)
        # Filtering (grab TR from header)
        # Confounds (from csv file or numpy array)
        # Normalizing

        signals = preprocessing.clean_signals(series, confounds=self.confounds,
                detrend=self.detrend)

        # For _later_: missing value removal or imputing of missing data
        # (i.e. we want to get rid of NaNs, if smoothing must be done
        # earlier)
        # Optionally: 'doctor_nan', remove voxels with NaNs, other option
        # for later: some form of imputation

        return signals
