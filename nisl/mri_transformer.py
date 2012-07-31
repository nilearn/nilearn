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

    def __init__(self, mask=None, args):
        # Mask is compulsory or computed
        # Must integrate memory, as in WardAgglomeration
        pass

    def fit(self, X, y=None):
        """
        X: list of filenames or NiImages
        """
        # Spatial
        # =======
        #
        # optional compute_mask (nipy.labs.mask.compute_mask_files)

        if mask is None:
             masking.compute_mask(np.mean(X, axis=0))

        # resampling:
        # nipy.labs.datasets.volumes.VolumeImg.as_volume_img
        # affine is either 4x4, 3x3
        # Function that does that exposes interpolation order, but not
        # this object
        # optional extract series (apply mask) (nipy.labs.mask.series_from_mask)
        #           -> smoothing

        # Temporal
        # ========
        # (parietal-python) parietal.time_series.preprocessing.clean_signals
        # Detrending (optional)
        # Filtering (grab TR from header)
        # Confounds (from csv file or numpy array)
        # Normalizing

        # For _later_: missing value removal or imputing of missing data
        # (i.e. we want to get rid of NaNs, if smoothing must be done
        # earlier)
        # Optionally: 'doctor_nan', remove voxels with NaNs, other option
        # for later: some form of imputation

