"""
Preprocessing functions for images

See also nisl.signal.
"""
# Authors: Philippe Gervais, Alexandre Abraham
# License: simplified BSD

import numpy as np
from . import signals
from . import utils
from . import masking


def high_variance_confounds(niimgs, n_confounds=10, percentile=1.,
                            detrend=True, mask_img=None):
    """ Return confounds signals extracted from input signals with highest
        variance.

        Parameters
        ==========
        niimgs: niimg
            4D image.

        mask_img: niimg
            If provided, confounds are extracted from voxels inside the mask.
            If not provided, all voxels are used.

        n_confounds (int)
            Number of confounds to return

        percentile: float
            Highest-variance signals percentile to keep before computing the
            singular value decomposition.
            signals.shape[0] * percentile must be greater than n_confounds.

        detrend: boolean
            If True, detrend signals before processing.

        Returns
        =======
        v: numpy.ndarray
            highest variance confounds. Shape: (number of scans, n_confounds)

        Notes
        ======
        This method is related to what has been published in the literature
        as 'CompCor' (Behzadi NeuroImage 2007).

        The implemented algorithm does the following:
        - compute sum of squares for each signals (no mean removal)
        - keep a given percentile of signals with highest variance (percentile)
        - compute an svd of the extracted signals
        - return a given number (n_confounds) of signals from the svd with
          highest singular values.

        See also
        ========
        nisl.signals.high_variance_confounds
    """

    niimgs = utils.check_niimgs(niimgs)
    if mask_img is not None:
        mask_img = utils.check_niimg(mask_img)
        sigs = masking.apply_mask(niimgs, mask_img)
    else:
        sigs = niimgs.get_data()
        # Not using apply_mask here saves memory in most cases.
        sigs = np.reshape(sigs, (-1, sigs.shape[-1])).T

    return signals.high_variance_confounds(sigs, n_confounds=n_confounds,
                                           percentile=percentile,
                                           detrend=detrend)
